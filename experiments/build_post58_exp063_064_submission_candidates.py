"""Build exploratory post-EXP-058 submission packages for EXP-063/064.

The frozen EXP-051 package is copied byte-for-byte before a single row-local
post-processing branch is added.  This builder never reads the canonical
``data/test.csv`` or ``data/sample_submission.csv``.  Package QA uses a small
source-derived synthetic 2025 fixture with unseen entity/team identifiers.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_exp021_final_candidates import build_zip
from temporal_multirate_features import attach_training_multirate_features
from temporal_residual_features import (
    add_static_features,
    attach_training_temporal_features,
)
from train_exp063_uncertain_region_residual import (
    CORRECTION_CLIP as EXP063_CORRECTION_CLIP,
    FEATURES as EXP063_FEATURES,
    new_model as new_exp063_model,
)
from train_exp064_invariant_uncertainty_group_eb import (
    CORRECTION_CLIP as EXP064_CORRECTION_CLIP,
    SMOOTHING as EXP064_SMOOTHING,
    row_keys as exp064_row_keys,
    season_map as exp064_season_map,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
FROZEN_BASE = ROOT / "submissions" / "EXP-051-TMDIRECT"
EXP063_DESTINATION = ROOT / "submissions" / "EXP-063-UNCERTAIN"
EXP064_DESTINATION = ROOT / "submissions" / "EXP-064-STABLERUNNERS"
READY_DIR = ROOT / "ready_to_submit" / "2026-08-20-post58"
EXP063_ZIP = READY_DIR / "EXP-063-UNCERTAIN.zip"
EXP064_ZIP = READY_DIR / "EXP-064-STABLERUNNERS.zip"
MANIFEST_PATH = READY_DIR / "post58_exp063_064_manifest.json"
PYTHON = ROOT / ".venv" / "bin" / "python"

LOWRANK_ROOT = ROOT / "artifacts" / "EXP-020" / "low_rank_pitcher_context_eb"
AGGRESSIVE_ROOT = ROOT / "artifacts" / "EXP-020" / "pitcher_count_eb_atop_team"
RECENCY_ROOT = ROOT / "artifacts" / "EXP-037" / "lowrank_source_policies"
TRACKMAN_ROOT = ROOT / "artifacts" / "EXP-043" / "exact_pitchtype_control_eb"
EXP063_REPORT = (
    ROOT / "artifacts" / "EXP-063" / "uncertain_region_residual" / "validation_metrics.json"
)
EXP064_REPORT = (
    ROOT
    / "artifacts"
    / "EXP-064"
    / "invariant_uncertainty_group_eb"
    / "validation_metrics.json"
)
EXP063_ARTIFACT_DIR = EXP063_REPORT.parent
EXP064_ARTIFACT_DIR = EXP064_REPORT.parent

ID_COL = "row_id"
TARGET_COL = "control_success"
SOURCE_SEASONS = (2021, 2022, 2023, 2024)
EXP063_SOURCE_SEASON = 2024
EXP063_THRESHOLD = 0.06
EXP063_WEIGHT = 0.25
EXP064_COLUMNS = ("count", "runners", "pbin")
EXP064_WEIGHT = 0.50

POST58_HELPERS = r'''

def _post58_apply_candidate(
    frame: pd.DataFrame,
    base_predictions: np.ndarray,
    candidate: str,
) -> np.ndarray:
    """Apply one frozen row-local post-EXP-058 correction."""
    state = json.loads(
        (MODEL_DIR / "post58_candidate_state.json").read_text(encoding="utf-8")
    )
    if str(state["candidate"]) != candidate:
        raise ValueError("post58 candidate metadata/state mismatch")

    if candidate == "close060_last_w025":
        feature_names = [str(value) for value in state["features"]]
        missing = sorted(set(feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"post58 EXP-063 features are missing: {missing}")
        model = lgb.Booster(
            model_file=str(MODEL_DIR / "post58_uncertain_residual_lightgbm.txt")
        )
        if int(model.num_feature()) != len(feature_names):
            raise ValueError("post58 EXP-063 feature schema/model width mismatch")
        eligible = np.abs(base_predictions - 0.5) < float(state["uncertainty_radius"])
        correction = np.zeros(len(frame), dtype=float)
        if eligible.any():
            values = model.predict(frame.loc[eligible, feature_names]).astype(float)
            correction[eligible] = np.clip(
                values,
                -float(state["correction_clip"]),
                float(state["correction_clip"]),
            )
        return np.clip(
            base_predictions + float(state["additive_weight"]) * correction,
            0.0,
            1.0,
        )

    if candidate == "stable_count_runners_pbin_w050":
        lookup = {
            (int(record["count"]), int(record["runners"]), int(record["pbin"])):
            float(record["effect"])
            for record in state["records"]
        }
        count = frame["count_index"].to_numpy(dtype=int)
        runners = frame["num_runners_on"].to_numpy(dtype=int)
        pbin = np.clip(
            ((base_predictions - 0.35) / 0.025).astype(int), 0, 12
        )
        correction = np.fromiter(
            (
                lookup.get((int(c), int(r), int(p)), 0.0)
                for c, r, p in zip(count, runners, pbin, strict=True)
            ),
            dtype=float,
            count=len(frame),
        )
        correction = np.clip(
            correction,
            -float(state["correction_clip"]),
            float(state["correction_clip"]),
        )
        return np.clip(
            base_predictions + float(state["additive_weight"]) * correction,
            0.0,
            1.0,
        )

    raise ValueError(f"unknown post58 candidate: {candidate}")
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def exp051_oof_prediction(season: int) -> np.ndarray:
    recency = np.load(RECENCY_ROOT / f"predictions_recency2_{season}.npy").astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    strict = np.load(
        LOWRANK_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
    ).astype(float)
    trackman = np.load(
        TRACKMAN_ROOT / f"predictions_fine_direct_w025_{season}.npy"
    ).astype(float)
    return np.clip(
        0.5 * recency + 0.5 * aggressive + 0.10 * (trackman - strict) / 0.25,
        0.0,
        1.0,
    )


def validate_frozen_base() -> dict[str, str]:
    required = [
        FROZEN_BASE / "script.py",
        FROZEN_BASE / "requirements.txt",
        FROZEN_BASE / "model" / "metadata.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen EXP-051 package is incomplete: {missing}")
    metadata = json.loads(required[-1].read_text(encoding="utf-8"))
    if metadata.get("candidate") != "trackman_direct_recent_w010":
        raise ValueError("frozen base is not EXP-051 trackman_direct_recent_w010")
    if int(metadata.get("history_through_season", -1)) != 2024:
        raise ValueError("frozen EXP-051 history is not through 2024")
    return {
        str(path.relative_to(FROZEN_BASE)): sha256(path)
        for path in sorted(FROZEN_BASE.rglob("*"))
        if path.is_file()
    }


def patch_inference_script(destination: Path, candidate: str) -> None:
    path = destination / "script.py"
    script = path.read_text(encoding="utf-8")
    helper_anchor = "\ndef validate_inputs(test: pd.DataFrame, sample: pd.DataFrame) -> None:\n"
    if script.count(helper_anchor) != 1:
        raise ValueError("EXP-051 inference helper anchor drifted")
    script = script.replace(helper_anchor, POST58_HELPERS + helper_anchor, 1)

    candidate_anchor = '    candidate = str(metadata["candidate"])\n'
    if script.count(candidate_anchor) != 1:
        raise ValueError("EXP-051 inference candidate anchor drifted")
    script = script.replace(
        candidate_anchor,
        '    post58_candidate = str(metadata["candidate"])\n'
        '    candidate = str(metadata["base_candidate"])\n',
        1,
    )

    prediction_anchor = "    prediction_map = dict(zip(test[ID_COL], predictions, strict=True))\n"
    if script.count(prediction_anchor) != 1:
        raise ValueError("EXP-051 inference prediction anchor drifted")
    script = script.replace(
        prediction_anchor,
        "    predictions = _post58_apply_candidate(\n"
        "        frame, predictions, post58_candidate\n"
        "    )\n"
        + prediction_anchor,
        1,
    )

    old_print = (
        "    print(\n"
        "        f\"Saved: {OUTPUT_PATH} | candidate={candidate} | rows={len(sample)} | \"\n"
        "        f\"mean={predictions.mean():.6f} | min={predictions.min():.6f} | \"\n"
        "        f\"max={predictions.max():.6f}\"\n"
        "    )\n"
    )
    if script.count(old_print) != 1:
        raise ValueError("EXP-051 inference completion-print anchor drifted")
    script = script.replace(
        old_print,
        "    print(\n"
        "        f\"Saved: {OUTPUT_PATH} | candidate={post58_candidate}\"\n"
        "    )\n",
        1,
    )
    script = script.replace(
        '"""EXP-021 final candidate inference (copied to the ZIP root as script.py)."""',
        f'"""Exploratory {candidate} inference over the frozen EXP-051 base."""',
        1,
    )
    path.write_text(script, encoding="utf-8")


def copy_frozen_base(destination: Path, candidate: str) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(FROZEN_BASE, destination)
    patch_inference_script(destination, candidate)


def exp063_cutoff2023_parity(frame: pd.DataFrame) -> dict[str, object]:
    """Rebuild the original 2023->2024 last-source fold and check its artifact."""
    source_mask = frame["season"].eq(2023).to_numpy()
    validation_mask = frame["season"].eq(2024).to_numpy()
    source_x = frame.loc[source_mask, list(EXP063_FEATURES)].reset_index(drop=True)
    validation_x = frame.loc[validation_mask, list(EXP063_FEATURES)].reset_index(
        drop=True
    )
    source_target = np.load(LOWRANK_ROOT / "targets_2023.npy").astype(float)
    validation_target = np.load(LOWRANK_ROOT / "targets_2024.npy").astype(float)
    source_base = exp051_oof_prediction(2023)
    validation_base = exp051_oof_prediction(2024)
    if not np.array_equal(
        source_target,
        frame.loc[source_mask, TARGET_COL].to_numpy(dtype=float),
    ) or not np.array_equal(
        validation_target,
        frame.loc[validation_mask, TARGET_COL].to_numpy(dtype=float),
    ):
        raise ValueError("EXP-063 historical parity target/order mismatch")
    source_eligible = np.abs(source_base - 0.5) < EXP063_THRESHOLD
    source_residual = source_target[source_eligible] - source_base[source_eligible]
    source_residual -= source_residual.mean()
    model = new_exp063_model()
    model.fit(source_x.loc[source_eligible], source_residual)
    validation_eligible = np.abs(validation_base - 0.5) < EXP063_THRESHOLD
    correction = np.zeros(len(validation_base), dtype=float)
    correction[validation_eligible] = np.clip(
        model.predict(validation_x.loc[validation_eligible]),
        -EXP063_CORRECTION_CLIP,
        EXP063_CORRECTION_CLIP,
    )
    rebuilt = np.clip(validation_base + EXP063_WEIGHT * correction, 0.0, 1.0)
    saved_path = (
        EXP063_ARTIFACT_DIR / "predictions_close060_last_w025_2024.npy"
    )
    saved = np.load(saved_path).astype(float)
    maximum = float(np.max(np.abs(rebuilt - saved)))
    if maximum > 1e-12:
        raise ValueError(f"EXP-063 cutoff-2023 parity failed: {maximum}")
    return {
        "source_season": 2023,
        "validation_season": 2024,
        "saved_prediction": str(saved_path),
        "rows": len(rebuilt),
        "max_abs_difference": maximum,
        "tolerance": 1e-12,
        "status": "passed",
    }


def build_exp063(
    raw: pd.DataFrame,
    frozen_hashes: dict[str, str],
) -> dict[str, object]:
    started = time.time()
    frame, _ = attach_training_temporal_features(add_static_features(raw))
    frame, _, multirate_audit = attach_training_multirate_features(frame)
    missing = sorted(set(EXP063_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"EXP-063 source features are missing: {missing}")
    historical_parity = exp063_cutoff2023_parity(frame)
    mask = frame["season"].eq(EXP063_SOURCE_SEASON).to_numpy()
    x = frame.loc[mask, list(EXP063_FEATURES)].reset_index(drop=True)
    target = np.load(LOWRANK_ROOT / "targets_2024.npy").astype(float)
    base = exp051_oof_prediction(EXP063_SOURCE_SEASON)
    observed = frame.loc[mask, TARGET_COL].to_numpy(dtype=float)
    if not (len(x) == len(target) == len(base)) or not np.array_equal(observed, target):
        raise ValueError("EXP-063 source OOF target/order mismatch")
    eligible = np.abs(base - 0.5) < EXP063_THRESHOLD
    residual = target[eligible] - base[eligible]
    residual_mean = float(residual.mean())
    centered = residual - residual_mean
    model = new_exp063_model()
    fit_started = time.time()
    model.fit(x.loc[eligible], centered)
    fit_seconds = time.time() - fit_started

    copy_frozen_base(EXP063_DESTINATION, "close060_last_w025")
    model_path = EXP063_DESTINATION / "model" / "post58_uncertain_residual_lightgbm.txt"
    model.booster_.save_model(model_path)
    state = {
        "format": "exp063_uncertain_residual_lgb_v1",
        "experiment": "EXP-063",
        "candidate": "close060_last_w025",
        "base_candidate": "trackman_direct_recent_w010",
        "source_season": EXP063_SOURCE_SEASON,
        "source_target": "2024 EXP-051 OOF residual",
        "residual_centered_within_eligible_source": True,
        "residual_center_removed": residual_mean,
        "uncertainty_center": 0.5,
        "uncertainty_radius": EXP063_THRESHOLD,
        "threshold": 0.5,
        "correction_clip": EXP063_CORRECTION_CLIP,
        "additive_weight": EXP063_WEIGHT,
        "features": list(EXP063_FEATURES),
        "source_rows": int(len(x)),
        "eligible_source_rows": int(eligible.sum()),
        "lightgbm": new_exp063_model().get_params(),
    }
    write_json(EXP063_DESTINATION / "model" / "post58_candidate_state.json", state)
    report = json.loads(EXP063_REPORT.read_text(encoding="utf-8"))
    metadata_path = EXP063_DESTINATION / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "experiment": "EXP-063",
            "candidate": "close060_last_w025",
            "base_candidate": "trackman_direct_recent_w010",
            "component_formula": (
                "EXP051 + 0.25 * clip(LGB(2024 centered EXP051 OOF residual), "
                "-0.03, 0.03) only where abs(EXP051-0.5)<0.06"
            ),
            "full_fit_state": state,
            "validation_aggregate_2022_2024": report["aggregate_2022_2024"][
                "close060_last_w025"
            ],
            "selection_status": (
                "exploratory submission candidate; original local hard gate failed; "
                "last-source variant was bounded post-hoc"
            ),
            "exploratory": True,
            "validation_gate_passed": False,
            "canonical_test_or_sample_opened_during_build": False,
            "saved_2024_cutoff2023_reconstruction_parity": historical_parity,
            "frozen_exp051_file_sha256_before_candidate_overlay": frozen_hashes,
        }
    )
    write_json(metadata_path, metadata)
    return {
        "fit_seconds": fit_seconds,
        "total_build_seconds": time.time() - started,
        "state": state,
        "saved_2024_cutoff2023_reconstruction_parity": historical_parity,
        "multirate_source_diagnostics": multirate_audit,
    }


def stable_runner_state(
    raw: pd.DataFrame,
    source_seasons: tuple[int, ...] = SOURCE_SEASONS,
) -> dict[str, object]:
    maps: list[pd.DataFrame] = []
    per_source_cells: dict[str, int] = {}
    if not source_seasons:
        raise ValueError("EXP-064 requires at least one source season")
    for season in source_seasons:
        rows = raw.loc[raw["season"].eq(season)].reset_index(drop=True)
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        base = exp051_oof_prediction(season)
        if not np.array_equal(target, rows[TARGET_COL].to_numpy(dtype=float)):
            raise ValueError(f"EXP-064 source OOF target/order mismatch: {season}")
        keys = exp064_row_keys(rows, base)
        source = exp064_season_map(
            keys,
            target - base,
            EXP064_COLUMNS,
        )
        maps.append(source)
        per_source_cells[str(season)] = int(len(source))

    union = maps[0].index
    for source in maps[1:]:
        union = union.union(source.index)
    effects = np.column_stack(
        [source["effect"].reindex(union).fillna(0.0).to_numpy() for source in maps]
    )
    counts = np.column_stack(
        [source["count"].reindex(union).fillna(0.0).to_numpy() for source in maps]
    )
    nonzero = counts > 0
    stable = nonzero.all(axis=1) & (
        (effects > 0).all(axis=1) | (effects < 0).all(axis=1)
    )
    mean_effect = np.clip(
        effects.mean(axis=1),
        -EXP064_CORRECTION_CLIP,
        EXP064_CORRECTION_CLIP,
    )
    records = []
    stable_positions = np.flatnonzero(stable)
    for position in stable_positions:
        key = union[position]
        if not isinstance(key, tuple):
            key = (key,)
        records.append(
            {
                column: int(value)
                for column, value in zip(EXP064_COLUMNS, key, strict=True)
            }
            | {"effect": float(mean_effect[position])}
        )
    records.sort(key=lambda row: tuple(int(row[column]) for column in EXP064_COLUMNS))
    return {
        "format": "stable_count_runners_pbin_v1",
        "experiment": "EXP-064",
        "candidate": "stable_count_runners_pbin_w050",
        "base_candidate": "trackman_direct_recent_w010",
        "source_seasons": list(source_seasons),
        "columns": list(EXP064_COLUMNS),
        "pbin": "clip(((EXP051-.35)/.025).astype(int),0,12)",
        "smoothing": EXP064_SMOOTHING,
        "stability": "nonzero in all source seasons and common non-zero sign",
        "source_season_equal_weight": True,
        "correction_clip": EXP064_CORRECTION_CLIP,
        "additive_weight": EXP064_WEIGHT,
        "per_source_cells": per_source_cells,
        "union_cells": int(len(union)),
        "stable_cells": int(stable.sum()),
        "records": records,
    }


def map_stable_runner_state(
    rows: pd.DataFrame,
    base: np.ndarray,
    state: dict[str, object],
) -> np.ndarray:
    keys = exp064_row_keys(rows.reset_index(drop=True), base)
    lookup = {
        (int(record["count"]), int(record["runners"]), int(record["pbin"])): float(
            record["effect"]
        )
        for record in state["records"]
    }
    correction = np.fromiter(
        (
            lookup.get((int(count), int(runners), int(pbin)), 0.0)
            for count, runners, pbin in keys.loc[
                :, list(EXP064_COLUMNS)
            ].itertuples(index=False, name=None)
        ),
        dtype=float,
        count=len(keys),
    )
    return np.clip(
        correction,
        -EXP064_CORRECTION_CLIP,
        EXP064_CORRECTION_CLIP,
    )


def exp064_cutoff2023_parity(raw: pd.DataFrame) -> dict[str, object]:
    """Rebuild the original prior-2024 state and check the saved prediction."""
    state = stable_runner_state(raw, source_seasons=(2021, 2022, 2023))
    rows = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    base = exp051_oof_prediction(2024)
    correction = map_stable_runner_state(rows, base, state)
    rebuilt = np.clip(base + EXP064_WEIGHT * correction, 0.0, 1.0)
    saved_path = (
        EXP064_ARTIFACT_DIR
        / "predictions_stable_count_runners_pbin_w050_2024.npy"
    )
    saved = np.load(saved_path).astype(float)
    maximum = float(np.max(np.abs(rebuilt - saved)))
    if maximum > 1e-12:
        raise ValueError(f"EXP-064 cutoff-2023 parity failed: {maximum}")
    return {
        "source_seasons": [2021, 2022, 2023],
        "validation_season": 2024,
        "saved_prediction": str(saved_path),
        "rows": len(rebuilt),
        "max_abs_difference": maximum,
        "tolerance": 1e-12,
        "status": "passed",
    }


def build_exp064(
    raw: pd.DataFrame,
    frozen_hashes: dict[str, str],
) -> dict[str, object]:
    started = time.time()
    historical_parity = exp064_cutoff2023_parity(raw)
    state = stable_runner_state(raw)
    copy_frozen_base(EXP064_DESTINATION, "stable_count_runners_pbin_w050")
    write_json(EXP064_DESTINATION / "model" / "post58_candidate_state.json", state)
    report = json.loads(EXP064_REPORT.read_text(encoding="utf-8"))
    metadata_path = EXP064_DESTINATION / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "experiment": "EXP-064",
            "candidate": "stable_count_runners_pbin_w050",
            "base_candidate": "trackman_direct_recent_w010",
            "component_formula": (
                "EXP051 + 0.50 * clip(mean of 2021-2024 season-centered "
                "count/runners/pbin EB effects retained only for common nonzero/sign "
                "stable cells, -0.02, 0.02)"
            ),
            "full_fit_state": {
                key: value for key, value in state.items() if key != "records"
            },
            "validation_aggregate_2022_2024": report["aggregate_2022_2024"][
                "stable_count_runners_pbin_w050"
            ],
            "selection_status": (
                "exploratory submission candidate; original local hard gate failed; "
                "candidate was predeclared but not the original report selection"
            ),
            "exploratory": True,
            "validation_gate_passed": False,
            "canonical_test_or_sample_opened_during_build": False,
            "saved_2024_cutoff2023_reconstruction_parity": historical_parity,
            "frozen_exp051_file_sha256_before_candidate_overlay": frozen_hashes,
        }
    )
    write_json(metadata_path, metadata)
    return {
        "total_build_seconds": time.time() - started,
        "state": state,
        "saved_2024_cutoff2023_reconstruction_parity": historical_parity,
    }


def source_only_fixture(raw: pd.DataFrame, rows: int = 4) -> pd.DataFrame:
    """Create legal synthetic 2025 inputs without frozen-history collisions."""
    fixture = raw.head(rows).drop(columns=[TARGET_COL]).copy()
    if len(fixture) != rows:
        raise ValueError("not enough source rows for synthetic fixture")
    fixture["season"] = 2025
    fixture[ID_COL] = [f"POST58_SYNTHETIC_{position}" for position in range(rows)]
    fixture["pitcher_id"] = np.arange(9_100_000, 9_100_000 + rows)
    fixture["batter_id"] = np.arange(9_200_000, 9_200_000 + rows)
    fixture["pitcher_team_id"] = np.arange(9_300_000, 9_300_000 + rows)
    fixture["batter_team_id"] = np.arange(9_400_000, 9_400_000 + rows)
    for column in ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"):
        fixture[column] = 0
    rate_columns = [
        column
        for column in fixture.columns
        if column.startswith("asof_") and column.endswith("_rate")
    ]
    fixture.loc[:, rate_columns] = fixture.loc[:, rate_columns].fillna(0.5)
    return fixture.reset_index(drop=True)


def run_fixture(stage: Path, fixture: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    data_dir = stage / "data"
    output_dir = stage / "output"
    data_dir.mkdir(exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    fixture.to_csv(data_dir / "test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            ID_COL: fixture[ID_COL].astype(str),
            TARGET_COL: np.full(len(fixture), 0.5),
        }
    ).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
        encoding="utf-8-sig",
    )
    started = time.time()
    result = subprocess.run(
        [str(PYTHON), "script.py"],
        cwd=stage,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = time.time() - started
    lower_stdout = result.stdout.lower()
    if any(
        token in lower_stdout for token in ("mean=", "min=", "max=", "rows=")
    ):
        raise ValueError("submission script disclosed forbidden prediction aggregate stats")
    output = pd.read_csv(output_dir / "submission.csv", encoding="utf-8-sig")
    if output[ID_COL].astype(str).tolist() != fixture[ID_COL].astype(str).tolist():
        raise ValueError("source-only smoke row order mismatch")
    values = output[TARGET_COL].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("source-only smoke probabilities are invalid")
    return output, runtime


def _prediction_map(output: pd.DataFrame) -> dict[str, float]:
    return dict(
        zip(
            output[ID_COL].astype(str),
            output[TARGET_COL].to_numpy(dtype=float),
            strict=True,
        )
    )


def source_only_smoke_and_invariance(
    zip_path: Path,
    fixture: pd.DataFrame,
) -> dict[str, object]:
    runtimes: dict[str, float] = {}
    differences: list[float] = []
    with tempfile.TemporaryDirectory(prefix="post58-source-smoke-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(stage)
        subprocess.run(
            [str(PYTHON), "-m", "py_compile", "script.py"],
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
        )
        full, runtimes["full"] = run_fixture(stage, fixture)
        reference = _prediction_map(full)

        reverse, runtimes["reverse"] = run_fixture(
            stage, fixture.iloc[::-1].reset_index(drop=True)
        )
        reverse_map = _prediction_map(reverse)
        differences.extend(abs(reference[key] - reverse_map[key]) for key in reference)

        singleton_seconds = 0.0
        for position in range(len(fixture)):
            single, runtime = run_fixture(
                stage, fixture.iloc[[position]].reset_index(drop=True)
            )
            singleton_seconds += runtime
            key, value = next(iter(_prediction_map(single).items()))
            differences.append(abs(reference[key] - value))
        runtimes["singletons_total"] = singleton_seconds

        split_seconds = 0.0
        split_map: dict[str, float] = {}
        for positions in np.array_split(np.arange(len(fixture)), 2):
            part, runtime = run_fixture(
                stage, fixture.iloc[positions].reset_index(drop=True)
            )
            split_seconds += runtime
            split_map.update(_prediction_map(part))
        runtimes["split_total"] = split_seconds
        differences.extend(abs(reference[key] - split_map[key]) for key in reference)

        duplicate = fixture.iloc[[0, 1, 0]].reset_index(drop=True).copy()
        duplicate.loc[2, ID_COL] = "POST58_SYNTHETIC_DUPLICATE_CONTENT"
        duplicate_output, runtimes["duplicate"] = run_fixture(stage, duplicate)
        duplicate_values = duplicate_output[TARGET_COL].to_numpy(dtype=float)
        differences.append(abs(float(duplicate_values[0]) - float(duplicate_values[2])))

    maximum = float(max(differences, default=0.0))
    if maximum > 1e-12:
        raise ValueError(f"row-independence audit failed: {maximum}")
    return {
        "fixture_source": (
            "first four train.csv rows with target removed; season=2025; "
            "unseen synthetic entity/team IDs; zero as-of counts"
        ),
        "canonical_data_test_csv_opened": False,
        "canonical_sample_submission_csv_opened": False,
        "rows": len(fixture),
        "full_reverse_singleton_split_duplicate": "passed",
        "max_abs_difference": maximum,
        "tolerance": 1e-12,
        "probability_range": "passed",
        "script_prediction_summary_stats_suppressed": True,
        "runtime_seconds": runtimes,
    }


def inspect_zip(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in {path.name}: {bad}")
        names = archive.namelist()
        if names[:2] != ["script.py", "requirements.txt"]:
            raise ValueError(f"invalid ZIP root order in {path.name}")
        if not all(
            name in {"script.py", "requirements.txt"} or name.startswith("model/")
            for name in names
        ):
            raise ValueError(f"invalid ZIP member in {path.name}")
        if any(
            Path(name).suffix.lower() in {".csv", ".npy", ".npz", ".pkl", ".pickle"}
            for name in names
        ):
            raise ValueError(f"forbidden data/pickle artifact in {path.name}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "crc": "passed",
        "files": names,
    }


def main() -> None:
    started = time.time()
    frozen_hashes = validate_frozen_base()
    raw = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    if not raw["season"].is_monotonic_increasing:
        raise ValueError("train.csv must be ordered by season")
    READY_DIR.mkdir(parents=True, exist_ok=True)

    exp063_build = build_exp063(raw, frozen_hashes)
    exp064_build = build_exp064(raw, frozen_hashes)
    exp063_zip = build_zip(EXP063_DESTINATION, EXP063_ZIP)
    exp064_zip = build_zip(EXP064_DESTINATION, EXP064_ZIP)
    del exp063_zip, exp064_zip

    fixture = source_only_fixture(raw)
    candidates = []
    for experiment, candidate, directory, zip_path, build in (
        (
            "EXP-063",
            "close060_last_w025",
            EXP063_DESTINATION,
            EXP063_ZIP,
            exp063_build,
        ),
        (
            "EXP-064",
            "stable_count_runners_pbin_w050",
            EXP064_DESTINATION,
            EXP064_ZIP,
            exp064_build,
        ),
    ):
        zip_report = inspect_zip(zip_path)
        smoke = source_only_smoke_and_invariance(zip_path, fixture)
        candidates.append(
            {
                "experiment": experiment,
                "candidate": candidate,
                "package_directory": str(directory),
                "exploratory": True,
                "validation_gate_passed": False,
                "canonical_test_or_sample_opened": False,
                "build": build,
                "zip": zip_report,
                "source_only_smoke_and_row_invariance": smoke,
            }
        )
        print(
            f"built={experiment} candidate={candidate} zip={zip_path} "
            f"sha256={zip_report['sha256']} crc=passed smoke=passed",
            flush=True,
        )

    manifest = {
        "generated_for_submission_date": "2026-08-20",
        "purpose": "exploratory post-EXP-058 candidates requested by the user",
        "canonical_data_test_csv_opened": False,
        "canonical_sample_submission_csv_opened": False,
        "frozen_base": str(FROZEN_BASE),
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "candidates": candidates,
        "total_seconds": time.time() - started,
    }
    write_json(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
