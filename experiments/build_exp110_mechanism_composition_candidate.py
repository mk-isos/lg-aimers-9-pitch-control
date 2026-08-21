"""Build the sole surviving EXP-105+ submission candidate: EXP-110.

The package starts from the frozen EXP-063 package (which itself is the frozen
EXP-051 package plus the exact deployed EXP-063 residual model), copies the
already-frozen EXP-064/071/072 source states, and applies the preregistered
mechanism-preserving rule.  It never opens canonical test.csv or
sample_submission.csv during build or QA.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_exp021_final_candidates import build_zip
from build_post58_exp063_064_submission_candidates import (
    inspect_zip,
    source_only_fixture,
    source_only_smoke_and_invariance,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "EXP-063-UNCERTAIN"
DESTINATION = ROOT / "submissions" / "EXP-110-MECHANISM-COMPOSITION"
READY_DIR = ROOT / "ready_to_submit" / "2026-08-21-correction-composition"
ZIP_PATH = READY_DIR / "EXP-110-MECHANISM-COMPOSITION.zip"
REPORT_PATH = READY_DIR / "build_exp110_report.json"
TRAIN_PATH = ROOT / "data" / "train.csv"
EVALUATION_REPORT = ROOT / "artifacts" / "EXP-105" / "correction_geometry" / "report.json"

STATE_SOURCES = {
    "exp064_state.json": ROOT
    / "submissions/EXP-064-STABLERUNNERS/model/post58_candidate_state.json",
    "exp071_state.json": ROOT
    / "submissions/EXP-071-PLAYERPHYS-RESID/model/post58_playerphysics_lookup.json",
    "exp072_state.json": ROOT
    / "submissions/EXP-072-DYNAMIC-AR/model/dynamic_pitcher_state.json",
}


COMPOSITION_HELPER = r'''
def _exp110_stable_cell_correction(
    frame: pd.DataFrame,
    base_predictions: np.ndarray,
    state: dict[str, object],
) -> np.ndarray:
    lookup = {
        (int(record["count"]), int(record["runners"]), int(record["pbin"])):
        float(record["effect"])
        for record in state["records"]
    }
    count = frame["count_index"].to_numpy(dtype=int)
    runners = frame["num_runners_on"].to_numpy(dtype=int)
    pbin = np.clip(((base_predictions - 0.35) / 0.025).astype(int), 0, 12)
    raw = np.fromiter(
        (
            lookup.get((int(c), int(r), int(p)), 0.0)
            for c, r, p in zip(count, runners, pbin, strict=True)
        ),
        dtype=float,
        count=len(frame),
    )
    raw = np.clip(raw, -float(state["correction_clip"]), float(state["correction_clip"]))
    return float(state["additive_weight"]) * raw


def _exp110_physical_correction(
    frame: pd.DataFrame,
    state: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    mapping = pd.Series(
        {int(key): int(value) for key, value in dict(state["pitcher_mapping"]).items()},
        dtype=float,
    )
    official_ids = pd.to_numeric(frame["pitcher_id"], errors="coerce")
    mapped = official_ids.map(mapping)

    fallback_frame = pd.DataFrame.from_records(list(state["pitcher_fallback"]))
    if fallback_frame.empty:
        fallback = np.full(len(frame), np.nan, dtype=float)
    else:
        fallback = fallback_frame.set_index("pitcher_trackman_id")["value"].reindex(mapped).to_numpy(float)

    context_frame = pd.DataFrame.from_records(list(state["context_lookup"]))
    if context_frame.empty:
        expected = np.full(len(frame), np.nan, dtype=float)
    else:
        keys = ["pitcher_trackman_id", "count_index", "batter_hand_code"]
        context = context_frame.set_index(keys)["value"]
        query = pd.MultiIndex.from_arrays(
            [mapped, frame["count_index"], frame["batter_hand"]], names=keys
        )
        expected = context.reindex(query).to_numpy(float)
    expected = np.where(np.isfinite(expected), expected, fallback)
    available = mapped.notna().to_numpy() & np.isfinite(expected)
    raw = np.zeros(len(frame), dtype=float)
    raw[available] = expected[available]
    raw = np.clip(raw, -float(state["correction_clip"]), float(state["correction_clip"]))
    return float(state["correction_weight"]) * raw, available


def _exp110_dynamic_correction(
    frame: pd.DataFrame,
    state: dict[str, object],
) -> np.ndarray:
    prior = {
        int(row["pitcher_id"]): (float(row["prior_n"]), float(row["prior_successes"]))
        for row in state["prior_career_states"]
    }
    latest = {
        int(row["pitcher_id"]): (int(row["last_season"]), float(row["latent_logit"]))
        for row in state["latest_latent_states"]
    }
    prediction_season = int(state["prediction_season"])
    league = float(state["league_prior"])
    rho = float(state["rho"])
    strength = float(state["current_season_prior_strength"])
    league_clipped = float(np.clip(league, 1e-6, 1.0 - 1e-6))
    league_logit = float(np.log(league_clipped / (1.0 - league_clipped)))
    raw = np.zeros(len(frame), dtype=float)
    for position, row in enumerate(frame.itertuples(index=False)):
        pitcher_id = int(getattr(row, "pitcher_id"))
        career_n = float(getattr(row, "asof_pitcher_n"))
        career_rate = float(getattr(row, "asof_pitcher_success_rate"))
        if not np.isfinite(career_rate):
            if career_n != 0.0:
                raise ValueError("missing pitcher success rate at positive career count")
            career_rate = 0.0
        career_success = float(np.rint(career_n * career_rate))
        prior_n, prior_success = prior.get(pitcher_id, (0.0, 0.0))
        season_n = career_n - prior_n
        season_success = career_success - prior_success
        if season_n < -1e-6 or season_success < -0.01 or season_success - season_n > 0.01:
            raise ValueError("current-season state is inconsistent with frozen history")
        season_n = max(season_n, 0.0)
        season_success = float(np.clip(season_success, 0.0, season_n))
        last_season, latent = latest.get(pitcher_id, (prediction_season, 0.0))
        gap = max(prediction_season - last_season, 0)
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(league_logit + latent * (rho ** gap), -30.0, 30.0))
        )
        dynamic = (season_success + strength * probability) / (season_n + strength)
        global_value = (season_success + strength * league) / (season_n + strength)
        raw[position] = dynamic - global_value
    return float(state["additive_delta_weight"]) * raw


def _exp110_apply_composition(
    frame: pd.DataFrame,
    base_predictions: np.ndarray,
) -> np.ndarray:
    # c063 is evaluated by the exact deployed EXP-063 helper/model.
    c063 = (
        _post58_apply_candidate(frame, base_predictions, "close060_last_w025")
        - base_predictions
    )
    state064 = json.loads((MODEL_DIR / "exp064_state.json").read_text(encoding="utf-8"))
    state071 = json.loads((MODEL_DIR / "exp071_state.json").read_text(encoding="utf-8"))
    state072 = json.loads((MODEL_DIR / "exp072_state.json").read_text(encoding="utf-8"))
    c064 = _exp110_stable_cell_correction(frame, base_predictions, state064)
    c071, physical_available = _exp110_physical_correction(frame, state071)
    c072 = _exp110_dynamic_correction(frame, state072)
    auxiliary_fields = np.column_stack(
        [c063, c064, np.where(physical_available, 0.0, c072)]
    )
    active = np.sum(np.abs(auxiliary_fields) > 0.0, axis=1)
    auxiliary = auxiliary_fields.sum(axis=1) / np.maximum(active, 1)
    # alpha=1.0 was derived from 2021--2024 historical OOF before packaging.
    return np.clip(base_predictions + c071 + auxiliary, 0.0, 1.0)
'''.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def render_script(source: str) -> str:
    helper_anchor = "\ndef validate_inputs(test: pd.DataFrame, sample: pd.DataFrame) -> None:\n"
    if source.count(helper_anchor) != 1:
        raise ValueError("EXP-063 helper anchor drifted")
    output = source.replace(helper_anchor, "\n\n" + COMPOSITION_HELPER + helper_anchor, 1)
    old = (
        "    predictions = _post58_apply_candidate(\n"
        "        frame, predictions, post58_candidate\n"
        "    )\n"
    )
    if output.count(old) != 1:
        raise ValueError("EXP-063 postprocessor anchor drifted")
    output = output.replace(
        old,
        "    predictions = _exp110_apply_composition(frame, predictions)\n",
        1,
    )
    output = output.replace(
        '"""Exploratory close060_last_w025 inference over the frozen EXP-051 base."""',
        '"""EXP-110 frozen mechanism-preserving correction composition."""',
        1,
    )
    return output


def main() -> None:
    started = time.time()
    if DESTINATION.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {DESTINATION}")
    report = json.loads(EVALUATION_REPORT.read_text(encoding="utf-8"))
    exp110 = report["results"]["EXP-110"]
    if exp110["candidate_tier"] != "B":
        raise ValueError("EXP-110 is no longer a tier-B candidate")
    alpha = float(exp110["final_2025"]["auxiliary_shrinkage"])
    if alpha != 1.0:
        raise ValueError("rendered EXP-110 formula is frozen for alpha=1.0")

    shutil.copytree(SOURCE, DESTINATION, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    script_path = DESTINATION / "script.py"
    script_path.write_text(render_script(script_path.read_text(encoding="utf-8")), encoding="utf-8")
    copied_states = {}
    for name, source in STATE_SOURCES.items():
        destination = DESTINATION / "model" / name
        shutil.copy2(source, destination)
        copied_states[name] = {"source": str(source), "sha256": sha256(destination)}

    metadata_path = DESTINATION / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "experiment": "EXP-110",
            "candidate": "mechanism_rule_composition_a100",
            "base_candidate": "trackman_direct_recent_w010",
            "formula": (
                "clip(p051 + c071 + mean(active c063, c064, "
                "c072 only when c071 physical lookup unavailable), 0, 1)"
            ),
            "auxiliary_shrinkage": alpha,
            "coefficient_source": "2021-2024 historical OOF only",
            "public_score_used_for_weight_or_threshold_fit": False,
            "component_state_files": copied_states,
            "deployment_audit": {
                "frozen_historical_state_only": True,
                "current_row_only": True,
                "query_row_aggregation": False,
                "inference_retraining": False,
                "other_test_rows_used": False,
            },
        }
    )
    write_json(metadata_path, metadata)

    READY_DIR.mkdir(parents=True, exist_ok=True)
    build_zip(DESTINATION, ZIP_PATH)
    zip_report = inspect_zip(ZIP_PATH)
    raw = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    fixture = source_only_fixture(raw, rows=6)
    smoke = source_only_smoke_and_invariance(ZIP_PATH, fixture)
    build_report = {
        "experiment": "EXP-110",
        "candidate": "mechanism_rule_composition_a100",
        "candidate_tier": "B",
        "package_directory": str(DESTINATION),
        "canonical_test_or_sample_opened": False,
        "formula": metadata["formula"],
        "auxiliary_shrinkage": alpha,
        "local_metrics": exp110["folds"],
        "states": copied_states,
        "zip": zip_report,
        "row_independence": smoke,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(REPORT_PATH, build_report)
    print(
        f"saved={ZIP_PATH} sha256={zip_report['sha256']} "
        f"max_diff={smoke['max_abs_difference']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
