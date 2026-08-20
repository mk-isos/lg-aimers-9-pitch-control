"""Build exploratory post-EXP-058 submission ZIPs for EXP-070 and EXP-071.

The builder refits the already-evaluated player-physics models through the
2024 cutoff, but it does not deploy either fitted LightGBM model.  The models
score only 2019--2024 TrackMan history.  Their pitcher/count/batter-hand
summaries, official-to-TrackMan pitcher mapping, and pitcher fallbacks are
frozen as JSON for row-local 2025 inference on top of the EXP-051 package.

The canonical competition test and sample-submission files are guarded and
never opened.  Package smoke tests use target-free rows derived from train.csv
and pass non-canonical fixture paths through environment variables.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_exp041_exact_game_trackman_sequence import mapping_from_aligned
from train_exp043_exact_pitchtype_control_eb import load_main
from train_exp066_partial_sequence_alignment_control import partial_aligned_rows
from train_exp070_partial_player_physics_integration import (
    CONTEXT_SMOOTHING,
    CORRECTION_CLIP,
    add_normalized_physics,
    encoded,
    load_trackman,
    new_model,
)
from train_exp071_partial_player_physics_residual import (
    attach_oof_residual,
    exp051_base,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
CANONICAL_TEST_PATH = (ROOT / "data" / "test.csv").resolve()
CANONICAL_SAMPLE_PATH = (ROOT / "data" / "sample_submission.csv").resolve()
BASE_PACKAGE = ROOT / "submissions" / "EXP-051-TMDIRECT"
READY_DIR = ROOT / "ready_to_submit" / "2026-08-20-post58"
REPORT_PATH = READY_DIR / "build_exp070_071_report.json"
PYTHON = ROOT / ".venv" / "bin" / "python"
LOOKUP_FILENAME = "post58_playerphysics_lookup.json"
ID_COL = "row_id"
TARGET_COL = "control_success"
FINAL_CUTOFF = 2024
PARITY_CUTOFF = 2023
TRACKMAN_FIRST_SEASON = 2019
ROW_TOLERANCE = 1e-9
PARITY_TOLERANCE = 1e-12
RAW_ID_COLUMNS = (
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
)


@dataclass(frozen=True)
class CandidateSpec:
    experiment: str
    candidate: str
    destination_name: str
    zip_name: str
    lookup_kind: str
    correction_weight: float
    target_description: str
    validation_artifact: Path
    saved_prediction: Path

    @property
    def destination(self) -> Path:
        return ROOT / "submissions" / self.destination_name

    @property
    def zip_path(self) -> Path:
        return READY_DIR / self.zip_name


SPECS = (
    CandidateSpec(
        experiment="EXP-070",
        candidate="playerphys_w015",
        destination_name="EXP-070-PLAYERPHYS",
        zip_name="EXP-070-PLAYERPHYS.zip",
        lookup_kind="expected_control",
        correction_weight=0.15,
        target_description=(
            "partial-aligned official control_success through cutoff"
        ),
        validation_artifact=(
            ROOT
            / "artifacts"
            / "EXP-070"
            / "partial_player_physics_integration"
            / "validation_metrics.json"
        ),
        saved_prediction=(
            ROOT
            / "artifacts"
            / "EXP-070"
            / "partial_player_physics_integration"
            / "predictions_playerphys_w015_2024.npy"
        ),
    ),
    CandidateSpec(
        experiment="EXP-071",
        candidate="playerphys_resid_w025",
        destination_name="EXP-071-PLAYERPHYS-RESID",
        zip_name="EXP-071-PLAYERPHYS-RESID.zip",
        lookup_kind="predicted_residual",
        correction_weight=0.25,
        target_description=(
            "2021-2024 EXP-051 OOF residual, centered independently by season"
        ),
        validation_artifact=(
            ROOT
            / "artifacts"
            / "EXP-071"
            / "partial_player_physics_residual"
            / "validation_metrics.json"
        ),
        saved_prediction=(
            ROOT
            / "artifacts"
            / "EXP-071"
            / "partial_player_physics_residual"
            / "predictions_playerphys_resid_w025_2024.npy"
        ),
    ),
)


INFERENCE_LOOKUP_HELPER = r'''
def map_post58_playerphysics_lookup(
    frame: pd.DataFrame,
    state: dict[str, object],
) -> np.ndarray:
    """Map a frozen historical lookup independently for every input row."""
    history_cutoff = int(state["history_cutoff_season"])
    if history_cutoff < 2019 or history_cutoff > 2024:
        raise ValueError("post-058 lookup cutoff is outside the supported window")
    if float(state["correction_clip"]) != 0.03:
        raise ValueError("post-058 lookup correction clip is unexpected")

    mapping_values = {
        int(key): int(value)
        for key, value in dict(state["pitcher_mapping"]).items()
    }
    mapping = pd.Series(mapping_values, dtype=float)
    official_ids = pd.to_numeric(frame["pitcher_id"], errors="coerce")
    mapped = official_ids.map(mapping)

    overall_records = list(state["pitcher_fallback"])
    if overall_records:
        overall_frame = pd.DataFrame.from_records(overall_records)
        overall = overall_frame.set_index("pitcher_trackman_id")["value"]
        fallback = overall.reindex(mapped).to_numpy(dtype=float)
    else:
        fallback = np.full(len(frame), np.nan, dtype=float)

    context_records = list(state["context_lookup"])
    if context_records:
        context_frame = pd.DataFrame.from_records(context_records)
        keys = ["pitcher_trackman_id", "count_index", "batter_hand_code"]
        context = context_frame.set_index(keys)["value"]
        query = pd.MultiIndex.from_arrays(
            [
                mapped,
                pd.to_numeric(frame["count_index"], errors="coerce"),
                pd.to_numeric(frame["batter_hand"], errors="coerce"),
            ],
            names=keys,
        )
        expected = context.reindex(query).to_numpy(dtype=float)
    else:
        expected = np.full(len(frame), np.nan, dtype=float)
    expected = np.where(np.isfinite(expected), expected, fallback)

    correction = np.zeros(len(frame), dtype=float)
    valid = mapped.notna().to_numpy() & np.isfinite(expected)
    lookup_kind = str(state["lookup_kind"])
    if lookup_kind == "expected_control":
        official = pd.to_numeric(
            frame["asof_pitcher_success_rate"], errors="coerce"
        ).to_numpy(dtype=float)
        valid &= np.isfinite(official)
        correction[valid] = expected[valid] - official[valid]
    elif lookup_kind == "predicted_residual":
        correction[valid] = expected[valid]
    else:
        raise ValueError(f"unknown post-058 lookup kind: {lookup_kind}")
    return np.clip(
        correction,
        -float(state["correction_clip"]),
        float(state["correction_clip"]),
    )
'''.strip()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


@contextmanager
def guard_canonical_competition_inputs() -> Iterator[dict[str, object]]:
    """Fail closed if this process tries to read canonical test/sample files."""
    original = pd.read_csv
    audit: dict[str, object] = {
        "canonical_test_csv_opened": False,
        "canonical_sample_submission_opened": False,
        "guard_active": True,
        "blocked_attempts": 0,
    }

    def guarded(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if isinstance(path, (str, os.PathLike)):
            resolved = Path(path).expanduser().resolve()
            if resolved in {CANONICAL_TEST_PATH, CANONICAL_SAMPLE_PATH}:
                audit["blocked_attempts"] = int(audit["blocked_attempts"]) + 1
                raise RuntimeError(
                    f"canonical competition input is forbidden during build: {resolved}"
                )
        return original(path, *args, **kwargs)

    pd.read_csv = guarded  # type: ignore[assignment]
    try:
        yield audit
    finally:
        pd.read_csv = original  # type: ignore[assignment]


def season_equal_weights(seasons: pd.Series) -> np.ndarray:
    counts = seasons.value_counts()
    weights = np.array([1.0 / counts[value] for value in seasons], dtype=float)
    weights *= len(weights) / weights.sum()
    return weights


def freeze_scored_history(
    history: pd.DataFrame,
    mapping: dict[int, int],
    *,
    spec: CandidateSpec,
    cutoff: int,
    value_column: str,
    fit_audit: dict[str, object],
) -> dict[str, object]:
    """Aggregate already-scored historical pitches into deployable JSON rows."""
    if history["season"].max() > cutoff:
        raise ValueError("TrackMan scoring history exceeds cutoff")
    if not history["season"].between(TRACKMAN_FIRST_SEASON, cutoff).all():
        raise ValueError("TrackMan scoring history is outside the declared window")
    if not np.isfinite(history[value_column].to_numpy(dtype=float)).all():
        raise ValueError("non-finite historical model score")

    overall = history.groupby("pitcher_trackman_id", sort=True)[value_column].agg(
        ["mean", "count"]
    )
    keys = ["pitcher_trackman_id", "count_index", "batter_hand_code"]
    context = history.groupby(keys, sort=True, observed=True)[value_column].agg(
        ["sum", "count"]
    )
    prior = overall["mean"].reindex(
        context.index.get_level_values("pitcher_trackman_id")
    ).to_numpy(dtype=float)
    context_values = (
        context["sum"].to_numpy(dtype=float) + CONTEXT_SMOOTHING * prior
    ) / (context["count"].to_numpy(dtype=float) + CONTEXT_SMOOTHING)

    pitcher_fallback = [
        {
            "pitcher_trackman_id": int(index),
            "value": float(row["mean"]),
            "count": int(row["count"]),
        }
        for index, row in overall.iterrows()
    ]
    context_lookup = []
    for (pitcher_id, count_index, batter_hand), value, count in zip(
        context.index,
        context_values,
        context["count"].to_numpy(dtype=int),
        strict=True,
    ):
        context_lookup.append(
            {
                "pitcher_trackman_id": int(pitcher_id),
                "count_index": int(count_index),
                "batter_hand_code": int(batter_hand),
                "value": float(value),
                "count": int(count),
            }
        )

    return {
        "schema_version": 1,
        "experiment": spec.experiment,
        "candidate": spec.candidate,
        "lookup_kind": spec.lookup_kind,
        "correction_weight": spec.correction_weight,
        "correction_clip": CORRECTION_CLIP,
        "history_cutoff_season": int(cutoff),
        "trackman_scoring_seasons": list(
            range(TRACKMAN_FIRST_SEASON, cutoff + 1)
        ),
        "context_smoothing": CONTEXT_SMOOTHING,
        "pitcher_mapping": {
            str(key): int(value) for key, value in sorted(mapping.items())
        },
        "pitcher_fallback": pitcher_fallback,
        "context_lookup": context_lookup,
        "fit_audit": fit_audit,
        "deployment": {
            "base": "EXP-051 trackman_direct_recent_w010",
            "current_trackman_features_required": False,
            "raw_trackman_history_required": False,
            "post58_lightgbm_booster_required": False,
            "row_local_official_inputs": [
                "pitcher_id",
                "balls_before",
                "strikes_before",
                "batter_hand",
                *(
                    ["asof_pitcher_success_rate"]
                    if spec.lookup_kind == "expected_control"
                    else []
                ),
            ],
            "test_row_aggregation": False,
        },
    }


def fit_frozen_lookup(
    aligned: pd.DataFrame,
    trackman: pd.DataFrame,
    spec: CandidateSpec,
    cutoff: int,
    base_cache: dict[int, np.ndarray],
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit through cutoff, score historical TrackMan, and discard the booster."""
    started = time.time()
    history = trackman.loc[
        trackman["season"].between(TRACKMAN_FIRST_SEASON, cutoff)
    ].copy()
    if spec.lookup_kind == "expected_control":
        source = aligned.loc[aligned["season"].le(cutoff)].copy()
        target_column = "control_success"
    elif spec.lookup_kind == "predicted_residual":
        source = attach_oof_residual(aligned, cutoff, base_cache)
        target_column = "centered_residual"
    else:
        raise ValueError(f"unknown lookup kind: {spec.lookup_kind}")
    if source.empty or history.empty:
        raise ValueError("full-fit source/history is empty")
    if int(source["season"].max()) > cutoff:
        raise ValueError("source label exceeds cutoff")

    source["pitcher_hand_code"] = source["pitcher_hand"].astype(np.int8)
    source["batter_hand_code"] = source["batter_hand"].astype(np.int8)
    source["velo_loss"] = source["rel_speed"] - source["zone_speed"]
    source = add_normalized_physics(source, history)
    history = add_normalized_physics(history, history)
    pitcher_categories = pd.Index(
        np.sort(source["pitcher_trackman_id"].dropna().unique())
    )
    source_x = encoded(source, pitcher_categories)
    history_x = encoded(history, pitcher_categories)
    model = new_model()
    model.fit(
        source_x,
        source[target_column].to_numpy(dtype=float),
        sample_weight=season_equal_weights(source["season"]),
        categorical_feature=["pitcher_code"],
    )
    value_column = (
        "predicted_control"
        if spec.lookup_kind == "expected_control"
        else "predicted_residual"
    )
    values = model.predict(history_x).astype(float)
    if spec.lookup_kind == "predicted_residual":
        values = np.clip(values, -CORRECTION_CLIP, CORRECTION_CLIP)
    history[value_column] = values

    mapping_result, mapping_audit = mapping_from_aligned(aligned, cutoff)
    importance = dict(
        zip(
            source_x.columns,
            model.booster_.feature_importance("gain"),
            strict=True,
        )
    )
    top = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:20]
    fit_audit: dict[str, object] = {
        **mapping_audit,
        "target": spec.target_description,
        "source_rows": int(len(source)),
        "source_seasons": [int(value) for value in sorted(source["season"].unique())],
        "source_rows_by_season": {
            str(int(key)): int(value)
            for key, value in source.groupby("season").size().items()
        },
        "source_max_season": int(source["season"].max()),
        "trackman_scored_rows": int(len(history)),
        "trackman_scored_rows_by_season": {
            str(int(key)): int(value)
            for key, value in history.groupby("season").size().items()
        },
        "trackman_max_season": int(history["season"].max()),
        "mapped_pitcher_categories": int(len(pitcher_categories)),
        "feature_count": int(source_x.shape[1]),
        "season_equal_weight": True,
        "current_2025_rows_used_for_fit_or_scoring": False,
        "current_2025_labels_used": False,
        "actual_2025_trackman_used": False,
        "top_gain_features": {name: float(value) for name, value in top},
        "fit_seconds": float(time.time() - started),
    }
    if spec.lookup_kind == "predicted_residual":
        centered = source.groupby("season")["centered_residual"].mean().abs()
        fit_audit["source_residual_center_max_abs"] = float(centered.max())
        fit_audit["oof_residual_source_seasons"] = [2021, 2022, 2023, 2024][
            : max(0, cutoff - 2020)
        ]

    state = freeze_scored_history(
        history,
        mapping_result.mapping,
        spec=spec,
        cutoff=cutoff,
        value_column=value_column,
        fit_audit=fit_audit,
    )
    fit_audit["context_groups"] = len(state["context_lookup"])
    fit_audit["pitcher_fallbacks"] = len(state["pitcher_fallback"])
    # Explicitly release the post-058 booster; it is never copied into a package.
    del model
    return state, fit_audit


def apply_frozen_lookup(
    frame: pd.DataFrame, state: dict[str, object]
) -> np.ndarray:
    """Builder-side implementation used for parity and active lookup QA."""
    namespace = {"np": np, "pd": pd}
    exec(INFERENCE_LOOKUP_HELPER, namespace)
    function = namespace["map_post58_playerphysics_lookup"]
    return function(frame, state)


def parity_against_saved(
    spec: CandidateSpec,
    state: dict[str, object],
    main_frame: pd.DataFrame,
    base_2024: np.ndarray,
) -> dict[str, object]:
    rows = main_frame.loc[main_frame["season"].eq(2024)].reset_index(drop=True)
    correction = apply_frozen_lookup(rows, state)
    if len(rows) != len(base_2024):
        raise ValueError("2024 official/base row order mismatch")
    reconstructed = np.clip(
        base_2024 + spec.correction_weight * correction, 0.0, 1.0
    )
    saved = np.load(spec.saved_prediction).astype(float)
    if len(saved) != len(reconstructed):
        raise ValueError("saved 2024 prediction length mismatch")
    prediction_max_abs = float(np.max(np.abs(reconstructed - saved)))
    raw = base_2024 + spec.correction_weight * correction
    interior = (raw > 0.0) & (raw < 1.0)
    if not interior.any():
        raise ValueError("no interior rows for correction parity")
    saved_correction = (saved[interior] - base_2024[interior]) / (
        spec.correction_weight
    )
    correction_max_abs = float(
        np.max(np.abs(saved_correction - correction[interior]))
    )
    if prediction_max_abs > PARITY_TOLERANCE:
        raise ValueError(
            f"{spec.experiment} saved prediction parity failed: {prediction_max_abs}"
        )
    if correction_max_abs > PARITY_TOLERANCE:
        raise ValueError(
            f"{spec.experiment} saved correction parity failed: {correction_max_abs}"
        )
    return {
        "reconstructed_cutoff": PARITY_CUTOFF,
        "validation_season": 2024,
        "saved_prediction": str(spec.saved_prediction),
        "rows": int(len(saved)),
        "prediction_max_abs_difference": prediction_max_abs,
        "correction_interior_rows": int(interior.sum()),
        "correction_max_abs_difference": correction_max_abs,
        "tolerance": PARITY_TOLERANCE,
        "passed": True,
    }


def lookup_invariance_audit(state: dict[str, object]) -> dict[str, object]:
    mappings = dict(state["pitcher_mapping"])
    fallbacks = list(state["pitcher_fallback"])
    if not mappings or not fallbacks:
        raise ValueError("lookup invariance fixture requires a mapped pitcher")
    reverse_mapping: dict[int, int] = {}
    for official, trackman_id in mappings.items():
        reverse_mapping.setdefault(int(trackman_id), int(official))
    fallback = next(
        row
        for row in fallbacks
        if int(row["pitcher_trackman_id"]) in reverse_mapping
    )
    official_pitcher = reverse_mapping[int(fallback["pitcher_trackman_id"])]
    rows = pd.DataFrame(
        {
            "pitcher_id": np.full(6, official_pitcher, dtype=np.int64),
            "count_index": np.arange(6, dtype=np.int8) % 12,
            "batter_hand": 1 + np.arange(6, dtype=np.int8) % 2,
            "asof_pitcher_success_rate": np.linspace(0.42, 0.57, 6),
        }
    )
    reference = apply_frozen_lookup(rows, state)
    reverse = apply_frozen_lookup(rows.iloc[::-1].reset_index(drop=True), state)[::-1]
    singleton = np.concatenate(
        [apply_frozen_lookup(rows.iloc[[index]], state) for index in range(len(rows))]
    )
    split = np.concatenate(
        [
            apply_frozen_lookup(rows.iloc[:3], state),
            apply_frozen_lookup(rows.iloc[3:], state),
        ]
    )
    duplicate_rows = rows.iloc[[0, 1, 0]].reset_index(drop=True)
    duplicate = apply_frozen_lookup(duplicate_rows, state)
    differences = [
        float(np.max(np.abs(reference - reverse))),
        float(np.max(np.abs(reference - singleton))),
        float(np.max(np.abs(reference - split))),
        float(abs(duplicate[0] - duplicate[2])),
    ]
    maximum = max(differences)
    if maximum > ROW_TOLERANCE:
        raise ValueError(f"active lookup row-invariance failed: {maximum}")
    return {
        "mapped_official_pitcher_fixture": int(official_pitcher),
        "full_reverse_singleton_split_duplicate": "passed",
        "max_abs_difference": float(maximum),
        "tolerance": ROW_TOLERANCE,
    }


def require_replace(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} block, observed {count}")
    return text.replace(old, new)


def patched_inference_script(source: str, spec: CandidateSpec) -> str:
    source = require_replace(
        source,
        '"""EXP-021 final candidate inference (copied to the ZIP root as script.py)."""',
        (
            f'"""{spec.experiment} exploratory frozen player-physics lookup '
            'on EXP-051."""'
        ),
        "module docstring",
    )
    source = require_replace(
        source,
        '''MODEL_DIR = Path("./model")
TEST_PATH = Path("./data/test.csv")
SAMPLE_PATH = Path("./data/sample_submission.csv")
OUTPUT_PATH = Path("./output/submission.csv")''',
        '''MODEL_DIR = Path(os.environ.get("SUBMISSION_MODEL_DIR", "./model"))
TEST_PATH = Path(os.environ.get("SUBMISSION_TEST_PATH", "./data/test.csv"))
SAMPLE_PATH = Path(
    os.environ.get("SUBMISSION_SAMPLE_PATH", "./data/sample_submission.csv")
)
OUTPUT_PATH = Path(
    os.environ.get("SUBMISSION_OUTPUT_PATH", "./output/submission.csv")
)''',
        "environment path",
    )
    token_count = source.count("trackman_direct_recent_w010")
    if token_count < 4:
        raise ValueError(f"EXP-051 candidate token count is too small: {token_count}")
    source = source.replace("trackman_direct_recent_w010", spec.candidate)
    marker = "\ndef validate_inputs(test: pd.DataFrame, sample: pd.DataFrame) -> None:"
    if source.count(marker) != 1:
        raise ValueError("validate_inputs insertion marker is ambiguous")
    source = source.replace(marker, "\n\n" + INFERENCE_LOOKUP_HELPER + "\n" + marker)

    old_branch = f'''    elif candidate == "{spec.candidate}":
        exact_state = json.loads(
            (MODEL_DIR / "exact_pitchtype_control.json").read_text(
                encoding="utf-8"
            )
        )
        exact_correction = map_exact_pitchtype_control(frame, exact_state)
        recent_prediction = 0.5 * recency_predictions + 0.5 * aggressive_predictions
        predictions = np.clip(
            recent_prediction + 0.10 * exact_correction, 0.0, 1.0
        )'''
    new_branch = f'''    elif candidate == "{spec.candidate}":
        exact_state = json.loads(
            (MODEL_DIR / "exact_pitchtype_control.json").read_text(
                encoding="utf-8"
            )
        )
        exact_correction = map_exact_pitchtype_control(frame, exact_state)
        exp051_prediction = np.clip(
            0.5 * recency_predictions
            + 0.5 * aggressive_predictions
            + 0.10 * exact_correction,
            0.0,
            1.0,
        )
        post58_state = json.loads(
            (MODEL_DIR / "{LOOKUP_FILENAME}").read_text(encoding="utf-8")
        )
        if str(post58_state["candidate"]) != candidate:
            raise ValueError("post-058 lookup candidate metadata mismatch")
        if int(post58_state["history_cutoff_season"]) != 2024:
            raise ValueError("deployed post-058 lookup is not frozen through 2024")
        post58_correction = map_post58_playerphysics_lookup(frame, post58_state)
        predictions = np.clip(
            exp051_prediction
            + float(post58_state["correction_weight"]) * post58_correction,
            0.0,
            1.0,
        )'''
    source = require_replace(source, old_branch, new_branch, "candidate branch")
    source = require_replace(
        source,
        '''    print(
        f"Saved: {OUTPUT_PATH} | candidate={candidate} | rows={len(sample)} | "
        f"mean={predictions.mean():.6f} | min={predictions.min():.6f} | "
        f"max={predictions.max():.6f}"
    )''',
        '''    print(f"Saved: {OUTPUT_PATH} | candidate={candidate}")''',
        "completion log",
    )
    forbidden = (
        "predictions.mean()",
        "predictions.min()",
        "predictions.max()",
        "test.groupby(",
        "sample.groupby(",
    )
    observed = [fragment for fragment in forbidden if fragment in source]
    if observed:
        raise ValueError(f"forbidden inference fragments remain: {observed}")
    return source


def load_validation(spec: CandidateSpec) -> dict[str, object]:
    report = json.loads(spec.validation_artifact.read_text(encoding="utf-8"))
    aggregate = report["aggregate_2022_2024"]
    selection = report["selection"]
    if bool(selection["adopt"]):
        raise ValueError(f"{spec.experiment} unexpectedly passed its adoption gate")
    return {
        "candidate": aggregate[spec.candidate],
        "exp051_base": aggregate["base"],
        "selection": selection,
        "status": "exploratory_gate_failed",
        "scientific_adoption": False,
    }


def package_metadata(
    base_metadata: dict[str, object],
    spec: CandidateSpec,
    state: dict[str, object],
    validation: dict[str, object],
    lookup_sha256: str,
) -> dict[str, object]:
    output = dict(base_metadata)
    output.update(
        {
            "experiment": spec.experiment,
            "candidate": spec.candidate,
            "candidate_status": "exploratory_submission_candidate_gate_failed",
            "scientific_adoption": False,
            "history_through_season": FINAL_CUTOFF,
            "component_formula": (
                "EXP-051 + "
                f"{spec.correction_weight:.2f} * frozen_{spec.lookup_kind}_correction"
            ),
            "component_weights": {
                "exp051_trackman_direct_recent_w010": 1.0,
                f"frozen_{spec.lookup_kind}_correction": spec.correction_weight,
            },
            "post58_full_fit": state["fit_audit"],
            "post58_lookup": {
                "file": f"model/{LOOKUP_FILENAME}",
                "sha256": lookup_sha256,
                "schema_version": state["schema_version"],
                "kind": spec.lookup_kind,
                "pitcher_mappings": len(state["pitcher_mapping"]),
                "pitcher_fallbacks": len(state["pitcher_fallback"]),
                "context_groups": len(state["context_lookup"]),
                "correction_clip": CORRECTION_CLIP,
                "context_smoothing": CONTEXT_SMOOTHING,
            },
            "validation_aggregate_2022_2024": validation["candidate"],
            "exp051_base_validation_aggregate_2022_2024": validation["exp051_base"],
            "selection": validation["selection"],
            "deployment": {
                "inference_is_row_local": True,
                "test_row_aggregation": False,
                "actual_current_trackman_used": False,
                "raw_trackman_history_packaged": False,
                "post58_lightgbm_booster_packaged": False,
                "post58_dependency_added": False,
                "base_package_dependency": "lightgbm==4.6.0",
                "base_package": "EXP-051-TMDIRECT",
            },
            "build_input_policy": {
                "canonical_test_csv_opened": False,
                "canonical_sample_submission_opened": False,
                "test_labels_available_or_used": False,
                "full_fit_cutoff": FINAL_CUTOFF,
            },
            "versions": {
                **dict(base_metadata.get("versions", {})),
                "builder_python": platform.python_version(),
                "builder_numpy": np.__version__,
                "builder_pandas": pd.__version__,
                "builder_lightgbm": lgb.__version__,
            },
        }
    )
    return output


def stage_package(
    spec: CandidateSpec,
    state: dict[str, object],
    validation: dict[str, object],
    stage: Path,
) -> dict[str, object]:
    shutil.copytree(
        BASE_PACKAGE,
        stage,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    script_path = stage / "script.py"
    script_path.write_text(
        patched_inference_script(script_path.read_text(encoding="utf-8"), spec),
        encoding="utf-8",
    )
    lookup_path = stage / "model" / LOOKUP_FILENAME
    write_json(lookup_path, state)
    lookup_hash = sha256(lookup_path)
    metadata_path = stage / "model" / "metadata.json"
    base_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = package_metadata(
        base_metadata, spec, state, validation, lookup_hash
    )
    write_json(metadata_path, metadata)
    return {
        "script_sha256": sha256(script_path),
        "lookup_sha256": lookup_hash,
        "metadata_sha256": sha256(metadata_path),
        "requirements_sha256": sha256(stage / "requirements.txt"),
    }


def build_zip(source: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing ZIP: {output}")
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        archive.write(source / "script.py", "script.py")
        archive.write(source / "requirements.txt", "requirements.txt")
        for path in sorted((source / "model").iterdir()):
            if path.is_file():
                archive.write(path, f"model/{path.name}")
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure: {bad}")
        names = archive.namelist()
        if names[:2] != ["script.py", "requirements.txt"]:
            raise ValueError("ZIP root member order is invalid")
        if not all(
            name in {"script.py", "requirements.txt"} or name.startswith("model/")
            for name in names
        ):
            raise ValueError("ZIP contains an invalid member path")
        forbidden_suffixes = {
            ".csv",
            ".npy",
            ".npz",
            ".pkl",
            ".pickle",
            ".joblib",
        }
        if any(Path(name).suffix.lower() in forbidden_suffixes for name in names):
            raise ValueError("ZIP contains raw data, predictions, or pickle state")
        if f"model/{LOOKUP_FILENAME}" not in names:
            raise ValueError("ZIP is missing the frozen post-058 lookup")
        member_crc32 = {
            info.filename: f"{info.CRC:08x}" for info in archive.infolist()
        }
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "crc": "passed",
        "root_order": names[:2],
        "members": names,
        "member_crc32": member_crc32,
    }


def source_fixture() -> pd.DataFrame:
    rows = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", nrows=6)
    rows = rows.drop(columns=[TARGET_COL])
    rows["season"] = 2025
    for offset, column in enumerate(RAW_ID_COLUMNS, start=1):
        rows[column] = (
            np.arange(len(rows), dtype=np.int64) + offset * 1_000_000_000
        )
    rows[ID_COL] = [f"SOURCE_SMOKE_{index:03d}" for index in range(len(rows))]
    return rows


def run_source_rows(
    stage: Path,
    rows: pd.DataFrame,
    *,
    sample_order: list[str] | None = None,
) -> tuple[pd.DataFrame, float, str]:
    fixture_dir = stage / "source_fixture"
    fixture_dir.mkdir(exist_ok=True)
    row_path = fixture_dir / "source_rows.csv"
    sample_path = fixture_dir / "source_submission_format.csv"
    output_path = fixture_dir / "source_output.csv"
    rows.to_csv(row_path, index=False, encoding="utf-8-sig")
    identifiers = sample_order if sample_order is not None else rows[ID_COL].tolist()
    pd.DataFrame(
        {ID_COL: identifiers, TARGET_COL: np.full(len(identifiers), 0.5)}
    ).to_csv(sample_path, index=False, encoding="utf-8-sig")
    if output_path.exists():
        output_path.unlink()
    environment = os.environ.copy()
    environment.update(
        {
            "SUBMISSION_TEST_PATH": str(row_path),
            "SUBMISSION_SAMPLE_PATH": str(sample_path),
            "SUBMISSION_OUTPUT_PATH": str(output_path),
        }
    )
    started = time.time()
    result = subprocess.run(
        [str(PYTHON), "script.py"],
        cwd=stage,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = time.time() - started
    prediction = pd.read_csv(output_path, encoding="utf-8-sig")
    if prediction.columns.tolist() != [ID_COL, TARGET_COL]:
        raise ValueError("source smoke output schema mismatch")
    if prediction[ID_COL].tolist() != identifiers:
        raise ValueError("source smoke output is not in sample order")
    values = prediction[TARGET_COL].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("source smoke probabilities are invalid")
    if any(token in result.stdout for token in ("mean=", "min=", "max=")):
        raise ValueError("submission script logged a test prediction aggregate")
    return prediction, runtime, result.stdout.strip()


def aligned_values(prediction: pd.DataFrame, ids: list[str]) -> np.ndarray:
    return (
        prediction.set_index(ID_COL)[TARGET_COL]
        .astype(float)
        .reindex(ids)
        .to_numpy(dtype=float)
    )


def package_source_smoke(zip_path: Path, rows: pd.DataFrame) -> dict[str, object]:
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
        ids = rows[ID_COL].tolist()
        full, full_runtime, stdout = run_source_rows(
            stage, rows, sample_order=list(reversed(ids))
        )
        reference = aligned_values(full, ids)
        reverse, reverse_runtime, _ = run_source_rows(
            stage, rows.iloc[::-1].reset_index(drop=True), sample_order=ids
        )
        first, first_runtime, _ = run_source_rows(
            stage, rows.iloc[:3].reset_index(drop=True)
        )
        second, second_runtime, _ = run_source_rows(
            stage, rows.iloc[3:].reset_index(drop=True)
        )
        singleton_frames = []
        singleton_runtime = 0.0
        for position in range(len(rows)):
            prediction, runtime, _ = run_source_rows(
                stage, rows.iloc[[position]].reset_index(drop=True)
            )
            singleton_frames.append(prediction)
            singleton_runtime += runtime
        duplicate_rows = rows.iloc[[0, 1, 0]].reset_index(drop=True).copy()
        duplicate_rows.loc[2, ID_COL] = "SOURCE_SMOKE_DUPLICATE"
        duplicate, duplicate_runtime, _ = run_source_rows(stage, duplicate_rows)

        split = pd.concat([first, second], ignore_index=True)
        singleton = pd.concat(singleton_frames, ignore_index=True)
        differences = [
            float(np.max(np.abs(reference - aligned_values(reverse, ids)))),
            float(np.max(np.abs(reference - aligned_values(split, ids)))),
            float(np.max(np.abs(reference - aligned_values(singleton, ids)))),
            float(
                abs(
                    duplicate.iloc[0][TARGET_COL]
                    - duplicate.iloc[2][TARGET_COL]
                )
            ),
        ]
        maximum = max(differences)
        if maximum > ROW_TOLERANCE:
            raise ValueError(f"full-package row-invariance failed: {maximum}")
    return {
        "fixture": (
            "six train rows; target removed; season=2025; raw IDs replaced "
            "with unseen values"
        ),
        "canonical_test_csv_opened": False,
        "canonical_sample_submission_opened": False,
        "temporary_input_filename": "source_rows.csv",
        "temporary_sample_filename": "source_submission_format.csv",
        "sample_order_remap": "passed",
        "full_reverse_singleton_split_duplicate": "passed",
        "max_abs_difference": float(maximum),
        "tolerance": ROW_TOLERANCE,
        "stdout_has_prediction_aggregates": False,
        "completion_message": stdout,
        "runtime_seconds": {
            "full": full_runtime,
            "reverse": reverse_runtime,
            "split_total": first_runtime + second_runtime,
            "singletons_total": singleton_runtime,
            "duplicate": duplicate_runtime,
        },
    }


def install_package(stage: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {destination}")
    shutil.copytree(stage, destination)


def main() -> None:
    started = time.time()
    if not PYTHON.exists():
        raise FileNotFoundError(f"workspace Python is missing: {PYTHON}")
    for spec in SPECS:
        if spec.destination.exists() or spec.zip_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output for {spec.experiment}"
            )
    READY_DIR.mkdir(parents=True, exist_ok=True)

    with guard_canonical_competition_inputs() as input_audit:
        load_started = time.time()
        main_frame = load_main()
        trackman = load_trackman()
        aligned, alignment_audit = partial_aligned_rows()
        load_seconds = time.time() - load_started
        base_cache = {season: exp051_base(season) for season in range(2021, 2025)}
        base_2024 = base_cache[2024]
        fixture = source_fixture()
        packages: dict[str, object] = {}

        for spec in SPECS:
            print(f"{spec.experiment}: rebuilding cutoff-2023 parity lookup", flush=True)
            parity_state, parity_fit = fit_frozen_lookup(
                aligned, trackman, spec, PARITY_CUTOFF, base_cache
            )
            # Round-trip through JSON to exercise the persisted representation.
            parity_state = json.loads(
                json.dumps(parity_state, allow_nan=False, separators=(",", ":"))
            )
            parity = parity_against_saved(
                spec, parity_state, main_frame, base_2024
            )

            print(f"{spec.experiment}: fitting cutoff-2024 frozen lookup", flush=True)
            state, full_fit = fit_frozen_lookup(
                aligned, trackman, spec, FINAL_CUTOFF, base_cache
            )
            active_lookup_invariance = lookup_invariance_audit(state)
            validation = load_validation(spec)
            with tempfile.TemporaryDirectory(
                prefix=f"{spec.experiment.lower()}-package-stage-"
            ) as temporary:
                stage = Path(temporary) / spec.destination_name
                hashes = stage_package(spec, state, validation, stage)
                install_package(stage, spec.destination)
            zip_audit = build_zip(spec.destination, spec.zip_path)
            smoke = package_source_smoke(spec.zip_path, fixture)
            packages[spec.experiment] = {
                "candidate": spec.candidate,
                "status": "exploratory_gate_failed",
                "destination": str(spec.destination),
                "validation": validation,
                "cutoff_2023_saved_2024_parity": parity,
                "parity_fit_audit": parity_fit,
                "full_fit_audit": full_fit,
                "active_lookup_invariance": active_lookup_invariance,
                "package_hashes": hashes,
                "zip": zip_audit,
                "full_package_source_smoke": smoke,
                "correction_assets": {
                    "frozen_lookup_json": f"model/{LOOKUP_FILENAME}",
                    "raw_trackman_packaged": False,
                    "post58_lightgbm_booster_packaged": False,
                    "base_exp051_lightgbm_retained": True,
                },
            }

        report = {
            "builder": str(Path(__file__).resolve()),
            "scope": "post-EXP-058 exploratory EXP-070/071 submission candidates",
            "input_policy": input_audit,
            "alignment_audit": alignment_audit,
            "load_and_alignment_seconds": load_seconds,
            "packages": packages,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "lightgbm": lgb.__version__,
                "requirements": "lightgbm==4.6.0",
            },
            "total_seconds": time.time() - started,
            "peak_rss_mb": peak_rss_mb(),
        }
        write_json(REPORT_PATH, report)
    print(
        f"built={len(SPECS)} report={REPORT_PATH} crc=passed smoke=passed",
        flush=True,
    )


if __name__ == "__main__":
    main()
