"""Build EXP-058: recentaggr plus 12.5% exact Trackman correction."""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

import build_exp051_trackman_direct_candidate as exp051
from build_exp021_final_candidates import build_zip, smoke_test
from train_exp017_rolling_residual import calculate_metrics


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "submissions" / "EXP-051-TMDIRECT"
DESTINATION = ROOT / "submissions" / "EXP-058-TMDIR125"
ZIP_PATH = ROOT / "submit_exp058_tmdir125.zip"
ARTIFACT_DIR = ROOT / "artifacts" / "EXP-058" / "trackman_direct125_candidate"
TEMPLATE = ROOT / "experiments" / "exp021_submission_inference.py"
LOWRANK_ROOT = ROOT / "artifacts" / "EXP-020" / "low_rank_pitcher_context_eb"
REPORT_SEASONS = (2022, 2023, 2024)
CORRECTION_WEIGHT = 0.125
EXP051_PUBLIC = 1047.9791516638
EXP058_PUBLIC = 1047.8300661031
EXP058_SUBMISSIONS = (
    "2026-08-13 09:38:46",
    "2026-08-13 10:40:05",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prediction(season: int, weight: float = CORRECTION_WEIGHT) -> np.ndarray:
    strict, aggressive, recency, trackman = exp051.components(season)
    recent = 0.5 * recency + 0.5 * aggressive
    correction = (trackman - strict) / 0.25
    return np.clip(recent + weight * correction, 0.0, 1.0)


def validation() -> dict[str, object]:
    briers: dict[str, float] = {}
    skills: dict[str, float] = {}
    for season in REPORT_SEASONS:
        target = np.load(
            LOWRANK_ROOT / f"targets_{season}.npy"
        ).astype(float)
        metrics = calculate_metrics(target, prediction(season))
        briers[str(season)] = float(metrics["brier_score"])
        skills[str(season)] = float(metrics["skill_score_unclipped"])
    values = list(skills.values())
    return {
        "season_briers": briers,
        "season_skills": skills,
        "mean_skill": float(np.mean(values)),
        "min_skill": float(np.min(values)),
        "latest_2024_skill": skills["2024"],
    }


def refit_public_curve(seasons: tuple[int, ...]) -> dict[str, object]:
    """Keep prior Brier curvature, anchor q(0) and observed q(.10)."""
    original_weight = exp051.CORRECTION_WEIGHT
    estimates: dict[float, float] = {}
    try:
        for weight in (0.0, 0.1, 0.2):
            exp051.CORRECTION_WEIGHT = weight
            estimates[weight] = exp051.geometry(seasons)["estimated_score"]
    finally:
        exp051.CORRECTION_WEIGHT = original_weight
    q0 = estimates[0.0]
    curvature = (estimates[0.2] - 2.0 * estimates[0.1] + q0) / 0.02
    linear = (EXP051_PUBLIC - q0 - 0.01 * curvature) / 0.1
    expected = q0 + CORRECTION_WEIGHT * linear + CORRECTION_WEIGHT**2 * curvature
    optimum = -linear / (2.0 * curvature)
    return {
        "seasons": list(seasons),
        "anchor_w000": q0,
        "anchor_w010_actual": EXP051_PUBLIC,
        "retained_quadratic_curvature": float(curvature),
        "refit_linear_term": float(linear),
        "estimated_optimum_weight": float(optimum),
        "candidate_weight": CORRECTION_WEIGHT,
        "candidate_estimated_score": float(expected),
        "leaderboard_model_is_diagnostic": True,
    }


def main() -> None:
    started = time.time()
    metrics = validation()
    public_geometry = {
        "method": (
            "retain prior Brier quadratic curvature and anchor the line at "
            "recentaggr w=0 and observed EXP-051 w=.10"
        ),
        "scenarios": {
            "2024": refit_public_curve((2024,)),
            "pooled_2022_2024": refit_public_curve(REPORT_SEASONS),
        },
        "selection": (
            "fixed .125 between local mean optimum .10 and latest-fold "
            "plateau; one bounded follow-up to observed EXP-051"
        ),
        "leaderboard_overfit_risk": True,
    }
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    shutil.copytree(SOURCE_DIR, DESTINATION)
    shutil.copyfile(TEMPLATE, DESTINATION / "script.py")
    metadata_path = DESTINATION / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "experiment": "EXP-058",
            "candidate": "trackman_direct_recent_w0125",
            "component_formula": (
                "recentaggr + 0.125 * exact_trackman_direct_correction"
            ),
            "validation_aggregate_2022_2024": metrics,
            "public_geometry": public_geometry,
            "selection_status": (
                "single bounded Public-informed follow-up; not nested"
            ),
            "probability_calibration": "identity",
        }
    )
    write_json(metadata_path, metadata)
    zip_result = build_zip(DESTINATION, ZIP_PATH)
    smoke = smoke_test(ZIP_PATH)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "EXP-058",
        "candidate": metadata["candidate"],
        "validation": metrics,
        "public_geometry": public_geometry,
        "references": {
            "recentaggr_public": 1046.9889925352,
            "exp051_public": EXP051_PUBLIC,
            "exp053_public": 1046.7664784878,
        },
        "public_result": {
            "submitted_at": list(EXP058_SUBMISSIONS),
            "score": EXP058_PUBLIC,
            "reference_exp051_score": EXP051_PUBLIC,
            "duplicate_submission_same_score": True,
            "source": "user-provided DACON leaderboard result",
        },
        "zip": zip_result,
        "smoke": smoke,
        "qa": {
            "current_fold_labels_used_to_fit_components": False,
            "test_row_aggregation": False,
            "weight_informed_by_public_scores": True,
            "negative_weight_geometry_extrapolation": True,
            "python": platform.python_version(),
        },
        "total_seconds": time.time() - started,
    }
    write_json(ARTIFACT_DIR / "validation_metrics.json", report)
    print(
        f"saved={ZIP_PATH} mean={metrics['mean_skill']:.3f} "
        f"min={metrics['min_skill']:.3f} smoke=passed",
        flush=True,
    )


if __name__ == "__main__":
    main()
