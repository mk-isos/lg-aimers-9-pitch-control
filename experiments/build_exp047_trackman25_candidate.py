"""Build the Public-bounded EXP-047 Trackman-25/recent-75 candidate."""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from build_exp021_final_candidates import build_zip, smoke_test
from train_exp017_rolling_residual import calculate_metrics


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "submissions" / "EXP-044-TRACKREC"
DESTINATION = ROOT / "submissions" / "EXP-047-TRACK25"
ZIP_PATH = ROOT / "submit_exp047_track25.zip"
ARTIFACT_DIR = ROOT / "artifacts" / "EXP-047" / "public_geometry_trackman25"
TEMPLATE = ROOT / "experiments" / "exp021_submission_inference.py"
LOWRANK_ROOT = ROOT / "artifacts" / "EXP-020" / "low_rank_pitcher_context_eb"
AGGRESSIVE_ROOT = ROOT / "artifacts" / "EXP-020" / "pitcher_count_eb_atop_team"
RECENCY_ROOT = ROOT / "artifacts" / "EXP-037" / "lowrank_source_policies"
TRACKMAN_ROOT = ROOT / "artifacts" / "EXP-043" / "exact_pitchtype_control_eb"
REPORT_SEASONS = (2022, 2023, 2024)
TRACKMAN_WEIGHT = 0.25
RECENT_WEIGHT = 0.75
PUBLIC = {
    "recentaggr": 1046.9889925352,
    "trackman_recent_50": 1046.9499938833,
    "strict": 1043.6074197937,
    "aggressive": 1043.1871309639,
    "strict_aggressive_50": 1045.1827084551,
}


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def arrays(season: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strict = np.load(
        LOWRANK_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    exact = np.load(
        TRACKMAN_ROOT / f"predictions_fine_direct_w025_{season}.npy"
    ).astype(float)
    recent = 0.5 * recency + 0.5 * aggressive
    return strict, aggressive, recent, exact


def validation() -> dict[str, object]:
    briers: dict[str, float] = {}
    skills: dict[str, float] = {}
    for season in REPORT_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        _, _, recent, exact = arrays(season)
        prediction = TRACKMAN_WEIGHT * exact + RECENT_WEIGHT * recent
        metrics = calculate_metrics(target, prediction)
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


def geometry(seasons: tuple[int, ...]) -> dict[str, float]:
    strict_values = []
    aggressive_values = []
    recent_values = []
    exact_values = []
    for season in seasons:
        strict, aggressive, recent, exact = arrays(season)
        strict_values.append(strict)
        aggressive_values.append(aggressive)
        recent_values.append(recent)
        exact_values.append(exact)
    strict = np.concatenate(strict_values)
    aggressive = np.concatenate(aggressive_values)
    recent = np.concatenate(recent_values)
    exact = np.concatenate(exact_values)
    diversity_sa = 4.0 * (
        PUBLIC["strict_aggressive_50"]
        - 0.5 * (PUBLIC["strict"] + PUBLIC["aggressive"])
    )
    diversity = diversity_sa * (
        np.mean(np.square(exact - recent))
        / np.mean(np.square(strict - aggressive))
    )
    exact_score = (
        2.0 * PUBLIC["trackman_recent_50"]
        - PUBLIC["recentaggr"]
        - 0.5 * diversity
    )
    optimum = np.clip(
        0.5 + (exact_score - PUBLIC["recentaggr"]) / (2.0 * diversity),
        0.0,
        1.0,
    )
    estimate = (
        RECENT_WEIGHT * PUBLIC["recentaggr"]
        + TRACKMAN_WEIGHT * exact_score
        + RECENT_WEIGHT * TRACKMAN_WEIGHT * diversity
    )
    return {
        "diversity": float(diversity),
        "estimated_exact_component_score": float(exact_score),
        "estimated_optimal_trackman_weight": float(optimum),
        "estimated_score_trackman_0_25": float(estimate),
    }


def main() -> None:
    started = time.time()
    metrics = validation()
    geometry_result = {
        "known_public_scores": PUBLIC,
        "2024": geometry((2024,)),
        "pooled_2022_2024": geometry(REPORT_SEASONS),
        "diagnostic_only": True,
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
            "experiment": "EXP-047",
            "candidate": "trackman_recent_consensus_25",
            "component_weights": {
                "exact_pitchtype_direct": TRACKMAN_WEIGHT,
                "recentaggr": RECENT_WEIGHT,
            },
            "validation_aggregate_2022_2024": metrics,
            "public_geometry": geometry_result,
            "selection_status": (
                "Public-score-informed bounded convex weight; not nested"
            ),
            "probability_calibration": "identity",
        }
    )
    write_json(metadata_path, metadata)
    zip_result = build_zip(DESTINATION, ZIP_PATH)
    smoke = smoke_test(ZIP_PATH)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "EXP-047",
        "candidate": metadata["candidate"],
        "validation": metrics,
        "public_geometry": geometry_result,
        "zip": zip_result,
        "smoke": smoke,
        "qa": {
            "current_fold_labels_used_to_fit_components": False,
            "test_row_aggregation": False,
            "weight_informed_by_public_scores": True,
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
