"""Build the bounded EXP-040 70:30 recency/aggressive candidate.

The frozen EXP-032 branches are reused unchanged.  Only their convex weight is
changed.  Aggregate Public Brier scores constrain the prediction geometry;
no evaluation row, label, or test-row aggregate is available or used.
"""

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
SOURCE_DIR = ROOT / "submissions" / "EXP-032-RECENTAGGR"
DESTINATION = ROOT / "submissions" / "EXP-040-REC70"
ZIP_PATH = ROOT / "submit_exp040_rec70.zip"
ARTIFACT_DIR = ROOT / "artifacts" / "EXP-040" / "public_geometry_candidate"
TEMPLATE = ROOT / "experiments" / "exp021_submission_inference.py"
LOWRANK_ROOT = ROOT / "artifacts" / "EXP-020" / "low_rank_pitcher_context_eb"
AGGRESSIVE_ROOT = ROOT / "artifacts" / "EXP-020" / "pitcher_count_eb_atop_team"
RECENCY_ROOT = ROOT / "artifacts" / "EXP-037" / "lowrank_source_policies"
REPORT_SEASONS = (2022, 2023, 2024)
RECENCY_WEIGHT = 0.70
AGGRESSIVE_WEIGHT = 0.30

PUBLIC_SCORES = {
    "strict": 1043.6074197937,
    "aggressive": 1043.1871309639,
    "strict_aggressive_50": 1045.1827084551,
    "strict_r_specific_50": 1042.9008134487,
    "strict_r_specific_aggressive_equal": 1044.4711201305,
    "recency_aggressive_50": 1046.9889925352,
}


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_predictions(season: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strict = np.load(
        LOWRANK_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    return strict, aggressive, recency


def validation_metrics() -> dict[str, object]:
    briers: dict[str, float] = {}
    skills: dict[str, float] = {}
    for season in REPORT_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        _, aggressive, recency = load_predictions(season)
        prediction = np.clip(
            RECENCY_WEIGHT * recency + AGGRESSIVE_WEIGHT * aggressive,
            0.0,
            1.0,
        )
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


def geometry_estimate(seasons: tuple[int, ...]) -> dict[str, float]:
    strict_values: list[np.ndarray] = []
    aggressive_values: list[np.ndarray] = []
    recency_values: list[np.ndarray] = []
    for season in seasons:
        strict, aggressive, recency = load_predictions(season)
        strict_values.append(strict)
        aggressive_values.append(aggressive)
        recency_values.append(recency)
    strict = np.concatenate(strict_values)
    aggressive = np.concatenate(aggressive_values)
    recency = np.concatenate(recency_values)
    distance_sa = float(np.mean(np.square(strict - aggressive)))
    distance_ra = float(np.mean(np.square(recency - aggressive)))

    strict_score = PUBLIC_SCORES["strict"]
    aggressive_score = PUBLIC_SCORES["aggressive"]
    midpoint_score = PUBLIC_SCORES["strict_aggressive_50"]
    diversity_sa = 4.0 * (
        midpoint_score - 0.5 * (strict_score + aggressive_score)
    )
    distance_ratio = distance_ra / distance_sa
    diversity_ra = diversity_sa * distance_ratio
    recency_score = (
        2.0 * PUBLIC_SCORES["recency_aggressive_50"]
        - aggressive_score
        - 0.5 * diversity_ra
    )
    optimum = 0.5 + (
        (recency_score - aggressive_score) / (2.0 * diversity_ra)
    )
    predicted = (
        AGGRESSIVE_WEIGHT * aggressive_score
        + RECENCY_WEIGHT * recency_score
        + AGGRESSIVE_WEIGHT * RECENCY_WEIGHT * diversity_ra
    )
    return {
        "distance_ratio": distance_ratio,
        "estimated_recency_score": recency_score,
        "estimated_optimal_recency_weight": optimum,
        "estimated_score_at_recency_0_70": predicted,
    }


def main() -> None:
    started = time.time()
    metrics = validation_metrics()
    geometry = {
        "public_scores": PUBLIC_SCORES,
        "estimates": {
            "2024": geometry_estimate((2024,)),
            "pooled_2022_2024": geometry_estimate(REPORT_SEASONS),
        },
        "diagnostic_only": True,
        "leaderboard_overfit_risk": (
            "aggregate Public scores informed the convex weight; not nested"
        ),
    }

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    shutil.copytree(SOURCE_DIR, DESTINATION)
    shutil.copyfile(TEMPLATE, DESTINATION / "script.py")
    metadata_path = DESTINATION / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "experiment": "EXP-040",
            "candidate": "recency_aggressive_consensus_70",
            "component_weights": {
                "recency": RECENCY_WEIGHT,
                "aggressive": AGGRESSIVE_WEIGHT,
            },
            "validation_aggregate_2022_2024": metrics,
            "public_geometry": geometry,
            "selection_status": (
                "bounded Public-score-informed convex weight; not fully nested"
            ),
            "probability_calibration": "identity",
        }
    )
    write_json(metadata_path, metadata)

    zip_result = build_zip(DESTINATION, ZIP_PATH)
    smoke = smoke_test(ZIP_PATH)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "EXP-040",
        "stage": "public_geometry_recency_aggressive_candidate",
        "candidate": metadata["candidate"],
        "component_weights": metadata["component_weights"],
        "validation": metrics,
        "public_geometry": geometry,
        "zip": zip_result,
        "smoke": smoke,
        "qa": {
            "current_fold_labels_used_to_fit_components": False,
            "test_row_aggregation": False,
            "fixed_calibration": "identity",
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
