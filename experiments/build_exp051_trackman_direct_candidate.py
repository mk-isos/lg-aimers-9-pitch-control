"""Build EXP-051: recentaggr plus 10% exact Trackman direct correction."""

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
DESTINATION = ROOT / "submissions" / "EXP-051-TMDIRECT"
ZIP_PATH = ROOT / "submit_exp051_tmdirect.zip"
ARTIFACT_DIR = ROOT / "artifacts" / "EXP-051" / "trackman_direct_candidate"
TEMPLATE = ROOT / "experiments" / "exp021_submission_inference.py"
LOWRANK_ROOT = ROOT / "artifacts" / "EXP-020" / "low_rank_pitcher_context_eb"
AGGRESSIVE_ROOT = ROOT / "artifacts" / "EXP-020" / "pitcher_count_eb_atop_team"
RECENCY_ROOT = ROOT / "artifacts" / "EXP-037" / "lowrank_source_policies"
TRACKMAN_ROOT = ROOT / "artifacts" / "EXP-043" / "exact_pitchtype_control_eb"
REPORT_SEASONS = (2022, 2023, 2024)
CORRECTION_WEIGHT = 0.10
PUBLIC = {
    "strict": 1043.6074197937,
    "aggressive": 1043.1871309639,
    "strict_aggressive_50": 1045.1827084551,
    "recentaggr": 1046.9889925352,
    "trackman_recent_50": 1046.9499938833,
}


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def components(season: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strict = np.load(
        LOWRANK_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    trackman = np.load(
        TRACKMAN_ROOT / f"predictions_fine_direct_w025_{season}.npy"
    ).astype(float)
    return strict, aggressive, recency, trackman


def prediction(season: int) -> np.ndarray:
    strict, aggressive, recency, trackman = components(season)
    recent = 0.5 * recency + 0.5 * aggressive
    # trackman = strict + 0.25 * direct_correction by construction.
    direct_correction = (trackman - strict) / 0.25
    return np.clip(recent + CORRECTION_WEIGHT * direct_correction, 0.0, 1.0)


def validation() -> dict[str, object]:
    briers: dict[str, float] = {}
    skills: dict[str, float] = {}
    for season in REPORT_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
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


def geometry(seasons: tuple[int, ...]) -> dict[str, float]:
    strict_values = []
    aggressive_values = []
    recency_values = []
    trackman_values = []
    for season in seasons:
        strict, aggressive, recency, trackman = components(season)
        strict_values.append(strict)
        aggressive_values.append(aggressive)
        recency_values.append(recency)
        trackman_values.append(trackman)
    strict = np.concatenate(strict_values)
    aggressive = np.concatenate(aggressive_values)
    recency = np.concatenate(recency_values)
    trackman = np.concatenate(trackman_values)
    names = ["strict", "aggressive", "recency", "trackman"]
    values = [strict, aggressive, recency, trackman]
    distance = np.array(
        [[np.mean(np.square(left - right)) for right in values] for left in values]
    )
    diversity_sa = 4.0 * (
        PUBLIC["strict_aggressive_50"]
        - 0.5 * (PUBLIC["strict"] + PUBLIC["aggressive"])
    )
    diversity = distance * (diversity_sa / distance[0, 1])
    vertex = np.zeros(4)
    vertex[0] = PUBLIC["strict"]
    vertex[1] = PUBLIC["aggressive"]
    vertex[2] = (
        2.0 * PUBLIC["recentaggr"] - vertex[1] - 0.5 * diversity[2, 1]
    )
    recent = 0.5 * recency + 0.5 * aggressive
    track_recent_diversity = (
        diversity_sa
        / distance[0, 1]
        * np.mean(np.square(trackman - recent))
    )
    vertex[3] = (
        2.0 * PUBLIC["trackman_recent_50"]
        - PUBLIC["recentaggr"]
        - 0.5 * track_recent_diversity
    )

    def score(weight: np.ndarray) -> float:
        output = float(weight @ vertex)
        for left in range(4):
            for right in range(left + 1, 4):
                output += weight[left] * weight[right] * diversity[left, right]
        return float(output)

    # recent + w * ((trackman-strict)/0.25)
    candidate_weight = np.array(
        [-4 * CORRECTION_WEIGHT, 0.5, 0.5, 4 * CORRECTION_WEIGHT]
    )
    return {
        "seasons": list(seasons),
        "affine_weights": {
            name: float(value)
            for name, value in zip(names, candidate_weight, strict=True)
        },
        "estimated_score": score(candidate_weight),
        "negative_weight_extrapolation": True,
    }


def main() -> None:
    started = time.time()
    metrics = validation()
    public_geometry = {
        "known_public_scores": PUBLIC,
        "correction_weight": CORRECTION_WEIGHT,
        "scenarios": {
            "2024": geometry((2024,)),
            "pooled_2022_2024": geometry(REPORT_SEASONS),
        },
        "selection": "fixed 0.10 where both scenario curves are near their maxima",
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
            "experiment": "EXP-051",
            "candidate": "trackman_direct_recent_w010",
            "component_formula": "recentaggr + 0.10 * exact_trackman_direct_correction",
            "validation_aggregate_2022_2024": metrics,
            "public_geometry": public_geometry,
            "selection_status": "Public-score-informed affine correction; not nested",
            "probability_calibration": "identity",
        }
    )
    write_json(metadata_path, metadata)
    zip_result = build_zip(DESTINATION, ZIP_PATH)
    smoke = smoke_test(ZIP_PATH)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "EXP-051",
        "candidate": metadata["candidate"],
        "validation": metrics,
        "public_geometry": public_geometry,
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
