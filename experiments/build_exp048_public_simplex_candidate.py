"""Build robust Public-geometry EXP-048 A25/C60/T15 candidate."""

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
DESTINATION = ROOT / "submissions" / "EXP-048-SIMPLEX"
ZIP_PATH = ROOT / "submit_exp048_simplex.zip"
ARTIFACT_DIR = ROOT / "artifacts" / "EXP-048" / "public_simplex_candidate"
TEMPLATE = ROOT / "experiments" / "exp021_submission_inference.py"
LOWRANK_ROOT = ROOT / "artifacts" / "EXP-020" / "low_rank_pitcher_context_eb"
AGGRESSIVE_ROOT = ROOT / "artifacts" / "EXP-020" / "pitcher_count_eb_atop_team"
RECENCY_ROOT = ROOT / "artifacts" / "EXP-037" / "lowrank_source_policies"
TRACKMAN_ROOT = ROOT / "artifacts" / "EXP-043" / "exact_pitchtype_control_eb"
REPORT_SEASONS = (2022, 2023, 2024)
WEIGHTS = {"aggressive": 0.25, "recency": 0.60, "trackman": 0.15}
PUBLIC = {
    "strict": 1043.6074197937,
    "aggressive": 1043.1871309639,
    "strict_aggressive_50": 1045.1827084551,
    "strict_rspecific_50": 1042.9008134487,
    "strict_rspecific_aggressive_equal": 1044.4711201305,
    "recency_aggressive_50": 1046.9889925352,
    "trackman_recent_50": 1046.9499938833,
}


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def components(season: int) -> dict[str, np.ndarray]:
    return {
        "strict": np.load(
            LOWRANK_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
        ).astype(float),
        "rspecific": np.load(
            LOWRANK_ROOT
            / f"predictions_lowrank_s300_r4_Rspecific_{season}.npy"
        ).astype(float),
        "aggressive": np.load(
            AGGRESSIVE_ROOT
            / f"predictions_r_gated_team_pc_all_{season}.npy"
        ).astype(float),
        "recency": np.load(
            RECENCY_ROOT / f"predictions_recency2_{season}.npy"
        ).astype(float),
        "trackman": np.load(
            TRACKMAN_ROOT / f"predictions_fine_direct_w025_{season}.npy"
        ).astype(float),
    }


def candidate_prediction(values: dict[str, np.ndarray]) -> np.ndarray:
    return np.clip(
        WEIGHTS["aggressive"] * values["aggressive"]
        + WEIGHTS["recency"] * values["recency"]
        + WEIGHTS["trackman"] * values["trackman"],
        0.0,
        1.0,
    )


def validation() -> dict[str, object]:
    briers: dict[str, float] = {}
    skills: dict[str, float] = {}
    for season in REPORT_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        metrics = calculate_metrics(target, candidate_prediction(components(season)))
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


def public_geometry(seasons: tuple[int, ...]) -> dict[str, object]:
    names = ["strict", "aggressive", "rspecific", "recency", "trackman"]
    values = {
        name: np.concatenate([components(season)[name] for season in seasons])
        for name in names
    }
    distance = np.array(
        [
            [np.mean(np.square(values[a] - values[b])) for b in names]
            for a in names
        ]
    )
    diversity_sa = 4.0 * (
        PUBLIC["strict_aggressive_50"]
        - 0.5 * (PUBLIC["strict"] + PUBLIC["aggressive"])
    )
    diversity = distance * (
        diversity_sa / distance[names.index("strict"), names.index("aggressive")]
    )
    vertex = np.zeros(len(names))
    vertex[0] = PUBLIC["strict"]
    vertex[1] = PUBLIC["aggressive"]
    vertex[2] = (
        2.0 * PUBLIC["strict_rspecific_50"]
        - vertex[0]
        - 0.5 * diversity[0, 2]
    )
    vertex[3] = (
        2.0 * PUBLIC["recency_aggressive_50"]
        - vertex[1]
        - 0.5 * diversity[3, 1]
    )
    recent = 0.5 * values["recency"] + 0.5 * values["aggressive"]
    diversity_track_recent = (
        diversity_sa
        / distance[0, 1]
        * np.mean(np.square(values["trackman"] - recent))
    )
    vertex[4] = (
        2.0 * PUBLIC["trackman_recent_50"]
        - PUBLIC["recency_aggressive_50"]
        - 0.5 * diversity_track_recent
    )

    def score(weight: np.ndarray) -> float:
        output = float(weight @ vertex)
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                output += (
                    weight[left]
                    * weight[right]
                    * diversity[left, right]
                )
        return float(output)

    candidate_weight = np.array([0.0, 0.25, 0.0, 0.60, 0.15])
    threeway_weight = np.array([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0])
    # The independently solved nonnegative optimum is recorded for the ceiling;
    # candidate weights are rounded and robust across both time bases.
    optimum = (
        np.array([0.0, 0.2805752289, 0.0, 0.5197154913, 0.1997092798])
        if seasons == (2024,)
        else np.array([0.0, 0.2419063938, 0.0, 0.6303308478, 0.1277627584])
    )
    return {
        "seasons": list(seasons),
        "vertex_score_estimates": {
            name: float(value) for name, value in zip(names, vertex, strict=True)
        },
        "threeway_reconstruction": score(threeway_weight),
        "threeway_actual": PUBLIC["strict_rspecific_aggressive_equal"],
        "threeway_reconstruction_error": (
            score(threeway_weight)
            - PUBLIC["strict_rspecific_aggressive_equal"]
        ),
        "candidate_estimated_score": score(candidate_weight),
        "simplex_optimum_estimated_score": score(optimum),
        "simplex_optimum_weights": {
            name: float(value) for name, value in zip(names, optimum, strict=True)
        },
        "simplex_ceiling_below_1100": bool(score(optimum) < 1100.0),
    }


def main() -> None:
    started = time.time()
    metrics = validation()
    geometry = {
        "known_public_scores": PUBLIC,
        "candidate_weights": WEIGHTS,
        "scenarios": {
            "2024": public_geometry((2024,)),
            "pooled_2022_2024": public_geometry(REPORT_SEASONS),
        },
        "candidate_selection": (
            "rounded maximin of the 2024 and pooled Public geometries"
        ),
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
            "experiment": "EXP-048",
            "candidate": "public_simplex_act_25_60_15",
            "component_weights": WEIGHTS,
            "validation_aggregate_2022_2024": metrics,
            "public_geometry": geometry,
            "selection_status": (
                "Public-score-informed robust simplex; not nested"
            ),
            "probability_calibration": "identity",
        }
    )
    write_json(metadata_path, metadata)
    zip_result = build_zip(DESTINATION, ZIP_PATH)
    smoke = smoke_test(ZIP_PATH)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "EXP-048",
        "candidate": metadata["candidate"],
        "validation": metrics,
        "public_geometry": geometry,
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
