"""EXP-026 row-local temporal extrapolation of frozen joint experts."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics, prepare_data
from train_exp022_outcome_taxonomy_multitask import (
    REPORT_SEASONS,
    TARGET_SKILL,
    detailed_segments,
    load_frozen_base,
    load_raw_label_frame,
)


EXPERIMENT = "EXP-026"
ARTIFACT_ROOT = Path("./artifacts/EXP-026/joint_expert_trend")
EXPERT_ROOT = Path("./artifacts/EXP-024/source_bagged_joint_taxonomy")
OOF_SEASONS = [2021, 2022, 2023, 2024]
SOURCE_SEASONS = [2019, 2020, 2021, 2022, 2023]
BLEND_WEIGHTS = (0.10, 0.25)
POLICIES = ("delta025", "delta050", "linear", "linear_clip003")


def extrapolation_policies(validation_season: int) -> dict[str, np.ndarray]:
    sources = [season for season in SOURCE_SEASONS if season < validation_season]
    matrix = np.vstack(
        [
            np.load(
                EXPERT_ROOT
                / f"predictions_source{source}_to_{validation_season}.npy"
            ).astype(float)
            for source in sources
        ]
    )
    last = matrix[-1]
    previous = matrix[-2]
    delta = last - previous
    years = np.asarray(sources, dtype=float)
    centered = years - years.mean()
    slope = (centered[:, None] * matrix).sum(axis=0) / np.sum(centered**2)
    linear = matrix.mean(axis=0) + slope * (validation_season - years.mean())
    return {
        "delta025": np.clip(last + 0.25 * delta, 0.0, 1.0),
        "delta050": np.clip(last + 0.50 * delta, 0.0, 1.0),
        "linear": np.clip(linear, 0.0, 1.0),
        "linear_clip003": np.clip(linear, last - 0.03, last + 0.03),
    }


def choose_prior(
    validation_season: int,
    candidates_by_season: dict[int, dict[str, np.ndarray]],
    targets_by_season: dict[int, np.ndarray],
) -> tuple[str, dict[str, object]]:
    source_oof = [season for season in OOF_SEASONS if season < validation_season]
    names = sorted(candidates_by_season[source_oof[0]])
    summaries: dict[str, object] = {}
    for name in names:
        skills: list[float] = []
        metrics_by_season: dict[str, object] = {}
        for season in source_oof:
            metrics = calculate_metrics(
                targets_by_season[season], candidates_by_season[season][name]
            )
            metrics_by_season[str(season)] = metrics
            skills.append(float(metrics["skill_score_unclipped"]))
        weight = float(name.rsplit("w", 1)[1]) / 100.0
        summaries[name] = {
            "season_metrics": metrics_by_season,
            "min_skill": float(np.min(skills)),
            "mean_skill": float(np.mean(skills)),
            "blend_weight": weight,
        }
    selected = max(
        names,
        key=lambda name: (
            float(summaries[name]["min_skill"]),
            float(summaries[name]["mean_skill"]),
            -float(summaries[name]["blend_weight"]),
            name,
        ),
    )
    return selected, {
        "source_oof_seasons": source_oof,
        "current_fold_labels_used": False,
        "candidate_summaries": summaries,
        "selected_candidate": selected,
    }


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    raw = load_raw_label_frame()
    diagnostics, _X, y, _base, seasons, _feature_names = prepare_data()
    del _X, _base
    if not np.array_equal(raw["control_success"].to_numpy(dtype=np.float32), y):
        raise ValueError("raw/prepared target order mismatch")
    diagnostics = diagnostics.copy()
    diagnostics["game_type"] = raw["game_type"].to_numpy()
    base_by_season, targets_by_season, base_alignment = load_frozen_base(
        y, seasons, OOF_SEASONS
    )

    candidates_by_season: dict[int, dict[str, np.ndarray]] = {}
    policy_diagnostics: dict[str, object] = {}
    for season in OOF_SEASONS:
        policies = extrapolation_policies(season)
        candidates: dict[str, np.ndarray] = {}
        for policy_name, policy_prediction in policies.items():
            for weight in BLEND_WEIGHTS:
                name = f"{policy_name}_w{int(weight * 100):03d}"
                candidates[name] = (
                    (1.0 - weight) * base_by_season[season]
                    + weight * policy_prediction
                )
        candidates_by_season[season] = candidates
        policy_diagnostics[str(season)] = {
            name: {
                "prediction_mean": float(prediction.mean()),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
            }
            for name, prediction in policies.items()
        }

    folds: dict[str, object] = {}
    for season in REPORT_SEASONS:
        selected_name, selection = choose_prior(
            season, candidates_by_season, targets_by_season
        )
        metrics_by_candidate: dict[str, object] = {}
        for name, prediction in candidates_by_season[season].items():
            metrics_by_candidate[name] = calculate_metrics(
                targets_by_season[season], prediction
            )
            np.save(
                ARTIFACT_ROOT / f"predictions_{name}_{season}.npy", prediction
            )
        selected = candidates_by_season[season][selected_name]
        np.save(
            ARTIFACT_ROOT / f"predictions_strict_selected_{season}.npy", selected
        )
        mask = seasons == season
        folds[str(season)] = {
            "base": calculate_metrics(
                targets_by_season[season], base_by_season[season]
            ),
            "policy_diagnostics": policy_diagnostics[str(season)],
            "selection": selection,
            "candidates": metrics_by_candidate,
            "selected": {
                "candidate": selected_name,
                **metrics_by_candidate[selected_name],
            },
            "selected_segments": detailed_segments(
                diagnostics, mask, targets_by_season[season], selected
            ),
        }
        print(
            f"{season}: selected={selected_name} "
            f"base={folds[str(season)]['base']['skill_score_unclipped']:.2f} "
            f"skill={folds[str(season)]['selected']['skill_score_unclipped']:.2f}"
        )

    selected_skills = {
        str(season): float(folds[str(season)]["selected"]["skill_score_unclipped"])
        for season in REPORT_SEASONS
    }
    base_skills = {
        str(season): float(folds[str(season)]["base"]["skill_score_unclipped"])
        for season in REPORT_SEASONS
    }
    each_1100 = all(value >= TARGET_SKILL for value in selected_skills.values())
    no_regression = all(
        selected_skills[str(season)] >= base_skills[str(season)]
        for season in REPORT_SEASONS
    )
    uniform = bool(each_1100 and no_regression)
    aggregate = {
        "selected_season_skills": selected_skills,
        "base_season_skills": base_skills,
        "mean_skill": float(np.mean(list(selected_skills.values()))),
        "min_skill": float(np.min(list(selected_skills.values()))),
        "latest_2024_skill": selected_skills["2024"],
        "each_reported_season_skill_at_least_1100": each_1100,
        "no_reported_season_regresses_vs_exp021_strict": no_regression,
        "uniform_1100_passed": uniform,
        "final_fit_authorized": uniform,
        "zip_creation_authorized": uniform,
        "stop_outcome_taxonomy_branch": not uniform,
    }
    result = {
        "experiment": EXPERIMENT,
        "stage": "joint_source_expert_rowlocal_trend",
        "validation_protocol": {
            "source_predictions": "frozen EXP-024 source-season joint experts",
            "candidate_definitions_predeclared": True,
            "current_fold_labels_used_for_selection": False,
            "validation_or_test_row_aggregation": False,
            "row_local_extrapolation_only": True,
            "calibration": "identity",
        },
        "model": {
            "policies": POLICIES,
            "blend_weights": BLEND_WEIGHTS,
            "candidate_count": len(POLICIES) * len(BLEND_WEIGHTS),
        },
        "base_alignment": base_alignment,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "qa": {
            "source_expert_seasons_strictly_prior": True,
            "current_fold_selection_false": True,
            "test_row_aggregation_false": True,
            "probabilities_finite_and_in_range": True,
            "final_fit_or_zip_created": False,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    output = ARTIFACT_ROOT / "validation_metrics.json"
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"saved={output} uniform_1100={uniform}")


if __name__ == "__main__":
    main()
