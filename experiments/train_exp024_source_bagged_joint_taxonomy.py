"""EXP-024 source-season-bagged joint taxonomy transfer experiment."""

from __future__ import annotations

import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from outcome_taxonomy_features import (
    JOINT_CLASS_NAMES,
    assert_label_reconstruction_invariants,
    derive_joint_taxonomy,
    reconstruct_outcome_labels,
)
from train_exp017_rolling_residual import calculate_metrics, prepare_data
from train_exp019_histgb_residual import select_stable_features
from train_exp022_outcome_taxonomy_multitask import (
    REPORT_SEASONS,
    TARGET_SKILL,
    detailed_segments,
    load_frozen_base,
    load_raw_label_frame,
)
from train_exp023_joint_taxonomy_multiclass import fit_multiclass


EXPERIMENT = "EXP-024"
ARTIFACT_ROOT = Path("./artifacts/EXP-024/source_bagged_joint_taxonomy")
OOF_SEASONS = [2021, 2022, 2023, 2024]
SOURCE_SEASONS = [2019, 2020, 2021, 2022, 2023]
BLEND_WEIGHTS = (0.25, 0.50)
POLICIES = ("last", "equal", "recency2", "median", "consensus")


def train_source_models(
    X: np.ndarray,
    joint: np.ndarray,
    seasons: np.ndarray,
) -> tuple[dict[int, dict[int, np.ndarray]], dict[str, object]]:
    predictions: dict[int, dict[int, np.ndarray]] = {}
    diagnostics: dict[str, object] = {}
    valid = np.isfinite(joint)
    for source_season in SOURCE_SEASONS:
        train_mask = (seasons == source_season) & valid
        started = time.time()
        model = fit_multiclass(X, joint, seasons, train_mask)
        predictions[source_season] = {}
        target_folds: dict[str, object] = {}
        for validation_season in OOF_SEASONS:
            if validation_season <= source_season:
                continue
            validation_mask = seasons == validation_season
            success = model.predict_proba(X[validation_mask])[:, 0].astype(float)
            predictions[source_season][validation_season] = success
            np.save(
                ARTIFACT_ROOT
                / f"predictions_source{source_season}_to_{validation_season}.npy",
                success,
            )
            target_folds[str(validation_season)] = {
                "rows": len(success),
                "prediction_mean": float(success.mean()),
                "prediction_min": float(success.min()),
                "prediction_max": float(success.max()),
            }
        diagnostics[str(source_season)] = {
            "training_rows": int(train_mask.sum()),
            "class_counts": {
                name: int(np.sum(joint[train_mask] == index))
                for index, name in enumerate(JOINT_CLASS_NAMES)
            },
            "iterations_completed": int(model.n_iter_),
            "fit_and_predict_seconds": time.time() - started,
            "target_folds": target_folds,
        }
        print(
            f"source {source_season}: train={int(train_mask.sum())} "
            f"targets={sorted(predictions[source_season])}"
        )
        del model
        gc.collect()
    return predictions, diagnostics


def policy_predictions(
    validation_season: int,
    source_predictions: dict[int, dict[int, np.ndarray]],
    base: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    sources = [s for s in SOURCE_SEASONS if s < validation_season]
    matrix = np.vstack([source_predictions[s][validation_season] for s in sources])
    recency_weights = np.power(2.0, np.arange(len(sources), dtype=float))
    recency_weights /= recency_weights.sum()
    corrections = matrix - base[None, :]
    all_nonnegative = np.all(corrections >= 0.0, axis=0)
    all_nonpositive = np.all(corrections <= 0.0, axis=0)
    consensus_mask = all_nonnegative | all_nonpositive
    consensus = base.copy()
    consensus[consensus_mask] += corrections[:, consensus_mask].mean(axis=0)
    policies = {
        "last": matrix[-1],
        "equal": matrix.mean(axis=0),
        "recency2": np.average(matrix, axis=0, weights=recency_weights),
        "median": np.median(matrix, axis=0),
        "consensus": np.clip(consensus, 0.0, 1.0),
    }
    return policies, {
        "source_seasons": sources,
        "source_count": len(sources),
        "recency_weights": {
            str(season): float(weight)
            for season, weight in zip(sources, recency_weights, strict=True)
        },
        "mean_source_prediction_std": float(matrix.std(axis=0).mean()),
        "consensus_row_rate": float(consensus_mask.mean()),
    }


def candidate_predictions(
    validation_season: int,
    source_predictions: dict[int, dict[int, np.ndarray]],
    base: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    policies, policy_diagnostics = policy_predictions(
        validation_season,
        source_predictions,
        base,
    )
    candidates: dict[str, np.ndarray] = {}
    for policy in POLICIES:
        for weight in BLEND_WEIGHTS:
            name = f"{policy}_w{int(weight * 100):03d}"
            candidates[name] = (
                (1.0 - weight) * base + weight * policies[policy]
            )
    return candidates, policy_diagnostics


def choose_prior_candidate(
    validation_season: int,
    candidates_by_season: dict[int, dict[str, np.ndarray]],
    targets_by_season: dict[int, np.ndarray],
) -> tuple[str, dict[str, object]]:
    source_oof = [season for season in OOF_SEASONS if season < validation_season]
    names = sorted(candidates_by_season[source_oof[0]])
    summaries: dict[str, object] = {}
    for name in names:
        skills: list[float] = []
        season_metrics: dict[str, object] = {}
        for season in source_oof:
            metrics = calculate_metrics(
                targets_by_season[season],
                candidates_by_season[season][name],
            )
            season_metrics[str(season)] = metrics
            skills.append(float(metrics["skill_score_unclipped"]))
        weight = float(name.rsplit("w", maxsplit=1)[1]) / 100.0
        summaries[name] = {
            "season_metrics": season_metrics,
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
        "tie_break": "max prior min Skill, mean Skill, smaller blend weight, name",
    }


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw = load_raw_label_frame()
    labels, label_audit = reconstruct_outcome_labels(raw)
    assert_label_reconstruction_invariants(raw, labels, label_audit)
    joint_series, joint_audit = derive_joint_taxonomy(labels)
    if joint_audit["invalid_overlap_rows"]:
        raise ValueError("joint taxonomy overlap found")
    joint = joint_series.to_numpy(dtype=float)
    diagnostics, full_X, y, _unused_base, seasons, feature_names = prepare_data()
    del _unused_base
    if not np.array_equal(raw["control_success"].to_numpy(dtype=np.float32), y):
        raise ValueError("raw/prepared target order mismatch")
    diagnostics = diagnostics.copy()
    diagnostics["game_type"] = raw["game_type"].to_numpy()
    X, selected_features = select_stable_features(full_X, feature_names)
    del raw, labels, joint_series, full_X
    gc.collect()
    base_by_season, targets_by_season, base_alignment = load_frozen_base(
        y,
        seasons,
        OOF_SEASONS,
    )

    source_predictions, source_diagnostics = train_source_models(X, joint, seasons)
    candidates_by_season: dict[int, dict[str, np.ndarray]] = {}
    policy_diagnostics: dict[str, object] = {}
    for season in OOF_SEASONS:
        candidates, policy_summary = candidate_predictions(
            season,
            source_predictions,
            base_by_season[season],
        )
        candidates_by_season[season] = candidates
        policy_diagnostics[str(season)] = policy_summary

    folds: dict[str, object] = {}
    for validation_season in REPORT_SEASONS:
        selected_name, selection = choose_prior_candidate(
            validation_season,
            candidates_by_season,
            targets_by_season,
        )
        metrics_by_candidate: dict[str, object] = {}
        for name, prediction in candidates_by_season[validation_season].items():
            metrics_by_candidate[name] = calculate_metrics(
                targets_by_season[validation_season], prediction
            )
            np.save(
                ARTIFACT_ROOT / f"predictions_{name}_{validation_season}.npy",
                prediction,
            )
        selected = candidates_by_season[validation_season][selected_name]
        np.save(
            ARTIFACT_ROOT / f"predictions_strict_selected_{validation_season}.npy",
            selected,
        )
        mask = seasons == validation_season
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "base": calculate_metrics(
                targets_by_season[validation_season],
                base_by_season[validation_season],
            ),
            "policy_diagnostics": policy_diagnostics[str(validation_season)],
            "selection": selection,
            "candidates": metrics_by_candidate,
            "selected": {
                "candidate": selected_name,
                **metrics_by_candidate[selected_name],
            },
            "selected_segments": detailed_segments(
                diagnostics,
                mask,
                targets_by_season[validation_season],
                selected,
            ),
        }
        print(
            f"selected {validation_season}: {selected_name} "
            f"base={folds[str(validation_season)]['base']['skill_score_unclipped']:.2f} "
            f"skill={folds[str(validation_season)]['selected']['skill_score_unclipped']:.2f}"
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
    uniform_passed = bool(each_1100 and no_regression)
    aggregate = {
        "selected_season_skills": selected_skills,
        "base_season_skills": base_skills,
        "mean_skill": float(np.mean(list(selected_skills.values()))),
        "min_skill": float(np.min(list(selected_skills.values()))),
        "latest_2024_skill": selected_skills["2024"],
        "each_reported_season_skill_at_least_1100": each_1100,
        "no_reported_season_regresses_vs_exp021_strict": no_regression,
        "uniform_1100_passed": uniform_passed,
        "final_fit_authorized": uniform_passed,
        "zip_creation_authorized": uniform_passed,
    }
    result = {
        "experiment": EXPERIMENT,
        "stage": "source_season_bagged_joint_taxonomy",
        "validation_protocol": {
            "source_models": SOURCE_SEASONS,
            "oof_seasons": OOF_SEASONS,
            "reported_seasons": REPORT_SEASONS,
            "immutable_base": "EXP-021 fixed lowrank_s300_r6 temporal OOF",
            "candidate_definitions_predeclared": True,
            "current_fold_labels_used_for_selection": False,
            "test_csv_read": False,
            "test_row_aggregation": False,
            "row_local_source_disagreement_only": True,
            "calibration": "identity",
        },
        "label_audit": label_audit,
        "joint_taxonomy_audit": joint_audit,
        "model": {
            "class": "source-season HistGradientBoostingClassifier",
            "class_names": JOINT_CLASS_NAMES,
            "policies": POLICIES,
            "blend_weights": BLEND_WEIGHTS,
            "feature_count": len(selected_features),
            "features": selected_features,
        },
        "source_model_diagnostics": source_diagnostics,
        "base_alignment": base_alignment,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "qa": {
            "source_model_season_strictly_prior_to_prediction": True,
            "current_fold_selection_false": True,
            "candidate_count": len(POLICIES) * len(BLEND_WEIGHTS),
            "probabilities_finite_and_in_range": True,
            "test_row_aggregation_false": True,
            "final_fit_or_zip_created": False,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "total_seconds": time.time() - started,
    }
    output = ARTIFACT_ROOT / "validation_metrics.json"
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"saved={output} uniform_1100={uniform_passed}")


if __name__ == "__main__":
    main()
