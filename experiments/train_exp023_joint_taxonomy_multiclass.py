"""EXP-023 bounded five-class joint outcome taxonomy experiment."""

from __future__ import annotations

import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from outcome_taxonomy_features import (
    JOINT_CLASS_NAMES,
    assert_label_reconstruction_invariants,
    derive_joint_taxonomy,
    reconstruct_outcome_labels,
)
from train_exp017_rolling_residual import calculate_metrics, prepare_data
from train_exp019_histgb_residual import (
    season_equal_weights,
    select_stable_features,
)
from train_exp022_outcome_taxonomy_multitask import (
    AUX_MODEL_CONFIG,
    REPORT_SEASONS,
    TARGET_SKILL,
    detailed_segments,
    load_frozen_base,
    load_raw_label_frame,
)


EXPERIMENT = "EXP-023"
ARTIFACT_ROOT = Path("./artifacts/EXP-023/joint_taxonomy_multiclass")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
BLEND_WEIGHTS = (0.10, 0.25, 0.50)


def fit_multiclass(
    X: np.ndarray,
    joint: np.ndarray,
    seasons: np.ndarray,
    train_mask: np.ndarray,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(**AUX_MODEL_CONFIG)
    model.fit(
        X[train_mask],
        joint[train_mask].astype(np.int8),
        sample_weight=season_equal_weights(seasons[train_mask]),
    )
    expected_classes = np.arange(len(JOINT_CLASS_NAMES), dtype=np.int64)
    if not np.array_equal(model.classes_, expected_classes):
        raise ValueError(f"missing multiclass label: {model.classes_}")
    return model


def build_temporal_predictions(
    X: np.ndarray,
    joint: np.ndarray,
    seasons: np.ndarray,
    base_by_season: dict[int, np.ndarray],
    targets_by_season: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    success_predictions: dict[int, np.ndarray] = {}
    folds: dict[str, object] = {}
    valid_joint = np.isfinite(joint)
    for validation_season in VALIDATION_SEASONS:
        train_mask = (seasons < validation_season) & valid_joint
        validation_mask = seasons == validation_season
        training_seasons = sorted(
            np.unique(seasons[train_mask]).astype(int).tolist()
        )
        if not training_seasons or max(training_seasons) >= validation_season:
            raise AssertionError("multiclass source season is not strictly prior")
        started = time.time()
        model = fit_multiclass(X, joint, seasons, train_mask)
        probabilities = model.predict_proba(X[validation_mask]).astype(float)
        fit_predict_seconds = time.time() - started
        if probabilities.shape != (int(validation_mask.sum()), 5):
            raise ValueError(f"invalid multiclass prediction shape: {probabilities.shape}")
        if not np.isfinite(probabilities).all():
            raise ValueError("non-finite multiclass probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("multiclass probability rows do not sum to one")
        success = probabilities[:, 0]
        success_predictions[validation_season] = success
        np.save(
            ARTIFACT_ROOT / f"predictions_multiclass_success_{validation_season}.npy",
            success,
        )

        local_joint = joint[validation_mask]
        local_valid = np.isfinite(local_joint)
        validation_distribution = {
            name: int(np.sum(local_joint[local_valid] == index))
            for index, name in enumerate(JOINT_CLASS_NAMES)
        }
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "training_seasons": training_seasons,
            "training_rows": int(train_mask.sum()),
            "training_class_counts": {
                name: int(np.sum(joint[train_mask] == index))
                for index, name in enumerate(JOINT_CLASS_NAMES)
            },
            "validation_rows": int(validation_mask.sum()),
            "validation_labeled_rows": int(local_valid.sum()),
            "validation_class_counts": validation_distribution,
            "validation_multiclass_log_loss": float(
                log_loss(
                    local_joint[local_valid].astype(np.int8),
                    probabilities[local_valid],
                    labels=np.arange(5),
                )
            ),
            "iterations_completed": int(model.n_iter_),
            "fit_predict_seconds": fit_predict_seconds,
            "direct_success_metrics": calculate_metrics(
                targets_by_season[validation_season], success
            ),
            "base_metrics": calculate_metrics(
                targets_by_season[validation_season],
                base_by_season[validation_season],
            ),
        }
        print(
            f"multiclass {validation_season}: rows={len(success)} "
            f"success_mean={success.mean():.4f} "
            f"direct_skill={folds[str(validation_season)]['direct_success_metrics']['skill_score_unclipped']:.2f}"
        )
        del model, probabilities
        gc.collect()
    return success_predictions, folds


def choose_prior_weight(
    validation_season: int,
    success_predictions: dict[int, np.ndarray],
    base_by_season: dict[int, np.ndarray],
    targets_by_season: dict[int, np.ndarray],
) -> tuple[float, dict[str, object]]:
    source_seasons = [s for s in VALIDATION_SEASONS if s < validation_season]
    candidates: dict[str, object] = {}
    for weight in BLEND_WEIGHTS:
        season_metrics: dict[str, object] = {}
        skills: list[float] = []
        for season in source_seasons:
            prediction = (
                (1.0 - weight) * base_by_season[season]
                + weight * success_predictions[season]
            )
            metrics = calculate_metrics(targets_by_season[season], prediction)
            season_metrics[str(season)] = metrics
            skills.append(float(metrics["skill_score_unclipped"]))
        key = f"w{int(weight * 100):03d}"
        candidates[key] = {
            "weight": weight,
            "season_metrics": season_metrics,
            "min_skill": float(np.min(skills)),
            "mean_skill": float(np.mean(skills)),
        }
    selected = max(
        BLEND_WEIGHTS,
        key=lambda weight: (
            float(candidates[f"w{int(weight * 100):03d}"]["min_skill"]),
            float(candidates[f"w{int(weight * 100):03d}"]["mean_skill"]),
            -weight,
        ),
    )
    return selected, {
        "source_seasons": source_seasons,
        "current_fold_labels_used": False,
        "candidates": candidates,
        "selected_weight": selected,
        "tie_break": "max prior min Skill, then mean Skill, then smaller weight",
    }


def samefold_ceiling(
    X: np.ndarray,
    joint: np.ndarray,
    seasons: np.ndarray,
    validation_season: int,
    base: np.ndarray,
    targets: np.ndarray,
) -> dict[str, object]:
    season_mask = seasons == validation_season
    train_mask = season_mask & np.isfinite(joint)
    model = fit_multiclass(X, joint, seasons, train_mask)
    success = model.predict_proba(X[season_mask])[:, 0].astype(float)
    candidates: dict[str, object] = {
        "direct": calculate_metrics(targets, success)
    }
    best_name = "direct"
    best_skill = float(candidates["direct"]["skill_score_unclipped"])
    for weight in BLEND_WEIGHTS:
        name = f"blend_w{int(weight * 100):03d}"
        prediction = (1.0 - weight) * base + weight * success
        candidates[name] = calculate_metrics(targets, prediction)
        skill = float(candidates[name]["skill_score_unclipped"])
        if skill > best_skill:
            best_name = name
            best_skill = skill
    return {
        "deployable": False,
        "current_fold_labels_used": True,
        "candidates": candidates,
        "best_candidate": best_name,
        "best_skill": best_skill,
    }


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw = load_raw_label_frame()
    labels, label_audit = reconstruct_outcome_labels(raw)
    assert_label_reconstruction_invariants(raw, labels, label_audit)
    joint_series, joint_audit = derive_joint_taxonomy(labels)
    if joint_audit["invalid_overlap_rows"] != 0:
        raise ValueError("joint taxonomy is not exhaustive")
    joint = joint_series.to_numpy(dtype=float)

    diagnostics, full_X, y, _unused_base, seasons, feature_names = prepare_data()
    del _unused_base
    if not np.array_equal(raw["control_success"].to_numpy(dtype=np.float32), y):
        raise ValueError("raw/prepared target order mismatch")
    diagnostics = diagnostics.copy()
    diagnostics["game_type"] = raw["game_type"].to_numpy()
    X, selected_features = select_stable_features(full_X, feature_names)
    del full_X, raw, labels, joint_series
    gc.collect()
    base_by_season, targets_by_season, base_alignment = load_frozen_base(
        y,
        seasons,
        VALIDATION_SEASONS,
    )

    success_predictions, model_folds = build_temporal_predictions(
        X,
        joint,
        seasons,
        base_by_season,
        targets_by_season,
    )
    folds: dict[str, object] = {}
    for validation_season in REPORT_SEASONS:
        selected_weight, selection = choose_prior_weight(
            validation_season,
            success_predictions,
            base_by_season,
            targets_by_season,
        )
        candidate_metrics: dict[str, object] = {}
        candidate_predictions: dict[float, np.ndarray] = {}
        for weight in BLEND_WEIGHTS:
            prediction = (
                (1.0 - weight) * base_by_season[validation_season]
                + weight * success_predictions[validation_season]
            )
            candidate_predictions[weight] = prediction
            candidate_metrics[f"w{int(weight * 100):03d}"] = calculate_metrics(
                targets_by_season[validation_season], prediction
            )
            np.save(
                ARTIFACT_ROOT
                / f"predictions_blend_w{int(weight * 100):03d}_{validation_season}.npy",
                prediction,
            )
        selected = candidate_predictions[selected_weight]
        np.save(
            ARTIFACT_ROOT / f"predictions_strict_selected_{validation_season}.npy",
            selected,
        )
        mask = seasons == validation_season
        folds[str(validation_season)] = {
            **model_folds[str(validation_season)],
            "selection": selection,
            "blend_candidates": candidate_metrics,
            "selected": {
                "weight": selected_weight,
                **calculate_metrics(targets_by_season[validation_season], selected),
            },
            "selected_segments": detailed_segments(
                diagnostics,
                mask,
                targets_by_season[validation_season],
                selected,
            ),
            "samefold_multiclass_ceiling": samefold_ceiling(
                X,
                joint,
                seasons,
                validation_season,
                base_by_season[validation_season],
                targets_by_season[validation_season],
            ),
        }
        print(
            f"selected {validation_season}: weight={selected_weight:.2f} "
            f"skill={folds[str(validation_season)]['selected']['skill_score_unclipped']:.2f} "
            f"samefold={folds[str(validation_season)]['samefold_multiclass_ceiling']['best_skill']:.2f}"
        )

    selected_skills = {
        str(season): float(folds[str(season)]["selected"]["skill_score_unclipped"])
        for season in REPORT_SEASONS
    }
    base_skills = {
        str(season): float(folds[str(season)]["base_metrics"]["skill_score_unclipped"])
        for season in REPORT_SEASONS
    }
    samefold_skills = {
        str(season): float(folds[str(season)]["samefold_multiclass_ceiling"]["best_skill"])
        for season in (2023, 2024)
    }
    each_1100 = all(value >= TARGET_SKILL for value in selected_skills.values())
    no_regression = all(
        selected_skills[str(season)] >= base_skills[str(season)]
        for season in REPORT_SEASONS
    )
    uniform_passed = bool(each_1100 and no_regression)
    samefold_passed = all(value >= TARGET_SKILL for value in samefold_skills.values())
    aggregate = {
        "selected_season_skills": selected_skills,
        "base_season_skills": base_skills,
        "samefold_best_2023_2024_skills": samefold_skills,
        "mean_skill": float(np.mean(list(selected_skills.values()))),
        "min_skill": float(np.min(list(selected_skills.values()))),
        "latest_2024_skill": selected_skills["2024"],
        "each_reported_season_skill_at_least_1100": each_1100,
        "no_reported_season_regresses_vs_exp021_strict": no_regression,
        "samefold_2023_2024_ceiling_passed": samefold_passed,
        "uniform_1100_passed": uniform_passed,
        "final_fit_authorized": uniform_passed,
        "zip_creation_authorized": uniform_passed,
        "stop_outcome_taxonomy_branch": not samefold_passed,
    }
    result = {
        "experiment": EXPERIMENT,
        "stage": "joint_taxonomy_multiclass",
        "validation_protocol": {
            "warmup_season": 2021,
            "reported_seasons": REPORT_SEASONS,
            "immutable_base": "EXP-021 fixed lowrank_s300_r6 temporal OOF",
            "current_fold_labels_used_for_temporal_model": False,
            "current_fold_labels_used_for_blend_selection": False,
            "samefold_ceiling_deployable": False,
            "test_csv_read": False,
            "test_row_aggregation": False,
            "calibration": "identity",
        },
        "label_audit": label_audit,
        "joint_taxonomy_audit": joint_audit,
        "model": {
            "class": "HistGradientBoostingClassifier",
            "config": AUX_MODEL_CONFIG,
            "class_names": JOINT_CLASS_NAMES,
            "feature_count": len(selected_features),
            "features": selected_features,
            "fixed_blend_weights": BLEND_WEIGHTS,
        },
        "base_alignment": base_alignment,
        "warmup_fold": model_folds["2021"],
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "qa": {
            "joint_taxonomy_exhaustive": joint_audit["invalid_overlap_rows"] == 0,
            "source_seasons_strictly_prior": True,
            "current_fold_selection_false": True,
            "probability_rows_sum_to_one": True,
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
    print(
        f"saved={output} uniform_1100={uniform_passed} "
        f"stop_branch={aggregate['stop_outcome_taxonomy_branch']}"
    )


if __name__ == "__main__":
    main()
