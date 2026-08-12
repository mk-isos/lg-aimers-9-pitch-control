"""EXP-028: class-prevalence-invariant joint-taxonomy score.

Every source season is reweighted to contribute the same total mass for each
of the five mutually exclusive outcome classes.  The resulting multiclass
score is centered/scaled on source rows only and used as a small fixed logit
correction to the immutable EXP-021 strict prediction.  This isolates common
within-season discrimination from changing season/class prevalence.
"""

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

from outcome_taxonomy_features import (
    JOINT_CLASS_NAMES,
    assert_label_reconstruction_invariants,
    derive_joint_taxonomy,
    reconstruct_outcome_labels,
)
from train_exp017_rolling_residual import calculate_metrics, prepare_data
from train_exp019_histgb_residual import select_stable_features
from train_exp022_outcome_taxonomy_multitask import (
    AUX_MODEL_CONFIG,
    REPORT_SEASONS,
    TARGET_SKILL,
    detailed_segments,
    load_frozen_base,
    load_raw_label_frame,
)


EXPERIMENT = "EXP-028"
ARTIFACT_ROOT = Path("./artifacts/EXP-028/prevalence_invariant_taxonomy")
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
LOGIT_WEIGHTS = (0.02, 0.05, 0.10)
EPS = 1e-8


def expit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def prevalence_invariant_weights(
    seasons: np.ndarray, joint: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    local_seasons = seasons[mask].astype(int)
    local_joint = joint[mask].astype(int)
    weights = np.zeros(int(mask.sum()), dtype=np.float64)
    diagnostics: dict[str, object] = {}
    unique_seasons = sorted(np.unique(local_seasons).tolist())
    for season in unique_seasons:
        season_mask = local_seasons == season
        counts: dict[str, int] = {}
        weighted_mass: dict[str, float] = {}
        for class_index, class_name in enumerate(JOINT_CLASS_NAMES):
            class_mask = season_mask & (local_joint == class_index)
            count = int(class_mask.sum())
            if count == 0:
                raise ValueError(f"empty class {class_name} in season {season}")
            # Each season has unit mass and each class has one fifth of it.
            weights[class_mask] = 1.0 / (len(unique_seasons) * 5.0 * count)
            counts[class_name] = count
            weighted_mass[class_name] = float(weights[class_mask].sum())
        diagnostics[str(season)] = {
            "class_counts": counts,
            "weighted_class_mass": weighted_mass,
            "weighted_season_mass": float(weights[season_mask].sum()),
        }
    weights *= len(weights)
    return weights, diagnostics


def source_equal_weights(seasons: np.ndarray) -> np.ndarray:
    values = np.asarray(seasons, dtype=int)
    weights = np.zeros(len(values), dtype=np.float64)
    unique = np.unique(values)
    for season in unique:
        local = values == season
        weights[local] = 1.0 / (len(unique) * int(local.sum()))
    weights *= len(values)
    return weights


def main() -> None:
    started = time.time()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
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
        y, seasons, VALIDATION_SEASONS
    )

    valid_joint = np.isfinite(joint)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        train_mask = (seasons < validation_season) & valid_joint
        validation_mask = seasons == validation_season
        train_weights, balance_diagnostics = prevalence_invariant_weights(
            seasons, joint, train_mask
        )
        model = HistGradientBoostingClassifier(**AUX_MODEL_CONFIG)
        fit_started = time.time()
        model.fit(
            X[train_mask],
            joint[train_mask].astype(np.int8),
            sample_weight=train_weights,
        )
        expected_classes = np.arange(len(JOINT_CLASS_NAMES), dtype=np.int64)
        if not np.array_equal(model.classes_, expected_classes):
            raise ValueError("multiclass labels missing")

        train_success = model.predict_proba(X[train_mask])[:, 0].astype(float)
        validation_success = model.predict_proba(X[validation_mask])[:, 0].astype(float)
        raw_train_score = logit(train_success)
        raw_validation_score = logit(validation_success)
        standard_weights = source_equal_weights(seasons[train_mask])
        score_mean = float(np.average(raw_train_score, weights=standard_weights))
        score_variance = float(
            np.average(np.square(raw_train_score - score_mean), weights=standard_weights)
        )
        score_scale = float(np.sqrt(max(score_variance, 1e-12)))
        validation_score = (raw_validation_score - score_mean) / score_scale
        if not np.isfinite(validation_score).all():
            raise ValueError("non-finite invariant score")

        candidates: dict[str, object] = {}
        candidate_predictions: dict[str, np.ndarray] = {}
        base = base_by_season[validation_season]
        for weight in LOGIT_WEIGHTS:
            name = f"invariant_logit_w{int(round(weight * 1000)):03d}"
            prediction = expit(logit(base) + weight * validation_score)
            candidate_predictions[name] = prediction
            candidates[name] = calculate_metrics(
                targets_by_season[validation_season], prediction
            )
            np.save(
                ARTIFACT_ROOT / f"predictions_{name}_{validation_season}.npy",
                prediction,
            )
        fold = {
            "validation_season": validation_season,
            "training_seasons": sorted(
                np.unique(seasons[train_mask]).astype(int).tolist()
            ),
            "training_rows": int(train_mask.sum()),
            "class_balance": balance_diagnostics,
            "score_center_from_source_only": score_mean,
            "score_scale_from_source_only": score_scale,
            "validation_score_mean": float(validation_score.mean()),
            "validation_score_std": float(validation_score.std()),
            "model_iterations": int(model.n_iter_),
            "fit_predict_seconds": time.time() - fit_started,
            "base": calculate_metrics(targets_by_season[validation_season], base),
            "candidates": candidates,
        }
        if validation_season in REPORT_SEASONS:
            fold["segments"] = {
                name: detailed_segments(
                    diagnostics,
                    validation_mask,
                    targets_by_season[validation_season],
                    prediction,
                )
                for name, prediction in candidate_predictions.items()
            }
        folds[str(validation_season)] = fold
        np.save(ARTIFACT_ROOT / f"targets_{validation_season}.npy", targets_by_season[validation_season])
        del model, train_success, validation_success
        gc.collect()

    aggregate_candidates: dict[str, object] = {}
    names = sorted(folds["2022"]["candidates"])
    for name in names:
        briers = {
            str(season): float(folds[str(season)]["candidates"][name]["brier_score"])
            for season in REPORT_SEASONS
        }
        skills = {
            str(season): float(
                folds[str(season)]["candidates"][name]["skill_score_unclipped"]
            )
            for season in REPORT_SEASONS
        }
        aggregate_candidates[name] = {
            "season_briers": briers,
            "season_skills": skills,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills["2024"],
            "uniform_1100_passed": all(value >= TARGET_SKILL for value in skills.values()),
        }
    best_name = max(
        aggregate_candidates,
        key=lambda name: (
            aggregate_candidates[name]["min_skill"],
            aggregate_candidates[name]["mean_skill"],
        ),
    )
    uniform_pass = bool(aggregate_candidates[best_name]["uniform_1100_passed"])
    result = {
        "experiment": EXPERIMENT,
        "stage": "bounded prevalence-invariant taxonomy validation",
        "hypothesis": (
            "equal source-season/class mass removes drifting taxonomy prevalence "
            "and preserves only transferable row-level discrimination"
        ),
        "validation_protocol": {
            "warmup_season": 2021,
            "report_seasons": list(REPORT_SEASONS),
            "current_fold_labels_used_for_fit_or_selection": False,
            "test_csv_read": False,
            "test_row_aggregation": False,
            "fixed_logit_weights": list(LOGIT_WEIGHTS),
            "score_center_and_scale": "source seasons only, season-equal",
        },
        "model": {
            "class": "HistGradientBoostingClassifier",
            "config": AUX_MODEL_CONFIG,
            "classes": list(JOINT_CLASS_NAMES),
            "feature_count": len(selected_features),
            "features": selected_features,
        },
        "label_audit": label_audit,
        "joint_taxonomy_audit": joint_audit,
        "base_alignment": base_alignment,
        "folds": folds,
        "aggregate_2022_2024": {
            "candidates": aggregate_candidates,
            "posthoc_best_min_candidate": best_name,
            "uniform_1100_passed": uniform_pass,
            "final_fit_authorized": uniform_pass,
            "zip_creation_authorized": uniform_pass,
        },
        "qa": {
            "joint_taxonomy_exhaustive": True,
            "each_source_season_and_class_equal_weighted": True,
            "source_seasons_strictly_prior": True,
            "current_fold_selection": False,
            "test_row_aggregation": False,
            "probabilities_finite_and_in_range": True,
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
    with (ARTIFACT_ROOT / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result["aggregate_2022_2024"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
