"""EXP-019: all-row group offset + R 전용 ordered CatBoost residual."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import catboost
import numpy as np
from catboost import CatBoostRegressor

import train_exp019_multirate_residual as multirate
from train_exp017_rolling_residual import calculate_metrics
from train_exp019_catboost_residual import (
    BLEND_WEIGHTS,
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
)
from train_exp019_r_full_residual import original_group_correction
from train_exp019_stable_monotonic import season_equal_weights


ARTIFACT_ROOT = Path("./artifacts/EXP-019/r_catboost")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]


def main() -> None:
    started = time.time()
    frame, _, y, base, seasons, reconstruction = multirate.prepare_multirate_data()
    game_types = frame["game_type"].astype(str).to_numpy()
    is_r = game_types == "R"
    initial_residual = multirate.centered_residual(y, base, seasons)
    group_all = np.empty(len(y), dtype=np.float64)
    group_reported: dict[int, np.ndarray] = {}
    for season in sorted(np.unique(seasons).astype(int).tolist()):
        mask = seasons == season
        correction = original_group_correction(frame, initial_residual, seasons, season)
        prediction = np.clip(base[mask].astype(float) + correction, 0.0, 1.0)
        group_all[mask] = prediction
        if season in VALIDATION_SEASONS:
            group_reported[season] = prediction
    residual_target = (y.astype(float) - group_all).astype(np.float32)
    for season in np.unique(seasons):
        mask = (seasons == season) & is_r
        residual_target[mask] -= residual_target[mask].mean()

    feature_names = [column for column in frame.columns if column not in DROP_COLUMNS]
    model_frame = frame[feature_names].copy()
    for column in CATEGORICAL_COLUMNS:
        model_frame[column] = model_frame[column].fillna("__MISSING__").astype(str)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}

    for validation_season in VALIDATION_SEASONS:
        train_mask = (seasons < validation_season) & is_r
        validation_mask = seasons == validation_season
        validation_r = validation_mask & is_r
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=200,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=12.0,
            random_strength=0.5,
            border_count=128,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            rsm=0.8,
            has_time=True,
            random_seed=42,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
        fit_started = time.time()
        model.fit(
            model_frame.loc[train_mask],
            residual_target[train_mask],
            cat_features=CATEGORICAL_COLUMNS,
            sample_weight=season_equal_weights(seasons[train_mask]),
        )
        correction = np.zeros(int(validation_mask.sum()), dtype=float)
        local_r = is_r[validation_mask]
        correction[local_r] = model.predict(model_frame.loc[validation_r]).astype(float)
        targets = y[validation_mask]
        validation_types = game_types[validation_mask]
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "fit_seconds": time.time() - fit_started,
            "correction_mean_R": float(correction[local_r].mean()),
            "feature_importance": {
                name: float(value)
                for name, value in sorted(
                    zip(feature_names, model.feature_importances_, strict=True),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
        }
        for weight in BLEND_WEIGHTS:
            candidate = f"branch_w{int(weight * 100):03d}"
            predictions = np.clip(
                group_reported[validation_season] + weight * correction,
                0.0,
                1.0,
            )
            fold[candidate] = calculate_metrics(targets, predictions)
            fold[f"regimes_{candidate}"] = {
                regime: calculate_metrics(
                    targets[validation_types == regime],
                    predictions[validation_types == regime],
                )
                for regime in sorted(np.unique(validation_types))
            }
            np.save(
                ARTIFACT_ROOT
                / f"predictions_{candidate}_{validation_season}.npy",
                predictions,
            )
        np.save(ARTIFACT_ROOT / f"targets_{validation_season}.npy", targets)
        folds[str(validation_season)] = fold
        print(
            f"r_catboost {validation_season}: "
            + " ".join(
                f"w{int(weight * 100):03d}="
                f"{fold[f'branch_w{int(weight * 100):03d}']['skill_score_unclipped']:.2f}"
                for weight in BLEND_WEIGHTS
            )
        )

    aggregate: dict[str, object] = {}
    for weight in BLEND_WEIGHTS:
        candidate = f"branch_w{int(weight * 100):03d}"
        scores = [
            folds[str(season)][candidate]["skill_score_unclipped"]
            for season in REPORT_SEASONS
        ]
        aggregate[candidate] = {
            "mean_skill": float(np.mean(scores)),
            "min_skill": float(np.min(scores)),
            "latest_2024_skill": float(scores[-1]),
        }
    result = {
        "experiment": "EXP-019",
        "candidate": "all_group_plus_R_ordered_catboost_d8_i200",
        "validation_protocol": {
            "outer_folds": VALIDATION_SEASONS,
            "reported_folds": REPORT_SEASONS,
            "R_model_training": "past R rows only",
            "F_prediction": "past-only all-row group offset",
            "current_fold_labels_used_for_training": False,
            "test_row_aggregation": False,
            "candidate_comparison_status": "diagnostic; nested selection required",
        },
        "model": {
            "features": feature_names,
            "categorical_features": CATEGORICAL_COLUMNS,
            "iterations": 200,
            "depth": 8,
            "learning_rate": 0.03,
            "l2_leaf_reg": 12.0,
        },
        "reconstruction_diagnostics": reconstruction,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "catboost": catboost.__version__,
        },
        "total_seconds": time.time() - started,
    }
    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
