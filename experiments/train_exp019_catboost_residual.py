"""EXP-019: ordered CatBoost temporal-group residual 비교.

LightGBM과 다른 ordered categorical inductive bias가 보완 신호를 제공하는지
검증한다. 각 outer fold는 과거 시즌만 학습하며 residual target은 각 학습 행의
past-only group OOF 예측을 뺀 값이다.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import train_exp019_multirate_residual as multirate
from train_exp017_rolling_residual import calculate_metrics
from train_exp019_r_full_residual import original_group_correction
from train_exp019_stable_monotonic import season_equal_weights


ARTIFACT_ROOT = Path("./artifacts/EXP-019/catboost_residual")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
CATEGORICAL_COLUMNS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "count_index",
]
DROP_COLUMNS = {"row_id", "control_success", "season"}


@dataclass(frozen=True)
class Config:
    name: str
    iterations: int
    depth: int
    l2_leaf_reg: float


CONFIGS = [
    Config("ordered_d6_i300", 300, 6, 10.0),
    Config("ordered_d8_i200", 200, 8, 12.0),
]


def main() -> None:
    started = time.time()
    frame, _, y, base, seasons, reconstruction = multirate.prepare_multirate_data()
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
        mask = seasons == season
        residual_target[mask] -= residual_target[mask].mean()

    feature_names = [column for column in frame.columns if column not in DROP_COLUMNS]
    model_frame = frame[feature_names].copy()
    for column in CATEGORICAL_COLUMNS:
        model_frame[column] = model_frame[column].fillna("__MISSING__").astype(str)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}

    for config in CONFIGS:
        artifact_dir = ARTIFACT_ROOT / config.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        folds: dict[str, object] = {}
        for validation_season in VALIDATION_SEASONS:
            train_mask = seasons < validation_season
            validation_mask = seasons == validation_season
            model = CatBoostRegressor(
                loss_function="RMSE",
                iterations=config.iterations,
                depth=config.depth,
                learning_rate=0.03,
                l2_leaf_reg=config.l2_leaf_reg,
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
            correction = model.predict(model_frame.loc[validation_mask]).astype(float)
            targets = y[validation_mask]
            fold: dict[str, object] = {
                "validation_season": validation_season,
                "training_seasons": sorted(
                    np.unique(seasons[train_mask]).astype(int).tolist()
                ),
                "fit_seconds": time.time() - fit_started,
                "correction_mean": float(correction.mean()),
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
                candidate = f"group_plus_cat_w{int(weight * 100):03d}"
                predictions = np.clip(
                    group_reported[validation_season] + weight * correction,
                    0.0,
                    1.0,
                )
                fold[candidate] = calculate_metrics(targets, predictions)
                np.save(
                    artifact_dir
                    / f"predictions_{candidate}_{validation_season}.npy",
                    predictions,
                )
            np.save(artifact_dir / f"targets_{validation_season}.npy", targets)
            folds[str(validation_season)] = fold
            print(
                f"{config.name} {validation_season}: "
                + " ".join(
                    f"w{int(weight * 100):03d}="
                    f"{fold[f'group_plus_cat_w{int(weight * 100):03d}']['skill_score_unclipped']:.2f}"
                    for weight in BLEND_WEIGHTS
                )
            )

        aggregate: dict[str, object] = {}
        for weight in BLEND_WEIGHTS:
            candidate = f"group_plus_cat_w{int(weight * 100):03d}"
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
            "candidate": config.name,
            "validation_protocol": {
                "outer_folds": VALIDATION_SEASONS,
                "reported_folds": REPORT_SEASONS,
                "target": "season-centered y minus past-only group OOF",
                "current_fold_labels_used_for_training": False,
                "test_row_aggregation": False,
                "candidate_comparison_status": "diagnostic; nested selection required",
            },
            "model": {
                "features": feature_names,
                "categorical_features": CATEGORICAL_COLUMNS,
                "iterations": config.iterations,
                "depth": config.depth,
                "learning_rate": 0.03,
                "l2_leaf_reg": config.l2_leaf_reg,
                "has_time": True,
            },
            "reconstruction_diagnostics": reconstruction,
            "folds": folds,
            "aggregate_2022_2024": aggregate,
            "environment": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "catboost": catboost.__version__,
            },
        }
        with (artifact_dir / "validation_metrics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        summaries[config.name] = aggregate

    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "experiment": "EXP-019",
                "stage": "ordered_catboost_group_residual",
                "selection_status": "not selected; nested selection required",
                "summaries": summaries,
                "total_seconds": time.time() - started,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
