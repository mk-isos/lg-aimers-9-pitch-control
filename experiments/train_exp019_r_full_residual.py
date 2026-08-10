"""EXP-019: 기존 all-row group offset + R 전용 full residual LightGBM.

F 체제 전환의 영향을 residual 학습에서 제거하되, F 예측에는 과거 전체에서
학습한 안정적 count/hand group 기준값을 유지한다. R 내부 모델 용량 부족인지
확인하기 위해 ID·team·season을 제외한 전체 공식/temporal 피처를 사용한다.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import train_exp019_multirate_residual as multirate
from train_exp017_rolling_residual import calculate_metrics
from train_exp019_stable_monotonic import season_equal_weights


ARTIFACT_ROOT = Path("./artifacts/EXP-019/r_full_residual")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
DROP_COLUMNS = {
    "row_id",
    "control_success",
    "season",
    "game_type",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
}
CATEGORICAL_COLUMNS = ["top_bottom", "base_state"]


@dataclass(frozen=True)
class Config:
    name: str
    iterations: int
    num_leaves: int
    min_child_samples: int


CONFIGS = [
    Config("rfull_l15_m3000_i200", 200, 15, 3000),
    Config("rfull_l31_m2000_i300", 300, 31, 2000),
    Config("rfull_l63_m1000_i300", 300, 63, 1000),
]


def original_group_correction(
    frame: pd.DataFrame,
    residual: np.ndarray,
    seasons: np.ndarray,
    validation_season: int,
) -> np.ndarray:
    train_mask = (
        (seasons < validation_season)
        & (seasons >= validation_season - 3)
    )
    validation_mask = seasons == validation_season
    if not train_mask.any():
        return np.zeros(int(validation_mask.sum()), dtype=float)
    keys = pd.DataFrame(
        {
            "count_index": frame["count_index"].to_numpy(dtype=np.int8),
            "pitcher_hand": frame["pitcher_hand"].to_numpy(dtype=np.int8),
            "batter_hand": frame["batter_hand"].to_numpy(dtype=np.int8),
            "reverse_bin": np.where(
                frame["asof_pitcher_reverse_rate"].notna(),
                np.floor(frame["asof_pitcher_reverse_rate"] / 0.05),
                -1,
            ).astype(np.int16),
        }
    )

    def effect(columns: list[str], smoothing: float) -> np.ndarray:
        grouped = keys.loc[train_mask, columns].copy()
        grouped["residual"] = residual[train_mask]
        stats = grouped.groupby(columns, sort=False)["residual"].agg(["sum", "count"])
        values = stats["sum"] / (stats["count"] + smoothing)
        index = pd.MultiIndex.from_frame(keys.loc[validation_mask, columns])
        return values.reindex(index).fillna(0.0).to_numpy(dtype=float)

    coarse_columns = ["count_index", "pitcher_hand", "batter_hand"]
    coarse = effect(coarse_columns, 100.0)
    reverse = effect(coarse_columns + ["reverse_bin"], 300.0)
    return 0.7 * coarse + 0.3 * reverse


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

    model_columns = [
        column
        for column in frame.columns
        if column not in DROP_COLUMNS and column not in CATEGORICAL_COLUMNS
    ]
    numeric = frame[model_columns].select_dtypes(include=[np.number]).astype(np.float32)
    categorical = pd.get_dummies(
        frame[CATEGORICAL_COLUMNS],
        dummy_na=True,
        dtype=np.int8,
    )
    feature_names = numeric.columns.tolist() + categorical.columns.tolist()
    X = np.column_stack(
        [numeric.to_numpy(dtype=np.float32), categorical.to_numpy(dtype=np.float32)]
    )
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}

    for config in CONFIGS:
        artifact_dir = ARTIFACT_ROOT / config.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        folds: dict[str, object] = {}
        for validation_season in VALIDATION_SEASONS:
            train_mask = (seasons < validation_season) & is_r
            validation_mask = seasons == validation_season
            validation_r = validation_mask & is_r
            model = LGBMRegressor(
                objective="regression_l2",
                metric="l2",
                n_estimators=config.iterations,
                learning_rate=0.015,
                num_leaves=config.num_leaves,
                min_child_samples=config.min_child_samples,
                max_bin=255,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                reg_alpha=0.5,
                reg_lambda=8.0,
                random_state=42,
                n_jobs=-1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            fit_started = time.time()
            model.fit(
                X[train_mask],
                residual_target[train_mask],
                sample_weight=season_equal_weights(seasons[train_mask]),
            )
            correction = np.zeros(int(validation_mask.sum()), dtype=float)
            local_r = is_r[validation_mask]
            correction[local_r] = model.predict(X[validation_r]).astype(float)
            targets = y[validation_mask]
            validation_types = game_types[validation_mask]
            fold: dict[str, object] = {
                "validation_season": validation_season,
                "fit_seconds": time.time() - fit_started,
                "correction_mean_R": float(correction[local_r].mean()),
                "feature_importance": {
                    name: int(value)
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
            "candidate": config.name,
            "validation_protocol": {
                "outer_folds": VALIDATION_SEASONS,
                "reported_folds": REPORT_SEASONS,
                "offset": "past-only all-row count/hand/reverse group",
                "R_model_training": "past R rows only",
                "F_prediction": "all-row temporal group offset only",
                "current_fold_labels_used_for_training": False,
                "test_row_aggregation": False,
                "candidate_comparison_status": "diagnostic; nested selection required",
            },
            "model": {
                "features": feature_names,
                "iterations": config.iterations,
                "num_leaves": config.num_leaves,
                "min_child_samples": config.min_child_samples,
            },
            "reconstruction_diagnostics": reconstruction,
            "folds": folds,
            "aggregate_2022_2024": aggregate,
            "environment": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "lightgbm": lgb.__version__,
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
                "stage": "all_group_plus_R_full_residual",
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
