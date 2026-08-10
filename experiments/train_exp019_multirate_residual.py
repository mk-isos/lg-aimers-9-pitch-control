"""EXP-019: 행별 복원한 현재 시즌 다중 비율 residual 후보.

경력 누적 success 외에도 reverse/middle 및 pitchmix를 이전 시즌 종료 상태와
현재 행 누적값의 차이로 복원한다. 테스트 행끼리 집계하지 않는다. 시즌별
동일 가중 residual LightGBM과 현재 시즌 reverse 조건부 그룹 효과를
2021~2024 rolling-origin으로 진단한다.
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

from temporal_multirate_features import attach_training_multirate_features
from temporal_residual_features import (
    TARGET,
    add_static_features,
    attach_training_temporal_features,
)
from train_exp017_rolling_residual import calculate_metrics, segment_metrics
from train_exp018_constrained_multiscale import centered_residual
from train_exp019_stable_monotonic import season_equal_weights


DATA_DIR = Path("./data")
ARTIFACT_ROOT = Path("./artifacts/EXP-019/multirate_residual")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
GROUP_SMOOTHING = 100.0
REVERSE_SMOOTHING = 300.0
REVERSE_WEIGHT = 0.30

BASE_FEATURES = {
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "pitcher_hand",
    "batter_hand",
    "count_index",
    "count_out_index",
    "is_full_count",
    "has_two_strikes",
    "has_three_balls",
    "count_advantage",
    "runner_in_scoring_position",
    "bases_loaded",
    "same_hand",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "multirate_pitcher_control_log_season_n",
    "multirate_pitcher_control_reliability_30",
    "multirate_pitcher_control_success_prior_shrunk_200",
    "multirate_pitcher_control_success_season_global_30",
    "multirate_pitcher_control_reverse_prior_shrunk_200",
    "multirate_pitcher_control_reverse_season_global_30",
    "multirate_pitcher_pitchmix_breaking_prior_shrunk_200",
    "multirate_pitcher_pitchmix_breaking_season_global_30",
}

RICH_FEATURES = BASE_FEATURES | {
    "game_month",
    "li",
    "late_inning",
    "close_game",
    "log_li",
    "temporal_pitcher_prior_exists",
    "temporal_pitcher_log_prior_n",
    "temporal_pitcher_log_season_n",
    "temporal_batter_prior_exists",
    "temporal_batter_log_prior_n",
    "temporal_batter_log_season_n",
    "multirate_pitcher_control_middle_prior_shrunk_200",
    "multirate_pitcher_control_middle_season_global_30",
    "multirate_batter_control_middle_season_global_30",
}

MONOTONE = {
    "asof_pitcher_prev1_game_success_rate": 1,
    "asof_pitcher_prev3_game_success_rate": 1,
    "asof_pitcher_prev5_game_success_rate": 1,
    "multirate_pitcher_control_success_prior_shrunk_200": 1,
    "multirate_pitcher_control_success_season_global_30": 1,
    "multirate_pitcher_control_reverse_prior_shrunk_200": -1,
    "multirate_pitcher_control_reverse_season_global_30": -1,
    "multirate_pitcher_pitchmix_breaking_prior_shrunk_200": -1,
    "multirate_pitcher_pitchmix_breaking_season_global_30": -1,
}


@dataclass(frozen=True)
class Config:
    name: str
    features: frozenset[str]
    monotonic: bool
    iterations: int
    num_leaves: int
    min_child_samples: int


CONFIGS = [
    Config("core_mono_l7_i200", frozenset(BASE_FEATURES), True, 200, 7, 5000),
    Config("core_mono_l7_i400", frozenset(BASE_FEATURES), True, 400, 7, 5000),
    Config("core_free_l7_i200", frozenset(BASE_FEATURES), False, 200, 7, 5000),
    Config("core_mono_l15_i200", frozenset(BASE_FEATURES), True, 200, 15, 5000),
    Config("rich_mono_l7_i200", frozenset(RICH_FEATURES), True, 200, 7, 5000),
]


def prepare_multirate_data():
    test_columns = pd.read_csv(
        DATA_DIR / "test.csv", encoding="utf-8-sig", nrows=0
    ).columns
    base_columns = [column for column in test_columns if column != "row_id"]
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=base_columns + [TARGET],
    )
    train = add_static_features(train)
    train, _ = attach_training_temporal_features(train, target=TARGET)
    train, _, reconstruction_diagnostics = attach_training_multirate_features(
        train, target=TARGET
    )
    y = train[TARGET].to_numpy(dtype=np.float32)
    seasons = train["season"].to_numpy(dtype=np.int16)
    base = train["temporal_base_global_30"].to_numpy(dtype=np.float32)
    diagnostics = train[
        [
            "season",
            "game_month",
            "game_type",
            "temporal_pitcher_season_n",
            "temporal_pitcher_prior_exists",
            "temporal_batter_prior_exists",
        ]
    ].copy()
    return train, diagnostics, y, base, seasons, reconstruction_diagnostics


def multirate_group_correction(
    frame: pd.DataFrame,
    residual: np.ndarray,
    seasons: np.ndarray,
    validation_season: int,
) -> np.ndarray:
    train_mask = (seasons < validation_season) & (seasons >= validation_season - 3)
    validation_mask = seasons == validation_season
    if not train_mask.any():
        return np.zeros(int(validation_mask.sum()), dtype=float)
    keys = pd.DataFrame(
        {
            "count_index": frame["count_index"].to_numpy(dtype=np.int8),
            "pitcher_hand": frame["pitcher_hand"].to_numpy(dtype=np.int8),
            "batter_hand": frame["batter_hand"].to_numpy(dtype=np.int8),
            "season_reverse_bin": np.floor(
                frame[
                    "multirate_pitcher_control_reverse_season_global_30"
                ].to_numpy(dtype=float)
                / 0.05
            ).astype(np.int16),
        }
    )

    def effect(columns: list[str], smoothing: float) -> np.ndarray:
        grouped = keys.loc[train_mask, columns].copy()
        grouped["residual"] = residual[train_mask]
        stats = grouped.groupby(columns, sort=False)["residual"].agg(["sum", "count"])
        values = stats["sum"] / (stats["count"] + smoothing)
        validation_keys = pd.MultiIndex.from_frame(keys.loc[validation_mask, columns])
        return values.reindex(validation_keys).fillna(0.0).to_numpy(dtype=float)

    base_columns = ["count_index", "pitcher_hand", "batter_hand"]
    base_effect = effect(base_columns, GROUP_SMOOTHING)
    reverse_effect = effect(base_columns + ["season_reverse_bin"], REVERSE_SMOOTHING)
    return (1.0 - REVERSE_WEIGHT) * base_effect + REVERSE_WEIGHT * reverse_effect


def game_type_metrics(
    diagnostics: pd.DataFrame,
    validation_mask: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    types = diagnostics.loc[validation_mask, "game_type"].astype(str).to_numpy()
    return {
        game_type: calculate_metrics(targets[types == game_type], predictions[types == game_type])
        for game_type in sorted(np.unique(types))
    }


def run_config(
    config: Config,
    frame: pd.DataFrame,
    diagnostics: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
    residual_target: np.ndarray,
    group_predictions: dict[int, np.ndarray],
) -> dict[str, object]:
    names = sorted(config.features)
    missing = sorted(config.features - set(frame.columns))
    if missing:
        raise ValueError(f"{config.name}: missing features {missing}")
    X = frame[names].to_numpy(dtype=np.float32)
    constraints = [MONOTONE.get(name, 0) if config.monotonic else 0 for name in names]
    artifact_dir = ARTIFACT_ROOT / config.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}

    for validation_season in VALIDATION_SEASONS:
        train_mask = seasons < validation_season
        validation_mask = seasons == validation_season
        targets = y[validation_mask]
        model = LGBMRegressor(
            objective="regression_l2",
            metric="l2",
            n_estimators=config.iterations,
            learning_rate=0.015,
            num_leaves=config.num_leaves,
            min_child_samples=config.min_child_samples,
            max_bin=127,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=1.0,
            reg_lambda=12.0,
            monotone_constraints=constraints,
            monotone_constraints_method="advanced",
            random_state=42,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        started = time.time()
        model.fit(
            X[train_mask],
            residual_target[train_mask],
            sample_weight=season_equal_weights(seasons[train_mask]),
        )
        fit_seconds = time.time() - started
        residual_prediction = model.predict(X[validation_mask]).astype(float)
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "training_seasons": sorted(
                np.unique(seasons[train_mask]).astype(int).tolist()
            ),
            "fit_seconds": fit_seconds,
            "residual_prediction_mean": float(residual_prediction.mean()),
            "base": calculate_metrics(targets, base[validation_mask]),
            "base_plus_group": calculate_metrics(
                targets, group_predictions[validation_season]
            ),
            "feature_importance": {
                name: int(value)
                for name, value in sorted(
                    zip(names, model.feature_importances_, strict=True),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
        }
        for weight in BLEND_WEIGHTS:
            candidate_name = f"group_plus_multirate_w{int(weight * 100):03d}"
            predictions = np.clip(
                group_predictions[validation_season] + weight * residual_prediction,
                0.0,
                1.0,
            )
            fold[candidate_name] = calculate_metrics(targets, predictions)
            fold[f"segments_{candidate_name}"] = segment_metrics(
                diagnostics, validation_mask, targets, predictions
            )
            fold[f"game_type_{candidate_name}"] = game_type_metrics(
                diagnostics, validation_mask, targets, predictions
            )
            np.save(
                artifact_dir
                / f"predictions_{candidate_name}_{validation_season}.npy",
                predictions,
            )
        np.save(artifact_dir / f"targets_{validation_season}.npy", targets)
        folds[str(validation_season)] = fold
        print(
            f"{config.name} {validation_season}: "
            + " ".join(
                f"w{int(weight * 100):03d}="
                f"{fold[f'group_plus_multirate_w{int(weight * 100):03d}']['skill_score_unclipped']:.2f}"
                for weight in BLEND_WEIGHTS
            )
        )

    aggregates: dict[str, object] = {}
    for weight in BLEND_WEIGHTS:
        candidate_name = f"group_plus_multirate_w{int(weight * 100):03d}"
        scores = [
            folds[str(season)][candidate_name]["skill_score_unclipped"]
            for season in REPORT_SEASONS
        ]
        aggregates[candidate_name] = {
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
            "current_fold_labels_used_for_training": False,
            "candidate_comparison_status": "diagnostic; nested selection required",
            "test_row_aggregation": False,
        },
        "model": {
            "features": names,
            "monotonic": config.monotonic,
            "monotone_constraints": dict(zip(names, constraints, strict=True)),
            "iterations": config.iterations,
            "num_leaves": config.num_leaves,
            "min_child_samples": config.min_child_samples,
        },
        "folds": folds,
        "aggregate_2022_2024": aggregates,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    with (artifact_dir / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    started = time.time()
    frame, diagnostics, y, base, seasons, reconstruction = prepare_multirate_data()
    residual_target = centered_residual(y, base, seasons)
    group_predictions: dict[int, np.ndarray] = {}
    for validation_season in VALIDATION_SEASONS:
        validation_mask = seasons == validation_season
        correction = multirate_group_correction(
            frame, residual_target, seasons, validation_season
        )
        group_predictions[validation_season] = np.clip(
            base[validation_mask].astype(float) + correction, 0.0, 1.0
        )

    summaries: dict[str, object] = {}
    for config in CONFIGS:
        result = run_config(
            config,
            frame,
            diagnostics,
            y,
            base,
            seasons,
            residual_target,
            group_predictions,
        )
        summaries[config.name] = result["aggregate_2022_2024"]
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "experiment": "EXP-019",
                "stage": "multirate_residual_candidate_search",
                "selection_status": "not selected; nested selection required",
                "reconstruction_diagnostics": reconstruction,
                "summaries": summaries,
                "total_seconds": time.time() - started,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
