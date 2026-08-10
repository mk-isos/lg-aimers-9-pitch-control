"""EXP-019: 시즌 방향성이 안정적인 피처만 사용하는 residual LightGBM.

EXP-018의 recent residual은 팀 ID와 시즌별 상관 방향이 바뀐 비율 피처에
의존했다. 이 실험은 팀 ID, season, ball/strike rate, pitchmix, 타자 성공률을
제외하고 count, hand, 투수 성공률, reverse/middle rate, 최근 성공률과
시점 안전한 투수 표본 신뢰도만 사용한다.

각 outer fold는 검증 시즌보다 이전 시즌만 학습한다. 이 스크립트의 config 및
blend 비교는 후보 탐색 진단이며, 최종 선택 점수는 별도의 nested temporal
selection으로 계산해야 한다.
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

from train_exp017_rolling_residual import calculate_metrics, prepare_data
from train_exp018_constrained_multiscale import (
    build_group_keys,
    centered_residual,
    group_correction,
)


ARTIFACT_ROOT = Path("./artifacts/EXP-019/stable_monotonic")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]

STRICT_FEATURES = {
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
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "count_index",
    "count_out_index",
    "is_full_count",
    "has_two_strikes",
    "has_three_balls",
    "count_advantage",
    "runner_in_scoring_position",
    "bases_loaded",
    "same_hand",
    "temporal_pitcher_prior_exists",
    "temporal_pitcher_log_prior_n",
    "temporal_pitcher_prior_rate_shrunk_200",
    "temporal_pitcher_log_season_n",
    "temporal_pitcher_season_global_30",
    "temporal_pitcher_reliability_30",
}

PRESSURE_FEATURES = STRICT_FEATURES | {
    "game_month",
    "game_dayofweek",
    "run_total_before",
    "score_diff_pitcher_team",
    "li",
    "runner_in_scoring_position",
    "late_inning",
    "close_game",
    "log_li",
    "score_pressure",
}

TEMPORAL_FEATURES = STRICT_FEATURES | {
    "temporal_pitcher_prior_rate",
    "temporal_pitcher_season_rate",
    "temporal_pitcher_season_minus_prior_rate",
    "temporal_pitcher_season_global_10",
    "temporal_pitcher_season_global_100",
    "temporal_pitcher_reliability_10",
    "temporal_pitcher_reliability_100",
    "temporal_batter_prior_exists",
    "temporal_batter_log_prior_n",
    "temporal_batter_log_season_n",
    "temporal_batter_season_global_30",
    "temporal_batter_reliability_30",
}

MONOTONE_DIRECTIONS = {
    "asof_pitcher_success_rate": 1,
    "asof_pitcher_reverse_rate": -1,
    "asof_pitcher_middle_rate": -1,
    "asof_pitcher_prev1_game_success_rate": 1,
    "asof_pitcher_prev3_game_success_rate": 1,
    "asof_pitcher_prev5_game_success_rate": 1,
    "temporal_pitcher_prior_rate": 1,
    "temporal_pitcher_prior_rate_shrunk_200": 1,
    "temporal_pitcher_season_rate": 1,
    "temporal_pitcher_season_global_10": 1,
    "temporal_pitcher_season_global_30": 1,
    "temporal_pitcher_season_global_100": 1,
    "temporal_batter_season_global_30": 1,
}


@dataclass(frozen=True)
class Config:
    name: str
    features: frozenset[str]
    monotonic: bool
    iterations: int
    num_leaves: int = 7
    min_child_samples: int = 5000


CONFIGS = [
    Config("strict_mono_i100", frozenset(STRICT_FEATURES), True, 100),
    Config("strict_mono_i200", frozenset(STRICT_FEATURES), True, 200),
    Config("strict_mono_i400", frozenset(STRICT_FEATURES), True, 400),
    Config("strict_free_i200", frozenset(STRICT_FEATURES), False, 200),
    Config("pressure_mono_i200", frozenset(PRESSURE_FEATURES), True, 200),
    Config("temporal_mono_i200", frozenset(TEMPORAL_FEATURES), True, 200),
]


def season_equal_weights(seasons: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(seasons), dtype=np.float32)
    unique, counts = np.unique(seasons, return_counts=True)
    for season, count in zip(unique, counts, strict=True):
        weights[seasons == season] = 1.0 / float(count)
    weights *= len(weights) / weights.sum()
    return weights


def fit_predict_residual(
    X: np.ndarray,
    target: np.ndarray,
    seasons: np.ndarray,
    validation_season: int,
    config: Config,
    monotone_constraints: list[int],
) -> tuple[np.ndarray, float, float, dict[str, float]]:
    train_mask = seasons < validation_season
    validation_mask = seasons == validation_season
    model = LGBMRegressor(
        objective="regression_l2",
        metric="l2",
        n_estimators=config.iterations,
        learning_rate=0.015,
        num_leaves=config.num_leaves,
        max_depth=-1,
        min_child_samples=config.min_child_samples,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=12.0,
        monotone_constraints=monotone_constraints,
        monotone_constraints_method="advanced",
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    weights = season_equal_weights(seasons[train_mask])
    started = time.time()
    model.fit(X[train_mask], target[train_mask], sample_weight=weights)
    fit_seconds = time.time() - started
    started = time.time()
    prediction = model.predict(X[validation_mask]).astype(np.float64)
    inference_seconds = time.time() - started
    importance = {
        str(index): float(value)
        for index, value in enumerate(model.feature_importances_)
    }
    return prediction, fit_seconds, inference_seconds, importance


def aggregate_metrics(
    folds: dict[str, object], prediction_name: str
) -> dict[str, float]:
    skills = [
        folds[str(season)][prediction_name]["skill_score_unclipped"]
        for season in REPORT_SEASONS
    ]
    return {
        "mean_skill": float(np.mean(skills)),
        "min_skill": float(np.min(skills)),
        "latest_2024_skill": float(skills[-1]),
    }


def run_config(
    config: Config,
    X_all: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
    feature_names: list[str],
    base_group_by_season: dict[int, np.ndarray],
) -> dict[str, object]:
    indices = [
        index for index, name in enumerate(feature_names) if name in config.features
    ]
    selected_names = [feature_names[index] for index in indices]
    missing = sorted(config.features - set(selected_names))
    if missing:
        raise ValueError(f"{config.name}: missing features: {missing}")
    X = X_all[:, indices]
    constraints = [
        MONOTONE_DIRECTIONS.get(name, 0) if config.monotonic else 0
        for name in selected_names
    ]
    residual_target = centered_residual(y, base, seasons)
    artifact_dir = ARTIFACT_ROOT / config.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}

    for validation_season in VALIDATION_SEASONS:
        validation_mask = seasons == validation_season
        targets = y[validation_mask]
        residual_prediction, fit_seconds, inference_seconds, importance = (
            fit_predict_residual(
                X,
                residual_target,
                seasons,
                validation_season,
                config,
                constraints,
            )
        )
        base_prediction = base[validation_mask].astype(np.float64)
        base_group_prediction = base_group_by_season[validation_season]
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "training_seasons": sorted(
                np.unique(seasons[seasons < validation_season]).astype(int).tolist()
            ),
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "residual_prediction_mean": float(residual_prediction.mean()),
            "base": calculate_metrics(targets, base_prediction),
            "base_plus_group": calculate_metrics(targets, base_group_prediction),
            "base_plus_stable_full": calculate_metrics(
                targets,
                np.clip(base_prediction + residual_prediction, 0.0, 1.0),
            ),
            "feature_importance": {
                selected_names[int(index)]: value
                for index, value in importance.items()
            },
        }
        for weight in BLEND_WEIGHTS:
            name = f"group_plus_stable_w{int(weight * 100):03d}"
            predictions = np.clip(
                base_group_prediction + weight * residual_prediction,
                0.0,
                1.0,
            )
            fold[name] = calculate_metrics(targets, predictions)
            np.save(
                artifact_dir / f"predictions_{name}_{validation_season}.npy",
                predictions,
            )
        np.save(artifact_dir / f"targets_{validation_season}.npy", targets)
        folds[str(validation_season)] = fold
        print(
            f"{config.name} {validation_season}: "
            f"base={fold['base']['skill_score_unclipped']:.2f} "
            f"group={fold['base_plus_group']['skill_score_unclipped']:.2f} "
            + " ".join(
                f"w{int(weight * 100):03d}="
                f"{fold[f'group_plus_stable_w{int(weight * 100):03d}']['skill_score_unclipped']:.2f}"
                for weight in BLEND_WEIGHTS
            )
        )

    prediction_names = [
        "base",
        "base_plus_group",
        "base_plus_stable_full",
        *[f"group_plus_stable_w{int(weight * 100):03d}" for weight in BLEND_WEIGHTS],
    ]
    result: dict[str, object] = {
        "experiment": "EXP-019",
        "candidate": config.name,
        "validation_protocol": {
            "outer_folds": VALIDATION_SEASONS,
            "reported_folds": REPORT_SEASONS,
            "current_fold_labels_used_for_training": False,
            "candidate_comparison_status": (
                "diagnostic only; requires nested temporal selection"
            ),
            "season_sample_weighting": "each training season has equal total weight",
        },
        "model": {
            "objective": "season-centered residual from temporal_base_global_30",
            "features": selected_names,
            "excluded_unstable_features": [
                "season",
                "pitcher_team_id",
                "batter_team_id",
                "asof_pitcher_ball_rate",
                "asof_pitcher_strike_rate",
                "asof_batter_success_rate",
                "pitchmix rates",
            ],
            "monotonic": config.monotonic,
            "monotone_constraints": dict(zip(selected_names, constraints, strict=True)),
            "iterations": config.iterations,
            "learning_rate": 0.015,
            "num_leaves": config.num_leaves,
            "min_child_samples": config.min_child_samples,
            "reg_alpha": 1.0,
            "reg_lambda": 12.0,
        },
        "folds": folds,
        "aggregate_2022_2024": {
            name: aggregate_metrics(folds, name) for name in prediction_names
        },
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
    return result


def main() -> None:
    started = time.time()
    diagnostics, X, y, base, seasons, feature_names = prepare_data()
    group_keys = build_group_keys(X, feature_names)
    residual = centered_residual(y, base, seasons)
    base_group_by_season: dict[int, np.ndarray] = {}
    for validation_season in VALIDATION_SEASONS:
        validation_mask = seasons == validation_season
        correction, _ = group_correction(
            group_keys, residual, seasons, validation_season
        )
        base_group_by_season[validation_season] = np.clip(
            base[validation_mask].astype(np.float64) + correction,
            0.0,
            1.0,
        )

    summaries: dict[str, object] = {}
    for config in CONFIGS:
        result = run_config(
            config,
            X,
            y,
            base,
            seasons,
            feature_names,
            base_group_by_season,
        )
        summaries[config.name] = result["aggregate_2022_2024"]

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "EXP-019",
        "stage": "stable_monotonic_candidate_search",
        "selection_status": "not selected; nested temporal selection required",
        "summaries": summaries,
        "total_seconds": time.time() - started,
    }
    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
