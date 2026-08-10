"""EXP-019: R/F 체제 분리 multirate residual.

game_type F의 성공률은 2022에서 2023으로 구조적으로 급변했다. R 행에서만
방향이 일관된 현재 시즌 success/reverse/ball/strike 및 pitchmix 신호를
학습하고 적용한다. F 행은 row-independent hierarchical base만 사용한다.
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
from train_exp017_rolling_residual import calculate_metrics, segment_metrics
from train_exp019_stable_monotonic import season_equal_weights


ARTIFACT_ROOT = Path("./artifacts/EXP-019/game_type_branch")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]

CORE_FEATURES = {
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
    "multirate_pitcher_control_ball_prior_shrunk_200",
    "multirate_pitcher_control_ball_season_global_30",
    "multirate_pitcher_control_strike_prior_shrunk_200",
    "multirate_pitcher_control_strike_season_global_30",
    "multirate_pitcher_pitchmix_fastball_season_global_30",
    "multirate_pitcher_pitchmix_breaking_season_global_30",
    "multirate_pitcher_pitchmix_offspeed_season_global_30",
}

MONOTONE = {
    "asof_pitcher_prev1_game_success_rate": 1,
    "asof_pitcher_prev3_game_success_rate": 1,
    "asof_pitcher_prev5_game_success_rate": 1,
    "multirate_pitcher_control_success_prior_shrunk_200": 1,
    "multirate_pitcher_control_success_season_global_30": 1,
    "multirate_pitcher_control_reverse_prior_shrunk_200": -1,
    "multirate_pitcher_control_reverse_season_global_30": -1,
    "multirate_pitcher_control_ball_prior_shrunk_200": -1,
    "multirate_pitcher_control_ball_season_global_30": -1,
    "multirate_pitcher_control_strike_prior_shrunk_200": 1,
    "multirate_pitcher_control_strike_season_global_30": 1,
    "multirate_pitcher_pitchmix_fastball_season_global_30": -1,
    "multirate_pitcher_pitchmix_breaking_season_global_30": -1,
    "multirate_pitcher_pitchmix_offspeed_season_global_30": 1,
}


@dataclass(frozen=True)
class Config:
    name: str
    monotonic: bool
    iterations: int
    num_leaves: int
    min_child_samples: int


CONFIGS = [
    Config("r_mono_l7_i200", True, 200, 7, 5000),
    Config("r_mono_l7_i400", True, 400, 7, 5000),
    Config("r_mono_l15_i200", True, 200, 15, 5000),
    Config("r_free_l7_i200", False, 200, 7, 5000),
]


def r_group_correction(
    frame: pd.DataFrame,
    residual: np.ndarray,
    seasons: np.ndarray,
    is_r: np.ndarray,
    validation_season: int,
) -> np.ndarray:
    train_mask = (
        (seasons < validation_season)
        & (seasons >= validation_season - 3)
        & is_r
    )
    validation_mask = seasons == validation_season
    validation_r = validation_mask & is_r
    correction = np.zeros(int(validation_mask.sum()), dtype=np.float64)
    if not train_mask.any() or not validation_r.any():
        return correction
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

    def mapped_effect(columns: list[str], smoothing: float) -> np.ndarray:
        grouped = keys.loc[train_mask, columns].copy()
        grouped["residual"] = residual[train_mask]
        stats = grouped.groupby(columns, sort=False)["residual"].agg(["sum", "count"])
        effects = stats["sum"] / (stats["count"] + smoothing)
        index = pd.MultiIndex.from_frame(keys.loc[validation_r, columns])
        return effects.reindex(index).fillna(0.0).to_numpy(dtype=float)

    coarse_columns = ["count_index", "pitcher_hand", "batter_hand"]
    coarse = mapped_effect(coarse_columns, 100.0)
    reverse = mapped_effect(coarse_columns + ["season_reverse_bin"], 300.0)
    validation_local_r = is_r[validation_mask]
    correction[validation_local_r] = 0.7 * coarse + 0.3 * reverse
    return correction


def regime_metrics(
    game_types: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        regime: calculate_metrics(
            targets[game_types == regime], predictions[game_types == regime]
        )
        for regime in sorted(np.unique(game_types))
    }


def main() -> None:
    started = time.time()
    frame, diagnostics, y, base, seasons, reconstruction = (
        multirate.prepare_multirate_data()
    )
    game_types = frame["game_type"].astype(str).to_numpy()
    is_r = game_types == "R"
    initial_residual = multirate.centered_residual(y, base, seasons)
    group_all = np.empty(len(y), dtype=np.float64)
    group_reported: dict[int, np.ndarray] = {}
    for season in sorted(np.unique(seasons).astype(int).tolist()):
        mask = seasons == season
        correction = r_group_correction(
            frame, initial_residual, seasons, is_r, season
        )
        prediction = np.clip(base[mask].astype(float) + correction, 0.0, 1.0)
        group_all[mask] = prediction
        if season in VALIDATION_SEASONS:
            group_reported[season] = prediction

    residual_target = (y.astype(float) - group_all).astype(np.float32)
    for season in np.unique(seasons):
        mask = (seasons == season) & is_r
        residual_target[mask] -= residual_target[mask].mean()

    feature_names = sorted(CORE_FEATURES)
    X = frame[feature_names].to_numpy(dtype=np.float32)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}

    for config in CONFIGS:
        constraints = [
            MONOTONE.get(name, 0) if config.monotonic else 0
            for name in feature_names
        ]
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
            fit_started = time.time()
            model.fit(
                X[train_mask],
                residual_target[train_mask],
                sample_weight=season_equal_weights(seasons[train_mask]),
            )
            correction = np.zeros(int(validation_mask.sum()), dtype=np.float64)
            local_r = is_r[validation_mask]
            correction[local_r] = model.predict(X[validation_r]).astype(float)
            targets = y[validation_mask]
            validation_types = game_types[validation_mask]
            fold: dict[str, object] = {
                "validation_season": validation_season,
                "r_train_rows": int(train_mask.sum()),
                "r_validation_rows": int(validation_r.sum()),
                "fit_seconds": time.time() - fit_started,
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
                fold[f"regimes_{candidate}"] = regime_metrics(
                    validation_types, targets, predictions
                )
                fold[f"segments_{candidate}"] = segment_metrics(
                    diagnostics, validation_mask, targets, predictions
                )
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
                "R_model_training": "past R rows only",
                "F_prediction": "temporal_base_global_30 only",
                "current_fold_labels_used_for_training": False,
                "test_row_aggregation": False,
                "candidate_comparison_status": "diagnostic; nested selection required",
            },
            "model": {
                "features": feature_names,
                "monotonic": config.monotonic,
                "monotone_constraints": dict(
                    zip(feature_names, constraints, strict=True)
                ),
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
                "stage": "game_type_R_F_branch",
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
