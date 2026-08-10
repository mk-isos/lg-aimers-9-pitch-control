"""EXP-017: nested rolling-origin residual LightGBM 비교.

검증 시즌의 정답은 최종 평가에만 쓴다. 트리 반복 수는 직전 시즌을 별도
튜닝 홀드아웃으로 사용해 정한 뒤, 검증 시즌 이전 데이터로 고정 반복 수만큼
다시 학습한다. 확률 보정도 현재 검증 시즌보다 앞선 OOF 예측만 사용한다.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import lsq_linear

from temporal_residual_features import (
    TARGET,
    add_static_features,
    attach_training_temporal_features,
)


DATA_DIR = Path("./data")
ARTIFACT_ROOT = Path("./artifacts/EXP-017")
ID = "row_id"
ONE_HOT_COLUMNS = ["top_bottom", "game_type", "base_state"]
DROP_MODEL_COLUMNS = [ID, TARGET, "pitcher_id", "batter_id"]
BASE_COLUMN = "temporal_base_global_30"
CANDIDATES = {
    "residual_all": {"objective": "residual", "window": None},
    "residual_recent3": {"objective": "residual", "window": 3},
    "residual_centered_all": {
        "objective": "residual_centered",
        "window": None,
    },
    "residual_centered_recent3": {
        "objective": "residual_centered",
        "window": 3,
    },
    "residual_centered_recent1": {
        "objective": "residual_centered",
        "window": 1,
    },
    "residual_centered_recent2": {
        "objective": "residual_centered",
        "window": 2,
    },
    "residual_context_all": {
        "objective": "residual_centered",
        "window": None,
        "feature_set": "context",
    },
    "binary_all": {"objective": "binary", "window": None},
}

CONTEXT_FEATURES = {
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_hand",
    "batter_hand",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "count_index",
    "count_out_index",
    "is_full_count",
    "has_two_strikes",
    "has_three_balls",
    "count_advantage",
    "runner_in_scoring_position",
    "bases_loaded",
    "same_hand",
    "late_inning",
    "close_game",
    "log_li",
    "score_pressure",
    "win_expectancy_gap",
    "pitcher_recent_success_delta_1_5",
    "pitcher_recent_success_delta_3_5",
    "pitcher_recent_middle_delta_1_5",
    "log_pitcher_n",
    "log_batter_n",
    "log_pitchmix_n",
    "temporal_pitcher_prior_exists",
    "temporal_pitcher_log_prior_n",
    "temporal_pitcher_log_season_n",
    "temporal_batter_prior_exists",
    "temporal_batter_log_prior_n",
    "temporal_batter_log_season_n",
}


def brier_metric(y_true: np.ndarray, predictions: np.ndarray):
    return "brier", float(np.mean((predictions - y_true) ** 2)), False


def calculate_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    actual_rate = float(y_true.mean())
    brier = float(np.mean((predictions - y_true) ** 2))
    baseline_brier = actual_rate * (1.0 - actual_rate)
    skill_unclipped = 100000.0 * (1.0 - brier / baseline_brier)
    design = np.column_stack([predictions, np.ones_like(predictions)])
    slope, intercept = np.linalg.lstsq(design, y_true, rcond=None)[0]
    return {
        "rows": int(len(y_true)),
        "actual_rate": actual_rate,
        "prediction_mean": float(predictions.mean()),
        "mean_gap": float(predictions.mean() - actual_rate),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "brier_score": brier,
        "baseline_brier": baseline_brier,
        "skill_score": float(max(0.0, skill_unclipped)),
        "skill_score_unclipped": float(skill_unclipped),
        "diagnostic_calibration_slope": float(slope),
        "diagnostic_calibration_intercept": float(intercept),
    }


def fit_prior_affine(
    targets: list[np.ndarray],
    predictions: list[np.ndarray],
) -> tuple[float, float]:
    if not targets:
        return 1.0, 0.0
    y = np.concatenate(targets).astype(float, copy=False)
    p = np.concatenate(predictions).astype(float, copy=False)
    design = np.column_stack([p, np.ones_like(p)])
    solution = lsq_linear(
        design,
        y,
        bounds=([0.75, -0.05], [1.25, 0.05]),
    )
    return float(solution.x[0]), float(solution.x[1])


def segment_metrics(
    diagnostics: pd.DataFrame,
    mask: np.ndarray,
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    pitcher_season_n = diagnostics.loc[mask, "temporal_pitcher_season_n"].to_numpy()
    count_segments = {
        "n_0": pitcher_season_n == 0,
        "n_1_19": (pitcher_season_n >= 1) & (pitcher_season_n < 20),
        "n_20_99": (pitcher_season_n >= 20) & (pitcher_season_n < 100),
        "n_100_499": (pitcher_season_n >= 100) & (pitcher_season_n < 500),
        "n_500_plus": pitcher_season_n >= 500,
    }
    pitcher_known = diagnostics.loc[
        mask, "temporal_pitcher_prior_exists"
    ].to_numpy(dtype=bool)
    batter_known = diagnostics.loc[
        mask, "temporal_batter_prior_exists"
    ].to_numpy(dtype=bool)
    status_segments = {
        "pitcher_new": ~pitcher_known,
        "pitcher_existing": pitcher_known,
        "batter_new": ~batter_known,
        "batter_existing": batter_known,
        "both_existing": pitcher_known & batter_known,
        "either_new": ~(pitcher_known & batter_known),
    }
    for name, segment_mask in {**count_segments, **status_segments}.items():
        if int(segment_mask.sum()) == 0:
            continue
        result[name] = calculate_metrics(
            y_true[segment_mask], predictions[segment_mask]
        )
    return result


def make_model(
    objective: str,
    n_estimators: int,
    num_leaves: int,
    min_child_samples: int,
):
    common = dict(
        n_estimators=n_estimators,
        learning_rate=0.015,
        num_leaves=num_leaves,
        max_depth=-1,
        min_child_samples=min_child_samples,
        max_bin=255,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=4.0,
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    if objective in {"residual", "residual_centered"}:
        return LGBMRegressor(objective="regression_l2", metric="l2", **common)
    return LGBMClassifier(objective="binary", metric="None", **common)


def make_residual_target(
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
    centered: bool,
) -> np.ndarray:
    residual = (y - base).astype(np.float32, copy=True)
    if centered:
        for season in np.unique(seasons):
            mask = seasons == season
            residual[mask] -= residual[mask].mean()
    return residual


def select_iteration(
    X: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
    tune_season: int,
    objective: str,
    window: int | None,
    num_leaves: int,
    min_child_samples: int,
) -> tuple[int, float]:
    train_mask = seasons < tune_season
    if window is not None:
        train_mask &= seasons >= tune_season - window
    tune_mask = seasons == tune_season
    if not train_mask.any() or not tune_mask.any():
        raise ValueError(f"튜닝 분할을 만들 수 없습니다: {tune_season}")
    model = make_model(
        objective,
        n_estimators=1200,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
    )
    started_at = time.time()
    if objective in {"residual", "residual_centered"}:
        residual_target = make_residual_target(
            y,
            base,
            seasons,
            centered=objective == "residual_centered",
        )
        model.fit(
            X[train_mask],
            residual_target[train_mask],
            eval_set=[(X[tune_mask], residual_target[tune_mask])],
            callbacks=[
                lgb.early_stopping(80, first_metric_only=True, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
    else:
        model.fit(
            X[train_mask],
            y[train_mask],
            eval_set=[(X[tune_mask], y[tune_mask])],
            eval_metric=brier_metric,
            callbacks=[
                lgb.early_stopping(80, first_metric_only=True, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
    best_iteration = int(model.best_iteration_)
    return best_iteration, time.time() - started_at


def fit_and_predict(
    X: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
    validation_season: int,
    objective: str,
    window: int | None,
    best_iteration: int,
    num_leaves: int,
    min_child_samples: int,
) -> tuple[np.ndarray, float, float]:
    train_mask = seasons < validation_season
    if window is not None:
        train_mask &= seasons >= validation_season - window
    validation_mask = seasons == validation_season
    model = make_model(
        objective,
        n_estimators=best_iteration,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
    )
    started_at = time.time()
    if objective in {"residual", "residual_centered"}:
        residual_target = make_residual_target(
            y,
            base,
            seasons,
            centered=objective == "residual_centered",
        )
        model.fit(X[train_mask], residual_target[train_mask])
    else:
        model.fit(X[train_mask], y[train_mask])
    fit_seconds = time.time() - started_at
    started_at = time.time()
    if objective in {"residual", "residual_centered"}:
        predictions = base[validation_mask] + model.predict(X[validation_mask])
    else:
        predictions = model.predict_proba(X[validation_mask])[:, 1]
    predictions = np.clip(predictions, 0.0, 1.0)
    return predictions, fit_seconds, time.time() - started_at


def prepare_data() -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    test_columns = pd.read_csv(
        DATA_DIR / "test.csv", encoding="utf-8-sig", nrows=0
    ).columns
    base_features = [column for column in test_columns if column != ID]
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=base_features + [TARGET],
    )
    train = add_static_features(train)
    train, _ = attach_training_temporal_features(train, target=TARGET)
    model_features = [
        column for column in train.columns if column not in DROP_MODEL_COLUMNS
    ]
    encoded = pd.get_dummies(
        train[model_features],
        columns=ONE_HOT_COLUMNS,
        dummy_na=True,
        dtype=np.int8,
    )
    for column in encoded.select_dtypes(include=["float64"]).columns:
        encoded[column] = encoded[column].astype("float32")
    for column in encoded.select_dtypes(include=["int64"]).columns:
        encoded[column] = encoded[column].astype("int32")
    X = encoded.to_numpy(dtype=np.float32)
    y = train[TARGET].to_numpy(dtype=np.float32)
    seasons = train["season"].to_numpy(dtype=np.int16)
    base = train[BASE_COLUMN].to_numpy(dtype=np.float32)
    diagnostics = train[
        [
            "season",
            "game_month",
            "temporal_pitcher_season_n",
            "temporal_pitcher_prior_exists",
            "temporal_batter_prior_exists",
        ]
    ].copy()
    return diagnostics, X, y, base, seasons, list(encoded.columns)


def run_candidate(
    candidate: str,
    diagnostics: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
    feature_names: list[str],
    validation_seasons: list[int],
    num_leaves: int,
    min_child_samples: int,
    fixed_iterations: int | None,
) -> dict[str, object]:
    config = CANDIDATES[candidate]
    objective = str(config["objective"])
    window = config["window"]
    feature_set = str(config.get("feature_set", "all"))
    if feature_set == "context":
        selected_indices = [
            index
            for index, name in enumerate(feature_names)
            if name in CONTEXT_FEATURES
            or name.startswith("top_bottom_")
            or name.startswith("game_type_")
            or name.startswith("base_state_")
        ]
        candidate_X = X[:, selected_indices]
        candidate_feature_names = [feature_names[index] for index in selected_indices]
    else:
        candidate_X = X
        candidate_feature_names = feature_names
    raw_targets: list[np.ndarray] = []
    raw_predictions: list[np.ndarray] = []
    folds: dict[str, object] = {}
    iteration_name = (
        f"fixed{fixed_iterations}" if fixed_iterations is not None else "nested"
    )
    artifact_dir = ARTIFACT_ROOT / (
        f"{candidate}_l{num_leaves}_m{min_child_samples}_{iteration_name}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for validation_season in validation_seasons:
        if fixed_iterations is None:
            best_iteration, tuning_seconds = select_iteration(
                candidate_X,
                y,
                base,
                seasons,
                tune_season=validation_season - 1,
                objective=objective,
                window=window,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
            )
            iteration_source = "previous_season_early_stopping"
        else:
            best_iteration = fixed_iterations
            tuning_seconds = 0.0
            iteration_source = "fixed_before_reported_fold_evaluation"
        predictions, fit_seconds, inference_seconds = fit_and_predict(
            candidate_X,
            y,
            base,
            seasons,
            validation_season=validation_season,
            objective=objective,
            window=window,
            best_iteration=best_iteration,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
        )
        validation_mask = seasons == validation_season
        targets = y[validation_mask]
        calibration_scale, calibration_intercept = fit_prior_affine(
            raw_targets, raw_predictions
        )
        calibrated = np.clip(
            calibration_scale * predictions + calibration_intercept,
            0.0,
            1.0,
        )
        fold = {
            "validation_season": validation_season,
            "training_seasons": sorted(
                np.unique(seasons[seasons < validation_season]).astype(int).tolist()
            ),
            "tuning_season": validation_season - 1,
            "best_iteration": best_iteration,
            "iteration_source": iteration_source,
            "tuning_seconds": tuning_seconds,
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "raw": calculate_metrics(targets, predictions),
            "prior_fold_calibration": {
                "scale": calibration_scale,
                "intercept": calibration_intercept,
                "trained_on_validation_seasons": validation_seasons[
                    : len(raw_targets)
                ],
                **calculate_metrics(targets, calibrated),
            },
            "segments_raw": segment_metrics(
                diagnostics, validation_mask, targets, predictions
            ),
        }
        folds[str(validation_season)] = fold
        np.save(artifact_dir / f"predictions_{validation_season}.npy", predictions)
        np.save(artifact_dir / f"targets_{validation_season}.npy", targets.astype(np.int8))
        raw_targets.append(targets.copy())
        raw_predictions.append(predictions.copy())
        print(
            f"{candidate} {validation_season}: iter={best_iteration} "
            f"raw={fold['raw']['skill_score_unclipped']:.2f} "
            f"cal={fold['prior_fold_calibration']['skill_score_unclipped']:.2f} "
            f"mean_gap={fold['raw']['mean_gap']:+.6f}"
        )
    raw_skills = [folds[str(s)]["raw"]["skill_score_unclipped"] for s in validation_seasons]
    calibrated_skills = [
        folds[str(s)]["prior_fold_calibration"]["skill_score_unclipped"]
        for s in validation_seasons
    ]
    final_scale, final_intercept = fit_prior_affine(raw_targets, raw_predictions)
    result: dict[str, object] = {
        "experiment": "EXP-017",
        "candidate": candidate,
        "objective": objective,
        "window": window,
        "validation_protocol": {
            "reported_seasons": validation_seasons,
            "iteration_selection": "previous season only; retrain before evaluation season",
            "calibration": "bounded affine fitted on earlier evaluated OOF seasons only",
            "calibration_bounds": {
                "scale": [0.75, 1.25],
                "intercept": [-0.05, 0.05],
            },
        },
        "model": {
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "learning_rate": 0.015,
            "fixed_iterations": fixed_iterations,
            "base_probability": BASE_COLUMN,
            "feature_set": feature_set,
            "features": len(candidate_feature_names),
            "dropped_raw_ids": ["pitcher_id", "batter_id"],
        },
        "folds": folds,
        "aggregate": {
            "raw_mean_skill": float(np.mean(raw_skills)),
            "raw_min_skill": float(np.min(raw_skills)),
            "prior_calibrated_mean_skill": float(np.mean(calibrated_skills)),
            "prior_calibrated_min_skill": float(np.min(calibrated_skills)),
        },
        "final_2025_calibration_from_all_oof": {
            "scale": final_scale,
            "intercept": final_intercept,
            "seasons": validation_seasons,
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
    with (artifact_dir / "feature_names.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(candidate_feature_names, file, ensure_ascii=False, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=sorted(CANDIDATES),
        default=["residual_all", "residual_recent3", "binary_all"],
    )
    parser.add_argument(
        "--validation-seasons",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023, 2024],
    )
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=1000)
    parser.add_argument("--fixed-iterations", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Prepare EXP-017 temporal features...")
    diagnostics, X, y, base, seasons, feature_names = prepare_data()
    print(f"rows={len(y)} features={len(feature_names)} matrix={X.nbytes / 2**20:.1f} MiB")
    for candidate in args.candidates:
        run_candidate(
            candidate,
            diagnostics,
            X,
            y,
            base,
            seasons,
            feature_names,
            validation_seasons=args.validation_seasons,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            fixed_iterations=args.fixed_iterations,
        )


if __name__ == "__main__":
    main()
