"""EXP-014: 시간 가중치와 범주형 ID를 적용한 LightGBM 실험.

투수/타자/팀 ID를 연속형 숫자가 아닌 범주형으로 처리하고, 오래된 시즌의
영향을 줄이는 시간 감쇠 가중치를 비교한다. 추가 피처는 모두 현재 투구 전에
제공되는 as-of 통계와 경기 상황만으로 생성한다.
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

from trackman_features import attach_trackman_features
from temporal_target_features import attach_temporal_target_features


DATA_DIR = Path("./data")
ARTIFACT_ROOT = Path("./artifacts/EXP-014")
ID = "row_id"
TARGET = "control_success"

BASE_CATEGORICAL_COLUMNS = [
    "top_bottom",
    "game_type",
    "base_state",
]
ID_CATEGORICAL_COLUMNS = BASE_CATEGORICAL_COLUMNS + [
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]
EXTENDED_CATEGORICAL_COLUMNS = ID_CATEGORICAL_COLUMNS + [
    "count_code",
    "count_out_state",
    "hand_matchup",
    "team_matchup",
]

LEGACY_FEATURES = [
    "count_index",
    "is_full_count",
    "runner_in_scoring_position",
    "same_hand",
    "pitcher_batter_success_gap",
    "pitcher_recent_success_delta_1_5",
]


def _shrunk_rate(
    rate: pd.Series,
    count: pd.Series,
    strength: float,
    prior: float = 0.5,
) -> pd.Series:
    safe_count = count.fillna(0).clip(lower=0)
    safe_rate = rate.fillna(prior)
    return (safe_count * safe_rate + strength * prior) / (
        safe_count + strength
    )


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """현재 투구 직전 정보만으로 모델 입력 피처를 만든다."""
    out = frame.copy()

    out["count_code"] = (
        out["balls_before"].astype(str)
        + "-"
        + out["strikes_before"].astype(str)
    )
    out["count_index"] = (
        out["balls_before"] * 4 + out["strikes_before"]
    ).astype("int8")
    out["count_out_state"] = (
        out["count_code"] + "-" + out["outs_before"].astype(str)
    )
    out["hand_matchup"] = (
        out["pitcher_hand"].astype(str)
        + "-"
        + out["batter_hand"].astype(str)
    )
    out["team_matchup"] = (
        out["pitcher_team_id"].astype(str)
        + "-"
        + out["batter_team_id"].astype(str)
    )

    out["is_full_count"] = (
        (out["balls_before"] == 3) & (out["strikes_before"] == 2)
    ).astype("int8")
    out["has_two_strikes"] = (out["strikes_before"] == 2).astype("int8")
    out["has_three_balls"] = (out["balls_before"] == 3).astype("int8")
    out["count_advantage"] = (
        out["strikes_before"] - out["balls_before"]
    ).astype("int8")
    out["runner_in_scoring_position"] = (
        (out["runner_on_2b"] == 1) | (out["runner_on_3b"] == 1)
    ).astype("int8")
    out["bases_loaded"] = (
        (out["runner_on_1b"] == 1)
        & (out["runner_on_2b"] == 1)
        & (out["runner_on_3b"] == 1)
    ).astype("int8")
    out["same_hand"] = (
        out["pitcher_hand"] == out["batter_hand"]
    ).astype("int8")
    out["late_inning"] = (out["inning"] >= 7).astype("int8")
    out["close_game"] = (
        out["score_diff_pitcher_team"].abs() <= 1
    ).astype("int8")
    out["log_li"] = np.log1p(out["li"].clip(lower=0))
    out["score_pressure"] = (
        out["score_diff_pitcher_team"].abs() * out["log_li"]
    )
    out["win_expectancy_gap"] = (
        out["home_win_expectancy"] - out["away_win_expectancy"]
    )

    out["pitcher_batter_success_gap"] = (
        out["asof_pitcher_success_rate"]
        - out["asof_batter_success_rate"]
    )
    out["pitcher_recent_success_delta_1_5"] = (
        out["asof_pitcher_prev1_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    )
    out["pitcher_recent_success_delta_3_5"] = (
        out["asof_pitcher_prev3_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    )
    out["pitcher_recent_middle_delta_1_5"] = (
        out["asof_pitcher_prev1_game_middle_rate"]
        - out["asof_pitcher_prev5_game_middle_rate"]
    )
    out["pitcher_recent_success_mean"] = out[
        [
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
        ]
    ].mean(axis=1)

    out["log_pitcher_n"] = np.log1p(out["asof_pitcher_n"].clip(lower=0))
    out["log_batter_n"] = np.log1p(out["asof_batter_n"].clip(lower=0))
    out["log_pitchmix_n"] = np.log1p(
        out["asof_pitcher_pitchmix_n"].clip(lower=0)
    )
    for strength in (50.0, 200.0, 500.0):
        suffix = int(strength)
        out[f"pitcher_success_shrunk_{suffix}"] = _shrunk_rate(
            out["asof_pitcher_success_rate"],
            out["asof_pitcher_n"],
            strength,
        )
        out[f"batter_success_shrunk_{suffix}"] = _shrunk_rate(
            out["asof_batter_success_rate"],
            out["asof_batter_n"],
            strength,
        )

    pitchmix = out[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].fillna(0.0).clip(lower=1e-12)
    out["pitchmix_entropy"] = -(pitchmix * np.log(pitchmix)).sum(axis=1)
    out["pitchmix_max_rate"] = pitchmix.max(axis=1)
    out["pitcher_failure_risk"] = out[
        [
            "asof_pitcher_reverse_rate",
            "asof_pitcher_middle_rate",
            "asof_pitcher_ball_rate",
        ]
    ].sum(axis=1, min_count=1)

    return out


def load_data(
    feature_set: str,
    use_trackman: bool,
    cutoff_season: int,
    max_mapping_cost: float,
    use_temporal_target_features: bool,
    target_smoothing: float,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    test_columns = pd.read_csv(
        DATA_DIR / "test.csv",
        encoding="utf-8-sig",
        nrows=0,
    ).columns
    base_features = [column for column in test_columns if column != ID]
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=base_features + [TARGET],
    )
    train = add_features(train)
    temporal_target_features: list[str] = []
    if use_temporal_target_features:
        train, temporal_target_features = attach_temporal_target_features(
            train,
            target=TARGET,
            max_season=cutoff_season + 1,
            smoothing=target_smoothing,
        )
    trackman_metadata: dict[str, object] = {"enabled": False}
    trackman_features: list[str] = []
    if use_trackman:
        trackman = pd.read_csv(
            DATA_DIR / "trackman_history.csv",
            encoding="utf-8-sig",
        )
        train, mapping_result, trackman_features = attach_trackman_features(
            train,
            trackman,
            cutoff_season=cutoff_season,
            max_mapping_cost=max_mapping_cost,
        )
        trackman_metadata = {
            "enabled": True,
            "cutoff_season": cutoff_season,
            "max_mapping_cost": max_mapping_cost,
            "mapped_pitchers": len(mapping_result.mapping),
            "candidate_main_ids": mapping_result.candidate_main_ids,
            "candidate_trackman_ids": mapping_result.candidate_trackman_ids,
            "feature_count": len(trackman_features),
        }
    if feature_set == "base":
        features = base_features + trackman_features + temporal_target_features
    elif feature_set == "legacy":
        features = (
            base_features
            + LEGACY_FEATURES
            + trackman_features
            + temporal_target_features
        )
    else:
        features = [column for column in train.columns if column != TARGET]
    return train, features, trackman_metadata


def prepare_categories(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_frame.copy()
    validation_out = validation_frame.copy()
    for column in categorical_columns:
        train_values = train_out[column].fillna("__MISSING__").astype(str)
        categories = pd.Index(train_values.unique())
        train_out[column] = pd.Categorical(
            train_values,
            categories=categories,
        )
        validation_out[column] = pd.Categorical(
            validation_out[column].fillna("__MISSING__").astype(str),
            categories=categories,
        )
    return train_out, validation_out


def prepare_one_hot(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = pd.get_dummies(
        train_frame,
        columns=BASE_CATEGORICAL_COLUMNS,
        dummy_na=True,
        dtype=np.int8,
    )
    validation_out = pd.get_dummies(
        validation_frame,
        columns=BASE_CATEGORICAL_COLUMNS,
        dummy_na=True,
        dtype=np.int8,
    )
    validation_out = validation_out.reindex(
        columns=train_out.columns,
        fill_value=0,
    )
    return train_out, validation_out


def brier_evaluation(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> tuple[str, float, bool]:
    return "brier", float(np.mean((predictions - y_true) ** 2)), False


def calculate_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    actual_rate = float(y_true.mean())
    brier = float(np.mean((predictions - y_true) ** 2))
    baseline_brier = actual_rate * (1.0 - actual_rate)
    score = max(0.0, 100000.0 * (1.0 - brier / baseline_brier))
    return {
        "actual_rate": actual_rate,
        "prediction_mean": float(predictions.mean()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "brier_score": brier,
        "baseline_brier": baseline_brier,
        "skill_score": score,
    }


def affine_calibration(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    design = np.column_stack([predictions, np.ones_like(predictions)])
    scale, intercept = np.linalg.lstsq(design, y_true, rcond=None)[0]
    calibrated = np.clip(scale * predictions + intercept, 0.0, 1.0)
    return calibrated, float(scale), float(intercept)


def run(
    validation_season: int,
    decay: float,
    window: int | None,
    num_leaves: int,
    feature_set: str,
    category_mode: str,
    use_trackman: bool,
    max_mapping_cost: float,
    objective: str,
    min_child_samples: int,
    seed: int,
    use_temporal_target_features: bool,
    target_smoothing: float,
) -> dict[str, object]:
    print("Environment")
    print(f" python={platform.python_version()}")
    print(f" pandas={pd.__version__}")
    print(f" numpy={np.__version__}")
    print(f" lightgbm={lgb.__version__}")

    started_at = time.time()
    train, features, trackman_metadata = load_data(
        feature_set,
        use_trackman=use_trackman,
        cutoff_season=validation_season - 1,
        max_mapping_cost=max_mapping_cost,
        use_temporal_target_features=use_temporal_target_features,
        target_smoothing=target_smoothing,
    )
    if category_mode == "onehot":
        categorical_columns = []
        string_only_columns = [
            "count_code",
            "count_out_state",
            "hand_matchup",
            "team_matchup",
        ]
        features = [
            column for column in features if column not in string_only_columns
        ]
    elif category_mode == "baseline":
        categorical_columns = BASE_CATEGORICAL_COLUMNS
    elif category_mode == "ids":
        categorical_columns = ID_CATEGORICAL_COLUMNS
    else:
        categorical_columns = EXTENDED_CATEGORICAL_COLUMNS
    categorical_columns = [
        column for column in categorical_columns if column in features
    ]
    train_mask = train["season"] < validation_season
    if window is not None:
        train_mask &= train["season"] >= validation_season - window
    validation_mask = train["season"] == validation_season
    train_frame = train.loc[train_mask, features]
    validation_frame = train.loc[validation_mask, features]
    y_train = train.loc[train_mask, TARGET].to_numpy()
    y_validation = train.loc[validation_mask, TARGET].to_numpy()
    if category_mode == "onehot":
        train_frame, validation_frame = prepare_one_hot(
            train_frame,
            validation_frame,
        )
    else:
        train_frame, validation_frame = prepare_categories(
            train_frame,
            validation_frame,
            categorical_columns,
        )
    latest_training_season = validation_season - 1
    ages = latest_training_season - train.loc[train_mask, "season"].to_numpy()
    sample_weight = np.power(decay, ages)
    print(
        f"features={len(features)} | categorical={len(categorical_columns)} | "
        f"train_rows={len(train_frame)} | validation_rows={len(validation_frame)}"
    )
    print(
        f"validation_season={validation_season} | decay={decay} | "
        f"window={window} | num_leaves={num_leaves} | "
        f"feature_set={feature_set} | category_mode={category_mode} | "
        f"trackman={use_trackman}"
    )

    model_class = LGBMClassifier if objective == "binary" else LGBMRegressor
    model = model_class(
        objective=objective,
        metric="None",
        n_estimators=1800,
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
        random_state=seed,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    fit_started_at = time.time()
    model.fit(
        train_frame,
        y_train,
        sample_weight=sample_weight,
        categorical_feature=categorical_columns,
        eval_set=[(validation_frame, y_validation)],
        eval_metric=brier_evaluation,
        callbacks=[
            lgb.early_stopping(100, first_metric_only=True, verbose=True),
            lgb.log_evaluation(100),
        ],
    )
    fit_seconds = time.time() - fit_started_at
    if objective == "binary":
        predictions = model.predict_proba(validation_frame)[:, 1]
    else:
        predictions = np.clip(model.predict(validation_frame), 0.0, 1.0)
    calibrated, scale, intercept = affine_calibration(
        y_validation,
        predictions,
    )
    raw_metrics = calculate_metrics(y_validation, predictions)
    calibrated_metrics = calculate_metrics(y_validation, calibrated)

    result: dict[str, object] = {
        "experiment": "EXP-014",
        "validation_season": validation_season,
        "decay": decay,
        "window": window,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "seed": seed,
        "feature_set": feature_set,
        "category_mode": category_mode,
        "objective": objective,
        "trackman": trackman_metadata,
        "temporal_target_features": {
            "enabled": use_temporal_target_features,
            "smoothing": target_smoothing,
        },
        "features": len(features),
        "categorical_features": categorical_columns,
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "best_iteration": int(model.best_iteration_),
        "fit_seconds": fit_seconds,
        "total_seconds": time.time() - started_at,
        "raw": raw_metrics,
        "calibration": {
            "scale": scale,
            "intercept": intercept,
            **calibrated_metrics,
        },
    }
    trackman_name = (
        f"tm{max_mapping_cost:g}" if use_trackman else "tmoff"
    )
    target_name = (
        f"te{target_smoothing:g}"
        if use_temporal_target_features
        else "teoff"
    )
    config_name = (
        f"v{validation_season}_d{decay:g}_"
        f"w{window if window is not None else 'all'}_l{num_leaves}_"
        f"f{feature_set}_c{category_mode}_{trackman_name}_o{objective}_"
        f"m{min_child_samples}_s{seed}_{target_name}"
    )
    artifact_dir = ARTIFACT_ROOT / config_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "validation_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    np.save(artifact_dir / "validation_predictions.npy", predictions)
    np.save(artifact_dir / "validation_targets.npy", y_validation.astype(np.int8))

    print("\nEXP-014 validation")
    print(
        f" raw_brier={raw_metrics['brier_score']:.9f} | "
        f"raw_score={raw_metrics['skill_score']:.2f} | "
        f"prediction_mean={raw_metrics['prediction_mean']:.6f}"
    )
    print(
        f" calibrated_brier={calibrated_metrics['brier_score']:.9f} | "
        f"calibrated_score={calibrated_metrics['skill_score']:.2f} | "
        f"scale={scale:.9f} | intercept={intercept:.9f}"
    )
    print(f"best_iteration={model.best_iteration_} | fit_seconds={fit_seconds:.1f}")
    print(f"artifacts={artifact_dir}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument(
        "--feature-set",
        choices=["base", "legacy", "engineered"],
        default="legacy",
    )
    parser.add_argument(
        "--category-mode",
        choices=["onehot", "baseline", "ids", "extended"],
        default="onehot",
    )
    parser.add_argument("--use-trackman", action="store_true")
    parser.add_argument("--max-mapping-cost", type=float, default=0.1)
    parser.add_argument(
        "--objective",
        choices=["binary", "regression_l2"],
        default="binary",
    )
    parser.add_argument("--min-child-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-temporal-target-features", action="store_true")
    parser.add_argument("--target-smoothing", type=float, default=100.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        validation_season=arguments.validation_season,
        decay=arguments.decay,
        window=arguments.window,
        num_leaves=arguments.num_leaves,
        feature_set=arguments.feature_set,
        category_mode=arguments.category_mode,
        use_trackman=arguments.use_trackman,
        max_mapping_cost=arguments.max_mapping_cost,
        objective=arguments.objective,
        min_child_samples=arguments.min_child_samples,
        seed=arguments.seed,
        use_temporal_target_features=arguments.use_temporal_target_features,
        target_smoothing=arguments.target_smoothing,
    )
