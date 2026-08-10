"""EXP-007: 범주형 피처를 직접 처리하는 LightGBM."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from train_exp002 import NEW_FEATURES, add_features


DATA_DIR = Path("./data")
ARTIFACT_DIR = Path("./artifacts/EXP-007/2024")
ID = "row_id"
TARGET = "control_success"
EXP003_BRIER = 0.2480752636926058
EXP003_SCORE = 693.202379652235

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
]


def apply_training_categories(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = X_train.copy()
    validation_out = X_validation.copy()
    for column in CATEGORICAL_COLUMNS:
        categories = pd.Index(train_out[column].dropna().unique())
        train_out[column] = pd.Categorical(
            train_out[column], categories=categories
        )
        validation_out[column] = pd.Categorical(
            validation_out[column], categories=categories
        )
    return train_out, validation_out


def brier_evaluation(y_true: np.ndarray, predictions: np.ndarray):
    return "brier", float(np.mean((predictions - y_true) ** 2)), False


def main() -> None:
    print("Environment")
    print(f" python={platform.python_version()}")
    print(f" pandas={pd.__version__}")
    print(f" numpy={np.__version__}")
    print(f" lightgbm={lgb.__version__}")

    test_columns = pd.read_csv(
        DATA_DIR / "test.csv", encoding="utf-8-sig", nrows=0
    ).columns
    base_features = [column for column in test_columns if column != ID]
    features = base_features + NEW_FEATURES

    print("\nLoad and build features...")
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=base_features + [TARGET],
    )
    train = add_features(train)
    is_validation = train["season"] == 2024
    X_train = train.loc[~is_validation, features]
    y_train = train.loc[~is_validation, TARGET]
    X_validation = train.loc[is_validation, features]
    y_validation = train.loc[is_validation, TARGET]
    X_train, X_validation = apply_training_categories(
        X_train, X_validation
    )
    print(
        f" features={len(features)} | categorical={len(CATEGORICAL_COLUMNS)} | "
        f"train_rows={len(X_train)} | validation_rows={len(X_validation)}"
    )

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=500,
        max_bin=255,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )

    started_at = time.time()
    model.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL_COLUMNS,
        eval_set=[(X_validation, y_validation)],
        eval_metric=brier_evaluation,
        callbacks=[
            lgb.early_stopping(80, first_metric_only=True, verbose=True),
            lgb.log_evaluation(100),
        ],
    )
    fit_seconds = time.time() - started_at
    started_at = time.time()
    predictions = model.predict_proba(X_validation)[:, 1]
    inference_seconds = time.time() - started_at

    actual_rate = float(y_validation.mean())
    brier = float(np.mean((predictions - y_validation.to_numpy()) ** 2))
    baseline_brier = actual_rate * (1 - actual_rate)
    score = max(0.0, 100000 * (1 - brier / baseline_brier))
    metrics = {
        "best_iteration": int(model.best_iteration_),
        "actual_rate": actual_rate,
        "prediction_mean": float(predictions.mean()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "brier_score": brier,
        "baseline_brier": baseline_brier,
        "skill_score": score,
        "brier_delta_vs_exp003": brier - EXP003_BRIER,
        "score_delta_vs_exp003": score - EXP003_SCORE,
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
    }

    print("\nEXP-007 validation")
    print(f" best_iteration={model.best_iteration_}")
    print(f" Brier={brier:.9f} | baseline={baseline_brier:.9f}")
    print(f" Validation Score={score:.2f}")
    print(
        f" vs EXP-003: brier={metrics['brier_delta_vs_exp003']:+.9f}, "
        f"score={metrics['score_delta_vs_exp003']:+.2f}"
    )
    print(
        f" actual_rate={actual_rate:.6f} | "
        f"prediction_mean={predictions.mean():.6f}"
    )
    print(f" fit_seconds={fit_seconds:.1f}")
    print(f" inference_seconds={inference_seconds:.1f}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    np.save(ARTIFACT_DIR / "validation_predictions.npy", predictions)
    np.save(
        ARTIFACT_DIR / "validation_targets.npy",
        y_validation.to_numpy(dtype=np.int8),
    )
    print(f" metrics_saved={ARTIFACT_DIR / 'validation_metrics.json'}")


if __name__ == "__main__":
    main()
