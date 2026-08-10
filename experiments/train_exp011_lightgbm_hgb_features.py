"""EXP-011: EXP-003과 같은 입력 표현을 사용하는 LightGBM."""

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
ARTIFACT_DIR = Path("./artifacts/EXP-011/2024")
ID = "row_id"
TARGET = "control_success"
EXP003_BRIER = 0.2480752636926058
EXP003_SCORE = 693.202379652235
ONE_HOT_COLUMNS = ["top_bottom", "game_type", "base_state"]


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
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=base_features + [TARGET],
    )
    train = add_features(train)
    print("One-hot encode three baseline categories...")
    encoded = pd.get_dummies(
        train[features],
        columns=ONE_HOT_COLUMNS,
        dummy_na=True,
        dtype=np.int8,
    )
    is_validation = train["season"] == 2024
    X_train = encoded.loc[~is_validation]
    y_train = train.loc[~is_validation, TARGET]
    X_validation = encoded.loc[is_validation]
    y_validation = train.loc[is_validation, TARGET]
    print(
        f" encoded_features={X_train.shape[1]} | train_rows={len(X_train)} | "
        f"validation_rows={len(X_validation)}"
    )

    model = LGBMClassifier(
        objective="binary",
        metric="None",
        n_estimators=1600,
        learning_rate=0.015,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=500,
        max_bin=255,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=3.0,
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
        eval_set=[(X_validation, y_validation)],
        eval_metric=brier_evaluation,
        callbacks=[
            lgb.early_stopping(100, first_metric_only=True, verbose=True),
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
    print("\nEXP-011 validation")
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
