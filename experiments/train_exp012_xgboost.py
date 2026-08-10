"""EXP-012: EXP-003과 같은 피처 표현을 사용하는 XGBoost."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from train_exp002 import NEW_FEATURES, add_features


DATA_DIR = Path("./data")
ARTIFACT_DIR = Path("./artifacts/EXP-012/2024")
ID = "row_id"
TARGET = "control_success"
EXP011_BRIER = 0.24804322476344168
EXP011_SCORE = 706.0260563004462
ONE_HOT_COLUMNS = ["top_bottom", "game_type", "base_state"]


def main() -> None:
    print("Environment")
    print(f" python={platform.python_version()}")
    print(f" pandas={pd.__version__}")
    print(f" numpy={np.__version__}")
    print(f" xgboost={xgb.__version__}")

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

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1600,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=200,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=5.0,
        gamma=0.0,
        tree_method="hist",
        max_bin=256,
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )
    started_at = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=100,
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
        "best_iteration": int(model.best_iteration),
        "actual_rate": actual_rate,
        "prediction_mean": float(predictions.mean()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "brier_score": brier,
        "baseline_brier": baseline_brier,
        "skill_score": score,
        "brier_delta_vs_exp011": brier - EXP011_BRIER,
        "score_delta_vs_exp011": score - EXP011_SCORE,
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
    }
    print("\nEXP-012 validation")
    print(f" best_iteration={model.best_iteration}")
    print(f" Brier={brier:.9f} | baseline={baseline_brier:.9f}")
    print(f" Validation Score={score:.2f}")
    print(
        f" vs EXP-011: brier={metrics['brier_delta_vs_exp011']:+.9f}, "
        f"score={metrics['score_delta_vs_exp011']:+.2f}"
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
