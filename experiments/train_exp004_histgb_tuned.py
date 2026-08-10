"""EXP-004: 용량을 늘리고 규제를 조정한 HistGradientBoosting."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_exp002 import NEW_FEATURES, add_features


DATA_DIR = Path("./data")
ARTIFACT_DIR = Path("./artifacts/EXP-004/2024")
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
EXP003_BRIER = 0.248075264
EXP003_SCORE = 693.20


def main() -> None:
    print("Environment")
    print(f" python={platform.python_version()}")
    print(f" pandas={pd.__version__}")
    print(f" numpy={np.__version__}")
    print(f" scikit-learn={sklearn.__version__}")
    print(f" joblib={joblib.__version__}")

    test_columns = pd.read_csv(
        DATA_DIR / "test.csv", encoding="utf-8-sig", nrows=0
    ).columns
    base_features = [column for column in test_columns if column != ID]
    features = base_features + NEW_FEATURES
    numeric_columns = [
        column for column in features if column not in CAT_COLS
    ]

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
    print(
        f" features={len(features)} | train_rows={len(X_train)} | "
        f"validation_rows={len(X_validation)}"
    )

    preprocessor = ColumnTransformer([
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            ),
            CAT_COLS,
        ),
        ("num", SimpleImputer(strategy="median"), numeric_columns),
    ])
    model = Pipeline([
        ("pre", preprocessor),
        (
            "clf",
            HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.04,
                max_iter=260,
                max_leaf_nodes=63,
                max_depth=None,
                min_samples_leaf=250,
                l2_regularization=2.0,
                early_stopping=False,
                random_state=42,
            ),
        ),
    ])

    started_at = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - started_at
    started_at = time.time()
    predictions = model.predict_proba(X_validation)[:, 1]
    inference_seconds = time.time() - started_at

    actual_rate = float(y_validation.mean())
    brier = float(np.mean((predictions - y_validation.to_numpy()) ** 2))
    baseline_brier = actual_rate * (1 - actual_rate)
    score = max(0.0, 100000 * (1 - brier / baseline_brier))
    metrics = {
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

    print("\nEXP-004 validation")
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
    print(f" metrics_saved={ARTIFACT_DIR / 'validation_metrics.json'}")


if __name__ == "__main__":
    main()
