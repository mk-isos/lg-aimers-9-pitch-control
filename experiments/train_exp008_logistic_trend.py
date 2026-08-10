"""EXP-008: 시즌 추세를 선형 외삽하는 로지스틱 회귀."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_exp002 import NEW_FEATURES, add_features


DATA_DIR = Path("./data")
ARTIFACT_DIR = Path("./artifacts/EXP-008/2024")
ID = "row_id"
TARGET = "control_success"
EXP003_BRIER = 0.2480752636926058
EXP003_SCORE = 693.202379652235

DROP_COLUMNS = ["pitcher_id", "batter_id"]
CATEGORICAL_COLUMNS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "count_code",
    "is_full_count",
    "runner_in_scoring_position",
    "same_hand",
]


def main() -> None:
    print("Environment")
    print(f" python={platform.python_version()}")
    print(f" pandas={pd.__version__}")
    print(f" numpy={np.__version__}")
    print(f" scikit-learn={sklearn.__version__}")

    test_columns = pd.read_csv(
        DATA_DIR / "test.csv", encoding="utf-8-sig", nrows=0
    ).columns
    base_features = [column for column in test_columns if column != ID]
    all_features = base_features + NEW_FEATURES
    features = [column for column in all_features if column not in DROP_COLUMNS]
    numeric_columns = [
        column for column in features if column not in CATEGORICAL_COLUMNS
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
        f" features={len(features)} | categorical={len(CATEGORICAL_COLUMNS)} | "
        f"numeric={len(numeric_columns)} | train_rows={len(X_train)} | "
        f"validation_rows={len(X_validation)}"
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20,
                dtype=np.float32,
            ),
            CATEGORICAL_COLUMNS,
        ),
        ("num", numeric_pipeline, numeric_columns),
    ])
    model = Pipeline([
        ("pre", preprocessor),
        (
            "clf",
            LogisticRegression(
                C=0.2,
                solver="lbfgs",
                max_iter=150,
                tol=1e-6,
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

    print("\nEXP-008 validation")
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
