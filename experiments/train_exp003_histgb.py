"""EXP-003: HistGradientBoosting + EXP-002 상황 피처.

평가 서버 기본 scikit-learn만 사용하며, 기본 실행은 2024년 검증만 수행한다.
"""

from __future__ import annotations

import argparse
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
ARTIFACT_ROOT = Path("./artifacts/EXP-003")
ID = "row_id"
TARGET = "control_success"
EXP001_BRIER = 0.248767
EXP001_SCORE = 416.18
EXP002_BRIER = 0.2486367339543425
EXP002_SCORE = 468.43889304349904
CAT_COLS = ["top_bottom", "game_type", "base_state"]


def load_training_data() -> tuple[pd.DataFrame, list[str]]:
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
    return train, base_features + NEW_FEATURES


def build_model(features: list[str]) -> Pipeline:
    numeric_columns = [
        column for column in features if column not in CAT_COLS
    ]
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
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_columns,
        ),
    ])
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )
    return Pipeline([
        ("pre", preprocessor),
        ("clf", classifier),
    ])


def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    actual_rate = float(y_true.mean())
    brier = float(np.mean((predictions - y_true.to_numpy()) ** 2))
    baseline_brier = actual_rate * (1 - actual_rate)
    score = max(0.0, 100000 * (1 - brier / baseline_brier))
    return {
        "actual_rate": actual_rate,
        "prediction_mean": float(predictions.mean()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "brier_score": brier,
        "baseline_brier": baseline_brier,
        "skill_score": score,
        "brier_delta_vs_exp001": brier - EXP001_BRIER,
        "score_delta_vs_exp001": score - EXP001_SCORE,
        "brier_delta_vs_exp002": brier - EXP002_BRIER,
        "score_delta_vs_exp002": score - EXP002_SCORE,
    }


def run(validation_season: int = 2024) -> dict[str, float]:
    print("Environment")
    print(f" python={platform.python_version()}")
    print(f" pandas={pd.__version__}")
    print(f" numpy={np.__version__}")
    print(f" scikit-learn={sklearn.__version__}")
    print(f" joblib={joblib.__version__}")

    print("\nLoad and build features...")
    train, features = load_training_data()
    train_mask = train["season"] < validation_season
    validation_mask = train["season"] == validation_season
    if not validation_mask.any():
        raise ValueError(f"검증 시즌 {validation_season} 데이터가 없습니다.")

    X_train = train.loc[train_mask, features]
    y_train = train.loc[train_mask, TARGET]
    X_validation = train.loc[validation_mask, features]
    y_validation = train.loc[validation_mask, TARGET]
    print(
        f" features={len(features)} | train_rows={len(X_train)} | "
        f"validation_year={validation_season} | "
        f"validation_rows={len(X_validation)}"
    )

    model = build_model(features)
    started_at = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - started_at

    started_at = time.time()
    predictions = model.predict_proba(X_validation)[:, 1]
    inference_seconds = time.time() - started_at
    metrics = calculate_metrics(y_validation, predictions)
    metrics["validation_season"] = validation_season
    metrics["fit_seconds"] = fit_seconds
    metrics["inference_seconds"] = inference_seconds

    print("\nEXP-003 validation")
    print(
        f" Brier={metrics['brier_score']:.9f} | "
        f"baseline={metrics['baseline_brier']:.9f}"
    )
    print(f" Validation Score={metrics['skill_score']:.2f}")
    if validation_season == 2024:
        print(
            f" vs EXP-002: brier={metrics['brier_delta_vs_exp002']:+.9f}, "
            f"score={metrics['score_delta_vs_exp002']:+.2f}"
        )
    print(
        f" actual_rate={metrics['actual_rate']:.6f} | "
        f"prediction_mean={metrics['prediction_mean']:.6f}"
    )
    print(f" fit_seconds={fit_seconds:.1f}")
    print(f" inference_seconds={inference_seconds:.1f}")

    artifact_dir = ARTIFACT_ROOT / str(validation_season)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    np.save(artifact_dir / "validation_predictions.npy", predictions)
    np.save(
        artifact_dir / "validation_targets.npy",
        y_validation.to_numpy(dtype=np.int8),
    )
    print(f" metrics_saved={artifact_dir / 'validation_metrics.json'}")
    print(
        " predictions_saved="
        f"{artifact_dir / 'validation_predictions.npy'}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-season", type=int, default=2024)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(validation_season=arguments.validation_season)
