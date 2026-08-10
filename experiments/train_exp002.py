"""EXP-002: RandomForest + 상황 조합 피처 6개.

기본 실행은 2019~2023년 학습, 2024년 검증만 수행한다.
검증 결과가 EXP-001보다 좋을 때 --save-final 옵션으로 전체 데이터 모델을 저장한다.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


DATA_DIR = Path("./data")
ARTIFACT_DIR = Path("./artifacts/EXP-002")
ID = "row_id"
TARGET = "control_success"
BASELINE_BRIER = 0.248767
BASELINE_SCORE = 416.18

CAT_COLS = ["top_bottom", "game_type", "base_state"]
NEW_FEATURES = [
    "count_code",
    "is_full_count",
    "runner_in_scoring_position",
    "same_hand",
    "pitcher_batter_success_gap",
    "pitcher_recent_success_delta",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """현재 투구 직전에 알 수 있는 정보로 상황 조합 피처를 만든다."""
    out = df.copy()

    out["count_code"] = (
        out["balls_before"] * 4 + out["strikes_before"]
    )
    out["is_full_count"] = (
        (out["balls_before"] == 3) & (out["strikes_before"] == 2)
    ).astype("int8")
    out["runner_in_scoring_position"] = (
        (out["runner_on_2b"] == 1) | (out["runner_on_3b"] == 1)
    ).astype("int8")
    out["same_hand"] = (
        out["pitcher_hand"] == out["batter_hand"]
    ).astype("int8")
    out["pitcher_batter_success_gap"] = (
        out["asof_pitcher_success_rate"]
        - out["asof_batter_success_rate"]
    )
    out["pitcher_recent_success_delta"] = (
        out["asof_pitcher_prev1_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    )

    return out


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
    features = base_features + NEW_FEATURES
    return train, features


def build_model(features: list[str]) -> Pipeline:
    numeric_columns = [
        column for column in features if column not in CAT_COLS
    ]
    preprocessor = ColumnTransformer([
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            CAT_COLS,
        ),
        ("num", SimpleImputer(strategy="median"), numeric_columns),
    ])
    return Pipeline([
        ("pre", preprocessor),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=200,
                n_jobs=-1,
                random_state=42,
            ),
        ),
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
        "brier_delta_vs_exp001": brier - BASELINE_BRIER,
        "score_delta_vs_exp001": score - BASELINE_SCORE,
    }


def print_environment() -> None:
    print("Environment")
    print(f" python={platform.python_version()}")
    print(f" pandas={pd.__version__}")
    print(f" numpy={np.__version__}")
    print(f" scikit-learn={sklearn.__version__}")
    print(f" joblib={joblib.__version__}")


def run(save_final: bool = False) -> dict[str, float]:
    print_environment()
    print("\nLoad and build features...")
    train, features = load_training_data()
    print(
        f" train={train.shape} | features={len(features)} "
        f"(base={len(features) - len(NEW_FEATURES)}, new={len(NEW_FEATURES)})"
    )
    print(f" new_features={NEW_FEATURES}")

    is_validation = train["season"] == 2024
    X_train = train.loc[~is_validation, features]
    y_train = train.loc[~is_validation, TARGET]
    X_validation = train.loc[is_validation, features]
    y_validation = train.loc[is_validation, TARGET]
    print(f" train_rows={len(X_train)} | validation_rows={len(X_validation)}")

    model = build_model(features)
    started_at = time.time()
    model.fit(X_train, y_train)
    validation_fit_seconds = time.time() - started_at
    print(f" validation_fit_seconds={validation_fit_seconds:.1f}")

    started_at = time.time()
    validation_predictions = model.predict_proba(X_validation)[:, 1]
    validation_inference_seconds = time.time() - started_at
    metrics = calculate_metrics(y_validation, validation_predictions)
    metrics["validation_fit_seconds"] = validation_fit_seconds
    metrics["validation_inference_seconds"] = validation_inference_seconds

    print("\nEXP-002 validation")
    print(
        f" Brier={metrics['brier_score']:.6f} | "
        f"baseline={metrics['baseline_brier']:.6f}"
    )
    print(f" Validation Score={metrics['skill_score']:.2f}")
    print(
        f" vs EXP-001: brier={metrics['brier_delta_vs_exp001']:+.6f}, "
        f"score={metrics['score_delta_vs_exp001']:+.2f}"
    )
    print(
        f" actual_rate={metrics['actual_rate']:.6f} | "
        f"prediction_mean={metrics['prediction_mean']:.6f}"
    )
    print(f" validation_inference_seconds={validation_inference_seconds:.1f}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(f" metrics_saved={ARTIFACT_DIR / 'validation_metrics.json'}")

    if save_final:
        print("\nTrain final model on 2019-2024...")
        final_model = build_model(features)
        started_at = time.time()
        final_model.fit(train[features], train[TARGET])
        final_fit_seconds = time.time() - started_at
        model_path = ARTIFACT_DIR / "rf_exp002.pkl"
        joblib.dump(final_model, model_path, compress=3)
        print(f" final_fit_seconds={final_fit_seconds:.1f}")
        print(f" model_saved={model_path}")
        print(f" model_size_bytes={os.path.getsize(model_path)}")
    else:
        print("\nFinal model was not saved.")
        print("If EXP-002 is selected, rerun with: --save-final")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="검증 후 2019~2024년 전체 모델도 저장한다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(save_final=arguments.save_final)
