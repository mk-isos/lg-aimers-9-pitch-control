"""EXP-013 최종 후보: CatBoost + LightGBM 전체 데이터 학습.

두 모델은 Python pickle이 아닌 각 라이브러리의 네이티브 형식으로 저장한다.
따라서 학습 Python 3.12와 평가 Python 3.11 사이의 pickle 호환 문제를 피한다.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from train_exp002 import NEW_FEATURES, add_features


DATA_DIR = Path("./data")
MODEL_DIR = Path("./submissions/EXP-013/model")
ID = "row_id"
TARGET = "control_success"
ONE_HOT_COLUMNS = ["top_bottom", "game_type", "base_state"]
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


def prepare_catboost_categories(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        out[column] = out[column].fillna("__MISSING__").astype(str)
    return out


def main() -> None:
    print("Environment")
    print(f" python={platform.python_version()}")
    print(f" pandas={pd.__version__}")
    print(f" numpy={np.__version__}")
    print(f" catboost={catboost.__version__}")
    print(f" lightgbm={lgb.__version__}")

    test_columns = pd.read_csv(
        DATA_DIR / "test.csv", encoding="utf-8-sig", nrows=0
    ).columns
    base_features = [column for column in test_columns if column != ID]
    features = base_features + NEW_FEATURES
    print("\nLoad full 2019-2024 training data...")
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=base_features + [TARGET],
    )
    train = add_features(train)
    print(f" train_rows={len(train)} | features={len(features)}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nTrain CatBoost final model...")
    catboost_train = prepare_catboost_categories(train[features])
    catboost_model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=400,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        border_count=128,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        rsm=0.8,
        has_time=True,
        random_seed=42,
        thread_count=-1,
        verbose=50,
        allow_writing_files=False,
    )
    started_at = time.time()
    catboost_model.fit(
        catboost_train,
        train[TARGET],
        cat_features=CATEGORICAL_COLUMNS,
    )
    catboost_seconds = time.time() - started_at
    catboost_path = MODEL_DIR / "catboost_model.cbm"
    catboost_model.save_model(catboost_path)
    print(f" catboost_seconds={catboost_seconds:.1f}")
    print(f" catboost_saved={catboost_path}")

    print("\nTrain LightGBM final model...")
    encoded_train = pd.get_dummies(
        train[features],
        columns=ONE_HOT_COLUMNS,
        dummy_na=True,
        dtype=np.int8,
    )
    lightgbm_model = LGBMClassifier(
        objective="binary",
        n_estimators=335,
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
    lightgbm_model.fit(encoded_train, train[TARGET])
    lightgbm_seconds = time.time() - started_at
    lightgbm_path = MODEL_DIR / "lightgbm_model.txt"
    lightgbm_model.booster_.save_model(lightgbm_path)
    columns_path = MODEL_DIR / "lightgbm_columns.json"
    with columns_path.open("w", encoding="utf-8") as file:
        json.dump(list(encoded_train.columns), file, ensure_ascii=False)
    print(f" lightgbm_seconds={lightgbm_seconds:.1f}")
    print(f" lightgbm_saved={lightgbm_path}")
    print(f" columns_saved={columns_path}")

    metadata = {
        "experiment": "EXP-013",
        "training_seasons": [2019, 2020, 2021, 2022, 2023, 2024],
        "training_rows": int(len(train)),
        "base_features": len(base_features),
        "new_features": NEW_FEATURES,
        "total_features": len(features),
        "catboost_version": catboost.__version__,
        "lightgbm_version": lgb.__version__,
        "catboost_seconds": catboost_seconds,
        "lightgbm_seconds": lightgbm_seconds,
        "ensemble_weights": {
            "catboost": 0.28719567,
            "lightgbm": 0.71280433,
        },
        "probability_calibration": {
            "type": "linear_affine_then_clip",
            "scale": 1.12708208,
            "intercept": -0.07336118,
        },
        "validation_brier": 0.247862497,
        "validation_score": 778.37,
    }
    metadata_path = MODEL_DIR / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f" metadata_saved={metadata_path}")

    print("\nModel sizes")
    for path in [catboost_path, lightgbm_path, columns_path, metadata_path]:
        print(f" {path.name}={os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
