"""EXP-015 최종 후보: 기존 CatBoost + 개선 LightGBM 전체 학습.

CatBoost는 EXP-013에서 전체 2019~2024 데이터로 학습한 네이티브 모델을
재사용한다. LightGBM은 2024 홀드아웃에서 가장 좋았던 EXP-014 설정과
엔지니어드 피처로 전체 데이터를 다시 학습한다.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from train_exp014_temporal_categorical_lgbm import (
    BASE_CATEGORICAL_COLUMNS,
    add_features,
)


DATA_DIR = Path("./data")
SOURCE_CATBOOST = Path("./submissions/EXP-013/model/catboost_model.cbm")
MODEL_DIR = Path("./submissions/EXP-015/model")
ID = "row_id"
TARGET = "control_success"
STRING_ONLY_COLUMNS = [
    "count_code",
    "count_out_state",
    "hand_matchup",
    "team_matchup",
]


def main() -> None:
    print("Environment")
    print(f" python={platform.python_version()}")
    print(f" pandas={pd.__version__}")
    print(f" numpy={np.__version__}")
    print(f" catboost={catboost.__version__}")
    print(f" lightgbm={lgb.__version__}")

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
    engineered_features = [
        column
        for column in train.columns
        if column != TARGET and column not in STRING_ONLY_COLUMNS
    ]
    encoded_train = pd.get_dummies(
        train[engineered_features],
        columns=BASE_CATEGORICAL_COLUMNS,
        dummy_na=True,
        dtype=np.int8,
    )
    print(
        f"train_rows={len(train)} | raw_features={len(engineered_features)} | "
        f"encoded_features={len(encoded_train.columns)}"
    )

    model = LGBMClassifier(
        objective="binary",
        n_estimators=278,
        learning_rate=0.015,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=1000,
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
    started_at = time.time()
    model.fit(encoded_train, train[TARGET])
    fit_seconds = time.time() - started_at

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lightgbm_path = MODEL_DIR / "lightgbm_model.txt"
    model.booster_.save_model(lightgbm_path)
    with (MODEL_DIR / "lightgbm_columns.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(list(encoded_train.columns), file, ensure_ascii=False)
    with (MODEL_DIR / "engineered_features.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(engineered_features, file, ensure_ascii=False)

    if not SOURCE_CATBOOST.exists():
        raise FileNotFoundError(f"CatBoost 모델이 없습니다: {SOURCE_CATBOOST}")
    shutil.copy2(SOURCE_CATBOOST, MODEL_DIR / "catboost_model.cbm")

    metadata = {
        "experiment": "EXP-015",
        "training_seasons": [2019, 2020, 2021, 2022, 2023, 2024],
        "training_rows": int(len(train)),
        "engineered_features": len(engineered_features),
        "encoded_features": len(encoded_train.columns),
        "lightgbm_best_iteration_from_2024_validation": 278,
        "lightgbm_fit_seconds": fit_seconds,
        "validation": {
            "split": "2019-2023 train / 2024 validation",
            "catboost_lightgbm_brier": 0.24781493248510744,
            "catboost_lightgbm_score": 797.4135455498965,
            "exp013_score": 778.37,
        },
        "ensemble": {
            "catboost_coefficient": 0.26157048,
            "lightgbm_coefficient": 0.85313287,
            "intercept": -0.06760608,
            "season_2025_adjustment": -0.005,
        },
        "format": {
            "catboost": "native cbm",
            "lightgbm": "native text",
            "pickle": False,
        },
    }
    with (MODEL_DIR / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"fit_seconds={fit_seconds:.1f}")
    for path in sorted(MODEL_DIR.iterdir()):
        print(f"{path.name}={os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
