"""EXP-013 구조를 2022~2024 rolling-origin으로 재평가한다.

기존 고정 앙상블 가중치와 affine 식은 2024에서 선택됐으므로 과거 fold에는
look-ahead 참고치일 뿐이다. 보정 전 고정 모델 구조와 고정 제출식 모두를
명시해 EXP-018과 비교할 때 낙관 편향을 숨기지 않는다.
"""

from __future__ import annotations

import json
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
from train_exp017_rolling_residual import calculate_metrics


DATA_DIR = Path("./data")
ARTIFACT_DIR = Path("./artifacts/EXP-013/rolling_2022_2024")
ID = "row_id"
TARGET = "control_success"
VALIDATION_SEASONS = [2022, 2023, 2024]
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
CATBOOST_WEIGHT = 0.28719567
LIGHTGBM_WEIGHT = 0.71280433
FIXED_SCALE = 1.12708208
FIXED_INTERCEPT = -0.07336118


def prepare_catboost(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        out[column] = out[column].fillna("__MISSING__").astype(str)
    return out


def main() -> None:
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
    catboost_data = prepare_catboost(train[features])
    lightgbm_data = pd.get_dummies(
        train[features],
        columns=ONE_HOT_COLUMNS,
        dummy_na=True,
        dtype=np.int8,
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        train_mask = train["season"].to_numpy() < validation_season
        validation_mask = train["season"].to_numpy() == validation_season
        y_train = train.loc[train_mask, TARGET]
        y_validation = train.loc[validation_mask, TARGET].to_numpy()

        cat_model = CatBoostClassifier(
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
            verbose=0,
            allow_writing_files=False,
        )
        started_at = time.time()
        cat_model.fit(
            catboost_data.loc[train_mask],
            y_train,
            cat_features=CATEGORICAL_COLUMNS,
        )
        cat_fit_seconds = time.time() - started_at
        cat_predictions = cat_model.predict_proba(
            catboost_data.loc[validation_mask]
        )[:, 1]

        lgb_model = LGBMClassifier(
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
        lgb_model.fit(lightgbm_data.loc[train_mask], y_train)
        lgb_fit_seconds = time.time() - started_at
        lgb_predictions = lgb_model.predict_proba(
            lightgbm_data.loc[validation_mask]
        )[:, 1]

        raw = (
            CATBOOST_WEIGHT * cat_predictions
            + LIGHTGBM_WEIGHT * lgb_predictions
        )
        fixed = np.clip(
            FIXED_SCALE * raw + FIXED_INTERCEPT,
            0.0,
            1.0,
        )
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "training_seasons": sorted(
                train.loc[train_mask, "season"].unique().astype(int).tolist()
            ),
            "catboost_fit_seconds": cat_fit_seconds,
            "lightgbm_fit_seconds": lgb_fit_seconds,
            "catboost": calculate_metrics(y_validation, cat_predictions),
            "lightgbm": calculate_metrics(y_validation, lgb_predictions),
            "raw_ensemble": calculate_metrics(y_validation, raw),
            "fixed_exp013_submission_formula": calculate_metrics(
                y_validation, fixed
            ),
        }
        np.save(
            ARTIFACT_DIR / f"raw_predictions_{validation_season}.npy",
            raw,
        )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            y_validation.astype(np.int8),
        )
        print(
            f"EXP-013 {validation_season}: "
            f"raw={folds[str(validation_season)]['raw_ensemble']['skill_score_unclipped']:.2f} "
            f"fixed={folds[str(validation_season)]['fixed_exp013_submission_formula']['skill_score_unclipped']:.2f} "
            f"cat_seconds={cat_fit_seconds:.1f} lgb_seconds={lgb_fit_seconds:.1f}"
        )
    raw_skills = [
        folds[str(season)]["raw_ensemble"]["skill_score_unclipped"]
        for season in VALIDATION_SEASONS
    ]
    fixed_skills = [
        folds[str(season)]["fixed_exp013_submission_formula"][
            "skill_score_unclipped"
        ]
        for season in VALIDATION_SEASONS
    ]
    result = {
        "experiment": "EXP-013",
        "evaluation": "rolling_2022_2024",
        "warning": (
            "ensemble weights and fixed affine coefficients were selected on "
            "2024; past-fold fixed-formula results are look-ahead diagnostics"
        ),
        "folds": folds,
        "aggregate": {
            "raw_mean_skill": float(np.mean(raw_skills)),
            "raw_min_skill": float(np.min(raw_skills)),
            "fixed_formula_mean_skill": float(np.mean(fixed_skills)),
            "fixed_formula_min_skill": float(np.min(fixed_skills)),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "catboost": catboost.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
