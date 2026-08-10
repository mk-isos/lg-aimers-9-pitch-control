"""EXP-013 CatBoost + LightGBM 보정 앙상블 추론."""

from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ID_COL = "row_id"
TARGET_COL = "control_success"
MODEL_DIR = Path("./model")
TEST_PATH = Path("./data/test.csv")
SAMPLE_SUBMISSION_PATH = Path("./data/sample_submission.csv")
OUTPUT_PATH = Path("./output/submission.csv")

NEW_FEATURES = [
    "count_code",
    "is_full_count",
    "runner_in_scoring_position",
    "same_hand",
    "pitcher_batter_success_gap",
    "pitcher_recent_success_delta",
]
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
CALIBRATION_SCALE = 1.12708208
CALIBRATION_INTERCEPT = -0.07336118


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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


def prepare_catboost_categories(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        out[column] = out[column].fillna("__MISSING__").astype(str)
    return out


def load_sample_submission(path: Path) -> pd.DataFrame:
    submission = pd.read_csv(path, encoding="utf-8-sig")
    if list(submission.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            "sample_submission 컬럼이 row_id, control_success가 아닙니다."
        )
    return submission


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    if len(submission) != len(test):
        raise ValueError("test와 submission의 행 개수가 다릅니다.")
    if submission[ID_COL].duplicated().any():
        raise ValueError("submission에 중복 row_id가 있습니다.")
    if submission[TARGET_COL].isna().any():
        raise ValueError("submission에 결측 예측값이 있습니다.")
    if not submission[TARGET_COL].between(0, 1).all():
        raise ValueError("예측 확률이 0~1 범위를 벗어났습니다.")


def main() -> None:
    print("Load test data...")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    submission = load_sample_submission(SAMPLE_SUBMISSION_PATH)
    ids = test[ID_COL].tolist()
    base_features = [column for column in test.columns if column != ID_COL]
    features = base_features + NEW_FEATURES
    test_features = add_features(test.drop(columns=[ID_COL]))
    test_features = test_features.loc[:, features]
    print(f" test={len(test)} | features={len(features)}")

    print("Load and predict CatBoost...")
    catboost_model = CatBoostClassifier()
    catboost_model.load_model(MODEL_DIR / "catboost_model.cbm")
    catboost_input = prepare_catboost_categories(test_features)
    catboost_predictions = catboost_model.predict_proba(catboost_input)[:, 1]

    print("Load and predict LightGBM...")
    with (MODEL_DIR / "lightgbm_columns.json").open(
        encoding="utf-8"
    ) as file:
        lightgbm_columns = json.load(file)
    lightgbm_input = pd.get_dummies(
        test_features,
        columns=ONE_HOT_COLUMNS,
        dummy_na=True,
        dtype=np.int8,
    )
    lightgbm_input = lightgbm_input.reindex(
        columns=lightgbm_columns,
        fill_value=0,
    )
    lightgbm_model = lgb.Booster(
        model_file=str(MODEL_DIR / "lightgbm_model.txt")
    )
    lightgbm_predictions = lightgbm_model.predict(lightgbm_input)

    print("Build calibrated ensemble...")
    raw_predictions = (
        CATBOOST_WEIGHT * catboost_predictions
        + LIGHTGBM_WEIGHT * lightgbm_predictions
    )
    predictions = np.clip(
        CALIBRATION_SCALE * raw_predictions + CALIBRATION_INTERCEPT,
        0.0,
        1.0,
    )

    prediction_map = dict(zip(ids, predictions))
    missing_ids = [
        row_id
        for row_id in submission[ID_COL]
        if row_id not in prediction_map
    ]
    if missing_ids:
        raise ValueError(f"예측이 없는 row_id가 {len(missing_ids)}개 있습니다.")
    submission[TARGET_COL] = submission[ID_COL].map(prediction_map)
    validate_submission(submission, test)
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | rows={len(submission)} | "
        f"mean={predictions.mean():.6f} | "
        f"min={predictions.min():.6f} | max={predictions.max():.6f}"
    )


if __name__ == "__main__":
    main()
