"""EXP-015 CatBoost + engineered LightGBM 앙상블 추론."""

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

CATBOOST_FEATURES = [
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
CATBOOST_COEFFICIENT = 0.26157048
LIGHTGBM_COEFFICIENT = 0.85313287
ENSEMBLE_INTERCEPT = -0.06760608
SEASON_2025_ADJUSTMENT = -0.005


def add_catboost_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["count_code"] = out["balls_before"] * 4 + out["strikes_before"]
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


def _shrunk_rate(
    rate: pd.Series,
    count: pd.Series,
    strength: float,
    prior: float = 0.5,
) -> pd.Series:
    safe_count = count.fillna(0).clip(lower=0)
    safe_rate = rate.fillna(prior)
    return (safe_count * safe_rate + strength * prior) / (
        safe_count + strength
    )


def add_lightgbm_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["count_code"] = (
        out["balls_before"].astype(str)
        + "-"
        + out["strikes_before"].astype(str)
    )
    out["count_index"] = (
        out["balls_before"] * 4 + out["strikes_before"]
    ).astype("int8")
    out["count_out_state"] = (
        out["count_code"] + "-" + out["outs_before"].astype(str)
    )
    out["hand_matchup"] = (
        out["pitcher_hand"].astype(str)
        + "-"
        + out["batter_hand"].astype(str)
    )
    out["team_matchup"] = (
        out["pitcher_team_id"].astype(str)
        + "-"
        + out["batter_team_id"].astype(str)
    )
    out["is_full_count"] = (
        (out["balls_before"] == 3) & (out["strikes_before"] == 2)
    ).astype("int8")
    out["has_two_strikes"] = (out["strikes_before"] == 2).astype("int8")
    out["has_three_balls"] = (out["balls_before"] == 3).astype("int8")
    out["count_advantage"] = (
        out["strikes_before"] - out["balls_before"]
    ).astype("int8")
    out["runner_in_scoring_position"] = (
        (out["runner_on_2b"] == 1) | (out["runner_on_3b"] == 1)
    ).astype("int8")
    out["bases_loaded"] = (
        (out["runner_on_1b"] == 1)
        & (out["runner_on_2b"] == 1)
        & (out["runner_on_3b"] == 1)
    ).astype("int8")
    out["same_hand"] = (
        out["pitcher_hand"] == out["batter_hand"]
    ).astype("int8")
    out["late_inning"] = (out["inning"] >= 7).astype("int8")
    out["close_game"] = (
        out["score_diff_pitcher_team"].abs() <= 1
    ).astype("int8")
    out["log_li"] = np.log1p(out["li"].clip(lower=0))
    out["score_pressure"] = (
        out["score_diff_pitcher_team"].abs() * out["log_li"]
    )
    out["win_expectancy_gap"] = (
        out["home_win_expectancy"] - out["away_win_expectancy"]
    )
    out["pitcher_batter_success_gap"] = (
        out["asof_pitcher_success_rate"]
        - out["asof_batter_success_rate"]
    )
    out["pitcher_recent_success_delta_1_5"] = (
        out["asof_pitcher_prev1_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    )
    out["pitcher_recent_success_delta_3_5"] = (
        out["asof_pitcher_prev3_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    )
    out["pitcher_recent_middle_delta_1_5"] = (
        out["asof_pitcher_prev1_game_middle_rate"]
        - out["asof_pitcher_prev5_game_middle_rate"]
    )
    out["pitcher_recent_success_mean"] = out[
        [
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
        ]
    ].mean(axis=1)
    out["log_pitcher_n"] = np.log1p(out["asof_pitcher_n"].clip(lower=0))
    out["log_batter_n"] = np.log1p(out["asof_batter_n"].clip(lower=0))
    out["log_pitchmix_n"] = np.log1p(
        out["asof_pitcher_pitchmix_n"].clip(lower=0)
    )
    for strength in (50.0, 200.0, 500.0):
        suffix = int(strength)
        out[f"pitcher_success_shrunk_{suffix}"] = _shrunk_rate(
            out["asof_pitcher_success_rate"],
            out["asof_pitcher_n"],
            strength,
        )
        out[f"batter_success_shrunk_{suffix}"] = _shrunk_rate(
            out["asof_batter_success_rate"],
            out["asof_batter_n"],
            strength,
        )
    pitchmix = out[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].fillna(0.0).clip(lower=1e-12)
    out["pitchmix_entropy"] = -(pitchmix * np.log(pitchmix)).sum(axis=1)
    out["pitchmix_max_rate"] = pitchmix.max(axis=1)
    out["pitcher_failure_risk"] = out[
        [
            "asof_pitcher_reverse_rate",
            "asof_pitcher_middle_rate",
            "asof_pitcher_ball_rate",
        ]
    ].sum(axis=1, min_count=1)
    return out


def prepare_catboost_categories(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        out[column] = out[column].fillna("__MISSING__").astype(str)
    return out


def main() -> None:
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH, encoding="utf-8-sig")
    if list(submission.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError("sample_submission 컬럼 형식이 올바르지 않습니다.")
    ids = test[ID_COL].tolist()
    base = test.drop(columns=[ID_COL])

    catboost_input = add_catboost_features(base)
    catboost_input = catboost_input.loc[
        :, list(base.columns) + CATBOOST_FEATURES
    ]
    catboost_input = prepare_catboost_categories(catboost_input)
    catboost_model = CatBoostClassifier()
    catboost_model.load_model(MODEL_DIR / "catboost_model.cbm")
    catboost_predictions = catboost_model.predict_proba(catboost_input)[:, 1]

    with (MODEL_DIR / "engineered_features.json").open(
        encoding="utf-8"
    ) as file:
        engineered_features = json.load(file)
    with (MODEL_DIR / "lightgbm_columns.json").open(
        encoding="utf-8"
    ) as file:
        encoded_columns = json.load(file)
    lightgbm_input = add_lightgbm_features(base)
    lightgbm_input = lightgbm_input.loc[:, engineered_features]
    lightgbm_input = pd.get_dummies(
        lightgbm_input,
        columns=ONE_HOT_COLUMNS,
        dummy_na=True,
        dtype=np.int8,
    ).reindex(columns=encoded_columns, fill_value=0)
    lightgbm_model = lgb.Booster(
        model_file=str(MODEL_DIR / "lightgbm_model.txt")
    )
    lightgbm_predictions = lightgbm_model.predict(lightgbm_input)

    predictions = (
        CATBOOST_COEFFICIENT * catboost_predictions
        + LIGHTGBM_COEFFICIENT * lightgbm_predictions
        + ENSEMBLE_INTERCEPT
    )
    season_adjustment = np.where(
        base["season"].to_numpy() >= 2025,
        SEASON_2025_ADJUSTMENT,
        0.0,
    )
    predictions = np.clip(predictions + season_adjustment, 0.0, 1.0)

    prediction_map = dict(zip(ids, predictions))
    submission[TARGET_COL] = submission[ID_COL].map(prediction_map)
    if len(submission) != len(test):
        raise ValueError("test와 submission의 행 수가 다릅니다.")
    if submission[ID_COL].duplicated().any():
        raise ValueError("중복 row_id가 있습니다.")
    if submission[TARGET_COL].isna().any():
        raise ValueError("결측 예측값이 있습니다.")
    if not submission[TARGET_COL].between(0, 1).all():
        raise ValueError("예측값이 0~1 범위를 벗어났습니다.")
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | rows={len(submission)} | "
        f"mean={predictions.mean():.6f} | min={predictions.min():.6f} | "
        f"max={predictions.max():.6f}"
    )


if __name__ == "__main__":
    main()
