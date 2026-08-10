"""EXP-002 제출용 추론 코드.

평가 서버에서 ./data/test.csv를 읽고 ./output/submission.csv를 생성한다.
"""

from __future__ import annotations

import os

import joblib
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
MODEL_PATH = "./model/rf_exp002.pkl"
TEST_PATH = "./data/test.csv"
SAMPLE_SUBMISSION_PATH = "./data/sample_submission.csv"
OUTPUT_PATH = "./output/submission.csv"


def load_test(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in frame.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없습니다.")
    return frame


def load_sample_submission(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    expected_columns = [ID_COL, TARGET_COL]
    if list(frame.columns[:2]) != expected_columns:
        raise ValueError(
            f"submission 컬럼이 {expected_columns}와 다릅니다: "
            f"{list(frame.columns)}"
        )
    return frame


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """학습 코드와 동일한 상황 조합 피처 6개를 만든다."""
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


def build_features(test: pd.DataFrame, model) -> pd.DataFrame:
    features = add_features(test.drop(columns=[ID_COL]))
    expected_features = list(model.feature_names_in_)
    missing = [column for column in expected_features if column not in features.columns]
    if missing:
        raise ValueError(f"추론 피처가 누락됐습니다: {missing}")
    return features.loc[:, expected_features]


def merge_predictions(
    submission: pd.DataFrame,
    ids: list[str],
    predictions,
) -> pd.DataFrame:
    prediction_map = dict(zip(ids, predictions))
    missing_ids = [row_id for row_id in submission[ID_COL] if row_id not in prediction_map]
    if missing_ids:
        raise ValueError(f"예측이 없는 row_id가 {len(missing_ids)}개 있습니다.")
    submission[TARGET_COL] = submission[ID_COL].map(prediction_map)
    return submission


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    if len(submission) != len(test):
        raise ValueError("test와 submission의 행 개수가 다릅니다.")
    if submission[ID_COL].duplicated().any():
        raise ValueError("submission에 중복 row_id가 있습니다.")
    if submission[TARGET_COL].isna().any():
        raise ValueError("submission에 결측 예측값이 있습니다.")
    if not submission[TARGET_COL].between(0, 1).all():
        raise ValueError("예측 확률이 0~1 범위를 벗어났습니다.")


def main() -> None:
    print("Load model...")
    model = joblib.load(MODEL_PATH)
    print(f" OK. n_features={getattr(model, 'n_features_in_', '?')}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    submission = load_sample_submission(SAMPLE_SUBMISSION_PATH)
    print(f" test={len(test)} submission={len(submission)}")

    print("Build EXP-002 features...")
    features = build_features(test, model)
    print(f" features={features.shape[1]}")

    print("Inference model...")
    predictions = model.predict_proba(features)[:, 1]
    print(f" predictions={len(predictions)}")

    print("Build submission...")
    submission = merge_predictions(
        submission,
        test[ID_COL].tolist(),
        predictions,
    )
    validate_submission(submission, test)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUTPUT_PATH} (rows={len(submission)})")


if __name__ == "__main__":
    main()
