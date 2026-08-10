"""EXP-018 규정 준수 constrained multiscale 추론."""

from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
MODEL_DIR = Path("./model")
TEST_PATH = Path("./data/test.csv")
SAMPLE_SUBMISSION_PATH = Path("./data/sample_submission.csv")
OUTPUT_PATH = Path("./output/submission.csv")
ONE_HOT_COLUMNS = ["top_bottom", "game_type", "base_state"]
DROP_MODEL_COLUMNS = [ID_COL, TARGET_COL, "pitcher_id", "batter_id"]
SHRINKAGE_STRENGTHS = (10.0, 30.0, 100.0, 300.0)
RECENT_WEIGHT = 0.15
REVERSE_GROUP_WEIGHT = 0.30


def add_static_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["count_index"] = (
        out["balls_before"] * 4 + out["strikes_before"]
    ).astype("int8")
    out["count_out_index"] = (
        out["count_index"] * 3 + out["outs_before"]
    ).astype("int8")
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
    out["log_li"] = np.log1p(out["li"].clip(lower=0)).astype("float32")
    out["score_pressure"] = (
        out["score_diff_pitcher_team"].abs() * out["log_li"]
    ).astype("float32")
    out["win_expectancy_gap"] = (
        out["home_win_expectancy"] - out["away_win_expectancy"]
    ).astype("float32")
    out["pitcher_batter_success_gap"] = (
        out["asof_pitcher_success_rate"]
        - out["asof_batter_success_rate"]
    ).astype("float32")
    out["pitcher_recent_success_delta_1_5"] = (
        out["asof_pitcher_prev1_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    ).astype("float32")
    out["pitcher_recent_success_delta_3_5"] = (
        out["asof_pitcher_prev3_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    ).astype("float32")
    out["pitcher_recent_middle_delta_1_5"] = (
        out["asof_pitcher_prev1_game_middle_rate"]
        - out["asof_pitcher_prev5_game_middle_rate"]
    ).astype("float32")
    out["log_pitcher_n"] = np.log1p(
        out["asof_pitcher_n"].clip(lower=0)
    ).astype("float32")
    out["log_batter_n"] = np.log1p(
        out["asof_batter_n"].clip(lower=0)
    ).astype("float32")
    out["log_pitchmix_n"] = np.log1p(
        out["asof_pitcher_pitchmix_n"].clip(lower=0)
    ).astype("float32")
    return out


def load_entity_state(
    records: list[dict[str, object]],
    id_column: str,
) -> pd.DataFrame:
    return pd.DataFrame.from_records(records).set_index(id_column)


def attach_entity_features(
    rows: pd.DataFrame,
    entity: str,
    state: pd.DataFrame,
    league_prior: float,
) -> pd.DataFrame:
    out = rows.copy()
    id_column = f"{entity}_id"
    count_column = f"asof_{entity}_n"
    rate_column = f"asof_{entity}_success_rate"
    ids = out[id_column]
    prior_n = ids.map(state["prior_n"]).fillna(0.0).to_numpy(dtype=float)
    prior_successes = (
        ids.map(state["prior_successes"])
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    prior_exists = ids.isin(state.index).to_numpy(dtype=np.int8)
    career_n = out[count_column].to_numpy(dtype=float)
    career_successes = np.rint(
        career_n * out[rate_column].fillna(0.0).to_numpy(dtype=float)
    )
    season_n = career_n - prior_n
    season_successes = career_successes - prior_successes
    if np.any(season_n < -1e-6):
        raise ValueError(f"{entity}: 누적 표본 수가 저장된 과거값보다 작습니다.")
    if np.any(season_successes < -0.01) or np.any(
        season_successes - season_n > 0.01
    ):
        raise ValueError(f"{entity}: 복원한 현재 시즌 성공 수가 잘못됐습니다.")
    season_n = np.maximum(season_n, 0.0)
    season_successes = np.clip(season_successes, 0.0, season_n)
    prior_rate = np.divide(
        prior_successes,
        prior_n,
        out=np.full(len(out), league_prior, dtype=float),
        where=prior_n > 0,
    )
    season_rate = np.divide(
        season_successes,
        season_n,
        out=np.full(len(out), league_prior, dtype=float),
        where=season_n > 0,
    )
    player_prior = (prior_successes + 200.0 * league_prior) / (
        prior_n + 200.0
    )
    prefix = f"temporal_{entity}"
    out[f"{prefix}_prior_exists"] = prior_exists
    out[f"{prefix}_prior_n"] = prior_n.astype("float32")
    out[f"{prefix}_log_prior_n"] = np.log1p(prior_n).astype("float32")
    out[f"{prefix}_prior_rate"] = prior_rate.astype("float32")
    out[f"{prefix}_prior_rate_shrunk_200"] = player_prior.astype("float32")
    out[f"{prefix}_season_n"] = season_n.astype("float32")
    out[f"{prefix}_log_season_n"] = np.log1p(season_n).astype("float32")
    out[f"{prefix}_season_rate"] = season_rate.astype("float32")
    out[f"{prefix}_season_minus_prior_rate"] = (
        season_rate - prior_rate
    ).astype("float32")
    for strength in SHRINKAGE_STRENGTHS:
        suffix = int(strength)
        out[f"{prefix}_season_global_{suffix}"] = (
            (season_successes + strength * league_prior)
            / (season_n + strength)
        ).astype("float32")
        out[f"{prefix}_season_player_{suffix}"] = (
            (season_successes + strength * player_prior)
            / (season_n + strength)
        ).astype("float32")
        out[f"{prefix}_reliability_{suffix}"] = (
            season_n / (season_n + strength)
        ).astype("float32")
    return out


def attach_temporal_features(
    frame: pd.DataFrame,
    history: dict[str, object],
) -> pd.DataFrame:
    through_season = int(history["through_season"])
    if (frame["season"] <= through_season).any():
        raise ValueError("평가 시즌이 학습 이력 이후가 아닙니다.")
    league_rate = float(history["league_rate"])
    out = frame.copy()
    out["temporal_prior_league_rate"] = np.float32(league_rate)
    for entity in ("pitcher", "batter"):
        state = load_entity_state(
            history[entity],
            f"{entity}_id",
        )
        out = attach_entity_features(
            out,
            entity,
            state,
            league_rate,
        )
    out["temporal_base_global_30"] = (
        0.7 * out["temporal_pitcher_season_global_30"]
        + 0.3 * out["temporal_batter_season_global_30"]
    ).astype("float32")
    out["temporal_base_player_30"] = (
        0.7 * out["temporal_pitcher_season_player_30"]
        + 0.3 * out["temporal_batter_season_player_30"]
    ).astype("float32")
    return out


def effect_map(
    records: list[dict[str, object]],
    columns: list[str],
) -> dict[tuple[int, ...], float]:
    return {
        tuple(int(record[column]) for column in columns): float(record["effect"])
        for record in records
    }


def map_group_effects(
    frame: pd.DataFrame,
    effects: dict[str, list[dict[str, object]]],
) -> np.ndarray:
    base_columns = ["count_index", "pitcher_hand", "batter_hand"]
    reverse_columns = base_columns + ["reverse_rate_bin"]
    keys = frame[base_columns].astype(int)
    base_lookup = effect_map(effects["base"], base_columns)
    base_effect = np.fromiter(
        (
            base_lookup.get(tuple(row), 0.0)
            for row in keys.itertuples(index=False, name=None)
        ),
        dtype=float,
        count=len(frame),
    )
    reverse_rate = frame["asof_pitcher_reverse_rate"].to_numpy(dtype=float)
    reverse_bin = np.where(
        np.isfinite(reverse_rate),
        np.floor(reverse_rate / 0.05),
        -1,
    ).astype(int)
    reverse_keys = keys.copy()
    reverse_keys["reverse_rate_bin"] = reverse_bin
    reverse_lookup = effect_map(effects["reverse"], reverse_columns)
    reverse_effect = np.fromiter(
        (
            reverse_lookup.get(tuple(row), 0.0)
            for row in reverse_keys.itertuples(index=False, name=None)
        ),
        dtype=float,
        count=len(frame),
    )
    return (
        (1.0 - REVERSE_GROUP_WEIGHT) * base_effect
        + REVERSE_GROUP_WEIGHT * reverse_effect
    )


def validate_inputs(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != [ID_COL, TARGET_COL]:
        raise ValueError("sample_submission 컬럼 형식이 올바르지 않습니다.")
    if len(test) != len(submission):
        raise ValueError("test와 sample_submission 행 수가 다릅니다.")
    if test[ID_COL].isna().any() or submission[ID_COL].isna().any():
        raise ValueError("row_id에 결측값이 있습니다.")
    if test[ID_COL].duplicated().any() or submission[ID_COL].duplicated().any():
        raise ValueError("row_id에 중복값이 있습니다.")
    if set(test[ID_COL]) != set(submission[ID_COL]):
        raise ValueError("test와 sample_submission의 row_id 집합이 다릅니다.")


def main() -> None:
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH, encoding="utf-8-sig")
    validate_inputs(test, submission)
    base_input = test.drop(columns=[ID_COL])
    engineered = add_static_features(base_input)
    history = json.loads(
        (MODEL_DIR / "history_state.json").read_text(encoding="utf-8")
    )
    engineered = attach_temporal_features(engineered, history)
    base_predictions = engineered["temporal_base_global_30"].to_numpy(
        dtype=float
    )

    feature_names = json.loads(
        (MODEL_DIR / "encoded_features.json").read_text(encoding="utf-8")
    )
    model_features = [
        column
        for column in engineered.columns
        if column not in DROP_MODEL_COLUMNS
    ]
    encoded = pd.get_dummies(
        engineered[model_features],
        columns=ONE_HOT_COLUMNS,
        dummy_na=True,
        dtype=np.int8,
    ).reindex(columns=feature_names, fill_value=0)
    residual_model = lgb.Booster(
        model_file=str(MODEL_DIR / "recent_residual_lightgbm.txt")
    )
    residual_predictions = residual_model.predict(
        encoded.to_numpy(dtype=np.float32)
    )
    effects = json.loads(
        (MODEL_DIR / "group_effects.json").read_text(encoding="utf-8")
    )
    group_correction = map_group_effects(engineered, effects)
    predictions = np.clip(
        base_predictions
        + group_correction
        + RECENT_WEIGHT * residual_predictions,
        0.0,
        1.0,
    )

    prediction_map = dict(zip(test[ID_COL], predictions))
    submission[TARGET_COL] = submission[ID_COL].map(prediction_map)
    if submission[TARGET_COL].isna().any():
        raise ValueError("submission에 결측 예측값이 있습니다.")
    if not submission[TARGET_COL].between(0.0, 1.0).all():
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
