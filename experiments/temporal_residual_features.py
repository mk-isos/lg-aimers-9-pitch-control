"""행별 공식 as-of 값에서 시점 안전한 현재 시즌 피처를 복원한다.

평가 행끼리 집계하지 않는다. 학습 데이터에서 확정한 직전 시즌 종료 상태를
고정한 뒤, 각 현재 행의 공식 누적값과의 차이만 계산한다. 따라서 평가 서버의
각 행은 다른 평가 행과 독립적으로 변환할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET = "control_success"
ENTITY_NAMES = ("pitcher", "batter")
SHRINKAGE_STRENGTHS = (10.0, 30.0, 100.0, 300.0)


@dataclass(frozen=True)
class HistoryState:
    """한 시즌이 끝난 시점의 선수별 누적 상태."""

    pitcher: pd.DataFrame
    batter: pd.DataFrame
    league_rate: float
    through_season: int


def add_static_features(frame: pd.DataFrame) -> pd.DataFrame:
    """현재 행의 공식 입력만으로 정적 상호작용 피처를 만든다."""
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

def _empty_entity_state(entity: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[f"{entity}_id", "prior_n", "prior_successes"]
    ).set_index(f"{entity}_id")


def _update_entity_state(
    season_rows: pd.DataFrame,
    entity: str,
    target: str,
) -> pd.DataFrame:
    id_column = f"{entity}_id"
    count_column = f"asof_{entity}_n"
    rate_column = f"asof_{entity}_success_rate"
    end_indices = season_rows.groupby(id_column, sort=False)[
        count_column
    ].idxmax()
    last = season_rows.loc[
        end_indices,
        [id_column, count_column, rate_column, target],
    ]
    counts_before = last[count_column].to_numpy(dtype=float)
    successes_before = np.rint(
        counts_before
        * last[rate_column].fillna(0.0).to_numpy(dtype=float)
    )
    state = pd.DataFrame(
        {
            id_column: last[id_column].to_numpy(),
            "prior_n": counts_before + 1.0,
            "prior_successes": (
                successes_before + last[target].to_numpy(dtype=float)
            ),
        }
    ).set_index(id_column)
    return state


def _attach_entity_features(
    rows: pd.DataFrame,
    entity: str,
    state: pd.DataFrame,
    league_prior: float,
) -> pd.DataFrame:
    """고정된 과거 상태와 현재 행 하나의 누적값만 사용한다."""
    out = rows.copy()
    id_column = f"{entity}_id"
    count_column = f"asof_{entity}_n"
    rate_column = f"asof_{entity}_success_rate"
    ids = out[id_column]
    if state.empty:
        prior_n = np.zeros(len(out), dtype=float)
        prior_successes = np.zeros(len(out), dtype=float)
        prior_exists = np.zeros(len(out), dtype=np.int8)
    else:
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
        raise ValueError(f"{entity}: 현재 누적 표본 수가 과거 종료값보다 작습니다.")
    if np.any(season_successes < -0.01) or np.any(
        season_successes - season_n > 0.01
    ):
        raise ValueError(f"{entity}: 복원한 현재 시즌 성공 수가 범위를 벗어났습니다.")
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
        global_posterior = (
            season_successes + strength * league_prior
        ) / (season_n + strength)
        player_posterior = (
            season_successes + strength * player_prior
        ) / (season_n + strength)
        out[f"{prefix}_season_global_{suffix}"] = (
            global_posterior.astype("float32")
        )
        out[f"{prefix}_season_player_{suffix}"] = (
            player_posterior.astype("float32")
        )
        out[f"{prefix}_reliability_{suffix}"] = (
            season_n / (season_n + strength)
        ).astype("float32")
    return out


def attach_training_temporal_features(
    frame: pd.DataFrame,
    target: str = TARGET,
) -> tuple[pd.DataFrame, HistoryState]:
    """각 학습 행에는 오직 이전 시즌까지 확정된 상태를 붙인다."""
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("학습 데이터는 시즌 오름차순이어야 합니다.")
    outputs: list[pd.DataFrame] = []
    states = {entity: _empty_entity_state(entity) for entity in ENTITY_NAMES}
    previous_league_rate = 0.5
    through_season = -1
    for season in sorted(frame["season"].astype(int).unique()):
        season_rows = frame.loc[frame["season"] == season].copy()
        season_rows["temporal_prior_league_rate"] = np.float32(
            previous_league_rate
        )
        for entity in ENTITY_NAMES:
            season_rows = _attach_entity_features(
                season_rows,
                entity,
                states[entity],
                previous_league_rate,
            )
        season_rows["temporal_base_global_30"] = (
            0.7 * season_rows["temporal_pitcher_season_global_30"]
            + 0.3 * season_rows["temporal_batter_season_global_30"]
        ).astype("float32")
        season_rows["temporal_base_player_30"] = (
            0.7 * season_rows["temporal_pitcher_season_player_30"]
            + 0.3 * season_rows["temporal_batter_season_player_30"]
        ).astype("float32")
        outputs.append(season_rows)
        for entity in ENTITY_NAMES:
            current_state = _update_entity_state(season_rows, entity, target)
            if states[entity].empty:
                states[entity] = current_state
            else:
                states[entity] = pd.concat(
                    [states[entity], current_state]
                )
                states[entity] = states[entity][
                    ~states[entity].index.duplicated(keep="last")
                ]
        previous_league_rate = float(season_rows[target].mean())
        through_season = season
    output = pd.concat(outputs).sort_index()
    return output, HistoryState(
        pitcher=states["pitcher"],
        batter=states["batter"],
        league_rate=previous_league_rate,
        through_season=through_season,
    )


def attach_inference_temporal_features(
    frame: pd.DataFrame,
    history_state: HistoryState,
) -> pd.DataFrame:
    """고정된 학습 이력으로 평가 행을 서로 독립적으로 변환한다."""
    if (frame["season"] <= history_state.through_season).any():
        raise ValueError("추론 시즌은 저장된 학습 이력 이후여야 합니다.")
    out = frame.copy()
    out["temporal_prior_league_rate"] = np.float32(
        history_state.league_rate
    )
    for entity in ENTITY_NAMES:
        out = _attach_entity_features(
            out,
            entity,
            getattr(history_state, entity),
            history_state.league_rate,
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
