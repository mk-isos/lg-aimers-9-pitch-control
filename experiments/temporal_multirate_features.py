"""공식 누적 비율에서 현재 시즌 다중 비율을 행별로 복원한다.

현재 행의 누적 표본/비율과 이전 시즌까지 저장한 선수 상태만 사용한다.
평가 데이터의 다른 행은 읽거나 집계하지 않는다. `control_success` 이외의
과거 마지막 투구 세부 분류는 train에 정답이 없으므로, 시즌 종료 상태에서
해당 분류 count는 마지막 투구 직전 값을 보수적으로 유지한다. 이로 인한
오차는 선수·비율당 최대 한 건이며 복원 후 유효 범위로 자른다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET = "control_success"
SHRINKAGE = 30.0
PLAYER_PRIOR_STRENGTH = 200.0


@dataclass(frozen=True)
class RateGroup:
    name: str
    id_column: str
    n_column: str
    rates: tuple[tuple[str, str], ...]
    exact_target_metric: str | None = None


RATE_GROUPS = (
    RateGroup(
        name="pitcher_control",
        id_column="pitcher_id",
        n_column="asof_pitcher_n",
        rates=(
            ("success", "asof_pitcher_success_rate"),
            ("reverse", "asof_pitcher_reverse_rate"),
            ("middle", "asof_pitcher_middle_rate"),
            ("ball", "asof_pitcher_ball_rate"),
            ("strike", "asof_pitcher_strike_rate"),
        ),
        exact_target_metric="success",
    ),
    RateGroup(
        name="batter_control",
        id_column="batter_id",
        n_column="asof_batter_n",
        rates=(
            ("success", "asof_batter_success_rate"),
            ("middle", "asof_batter_middle_rate"),
        ),
        exact_target_metric="success",
    ),
    RateGroup(
        name="pitcher_pitchmix",
        id_column="pitcher_id",
        n_column="asof_pitcher_pitchmix_n",
        rates=(
            ("fastball", "asof_pitcher_fastball_rate"),
            ("breaking", "asof_pitcher_breaking_rate"),
            ("offspeed", "asof_pitcher_offspeed_rate"),
        ),
    ),
)

DEFAULT_GLOBAL_RATES = {
    "pitcher_control_success": 0.5,
    "pitcher_control_reverse": 0.2,
    "pitcher_control_middle": 0.15,
    "pitcher_control_ball": 0.35,
    "pitcher_control_strike": 0.45,
    "batter_control_success": 0.5,
    "batter_control_middle": 0.15,
    "pitcher_pitchmix_fastball": 0.5,
    "pitcher_pitchmix_breaking": 0.35,
    "pitcher_pitchmix_offspeed": 0.15,
}


@dataclass
class MultirateState:
    tables: dict[str, pd.DataFrame]
    global_rates: dict[str, float]
    through_season: int


def _empty_state(group: RateGroup) -> pd.DataFrame:
    columns = [group.id_column, "prior_n"] + [
        f"prior_{metric}_count" for metric, _ in group.rates
    ]
    return pd.DataFrame(columns=columns).set_index(group.id_column)


def _attach_group_features(
    rows: pd.DataFrame,
    group: RateGroup,
    state: pd.DataFrame,
    global_rates: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = rows.copy()
    ids = out[group.id_column]
    if state.empty:
        prior_n = np.zeros(len(out), dtype=np.float64)
    else:
        prior_n = ids.map(state["prior_n"]).fillna(0.0).to_numpy(dtype=float)
    career_n = out[group.n_column].fillna(0.0).to_numpy(dtype=float)
    season_n_raw = career_n - prior_n
    if np.any(season_n_raw < -1e-6):
        raise ValueError(f"{group.name}: career n is smaller than stored prior n")
    season_n = np.maximum(season_n_raw, 0.0)
    prefix = f"multirate_{group.name}"
    out[f"{prefix}_season_n"] = season_n.astype(np.float32)
    out[f"{prefix}_log_season_n"] = np.log1p(season_n).astype(np.float32)
    out[f"{prefix}_reliability_30"] = (
        season_n / (season_n + SHRINKAGE)
    ).astype(np.float32)
    diagnostics = {"negative_count": 0, "above_n_count": 0}

    for metric, rate_column in group.rates:
        key = f"{group.name}_{metric}"
        global_rate = float(global_rates[key])
        if state.empty:
            prior_count = np.zeros(len(out), dtype=np.float64)
        else:
            prior_count = (
                ids.map(state[f"prior_{metric}_count"])
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
        rate = out[rate_column].fillna(global_rate).to_numpy(dtype=float)
        career_count = np.rint(career_n * rate)
        season_count_raw = career_count - prior_count
        diagnostics["negative_count"] += int((season_count_raw < -0.01).sum())
        diagnostics["above_n_count"] += int(
            (season_count_raw - season_n > 0.01).sum()
        )
        season_count = np.clip(season_count_raw, 0.0, season_n)
        prior_rate = np.divide(
            prior_count,
            prior_n,
            out=np.full(len(out), global_rate, dtype=float),
            where=prior_n > 0,
        )
        player_prior = (prior_count + PLAYER_PRIOR_STRENGTH * global_rate) / (
            prior_n + PLAYER_PRIOR_STRENGTH
        )
        season_rate = np.divide(
            season_count,
            season_n,
            out=np.full(len(out), global_rate, dtype=float),
            where=season_n > 0,
        )
        global_posterior = (season_count + SHRINKAGE * global_rate) / (
            season_n + SHRINKAGE
        )
        player_posterior = (season_count + SHRINKAGE * player_prior) / (
            season_n + SHRINKAGE
        )
        metric_prefix = f"{prefix}_{metric}"
        out[f"{metric_prefix}_prior_rate"] = prior_rate.astype(np.float32)
        out[f"{metric_prefix}_prior_shrunk_200"] = player_prior.astype(
            np.float32
        )
        out[f"{metric_prefix}_season_rate"] = season_rate.astype(np.float32)
        out[f"{metric_prefix}_season_global_30"] = global_posterior.astype(
            np.float32
        )
        out[f"{metric_prefix}_season_player_30"] = player_posterior.astype(
            np.float32
        )
        out[f"{metric_prefix}_season_minus_prior"] = (
            season_rate - prior_rate
        ).astype(np.float32)
    return out, diagnostics


def _updated_state(
    season_rows: pd.DataFrame,
    group: RateGroup,
    target: str,
) -> pd.DataFrame:
    end_indices = season_rows.groupby(group.id_column, sort=False)[
        group.n_column
    ].idxmax()
    columns = [group.id_column, group.n_column, target] + [
        rate_column for _, rate_column in group.rates
    ]
    last = season_rows.loc[end_indices, columns]
    n_before = last[group.n_column].fillna(0.0).to_numpy(dtype=float)
    data: dict[str, np.ndarray] = {
        group.id_column: last[group.id_column].to_numpy(),
        "prior_n": n_before + 1.0,
    }
    for metric, rate_column in group.rates:
        count = np.rint(
            n_before
            * last[rate_column]
            .fillna(DEFAULT_GLOBAL_RATES[f"{group.name}_{metric}"])
            .to_numpy(dtype=float)
        )
        if metric == group.exact_target_metric:
            count += last[target].to_numpy(dtype=float)
        data[f"prior_{metric}_count"] = count
    return pd.DataFrame(data).set_index(group.id_column)


def _merge_state(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        return new
    combined = pd.concat([old, new])
    return combined[~combined.index.duplicated(keep="last")]


def _global_rates_from_states(
    tables: dict[str, pd.DataFrame],
) -> dict[str, float]:
    rates = dict(DEFAULT_GLOBAL_RATES)
    for group in RATE_GROUPS:
        state = tables[group.name]
        total_n = float(state["prior_n"].sum())
        if total_n <= 0:
            continue
        for metric, _ in group.rates:
            key = f"{group.name}_{metric}"
            rates[key] = float(state[f"prior_{metric}_count"].sum() / total_n)
    return rates


def attach_training_multirate_features(
    frame: pd.DataFrame,
    target: str = TARGET,
) -> tuple[pd.DataFrame, MultirateState, dict[str, dict[str, int]]]:
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("training frame must be ordered by season")
    tables = {group.name: _empty_state(group) for group in RATE_GROUPS}
    global_rates = dict(DEFAULT_GLOBAL_RATES)
    diagnostics: dict[str, dict[str, int]] = {}
    outputs: list[pd.DataFrame] = []
    through_season = -1
    for season in sorted(frame["season"].astype(int).unique()):
        rows = frame.loc[frame["season"] == season].copy()
        for group in RATE_GROUPS:
            rows, group_diagnostics = _attach_group_features(
                rows, group, tables[group.name], global_rates
            )
            diagnostics[f"{season}_{group.name}"] = group_diagnostics
        outputs.append(rows)
        for group in RATE_GROUPS:
            tables[group.name] = _merge_state(
                tables[group.name], _updated_state(rows, group, target)
            )
        global_rates = _global_rates_from_states(tables)
        through_season = season
    return (
        pd.concat(outputs).sort_index(),
        MultirateState(tables, global_rates, through_season),
        diagnostics,
    )


def attach_inference_multirate_features(
    frame: pd.DataFrame,
    state: MultirateState,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    if (frame["season"] <= state.through_season).any():
        raise ValueError("inference season must follow stored history")
    rows = frame.copy()
    diagnostics: dict[str, dict[str, int]] = {}
    for group in RATE_GROUPS:
        rows, group_diagnostics = _attach_group_features(
            rows,
            group,
            state.tables[group.name],
            state.global_rates,
        )
        diagnostics[group.name] = group_diagnostics
    return rows, diagnostics
