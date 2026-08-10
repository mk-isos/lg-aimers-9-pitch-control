"""과거 시즌의 정답만 사용하는 시간 안전 Target Encoding."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


DEFAULT_GROUPS: list[tuple[str, ...]] = [
    ("pitcher_id",),
    ("batter_id",),
    ("pitcher_id", "count_index"),
    ("pitcher_id", "batter_hand"),
    ("batter_id", "pitcher_hand"),
    ("pitcher_team_id", "count_index"),
    ("count_index", "base_state", "outs_before"),
    ("inning", "count_index"),
]


def _safe_name(columns: tuple[str, ...]) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", "_".join(columns))


def _map_group_statistics(
    reference: pd.DataFrame,
    rows: pd.DataFrame,
    group_columns: tuple[str, ...],
    target: str,
    smoothing: float,
    fallback: float,
) -> tuple[np.ndarray, np.ndarray]:
    if reference.empty:
        return (
            np.full(len(rows), fallback, dtype=np.float32),
            np.zeros(len(rows), dtype=np.float32),
        )
    statistics = reference.groupby(list(group_columns), dropna=False)[target].agg(
        ["sum", "count"]
    )
    if len(group_columns) == 1:
        keys = rows[group_columns[0]]
    else:
        keys = pd.MultiIndex.from_frame(rows[list(group_columns)])
    sums = statistics["sum"].reindex(keys).to_numpy(dtype=float)
    counts = statistics["count"].reindex(keys).to_numpy(dtype=float)
    counts = np.nan_to_num(counts, nan=0.0)
    sums = np.nan_to_num(sums, nan=0.0)
    rates = (sums + smoothing * fallback) / (counts + smoothing)
    return rates.astype(np.float32), np.log1p(counts).astype(np.float32)


def attach_temporal_target_features(
    frame: pd.DataFrame,
    target: str,
    max_season: int,
    smoothing: float = 100.0,
    groups: list[tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """각 행의 시즌보다 과거인 정답만 이용해 TE 피처를 붙인다."""
    selected_groups = groups or DEFAULT_GROUPS
    output = frame.copy()
    feature_names: list[str] = []
    seasons = sorted(
        season
        for season in output["season"].dropna().astype(int).unique()
        if season <= max_season
    )
    for group_columns in selected_groups:
        group_name = _safe_name(group_columns)
        history_rate_name = f"te_hist_{group_name}_rate"
        history_count_name = f"te_hist_{group_name}_log_n"
        last_rate_name = f"te_last_{group_name}_rate"
        last_count_name = f"te_last_{group_name}_log_n"
        output[history_rate_name] = np.float32(0.5)
        output[history_count_name] = np.float32(0.0)
        output[last_rate_name] = np.float32(0.5)
        output[last_count_name] = np.float32(0.0)
        feature_names.extend(
            [
                history_rate_name,
                history_count_name,
                last_rate_name,
                last_count_name,
            ]
        )

        for season in seasons:
            row_mask = output["season"] == season
            rows = output.loc[row_mask]
            historical = output.loc[
                (output["season"] < season) & output[target].notna()
            ]
            fallback = float(historical[target].mean()) if len(historical) else 0.5
            history_rates, history_counts = _map_group_statistics(
                historical,
                rows,
                group_columns,
                target,
                smoothing,
                fallback,
            )
            last_season = historical.loc[historical["season"] == season - 1]
            last_fallback = (
                float(last_season[target].mean())
                if len(last_season)
                else fallback
            )
            last_rates, last_counts = _map_group_statistics(
                last_season,
                rows,
                group_columns,
                target,
                smoothing,
                last_fallback,
            )
            output.loc[row_mask, history_rate_name] = history_rates
            output.loc[row_mask, history_count_name] = history_counts
            output.loc[row_mask, last_rate_name] = last_rates
            output.loc[row_mask, last_count_name] = last_counts
    return output, feature_names
