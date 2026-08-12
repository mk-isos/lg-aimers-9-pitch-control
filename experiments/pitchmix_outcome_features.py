"""Row-order-independent train labels from cumulative pitch-mix transitions."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


PITCH_GROUP_NAMES: Final[tuple[str, ...]] = (
    "fastball",
    "breaking",
    "offspeed",
)
PITCH_RATE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"asof_pitcher_{name}_rate" for name in PITCH_GROUP_NAMES
)
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "row_id",
    "pitcher_id",
    "season",
    "asof_pitcher_n",
    "asof_pitcher_pitchmix_n",
    *PITCH_RATE_COLUMNS,
)


def reconstruct_pitch_group(
    frame: pd.DataFrame,
) -> tuple[pd.Series, dict[str, object]]:
    """Recover the current train pitch group from the next cumulative state."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing pitch-mix columns: {missing}")
    pitcher_n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy(float)
    key = pd.MultiIndex.from_arrays(
        [frame["pitcher_id"], frame["season"], pitcher_n],
        names=["pitcher_id", "season", "asof_pitcher_n"],
    )
    duplicate = key.duplicated(keep=False)
    key_valid = np.isfinite(pitcher_n) & (pitcher_n >= 0) & ~duplicate
    positions = np.flatnonzero(key_valid)
    lookup = pd.Series(positions, index=key[key_valid], dtype="int64")
    successor_key = pd.MultiIndex.from_arrays(
        [frame["pitcher_id"], frame["season"], pitcher_n + 1],
        names=key.names,
    )
    successor = lookup.reindex(successor_key).to_numpy(dtype=float)
    pair = key_valid & np.isfinite(successor)

    mix_n = pd.to_numeric(
        frame["asof_pitcher_pitchmix_n"], errors="coerce"
    ).to_numpy(float)
    rates = frame.loc[:, PITCH_RATE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    zero_history_missing = (mix_n[:, None] == 0.0) & ~np.isfinite(rates)
    safe_rates = np.where(zero_history_missing, 0.0, rates)
    counts = np.rint(mix_n[:, None] * safe_rates)
    deltas = np.full((len(frame), len(PITCH_GROUP_NAMES)), np.nan, dtype=float)
    delta_n = np.full(len(frame), np.nan, dtype=float)
    current = np.flatnonzero(pair)
    if len(current):
        following = successor[current].astype(np.int64)
        delta_n[current] = mix_n[following] - mix_n[current]
        deltas[current] = counts[following] - counts[current]
    onehot = (
        np.isfinite(deltas).all(axis=1)
        & np.isin(deltas, [0.0, 1.0]).all(axis=1)
        & (deltas.sum(axis=1) == 1.0)
    )
    valid = pair & np.isfinite(delta_n) & (delta_n == 1.0) & onehot
    group = np.full(len(frame), np.nan, dtype=float)
    group[valid] = np.argmax(deltas[valid], axis=1)

    seasons = frame["season"].to_numpy()
    per_season: dict[str, object] = {}
    for season in sorted(pd.unique(frame["season"]).tolist()):
        local = valid & (seasons == season)
        per_season[str(int(season))] = {
            "rows": int(np.sum(seasons == season)),
            "valid_rows": int(local.sum()),
            "class_counts": {
                name: int(np.sum(group[local] == index))
                for index, name in enumerate(PITCH_GROUP_NAMES)
            },
        }
    diagnostics = {
        "rows": int(len(frame)),
        "unique_key_rows": int(key_valid.sum()),
        "duplicate_key_rows": int(duplicate.sum()),
        "candidate_pair_rows": int(pair.sum()),
        "valid_onehot_rows": int(valid.sum()),
        "invalid_pair_rows": int((pair & ~valid).sum()),
        "all_valid_delta_pitchmix_n_equal_one": bool(
            np.all(delta_n[valid] == 1.0)
        ),
        "all_valid_group_deltas_onehot": bool(onehot[valid].all()),
        "row_order_dependency": False,
        "pair_lookup": "unique (pitcher_id, season, asof_pitcher_n + 1)",
        "class_names": list(PITCH_GROUP_NAMES),
        "per_season": per_season,
    }
    return pd.Series(group, index=frame.index, name="pitch_group_class"), diagnostics
