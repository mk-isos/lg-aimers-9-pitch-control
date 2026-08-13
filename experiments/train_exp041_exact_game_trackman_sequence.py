"""EXP-041: exact game-sequence Trackman ID alignment.

Official rows are contiguous by game.  A game boundary is reconstructed from
team-pair and inning-half resets, then its ordered public pre-pitch state
sequence is compared with each Trackman game after sorting by ``pitch_no``.
Only exact, full-sequence matches are accepted.  For every outer fold, player
ID mappings are learned exclusively from matched games in earlier seasons.

The downstream feature/model definition is deliberately held identical to
EXP-033, isolating the value of exact alignment from feature changes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import train_exp033_trackman_sequence_trend as sequence
from trackman_features import MappingResult, TEAM_ID_TO_TRACKMAN


DATA_DIR = Path("./data")
EXPERIMENT = os.environ.get("EXACT_TRACKMAN_EXPERIMENT", "EXP-041")
ARTIFACT_DIR = Path(
    os.environ.get(
        "EXACT_TRACKMAN_ARTIFACT_DIR",
        "./artifacts/EXP-041/exact_game_trackman_sequence",
    )
)
SOURCE_POLICY = os.environ.get("EXACT_TRACKMAN_SOURCE_POLICY", "pooled_equal")
ALIASES = {"SK_WYV": "SSG", "SSG_LAN": "SSG"}
MIN_ALIGNED_ROWS = 5
MIN_MAPPING_PURITY = 0.99

OFFICIAL_COLUMNS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_id",
    "batter_id",
    "control_success",
]
TRACKMAN_COLUMNS = [
    "season",
    "game_month",
    "game_dayofweek",
    "trackman_game_id",
    "pitch_no",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team",
    "batter_team",
    "pitcher_trackman_id",
    "batter_trackman_id",
    "tagged_pitch_type",
    "auto_pitch_type",
    "pitch_type_group",
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]


def official_games(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    phase = 2 * (frame["inning"].to_numpy(dtype=np.int16) - 1)
    phase += frame["top_bottom"].astype(str).eq("B").to_numpy(dtype=np.int16)
    team_low = np.minimum(frame["pitcher_team_id"], frame["batter_team_id"])
    team_high = np.maximum(frame["pitcher_team_id"], frame["batter_team_id"])
    boundary = (
        frame["season"].ne(frame["season"].shift()).to_numpy()
        | pd.Series(team_low).ne(pd.Series(team_low).shift()).to_numpy()
        | pd.Series(team_high).ne(pd.Series(team_high).shift()).to_numpy()
        | (phase < np.roll(phase, 1))
    )
    boundary[0] = True
    reset = (
        (phase == 0)
        & frame["run_top_before"].eq(0).to_numpy()
        & frame["run_bot_before"].eq(0).to_numpy()
        & (
            (np.roll(phase, 1) > 0)
            | (frame["run_top_before"].shift(fill_value=0).to_numpy() > 0)
            | (frame["run_bot_before"].shift(fill_value=0).to_numpy() > 0)
        )
    )
    boundary |= reset
    game_id = np.cumsum(boundary)
    matrix = np.column_stack(
        [
            frame["inning"],
            frame["top_bottom"].astype(str).eq("B").astype(np.int8),
            frame["balls_before"],
            frame["strikes_before"],
            frame["outs_before"],
            frame["pitcher_hand"],
            frame["batter_hand"],
            frame["pitcher_team_id"],
            frame["batter_team_id"],
        ]
    ).astype(np.int16)
    summaries: list[dict[str, object]] = []
    for value in np.unique(game_id):
        indices = np.flatnonzero(game_id == value)
        first = frame.iloc[int(indices[0])]
        summaries.append(
            {
                "official_game_id": int(value),
                "season": int(first["season"]),
                "month": int(first["game_month"]),
                "dayofweek": int(first["game_dayofweek"]),
                "team_low": int(
                    min(first["pitcher_team_id"], first["batter_team_id"])
                ),
                "team_high": int(
                    max(first["pitcher_team_id"], first["batter_team_id"])
                ),
                "rows": int(len(indices)),
                "state_hash": hashlib.sha256(matrix[indices].tobytes()).hexdigest(),
            }
        )
    return pd.DataFrame(summaries), game_id


def trackman_games(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    inverse_teams = {name: code for code, name in TEAM_ID_TO_TRACKMAN.items()}
    work = frame.copy()
    work["fine_pitch_type"] = sequence.canonical_pitch_type(
        work["tagged_pitch_type"]
    )
    work["pitcher_team"] = work["pitcher_team"].replace(ALIASES)
    work["batter_team"] = work["batter_team"].replace(ALIASES)
    work["pitcher_team_code"] = work["pitcher_team"].map(inverse_teams)
    work["batter_team_code"] = work["batter_team"].map(inverse_teams)
    work = work.loc[
        work["pitcher_team_code"].notna()
        & work["batter_team_code"].notna()
    ].copy()
    work = work.sort_values(
        ["season", "trackman_game_id", "pitch_no"]
    ).reset_index(drop=True)
    matrix = np.column_stack(
        [
            work["inning"],
            work["top_bottom"].astype(str).eq("Bottom").astype(np.int8),
            work["balls_before"],
            work["strikes_before"],
            work["outs_before"],
            work["pitcher_hand"].map({"Left": 1, "Right": 2}),
            work["batter_hand"].map({"Left": 1, "Right": 2}),
            work["pitcher_team_code"],
            work["batter_team_code"],
        ]
    ).astype(np.int16)
    summaries: list[dict[str, object]] = []
    grouped = work.groupby(["season", "trackman_game_id"], sort=False).indices
    for (season, game_id), indices in grouped.items():
        indices = np.asarray(indices, dtype=np.int64)
        first = work.iloc[int(indices[0])]
        summaries.append(
            {
                "trackman_game_id": str(game_id),
                "season": int(season),
                "month": int(first["game_month"]),
                "dayofweek": int(first["game_dayofweek"]),
                "team_low": int(
                    min(first["pitcher_team_code"], first["batter_team_code"])
                ),
                "team_high": int(
                    max(first["pitcher_team_code"], first["batter_team_code"])
                ),
                "rows": int(len(indices)),
                "state_hash": hashlib.sha256(matrix[indices].tobytes()).hexdigest(),
            }
        )
    row_group = pd.MultiIndex.from_frame(work[["season", "trackman_game_id"]])
    return pd.DataFrame(summaries), work, row_group


def exact_aligned_rows() -> tuple[pd.DataFrame, dict[str, object]]:
    official = pd.read_csv(
        DATA_DIR / "train.csv", encoding="utf-8-sig", usecols=OFFICIAL_COLUMNS
    )
    trackman = pd.read_csv(
        DATA_DIR / "trackman_history.csv",
        encoding="utf-8-sig",
        usecols=TRACKMAN_COLUMNS,
    )
    official_summary, official_game_id = official_games(official)
    trackman_summary, trackman_work, _ = trackman_games(trackman)
    keys = [
        "season",
        "month",
        "dayofweek",
        "team_low",
        "team_high",
        "rows",
        "state_hash",
    ]
    matches = official_summary.merge(trackman_summary, on=keys, how="inner")
    if matches["official_game_id"].duplicated().any():
        raise ValueError("ambiguous exact official game match")
    if matches["trackman_game_id"].duplicated().any():
        raise ValueError("ambiguous exact Trackman game match")

    official_indices = pd.Series(np.arange(len(official))).groupby(
        official_game_id, sort=False
    ).apply(lambda values: values.to_numpy())
    trackman_indices = trackman_work.groupby(
        ["season", "trackman_game_id"], sort=False
    ).indices
    aligned: list[pd.DataFrame] = []
    for match in matches.itertuples(index=False):
        left = official_indices.loc[int(match.official_game_id)]
        right = np.asarray(
            trackman_indices[(int(match.season), str(match.trackman_game_id))]
        )
        if len(left) != len(right):
            raise ValueError("exact-game length mismatch")
        aligned.append(
            pd.DataFrame(
                {
                    "season": int(match.season),
                    "pitcher_id": official.iloc[left]["pitcher_id"].to_numpy(),
                    "pitcher_trackman_id": trackman_work.iloc[right][
                        "pitcher_trackman_id"
                    ].to_numpy(),
                    "batter_id": official.iloc[left]["batter_id"].to_numpy(),
                    "batter_trackman_id": trackman_work.iloc[right][
                        "batter_trackman_id"
                    ].to_numpy(),
                    "control_success": official.iloc[left][
                        "control_success"
                    ].to_numpy(),
                    "count_index": (
                        4 * official.iloc[left]["balls_before"].to_numpy()
                        + official.iloc[left]["strikes_before"].to_numpy()
                    ),
                    "batter_hand": official.iloc[left][
                        "batter_hand"
                    ].to_numpy(),
                    "pitcher_hand": official.iloc[left][
                        "pitcher_hand"
                    ].to_numpy(),
                    "fine_pitch_type": trackman_work.iloc[right][
                        "fine_pitch_type"
                    ].to_numpy(),
                    "pitch_type_group": trackman_work.iloc[right][
                        "pitch_type_group"
                    ].to_numpy(),
                    "rel_speed": trackman_work.iloc[right]["rel_speed"].to_numpy(),
                    "spin_rate": trackman_work.iloc[right]["spin_rate"].to_numpy(),
                    "induced_vert_break": trackman_work.iloc[right][
                        "induced_vert_break"
                    ].to_numpy(),
                    "horz_break": trackman_work.iloc[right]["horz_break"].to_numpy(),
                    "extension": trackman_work.iloc[right]["extension"].to_numpy(),
                    "rel_height": trackman_work.iloc[right]["rel_height"].to_numpy(),
                    "rel_side": trackman_work.iloc[right]["rel_side"].to_numpy(),
                    "zone_speed": trackman_work.iloc[right]["zone_speed"].to_numpy(),
                }
            )
        )
    output = pd.concat(aligned, ignore_index=True)
    audit = {
        "official_games": int(len(official_summary)),
        "core_trackman_games": int(len(trackman_summary)),
        "exact_matched_games": int(len(matches)),
        "exact_aligned_rows": int(len(output)),
        "exact_aligned_fraction_of_train": float(len(output) / len(official)),
        "matched_games_by_season": {
            str(key): int(value)
            for key, value in matches.groupby("season").size().items()
        },
    }
    return output, audit


def mapping_from_aligned(
    aligned: pd.DataFrame, cutoff_season: int
) -> tuple[MappingResult, dict[str, object]]:
    source = aligned.loc[aligned["season"].le(cutoff_season)]
    counts = source.groupby(["pitcher_id", "pitcher_trackman_id"]).size()
    totals = counts.groupby(level=0).sum()
    maxima = counts.groupby(level=0).max()
    best_pairs = counts.groupby(level=0).idxmax()
    purity = maxima / totals
    accepted_ids = purity.index[
        purity.ge(MIN_MAPPING_PURITY) & totals.ge(MIN_ALIGNED_ROWS)
    ]
    mapping = {
        int(player_id): int(best_pairs.loc[player_id][1])
        for player_id in accepted_ids
    }
    result = MappingResult(
        mapping=mapping,
        costs={player_id: 0.0 for player_id in mapping},
        candidate_main_ids=int(source["pitcher_id"].nunique()),
        candidate_trackman_ids=int(source["pitcher_trackman_id"].nunique()),
    )
    audit = {
        "cutoff_season": int(cutoff_season),
        "source_aligned_rows": int(len(source)),
        "observed_official_pitchers": int(source["pitcher_id"].nunique()),
        "accepted_pitchers": int(len(mapping)),
        "minimum_rows": MIN_ALIGNED_ROWS,
        "minimum_purity": MIN_MAPPING_PURITY,
        "weighted_majority_purity": float(maxima.sum() / totals.sum()),
    }
    return result, audit


def main() -> None:
    aligned, alignment_audit = exact_aligned_rows()
    mappings: dict[int, MappingResult] = {}
    mapping_audits: dict[str, object] = {}
    for cutoff in (2020, 2021, 2022, 2023, 2024):
        mappings[cutoff], mapping_audits[str(cutoff)] = mapping_from_aligned(
            aligned, cutoff
        )

    def exact_builder(
        main: pd.DataFrame,
        trackman: pd.DataFrame,
        cutoff_season: int,
        max_cost: float = 0.1,
    ) -> MappingResult:
        del main, trackman, max_cost
        return mappings[int(cutoff_season)]

    sequence.EXPERIMENT = EXPERIMENT
    sequence.ARTIFACT_DIR = ARTIFACT_DIR
    sequence.SOURCE_POLICY = SOURCE_POLICY
    sequence.build_pitcher_mapping = exact_builder
    sequence.main()

    metrics_path = ARTIFACT_DIR / "validation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["exact_alignment"] = {
        **alignment_audit,
        "mapping_by_cutoff": mapping_audits,
        "current_fold_or_future_games_used_for_mapping": False,
        "match_fields": [
            "season",
            "game_month",
            "game_dayofweek",
            "team_pair",
            "ordered inning/top_bottom/count/outs/hands/team sequence",
        ],
    }
    metrics["validation_protocol"]["mapping"] = (
        "exact full-game sequence matches from seasons <= validation season-1"
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
