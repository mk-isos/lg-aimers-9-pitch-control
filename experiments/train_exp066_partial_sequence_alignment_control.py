"""EXP-066: high-confidence partial game-sequence TrackMan alignment.

Within the same season/month/day-of-week/team pair, each official game is
matched to the TrackMan game with the highest ordered core-state sequence
similarity.  Matches require SequenceMatcher ratio >= .98, mutual one-to-one
assignment, and only exactly equal matching blocks are aligned.  On all 2,418
previous full-sequence matches this procedure must reproduce the same game
with 100% accuracy before any downstream model is evaluated.
"""

from __future__ import annotations

import difflib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics
from train_exp041_exact_game_trackman_sequence import (
    OFFICIAL_COLUMNS,
    TRACKMAN_COLUMNS,
    official_games,
    trackman_games,
)
from train_exp043_exact_pitchtype_control_eb import (
    build_features,
    load_main,
    load_trackman,
)


DATA_DIR = Path("./data")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
EXACT_ROOT = Path("./artifacts/EXP-043/exact_pitchtype_control_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-066/partial_sequence_alignment_control")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
MIN_SEQUENCE_SCORE = 0.98
MIN_LENGTH_RATIO = 0.75
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "partial_direct_w010",
    "exact_partial_blend_w010",
    "exact010_partial_delta_w025",
)


def state_sequences(
    official: pd.DataFrame,
    official_game_id: np.ndarray,
    trackman_work: pd.DataFrame,
) -> tuple[
    dict[int, np.ndarray],
    dict[tuple[int, str], np.ndarray],
    dict[int, list[tuple[int, ...]]],
    dict[tuple[int, str], list[tuple[int, ...]]],
]:
    official_matrix = np.column_stack(
        [
            official["inning"],
            official["top_bottom"].astype(str).eq("B").astype(np.int8),
            official["balls_before"],
            official["strikes_before"],
            official["outs_before"],
            official["pitcher_team_id"],
            official["batter_team_id"],
        ]
    ).astype(np.int16)
    trackman_matrix = np.column_stack(
        [
            trackman_work["inning"],
            trackman_work["top_bottom"].astype(str).eq("Bottom").astype(np.int8),
            trackman_work["balls_before"],
            trackman_work["strikes_before"],
            trackman_work["outs_before"],
            trackman_work["pitcher_team_code"],
            trackman_work["batter_team_code"],
        ]
    ).astype(np.int16)
    official_indices = {
        int(key): values.to_numpy()
        for key, values in pd.Series(np.arange(len(official))).groupby(
            official_game_id, sort=False
        )
    }
    trackman_indices = {
        (int(season), str(game)): np.asarray(values)
        for (season, game), values in trackman_work.groupby(
            ["season", "trackman_game_id"], sort=False
        ).indices.items()
    }
    official_sequence = {
        key: [tuple(value) for value in official_matrix[index]]
        for key, index in official_indices.items()
    }
    trackman_sequence = {
        key: [tuple(value) for value in trackman_matrix[index]]
        for key, index in trackman_indices.items()
    }
    return official_indices, trackman_indices, official_sequence, trackman_sequence


def partial_aligned_rows() -> tuple[pd.DataFrame, dict[str, object]]:
    official = pd.read_csv(
        DATA_DIR / "train.csv", encoding="utf-8-sig", usecols=OFFICIAL_COLUMNS
    )
    official_season_row_index = official.groupby("season", sort=False).cumcount().to_numpy()
    trackman = pd.read_csv(
        DATA_DIR / "trackman_history.csv",
        encoding="utf-8-sig",
        usecols=TRACKMAN_COLUMNS,
    )
    official_summary, official_game_id = official_games(official)
    trackman_summary, trackman_work, _ = trackman_games(trackman)
    (
        official_indices,
        trackman_indices,
        official_sequence,
        trackman_sequence,
    ) = state_sequences(official, official_game_id, trackman_work)
    meta_columns = ["season", "month", "dayofweek", "team_low", "team_high"]
    trackman_groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for row in trackman_summary.itertuples(index=False):
        key = tuple(int(getattr(row, column)) for column in meta_columns)
        trackman_groups[key].append(str(row.trackman_game_id))
    exact = official_summary.merge(
        trackman_summary,
        on=[*meta_columns, "rows", "state_hash"],
        how="inner",
    )
    exact_map = {
        int(row.official_game_id): str(row.trackman_game_id)
        for row in exact.itertuples(index=False)
    }
    records: list[dict[str, object]] = []
    for row in official_summary.itertuples(index=False):
        official_id = int(row.official_game_id)
        key = tuple(int(getattr(row, column)) for column in meta_columns)
        left_sequence = official_sequence[official_id]
        candidates = []
        for trackman_id in trackman_groups.get(key, []):
            right_sequence = trackman_sequence[(int(row.season), trackman_id)]
            length_ratio = min(len(left_sequence), len(right_sequence)) / max(
                len(left_sequence), len(right_sequence)
            )
            if length_ratio < MIN_LENGTH_RATIO:
                continue
            matcher = difflib.SequenceMatcher(
                None, left_sequence, right_sequence, autojunk=False
            )
            blocks = [
                (block.a, block.b, block.size)
                for block in matcher.get_matching_blocks()
                if block.size > 0
            ]
            matched_rows = sum(block[2] for block in blocks)
            score = 2.0 * matched_rows / (
                len(left_sequence) + len(right_sequence)
            )
            candidates.append((score, matched_rows, trackman_id, blocks))
        if not candidates:
            continue
        candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
        best = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else 0.0
        if best[0] < MIN_SEQUENCE_SCORE:
            continue
        records.append(
            {
                "official_game_id": official_id,
                "season": int(row.season),
                "trackman_game_id": best[2],
                "score": float(best[0]),
                "margin": float(best[0] - second),
                "matched_rows": int(best[1]),
                "blocks": best[3],
                "exact_expected": exact_map.get(official_id),
            }
        )
    # Mutual one-to-one: retain a TrackMan game only for its unique best official game.
    best_by_trackman: dict[tuple[int, str], dict[str, object]] = {}
    for record in records:
        key = (int(record["season"]), str(record["trackman_game_id"]))
        current = best_by_trackman.get(key)
        if current is None or (
            float(record["score"]), int(record["matched_rows"])
        ) > (float(current["score"]), int(current["matched_rows"])):
            best_by_trackman[key] = record
    accepted = sorted(
        best_by_trackman.values(), key=lambda value: int(value["official_game_id"])
    )
    accepted_exact = [value for value in accepted if value["exact_expected"] is not None]
    exact_correct = sum(
        str(value["trackman_game_id"]) == str(value["exact_expected"])
        for value in accepted_exact
    )
    if exact_correct != len(exact_map) or len(accepted_exact) != len(exact_map):
        raise ValueError(
            "partial sequence matcher did not reproduce every exact match"
        )

    aligned: list[pd.DataFrame] = []
    for record in accepted:
        left_all = official_indices[int(record["official_game_id"])]
        right_all = trackman_indices[
            (int(record["season"]), str(record["trackman_game_id"]))
        ]
        left_parts = []
        right_parts = []
        for left_start, right_start, size in record["blocks"]:
            left_parts.append(left_all[left_start : left_start + size])
            right_parts.append(right_all[right_start : right_start + size])
        left = np.concatenate(left_parts)
        right = np.concatenate(right_parts)
        aligned.append(
            pd.DataFrame(
                {
                    "season": int(record["season"]),
                    "official_row_index": left,
                    "official_season_row_index": official_season_row_index[left],
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
                    "inning": official.iloc[left]["inning"].to_numpy(),
                    "outs_before": official.iloc[left]["outs_before"].to_numpy(),
                    "batter_hand": official.iloc[left]["batter_hand"].to_numpy(),
                    "pitcher_hand": official.iloc[left]["pitcher_hand"].to_numpy(),
                    "fine_pitch_type": trackman_work.iloc[right][
                        "fine_pitch_type"
                    ].to_numpy(),
                    "auto_fine_type": trackman_work.iloc[right][
                        "auto_fine_type"
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
    return output, {
        "official_games": len(official_summary),
        "trackman_games": len(trackman_summary),
        "exact_reference_games": len(exact_map),
        "accepted_games": len(accepted),
        "new_partial_games": len(accepted) - len(exact_map),
        "aligned_rows": len(output),
        "new_partial_matched_rows": int(
            sum(
                int(value["matched_rows"])
                for value in accepted
                if value["exact_expected"] is None
            )
        ),
        "exact_reidentification_accuracy": float(exact_correct / len(exact_map)),
        "minimum_sequence_score": MIN_SEQUENCE_SCORE,
        "minimum_length_ratio": MIN_LENGTH_RATIO,
        "minimum_accepted_score": float(min(value["score"] for value in accepted)),
        "minimum_accepted_margin": float(min(value["margin"] for value in accepted)),
        "matched_games_by_season": {
            str(season): int(sum(int(value["season"]) == season for value in accepted))
            for season in sorted(official["season"].unique())
        },
    }


def base_components(season: int) -> tuple[np.ndarray, np.ndarray]:
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    strict = np.load(
        LOWRANK_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
    ).astype(float)
    exact = np.load(EXACT_ROOT / f"predictions_fine_direct_w025_{season}.npy").astype(float)
    return 0.5 * recency + 0.5 * aggressive, (exact - strict) / 0.25


def main() -> None:
    started = time.time()
    main = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = partial_aligned_rows()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        recent, exact_correction = base_components(season)
        features, audits[str(season)] = build_features(
            main, aligned, trackman, season
        )
        mapped = features["trackman_mapped"].eq(1).to_numpy()
        partial = np.zeros(len(target), dtype=float)
        partial[mapped] = features.loc[
            mapped, "expected_minus_official"
        ].to_numpy(float)
        partial = np.clip(partial, -CORRECTION_CLIP, CORRECTION_CLIP)
        base = np.clip(recent + 0.10 * exact_correction, 0, 1)
        predictions = {
            "base": base,
            "partial_direct_w010": np.clip(recent + 0.10 * partial, 0, 1),
            "exact_partial_blend_w010": np.clip(
                recent + 0.05 * exact_correction + 0.05 * partial, 0, 1
            ),
            "exact010_partial_delta_w025": np.clip(
                base + 0.25 * (partial - exact_correction), 0, 1
            ),
        }
        fold: dict[str, object] = {
            "validation_season": season,
            "history_cutoff": season - 1,
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(target, prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{season}.npy", prediction)
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", target.astype(np.int8))
        folds[str(season)] = fold
        print(
            f"fold {season}: "
            + " ".join(
                f"{name}={fold[name]['skill_score_unclipped']:.2f}"
                for name in CANDIDATES
            ),
            flush=True,
        )
    aggregate: dict[str, object] = {}
    for name in ("base", *CANDIDATES):
        skills = {
            str(season): float(folds[str(season)][name]["skill_score_unclipped"])
            for season in REPORT_SEASONS
        }
        briers = {
            str(season): float(folds[str(season)][name]["brier_score"])
            for season in REPORT_SEASONS
        }
        aggregate[name] = {
            "season_briers": briers,
            "season_skills": skills,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills["2024"],
        }
    best = max(
        CANDIDATES,
        key=lambda name: (aggregate[name]["min_skill"], aggregate[name]["mean_skill"]),
    )
    result = {
        "experiment": "EXP-066",
        "candidate_family": "high_confidence_partial_sequence_alignment_control",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "alignment_and_tables_cutoff": "validation season-1",
            "same_state_matching_blocks_only": True,
            "exact_reference_reidentification_required": 1.0,
            "current_fold_labels_used_for_fit_or_selection": False,
            "actual_current_pitch_type_used": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "alignment_audit": alignment_audit,
        "fold_feature_audit": audits,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_fixed_candidate": best,
            "best_min_skill": aggregate[best]["min_skill"],
            "best_mean_skill": aggregate[best]["mean_skill"],
            "gate_each_season_1000": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
            ),
            "gate_mean_1100": bool(aggregate[best]["mean_skill"] >= 1100.0),
            "adopt": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
                and aggregate[best]["mean_skill"] >= 1100.0
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "total_seconds": time.time() - started,
    }
    (ARTIFACT_DIR / "validation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"best={best} mean={aggregate[best]['mean_skill']:.2f} "
        f"min={aggregate[best]['min_skill']:.2f} adopt={result['selection']['adopt']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
