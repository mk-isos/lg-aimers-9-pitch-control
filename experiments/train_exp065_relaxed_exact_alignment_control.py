"""EXP-065: conflict-free relaxed exact-game TrackMan alignment.

The existing alignment hashes hands as part of the full ordered state.  A
hand-independent core hash (inning half, count, outs and team orientation)
recovers every exact match without conflict and adds 97 unique games.  This
experiment rebuilds the tagged pitch-type control correction on that expanded
past-only label set and compares it with the immutable EXP-051 base.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
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
ARTIFACT_DIR = Path("./artifacts/EXP-065/relaxed_exact_alignment_control")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "relaxed_direct_w010",
    "exact_relaxed_blend_w010",
    "exact010_relaxed_delta_w050",
)


def core_aligned_rows() -> tuple[pd.DataFrame, dict[str, object]]:
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
    official_indices = pd.Series(np.arange(len(official))).groupby(
        official_game_id, sort=False
    ).apply(lambda values: values.to_numpy())
    trackman_indices = trackman_work.groupby(
        ["season", "trackman_game_id"], sort=False
    ).indices
    official_hash = {
        int(game_id): hashlib.sha256(official_matrix[index].tobytes()).hexdigest()
        for game_id, index in official_indices.items()
    }
    trackman_hash = {
        (int(season), str(game_id)): hashlib.sha256(
            trackman_matrix[np.asarray(index)].tobytes()
        ).hexdigest()
        for (season, game_id), index in trackman_indices.items()
    }
    official_summary = official_summary.copy()
    trackman_summary = trackman_summary.copy()
    official_summary["core_hash"] = official_summary["official_game_id"].map(
        official_hash
    )
    trackman_summary["core_hash"] = [
        trackman_hash[(int(row.season), str(row.trackman_game_id))]
        for row in trackman_summary.itertuples(index=False)
    ]
    keys = [
        "season",
        "month",
        "dayofweek",
        "team_low",
        "team_high",
        "rows",
        "core_hash",
    ]
    matches = official_summary.merge(trackman_summary, on=keys, how="inner")
    if matches["official_game_id"].duplicated().any():
        raise ValueError("ambiguous relaxed official match")
    if matches["trackman_game_id"].duplicated().any():
        raise ValueError("ambiguous relaxed TrackMan match")
    aligned: list[pd.DataFrame] = []
    for match in matches.itertuples(index=False):
        left = official_indices.loc[int(match.official_game_id)]
        right = np.asarray(
            trackman_indices[(int(match.season), str(match.trackman_game_id))]
        )
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
                }
            )
        )
    output = pd.concat(aligned, ignore_index=True)
    return output, {
        "official_games": len(official_summary),
        "trackman_games": len(trackman_summary),
        "relaxed_matched_games": len(matches),
        "relaxed_aligned_rows": len(output),
        "matched_games_by_season": {
            str(key): int(value)
            for key, value in matches.groupby("season").size().items()
        },
        "match_fields": keys,
        "ambiguous_official": 0,
        "ambiguous_trackman": 0,
    }


def base_components(season: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return 0.5 * recency + 0.5 * aggressive, strict, (exact - strict) / 0.25


def main() -> None:
    started = time.time()
    main = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = core_aligned_rows()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        recent, _, exact_correction = base_components(season)
        features, audits[str(season)] = build_features(
            main, aligned, trackman, season
        )
        mapped = features["trackman_mapped"].eq(1).to_numpy()
        relaxed = np.zeros(len(target), dtype=float)
        relaxed[mapped] = features.loc[
            mapped, "expected_minus_official"
        ].to_numpy(float)
        relaxed = np.clip(relaxed, -CORRECTION_CLIP, CORRECTION_CLIP)
        base = np.clip(recent + 0.10 * exact_correction, 0, 1)
        predictions = {
            "base": base,
            "relaxed_direct_w010": np.clip(recent + 0.10 * relaxed, 0, 1),
            "exact_relaxed_blend_w010": np.clip(
                recent + 0.05 * exact_correction + 0.05 * relaxed, 0, 1
            ),
            "exact010_relaxed_delta_w050": np.clip(
                base + 0.50 * (relaxed - exact_correction), 0, 1
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
        "experiment": "EXP-065",
        "candidate_family": "conflict_free_relaxed_exact_alignment_control",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "alignment_and_tables_cutoff": "validation season-1",
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
