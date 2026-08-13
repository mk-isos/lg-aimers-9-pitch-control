"""EXP-052: temporal source policies for exact Trackman control correction.

Instead of row-pooling all earlier exact-aligned seasons, this experiment fits
one hierarchical fine-pitch control table per source season and combines the
resulting current-row corrections with fixed equal/recency/last policies.
Evaluation rows are never aggregated and the actual current pitch type is not
used.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics
from train_exp041_exact_game_trackman_sequence import (
    exact_aligned_rows,
    mapping_from_aligned,
)
from train_exp043_exact_pitchtype_control_eb import (
    FINE_TYPES,
    load_main,
    load_trackman,
    propensity_table,
)


LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
POOLED_ROOT = Path("./artifacts/EXP-043/exact_pitchtype_control_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-052/trackman_control_source_policies")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
PITCHER_SMOOTHING = 150.0
TYPE_SMOOTHING = 60.0
CONTEXT_SMOOTHING = 30.0
CORRECTION_CLIP = 0.03
CORRECTION_WEIGHT = 0.10
CANDIDATES = (
    "pooled_w010",
    "source_equal_w010",
    "source_recency2_w010",
    "source_last_w010",
)


def source_tables(
    aligned: pd.DataFrame, source_season: int
) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    history = aligned.loc[aligned["season"].eq(source_season)]
    league = float(history["control_success"].mean())
    pitcher_stats = history.groupby("pitcher_trackman_id")["control_success"].agg(
        ["sum", "count"]
    )
    pitcher_rate = (
        pitcher_stats["sum"] + PITCHER_SMOOTHING * league
    ) / (pitcher_stats["count"] + PITCHER_SMOOTHING)
    type_keys = ["pitcher_trackman_id", "fine_pitch_type"]
    type_stats = history.groupby(type_keys)["control_success"].agg(["sum", "count"])
    pitcher_index = type_stats.index.get_level_values("pitcher_trackman_id")
    type_prior = pitcher_rate.reindex(pitcher_index).fillna(league).to_numpy(float)
    type_rate = pd.Series(
        (type_stats["sum"].to_numpy(float) + TYPE_SMOOTHING * type_prior)
        / (type_stats["count"].to_numpy(float) + TYPE_SMOOTHING),
        index=type_stats.index,
    )
    context_keys = [
        "pitcher_trackman_id",
        "fine_pitch_type",
        "count_index",
        "batter_hand",
    ]
    context_stats = history.groupby(context_keys)["control_success"].agg(
        ["sum", "count"]
    )
    context_type_index = pd.MultiIndex.from_arrays(
        [
            context_stats.index.get_level_values("pitcher_trackman_id"),
            context_stats.index.get_level_values("fine_pitch_type"),
        ],
        names=type_keys,
    )
    context_prior = type_rate.reindex(context_type_index).to_numpy(float)
    context_rate = pd.Series(
        (context_stats["sum"].to_numpy(float) + CONTEXT_SMOOTHING * context_prior)
        / (context_stats["count"].to_numpy(float) + CONTEXT_SMOOTHING),
        index=context_stats.index,
    )
    return pitcher_rate, type_rate, context_rate, league


def source_correction(
    rows: pd.DataFrame,
    mapped: pd.Series,
    propensity: pd.DataFrame,
    tables: tuple[pd.Series, pd.Series, pd.Series, float],
) -> np.ndarray:
    pitcher_rate, type_rate, context_rate, league = tables
    query_index = pd.MultiIndex.from_arrays(
        [mapped, rows["count_index"], rows["batter_hand"]],
        names=["pitcher_trackman_id", "count_index", "batter_hand_code"],
    )
    weights = propensity.reindex(query_index).to_numpy(float)
    overall = pitcher_rate.reindex(mapped).to_numpy(float)
    valid = np.isfinite(overall) & np.isfinite(weights).all(axis=1)
    rate_matrix = np.empty((len(rows), len(FINE_TYPES)), dtype=np.float32)
    for position, pitch_type in enumerate(FINE_TYPES):
        context_index = pd.MultiIndex.from_arrays(
            [
                mapped,
                np.full(len(rows), pitch_type, dtype=object),
                rows["count_index"],
                rows["batter_hand"],
            ],
            names=[
                "pitcher_trackman_id",
                "fine_pitch_type",
                "count_index",
                "batter_hand",
            ],
        )
        type_index = pd.MultiIndex.from_arrays(
            [mapped, np.full(len(rows), pitch_type, dtype=object)],
            names=["pitcher_trackman_id", "fine_pitch_type"],
        )
        rate = context_rate.reindex(context_index).to_numpy(float)
        fallback = type_rate.reindex(type_index).to_numpy(float)
        rate = np.where(np.isfinite(rate), rate, fallback)
        rate_matrix[:, position] = np.where(np.isfinite(rate), rate, overall)
    expected = np.sum(np.nan_to_num(weights) * rate_matrix, axis=1)
    official = rows["asof_pitcher_success_rate"].to_numpy(float)
    correction = np.zeros(len(rows))
    valid &= np.isfinite(official)
    correction[valid] = np.clip(
        expected[valid] - official[valid], -CORRECTION_CLIP, CORRECTION_CLIP
    )
    return correction


def recent_base(season: int) -> np.ndarray:
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    return 0.5 * recency + 0.5 * aggressive


def main() -> None:
    started = time.time()
    main_frame = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = exact_aligned_rows()
    all_tables = {
        season: source_tables(aligned, season)
        for season in sorted(aligned["season"].unique())
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    audits: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        rows = main_frame.loc[
            main_frame["season"].eq(validation_season)
        ].reset_index(drop=True)
        target = np.load(
            LOWRANK_ROOT / f"targets_{validation_season}.npy"
        ).astype(float)
        base = recent_base(validation_season)
        mapping, mapping_audit = mapping_from_aligned(
            aligned, validation_season - 1
        )
        mapped = rows["pitcher_id"].map(mapping.mapping)
        propensity = propensity_table(trackman, validation_season - 1)
        source_seasons = [
            int(season)
            for season in sorted(all_tables)
            if season < validation_season
        ]
        source_values = np.vstack(
            [
                source_correction(
                    rows, mapped, propensity, all_tables[source_season]
                )
                for source_season in source_seasons
            ]
        )
        equal = np.mean(source_values, axis=0)
        recency_weights = np.power(2.0, np.arange(len(source_seasons)))
        recency = np.average(source_values, axis=0, weights=recency_weights)
        last = source_values[-1]
        strict = np.load(
            LOWRANK_ROOT
            / f"predictions_lowrank_s300_r6_{validation_season}.npy"
        ).astype(float)
        pooled_prediction = np.load(
            POOLED_ROOT
            / f"predictions_fine_direct_w025_{validation_season}.npy"
        ).astype(float)
        pooled_correction = (pooled_prediction - strict) / 0.25
        predictions = {
            "base": base,
            "pooled_w010": np.clip(
                base + CORRECTION_WEIGHT * pooled_correction, 0.0, 1.0
            ),
            "source_equal_w010": np.clip(
                base + CORRECTION_WEIGHT * equal, 0.0, 1.0
            ),
            "source_recency2_w010": np.clip(
                base + CORRECTION_WEIGHT * recency, 0.0, 1.0
            ),
            "source_last_w010": np.clip(
                base + CORRECTION_WEIGHT * last, 0.0, 1.0
            ),
        }
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_seasons": source_seasons,
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(target, prediction)
            np.save(
                ARTIFACT_DIR / f"predictions_{name}_{validation_season}.npy",
                prediction,
            )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            target.astype(np.int8),
        )
        folds[str(validation_season)] = fold
        audits[str(validation_season)] = {
            **mapping_audit,
            "row_mapping_coverage": float(mapped.notna().mean()),
            "propensity_contexts": int(len(propensity)),
            "source_seasons": source_seasons,
            "source_correction_mean_abs": {
                str(season): float(np.mean(np.abs(source_values[position])))
                for position, season in enumerate(source_seasons)
            },
        }
        print(
            f"fold {validation_season}: "
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
        "experiment": "EXP-052",
        "candidate_family": "exact_trackman_control_temporal_source_policies",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "mapping_and_propensity_cutoff": "validation season-1",
            "source_control_tables": "each historical season independently",
            "current_fold_labels_used_for_fit_or_selection": False,
            "actual_current_pitch_type_used": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "model": {
            "pitcher_smoothing": PITCHER_SMOOTHING,
            "type_smoothing": TYPE_SMOOTHING,
            "context_smoothing": CONTEXT_SMOOTHING,
            "correction_clip": CORRECTION_CLIP,
            "correction_weight": CORRECTION_WEIGHT,
        },
        "exact_alignment": alignment_audit,
        "fold_feature_audit": audits,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_fixed_candidate": best,
            "best_min_skill": aggregate[best]["min_skill"],
            "gate_each_season_1000": bool(min(aggregate[best]["season_skills"].values()) >= 1000.0),
            "gate_mean_1050": bool(aggregate[best]["mean_skill"] >= 1050.0),
            "adopt": bool(min(aggregate[best]["season_skills"].values()) >= 1000.0),
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
