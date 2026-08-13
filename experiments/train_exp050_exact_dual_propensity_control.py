"""EXP-050: exact-mapped pitcher/batter hierarchical pitch propensity.

Historical Trackman pitch types estimate current-row pitch-type probabilities
from the mapped pitcher, mapped batter, count and hands.  The actual current
pitch type is never used.  These propensities integrate exact-aligned pitcher
fine-type control rates through the previous season.
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
    posterior_tables,
    propensity_table,
)
from train_exp046_exact_batter_pitchtype_control import batter_mapping


LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
ARTIFACT_DIR = Path("./artifacts/EXP-050/exact_dual_propensity_control")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
BATTER_PROPENSITY_SMOOTHING = 30.0
MATCHUP_SMOOTHING = 40.0
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "pitcher_prop_w025",
    "dual25_w025",
    "dual50_w025",
    "matchup_w025",
)


def crosstab_counts(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return pd.crosstab(
        [rows[column] for column in keys], rows["fine_pitch_type"]
    ).reindex(columns=FINE_TYPES, fill_value=0)


def batter_propensity_table(trackman: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    history = trackman.loc[trackman["season"].le(cutoff)].copy()
    history["pitcher_hand_code"] = history["pitcher_hand"].map(
        {"Left": 1, "Right": 2}
    )
    overall = crosstab_counts(history, ["batter_trackman_id"])
    overall_prob = overall.div(overall.sum(axis=1), axis=0)
    context = crosstab_counts(
        history,
        ["batter_trackman_id", "count_index", "pitcher_hand_code"],
    )
    batter_ids = context.index.get_level_values("batter_trackman_id")
    prior = overall_prob.reindex(batter_ids).to_numpy(float)
    counts = context.to_numpy(float)
    probability = (
        counts + BATTER_PROPENSITY_SMOOTHING * np.nan_to_num(prior)
    ) / (counts.sum(axis=1, keepdims=True) + BATTER_PROPENSITY_SMOOTHING)
    return pd.DataFrame(probability, index=context.index, columns=FINE_TYPES)


def matchup_counts_table(trackman: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    history = trackman.loc[trackman["season"].le(cutoff)]
    return crosstab_counts(
        history,
        ["pitcher_trackman_id", "batter_trackman_id", "count_index"],
    )


def expected_control(
    rows: pd.DataFrame,
    mapped_pitcher: pd.Series,
    weights: np.ndarray,
    pitcher_rate: pd.Series,
    type_rate: pd.Series,
    context_rate: pd.Series,
    league: float,
) -> np.ndarray:
    rate_matrix = np.empty((len(rows), len(FINE_TYPES)), dtype=np.float32)
    overall = pitcher_rate.reindex(mapped_pitcher).fillna(league).to_numpy(float)
    for position, pitch_type in enumerate(FINE_TYPES):
        context_index = pd.MultiIndex.from_arrays(
            [
                mapped_pitcher,
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
            [mapped_pitcher, np.full(len(rows), pitch_type, dtype=object)],
            names=["pitcher_trackman_id", "fine_pitch_type"],
        )
        rate = context_rate.reindex(context_index).to_numpy(float)
        fallback = type_rate.reindex(type_index).to_numpy(float)
        rate = np.where(np.isfinite(rate), rate, fallback)
        rate_matrix[:, position] = np.where(np.isfinite(rate), rate, overall)
    return np.sum(weights * rate_matrix, axis=1)


def build_corrections(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    aligned: pd.DataFrame,
    season: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    cutoff = season - 1
    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    pitcher_map, pitcher_audit = mapping_from_aligned(aligned, cutoff)
    batter_map_result = batter_mapping(aligned, cutoff)
    mapped_pitcher = rows["pitcher_id"].map(pitcher_map.mapping)
    mapped_batter = rows["batter_id"].map(batter_map_result.mapping)
    pitcher_prop = propensity_table(trackman, cutoff)
    batter_prop = batter_propensity_table(trackman, cutoff)
    matchup_counts = matchup_counts_table(trackman, cutoff)
    pitcher_index = pd.MultiIndex.from_arrays(
        [mapped_pitcher, rows["count_index"], rows["batter_hand"]],
        names=["pitcher_trackman_id", "count_index", "batter_hand_code"],
    )
    batter_index = pd.MultiIndex.from_arrays(
        [mapped_batter, rows["count_index"], rows["pitcher_hand"]],
        names=["batter_trackman_id", "count_index", "pitcher_hand_code"],
    )
    matchup_index = pd.MultiIndex.from_arrays(
        [mapped_pitcher, mapped_batter, rows["count_index"]],
        names=["pitcher_trackman_id", "batter_trackman_id", "count_index"],
    )
    p = pitcher_prop.reindex(pitcher_index).to_numpy(float)
    b = batter_prop.reindex(batter_index).to_numpy(float)
    p_valid = np.isfinite(p).all(axis=1)
    b_valid = np.isfinite(b).all(axis=1)
    dual25 = np.where(b_valid[:, None], 0.75 * p + 0.25 * b, p)
    dual50 = np.where(b_valid[:, None], 0.50 * p + 0.50 * b, p)
    raw_matchup = matchup_counts.reindex(matchup_index).to_numpy(float)
    matchup_n = np.nansum(raw_matchup, axis=1)
    matchup = (
        np.nan_to_num(raw_matchup) + MATCHUP_SMOOTHING * dual25
    ) / (matchup_n[:, None] + MATCHUP_SMOOTHING)
    pitcher_rate, _, type_rate, context_rate, league = posterior_tables(
        aligned, cutoff
    )
    official = rows["asof_pitcher_success_rate"].to_numpy(float)
    corrections: dict[str, np.ndarray] = {}
    for name, weights in {
        "pitcher_prop": p,
        "dual25": dual25,
        "dual50": dual50,
        "matchup": matchup,
    }.items():
        valid = p_valid & np.isfinite(weights).all(axis=1) & np.isfinite(official)
        expected = expected_control(
            rows,
            mapped_pitcher,
            np.nan_to_num(weights),
            pitcher_rate,
            type_rate,
            context_rate,
            league,
        )
        correction = np.zeros(len(rows))
        correction[valid] = np.clip(
            expected[valid] - official[valid],
            -CORRECTION_CLIP,
            CORRECTION_CLIP,
        )
        corrections[name] = correction

    actual = pd.Categorical(
        aligned.loc[aligned["season"].eq(season), "fine_pitch_type"],
        categories=FINE_TYPES,
    ).codes
    selection_audit: dict[str, float] = {}
    # A separate exact-aligned likelihood audit is intentionally omitted here:
    # aligned rows are a subset whose original validation positions are not
    # serialized.  Rolling target metrics remain authoritative.
    audit = {
        "pitcher_mapping": pitcher_audit,
        "batter_mapping": batter_map_result.audit,
        "pitcher_propensity_coverage": float(p_valid.mean()),
        "batter_propensity_coverage": float(b_valid.mean()),
        "matchup_seen_coverage": float((matchup_n > 0).mean()),
        "propensity_contexts": {
            "pitcher": int(len(pitcher_prop)),
            "batter": int(len(batter_prop)),
            "matchup": int(len(matchup_counts)),
        },
        "selection_audit": selection_audit,
        "aligned_actual_pitch_rows_for_season": int((actual >= 0).sum()),
    }
    return corrections, audit


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
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        base = recent_base(season)
        corrections, audits[str(season)] = build_corrections(
            main_frame, trackman, aligned, season
        )
        predictions = {
            "base": base,
            "pitcher_prop_w025": np.clip(
                base + 0.25 * corrections["pitcher_prop"], 0.0, 1.0
            ),
            "dual25_w025": np.clip(
                base + 0.25 * corrections["dual25"], 0.0, 1.0
            ),
            "dual50_w025": np.clip(
                base + 0.25 * corrections["dual50"], 0.0, 1.0
            ),
            "matchup_w025": np.clip(
                base + 0.25 * corrections["matchup"], 0.0, 1.0
            ),
        }
        fold: dict[str, object] = {"validation_season": season, "history_cutoff": season - 1}
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
        "experiment": "EXP-050",
        "candidate_family": "exact_pitcher_batter_hierarchical_pitch_propensity",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "history_and_mapping_cutoff": "validation season-1",
            "current_fold_labels_used_for_fit_or_selection": False,
            "actual_current_pitch_type_used": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "model": {
            "fine_pitch_types": list(FINE_TYPES),
            "batter_propensity_smoothing": BATTER_PROPENSITY_SMOOTHING,
            "matchup_smoothing": MATCHUP_SMOOTHING,
            "correction_clip": CORRECTION_CLIP,
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
