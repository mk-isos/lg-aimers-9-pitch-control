"""EXP-046: exact-aligned batter x fine-pitch-type control history.

The exact full-game alignment supplies a high-purity official/Trackman batter
map.  Earlier aligned labels estimate a hierarchical batter response by fine
pitch type, count, and pitcher hand.  Historical pitcher pitch-type propensity
integrates the unknown current pitch type without using evaluation-row peers.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from train_exp017_rolling_residual import calculate_metrics
from train_exp041_exact_game_trackman_sequence import (
    MIN_ALIGNED_ROWS,
    MIN_MAPPING_PURITY,
    exact_aligned_rows,
    mapping_from_aligned,
)
from train_exp043_exact_pitchtype_control_eb import (
    FINE_TYPES,
    load_main,
    load_trackman,
    propensity_table,
)


DATA_DIR = Path("./data")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
ARTIFACT_DIR = Path("./artifacts/EXP-046/exact_batter_pitchtype_control")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
BATTER_SMOOTHING = 700.0
TYPE_SMOOTHING = 300.0
CONTEXT_SMOOTHING = 150.0
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "batter_direct_w010",
    "batter_direct_w025",
    "batter_lgb_w025",
    "batter_blend_w025",
)


@dataclass(frozen=True)
class ExactMapping:
    mapping: dict[int, int]
    audit: dict[str, object]


def batter_mapping(aligned: pd.DataFrame, cutoff: int) -> ExactMapping:
    source = aligned.loc[aligned["season"].le(cutoff)]
    counts = source.groupby(["batter_id", "batter_trackman_id"]).size()
    totals = counts.groupby(level=0).sum()
    maxima = counts.groupby(level=0).max()
    best = counts.groupby(level=0).idxmax()
    purity = maxima / totals
    accepted = purity.index[
        purity.ge(MIN_MAPPING_PURITY) & totals.ge(MIN_ALIGNED_ROWS)
    ]
    mapping = {
        int(player_id): int(best.loc[player_id][1]) for player_id in accepted
    }
    return ExactMapping(
        mapping=mapping,
        audit={
            "cutoff_season": int(cutoff),
            "source_aligned_rows": int(len(source)),
            "observed_official_batters": int(source["batter_id"].nunique()),
            "accepted_batters": int(len(mapping)),
            "minimum_rows": MIN_ALIGNED_ROWS,
            "minimum_purity": MIN_MAPPING_PURITY,
            "weighted_majority_purity": float(maxima.sum() / totals.sum()),
        },
    )


def batter_tables(
    aligned: pd.DataFrame, cutoff: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, float]:
    history = aligned.loc[aligned["season"].le(cutoff)].copy()
    league = float(history["control_success"].mean())
    batter_stats = history.groupby("batter_trackman_id")["control_success"].agg(
        ["sum", "count"]
    )
    batter_rate = (
        batter_stats["sum"] + BATTER_SMOOTHING * league
    ) / (batter_stats["count"] + BATTER_SMOOTHING)
    type_keys = ["batter_trackman_id", "fine_pitch_type"]
    type_stats = history.groupby(type_keys)["control_success"].agg(["sum", "count"])
    type_batter = type_stats.index.get_level_values("batter_trackman_id")
    type_prior = batter_rate.reindex(type_batter).fillna(league).to_numpy()
    type_rate = pd.Series(
        (type_stats["sum"].to_numpy(float) + TYPE_SMOOTHING * type_prior)
        / (type_stats["count"].to_numpy(float) + TYPE_SMOOTHING),
        index=type_stats.index,
    )
    context_keys = [
        "batter_trackman_id",
        "fine_pitch_type",
        "count_index",
        "pitcher_hand",
    ]
    context_stats = history.groupby(context_keys)["control_success"].agg(
        ["sum", "count"]
    )
    context_type = pd.MultiIndex.from_arrays(
        [
            context_stats.index.get_level_values("batter_trackman_id"),
            context_stats.index.get_level_values("fine_pitch_type"),
        ],
        names=type_keys,
    )
    prior = type_rate.reindex(context_type).to_numpy(float)
    context_rate = pd.Series(
        (context_stats["sum"].to_numpy(float) + CONTEXT_SMOOTHING * prior)
        / (context_stats["count"].to_numpy(float) + CONTEXT_SMOOTHING),
        index=context_stats.index,
    )
    return batter_rate, batter_stats["count"], type_rate, context_rate, league


def build_features(
    main: pd.DataFrame,
    aligned: pd.DataFrame,
    trackman: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cutoff = season - 1
    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    pitcher_map, pitcher_audit = mapping_from_aligned(aligned, cutoff)
    batter_map = batter_mapping(aligned, cutoff)
    batter_rate, batter_n, type_rate, context_rate, league = batter_tables(
        aligned, cutoff
    )
    propensity = propensity_table(trackman, cutoff)
    mapped_pitcher = rows["pitcher_id"].map(pitcher_map.mapping)
    mapped_batter = rows["batter_id"].map(batter_map.mapping)
    propensity_index = pd.MultiIndex.from_arrays(
        [mapped_pitcher, rows["count_index"], rows["batter_hand"]],
        names=["pitcher_trackman_id", "count_index", "batter_hand_code"],
    )
    weights = propensity.reindex(propensity_index).to_numpy(float)
    rate_matrix = np.empty((len(rows), len(FINE_TYPES)), dtype=np.float32)
    for position, pitch_type in enumerate(FINE_TYPES):
        context_index = pd.MultiIndex.from_arrays(
            [
                mapped_batter,
                np.full(len(rows), pitch_type, dtype=object),
                rows["count_index"],
                rows["pitcher_hand"],
            ],
            names=[
                "batter_trackman_id",
                "fine_pitch_type",
                "count_index",
                "pitcher_hand",
            ],
        )
        type_index = pd.MultiIndex.from_arrays(
            [mapped_batter, np.full(len(rows), pitch_type, dtype=object)],
            names=["batter_trackman_id", "fine_pitch_type"],
        )
        rate = context_rate.reindex(context_index).to_numpy(float)
        fallback = type_rate.reindex(type_index).to_numpy(float)
        overall = batter_rate.reindex(mapped_batter).fillna(league).to_numpy(float)
        rate = np.where(np.isfinite(rate), rate, fallback)
        rate_matrix[:, position] = np.where(np.isfinite(rate), rate, overall)
    valid = (
        mapped_pitcher.notna().to_numpy()
        & mapped_batter.notna().to_numpy()
        & np.isfinite(weights).all(axis=1)
    )
    expected = np.sum(np.nan_to_num(weights) * rate_matrix, axis=1)
    expected[~valid] = np.nan
    official = rows["asof_batter_success_rate"].to_numpy(float)
    direct = expected - official
    features = pd.DataFrame(
        {
            "trackman_mapped": valid.astype(np.int8),
            "expected_batter_fine_control": expected,
            "expected_minus_official_batter": direct,
            "batter_aligned_control": batter_rate.reindex(mapped_batter).to_numpy(float),
            "batter_aligned_log_n": np.log1p(batter_n.reindex(mapped_batter).to_numpy(float)),
            "fine_control_dispersion": np.sqrt(
                np.sum(np.nan_to_num(weights) * np.square(rate_matrix - expected[:, None]), axis=1)
            ),
            "fine_selection_entropy": -np.sum(
                np.nan_to_num(weights) * np.log(np.clip(np.nan_to_num(weights), 1e-12, 1.0)), axis=1
            ),
            "official_batter_success": official,
            "official_batter_log_n": np.log1p(rows["asof_batter_n"].to_numpy(float)),
            "official_pitcher_success": rows["asof_pitcher_success_rate"].to_numpy(float),
            "official_pitcher_log_n": np.log1p(rows["asof_pitcher_n"].to_numpy(float)),
            "count_index": rows["count_index"].to_numpy(float),
            "pitcher_hand": rows["pitcher_hand"].to_numpy(float),
            "batter_hand": rows["batter_hand"].to_numpy(float),
        }
    )
    return features, {
        "pitcher_mapping": pitcher_audit,
        "batter_mapping": batter_map.audit,
        "row_joint_mapping_coverage": float(valid.mean()),
        "aligned_label_rows": int(aligned["season"].le(cutoff).sum()),
        "propensity_contexts": int(len(propensity)),
    }


def recent_base(season: int) -> np.ndarray:
    recency = np.load(RECENCY_ROOT / f"predictions_recency2_{season}.npy").astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    return 0.5 * recency + 0.5 * aggressive


def new_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=160,
        learning_rate=0.015,
        num_leaves=7,
        min_child_samples=3000,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=15.0,
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def main() -> None:
    started = time.time()
    main_frame = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = exact_aligned_rows()
    features: dict[int, pd.DataFrame] = {}
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        base[season] = recent_base(season)
        features[season], audits[str(season)] = build_features(
            main_frame, aligned, trackman, season
        )
        rows = main_frame.loc[main_frame["season"].eq(season)]
        if not np.array_equal(rows["control_success"].to_numpy(float), targets[season]):
            raise ValueError(f"target/order mismatch {season}")
        print(
            f"features {season}: coverage={audits[str(season)]['row_joint_mapping_coverage']:.3f}",
            flush=True,
        )
    names = [column for column in features[2021] if column != "trackman_mapped"]
    residual = {season: targets[season] - base[season] for season in EVALUATED_SEASONS}
    for season in residual:
        residual[season] -= residual[season].mean()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        sources = [season for season in EVALUATED_SEASONS if season < validation_season]
        lgb_correction = np.zeros(len(targets[validation_season]))
        if sources:
            train_x = pd.concat([features[s] for s in sources], ignore_index=True)
            train_y = np.concatenate([residual[s] for s in sources])
            source = np.concatenate([np.full(len(features[s]), s) for s in sources])
            eligible = train_x["trackman_mapped"].eq(1).to_numpy()
            counts = pd.Series(source[eligible]).value_counts()
            weight = np.array([1.0 / counts[value] for value in source[eligible]])
            weight *= len(weight) / weight.sum()
            model = new_model()
            model.fit(train_x.loc[eligible, names], train_y[eligible], sample_weight=weight)
            valid = features[validation_season]["trackman_mapped"].eq(1).to_numpy()
            lgb_correction[valid] = model.predict(
                features[validation_season].loc[valid, names]
            )
        lgb_correction = np.clip(lgb_correction, -CORRECTION_CLIP, CORRECTION_CLIP)
        direct = np.zeros(len(lgb_correction))
        mapped = features[validation_season]["trackman_mapped"].eq(1).to_numpy()
        values = features[validation_season]["expected_minus_official_batter"].to_numpy(float)
        direct[mapped] = values[mapped]
        direct = np.clip(direct, -CORRECTION_CLIP, CORRECTION_CLIP)
        predictions = {
            "base": base[validation_season],
            "batter_direct_w010": np.clip(base[validation_season] + 0.10 * direct, 0.0, 1.0),
            "batter_direct_w025": np.clip(base[validation_season] + 0.25 * direct, 0.0, 1.0),
            "batter_lgb_w025": np.clip(base[validation_season] + 0.25 * lgb_correction, 0.0, 1.0),
            "batter_blend_w025": np.clip(base[validation_season] + 0.125 * direct + 0.125 * lgb_correction, 0.0, 1.0),
        }
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_oof_seasons": sources,
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(targets[validation_season], prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{validation_season}.npy", prediction)
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets[validation_season].astype(np.int8))
        folds[str(validation_season)] = fold
        print(
            f"fold {validation_season}: "
            + " ".join(f"{name}={fold[name]['skill_score_unclipped']:.2f}" for name in CANDIDATES),
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
    best = max(CANDIDATES, key=lambda name: (aggregate[name]["min_skill"], aggregate[name]["mean_skill"]))
    result = {
        "experiment": "EXP-046",
        "candidate_family": "exact_aligned_batter_fine_pitchtype_control",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "history_and_mapping_cutoff": "validation season-1",
            "source_residuals": "earlier OOF seasons, centered and season-equal",
            "current_fold_labels_used_for_fit_or_selection": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "model": {
            "fine_pitch_types": list(FINE_TYPES),
            "batter_smoothing": BATTER_SMOOTHING,
            "type_smoothing": TYPE_SMOOTHING,
            "context_smoothing": CONTEXT_SMOOTHING,
            "correction_clip": CORRECTION_CLIP,
            "feature_names": names,
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
            "lightgbm": lgb.__version__,
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
