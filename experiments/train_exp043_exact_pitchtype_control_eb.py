"""EXP-043: exact-aligned pitcher x fine-pitch-type control EB.

Exact full-game sequence matches from EXP-041 attach historical Trackman pitch
types to official ``control_success`` labels.  For every outer fold, only
earlier seasons fit a hierarchy of pitcher, pitcher x fine type, and pitcher x
fine type x count x batter-hand success rates.  Prior Trackman usage supplies
the current-row pitch-type propensity, so the hierarchy is integrated without
using the unknown current pitch type.  Test rows are never aggregated.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from train_exp017_rolling_residual import calculate_metrics
from train_exp033_trackman_sequence_trend import FINE_TYPES, canonical_pitch_type
from train_exp041_exact_game_trackman_sequence import (
    exact_aligned_rows,
    mapping_from_aligned,
)


DATA_DIR = Path("./data")
BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-043/exact_pitchtype_control_eb")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
PITCHER_SMOOTHING = 500.0
TYPE_SMOOTHING = 200.0
CONTEXT_SMOOTHING = 100.0
PROPENSITY_SMOOTHING = 20.0
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "fine_lgb_w025",
    "fine_lgb_w050",
    "fine_direct_w025",
    "fine_blend_w025",
)


def load_main() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "control_success",
    ]
    frame = pd.read_csv(
        DATA_DIR / "train.csv", encoding="utf-8-sig", usecols=columns
    )
    frame["count_index"] = (
        4 * frame["balls_before"] + frame["strikes_before"]
    ).astype(np.int8)
    return frame


def load_trackman() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_trackman_id",
        "batter_trackman_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "pitcher_hand",
        "tagged_pitch_type",
    ]
    frame = pd.read_csv(
        DATA_DIR / "trackman_history.csv",
        encoding="utf-8-sig",
        usecols=columns,
    )
    frame["count_index"] = (
        4 * frame["balls_before"] + frame["strikes_before"]
    ).astype(np.int8)
    frame["batter_hand_code"] = frame["batter_hand"].map(
        {"Left": 1, "Right": 2}
    )
    frame["fine_pitch_type"] = canonical_pitch_type(
        frame["tagged_pitch_type"]
    )
    return frame


def posterior_tables(
    aligned: pd.DataFrame, cutoff: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, float]:
    history = aligned.loc[aligned["season"].le(cutoff)].copy()
    y = history["control_success"].astype(float)
    league = float(y.mean())

    pitcher_stats = history.groupby("pitcher_trackman_id")[
        "control_success"
    ].agg(["sum", "count"])
    pitcher_rate = (
        pitcher_stats["sum"] + PITCHER_SMOOTHING * league
    ) / (pitcher_stats["count"] + PITCHER_SMOOTHING)

    type_keys = ["pitcher_trackman_id", "fine_pitch_type"]
    type_stats = history.groupby(type_keys)["control_success"].agg(
        ["sum", "count"]
    )
    type_pitcher = type_stats.index.get_level_values("pitcher_trackman_id")
    type_prior = pitcher_rate.reindex(type_pitcher).fillna(league).to_numpy()
    type_rate = pd.Series(
        (
            type_stats["sum"].to_numpy(dtype=float)
            + TYPE_SMOOTHING * type_prior
        )
        / (type_stats["count"].to_numpy(dtype=float) + TYPE_SMOOTHING),
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
    context_prior = type_rate.reindex(context_type_index).to_numpy(dtype=float)
    context_rate = pd.Series(
        (
            context_stats["sum"].to_numpy(dtype=float)
            + CONTEXT_SMOOTHING * context_prior
        )
        / (context_stats["count"].to_numpy(dtype=float) + CONTEXT_SMOOTHING),
        index=context_stats.index,
    )
    return pitcher_rate, pitcher_stats["count"], type_rate, context_rate, league


def propensity_table(trackman: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    history = trackman.loc[trackman["season"].le(cutoff)]
    pitcher_counts = pd.crosstab(
        history["pitcher_trackman_id"], history["fine_pitch_type"]
    ).reindex(columns=FINE_TYPES, fill_value=0)
    pitcher_mix = pitcher_counts.div(
        pitcher_counts.sum(axis=1).replace(0, np.nan), axis=0
    )
    context_index = [
        history["pitcher_trackman_id"],
        history["count_index"],
        history["batter_hand_code"],
    ]
    context_counts = pd.crosstab(
        context_index, history["fine_pitch_type"]
    ).reindex(columns=FINE_TYPES, fill_value=0)
    context_pitcher = context_counts.index.get_level_values(
        "pitcher_trackman_id"
    )
    prior = pitcher_mix.reindex(context_pitcher).to_numpy(dtype=float)
    counts = context_counts.to_numpy(dtype=float)
    probabilities = (
        counts + PROPENSITY_SMOOTHING * np.nan_to_num(prior, nan=0.0)
    ) / (counts.sum(axis=1, keepdims=True) + PROPENSITY_SMOOTHING)
    return pd.DataFrame(
        probabilities, index=context_counts.index, columns=FINE_TYPES
    )


def build_features(
    main: pd.DataFrame,
    aligned: pd.DataFrame,
    trackman: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cutoff = season - 1
    mapping, map_audit = mapping_from_aligned(aligned, cutoff)
    pitcher_rate, pitcher_n, type_rate, context_rate, league = posterior_tables(
        aligned, cutoff
    )
    propensity = propensity_table(trackman, cutoff)
    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    mapped = rows["pitcher_id"].map(mapping.mapping)
    query_index = pd.MultiIndex.from_arrays(
        [mapped, rows["count_index"], rows["batter_hand"]],
        names=["pitcher_trackman_id", "count_index", "batter_hand_code"],
    )
    prop = propensity.reindex(query_index).to_numpy(dtype=float)

    expected = np.full(len(rows), np.nan, dtype=float)
    dispersion = np.full(len(rows), np.nan, dtype=float)
    for type_position, pitch_type in enumerate(FINE_TYPES):
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
        rate = context_rate.reindex(context_index).to_numpy(dtype=float)
        fallback = type_rate.reindex(type_index).to_numpy(dtype=float)
        overall = pitcher_rate.reindex(mapped).fillna(league).to_numpy(dtype=float)
        rate = np.where(np.isfinite(rate), rate, fallback)
        rate = np.where(np.isfinite(rate), rate, overall)
        contribution = prop[:, type_position] * rate
        expected = np.nansum(
            np.column_stack([np.nan_to_num(expected, nan=0.0), contribution]),
            axis=1,
        )
        if type_position == 0:
            rate_matrix = rate[:, None]
        else:
            rate_matrix = np.column_stack([rate_matrix, rate])
    prop_valid = np.isfinite(prop).all(axis=1)
    expected[~prop_valid] = np.nan
    overall = pitcher_rate.reindex(mapped).to_numpy(dtype=float)
    weights = np.nan_to_num(prop, nan=0.0)
    centered = rate_matrix - expected[:, None]
    dispersion[prop_valid] = np.sqrt(
        np.sum(weights[prop_valid] * np.square(centered[prop_valid]), axis=1)
    )
    entropy = -np.sum(
        weights * np.log(np.clip(weights, 1e-12, 1.0)), axis=1
    )
    mapped_mask = mapped.notna().to_numpy() & prop_valid & np.isfinite(expected)
    features = pd.DataFrame(
        {
            "trackman_mapped": mapped_mask.astype(np.int8),
            "expected_fine_control": expected,
            "pitcher_aligned_control": overall,
            "pitcher_aligned_log_n": np.log1p(
                pitcher_n.reindex(mapped).to_numpy(dtype=float)
            ),
            "fine_control_dispersion": dispersion,
            "fine_selection_entropy": entropy,
            "expected_minus_official": (
                expected - rows["asof_pitcher_success_rate"].to_numpy(dtype=float)
            ),
            "official_success_rate": rows[
                "asof_pitcher_success_rate"
            ].to_numpy(dtype=float),
            "official_log_n": np.log1p(
                rows["asof_pitcher_n"].to_numpy(dtype=float)
            ),
            "count_index": rows["count_index"].to_numpy(dtype=float),
            "batter_hand": rows["batter_hand"].to_numpy(dtype=float),
        }
    )
    audit = {
        **map_audit,
        "row_mapping_coverage": float(mapped_mask.mean()),
        "aligned_label_rows": int(aligned["season"].le(cutoff).sum()),
        "propensity_contexts": int(len(propensity)),
        "pitch_types": list(FINE_TYPES),
    }
    return features, audit


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
        reg_lambda=12.0,
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def main() -> None:
    started = time.time()
    main = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = exact_aligned_rows()
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    features: dict[int, pd.DataFrame] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(BASE_ROOT / f"targets_{season}.npy").astype(float)
        base[season] = np.load(
            BASE_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
        ).astype(float)
        rows = main.loc[main["season"].eq(season)]
        if not np.array_equal(
            rows["control_success"].to_numpy(dtype=float), targets[season]
        ):
            raise ValueError(f"target/order mismatch {season}")
        features[season], audits[str(season)] = build_features(
            main, aligned, trackman, season
        )
        print(
            f"features {season}: coverage={audits[str(season)]['row_mapping_coverage']:.3f}",
            flush=True,
        )

    feature_names = [column for column in features[2021] if column != "trackman_mapped"]
    residuals = {
        season: targets[season] - base[season] for season in EVALUATED_SEASONS
    }
    for season in residuals:
        residuals[season] -= residuals[season].mean()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        source_seasons = [s for s in EVALUATED_SEASONS if s < validation_season]
        lgb_correction = np.zeros(len(targets[validation_season]), dtype=float)
        if source_seasons:
            train_x = pd.concat(
                [features[s] for s in source_seasons], ignore_index=True
            )
            train_y = np.concatenate([residuals[s] for s in source_seasons])
            source = np.concatenate(
                [np.full(len(features[s]), s) for s in source_seasons]
            )
            eligible = train_x["trackman_mapped"].eq(1).to_numpy()
            counts = pd.Series(source[eligible]).value_counts()
            weight = np.array([1.0 / counts[value] for value in source[eligible]])
            weight *= len(weight) / weight.sum()
            model = new_model()
            model.fit(
                train_x.loc[eligible, feature_names],
                train_y[eligible],
                sample_weight=weight,
            )
            validation_eligible = features[validation_season][
                "trackman_mapped"
            ].eq(1).to_numpy()
            lgb_correction[validation_eligible] = model.predict(
                features[validation_season].loc[validation_eligible, feature_names]
            )
        lgb_correction = np.clip(
            lgb_correction, -CORRECTION_CLIP, CORRECTION_CLIP
        )
        direct_correction = np.zeros(len(lgb_correction), dtype=float)
        mapped = features[validation_season]["trackman_mapped"].eq(1).to_numpy()
        direct_correction[mapped] = features[validation_season].loc[
            mapped, "expected_minus_official"
        ].to_numpy(dtype=float)
        direct_correction = np.clip(
            direct_correction, -CORRECTION_CLIP, CORRECTION_CLIP
        )
        predictions = {
            "base": base[validation_season],
            "fine_lgb_w025": np.clip(
                base[validation_season] + 0.25 * lgb_correction, 0.0, 1.0
            ),
            "fine_lgb_w050": np.clip(
                base[validation_season] + 0.50 * lgb_correction, 0.0, 1.0
            ),
            "fine_direct_w025": np.clip(
                base[validation_season] + 0.25 * direct_correction, 0.0, 1.0
            ),
            "fine_blend_w025": np.clip(
                base[validation_season]
                + 0.125 * lgb_correction
                + 0.125 * direct_correction,
                0.0,
                1.0,
            ),
        }
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(targets[validation_season], prediction)
            np.save(
                ARTIFACT_DIR / f"predictions_{name}_{validation_season}.npy",
                prediction,
            )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets[validation_season].astype(np.int8),
        )
        folds[str(validation_season)] = fold
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
        key=lambda name: (
            aggregate[name]["min_skill"],
            aggregate[name]["mean_skill"],
        ),
    )
    result = {
        "experiment": "EXP-043",
        "candidate_family": "exact_aligned_fine_pitchtype_control_eb",
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
            "pitcher_smoothing": PITCHER_SMOOTHING,
            "type_smoothing": TYPE_SMOOTHING,
            "context_smoothing": CONTEXT_SMOOTHING,
            "propensity_smoothing": PROPENSITY_SMOOTHING,
            "correction_clip": CORRECTION_CLIP,
            "residual_feature_names": feature_names,
        },
        "exact_alignment": alignment_audit,
        "fold_feature_audit": audits,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_fixed_candidate": best,
            "best_min_skill": aggregate[best]["min_skill"],
            "gate_each_season_1000": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
            ),
            "gate_mean_1050": bool(aggregate[best]["mean_skill"] >= 1050.0),
            "adopt_for_full_fit": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
                and aggregate[best]["mean_skill"] >= 1050.0
            ),
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
        f"min={aggregate[best]['min_skill']:.2f} "
        f"adopt={result['selection']['adopt_for_full_fit']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
