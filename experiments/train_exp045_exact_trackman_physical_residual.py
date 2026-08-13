"""EXP-045: exact-mapped Trackman physical-profile residual.

This bounded experiment tests whether exact game-sequence pitcher mapping makes
historical Trackman physical measurements temporally useful.  Every validation
row uses only a frozen pitcher map and Trackman summaries through the previous
season.  No validation/test-row aggregation is performed.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_exp017_rolling_residual import calculate_metrics
from train_exp033_trackman_sequence_trend import FINE_TYPES, canonical_pitch_type
from train_exp041_exact_game_trackman_sequence import (
    exact_aligned_rows,
    mapping_from_aligned,
)
from train_exp043_exact_pitchtype_control_eb import load_main, propensity_table


DATA_DIR = Path("./data")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
ARTIFACT_DIR = Path("./artifacts/EXP-045/exact_trackman_physical_residual")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
PHYSICAL = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "ridge_a1000_w025",
    "ridge_a10000_w025",
    "lgb_l7_w025",
    "lgb_l7_w050",
)


def load_trackman() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_trackman_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "tagged_pitch_type",
        *PHYSICAL,
    ]
    frame = pd.read_csv(
        DATA_DIR / "trackman_history.csv", encoding="utf-8-sig", usecols=columns
    )
    frame["count_index"] = (
        4 * frame["balls_before"] + frame["strikes_before"]
    ).astype(np.int8)
    frame["batter_hand_code"] = frame["batter_hand"].map(
        {"Left": 1, "Right": 2}
    )
    frame["fine_pitch_type"] = canonical_pitch_type(frame["tagged_pitch_type"])
    frame["velo_loss"] = frame["rel_speed"] - frame["zone_speed"]
    return frame


def grouped_physical(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [*PHYSICAL, "velo_loss"]
    grouped = history.groupby(
        ["pitcher_trackman_id", "fine_pitch_type"], sort=True
    )[metrics].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    # All physical columns share approximately the same availability, but use
    # rel_speed count as the reliability measure to avoid duplicate count fields.
    count_columns = [name for name in grouped if name.endswith("_count")]
    keep = [name for name in grouped if not name.endswith("_count")]
    grouped = grouped[keep].copy()
    grouped["physical_n"] = history.groupby(
        ["pitcher_trackman_id", "fine_pitch_type"], sort=True
    ).size()
    overall = history.groupby("pitcher_trackman_id", sort=True)[metrics].agg(
        ["mean", "std"]
    )
    overall.columns = [f"overall_{metric}_{stat}" for metric, stat in overall.columns]
    overall["overall_physical_n"] = history.groupby(
        "pitcher_trackman_id", sort=True
    ).size()
    if not count_columns:
        raise ValueError("physical summary count columns missing")
    return grouped, overall


def expected_summary(
    rows: pd.DataFrame,
    mapped: pd.Series,
    propensity: pd.DataFrame,
    by_type: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    query = pd.MultiIndex.from_arrays(
        [mapped, rows["count_index"], rows["batter_hand"]],
        names=["pitcher_trackman_id", "count_index", "batter_hand_code"],
    )
    weights = propensity.reindex(query).to_numpy(dtype=float)
    valid = np.isfinite(weights).all(axis=1)
    summary_columns = [name for name in by_type if name != "physical_n"]
    output: dict[str, np.ndarray] = {}
    for column in summary_columns:
        matrix = np.empty((len(rows), len(FINE_TYPES)), dtype=np.float32)
        for position, pitch_type in enumerate(FINE_TYPES):
            index = pd.MultiIndex.from_arrays(
                [mapped, np.full(len(rows), pitch_type, dtype=object)],
                names=["pitcher_trackman_id", "fine_pitch_type"],
            )
            matrix[:, position] = by_type[column].reindex(index).to_numpy(
                dtype=np.float32
            )
        present = np.isfinite(matrix)
        effective = np.where(present, weights, 0.0)
        denominator = effective.sum(axis=1)
        values = np.divide(
            np.nansum(effective * matrix, axis=1),
            denominator,
            out=np.full(len(rows), np.nan),
            where=denominator > 0,
        )
        values[~valid] = np.nan
        output[f"{prefix}_{column}"] = values
    output[f"{prefix}_mix_entropy"] = np.where(
        valid,
        -np.sum(np.nan_to_num(weights) * np.log(np.clip(np.nan_to_num(weights), 1e-12, 1.0)), axis=1),
        np.nan,
    )
    return pd.DataFrame(output)


def build_features(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    aligned: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cutoff = season - 1
    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    mapping, map_audit = mapping_from_aligned(aligned, cutoff)
    mapped = rows["pitcher_id"].map(mapping.mapping)
    all_history = trackman.loc[trackman["season"].le(cutoff)]
    last_history = trackman.loc[trackman["season"].eq(cutoff)]
    prior_history = trackman.loc[trackman["season"].lt(cutoff)]
    all_type, all_overall = grouped_physical(all_history)
    last_type, _ = grouped_physical(last_history)
    prior_type, _ = grouped_physical(prior_history)
    propensity = propensity_table(trackman, cutoff)
    all_expected = expected_summary(
        rows, mapped, propensity, all_type, "tm_all"
    )
    last_expected = expected_summary(
        rows, mapped, propensity, last_type, "tm_last"
    )
    prior_expected = expected_summary(
        rows, mapped, propensity, prior_type, "tm_prior"
    )
    features = pd.concat([all_expected, last_expected, prior_expected], axis=1)
    for column in all_expected:
        if column.endswith("mix_entropy"):
            continue
        suffix = column.removeprefix("tm_all_")
        features[f"tm_delta_{suffix}"] = (
            last_expected[f"tm_last_{suffix}"]
            - prior_expected[f"tm_prior_{suffix}"]
        )
    overall_values = all_overall.reindex(mapped).reset_index(drop=True)
    overall_values.columns = [f"tm_{name}" for name in overall_values.columns]
    features = pd.concat([features, overall_values], axis=1)
    features["trackman_mapped"] = mapped.notna().astype(np.int8)
    features["official_success_rate"] = rows[
        "asof_pitcher_success_rate"
    ].to_numpy(dtype=float)
    features["official_log_n"] = np.log1p(
        rows["asof_pitcher_n"].to_numpy(dtype=float)
    )
    features["count_index"] = rows["count_index"].to_numpy(dtype=float)
    features["batter_hand"] = rows["batter_hand"].to_numpy(dtype=float)
    features["balls_before"] = rows["balls_before"].to_numpy(dtype=float)
    features["strikes_before"] = rows["strikes_before"].to_numpy(dtype=float)
    audit = {
        **map_audit,
        "row_mapping_coverage": float(mapped.notna().mean()),
        "feature_count": int(features.shape[1] - 1),
        "trackman_history_rows": int(len(all_history)),
    }
    return features, audit


def recent_base(season: int) -> np.ndarray:
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    return 0.5 * recency + 0.5 * aggressive


def new_lgb() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=200,
        learning_rate=0.015,
        num_leaves=7,
        min_child_samples=3000,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
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
        features[season], audits[str(season)] = build_features(
            main_frame, trackman, aligned, season
        )
        targets[season] = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        base[season] = recent_base(season)
        expected_target = main_frame.loc[
            main_frame["season"].eq(season), "control_success"
        ].to_numpy(dtype=float)
        if not np.array_equal(targets[season], expected_target):
            raise ValueError(f"target/order mismatch {season}")
        print(
            f"features {season}: coverage={audits[str(season)]['row_mapping_coverage']:.3f}",
            flush=True,
        )

    names = [name for name in features[2021] if name != "trackman_mapped"]
    residual = {season: targets[season] - base[season] for season in EVALUATED_SEASONS}
    for season in residual:
        residual[season] -= residual[season].mean()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        sources = [season for season in EVALUATED_SEASONS if season < validation_season]
        correction = {
            "ridge_a1000": np.zeros(len(targets[validation_season])),
            "ridge_a10000": np.zeros(len(targets[validation_season])),
            "lgb_l7": np.zeros(len(targets[validation_season])),
        }
        if sources:
            train_x = pd.concat([features[s][names] for s in sources], ignore_index=True)
            train_y = np.concatenate([residual[s] for s in sources])
            source = np.concatenate([np.full(len(features[s]), s) for s in sources])
            eligible = pd.concat(
                [features[s]["trackman_mapped"] for s in sources], ignore_index=True
            ).eq(1).to_numpy()
            valid = features[validation_season]["trackman_mapped"].eq(1).to_numpy()
            counts = pd.Series(source[eligible]).value_counts()
            sample_weight = np.array([1.0 / counts[value] for value in source[eligible]])
            sample_weight *= len(sample_weight) / sample_weight.sum()
            for alpha in (1000.0, 10000.0):
                model = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                    Ridge(alpha=alpha),
                )
                model.fit(train_x.loc[eligible], train_y[eligible], ridge__sample_weight=sample_weight)
                correction[f"ridge_a{int(alpha)}"][valid] = model.predict(
                    features[validation_season].loc[valid, names]
                )
            model = new_lgb()
            model.fit(
                train_x.loc[eligible], train_y[eligible], sample_weight=sample_weight
            )
            correction["lgb_l7"][valid] = model.predict(
                features[validation_season].loc[valid, names]
            )
        for name in correction:
            correction[name] = np.clip(
                correction[name], -CORRECTION_CLIP, CORRECTION_CLIP
            )
        predictions = {
            "base": base[validation_season],
            "ridge_a1000_w025": np.clip(
                base[validation_season] + 0.25 * correction["ridge_a1000"], 0.0, 1.0
            ),
            "ridge_a10000_w025": np.clip(
                base[validation_season] + 0.25 * correction["ridge_a10000"], 0.0, 1.0
            ),
            "lgb_l7_w025": np.clip(
                base[validation_season] + 0.25 * correction["lgb_l7"], 0.0, 1.0
            ),
            "lgb_l7_w050": np.clip(
                base[validation_season] + 0.50 * correction["lgb_l7"], 0.0, 1.0
            ),
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
        "experiment": "EXP-045",
        "candidate_family": "exact_mapped_trackman_physical_residual",
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
            "physical_columns": list(PHYSICAL),
            "fine_pitch_types": list(FINE_TYPES),
            "feature_count": len(names),
            "correction_clip": CORRECTION_CLIP,
            "ridge_alphas": [1000.0, 10000.0],
            "lgb": new_lgb().get_params(),
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
            "sklearn": sklearn.__version__,
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
