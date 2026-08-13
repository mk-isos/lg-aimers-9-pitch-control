"""EXP-033: unused Trackman sequence, fine-pitch, and temporal trend signals.

This bounded experiment deliberately avoids repeating the existing Trackman
mean/std and coarse pitch-group residual tests.  It uses prior Trackman logs
to build pitcher profiles from detailed tagged/automatic pitch types, robust
physical quantiles, within-game pitch-number slopes, within-PA/inning/month
slopes, early/late deltas, and workload summaries.  Anonymous pitcher mapping
is rebuilt at every outer season using official history only through season-1.

The immutable base is the fixed EXP-021 lowrank-s300-r6 OOF prediction.  A
small LightGBM learns season-centered residuals from earlier OOF seasons only.
Validation labels and test-row aggregates are never used to construct, fit,
select, or calibrate a candidate.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from trackman_features import TRACKMAN_NUMERIC_COLUMNS, build_pitcher_mapping
from train_exp017_rolling_residual import calculate_metrics


EXPERIMENT = os.environ.get("TRACKMAN_EXPERIMENT", "EXP-033")
DATA_DIR = Path("./data")
ARTIFACT_DIR = Path(
    os.environ.get(
        "TRACKMAN_ARTIFACT_DIR",
        "./artifacts/EXP-033/trackman_sequence_trend",
    )
)
BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
MAX_MAPPING_COST = float(os.environ.get("TRACKMAN_MAX_MAPPING_COST", "0.10"))
SOURCE_POLICY = os.environ.get("TRACKMAN_SOURCE_POLICY", "pooled_equal")
CORRECTION_CLIP = 0.03
FINE_TYPES = (
    "fastball",
    "sinker",
    "cutter",
    "slider",
    "curveball",
    "changeup",
    "splitter",
    "other",
)
DERIVED_COLUMNS = (
    "speed_loss",
    "movement_magnitude",
    "release_radius",
)
PROFILE_COLUMNS = (*TRACKMAN_NUMERIC_COLUMNS, *DERIVED_COLUMNS)
CANDIDATES = (
    "sequence_w025",
    "sequence_reliable_w025",
    "sequence_reliable_w050",
)


def canonical_pitch_type(values: pd.Series) -> pd.Series:
    text = values.fillna("other").astype(str).str.lower()
    output = pd.Series("other", index=values.index, dtype=object)
    rules = (
        ("fastball", r"fastball|four-seam"),
        ("sinker", r"sinker"),
        ("cutter", r"cutter"),
        ("slider", r"slider|sweeper|slurve"),
        ("curveball", r"curve"),
        ("changeup", r"change"),
        ("splitter", r"splitter"),
    )
    for label, pattern in rules:
        output.loc[text.str.contains(pattern, regex=True)] = label
    return output


def add_trackman_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["tagged_fine_type"] = canonical_pitch_type(
        output["tagged_pitch_type"]
    )
    output["auto_fine_type"] = canonical_pitch_type(output["auto_pitch_type"])
    output["tag_auto_agree"] = (
        output["tagged_fine_type"].eq(output["auto_fine_type"]).astype(np.int8)
    )
    output["speed_loss"] = output["rel_speed"] - output["zone_speed"]
    output["movement_magnitude"] = np.hypot(
        output["induced_vert_break"], output["horz_break"]
    )
    output["release_radius"] = np.hypot(
        output["rel_height"], output["rel_side"]
    )
    return output


def normalized_rates(
    rows: pd.DataFrame,
    category_column: str,
    prefix: str,
) -> pd.DataFrame:
    counts = pd.crosstab(
        rows["pitcher_trackman_id"], rows[category_column]
    ).reindex(columns=FINE_TYPES, fill_value=0)
    rates = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    rates.columns = [f"{prefix}_{value}_rate" for value in rates.columns]
    entropy = -(rates * np.log(np.clip(rates, 1e-12, 1.0))).sum(axis=1)
    rates[f"{prefix}_entropy"] = entropy
    return rates


def centered_slope(
    rows: pd.DataFrame,
    x_column: str,
    center_keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    group = rows.groupby(center_keys, sort=False)[x_column]
    centered_x = rows[x_column].astype(float) - group.transform("mean")
    denominator = np.square(centered_x).groupby(
        rows["pitcher_trackman_id"]
    ).sum()
    output: dict[str, pd.Series] = {}
    for column in PROFILE_COLUMNS:
        numerator = (centered_x * rows[column].astype(float)).groupby(
            rows["pitcher_trackman_id"]
        ).sum(min_count=1)
        output[f"{prefix}_{column}_slope"] = numerator / denominator.replace(
            0.0, np.nan
        )
    return pd.DataFrame(output)


def period_delta(
    rows: pd.DataFrame,
    early_mask: np.ndarray,
    late_mask: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    early = rows.loc[early_mask].groupby("pitcher_trackman_id")[
        list(PROFILE_COLUMNS)
    ].mean()
    late = rows.loc[late_mask].groupby("pitcher_trackman_id")[
        list(PROFILE_COLUMNS)
    ].mean()
    delta = late.subtract(early, fill_value=np.nan)
    delta.columns = [f"{prefix}_{column}_delta" for column in delta.columns]
    return delta


def summarize_last_season(rows: pd.DataFrame, season: int) -> pd.DataFrame:
    last = rows.loc[rows["season"].eq(season - 1)].copy()
    if last.empty:
        return pd.DataFrame()
    grouped = last.groupby("pitcher_trackman_id", sort=False)
    summary = grouped.size().rename("tm_last_n").to_frame()
    summary["tm_last_log_n"] = np.log1p(summary["tm_last_n"])
    summary["tm_last_tag_auto_agree"] = grouped["tag_auto_agree"].mean()

    for column in PROFILE_COLUMNS:
        values = grouped[column]
        summary[f"tm_last_{column}_mean"] = values.mean()
        summary[f"tm_last_{column}_std"] = values.std()
        quantiles = values.quantile([0.1, 0.5, 0.9]).unstack()
        summary[f"tm_last_{column}_q10"] = quantiles[0.1]
        summary[f"tm_last_{column}_q50"] = quantiles[0.5]
        summary[f"tm_last_{column}_q90"] = quantiles[0.9]
        summary[f"tm_last_{column}_q80_width"] = (
            quantiles[0.9] - quantiles[0.1]
        )

    summary = summary.join(
        normalized_rates(last, "tagged_fine_type", "tm_last_tag"), how="left"
    )
    summary = summary.join(
        normalized_rates(last, "auto_fine_type", "tm_last_auto"), how="left"
    )

    slope_specs = (
        ("pitch_no", ["pitcher_trackman_id", "trackman_game_id"], "tm_game_pitchno"),
        ("pitch_of_pa", ["pitcher_trackman_id", "trackman_game_id"], "tm_game_pa_depth"),
        ("inning", ["pitcher_trackman_id"], "tm_season_inning"),
        ("game_month", ["pitcher_trackman_id"], "tm_season_month"),
    )
    for x_column, keys, prefix in slope_specs:
        summary = summary.join(
            centered_slope(last, x_column, keys, prefix), how="left"
        )

    summary = summary.join(
        period_delta(
            last,
            last["game_month"].le(5).to_numpy(),
            last["game_month"].ge(8).to_numpy(),
            "tm_month_late_early",
        ),
        how="left",
    )
    summary = summary.join(
        period_delta(
            last,
            last["inning"].le(3).to_numpy(),
            last["inning"].ge(7).to_numpy(),
            "tm_inning_late_early",
        ),
        how="left",
    )

    game_load = (
        last.groupby(["pitcher_trackman_id", "trackman_game_id"], sort=False)
        .size()
        .rename("pitches")
        .reset_index()
    )
    load_group = game_load.groupby("pitcher_trackman_id")["pitches"]
    summary["tm_last_games"] = load_group.size()
    summary["tm_last_pitches_per_game_mean"] = load_group.mean()
    summary["tm_last_pitches_per_game_std"] = load_group.std()
    summary["tm_last_pitches_per_game_q90"] = load_group.quantile(0.9)
    return summary


def summarize_history_mix(rows: pd.DataFrame, season: int) -> pd.DataFrame:
    history = rows.loc[rows["season"].lt(season)]
    if history.empty:
        return pd.DataFrame()
    tagged = normalized_rates(history, "tagged_fine_type", "tm_hist_tag")
    auto = normalized_rates(history, "auto_fine_type", "tm_hist_auto")
    output = tagged.join(auto, how="outer")
    output["tm_hist_n"] = history.groupby("pitcher_trackman_id").size()
    output["tm_hist_log_n"] = np.log1p(output["tm_hist_n"])
    return output


def build_feature_rows(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    mapping = build_pitcher_mapping(
        main,
        trackman,
        cutoff_season=season - 1,
        max_cost=MAX_MAPPING_COST,
    )
    last = summarize_last_season(trackman, season)
    history = summarize_history_mix(trackman, season)
    summary = history.join(last, how="outer")
    rows = main.loc[main["season"].eq(season)].copy()
    mapped_id = rows["pitcher_id"].map(mapping.mapping)
    features = summary.reindex(mapped_id.to_numpy()).reset_index(drop=True)
    features.index = rows.index
    features["trackman_mapping_cost"] = rows["pitcher_id"].map(mapping.costs)
    features["trackman_mapped"] = features["trackman_mapping_cost"].notna().astype(
        np.int8
    )
    features["current_game_month"] = rows["game_month"].astype(float)
    features["current_inning"] = rows["inning"].astype(float)
    features["current_count_index"] = (
        rows["balls_before"].astype(float) * 4.0
        + rows["strikes_before"].astype(float)
    )
    features["current_outs"] = rows["outs_before"].astype(float)
    features["current_pitcher_hand"] = rows["pitcher_hand"].astype(float)
    features["current_batter_hand"] = rows["batter_hand"].astype(float)
    features["current_is_regular"] = rows["game_type"].astype(str).eq("R").astype(
        np.int8
    )
    for column in PROFILE_COLUMNS:
        month_slope = f"tm_season_month_{column}_slope"
        inning_slope = f"tm_season_inning_{column}_slope"
        if month_slope in features:
            features[f"tm_expected_month_{column}"] = (
                features[month_slope] * (features["current_game_month"] - 6.5)
            )
        if inning_slope in features:
            features[f"tm_expected_inning_{column}"] = (
                features[inning_slope] * (features["current_inning"] - 5.0)
            )
    audit = {
        "mapping_cutoff_season": season - 1,
        "trackman_feature_seasons": sorted(
            trackman.loc[trackman["season"].lt(season), "season"]
            .astype(int)
            .unique()
            .tolist()
        ),
        "mapped_pitchers": len(mapping.mapping),
        "candidate_main_ids": mapping.candidate_main_ids,
        "candidate_trackman_ids": mapping.candidate_trackman_ids,
        "row_mapping_coverage": float(features["trackman_mapped"].mean()),
        "feature_count": int(features.shape[1]),
    }
    return features.reset_index(drop=True), audit


def load_main() -> pd.DataFrame:
    columns = [
        "season",
        "game_month",
        "inning",
        "game_type",
        "balls_before",
        "strikes_before",
        "outs_before",
        "pitcher_id",
        "pitcher_team_id",
        "pitcher_hand",
        "batter_hand",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
        "control_success",
    ]
    return pd.read_csv(
        DATA_DIR / "train.csv", encoding="utf-8-sig", usecols=columns
    )


def load_base(main: pd.DataFrame) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        mask = main["season"].eq(season).to_numpy()
        targets[season] = np.load(BASE_ROOT / f"targets_{season}.npy").astype(float)
        base[season] = np.load(
            BASE_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
        ).astype(float)
        csv_target = main.loc[mask, "control_success"].to_numpy(dtype=float)
        if not (
            len(csv_target) == len(targets[season]) == len(base[season])
            and np.array_equal(csv_target, targets[season])
        ):
            raise ValueError(f"base target/order mismatch for {season}")
    return targets, base


def aggregate(folds: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for candidate in ("base", *CANDIDATES):
        skills = {
            str(season): float(
                folds[str(season)][candidate]["skill_score_unclipped"]
            )
            for season in REPORT_SEASONS
        }
        briers = {
            str(season): float(folds[str(season)][candidate]["brier_score"])
            for season in REPORT_SEASONS
        }
        output[candidate] = {
            "season_briers": briers,
            "season_skills": skills,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills["2024"],
        }
    base = output["base"]
    for candidate in CANDIDATES:
        current = output[candidate]
        current["season_skill_change_vs_base"] = {
            season: float(current["season_skills"][season] - base["season_skills"][season])
            for season in map(str, REPORT_SEASONS)
        }
    return output


def new_residual_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        metric="l2",
        n_estimators=200,
        learning_rate=0.015,
        num_leaves=7,
        min_child_samples=2000,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
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
    trackman = add_trackman_derivatives(
        pd.read_csv(DATA_DIR / "trackman_history.csv", encoding="utf-8-sig")
    )
    targets, base = load_base(main)
    feature_rows: dict[int, pd.DataFrame] = {}
    mapping_audit: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        feature_rows[season], mapping_audit[str(season)] = build_feature_rows(
            main, trackman, season
        )
        print(
            f"features {season}: rows={len(feature_rows[season])} "
            f"coverage={mapping_audit[str(season)]['row_mapping_coverage']:.3f}",
            flush=True,
        )
    feature_names = list(feature_rows[EVALUATED_SEASONS[0]].columns)
    if any(list(feature_rows[s].columns) != feature_names for s in EVALUATED_SEASONS):
        raise ValueError("seasonal Trackman feature schema drift")

    residuals = {
        season: targets[season] - base[season] for season in EVALUATED_SEASONS
    }
    for season in EVALUATED_SEASONS:
        residuals[season] = residuals[season] - residuals[season].mean()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        source_seasons = [s for s in EVALUATED_SEASONS if s < validation_season]
        correction = np.zeros(len(targets[validation_season]), dtype=float)
        fit_seconds = 0.0
        if source_seasons:
            validation_eligible = feature_rows[validation_season][
                "trackman_mapped"
            ].eq(1).to_numpy()
            validation_matrix = (
                feature_rows[validation_season]
                .loc[validation_eligible, feature_names]
                .replace([np.inf, -np.inf], np.nan)
            )
            if SOURCE_POLICY == "pooled_equal":
                train_x = pd.concat(
                    [feature_rows[s] for s in source_seasons], ignore_index=True
                )
                train_y = np.concatenate([residuals[s] for s in source_seasons])
                source_labels = np.concatenate(
                    [
                        np.full(len(residuals[s]), s, dtype=np.int16)
                        for s in source_seasons
                    ]
                )
                eligible = train_x["trackman_mapped"].eq(1).to_numpy()
                counts = pd.Series(source_labels[eligible]).value_counts()
                sample_weight = np.array(
                    [
                        1.0 / counts[int(season)]
                        for season in source_labels[eligible]
                    ],
                    dtype=float,
                )
                sample_weight *= len(sample_weight) / sample_weight.sum()
                model = new_residual_model()
                fit_started = time.time()
                model.fit(
                    train_x.loc[eligible, feature_names].replace(
                        [np.inf, -np.inf], np.nan
                    ),
                    train_y[eligible],
                    sample_weight=sample_weight,
                )
                fit_seconds = time.time() - fit_started
                correction[validation_eligible] = model.predict(validation_matrix)
            elif SOURCE_POLICY == "per_source_recency2":
                source_predictions: list[np.ndarray] = []
                for source_season in source_seasons:
                    train_x = feature_rows[source_season]
                    eligible = train_x["trackman_mapped"].eq(1).to_numpy()
                    model = new_residual_model()
                    fit_started = time.time()
                    model.fit(
                        train_x.loc[eligible, feature_names].replace(
                            [np.inf, -np.inf], np.nan
                        ),
                        residuals[source_season][eligible],
                    )
                    fit_seconds += time.time() - fit_started
                    source_predictions.append(model.predict(validation_matrix))
                source_weights = np.power(
                    2.0, np.arange(len(source_predictions), dtype=float)
                )
                correction[validation_eligible] = np.average(
                    np.vstack(source_predictions), axis=0, weights=source_weights
                )
            else:
                raise ValueError(f"unknown Trackman source policy: {SOURCE_POLICY}")
            correction = np.clip(correction, -CORRECTION_CLIP, CORRECTION_CLIP)

        cost = feature_rows[validation_season]["trackman_mapping_cost"].to_numpy(
            dtype=float
        )
        reliability = np.clip(1.0 - np.nan_to_num(cost, nan=MAX_MAPPING_COST) / MAX_MAPPING_COST, 0.0, 1.0)
        predictions = {
            "base": base[validation_season],
            "sequence_w025": np.clip(base[validation_season] + 0.25 * correction, 0.0, 1.0),
            "sequence_reliable_w025": np.clip(base[validation_season] + 0.25 * reliability * correction, 0.0, 1.0),
            "sequence_reliable_w050": np.clip(base[validation_season] + 0.50 * reliability * correction, 0.0, 1.0),
        }
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "fit_seconds": fit_seconds,
            "validation_mapping_coverage": mapping_audit[str(validation_season)][
                "row_mapping_coverage"
            ],
        }
        for candidate, values in predictions.items():
            if not np.isfinite(values).all() or not (
                (values >= 0.0).all() and (values <= 1.0).all()
            ):
                raise ValueError(f"invalid prediction {candidate} {validation_season}")
            fold[candidate] = calculate_metrics(targets[validation_season], values)
            np.save(
                ARTIFACT_DIR / f"predictions_{candidate}_{validation_season}.npy",
                values,
            )
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets[validation_season].astype(np.int8))
        np.save(ARTIFACT_DIR / f"correction_{validation_season}.npy", correction)
        folds[str(validation_season)] = fold
        print(
            f"fold {validation_season}: "
            + " ".join(
                f"{name}={fold[name]['skill_score_unclipped']:.2f}"
                for name in CANDIDATES
            ),
            flush=True,
        )

    aggregate_metrics = aggregate(folds)
    best = max(
        CANDIDATES,
        key=lambda name: (
            aggregate_metrics[name]["min_skill"],
            aggregate_metrics[name]["latest_2024_skill"],
            aggregate_metrics[name]["mean_skill"],
        ),
    )
    result = {
        "experiment": EXPERIMENT,
        "candidate_family": "trackman_sequence_finepitch_temporal_trend",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "base": "fixed EXP-021 lowrank_s300_r6 OOF",
            "residual_training": "earlier OOF seasons only; season-centered and season-equal",
            "trackman_cutoff": "strictly before validation season",
            "mapping_cutoff": "official main history through validation season-1",
            "current_fold_labels_used_for_training_or_selection": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
            "source_policy": SOURCE_POLICY,
        },
        "model": {
            "type": "LightGBM regression residual",
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "physical_columns": list(PROFILE_COLUMNS),
            "fine_pitch_types": list(FINE_TYPES),
            "iterations": 200,
            "learning_rate": 0.015,
            "num_leaves": 7,
            "min_child_samples": 2000,
            "max_mapping_cost": MAX_MAPPING_COST,
            "correction_clip": CORRECTION_CLIP,
            "source_policy": SOURCE_POLICY,
        },
        "mapping_audit": mapping_audit,
        "folds": folds,
        "aggregate_2022_2024": aggregate_metrics,
        "selection": {
            "best_fixed_candidate": best,
            "best_min_skill": aggregate_metrics[best]["min_skill"],
            "gate_each_season_1000": bool(
                min(aggregate_metrics[best]["season_skills"].values()) >= 1000.0
            ),
            "gate_mean_1050": bool(aggregate_metrics[best]["mean_skill"] >= 1050.0),
            "adopt_for_full_fit": bool(
                min(aggregate_metrics[best]["season_skills"].values()) >= 1000.0
                and aggregate_metrics[best]["mean_skill"] >= 1050.0
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
        f"best={best} mean={aggregate_metrics[best]['mean_skill']:.2f} "
        f"min={aggregate_metrics[best]['min_skill']:.2f} "
        f"adopt={result['selection']['adopt_for_full_fit']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
