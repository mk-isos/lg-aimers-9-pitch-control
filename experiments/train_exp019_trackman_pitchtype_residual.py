"""EXP-019: pitch-type-specific temporal Trackman residual correction.

For every outer season, the anonymous pitcher mapping is rebuilt using only
official rows through season-1 and is capped at mapping cost 0.10.  Trackman
features compare the immediately preceding season with the history before that
season, separately for fastball, breaking, and offspeed pitches.  The current
validation season and 2025 Trackman data are never used.

The immutable prediction base is the fixed 50/50 OOF average of R-full
LightGBM and HistGradientBoosting.  A small LightGBM learns season-centered OOF
residuals only on reliably mapped rows from earlier evaluated seasons.  No raw
ID, team, season, validation-row aggregation, or test-row aggregation enters
the residual model.
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

from trackman_features import (
    PITCH_GROUPS,
    TRACKMAN_NUMERIC_COLUMNS,
    build_pitcher_mapping,
)
from train_exp017_rolling_residual import calculate_metrics
from train_exp019_stable_monotonic import season_equal_weights


DATA_DIR = Path("./data")
TRACKMAN_PATH = DATA_DIR / "trackman_history.csv"
ARTIFACT_DIR = Path("./artifacts/EXP-019/trackman_pitchtype_residual")
LGB_ROOT = Path(
    "./artifacts/EXP-019/r_full_residual/rfull_l63_m1000_i300"
)
HGB_ROOT = Path(
    "./artifacts/EXP-019/histgb_residual/hist_l15_d4_m3000_i160"
)
LGB_VARIANT = "branch_w075"
HGB_VARIANT = "branch_w100"
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
BASE_WEIGHTS = {"lightgbm": 0.50, "histgradientboosting": 0.50}
CORRECTION_WEIGHTS = (0.25, 0.50)
MAX_MAPPING_COST = 0.10
ITERATIONS = 200
LEARNING_RATE = 0.015
NUM_LEAVES = 7
MIN_CHILD_SAMPLES = 2000
TEAM_EB_METRICS = Path(
    "./artifacts/EXP-019/team_eb_ensemble/validation_metrics.json"
)
TEAM_EB_REFERENCE = "all_prior_s1000"
RELEASE_COLUMNS = ("extension", "rel_height", "rel_side")


def load_main() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_id",
        "pitcher_team_id",
        "pitcher_hand",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
        "control_success",
    ]
    return pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=columns,
    )


def prediction_path(root: Path, variant: str, season: int) -> Path:
    return root / f"predictions_{variant}_{season}.npy"


def target_path(root: Path, season: int) -> Path:
    return root / f"targets_{season}.npy"


def load_fixed_oof(
    main: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
]:
    main_seasons = main["season"].to_numpy(dtype=np.int16)
    main_target = main["control_success"].to_numpy(dtype=np.int8)
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_seasons: list[np.ndarray] = []
    indices_by_season: dict[int, np.ndarray] = {}
    for season in VALIDATION_SEASONS:
        indices = np.flatnonzero(main_seasons == season)
        lgb_prediction = np.load(
            prediction_path(LGB_ROOT, LGB_VARIANT, season)
        ).astype(float)
        hgb_prediction = np.load(
            prediction_path(HGB_ROOT, HGB_VARIANT, season)
        ).astype(float)
        lgb_target = np.load(target_path(LGB_ROOT, season)).astype(np.int8)
        hgb_target = np.load(target_path(HGB_ROOT, season)).astype(np.int8)
        current_target = main_target[indices]
        if not (
            len(indices) == len(lgb_prediction) == len(hgb_prediction)
            and np.array_equal(current_target, lgb_target)
            and np.array_equal(current_target, hgb_target)
        ):
            raise ValueError(f"fixed OOF alignment mismatch for {season}")
        prediction = np.clip(
            BASE_WEIGHTS["lightgbm"] * lgb_prediction
            + BASE_WEIGHTS["histgradientboosting"] * hgb_prediction,
            0.0,
            1.0,
        )
        all_predictions.append(prediction)
        all_targets.append(current_target.astype(float))
        all_seasons.append(np.full(len(indices), season, dtype=np.int16))
        indices_by_season[season] = indices
    return (
        np.concatenate(all_predictions),
        np.concatenate(all_targets),
        np.concatenate(all_seasons),
        indices_by_season,
    )


def aggregate_period(
    rows: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    pitcher_ids = pd.Index(
        np.sort(rows["pitcher_trackman_id"].dropna().unique()),
        name="pitcher_trackman_id",
    )
    output = pd.DataFrame(index=pitcher_ids)
    group_mean_columns: dict[str, list[str]] = {
        column: [] for column in RELEASE_COLUMNS
    }
    for pitch_group in PITCH_GROUPS:
        group_rows = rows.loc[rows["pitch_type_group"] == pitch_group]
        grouped = group_rows.groupby("pitcher_trackman_id", sort=False)
        count_name = f"{prefix}_{pitch_group}_n"
        counts = grouped.size().reindex(pitcher_ids, fill_value=0).astype(float)
        output[count_name] = counts.astype(np.float32)
        output[f"{prefix}_{pitch_group}_log_n"] = np.log1p(counts).astype(
            np.float32
        )
        for column in TRACKMAN_NUMERIC_COLUMNS:
            mean_name = f"{prefix}_{pitch_group}_{column}_mean"
            std_name = f"{prefix}_{pitch_group}_{column}_std"
            output[mean_name] = grouped[column].mean().reindex(
                pitcher_ids
            ).astype(np.float32)
            output[std_name] = grouped[column].std().reindex(
                pitcher_ids
            ).astype(np.float32)
            if column in RELEASE_COLUMNS:
                group_mean_columns[column].append(mean_name)
        release_std_names = [
            f"{prefix}_{pitch_group}_{column}_std"
            for column in RELEASE_COLUMNS
        ]
        output[f"{prefix}_{pitch_group}_release_within_std_rms"] = np.sqrt(
            np.nanmean(
                np.square(
                    output[release_std_names].to_numpy(dtype=float)
                ),
                axis=1,
            )
        ).astype(np.float32)

    between_parts: list[np.ndarray] = []
    for column in RELEASE_COLUMNS:
        values = output[group_mean_columns[column]].to_numpy(dtype=float)
        between_parts.append(np.nanstd(values, axis=1))
    output[f"{prefix}_release_between_pitchtype_std_rms"] = np.sqrt(
        np.nanmean(np.square(np.column_stack(between_parts)), axis=1)
    ).astype(np.float32)
    return output


def build_pitchtype_summary(
    trackman: pd.DataFrame,
    prediction_season: int,
) -> pd.DataFrame:
    prior_rows = trackman.loc[trackman["season"] < prediction_season - 1]
    last_rows = trackman.loc[trackman["season"] == prediction_season - 1]
    prior = aggregate_period(prior_rows, "tm_prior")
    last = aggregate_period(last_rows, "tm_last")
    summary = prior.join(last, how="outer")

    for pitch_group in PITCH_GROUPS:
        for suffix in ("n", "log_n"):
            prior_name = f"tm_prior_{pitch_group}_{suffix}"
            last_name = f"tm_last_{pitch_group}_{suffix}"
            summary[prior_name] = summary[prior_name].fillna(0.0)
            summary[last_name] = summary[last_name].fillna(0.0)
            summary[f"tm_delta_{pitch_group}_{suffix}"] = (
                summary[last_name] - summary[prior_name]
            ).astype(np.float32)
        for column in TRACKMAN_NUMERIC_COLUMNS:
            for statistic in ("mean", "std"):
                prior_name = (
                    f"tm_prior_{pitch_group}_{column}_{statistic}"
                )
                last_name = f"tm_last_{pitch_group}_{column}_{statistic}"
                summary[f"tm_delta_{pitch_group}_{column}_{statistic}"] = (
                    summary[last_name] - summary[prior_name]
                ).astype(np.float32)
        prior_release = (
            f"tm_prior_{pitch_group}_release_within_std_rms"
        )
        last_release = f"tm_last_{pitch_group}_release_within_std_rms"
        summary[f"tm_delta_{pitch_group}_release_within_std_rms"] = (
            summary[last_release] - summary[prior_release]
        ).astype(np.float32)

    prior_between = "tm_prior_release_between_pitchtype_std_rms"
    last_between = "tm_last_release_between_pitchtype_std_rms"
    summary["tm_delta_release_between_pitchtype_std_rms"] = (
        summary[last_between] - summary[prior_between]
    ).astype(np.float32)
    summary["tm_prior_release_consistency"] = (
        1.0 / (1.0 + summary[prior_between])
    ).astype(np.float32)
    summary["tm_last_release_consistency"] = (
        1.0 / (1.0 + summary[last_between])
    ).astype(np.float32)
    summary["tm_delta_release_consistency"] = (
        summary["tm_last_release_consistency"]
        - summary["tm_prior_release_consistency"]
    ).astype(np.float32)
    return summary


def build_temporal_feature_matrix(
    main: pd.DataFrame,
    summary_trackman: pd.DataFrame,
    mapping_trackman: pd.DataFrame,
    indices_by_season: dict[int, np.ndarray],
) -> tuple[
    np.ndarray,
    list[str],
    np.ndarray,
    dict[str, object],
]:
    summaries = {
        season: build_pitchtype_summary(summary_trackman, season)
        for season in VALIDATION_SEASONS
    }
    feature_names = sorted(
        set().union(*(summary.columns for summary in summaries.values()))
    )
    total_rows = sum(len(indices_by_season[season]) for season in VALIDATION_SEASONS)
    X = np.full((total_rows, len(feature_names)), np.nan, dtype=np.float32)
    costs = np.full(total_rows, np.nan, dtype=float)
    audit: dict[str, object] = {}
    position = 0
    for season in VALIDATION_SEASONS:
        indices = indices_by_season[season]
        mapping = build_pitcher_mapping(
            main,
            mapping_trackman,
            cutoff_season=season - 1,
            max_cost=MAX_MAPPING_COST,
        )
        mapped_ids = main.loc[indices, "pitcher_id"].map(mapping.mapping)
        matched = summaries[season].reindex(mapped_ids.to_numpy())
        matched = matched.reindex(columns=feature_names)
        row_slice = slice(position, position + len(indices))
        X[row_slice] = matched.to_numpy(dtype=np.float32)
        season_costs = main.loc[indices, "pitcher_id"].map(mapping.costs)
        costs[row_slice] = season_costs.to_numpy(dtype=float)
        has_features = matched.notna().any(axis=1).to_numpy()
        audit[str(season)] = {
            "mapping_cutoff_season": season - 1,
            "trackman_prior_seasons": sorted(
                summary_trackman.loc[
                    summary_trackman["season"] < season - 1, "season"
                ].astype(int).unique().tolist()
            ),
            "trackman_last_season": season - 1,
            "mapped_pitchers": len(mapping.mapping),
            "candidate_main_ids": mapping.candidate_main_ids,
            "candidate_trackman_ids": mapping.candidate_trackman_ids,
            "row_mapping_coverage_cost_0_10": float(
                season_costs.notna().mean()
            ),
            "row_feature_coverage": float(has_features.mean()),
        }
        position += len(indices)
    if position != total_rows:
        raise RuntimeError("temporal Trackman matrix row mismatch")
    return X, feature_names, costs, audit


def aggregate_folds(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for weight in CORRECTION_WEIGHTS:
        candidate = f"trackman_w{int(weight * 100):03d}"
        skills = {
            season: float(
                folds[str(season)][candidate]["skill_score_unclipped"]
            )
            for season in REPORT_SEASONS
        }
        briers = {
            season: float(folds[str(season)][candidate]["brier_score"])
            for season in REPORT_SEASONS
        }
        aggregate[candidate] = {
            "season_skills": {
                str(season): value for season, value in skills.items()
            },
            "season_briers": {
                str(season): value for season, value in briers.items()
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
        }
    return aggregate


def main() -> None:
    started = time.time()
    main = load_main()
    fixed_base, y, seasons, indices_by_season = load_fixed_oof(main)
    trackman_full = pd.read_csv(TRACKMAN_PATH, encoding="utf-8-sig")
    trackman_pitchtypes = trackman_full.loc[
        trackman_full["pitch_type_group"].isin(PITCH_GROUPS)
    ].copy()
    X, feature_names, costs, mapping_audit = build_temporal_feature_matrix(
        main,
        trackman_pitchtypes,
        trackman_full,
        indices_by_season,
    )
    eligible = (
        np.isfinite(costs)
        & (costs <= MAX_MAPPING_COST)
        & np.isfinite(X).any(axis=1)
    )
    residual_target = (y - fixed_base).astype(np.float32)
    for season in VALIDATION_SEASONS:
        mask = seasons == season
        residual_target[mask] -= residual_target[mask].mean()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        validation_mask = seasons == validation_season
        validation_eligible = validation_mask & eligible
        train_mask = (seasons < validation_season) & eligible
        targets = y[validation_mask]
        correction = np.zeros(int(validation_mask.sum()), dtype=np.float64)
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "training_oof_seasons": sorted(
                np.unique(seasons[train_mask]).astype(int).tolist()
            ),
            "training_rows_mapped": int(train_mask.sum()),
            "validation_rows_mapped": int(validation_eligible.sum()),
            "validation_row_coverage": float(
                eligible[validation_mask].mean()
            ),
        }
        if train_mask.any() and validation_eligible.any():
            model = LGBMRegressor(
                objective="regression_l2",
                metric="l2",
                n_estimators=ITERATIONS,
                learning_rate=LEARNING_RATE,
                num_leaves=NUM_LEAVES,
                min_child_samples=MIN_CHILD_SAMPLES,
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
            fit_started = time.time()
            model.fit(
                X[train_mask],
                residual_target[train_mask],
                sample_weight=season_equal_weights(seasons[train_mask]),
            )
            local_eligible = eligible[validation_mask]
            correction[local_eligible] = model.booster_.predict(
                X[validation_eligible]
            ).astype(float)
            fold["fit_seconds"] = time.time() - fit_started
            fold["feature_importance"] = {
                name: int(value)
                for name, value in sorted(
                    zip(feature_names, model.feature_importances_, strict=True),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }
        else:
            fold["fit_seconds"] = 0.0
            fold["feature_importance"] = {}
            fold["no_correction_reason"] = (
                "no earlier fixed OOF season is available"
            )
        fold["correction_mean_mapped"] = float(
            correction[eligible[validation_mask]].mean()
        )
        fold["correction_std_mapped"] = float(
            correction[eligible[validation_mask]].std()
        )
        fold["fixed_base"] = calculate_metrics(targets, fixed_base[validation_mask])
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets)
        np.save(
            ARTIFACT_DIR / f"trackman_correction_{validation_season}.npy",
            correction,
        )
        for weight in CORRECTION_WEIGHTS:
            candidate = f"trackman_w{int(weight * 100):03d}"
            predictions = np.clip(
                fixed_base[validation_mask] + weight * correction,
                0.0,
                1.0,
            )
            fold[candidate] = calculate_metrics(targets, predictions)
            mapped = eligible[validation_mask]
            fold[f"mapped_{candidate}"] = calculate_metrics(
                targets[mapped], predictions[mapped]
            )
            fold[f"unmapped_{candidate}"] = calculate_metrics(
                targets[~mapped], predictions[~mapped]
            )
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate}_{validation_season}.npy",
                predictions,
            )
        folds[str(validation_season)] = fold
        print(
            f"tm_pitchtype {validation_season} "
            f"coverage={fold['validation_row_coverage']:.3f}: "
            + " ".join(
                f"w{int(weight * 100):03d}="
                f"{fold[f'trackman_w{int(weight * 100):03d}']['skill_score_unclipped']:.2f}"
                for weight in CORRECTION_WEIGHTS
            ),
            flush=True,
        )

    aggregate = aggregate_folds(folds)
    best_candidate = max(
        aggregate,
        key=lambda name: (
            float(aggregate[name]["min_skill"]),
            float(aggregate[name]["latest_2024_skill"]),
            float(aggregate[name]["mean_skill"]),
        ),
    )
    team_eb = json.loads(TEAM_EB_METRICS.read_text(encoding="utf-8"))
    benchmark_min = float(
        team_eb["aggregate_2022_2024"][TEAM_EB_REFERENCE][
            "team_eb_min_skill"
        ]
    )
    result: dict[str, object] = {
        "experiment": "EXP-019",
        "candidate_family": "pitchtype_temporal_trackman_residual",
        "validation_protocol": {
            "outer_folds": list(VALIDATION_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "fixed_base": (
                "50/50 saved OOF rfull LightGBM and HistGradientBoosting"
            ),
            "residual_training": (
                "mapped rows from earlier evaluated OOF seasons only; "
                "season-centered and season-equal"
            ),
            "mapping": "main history through prediction season-1; max cost 0.10",
            "trackman_prior": "seasons strictly before prediction season-1",
            "trackman_last": "prediction season-1 only",
            "current_fold_labels_used_for_training": False,
            "test_row_aggregation": False,
            "candidate_comparison_status": "single bounded predeclared test",
        },
        "model": {
            "features": feature_names,
            "feature_count": len(feature_names),
            "pitch_type_groups": list(PITCH_GROUPS),
            "physical_columns": list(TRACKMAN_NUMERIC_COLUMNS),
            "release_columns": list(RELEASE_COLUMNS),
            "max_mapping_cost": MAX_MAPPING_COST,
            "iterations": ITERATIONS,
            "learning_rate": LEARNING_RATE,
            "num_leaves": NUM_LEAVES,
            "min_child_samples": MIN_CHILD_SAMPLES,
            "correction_weights": list(CORRECTION_WEIGHTS),
            "excluded": [
                "pitcher_id",
                "team IDs",
                "season",
                "test-row aggregates",
            ],
        },
        "mapping_audit": mapping_audit,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "reference": {
            "team_eb_candidate": TEAM_EB_REFERENCE,
            "team_eb_min_skill": benchmark_min,
        },
        "selection": {
            "best_fixed_weight": best_candidate,
            "best_min_skill": aggregate[best_candidate]["min_skill"],
            "beats_team_eb_850_4": bool(
                float(aggregate[best_candidate]["min_skill"])
                > benchmark_min
            ),
            "stop_rule_triggered": bool(
                float(aggregate[best_candidate]["min_skill"])
                < benchmark_min
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
        "total_seconds": time.time() - started,
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f"selection={result['selection']}", flush=True)
    print(f"saved={ARTIFACT_DIR / 'validation_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
