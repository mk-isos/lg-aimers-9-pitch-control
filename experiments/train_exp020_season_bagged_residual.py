"""EXP-020: source-season-bagged R-only residual LightGBM.

The immutable base is the temporal-safe EXP-019 all-prior team-EB OOF
prediction.  For an outer validation season, one shallow residual model is
trained independently for every earlier evaluated OOF season.  Models are not
pooled: their current-row R corrections are combined by mean, median, or
unanimous sign consensus.  F rows always retain the immutable base.

The model configuration, feature whitelist, aggregation rules, and correction
weights are declared before evaluation.  Current-fold labels, validation-row
aggregates, raw player/team IDs, and season are never model inputs.
"""

from __future__ import annotations

import gc
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRegressor

import train_exp019_multirate_residual as multirate
from train_exp017_rolling_residual import calculate_metrics, segment_metrics
from train_exp019_game_type_branch import CORE_FEATURES, MONOTONE


SOURCE_DIR = Path("./artifacts/EXP-019/team_eb_ensemble")
ARTIFACT_DIR = Path("./artifacts/EXP-020/season_bagged_residual")
SOURCE_CANDIDATE = "all_prior_s1000"
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
AGGREGATIONS = ("mean", "median", "sign_consensus")
CORRECTION_WEIGHTS = (0.50, 0.75)

MODEL_CONFIG = {
    "name": "R_mono_l7_m5000_i200",
    "objective": "regression_l2",
    "iterations": 200,
    "learning_rate": 0.015,
    "num_leaves": 7,
    "min_child_samples": 5000,
    "max_bin": 127,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.90,
    "reg_alpha": 1.0,
    "reg_lambda": 12.0,
    "random_state": 42,
}

FORBIDDEN_FEATURES = {
    "row_id",
    "control_success",
    "season",
    "game_type",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
}


def candidate_name(aggregation: str, weight: float) -> str:
    return f"{aggregation}_w{int(weight * 100):03d}"


def regime_metrics(
    game_types: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        regime: calculate_metrics(
            targets[game_types == regime],
            predictions[game_types == regime],
        )
        for regime in sorted(np.unique(game_types))
    }


def make_model(constraints: list[int]) -> LGBMRegressor:
    return LGBMRegressor(
        objective=MODEL_CONFIG["objective"],
        metric="l2",
        n_estimators=MODEL_CONFIG["iterations"],
        learning_rate=MODEL_CONFIG["learning_rate"],
        num_leaves=MODEL_CONFIG["num_leaves"],
        max_depth=-1,
        min_child_samples=MODEL_CONFIG["min_child_samples"],
        max_bin=MODEL_CONFIG["max_bin"],
        subsample=MODEL_CONFIG["subsample"],
        subsample_freq=MODEL_CONFIG["subsample_freq"],
        colsample_bytree=MODEL_CONFIG["colsample_bytree"],
        reg_alpha=MODEL_CONFIG["reg_alpha"],
        reg_lambda=MODEL_CONFIG["reg_lambda"],
        monotone_constraints=constraints,
        monotone_constraints_method="advanced",
        random_state=MODEL_CONFIG["random_state"],
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def disagreement_metrics(corrections: np.ndarray) -> dict[str, object]:
    model_count, row_count = corrections.shape
    if model_count == 0:
        return {
            "source_model_count": 0,
            "R_rows": int(row_count),
            "across_model_std_mean": 0.0,
            "across_model_std_p90": 0.0,
            "across_model_std_max": 0.0,
            "mean_absolute_pairwise_difference": 0.0,
            "unanimous_sign_rows": 0,
            "unanimous_sign_rate": 0.0,
            "mean_median_absolute_difference": 0.0,
        }
    standard_deviation = corrections.std(axis=0)
    positive_consensus = np.all(corrections > 0.0, axis=0)
    negative_consensus = np.all(corrections < 0.0, axis=0)
    consensus = positive_consensus | negative_consensus
    pairwise_differences = [
        np.abs(corrections[left] - corrections[right]).mean()
        for left in range(model_count)
        for right in range(left + 1, model_count)
    ]
    return {
        "source_model_count": int(model_count),
        "R_rows": int(row_count),
        "across_model_std_mean": float(standard_deviation.mean()),
        "across_model_std_p90": float(
            np.quantile(standard_deviation, 0.90)
        ),
        "across_model_std_max": float(standard_deviation.max()),
        "mean_absolute_pairwise_difference": float(
            np.mean(pairwise_differences) if pairwise_differences else 0.0
        ),
        "unanimous_positive_rows": int(positive_consensus.sum()),
        "unanimous_negative_rows": int(negative_consensus.sum()),
        "unanimous_sign_rows": int(consensus.sum()),
        "unanimous_sign_rate": float(consensus.mean()),
        "mean_median_absolute_difference": float(
            np.abs(
                corrections.mean(axis=0)
                - np.median(corrections, axis=0)
            ).mean()
        ),
    }


def aggregate_correction(
    corrections: np.ndarray,
    aggregation: str,
) -> np.ndarray:
    if corrections.shape[0] == 0:
        return np.zeros(corrections.shape[1], dtype=float)
    if aggregation == "mean":
        return corrections.mean(axis=0)
    if aggregation == "median":
        return np.median(corrections, axis=0)
    if aggregation == "sign_consensus":
        mean = corrections.mean(axis=0)
        consensus = np.all(corrections > 0.0, axis=0) | np.all(
            corrections < 0.0, axis=0
        )
        return np.where(consensus, mean, 0.0)
    raise ValueError(f"unknown aggregation: {aggregation}")


def aggregate_metrics(
    folds: dict[str, object],
    metric_key: str,
) -> dict[str, object]:
    metrics = {
        season: folds[str(season)][metric_key]
        for season in REPORT_SEASONS
    }
    skills = {
        season: float(value["skill_score_unclipped"])
        for season, value in metrics.items()
    }
    return {
        "season_briers": {
            str(season): float(value["brier_score"])
            for season, value in metrics.items()
        },
        "season_skills": {
            str(season): value for season, value in skills.items()
        },
        "season_mean_gaps": {
            str(season): float(value["mean_gap"])
            for season, value in metrics.items()
        },
        "season_calibration_slopes": {
            str(season): float(value["diagnostic_calibration_slope"])
            for season, value in metrics.items()
        },
        "season_calibration_intercepts": {
            str(season): float(value["diagnostic_calibration_intercept"])
            for season, value in metrics.items()
        },
        "mean_skill": float(np.mean(list(skills.values()))),
        "min_skill": float(np.min(list(skills.values()))),
        "latest_2024_skill": skills[2024],
        "mean_absolute_gap": float(
            np.mean([abs(value["mean_gap"]) for value in metrics.values()])
        ),
        "max_absolute_gap": float(
            np.max([abs(value["mean_gap"]) for value in metrics.values()])
        ),
        "uniform_1100_passed": bool(
            all(value >= 1100.0 for value in skills.values())
        ),
    }


def main() -> None:
    started_at = time.time()
    frame, diagnostics, y, _, seasons, reconstruction = (
        multirate.prepare_multirate_data()
    )
    evaluation_mask = np.isin(seasons, VALIDATION_SEASONS)
    feature_names = sorted(CORE_FEATURES)
    forbidden_present = sorted(set(feature_names) & FORBIDDEN_FEATURES)
    if forbidden_present:
        raise ValueError(f"forbidden model features: {forbidden_present}")
    missing_features = sorted(set(feature_names) - set(frame.columns))
    if missing_features:
        raise ValueError(f"missing whitelist features: {missing_features}")
    X = frame.loc[evaluation_mask, feature_names].to_numpy(dtype=np.float32)
    game_types = frame.loc[evaluation_mask, "game_type"].astype(str).to_numpy()
    diagnostics = diagnostics.loc[evaluation_mask].reset_index(drop=True)
    y = y[evaluation_mask].astype(np.float64)
    seasons = seasons[evaluation_mask]
    del frame
    gc.collect()
    if set(np.unique(game_types)) != {"F", "R"}:
        raise ValueError(f"unexpected game_type values: {np.unique(game_types)}")
    is_r = game_types == "R"
    constraints = [MONOTONE.get(name, 0) for name in feature_names]

    base_by_season: dict[int, np.ndarray] = {}
    targets_by_season: dict[int, np.ndarray] = {}
    for season in VALIDATION_SEASONS:
        mask = seasons == season
        base = np.load(
            SOURCE_DIR / f"predictions_{SOURCE_CANDIDATE}_{season}.npy"
        ).astype(float)
        targets = np.load(SOURCE_DIR / f"targets_{season}.npy").astype(
            np.int8
        )
        if not (
            int(mask.sum()) == len(base) == len(targets)
            and np.array_equal(y[mask].astype(np.int8), targets)
        ):
            raise ValueError(f"OOF alignment mismatch for {season}")
        if not np.isfinite(base).all() or not (
            (base >= 0.0).all() and (base <= 1.0).all()
        ):
            raise ValueError(f"invalid base prediction for {season}")
        base_by_season[season] = base
        targets_by_season[season] = targets

    model_cache: dict[int, LGBMRegressor] = {}
    training_diagnostics: dict[str, object] = {}

    def get_source_model(source_season: int) -> LGBMRegressor:
        if source_season in model_cache:
            return model_cache[source_season]
        source_mask = (seasons == source_season) & is_r
        source_base = base_by_season[source_season]
        source_all_mask = seasons == source_season
        source_targets = y[source_all_mask]
        source_types = is_r[source_all_mask]
        raw_residual = source_targets[source_types] - source_base[source_types]
        center = float(raw_residual.mean())
        residual_target = (raw_residual - center).astype(np.float32)
        model = make_model(constraints)
        fit_started = time.time()
        model.fit(X[source_mask], residual_target)
        fit_seconds = time.time() - fit_started
        model_cache[source_season] = model
        training_diagnostics[str(source_season)] = {
            "source_oof_season": source_season,
            "R_training_rows": int(source_mask.sum()),
            "raw_residual_mean_before_centering": center,
            "residual_mean_after_centering": float(
                residual_target.mean()
            ),
            "fit_seconds": fit_seconds,
            "feature_importance": {
                name: int(value)
                for name, value in sorted(
                    zip(
                        feature_names,
                        model.feature_importances_,
                        strict=True,
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
        }
        return model

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        validation_mask = seasons == validation_season
        validation_r = validation_mask & is_r
        local_r = is_r[validation_mask]
        validation_types = game_types[validation_mask]
        targets = targets_by_season[validation_season]
        base = base_by_season[validation_season]
        source_seasons = [
            season
            for season in VALIDATION_SEASONS
            if season < validation_season
        ]
        source_predictions: list[np.ndarray] = []
        source_summaries: dict[str, object] = {}
        for source_season in source_seasons:
            model = get_source_model(source_season)
            correction = model.predict(X[validation_r]).astype(float)
            source_predictions.append(correction)
            source_summaries[str(source_season)] = {
                "mean": float(correction.mean()),
                "standard_deviation": float(correction.std()),
                "mean_absolute": float(np.abs(correction).mean()),
                "min": float(correction.min()),
                "max": float(correction.max()),
            }
        if source_predictions:
            correction_matrix = np.vstack(source_predictions)
        else:
            correction_matrix = np.empty(
                (0, int(validation_r.sum())), dtype=float
            )
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "R_validation_rows": int(validation_r.sum()),
            "F_validation_rows": int((validation_mask & ~is_r).sum()),
            "base_team_all_prior": calculate_metrics(targets, base),
            "base_regime_segments": regime_metrics(
                validation_types, targets, base
            ),
            "base_sample_segments": segment_metrics(
                diagnostics,
                validation_mask,
                targets,
                base,
            ),
            "source_model_corrections": source_summaries,
            "model_disagreement_R": disagreement_metrics(
                correction_matrix
            ),
            "candidates": {},
            "temporal_invariants": {
                "all_source_seasons_precede_validation": bool(
                    all(
                        season < validation_season
                        for season in source_seasons
                    )
                ),
                "current_fold_labels_used_for_model_fit": False,
                "models_trained_on_R_only": True,
                "F_predictions_equal_base": True,
                "validation_row_aggregation_used": False,
            },
        }
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets)
        for aggregation in AGGREGATIONS:
            R_correction = aggregate_correction(
                correction_matrix, aggregation
            )
            for weight in CORRECTION_WEIGHTS:
                name = candidate_name(aggregation, weight)
                predictions = base.copy()
                predictions[local_r] = np.clip(
                    base[local_r] + weight * R_correction,
                    0.0,
                    1.0,
                )
                if not np.array_equal(predictions[~local_r], base[~local_r]):
                    raise ValueError(
                        f"F invariant failed: {validation_season} {name}"
                    )
                if not np.isfinite(predictions).all() or not (
                    (predictions >= 0.0).all()
                    and (predictions <= 1.0).all()
                ):
                    raise ValueError(
                        f"invalid prediction: {validation_season} {name}"
                    )
                metrics = calculate_metrics(targets, predictions)
                fold["candidates"][name] = {
                    "aggregation": aggregation,
                    "correction_weight": weight,
                    "R_correction_mean": float(R_correction.mean()),
                    "R_correction_mean_absolute": float(
                        np.abs(R_correction).mean()
                    ),
                    "metrics": metrics,
                    "regime_segments": regime_metrics(
                        validation_types, targets, predictions
                    ),
                    "sample_segments": segment_metrics(
                        diagnostics,
                        validation_mask,
                        targets,
                        predictions,
                    ),
                }
                np.save(
                    ARTIFACT_DIR
                    / f"predictions_{name}_{validation_season}.npy",
                    predictions,
                )
                print(
                    f"season_bag {validation_season} {name}: "
                    f"skill={metrics['skill_score_unclipped']:.2f} "
                    f"gap={metrics['mean_gap']:+.6f}"
                )
        folds[str(validation_season)] = fold

    aggregate_folds: dict[str, object] = {}
    names = [
        candidate_name(aggregation, weight)
        for aggregation in AGGREGATIONS
        for weight in CORRECTION_WEIGHTS
    ]
    for season in VALIDATION_SEASONS:
        source_fold = folds[str(season)]
        aggregate_folds[str(season)] = {
            "base_team_all_prior": source_fold["base_team_all_prior"],
            **{
                name: source_fold["candidates"][name]["metrics"]
                for name in names
            },
        }
    aggregate = {
        "base_team_all_prior": aggregate_metrics(
            aggregate_folds, "base_team_all_prior"
        ),
        **{
            name: aggregate_metrics(aggregate_folds, name)
            for name in names
        },
    }
    base_skills = aggregate["base_team_all_prior"]["season_skills"]
    for name in names:
        change = {
            str(season): float(
                aggregate[name]["season_skills"][str(season)]
                - base_skills[str(season)]
            )
            for season in REPORT_SEASONS
        }
        aggregate[name]["season_skill_change_vs_base"] = change
        aggregate[name]["mean_skill_change_vs_base"] = float(
            aggregate[name]["mean_skill"]
            - aggregate["base_team_all_prior"]["mean_skill"]
        )
        aggregate[name]["min_skill_change_vs_base"] = float(
            aggregate[name]["min_skill"]
            - aggregate["base_team_all_prior"]["min_skill"]
        )
        aggregate[name]["improved_every_reported_season"] = bool(
            all(value > 0.0 for value in change.values())
        )
    best_mean = max(
        names, key=lambda name: float(aggregate[name]["mean_skill"])
    )
    best_min = max(
        names, key=lambda name: float(aggregate[name]["min_skill"])
    )
    result = {
        "experiment": "EXP-020",
        "candidate": "source_season_bagged_R_residual_atop_team_EB",
        "validation_protocol": {
            "evaluated_oof_seasons": list(VALIDATION_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": (
                "EXP-019 team_eb_ensemble predictions_all_prior_s1000"
            ),
            "source_model_training": (
                "one independent model per earlier evaluated OOF season, "
                "R rows only, R residual centered inside source season"
            ),
            "pooled_source_training_used": False,
            "current_fold_labels_used_for_training": False,
            "validation_or_test_row_aggregation": False,
            "F_prediction": "immutable base exactly",
            "candidate_grid_predeclared": True,
            "candidate_comparison_nested": False,
        },
        "model": {
            **MODEL_CONFIG,
            "features": feature_names,
            "feature_count": len(feature_names),
            "forbidden_features_absent": True,
            "monotone_constraints": dict(
                zip(feature_names, constraints, strict=True)
            ),
        },
        "aggregation_grid": {
            "methods": list(AGGREGATIONS),
            "correction_weights": list(CORRECTION_WEIGHTS),
            "sign_consensus": (
                "use source-model mean only when every model is positive "
                "or every model is negative for the current row; else zero"
            ),
        },
        "reconstruction_diagnostics": reconstruction,
        "source_model_training_diagnostics": training_diagnostics,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "diagnostic only; candidate comparison is non-nested",
            "posthoc_best_mean_candidate": best_mean,
            "posthoc_best_min_candidate": best_min,
            "any_candidate_improved_every_reported_season": bool(
                any(
                    aggregate[name]["improved_every_reported_season"]
                    for name in names
                )
            ),
            "adopt_without_nested_confirmation": False,
        },
        "qa": {
            "source_alignment_checked": True,
            "source_and_output_probability_ranges_checked": True,
            "strict_F_base_equality_checked": True,
            "source_season_order_checked": True,
            "forbidden_features_checked": True,
            "saved_prediction_arrays": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
        "total_seconds": float(time.time() - started_at),
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
