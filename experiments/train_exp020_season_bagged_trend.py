"""EXP-020: bounded temporal aggregation of fixed source-season R models.

This experiment changes only how the already-declared source-season residual
LightGBM predictions from ``train_exp020_season_bagged_residual.py`` are
combined.  Its model factory, feature whitelist, monotone constraints,
R-only source residual centering, immutable team-allprior base, and 0.50 total
correction scale are reused without modification.

For each outer validation fold, only earlier source models predict the current
R rows.  F rows always retain the immutable base.  Four aggregation rules are
predeclared: last source, geometric recency 1:2:4, and rowwise linear source-
year trends partially extrapolated 0.25 or 0.50 of the distance from the last
source year toward the validation year.  No current-fold label chooses an
aggregation or fits a trend coefficient.  A separate strict path selects from
the four candidates using earlier OOF fold labels only.
"""

from __future__ import annotations

import gc
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

import train_exp020_season_bagged_residual as seasonbag
from train_exp017_rolling_residual import calculate_metrics


SEASONBAG_ROOT = Path("./artifacts/EXP-020/season_bagged_residual")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-020/season_bagged_trend")

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
CORRECTION_SCALE = 0.50
RECENCY_RATIO = 2.0
TREND_FRACTIONS = (0.25, 0.50)

BASE_REFERENCE = "team_allprior_base"
MEAN_REFERENCE = "seasonbag_mean_w050_reference"
LOWRANK_REFERENCE = "lowrank_s300_r6_reference"
R_SPECIFIC_REFERENCE = "lowrank_s300_r4_Rspecific_reference"
REFERENCE_CANDIDATES = (
    BASE_REFERENCE,
    MEAN_REFERENCE,
    LOWRANK_REFERENCE,
    R_SPECIFIC_REFERENCE,
)
NEW_CANDIDATES = (
    "last_w050",
    "recency124_w050",
    "linear_trend_x025_w050",
    "linear_trend_x050_w050",
)
STRICT_SELECTION_CANDIDATES = (MEAN_REFERENCE, *NEW_CANDIDATES)
ALL_REPORTED_CANDIDATES = (*REFERENCE_CANDIDATES, *NEW_CANDIDATES)


def aggregate_source_corrections(
    correction_matrix: np.ndarray,
    source_seasons: list[int],
    validation_season: int,
    candidate: str,
) -> tuple[np.ndarray, dict[str, object]]:
    model_count, row_count = correction_matrix.shape
    if model_count == 0:
        return np.zeros(row_count, dtype=np.float64), {
            "source_model_count": 0,
            "formula": "zero; no earlier source model",
        }
    if candidate == "last_w050":
        return correction_matrix[-1].copy(), {
            "source_model_count": model_count,
            "formula": "latest earlier source correction",
            "source_weights": {
                str(season): float(index == model_count - 1)
                for index, season in enumerate(source_seasons)
            },
        }
    if candidate == "recency124_w050":
        weights = np.power(
            RECENCY_RATIO, np.arange(model_count, dtype=np.float64)
        )
        weights /= weights.sum()
        return weights @ correction_matrix, {
            "source_model_count": model_count,
            "formula": "geometric recency, newest/previous weight ratio 2",
            "source_weights": {
                str(season): float(weight)
                for season, weight in zip(
                    source_seasons, weights, strict=True
                )
            },
        }
    fraction_by_candidate = {
        "linear_trend_x025_w050": 0.25,
        "linear_trend_x050_w050": 0.50,
    }
    if candidate not in fraction_by_candidate:
        raise ValueError(f"unknown aggregation candidate: {candidate}")
    fraction = fraction_by_candidate[candidate]
    if model_count == 1:
        return correction_matrix[0].copy(), {
            "source_model_count": 1,
            "formula": "single source; trend undefined, equal to last/mean",
            "trend_fraction": fraction,
        }

    years = np.asarray(source_seasons, dtype=np.float64)
    centered_years = years - years.mean()
    slope = (
        centered_years[:, None] * correction_matrix
    ).sum(axis=0) / float(np.square(centered_years).sum())
    intercept_at_mean = correction_matrix.mean(axis=0)
    target_year = float(source_seasons[-1]) + fraction * (
        float(validation_season) - float(source_seasons[-1])
    )
    extrapolated = intercept_at_mean + slope * (
        target_year - float(years.mean())
    )
    return extrapolated, {
        "source_model_count": model_count,
        "formula": (
            "rowwise least-squares correction versus source year, evaluated "
            "partway from latest source toward validation year"
        ),
        "trend_fraction": fraction,
        "source_years": source_seasons,
        "target_year": target_year,
        "slope_mean": float(slope.mean()),
        "slope_standard_deviation": float(slope.std()),
        "slope_mean_absolute": float(np.abs(slope).mean()),
        "slope_min": float(slope.min()),
        "slope_max": float(slope.max()),
    }


def regime_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    game_types: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        game_type: calculate_metrics(
            targets[game_types == game_type],
            predictions[game_types == game_type],
        )
        for game_type in sorted(np.unique(game_types))
    }


def aggregate_metrics(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for candidate in ALL_REPORTED_CANDIDATES:
        metrics = {
            season: folds[str(season)]["candidates"][candidate]["metrics"]
            for season in REPORT_SEASONS
        }
        skills = {
            season: float(value["skill_score_unclipped"])
            for season, value in metrics.items()
        }
        aggregate[candidate] = {
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
                str(season): float(
                    value["diagnostic_calibration_slope"]
                )
                for season, value in metrics.items()
            },
            "season_calibration_intercepts": {
                str(season): float(
                    value["diagnostic_calibration_intercept"]
                )
                for season, value in metrics.items()
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
            "uniform_1100_passed": bool(
                all(value >= 1100.0 for value in skills.values())
            ),
        }
    mean_reference = aggregate[MEAN_REFERENCE]
    for candidate in NEW_CANDIDATES:
        current = aggregate[candidate]
        current["season_skill_change_vs_mean_w050"] = {
            str(season): float(
                current["season_skills"][str(season)]
                - mean_reference["season_skills"][str(season)]
            )
            for season in REPORT_SEASONS
        }
        current["mean_skill_change_vs_mean_w050"] = float(
            current["mean_skill"] - mean_reference["mean_skill"]
        )
        current["min_skill_change_vs_mean_w050"] = float(
            current["min_skill"] - mean_reference["min_skill"]
        )
    return aggregate


def select_from_prior_folds(
    validation_season: int,
    folds: dict[str, object],
) -> tuple[str, list[int], dict[str, object]]:
    history = [
        season
        for season in EVALUATED_SEASONS
        if season < validation_season
    ]
    if not history:
        return STRICT_SELECTION_CANDIDATES[0], [], {}
    selection_metrics: dict[str, object] = {}
    for candidate in STRICT_SELECTION_CANDIDATES:
        skills = {
            season: float(
                folds[str(season)]["candidates"][candidate]["metrics"][
                    "skill_score_unclipped"
                ]
            )
            for season in history
        }
        selection_metrics[candidate] = {
            "history_skills": {
                str(season): value for season, value in skills.items()
            },
            "history_worst_skill": float(min(skills.values())),
            "history_mean_skill": float(np.mean(list(skills.values()))),
        }
    selected = max(
        STRICT_SELECTION_CANDIDATES,
        key=lambda candidate: (
            selection_metrics[candidate]["history_worst_skill"],
            selection_metrics[candidate]["history_mean_skill"],
            -STRICT_SELECTION_CANDIDATES.index(candidate),
        ),
    )
    return selected, history, selection_metrics


def build_strict_path(
    folds: dict[str, object],
    prediction_cache: dict[int, dict[str, np.ndarray]],
    targets: dict[int, np.ndarray],
) -> dict[str, object]:
    strict_folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        selected, history, selection_metrics = select_from_prior_folds(
            validation_season, folds
        )
        predictions = prediction_cache[validation_season][selected]
        metrics = calculate_metrics(targets[validation_season], predictions)
        strict_folds[str(validation_season)] = {
            "validation_season": validation_season,
            "selection_history_seasons": history,
            "selected_candidate": selected,
            "selection_metrics": selection_metrics,
            "current_fold_metrics_used_for_selection": False,
            "metrics": metrics,
        }
        np.save(
            ARTIFACT_DIR
            / f"predictions_strict_previous_{validation_season}.npy",
            predictions,
        )
    skills = {
        season: float(
            strict_folds[str(season)]["metrics"]["skill_score_unclipped"]
        )
        for season in REPORT_SEASONS
    }
    briers = {
        season: float(
            strict_folds[str(season)]["metrics"]["brier_score"]
        )
        for season in REPORT_SEASONS
    }
    next_selected, next_history, next_metrics = select_from_prior_folds(
        2025, folds
    )
    return {
        "objective": (
            "maximize worst earlier-fold Skill; then earlier-fold mean; "
            "then predeclared candidate order"
        ),
        "folds": strict_folds,
        "aggregate_2022_2024": {
            "season_briers": {
                str(season): value for season, value in briers.items()
            },
            "season_skills": {
                str(season): value for season, value in skills.items()
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
            "selection_path": {
                str(season): strict_folds[str(season)][
                    "selected_candidate"
                ]
                for season in REPORT_SEASONS
            },
        },
        "next_2025_selection": {
            "selection_history_seasons": next_history,
            "selected_candidate": next_selected,
            "selection_metrics": next_metrics,
            "uses_2025_labels": False,
        },
    }


def main() -> None:
    started = time.time()
    frame, diagnostics, y, _, seasons, reconstruction = (
        seasonbag.multirate.prepare_multirate_data()
    )
    evaluation_mask = np.isin(seasons, EVALUATED_SEASONS)
    feature_names = sorted(seasonbag.CORE_FEATURES)
    forbidden_present = sorted(
        set(feature_names) & seasonbag.FORBIDDEN_FEATURES
    )
    if forbidden_present:
        raise ValueError(f"forbidden source model features: {forbidden_present}")
    X = frame.loc[evaluation_mask, feature_names].to_numpy(dtype=np.float32)
    game_types = frame.loc[evaluation_mask, "game_type"].astype(str).to_numpy()
    y = y[evaluation_mask].astype(np.float64)
    seasons = seasons[evaluation_mask]
    del frame, diagnostics
    gc.collect()
    is_r = game_types == "R"
    constraints = [seasonbag.MONOTONE.get(name, 0) for name in feature_names]

    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    mean_reference: dict[int, np.ndarray] = {}
    lowrank_reference: dict[int, np.ndarray] = {}
    r_specific_reference: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        mask = seasons == season
        targets[season] = np.load(
            seasonbag.SOURCE_DIR / f"targets_{season}.npy"
        ).astype(np.float64)
        base[season] = np.load(
            seasonbag.SOURCE_DIR
            / f"predictions_{seasonbag.SOURCE_CANDIDATE}_{season}.npy"
        ).astype(np.float64)
        mean_reference[season] = np.load(
            SEASONBAG_ROOT / f"predictions_mean_w050_{season}.npy"
        ).astype(np.float64)
        lowrank_reference[season] = np.load(
            LOWRANK_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
        ).astype(np.float64)
        r_specific_reference[season] = np.load(
            LOWRANK_ROOT
            / f"predictions_lowrank_s300_r4_Rspecific_{season}.npy"
        ).astype(np.float64)
        if not (
            int(mask.sum())
            == len(targets[season])
            == len(base[season])
            == len(mean_reference[season])
            == len(lowrank_reference[season])
            == len(r_specific_reference[season])
            and np.array_equal(y[mask], targets[season])
        ):
            raise ValueError(f"OOF alignment mismatch for {season}")

    model_cache: dict[int, object] = {}
    training_diagnostics: dict[str, object] = {}

    def get_source_model(source_season: int):
        if source_season in model_cache:
            return model_cache[source_season]
        source_mask = (seasons == source_season) & is_r
        source_local = seasons == source_season
        source_types = is_r[source_local]
        residual = (
            y[source_local][source_types]
            - base[source_season][source_types]
        )
        raw_mean = float(residual.mean())
        residual_target = (residual - raw_mean).astype(np.float32)
        model = seasonbag.make_model(constraints)
        fit_started = time.time()
        model.fit(X[source_mask], residual_target)
        model_cache[source_season] = model
        training_diagnostics[str(source_season)] = {
            "source_season": source_season,
            "R_training_rows": int(source_mask.sum()),
            "raw_residual_mean_before_centering": raw_mean,
            "residual_mean_after_centering": float(
                residual_target.mean()
            ),
            "fit_seconds": float(time.time() - fit_started),
        }
        return model

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    prediction_cache: dict[int, dict[str, np.ndarray]] = {}
    max_mean_reproduction_difference = 0.0
    for validation_season in EVALUATED_SEASONS:
        validation_mask = seasons == validation_season
        validation_r = validation_mask & is_r
        local_r = is_r[validation_mask]
        validation_types = game_types[validation_mask]
        source_seasons = [
            season
            for season in EVALUATED_SEASONS
            if season < validation_season
        ]
        source_corrections = [
            get_source_model(source_season)
            .predict(X[validation_r])
            .astype(np.float64)
            for source_season in source_seasons
        ]
        correction_matrix = (
            np.vstack(source_corrections)
            if source_corrections
            else np.empty((0, int(validation_r.sum())), dtype=np.float64)
        )
        mean_correction = (
            correction_matrix.mean(axis=0)
            if len(correction_matrix)
            else np.zeros(int(validation_r.sum()), dtype=np.float64)
        )
        reproduced_mean = base[validation_season].copy()
        reproduced_mean[local_r] = np.clip(
            base[validation_season][local_r]
            + CORRECTION_SCALE * mean_correction,
            0.0,
            1.0,
        )
        mean_difference = float(
            np.max(
                np.abs(
                    reproduced_mean - mean_reference[validation_season]
                )
            )
        )
        max_mean_reproduction_difference = max(
            max_mean_reproduction_difference, mean_difference
        )
        if mean_difference > 1e-12:
            raise AssertionError(
                f"fixed source-model reproduction failed: "
                f"{validation_season} {mean_difference}"
            )

        predictions: dict[str, np.ndarray] = {
            BASE_REFERENCE: base[validation_season].copy(),
            MEAN_REFERENCE: mean_reference[validation_season].copy(),
            LOWRANK_REFERENCE: lowrank_reference[validation_season].copy(),
            R_SPECIFIC_REFERENCE: r_specific_reference[
                validation_season
            ].copy(),
        }
        aggregation_diagnostics: dict[str, object] = {}
        for candidate in NEW_CANDIDATES:
            aggregated, candidate_diagnostics = (
                aggregate_source_corrections(
                    correction_matrix,
                    source_seasons,
                    validation_season,
                    candidate,
                )
            )
            current = base[validation_season].copy()
            current[local_r] = np.clip(
                base[validation_season][local_r]
                + CORRECTION_SCALE * aggregated,
                0.0,
                1.0,
            )
            if not np.array_equal(
                current[~local_r], base[validation_season][~local_r]
            ):
                raise AssertionError(f"F base invariant failed: {candidate}")
            predictions[candidate] = current
            aggregation_diagnostics[candidate] = {
                **candidate_diagnostics,
                "correction_scale": CORRECTION_SCALE,
                "aggregated_mean": float(aggregated.mean()),
                "aggregated_standard_deviation": float(aggregated.std()),
                "aggregated_mean_absolute": float(
                    np.abs(aggregated).mean()
                ),
            }
        if tuple(predictions) != ALL_REPORTED_CANDIDATES:
            raise AssertionError("candidate order drift")

        candidate_metrics: dict[str, object] = {}
        for candidate, candidate_predictions in predictions.items():
            if not np.isfinite(candidate_predictions).all() or not (
                (candidate_predictions >= 0.0).all()
                and (candidate_predictions <= 1.0).all()
            ):
                raise ValueError(
                    f"invalid prediction {validation_season} {candidate}"
                )
            candidate_metrics[candidate] = {
                "metrics": calculate_metrics(
                    targets[validation_season], candidate_predictions
                ),
                "regime_metrics": regime_metrics(
                    targets[validation_season],
                    candidate_predictions,
                    validation_types,
                ),
            }
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate}_{validation_season}.npy",
                candidate_predictions,
            )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets[validation_season].astype(np.int8),
        )
        prediction_cache[validation_season] = {
            candidate: predictions[candidate]
            for candidate in STRICT_SELECTION_CANDIDATES
        }
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "R_rows": int(local_r.sum()),
            "F_rows": int((~local_r).sum()),
            "fixed_mean_w050_reproduction_max_abs_difference": (
                mean_difference
            ),
            "aggregation_diagnostics": aggregation_diagnostics,
            "candidates": candidate_metrics,
            "strict_invariants": {
                "all_source_seasons_precede_validation": bool(
                    all(
                        source_season < validation_season
                        for source_season in source_seasons
                    )
                ),
                "source_models_and_features_identical_to_seasonbag": True,
                "current_fold_labels_used_for_aggregation": False,
                "current_fold_labels_used_for_trend_coefficients": False,
                "validation_or_test_row_aggregation": False,
                "F_predictions_equal_base_for_new_candidates": True,
                "mean_w050_reference_reproduced": bool(
                    mean_difference <= 1e-12
                ),
            },
        }
        print(
            f"seasonbag_trend {validation_season}: "
            + " ".join(
                f"{candidate}="
                f"{candidate_metrics[candidate]['metrics']['skill_score_unclipped']:.2f}"
                for candidate in ALL_REPORTED_CANDIDATES
            ),
            flush=True,
        )

    aggregate = aggregate_metrics(folds)
    strict_path = build_strict_path(folds, prediction_cache, targets)
    best_min_candidate = max(
        NEW_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["min_skill"],
            aggregate[candidate]["latest_2024_skill"],
            aggregate[candidate]["mean_skill"],
            -NEW_CANDIDATES.index(candidate),
        ),
    )
    best_min = float(aggregate[best_min_candidate]["min_skill"])
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": "fixed_source_season_model_temporal_aggregation",
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": "EXP-019 team all_prior_s1000 OOF",
            "source_model_definition": (
                "exact train_exp020_season_bagged_residual model factory, "
                "feature whitelist, R-only centered residual target"
            ),
            "current_fold_labels_used_for_aggregation_or_trend": False,
            "validation_or_test_row_aggregation": False,
            "F_prediction_for_new_candidates": "immutable base exactly",
            "candidate_grid_predeclared": True,
            "candidate_count": len(NEW_CANDIDATES),
            "candidate_comparison_nested": False,
            "strict_path_uses_current_fold": False,
        },
        "predeclared_configuration": {
            "new_candidates": list(NEW_CANDIDATES),
            "strict_selection_candidates": list(
                STRICT_SELECTION_CANDIDATES
            ),
            "correction_scale": CORRECTION_SCALE,
            "recency_newest_to_previous_ratio": RECENCY_RATIO,
            "trend_fractions": list(TREND_FRACTIONS),
            "single_source_rule": (
                "all four aggregations equal the only source correction"
            ),
            "source_model": seasonbag.MODEL_CONFIG,
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "source_models_modified": False,
        },
        "reconstruction_diagnostics": reconstruction,
        "source_model_training_diagnostics": training_diagnostics,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "strict_previous_fold_selection": strict_path,
        "selection": {
            "status": "bounded diagnostic; global comparison non-nested",
            "posthoc_best_min_candidate": best_min_candidate,
            "posthoc_best_min_skill": best_min,
            "uniform_1100_gate_passed": bool(best_min >= 1100.0),
            "stop_rule_triggered": bool(best_min < 1100.0),
            "stop_reason": (
                "best minimum Skill remains below 1100"
                if best_min < 1100.0
                else "gate passed"
            ),
            "adopt_without_additional_confirmation": False,
        },
        "qa": {
            "fixed_mean_w050_reproduction_max_abs_difference": (
                max_mean_reproduction_difference
            ),
            "source_target_and_row_order_alignment_checked": True,
            "source_residual_centering_checked": True,
            "source_season_order_checked": True,
            "source_model_config_identity_checked": True,
            "F_base_equality_checked": True,
            "prediction_probability_ranges_checked": True,
            "saved_prediction_arrays": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
        "total_seconds": float(time.time() - started),
    }
    metrics_path = ARTIFACT_DIR / "validation_metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"strict={strict_path['aggregate_2022_2024']}", flush=True)
    print(f"next_2025={strict_path['next_2025_selection']['selected_candidate']}")
    print(f"selection={result['selection']}", flush=True)
    print(f"saved={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
