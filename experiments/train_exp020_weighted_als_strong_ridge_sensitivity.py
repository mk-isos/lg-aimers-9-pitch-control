"""EXP-020 follow-up: bounded strong-ridge weighted-ALS sensitivity.

This is a separate sensitivity check requested after the primary weighted-ALS
grid had completed.  It is not mixed into the original four-candidate
selection.  Rank is fixed at four and only ridge 300, 600, and 3000 are
evaluated to determine whether collapsing the interaction toward zero makes
the remaining two-way bias correction competitive with the basic SVD.

The temporal protocol is otherwise identical: immutable team all-prior OOF
base, residual centering inside each earlier OOF source season, count-weighted
observed-cell ALS, source-season equal averaging, and zero correction for an
unseen pitcher.  No current-fold labels, test rows, validation/test-row
aggregates, or further sensitivity expansion are used.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics
from train_exp020_weighted_als_pitcher_context import (
    ALS_ITERATIONS,
    BASE_ROOT,
    BASIC_REFERENCE_ROOT,
    CONTEXTS,
    EVALUATED_SEASONS,
    REPORT_SEASONS,
    SATURATED_REFERENCE_ROOT,
    correction_diagnostics,
    fit_weighted_als,
    load_oof,
    load_rows,
    summarize_coverage,
)


ARTIFACT_DIR = Path(
    "./artifacts/EXP-020/weighted_als_strong_ridge_sensitivity"
)
SENSITIVITY_RIDGES = (300.0, 600.0, 3000.0)
FIXED_RANK = 4

BASE_CANDIDATE = "base_team_all_prior"
BASIC_REFERENCE = "basic_svd_s300_r4_reference"
SATURATED_REFERENCE = "saturated_pctx_s600_reference"
SENSITIVITY_CANDIDATES = tuple(
    f"weighted_als_ridge{int(ridge)}_rank{FIXED_RANK}"
    for ridge in SENSITIVITY_RIDGES
)
CANDIDATES = (
    BASE_CANDIDATE,
    BASIC_REFERENCE,
    SATURATED_REFERENCE,
    *SENSITIVITY_CANDIDATES,
)


def candidate_name(ridge: float) -> str:
    return f"weighted_als_ridge{int(ridge)}_rank{FIXED_RANK}"


def fit_source_sensitivity(
    source_season: int,
    source_rows: pd.DataFrame,
    targets: np.ndarray,
    base: np.ndarray,
) -> dict[str, object]:
    raw_residual = targets - base
    raw_residual_mean = float(raw_residual.mean())
    residual = raw_residual - raw_residual_mean
    if abs(float(residual.mean())) > 1e-12:
        raise AssertionError("source residual centering failed")

    pitcher_codes, pitcher_ids = pd.factorize(
        source_rows["pitcher_id"], sort=True
    )
    context_positions = source_rows["context_position"].to_numpy(
        dtype=np.int16
    )
    matrix_shape = (len(pitcher_ids), len(CONTEXTS))
    residual_sums = np.zeros(matrix_shape, dtype=np.float64)
    counts = np.zeros(matrix_shape, dtype=np.int64)
    np.add.at(
        residual_sums, (pitcher_codes, context_positions), residual
    )
    np.add.at(counts, (pitcher_codes, context_positions), 1)
    if int(counts.sum()) != len(source_rows):
        raise AssertionError("source count matrix mismatch")
    cell_means = np.divide(
        residual_sums,
        counts,
        out=np.zeros_like(residual_sums),
        where=counts > 0,
    )

    fitted_matrices: dict[float, np.ndarray] = {}
    candidate_diagnostics: dict[str, object] = {}
    for ridge in SENSITIVITY_RIDGES:
        fitted, diagnostics = fit_weighted_als(
            cell_means,
            counts.astype(np.float64),
            ridge,
            FIXED_RANK,
        )
        fitted_matrices[ridge] = fitted
        candidate_diagnostics[candidate_name(ridge)] = diagnostics
    return {
        "source_season": source_season,
        "pitcher_ids": np.asarray(pitcher_ids),
        "counts": counts,
        "fitted_matrices": fitted_matrices,
        "diagnostics": {
            "source_rows": int(len(source_rows)),
            "source_pitchers": int(len(pitcher_ids)),
            "matrix_shape": list(matrix_shape),
            "observed_cells": int((counts > 0).sum()),
            "observed_density": float((counts > 0).mean()),
            "raw_residual_mean_before_centering": raw_residual_mean,
            "residual_mean_after_centering": float(residual.mean()),
            "candidates": candidate_diagnostics,
        },
    }


def map_source_sensitivity(
    source_model: dict[str, object],
    validation_rows: pd.DataFrame,
) -> dict[str, object]:
    source_row_indices = pd.Index(
        source_model["pitcher_ids"]
    ).get_indexer(validation_rows["pitcher_id"])
    pitcher_seen = source_row_indices >= 0
    safe_rows = np.where(pitcher_seen, source_row_indices, 0)
    context_positions = validation_rows["context_position"].to_numpy(
        dtype=np.int16
    )
    exact_context_seen = pitcher_seen & (
        source_model["counts"][safe_rows, context_positions] > 0
    )
    corrections: dict[float, np.ndarray] = {}
    for ridge in SENSITIVITY_RIDGES:
        values = np.zeros(len(validation_rows), dtype=np.float64)
        fitted = source_model["fitted_matrices"][ridge]
        values[pitcher_seen] = fitted[
            source_row_indices[pitcher_seen],
            context_positions[pitcher_seen],
        ]
        if np.any(values[~pitcher_seen] != 0.0):
            raise AssertionError("unseen pitcher received correction")
        corrections[ridge] = values
    return {
        "pitcher_seen": pitcher_seen,
        "exact_context_seen": exact_context_seen,
        "corrections": corrections,
    }


def aggregate_metrics(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for candidate in CANDIDATES:
        briers = {
            season: float(
                folds[str(season)]["candidates"][candidate]["brier_score"]
            )
            for season in REPORT_SEASONS
        }
        skills = {
            season: float(
                folds[str(season)]["candidates"][candidate][
                    "skill_score_unclipped"
                ]
            )
            for season in REPORT_SEASONS
        }
        aggregate[candidate] = {
            "season_briers": {
                str(season): value for season, value in briers.items()
            },
            "season_skills": {
                str(season): value for season, value in skills.items()
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
        }

    for candidate in SENSITIVITY_CANDIDATES:
        summary = aggregate[candidate]
        for label, reference in (
            ("base", BASE_CANDIDATE),
            ("basic_svd_r4", BASIC_REFERENCE),
            ("saturated_s600", SATURATED_REFERENCE),
        ):
            reference_summary = aggregate[reference]
            summary[f"season_skill_change_vs_{label}"] = {
                str(season): float(
                    summary["season_skills"][str(season)]
                    - reference_summary["season_skills"][str(season)]
                )
                for season in REPORT_SEASONS
            }
            summary[f"mean_skill_change_vs_{label}"] = float(
                summary["mean_skill"] - reference_summary["mean_skill"]
            )
            summary[f"min_skill_change_vs_{label}"] = float(
                summary["min_skill"] - reference_summary["min_skill"]
            )
    return aggregate


def main() -> None:
    started = time.time()
    rows = load_rows()
    (
        targets,
        base,
        basic_reference,
        saturated_reference,
    ) = load_oof(rows)
    source_models: dict[int, dict[str, object]] = {}

    def get_source_model(source_season: int) -> dict[str, object]:
        if source_season not in source_models:
            source_models[source_season] = fit_source_sensitivity(
                source_season,
                rows[source_season],
                targets[source_season],
                base[source_season],
            )
        return source_models[source_season]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        source_seasons = [
            season
            for season in EVALUATED_SEASONS
            if season < validation_season
        ]
        mapped_sources = {
            source_season: map_source_sensitivity(
                get_source_model(source_season), rows[validation_season]
            )
            for source_season in source_seasons
        }
        coverage, pitcher_source_count, exact_source_count = (
            summarize_coverage(
                mapped_sources, len(rows[validation_season])
            )
        )
        predictions: dict[str, np.ndarray] = {
            BASE_CANDIDATE: base[validation_season].copy(),
            BASIC_REFERENCE: basic_reference[validation_season].copy(),
            SATURATED_REFERENCE: saturated_reference[
                validation_season
            ].copy(),
        }
        corrections: dict[str, np.ndarray] = {}
        for ridge in SENSITIVITY_RIDGES:
            candidate = candidate_name(ridge)
            if mapped_sources:
                correction = np.mean(
                    np.vstack(
                        [
                            mapped["corrections"][ridge]
                            for mapped in mapped_sources.values()
                        ]
                    ),
                    axis=0,
                )
            else:
                correction = np.zeros(
                    len(rows[validation_season]), dtype=np.float64
                )
            if np.any(correction[pitcher_source_count == 0] != 0.0):
                raise AssertionError("unseen pitcher correction nonzero")
            prediction = np.clip(
                base[validation_season] + correction, 0.0, 1.0
            )
            corrections[candidate] = correction
            predictions[candidate] = prediction
        if tuple(predictions) != CANDIDATES:
            raise AssertionError("candidate declaration/order drift")

        metrics = {
            candidate: calculate_metrics(
                targets[validation_season], prediction
            )
            for candidate, prediction in predictions.items()
        }
        for candidate, prediction in predictions.items():
            if not np.isfinite(prediction).all() or not (
                (prediction >= 0.0).all()
                and (prediction <= 1.0).all()
            ):
                raise ValueError(
                    f"invalid prediction {validation_season} {candidate}"
                )
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate}_{validation_season}.npy",
                prediction,
            )
        for candidate, correction in corrections.items():
            np.save(
                ARTIFACT_DIR
                / f"correction_{candidate}_{validation_season}.npy",
                correction,
            )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets[validation_season].astype(np.int8),
        )
        np.save(
            ARTIFACT_DIR
            / f"pitcher_seen_source_count_{validation_season}.npy",
            pitcher_source_count,
        )
        np.save(
            ARTIFACT_DIR
            / f"exact_context_source_count_{validation_season}.npy",
            exact_source_count,
        )
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "rows": int(len(rows[validation_season])),
            "coverage": coverage,
            "correction_diagnostics": {
                candidate: correction_diagnostics(correction)
                for candidate, correction in corrections.items()
            },
            "candidates": metrics,
            "strict_source_invariants": {
                "all_source_seasons_precede_validation": bool(
                    all(
                        source_season < validation_season
                        for source_season in source_seasons
                    )
                ),
                "current_fold_labels_used_for_fit": False,
                "validation_or_test_rows_used_for_aggregation": False,
                "validation_rows_used_only_for_key_mapping": True,
                "source_residuals_centered_within_season": True,
                "source_seasons_combined_with_equal_weight": True,
                "unseen_pitcher_correction_is_zero": True,
            },
        }
        print(
            f"strong_ridge_sensitivity {validation_season}: "
            + " ".join(
                f"{candidate}="
                f"{metrics[candidate]['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    aggregate = aggregate_metrics(folds)
    best_sensitivity = max(
        SENSITIVITY_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["min_skill"],
            aggregate[candidate]["mean_skill"],
            -SENSITIVITY_CANDIDATES.index(candidate),
        ),
    )
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": (
            "bounded_strong_ridge_weighted_ALS_sensitivity"
        ),
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": (
                "EXP-019 team_eb_ensemble all_prior_s1000 OOF"
            ),
            "effect_target": (
                "source-season-centered target minus immutable base OOF"
            ),
            "source_season_combination": (
                "equal mean of earlier source corrections; absent pitcher "
                "contributes zero"
            ),
            "current_fold_labels_used_for_fit": False,
            "validation_or_test_row_aggregation": False,
            "validation_current_row_keys_only": True,
            "test_csv_read": False,
            "followup_requested_after_primary_grid": True,
            "mixed_into_primary_candidate_selection": False,
            "further_posthoc_expansion_allowed": False,
        },
        "bounded_configuration": {
            "ridges": list(SENSITIVITY_RIDGES),
            "fixed_rank": FIXED_RANK,
            "candidate_count": len(SENSITIVITY_CANDIDATES),
            "als_iterations": ALS_ITERATIONS,
            "objective": (
                "sum_observed cell_count*(cell_mean-fitted)^2 + "
                "ridge*(all bias/factor squared norms)"
            ),
            "purpose": (
                "test bias-dominated limit after interaction collapses"
            ),
        },
        "source_model_diagnostics": {
            str(source_season): source_model["diagnostics"]
            for source_season, source_model in source_models.items()
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": (
                "bounded sensitivity only; do not mix with primary "
                "post-hoc selection"
            ),
            "best_sensitivity_by_min_skill": best_sensitivity,
            "best_sensitivity_min_skill": aggregate[best_sensitivity][
                "min_skill"
            ],
            "best_sensitivity_beats_basic_svd_r4_min": bool(
                aggregate[best_sensitivity]["min_skill"]
                > aggregate[BASIC_REFERENCE]["min_skill"]
            ),
            "best_sensitivity_exceeds_1100": bool(
                aggregate[best_sensitivity]["min_skill"] >= 1100.0
            ),
            "stop_weighted_als_family": True,
            "adopt": False,
        },
        "qa": {
            "source_target_and_row_order_alignment_checked": True,
            "source_residual_centering_checked": True,
            "source_season_order_checked": True,
            "fixed_iteration_count_checked": True,
            "objective_monotonicity_checked": True,
            "prediction_probability_ranges_checked": True,
            "unseen_pitcher_zero_correction_checked": True,
            "saved_prediction_correction_and_coverage_arrays": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "total_seconds": float(time.time() - started),
    }
    metrics_path = ARTIFACT_DIR / "validation_metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"selection={result['selection']}", flush=True)
    print(f"saved={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
