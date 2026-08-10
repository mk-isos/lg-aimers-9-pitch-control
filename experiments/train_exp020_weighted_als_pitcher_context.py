"""EXP-020: count-weighted ALS for pitcher-by-count-context residuals.

The immutable prediction base is the temporal-safe EXP-019 team
``all_prior_s1000`` OOF.  For each outer validation season, every earlier OOF
season independently fits a two-way residual model over the static 24-cell
``pitcher_id x (count_index, batter_hand)`` matrix:

    pitcher bias + context bias + low-rank interaction

Only observed cells enter the squared-error objective, weighted by their row
counts.  All parameters receive the same ridge penalty and are optimized by
deterministic alternating least squares for a fixed iteration count.  The
source residual is centered inside its season before aggregation.  At
validation time, source-season corrections are averaged equally and an
unseen pitcher receives zero, matching the existing basic SVD protocol.

The rank/ridge grid is fixed below.  Ridge 300/600 was rejected before any
validation result was inspected because an earliest-source-2021
training-objective-only scale audit showed that symmetric ridge >=150
collapses the interaction to numerical zero.  The declared 30/60 grid retains
strong regularization while remaining non-degenerate.  No R-specific branch
is evaluated.

No current-fold label, validation/test aggregate, test row, or post-result
tuning is used.  Validation rows supply only current-row official keys.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics


DATA_PATH = Path("./data/train.csv")
BASE_ROOT = Path("./artifacts/EXP-019/team_eb_ensemble")
BASIC_REFERENCE_ROOT = Path(
    "./artifacts/EXP-020/low_rank_pitcher_context_eb"
)
SATURATED_REFERENCE_ROOT = Path(
    "./artifacts/EXP-020/pitcher_count_eb_atop_team"
)
ARTIFACT_DIR = Path(
    "./artifacts/EXP-020/weighted_als_pitcher_context"
)

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
RIDGES = (30.0, 60.0)
RANKS = (2, 4)
SCALE_AUDIT_RIDGES = (300.0, 600.0)
SCALE_AUDIT_RANK = 4
BIAS_INITIALIZATION_ITERATIONS = 10
ALS_ITERATIONS = 30
OBJECTIVE_INCREASE_TOLERANCE = 1e-9

BATTER_HANDS = (1, 2)
COUNT_INDICES = tuple(
    balls * 4 + strikes
    for balls in range(4)
    for strikes in range(3)
)
CONTEXTS = tuple(
    (count_index, batter_hand)
    for count_index in COUNT_INDICES
    for batter_hand in BATTER_HANDS
)
CONTEXT_TO_POSITION = {
    context: position for position, context in enumerate(CONTEXTS)
}

BASE_CANDIDATE = "base_team_all_prior"
BASIC_REFERENCE = "basic_svd_s300_r4_reference"
SATURATED_REFERENCE = "saturated_pctx_s600_reference"
ALS_CANDIDATES = tuple(
    f"weighted_als_ridge{int(ridge)}_rank{rank}"
    for ridge in RIDGES
    for rank in RANKS
)
CANDIDATES = (
    BASE_CANDIDATE,
    BASIC_REFERENCE,
    SATURATED_REFERENCE,
    *ALS_CANDIDATES,
)


def als_name(ridge: float, rank: int) -> str:
    return f"weighted_als_ridge{int(ridge)}_rank{rank}"


def load_rows() -> dict[int, pd.DataFrame]:
    columns = [
        "season",
        "pitcher_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "control_success",
    ]
    frame = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=columns,
    )
    frame = frame.loc[frame["season"].isin(EVALUATED_SEASONS)].copy()
    if frame[
        [
            "pitcher_id",
            "balls_before",
            "strikes_before",
            "batter_hand",
            "control_success",
        ]
    ].isna().any().any():
        raise ValueError("missing required pitcher-context field")

    frame["count_index"] = (
        frame["balls_before"].to_numpy(dtype=np.int8) * 4
        + frame["strikes_before"].to_numpy(dtype=np.int8)
    ).astype(np.int8)
    observed_counts = set(frame["count_index"].astype(int).unique())
    observed_hands = set(frame["batter_hand"].astype(int).unique())
    if not observed_counts.issubset(set(COUNT_INDICES)):
        raise ValueError(f"unexpected count_index: {observed_counts}")
    if not observed_hands.issubset(set(BATTER_HANDS)):
        raise ValueError(f"unexpected batter_hand: {observed_hands}")

    frame["context_position"] = [
        CONTEXT_TO_POSITION[(int(count_index), int(batter_hand))]
        for count_index, batter_hand in zip(
            frame["count_index"], frame["batter_hand"], strict=True
        )
    ]
    frame["context_position"] = frame["context_position"].astype(
        np.int8
    )
    return {
        season: frame.loc[frame["season"].eq(season)].reset_index(drop=True)
        for season in EVALUATED_SEASONS
    }


def load_oof(
    rows: dict[int, pd.DataFrame],
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    basic_reference: dict[int, np.ndarray] = {}
    saturated_reference: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(
            BASE_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        base[season] = np.load(
            BASE_ROOT / f"predictions_all_prior_s1000_{season}.npy"
        ).astype(np.float64)
        basic_reference[season] = np.load(
            BASIC_REFERENCE_ROOT
            / f"predictions_lowrank_s300_r4_{season}.npy"
        ).astype(np.float64)
        saturated_reference[season] = np.load(
            SATURATED_REFERENCE_ROOT
            / f"predictions_team_pc_all_{season}.npy"
        ).astype(np.float64)
        csv_targets = rows[season]["control_success"].to_numpy(
            dtype=np.float64
        )
        lengths = {
            len(csv_targets),
            len(targets[season]),
            len(base[season]),
            len(basic_reference[season]),
            len(saturated_reference[season]),
        }
        if len(lengths) != 1 or not np.array_equal(
            csv_targets, targets[season]
        ):
            raise ValueError(f"OOF target/order mismatch for {season}")
        for label, predictions in (
            ("base", base[season]),
            ("basic_reference", basic_reference[season]),
            ("saturated_reference", saturated_reference[season]),
        ):
            if not np.isfinite(predictions).all() or not (
                (predictions >= 0.0).all()
                and (predictions <= 1.0).all()
            ):
                raise ValueError(
                    f"invalid {label} predictions for {season}"
                )
    return targets, base, basic_reference, saturated_reference


def objective_components(
    cell_means: np.ndarray,
    counts: np.ndarray,
    pitcher_bias: np.ndarray,
    context_bias: np.ndarray,
    pitcher_factors: np.ndarray,
    context_factors: np.ndarray,
    ridge: float,
) -> tuple[float, float, float]:
    fitted = (
        pitcher_bias[:, None]
        + context_bias[None, :]
        + pitcher_factors @ context_factors.T
    )
    error = cell_means - fitted
    data_loss = float(np.sum(counts * np.square(error)))
    regularization_loss = float(
        ridge
        * (
            np.square(pitcher_bias).sum()
            + np.square(context_bias).sum()
            + np.square(pitcher_factors).sum()
            + np.square(context_factors).sum()
        )
    )
    return data_loss, regularization_loss, data_loss + regularization_loss


def fit_weighted_als(
    cell_means: np.ndarray,
    counts: np.ndarray,
    ridge: float,
    rank: int,
) -> tuple[np.ndarray, dict[str, object]]:
    pitcher_count = counts.sum(axis=1)
    context_count = counts.sum(axis=0)
    pitcher_bias = np.zeros(counts.shape[0], dtype=np.float64)
    context_bias = np.zeros(counts.shape[1], dtype=np.float64)

    for _ in range(BIAS_INITIALIZATION_ITERATIONS):
        pitcher_bias = np.sum(
            counts * (cell_means - context_bias[None, :]), axis=1
        ) / (pitcher_count + ridge)
        context_bias = np.sum(
            counts * (cell_means - pitcher_bias[:, None]), axis=0
        ) / (context_count + ridge)

    observed = counts > 0
    bias_residual = (
        cell_means
        - pitcher_bias[:, None]
        - context_bias[None, :]
    )
    initialization_matrix = np.where(observed, bias_residual, 0.0)
    left, singular_values, right = np.linalg.svd(
        initialization_matrix, full_matrices=False
    )
    effective_rank = min(rank, len(singular_values))
    sqrt_singular = np.sqrt(singular_values[:effective_rank])
    pitcher_factors = left[:, :effective_rank] * sqrt_singular
    context_factors = right[:effective_rank, :].T * sqrt_singular
    if effective_rank < rank:
        pitcher_factors = np.pad(
            pitcher_factors, ((0, 0), (0, rank - effective_rank))
        )
        context_factors = np.pad(
            context_factors, ((0, 0), (0, rank - effective_rank))
        )

    data_loss, regularization_loss, total_objective = (
        objective_components(
            cell_means,
            counts,
            pitcher_bias,
            context_bias,
            pitcher_factors,
            context_factors,
            ridge,
        )
    )
    objective_history = [total_objective]
    data_loss_history = [data_loss]
    regularization_history = [regularization_loss]
    identity = np.eye(rank, dtype=np.float64)

    for _ in range(ALS_ITERATIONS):
        interaction = pitcher_factors @ context_factors.T
        pitcher_bias = np.sum(
            counts
            * (
                cell_means
                - context_bias[None, :]
                - interaction
            ),
            axis=1,
        ) / (pitcher_count + ridge)
        context_bias = np.sum(
            counts
            * (
                cell_means
                - pitcher_bias[:, None]
                - interaction
            ),
            axis=0,
        ) / (context_count + ridge)

        for pitcher_position in range(counts.shape[0]):
            weights = counts[pitcher_position]
            response = (
                cell_means[pitcher_position]
                - pitcher_bias[pitcher_position]
                - context_bias
            )
            normal = (
                (context_factors.T * weights) @ context_factors
                + ridge * identity
            )
            rhs = context_factors.T @ (weights * response)
            pitcher_factors[pitcher_position] = np.linalg.solve(
                normal, rhs
            )

        for context_position in range(counts.shape[1]):
            weights = counts[:, context_position]
            response = (
                cell_means[:, context_position]
                - pitcher_bias
                - context_bias[context_position]
            )
            normal = (
                (pitcher_factors.T * weights) @ pitcher_factors
                + ridge * identity
            )
            rhs = pitcher_factors.T @ (weights * response)
            context_factors[context_position] = np.linalg.solve(
                normal, rhs
            )

        data_loss, regularization_loss, total_objective = (
            objective_components(
                cell_means,
                counts,
                pitcher_bias,
                context_bias,
                pitcher_factors,
                context_factors,
                ridge,
            )
        )
        objective_history.append(total_objective)
        data_loss_history.append(data_loss)
        regularization_history.append(regularization_loss)

    objective_change = np.diff(objective_history)
    max_increase = float(max(0.0, objective_change.max()))
    if max_increase > OBJECTIVE_INCREASE_TOLERANCE:
        raise AssertionError(
            f"ALS objective increased: ridge={ridge} rank={rank} "
            f"increase={max_increase}"
        )
    fitted = (
        pitcher_bias[:, None]
        + context_bias[None, :]
        + pitcher_factors @ context_factors.T
    )
    if not np.isfinite(fitted).all():
        raise ValueError("non-finite ALS fitted matrix")
    observed_weight = float(counts.sum())
    diagnostics: dict[str, object] = {
        "ridge": ridge,
        "rank": rank,
        "bias_initialization_iterations": BIAS_INITIALIZATION_ITERATIONS,
        "als_iterations_completed": ALS_ITERATIONS,
        "objective_history": [float(value) for value in objective_history],
        "data_loss_history": [float(value) for value in data_loss_history],
        "regularization_loss_history": [
            float(value) for value in regularization_history
        ],
        "initial_objective": float(objective_history[0]),
        "final_objective": float(objective_history[-1]),
        "relative_objective_decrease": float(
            (objective_history[0] - objective_history[-1])
            / objective_history[0]
        ),
        "last_iteration_relative_decrease": float(
            (objective_history[-2] - objective_history[-1])
            / objective_history[-2]
        ),
        "max_objective_increase": max_increase,
        "objective_monotonic_within_tolerance": bool(
            max_increase <= OBJECTIVE_INCREASE_TOLERANCE
        ),
        "final_weighted_rmse": float(
            np.sqrt(data_loss_history[-1] / observed_weight)
        ),
        "pitcher_bias_l2_norm": float(np.linalg.norm(pitcher_bias)),
        "context_bias_l2_norm": float(np.linalg.norm(context_bias)),
        "pitcher_factor_l2_norm": float(
            np.linalg.norm(pitcher_factors)
        ),
        "context_factor_l2_norm": float(
            np.linalg.norm(context_factors)
        ),
        "interaction_frobenius_norm": float(
            np.linalg.norm(pitcher_factors @ context_factors.T)
        ),
        "fitted_mean_absolute_effect": float(np.abs(fitted).mean()),
        "fitted_max_absolute_effect": float(np.abs(fitted).max()),
        "initialization_singular_values": [
            float(value) for value in singular_values
        ],
    }
    return fitted, diagnostics


def fit_source_model(
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

    fitted_matrices: dict[tuple[float, int], np.ndarray] = {}
    candidate_diagnostics: dict[str, object] = {}
    for ridge in RIDGES:
        for rank in RANKS:
            candidate = als_name(ridge, rank)
            fitted, diagnostics = fit_weighted_als(
                cell_means,
                counts.astype(np.float64),
                ridge,
                rank,
            )
            fitted_matrices[(ridge, rank)] = fitted
            candidate_diagnostics[candidate] = diagnostics

    return {
        "source_season": source_season,
        "pitcher_ids": np.asarray(pitcher_ids),
        "counts": counts,
        "cell_means": cell_means,
        "fitted_matrices": fitted_matrices,
        "diagnostics": {
            "source_rows": int(len(source_rows)),
            "source_pitchers": int(len(pitcher_ids)),
            "matrix_shape": [matrix_shape[0], matrix_shape[1]],
            "matrix_cells": int(counts.size),
            "observed_cells": int((counts > 0).sum()),
            "observed_density": float((counts > 0).mean()),
            "cell_count_min_observed": int(counts[counts > 0].min()),
            "cell_count_median_observed": float(
                np.median(counts[counts > 0])
            ),
            "cell_count_max": int(counts.max()),
            "raw_residual_mean_before_centering": raw_residual_mean,
            "residual_mean_after_centering": float(residual.mean()),
            "candidates": candidate_diagnostics,
        },
    }


def map_source_model(
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

    corrections: dict[tuple[float, int], np.ndarray] = {}
    for ridge in RIDGES:
        for rank in RANKS:
            values = np.zeros(len(validation_rows), dtype=np.float64)
            fitted = source_model["fitted_matrices"][(ridge, rank)]
            values[pitcher_seen] = fitted[
                source_row_indices[pitcher_seen],
                context_positions[pitcher_seen],
            ]
            if np.any(values[~pitcher_seen] != 0.0):
                raise AssertionError("unseen pitcher received correction")
            corrections[(ridge, rank)] = values
    return {
        "pitcher_seen": pitcher_seen,
        "exact_context_seen": exact_context_seen,
        "corrections": corrections,
    }


def summarize_coverage(
    mapped_sources: dict[int, dict[str, object]],
    row_count: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    if not mapped_sources:
        zero = np.zeros(row_count, dtype=np.int8)
        return (
            {
                "source_count": 0,
                "rows": row_count,
                "pitcher_seen_any_source_rows": 0,
                "pitcher_seen_any_source_rate": 0.0,
                "exact_context_seen_any_source_rows": 0,
                "exact_context_seen_any_source_rate": 0.0,
                "unseen_pitcher_all_sources_rows": row_count,
                "unseen_pitcher_all_sources_rate": 1.0,
                "per_source": {},
            },
            zero,
            zero.copy(),
        )
    pitcher_matrix = np.vstack(
        [mapped["pitcher_seen"] for mapped in mapped_sources.values()]
    )
    exact_matrix = np.vstack(
        [mapped["exact_context_seen"] for mapped in mapped_sources.values()]
    )
    pitcher_count = pitcher_matrix.sum(axis=0).astype(np.int8)
    exact_count = exact_matrix.sum(axis=0).astype(np.int8)
    pitcher_any = pitcher_count > 0
    exact_any = exact_count > 0
    if np.any(exact_count > pitcher_count):
        raise AssertionError("exact coverage exceeds pitcher coverage")
    return (
        {
            "source_count": int(len(mapped_sources)),
            "rows": row_count,
            "pitcher_seen_any_source_rows": int(pitcher_any.sum()),
            "pitcher_seen_any_source_rate": float(pitcher_any.mean()),
            "pitcher_seen_every_source_rows": int(
                (pitcher_count == len(mapped_sources)).sum()
            ),
            "pitcher_seen_every_source_rate": float(
                (pitcher_count == len(mapped_sources)).mean()
            ),
            "exact_context_seen_any_source_rows": int(exact_any.sum()),
            "exact_context_seen_any_source_rate": float(exact_any.mean()),
            "exact_context_seen_every_source_rows": int(
                (exact_count == len(mapped_sources)).sum()
            ),
            "exact_context_seen_every_source_rate": float(
                (exact_count == len(mapped_sources)).mean()
            ),
            "pitcher_seen_but_never_exact_rows": int(
                (pitcher_any & ~exact_any).sum()
            ),
            "pitcher_seen_but_never_exact_rate": float(
                (pitcher_any & ~exact_any).mean()
            ),
            "unseen_pitcher_all_sources_rows": int((~pitcher_any).sum()),
            "unseen_pitcher_all_sources_rate": float(
                (~pitcher_any).mean()
            ),
            "per_source": {
                str(source_season): {
                    "pitcher_seen_rows": int(
                        mapped["pitcher_seen"].sum()
                    ),
                    "pitcher_seen_rate": float(
                        mapped["pitcher_seen"].mean()
                    ),
                    "exact_context_seen_rows": int(
                        mapped["exact_context_seen"].sum()
                    ),
                    "exact_context_seen_rate": float(
                        mapped["exact_context_seen"].mean()
                    ),
                }
                for source_season, mapped in mapped_sources.items()
            },
        },
        pitcher_count,
        exact_count,
    )


def correction_diagnostics(correction: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(correction.mean()),
        "standard_deviation": float(correction.std()),
        "mean_absolute": float(np.abs(correction).mean()),
        "min": float(correction.min()),
        "max": float(correction.max()),
        "nonzero_rows": int(np.count_nonzero(correction)),
        "nonzero_rate": float(np.count_nonzero(correction) / len(correction)),
    }


def aggregate_metrics(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for candidate in CANDIDATES:
        fold_metrics = {
            season: folds[str(season)]["candidates"][candidate]
            for season in REPORT_SEASONS
        }
        briers = {
            season: float(metrics["brier_score"])
            for season, metrics in fold_metrics.items()
        }
        skills = {
            season: float(metrics["skill_score_unclipped"])
            for season, metrics in fold_metrics.items()
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

    references = {
        "base": BASE_CANDIDATE,
        "basic_svd_r4": BASIC_REFERENCE,
        "saturated_s600": SATURATED_REFERENCE,
    }
    for candidate in ALS_CANDIDATES:
        summary = aggregate[candidate]
        for label, reference in references.items():
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
        summary["beats_base_every_report_season"] = bool(
            all(
                value > 0.0
                for value in summary[
                    "season_skill_change_vs_base"
                ].values()
            )
        )
        summary["beats_basic_svd_r4_every_report_season"] = bool(
            all(
                value > 0.0
                for value in summary[
                    "season_skill_change_vs_basic_svd_r4"
                ].values()
            )
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
            source_models[source_season] = fit_source_model(
                source_season,
                rows[source_season],
                targets[source_season],
                base[source_season],
            )
        return source_models[source_season]

    earliest_source_model = get_source_model(EVALUATED_SEASONS[0])
    scale_audit_diagnostics: dict[str, object] = {}
    for rejected_ridge in SCALE_AUDIT_RIDGES:
        _, audit_diagnostics = fit_weighted_als(
            earliest_source_model["cell_means"],
            earliest_source_model["counts"].astype(np.float64),
            rejected_ridge,
            SCALE_AUDIT_RANK,
        )
        scale_audit_diagnostics[str(int(rejected_ridge))] = (
            audit_diagnostics
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        source_seasons = [
            season
            for season in EVALUATED_SEASONS
            if season < validation_season
        ]
        mapped_sources = {
            source_season: map_source_model(
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
        for ridge in RIDGES:
            for rank in RANKS:
                candidate = als_name(ridge, rank)
                if mapped_sources:
                    correction = np.mean(
                        np.vstack(
                            [
                                mapped["corrections"][(ridge, rank)]
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
                if not np.isfinite(prediction).all() or not (
                    (prediction >= 0.0).all()
                    and (prediction <= 1.0).all()
                ):
                    raise ValueError(
                        f"invalid prediction {validation_season} {candidate}"
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
            f"weighted_als {validation_season}: "
            + " ".join(
                f"{candidate}="
                f"{metrics[candidate]['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    aggregate = aggregate_metrics(folds)
    best_mean = max(
        ALS_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["mean_skill"],
            aggregate[candidate]["min_skill"],
            -ALS_CANDIDATES.index(candidate),
        ),
    )
    best_min = max(
        ALS_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["min_skill"],
            aggregate[candidate]["latest_2024_skill"],
            aggregate[candidate]["mean_skill"],
            -ALS_CANDIDATES.index(candidate),
        ),
    )
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": (
            "count_weighted_bias_plus_ALS_pitcher_context_atop_team_OOF"
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
            "candidate_grid_predeclared_before_validation_results": True,
            "candidate_comparison_nested": False,
            "r_specific_candidate_predeclared": False,
            "post_result_tuning": False,
        },
        "predeclared_configuration": {
            "ridges": list(RIDGES),
            "ranks": list(RANKS),
            "candidate_count": len(ALS_CANDIDATES),
            "bias_initialization_iterations": (
                BIAS_INITIALIZATION_ITERATIONS
            ),
            "als_iterations": ALS_ITERATIONS,
            "objective": (
                "sum_observed cell_count*(cell_mean-fitted)^2 + "
                "ridge*(all bias/factor squared norms)"
            ),
            "model": (
                "pitcher_bias + context_bias + pitcher_factor dot "
                "context_factor"
            ),
            "initialization": (
                "ridge two-way bias initialization then deterministic SVD "
                "of observed bias residual with missing cells zero"
            ),
            "ridge_scale_audit": {
                "used_report_season_validation_labels_or_results": False,
                "used_only_earliest_prior_source_training_objective": True,
                "earliest_source_oof_season": EVALUATED_SEASONS[0],
                "rejected_grid": list(SCALE_AUDIT_RIDGES),
                "audit_rank": SCALE_AUDIT_RANK,
                "reason": (
                    "earliest-source training-objective audit showed "
                    "symmetric ridge >=150 collapses interaction to "
                    "numerical zero"
                ),
                "final_grid_fixed_before_validation": list(RIDGES),
                "machine_generated_diagnostics": scale_audit_diagnostics,
            },
            "context_domain": [
                {
                    "position": position,
                    "count_index": context[0],
                    "batter_hand": context[1],
                }
                for position, context in enumerate(CONTEXTS)
            ],
            "context_count": len(CONTEXTS),
            "correction_weight": 1.0,
        },
        "source_model_diagnostics": {
            str(source_season): source_model["diagnostics"]
            for source_season, source_model in source_models.items()
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "diagnostic only; candidate comparison is post-hoc",
            "posthoc_best_mean_candidate": best_mean,
            "posthoc_best_min_candidate": best_min,
            "best_mean_beats_base_every_report_season": bool(
                aggregate[best_mean]["beats_base_every_report_season"]
            ),
            "best_mean_beats_basic_svd_r4_every_report_season": bool(
                aggregate[best_mean][
                    "beats_basic_svd_r4_every_report_season"
                ]
            ),
            "best_min_exceeds_1100": bool(
                aggregate[best_min]["min_skill"] >= 1100.0
            ),
            "stop_family_if_best_min_below_1100": bool(
                aggregate[best_min]["min_skill"] < 1100.0
            ),
            "adopt_without_nested_confirmation": False,
        },
        "qa": {
            "source_target_and_row_order_alignment_checked": True,
            "source_residual_centering_checked": True,
            "source_season_order_checked": True,
            "static_context_domain_checked": True,
            "count_weighted_observed_cell_objective_checked": True,
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
