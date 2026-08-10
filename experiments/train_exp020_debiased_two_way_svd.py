"""EXP-020: debiased two-way SVD pitcher-context residual correction.

This final bounded factorization diagnostic tests whether the basic zero-filled
SVD wastes low-rank capacity on pitcher and context main effects.  For every
earlier OOF source season, a strongly regularized two-way pitcher/context bias
model is fitted first to the source-season-centered residual.  Those main
effects are removed at the cell-sum level, and only the remaining interaction
matrix is EB-smoothed and decomposed by rank-2/rank-4 SVD.

The four candidates are fixed before validation results:

* bias scale 0.25 or 0.50
* interaction rank 2 or 4

Interaction weight is one.  Source seasons are averaged equally.  For a
pitcher unseen in a source season, the source-only global context main effect
is explicitly allowed, while pitcher main and interaction are exactly zero.
The existing basic and R-specific rank-4 OOF arrays are read-only references;
no R-only candidate is trained here.

No current-fold label, validation/test-row aggregate, test row, or post-result
tuning is used.  Validation rows provide only current-row official keys.
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
REFERENCE_ROOT = Path(
    "./artifacts/EXP-020/low_rank_pitcher_context_eb"
)
ARTIFACT_DIR = Path(
    "./artifacts/EXP-020/debiased_two_way_svd"
)

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
MAIN_RIDGE = 600.0
MAIN_BIAS_ITERATIONS = 20
INTERACTION_SMOOTHING = 300.0
BIAS_SCALES = (0.25, 0.50)
INTERACTION_RANKS = (2, 4)

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
R_SPECIFIC_REFERENCE = "basic_svd_s300_r4_Rspecific_reference"
DEBIASED_CANDIDATES = tuple(
    f"debiased_bias{int(scale * 100):03d}_rank{rank}"
    for scale in BIAS_SCALES
    for rank in INTERACTION_RANKS
)
CANDIDATES = (
    BASE_CANDIDATE,
    BASIC_REFERENCE,
    R_SPECIFIC_REFERENCE,
    *DEBIASED_CANDIDATES,
)


def candidate_name(bias_scale: float, rank: int) -> str:
    return f"debiased_bias{int(bias_scale * 100):03d}_rank{rank}"


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
    required = [
        "pitcher_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "control_success",
    ]
    if frame[required].isna().any().any():
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
    r_specific_reference: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(
            BASE_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        base[season] = np.load(
            BASE_ROOT / f"predictions_all_prior_s1000_{season}.npy"
        ).astype(np.float64)
        basic_reference[season] = np.load(
            REFERENCE_ROOT
            / f"predictions_lowrank_s300_r4_{season}.npy"
        ).astype(np.float64)
        r_specific_reference[season] = np.load(
            REFERENCE_ROOT
            / f"predictions_lowrank_s300_r4_Rspecific_{season}.npy"
        ).astype(np.float64)
        csv_targets = rows[season]["control_success"].to_numpy(
            dtype=np.float64
        )
        lengths = {
            len(csv_targets),
            len(targets[season]),
            len(base[season]),
            len(basic_reference[season]),
            len(r_specific_reference[season]),
        }
        if len(lengths) != 1 or not np.array_equal(
            csv_targets, targets[season]
        ):
            raise ValueError(f"OOF target/order mismatch for {season}")
        for label, predictions in (
            ("base", base[season]),
            ("basic_reference", basic_reference[season]),
            ("r_specific_reference", r_specific_reference[season]),
        ):
            if not np.isfinite(predictions).all() or not (
                (predictions >= 0.0).all()
                and (predictions <= 1.0).all()
            ):
                raise ValueError(
                    f"invalid {label} predictions for {season}"
                )
    return targets, base, basic_reference, r_specific_reference


def bias_objective(
    cell_means: np.ndarray,
    counts: np.ndarray,
    pitcher_bias: np.ndarray,
    context_bias: np.ndarray,
) -> tuple[float, float, float]:
    error = (
        cell_means
        - pitcher_bias[:, None]
        - context_bias[None, :]
    )
    data_loss = float(np.sum(counts * np.square(error)))
    ridge_loss = float(
        MAIN_RIDGE
        * (
            np.square(pitcher_bias).sum()
            + np.square(context_bias).sum()
        )
    )
    return data_loss, ridge_loss, data_loss + ridge_loss


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
    if (pitcher_codes < 0).any():
        raise ValueError(f"missing pitcher ID in source {source_season}")
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
    count_float = counts.astype(np.float64)
    pitcher_total = count_float.sum(axis=1)
    context_total = count_float.sum(axis=0)
    pitcher_bias = np.zeros(len(pitcher_ids), dtype=np.float64)
    context_bias = np.zeros(len(CONTEXTS), dtype=np.float64)
    data_loss, ridge_loss, total_objective = bias_objective(
        cell_means,
        count_float,
        pitcher_bias,
        context_bias,
    )
    objective_history = [total_objective]
    data_loss_history = [data_loss]
    ridge_loss_history = [ridge_loss]
    for _ in range(MAIN_BIAS_ITERATIONS):
        pitcher_bias = np.sum(
            residual_sums - count_float * context_bias[None, :],
            axis=1,
        ) / (pitcher_total + MAIN_RIDGE)
        context_bias = np.sum(
            residual_sums - count_float * pitcher_bias[:, None],
            axis=0,
        ) / (context_total + MAIN_RIDGE)
        data_loss, ridge_loss, total_objective = bias_objective(
            cell_means,
            count_float,
            pitcher_bias,
            context_bias,
        )
        objective_history.append(total_objective)
        data_loss_history.append(data_loss)
        ridge_loss_history.append(ridge_loss)
    if np.any(np.diff(objective_history) > 1e-9):
        raise AssertionError("two-way bias objective increased")

    interaction_sums = residual_sums - count_float * (
        pitcher_bias[:, None] + context_bias[None, :]
    )
    smoothed_interaction = interaction_sums / (
        count_float + INTERACTION_SMOOTHING
    )
    left, singular_values, right = np.linalg.svd(
        smoothed_interaction, full_matrices=False
    )
    total_energy = float(np.square(singular_values).sum())
    interaction_reconstructions: dict[int, np.ndarray] = {}
    retained_energy: dict[str, float] = {}
    for rank in INTERACTION_RANKS:
        effective_rank = min(rank, len(singular_values))
        reconstruction = (
            left[:, :effective_rank]
            * singular_values[:effective_rank]
        ) @ right[:effective_rank, :]
        interaction_reconstructions[rank] = reconstruction
        retained = float(
            np.square(singular_values[:effective_rank]).sum()
        )
        retained_energy[str(rank)] = (
            retained / total_energy if total_energy > 0.0 else 0.0
        )
    if not all(
        np.isfinite(matrix).all()
        for matrix in interaction_reconstructions.values()
    ):
        raise ValueError("non-finite interaction reconstruction")

    return {
        "source_season": source_season,
        "pitcher_ids": np.asarray(pitcher_ids),
        "counts": counts,
        "pitcher_bias": pitcher_bias,
        "context_bias": context_bias,
        "interaction_reconstructions": interaction_reconstructions,
        "diagnostics": {
            "source_rows": int(len(source_rows)),
            "source_pitchers": int(len(pitcher_ids)),
            "matrix_shape": list(matrix_shape),
            "matrix_cells": int(counts.size),
            "observed_cells": int((counts > 0).sum()),
            "observed_density": float((counts > 0).mean()),
            "raw_residual_mean_before_centering": raw_residual_mean,
            "residual_mean_after_centering": float(residual.mean()),
            "main_effects": {
                "ridge": MAIN_RIDGE,
                "iterations_completed": MAIN_BIAS_ITERATIONS,
                "objective_history": [
                    float(value) for value in objective_history
                ],
                "data_loss_history": [
                    float(value) for value in data_loss_history
                ],
                "ridge_loss_history": [
                    float(value) for value in ridge_loss_history
                ],
                "max_objective_increase": float(
                    max(0.0, np.diff(objective_history).max())
                ),
                "last_iteration_relative_decrease": float(
                    (objective_history[-2] - objective_history[-1])
                    / objective_history[-2]
                ),
                "pitcher_bias_l2_norm": float(
                    np.linalg.norm(pitcher_bias)
                ),
                "context_bias_l2_norm": float(
                    np.linalg.norm(context_bias)
                ),
                "pitcher_bias_mean_absolute": float(
                    np.abs(pitcher_bias).mean()
                ),
                "context_bias_mean_absolute": float(
                    np.abs(context_bias).mean()
                ),
            },
            "interaction": {
                "smoothing": INTERACTION_SMOOTHING,
                "residual_sum_after_main_effect_removal": float(
                    interaction_sums.sum()
                ),
                "smoothed_frobenius_norm": float(
                    np.linalg.norm(smoothed_interaction)
                ),
                "smoothed_mean_absolute_effect": float(
                    np.abs(smoothed_interaction).mean()
                ),
                "smoothed_max_absolute_effect": float(
                    np.abs(smoothed_interaction).max()
                ),
                "singular_values": [
                    float(value) for value in singular_values
                ],
                "retained_energy_fraction": retained_energy,
                "reconstruction_frobenius_norm": {
                    str(rank): float(
                        np.linalg.norm(
                            interaction_reconstructions[rank]
                        )
                    )
                    for rank in INTERACTION_RANKS
                },
            },
        },
    }


def map_source_model(
    source_model: dict[str, object],
    validation_rows: pd.DataFrame,
) -> dict[str, object]:
    source_row_indices = pd.Index(
        source_model["pitcher_ids"]
    ).get_indexer(validation_rows["pitcher_id"])
    context_positions = validation_rows["context_position"].to_numpy(
        dtype=np.int16
    )
    pitcher_seen = source_row_indices >= 0
    safe_rows = np.where(pitcher_seen, source_row_indices, 0)
    exact_context_seen = pitcher_seen & (
        source_model["counts"][safe_rows, context_positions] > 0
    )
    context_main = source_model["context_bias"][context_positions]
    pitcher_main = np.zeros(len(validation_rows), dtype=np.float64)
    pitcher_main[pitcher_seen] = source_model["pitcher_bias"][
        source_row_indices[pitcher_seen]
    ]
    interactions: dict[int, np.ndarray] = {}
    for rank in INTERACTION_RANKS:
        values = np.zeros(len(validation_rows), dtype=np.float64)
        reconstruction = source_model["interaction_reconstructions"][rank]
        values[pitcher_seen] = reconstruction[
            source_row_indices[pitcher_seen],
            context_positions[pitcher_seen],
        ]
        interactions[rank] = values
    if np.any(pitcher_main[~pitcher_seen] != 0.0):
        raise AssertionError("unseen pitcher main effect nonzero")
    if any(
        np.any(values[~pitcher_seen] != 0.0)
        for values in interactions.values()
    ):
        raise AssertionError("unseen pitcher interaction nonzero")
    return {
        "pitcher_seen": pitcher_seen,
        "exact_context_seen": exact_context_seen,
        "context_main": context_main,
        "pitcher_main": pitcher_main,
        "interactions": interactions,
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
                "global_context_main_coverage_rows": 0,
                "global_context_main_coverage_rate": 0.0,
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
            "global_context_main_coverage_rows": row_count,
            "global_context_main_coverage_rate": 1.0,
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
            "unseen_pitcher_all_sources_rows": int((~pitcher_any).sum()),
            "unseen_pitcher_all_sources_rate": float(
                (~pitcher_any).mean()
            ),
            "per_source": {
                str(source_season): {
                    "global_context_main_rows": row_count,
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


def component_diagnostics(component: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(component.mean()),
        "standard_deviation": float(component.std()),
        "l2_norm": float(np.linalg.norm(component)),
        "mean_absolute": float(np.abs(component).mean()),
        "min": float(component.min()),
        "max": float(component.max()),
        "nonzero_rows": int(np.count_nonzero(component)),
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
    for candidate in DEBIASED_CANDIDATES:
        summary = aggregate[candidate]
        for label, reference in (
            ("base", BASE_CANDIDATE),
            ("basic_r4", BASIC_REFERENCE),
            ("r_specific_r4", R_SPECIFIC_REFERENCE),
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
        summary["beats_base_every_report_season"] = bool(
            all(
                value > 0.0
                for value in summary[
                    "season_skill_change_vs_base"
                ].values()
            )
        )
        summary["beats_basic_r4_every_report_season"] = bool(
            all(
                value > 0.0
                for value in summary[
                    "season_skill_change_vs_basic_r4"
                ].values()
            )
        )
        summary["beats_r_specific_r4_every_report_season"] = bool(
            all(
                value > 0.0
                for value in summary[
                    "season_skill_change_vs_r_specific_r4"
                ].values()
            )
        )
    return aggregate


def main() -> None:
    started = time.time()
    rows = load_rows()
    targets, base, basic_reference, r_specific_reference = load_oof(rows)
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
        if mapped_sources:
            context_main = np.mean(
                np.vstack(
                    [
                        mapped["context_main"]
                        for mapped in mapped_sources.values()
                    ]
                ),
                axis=0,
            )
            pitcher_main = np.mean(
                np.vstack(
                    [
                        mapped["pitcher_main"]
                        for mapped in mapped_sources.values()
                    ]
                ),
                axis=0,
            )
            interactions = {
                rank: np.mean(
                    np.vstack(
                        [
                            mapped["interactions"][rank]
                            for mapped in mapped_sources.values()
                        ]
                    ),
                    axis=0,
                )
                for rank in INTERACTION_RANKS
            }
        else:
            context_main = np.zeros(
                len(rows[validation_season]), dtype=np.float64
            )
            pitcher_main = np.zeros_like(context_main)
            interactions = {
                rank: np.zeros_like(context_main)
                for rank in INTERACTION_RANKS
            }
        if np.any(pitcher_main[pitcher_source_count == 0] != 0.0):
            raise AssertionError("unseen pitcher averaged main nonzero")
        for interaction in interactions.values():
            if np.any(interaction[pitcher_source_count == 0] != 0.0):
                raise AssertionError(
                    "unseen pitcher averaged interaction nonzero"
                )

        bias_main = context_main + pitcher_main
        predictions: dict[str, np.ndarray] = {
            BASE_CANDIDATE: base[validation_season].copy(),
            BASIC_REFERENCE: basic_reference[validation_season].copy(),
            R_SPECIFIC_REFERENCE: r_specific_reference[
                validation_season
            ].copy(),
        }
        corrections: dict[str, np.ndarray] = {}
        for bias_scale in BIAS_SCALES:
            for rank in INTERACTION_RANKS:
                candidate = candidate_name(bias_scale, rank)
                correction = (
                    bias_scale * bias_main + interactions[rank]
                )
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
                unseen = pitcher_source_count == 0
                expected_unseen = bias_scale * context_main[unseen]
                if not np.array_equal(
                    correction[unseen], expected_unseen
                ):
                    raise AssertionError(
                        "unseen pitcher context-only rule failed"
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
            / f"context_main_component_{validation_season}.npy",
            context_main,
        )
        np.save(
            ARTIFACT_DIR
            / f"pitcher_main_component_{validation_season}.npy",
            pitcher_main,
        )
        for rank, interaction in interactions.items():
            np.save(
                ARTIFACT_DIR
                / f"interaction_component_rank{rank}_{validation_season}.npy",
                interaction,
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
            "component_diagnostics": {
                "context_main": component_diagnostics(context_main),
                "pitcher_main": component_diagnostics(pitcher_main),
                "combined_unscaled_main": component_diagnostics(bias_main),
                **{
                    f"interaction_rank{rank}": component_diagnostics(
                        interaction
                    )
                    for rank, interaction in interactions.items()
                },
                **{
                    candidate: component_diagnostics(correction)
                    for candidate, correction in corrections.items()
                },
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
                "unseen_pitcher_global_context_main_allowed": True,
                "unseen_pitcher_pitcher_main_is_zero": True,
                "unseen_pitcher_interaction_is_zero": True,
                "r_specific_model_trained": False,
            },
        }
        print(
            f"debiased_svd {validation_season}: "
            + " ".join(
                f"{candidate}="
                f"{metrics[candidate]['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    aggregate = aggregate_metrics(folds)
    best_mean = max(
        DEBIASED_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["mean_skill"],
            aggregate[candidate]["min_skill"],
            -DEBIASED_CANDIDATES.index(candidate),
        ),
    )
    best_min = max(
        DEBIASED_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["min_skill"],
            aggregate[candidate]["latest_2024_skill"],
            aggregate[candidate]["mean_skill"],
            -DEBIASED_CANDIDATES.index(candidate),
        ),
    )
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": (
            "debiased_two_way_main_effect_plus_interaction_SVD"
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
                "equal mean of earlier source components"
            ),
            "current_fold_labels_used_for_fit": False,
            "validation_or_test_row_aggregation": False,
            "validation_current_row_keys_only": True,
            "test_csv_read": False,
            "candidate_grid_predeclared": True,
            "candidate_comparison_nested": False,
            "r_specific_model_trained_or_created": False,
        },
        "predeclared_configuration": {
            "main_ridge": MAIN_RIDGE,
            "main_bias_iterations": MAIN_BIAS_ITERATIONS,
            "interaction_smoothing": INTERACTION_SMOOTHING,
            "bias_scales": list(BIAS_SCALES),
            "interaction_ranks": list(INTERACTION_RANKS),
            "candidate_count": len(DEBIASED_CANDIDATES),
            "interaction_scale": 1.0,
            "model": (
                "bias_scale*(source context main + source pitcher main) + "
                "rank-r SVD of EB-smoothed residual interaction"
            ),
            "unseen_pitcher_rule": (
                "allow source-only global context main; pitcher main and "
                "interaction are zero"
            ),
            "context_domain": [
                {
                    "position": position,
                    "count_index": context[0],
                    "batter_hand": context[1],
                }
                for position, context in enumerate(CONTEXTS)
            ],
            "basic_reference": (
                "saved low_rank_pitcher_context_eb lowrank_s300_r4 OOF"
            ),
            "r_specific_reference": (
                "saved low_rank_pitcher_context_eb "
                "lowrank_s300_r4_Rspecific OOF; read-only"
            ),
        },
        "source_model_diagnostics": {
            str(source_season): source_model["diagnostics"]
            for source_season, source_model in source_models.items()
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "diagnostic only; comparison is non-nested/post-hoc",
            "posthoc_best_mean_candidate": best_mean,
            "posthoc_best_min_candidate": best_min,
            "best_mean_beats_base_every_report_season": bool(
                aggregate[best_mean]["beats_base_every_report_season"]
            ),
            "best_mean_beats_basic_r4_every_report_season": bool(
                aggregate[best_mean][
                    "beats_basic_r4_every_report_season"
                ]
            ),
            "best_mean_beats_r_specific_r4_every_report_season": bool(
                aggregate[best_mean][
                    "beats_r_specific_r4_every_report_season"
                ]
            ),
            "best_min_exceeds_1100": bool(
                aggregate[best_min]["min_skill"] >= 1100.0
            ),
            "stop_factorization_branch": bool(
                aggregate[best_min]["min_skill"] < 1100.0
            ),
            "adopt_without_nested_confirmation": False,
        },
        "qa": {
            "source_target_and_row_order_alignment_checked": True,
            "source_residual_centering_checked": True,
            "source_season_order_checked": True,
            "main_bias_objective_monotonicity_checked": True,
            "component_decomposition_saved": True,
            "static_context_domain_checked": True,
            "prediction_probability_ranges_checked": True,
            "unseen_pitcher_context_only_rule_checked": True,
            "saved_prediction_correction_component_coverage_arrays": True,
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
