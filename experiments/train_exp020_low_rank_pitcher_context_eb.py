"""EXP-020: low-rank pitcher-by-count-context EB atop team OOF.

The immutable base is the temporal-safe EXP-019 team ``all_prior_s1000``
OOF prediction.  For every outer validation season, one empirical-Bayes
pitcher-by-context matrix is fitted independently for each earlier OOF
season.  The residual is centered inside its source season before the
pitcher_id x (count_index, batter_hand) sum/count matrix is constructed.

Strong EB smoothing is applied before a deterministic truncated SVD.  Rank-2
and rank-4 reconstructions let a previously seen pitcher borrow information
across the 24 statically declared count/hand contexts.  An unseen pitcher
always receives zero.  Source-season corrections are averaged equally,
including zero for a pitcher absent from a source season, matching the
existing saturated pctx_s600 reference protocol.

No current-fold label, test row, validation-row aggregate, raw prediction
array derived from test rows, or post-result tuning is used.  The small grid
of two smoothing strengths and two ranks is declared below.  Its comparison
is still reported as non-nested/post-hoc.
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
SATURATED_REFERENCE_ROOT = Path(
    "./artifacts/EXP-020/pitcher_count_eb_atop_team"
)
ARTIFACT_DIR = Path(
    "./artifacts/EXP-020/low_rank_pitcher_context_eb"
)

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
SMOOTHING_GRID = (300.0, 600.0)
REFERENCE_RANKS = (2, 4)
EXTENSION_RANKS = (6, 8, 12)
RANK_GRID = (*REFERENCE_RANKS, *EXTENSION_RANKS)
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
SATURATED_CANDIDATE = "saturated_pctx_s600_reference"
LOW_RANK_CANDIDATES = tuple(
    f"lowrank_s{int(smoothing)}_r{rank}"
    for smoothing in SMOOTHING_GRID
    for rank in RANK_GRID
)
R_SPECIFIC_CANDIDATE = "lowrank_s300_r4_Rspecific"
CANDIDATES = (
    BASE_CANDIDATE,
    SATURATED_CANDIDATE,
    *LOW_RANK_CANDIDATES,
    R_SPECIFIC_CANDIDATE,
)
EFFECT_CANDIDATES = (*LOW_RANK_CANDIDATES, R_SPECIFIC_CANDIDATE)


def low_rank_name(smoothing: float, rank: int) -> str:
    return f"lowrank_s{int(smoothing)}_r{rank}"


def load_rows() -> dict[int, pd.DataFrame]:
    columns = [
        "season",
        "pitcher_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "game_type",
        "control_success",
    ]
    frame = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=columns,
    )
    frame = frame.loc[frame["season"].isin(EVALUATED_SEASONS)].copy()
    frame["count_index"] = (
        frame["balls_before"].to_numpy(dtype=np.int8) * 4
        + frame["strikes_before"].to_numpy(dtype=np.int8)
    ).astype(np.int8)

    observed_counts = set(
        frame["count_index"].dropna().astype(int).unique().tolist()
    )
    observed_hands = set(
        frame["batter_hand"].dropna().astype(int).unique().tolist()
    )
    if not observed_counts.issubset(set(COUNT_INDICES)):
        raise ValueError(f"unexpected count_index values: {observed_counts}")
    if not observed_hands.issubset(set(BATTER_HANDS)):
        raise ValueError(f"unexpected batter_hand values: {observed_hands}")
    if frame[
        [
            "pitcher_id",
            "count_index",
            "batter_hand",
            "game_type",
            "control_success",
        ]
    ].isna().any().any():
        raise ValueError("missing required pitcher-context field")
    if set(frame["game_type"].astype(str).unique()) != {"F", "R"}:
        raise ValueError("unexpected game_type domain")

    frame["context_position"] = [
        CONTEXT_TO_POSITION[(int(count_index), int(batter_hand))]
        for count_index, batter_hand in zip(
            frame["count_index"],
            frame["batter_hand"],
            strict=True,
        )
    ]
    frame["context_position"] = frame["context_position"].astype(np.int8)
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
]:
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    saturated_reference: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(
            BASE_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        base[season] = np.load(
            BASE_ROOT / f"predictions_all_prior_s1000_{season}.npy"
        ).astype(np.float64)
        saturated_reference[season] = np.load(
            SATURATED_REFERENCE_ROOT
            / f"predictions_team_pc_all_{season}.npy"
        ).astype(np.float64)
        csv_targets = rows[season]["control_success"].to_numpy(
            dtype=np.float64
        )
        if not (
            len(csv_targets)
            == len(targets[season])
            == len(base[season])
            == len(saturated_reference[season])
            and np.array_equal(csv_targets, targets[season])
        ):
            raise ValueError(f"OOF target/order mismatch for {season}")
        for label, predictions in (
            ("base", base[season]),
            ("saturated", saturated_reference[season]),
        ):
            if not np.isfinite(predictions).all() or not (
                (predictions >= 0.0).all()
                and (predictions <= 1.0).all()
            ):
                raise ValueError(f"invalid {label} predictions for {season}")
    return targets, base, saturated_reference


def fit_source_matrix(
    source_season: int,
    source_rows: pd.DataFrame,
    targets: np.ndarray,
    base: np.ndarray,
    smoothing_grid: tuple[float, ...] = SMOOTHING_GRID,
    rank_grid: tuple[int, ...] = RANK_GRID,
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
    observation_counts = np.zeros(matrix_shape, dtype=np.int64)
    np.add.at(
        residual_sums,
        (pitcher_codes, context_positions),
        residual,
    )
    np.add.at(
        observation_counts,
        (pitcher_codes, context_positions),
        1,
    )
    if int(observation_counts.sum()) != len(source_rows):
        raise AssertionError("source matrix row count mismatch")

    matrices: dict[float, dict[str, object]] = {}
    smoothing_diagnostics: dict[str, object] = {}
    for smoothing in smoothing_grid:
        saturated = residual_sums / (
            observation_counts.astype(np.float64) + smoothing
        )
        left, singular_values, right = np.linalg.svd(
            saturated,
            full_matrices=False,
        )
        reconstructions: dict[int, np.ndarray] = {}
        total_energy = float(np.square(singular_values).sum())
        rank_energy: dict[str, float] = {}
        for rank in rank_grid:
            effective_rank = min(rank, len(singular_values))
            reconstructions[rank] = (
                left[:, :effective_rank]
                * singular_values[:effective_rank]
            ) @ right[:effective_rank, :]
            retained = float(
                np.square(singular_values[:effective_rank]).sum()
            )
            rank_energy[str(rank)] = (
                retained / total_energy if total_energy > 0.0 else 0.0
            )
        matrices[smoothing] = {
            "saturated": saturated,
            "reconstructions": reconstructions,
        }
        smoothing_diagnostics[str(int(smoothing))] = {
            "singular_values": [
                float(value) for value in singular_values
            ],
            "matrix_frobenius_norm": float(np.sqrt(total_energy)),
            "retained_energy_fraction": rank_energy,
            "saturated_mean_absolute_effect": float(
                np.abs(saturated).mean()
            ),
            "saturated_max_absolute_effect": float(
                np.abs(saturated).max()
            ),
        }

    return {
        "source_season": source_season,
        "pitcher_ids": np.asarray(pitcher_ids),
        "observation_counts": observation_counts,
        "matrices": matrices,
        "diagnostics": {
            "source_rows": int(len(source_rows)),
            "source_pitchers": int(len(pitcher_ids)),
            "observed_pitcher_context_cells": int(
                (observation_counts > 0).sum()
            ),
            "matrix_cells": int(observation_counts.size),
            "matrix_observed_density": float(
                (observation_counts > 0).mean()
            ),
            "raw_residual_mean_before_centering": raw_residual_mean,
            "residual_mean_after_centering": float(residual.mean()),
            "smoothing": smoothing_diagnostics,
        },
    }


def map_source_matrix(
    source_model: dict[str, object],
    validation_rows: pd.DataFrame,
) -> dict[str, object]:
    pitcher_ids = source_model["pitcher_ids"]
    source_row_indices = pd.Index(pitcher_ids).get_indexer(
        validation_rows["pitcher_id"]
    )
    context_positions = validation_rows["context_position"].to_numpy(
        dtype=np.int16
    )
    pitcher_seen = source_row_indices >= 0
    safe_rows = np.where(pitcher_seen, source_row_indices, 0)
    observation_counts = source_model["observation_counts"]
    exact_context_seen = pitcher_seen & (
        observation_counts[safe_rows, context_positions] > 0
    )

    saturated_values: dict[float, np.ndarray] = {}
    low_rank_values: dict[tuple[float, int], np.ndarray] = {}
    for smoothing, matrix_details in source_model["matrices"].items():
        saturated_matrix = matrix_details[
            "saturated"
        ]
        saturated = np.zeros(len(validation_rows), dtype=np.float64)
        saturated[pitcher_seen] = saturated_matrix[
            source_row_indices[pitcher_seen],
            context_positions[pitcher_seen],
        ]
        saturated_values[smoothing] = saturated
        for rank, reconstruction in matrix_details[
            "reconstructions"
        ].items():
            values = np.zeros(len(validation_rows), dtype=np.float64)
            values[pitcher_seen] = reconstruction[
                source_row_indices[pitcher_seen],
                context_positions[pitcher_seen],
            ]
            low_rank_values[(smoothing, rank)] = values

    return {
        "pitcher_seen": pitcher_seen,
        "exact_context_seen": exact_context_seen,
        "saturated_values": saturated_values,
        "low_rank_values": low_rank_values,
    }


def summarize_coverage(
    mapped_sources: dict[int, dict[str, object]],
    row_count: int,
) -> dict[str, object]:
    if not mapped_sources:
        return {
            "source_count": 0,
            "rows": row_count,
            "pitcher_seen_any_source_rows": 0,
            "pitcher_seen_any_source_rate": 0.0,
            "pitcher_seen_every_source_rows": 0,
            "pitcher_seen_every_source_rate": 0.0,
            "exact_context_seen_any_source_rows": 0,
            "exact_context_seen_any_source_rate": 0.0,
            "exact_context_seen_every_source_rows": 0,
            "exact_context_seen_every_source_rate": 0.0,
            "shared_only_any_source_rows": 0,
            "shared_only_any_source_rate": 0.0,
            "per_source": {},
        }

    pitcher_matrix = np.vstack(
        [mapped["pitcher_seen"] for mapped in mapped_sources.values()]
    )
    exact_matrix = np.vstack(
        [mapped["exact_context_seen"] for mapped in mapped_sources.values()]
    )
    pitcher_any = pitcher_matrix.any(axis=0)
    pitcher_all = pitcher_matrix.all(axis=0)
    exact_any = exact_matrix.any(axis=0)
    exact_all = exact_matrix.all(axis=0)
    shared_only_any = np.any(pitcher_matrix & ~exact_matrix, axis=0)
    return {
        "source_count": int(len(mapped_sources)),
        "rows": row_count,
        "pitcher_seen_any_source_rows": int(pitcher_any.sum()),
        "pitcher_seen_any_source_rate": float(pitcher_any.mean()),
        "pitcher_seen_every_source_rows": int(pitcher_all.sum()),
        "pitcher_seen_every_source_rate": float(pitcher_all.mean()),
        "exact_context_seen_any_source_rows": int(exact_any.sum()),
        "exact_context_seen_any_source_rate": float(exact_any.mean()),
        "exact_context_seen_every_source_rows": int(exact_all.sum()),
        "exact_context_seen_every_source_rate": float(exact_all.mean()),
        "shared_only_any_source_rows": int(shared_only_any.sum()),
        "shared_only_any_source_rate": float(shared_only_any.mean()),
        "per_source": {
            str(source_season): {
                "pitcher_seen_rows": int(mapped["pitcher_seen"].sum()),
                "pitcher_seen_rate": float(mapped["pitcher_seen"].mean()),
                "exact_context_seen_rows": int(
                    mapped["exact_context_seen"].sum()
                ),
                "exact_context_seen_rate": float(
                    mapped["exact_context_seen"].mean()
                ),
                "shared_only_rows": int(
                    (
                        mapped["pitcher_seen"]
                        & ~mapped["exact_context_seen"]
                    ).sum()
                ),
                "shared_only_rate": float(
                    np.mean(
                        mapped["pitcher_seen"]
                        & ~mapped["exact_context_seen"]
                    )
                ),
            }
            for source_season, mapped in mapped_sources.items()
        },
    }


def aggregate_metrics(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for candidate in CANDIDATES:
        fold_metrics = {
            season: folds[str(season)]["candidates"][candidate]
            for season in REPORT_SEASONS
        }
        skills = {
            season: float(metrics["skill_score_unclipped"])
            for season, metrics in fold_metrics.items()
        }
        briers = {
            season: float(metrics["brier_score"])
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

    base = aggregate[BASE_CANDIDATE]
    saturated = aggregate[SATURATED_CANDIDATE]
    for candidate in EFFECT_CANDIDATES:
        current = aggregate[candidate]
        current["season_skill_change_vs_base"] = {
            str(season): float(
                current["season_skills"][str(season)]
                - base["season_skills"][str(season)]
            )
            for season in REPORT_SEASONS
        }
        current["season_skill_change_vs_saturated_s600"] = {
            str(season): float(
                current["season_skills"][str(season)]
                - saturated["season_skills"][str(season)]
            )
            for season in REPORT_SEASONS
        }
        current["mean_skill_change_vs_base"] = float(
            current["mean_skill"] - base["mean_skill"]
        )
        current["min_skill_change_vs_base"] = float(
            current["min_skill"] - base["min_skill"]
        )
        current["mean_skill_change_vs_saturated_s600"] = float(
            current["mean_skill"] - saturated["mean_skill"]
        )
        current["min_skill_change_vs_saturated_s600"] = float(
            current["min_skill"] - saturated["min_skill"]
        )
    return aggregate


def select_rank_from_prior_folds(
    smoothing: float,
    validation_season: int,
    folds: dict[str, object],
) -> tuple[int, list[int], dict[str, object]]:
    """Select rank using only earlier OOF folds for one fixed smoothing."""
    history = [
        season
        for season in EVALUATED_SEASONS
        if season < validation_season
    ]
    if not history:
        return RANK_GRID[0], [], {}

    selection_metrics: dict[str, object] = {}
    for rank in RANK_GRID:
        candidate = low_rank_name(smoothing, rank)
        skills = {
            season: float(
                folds[str(season)]["candidates"][candidate][
                    "skill_score_unclipped"
                ]
            )
            for season in history
        }
        selection_metrics[str(rank)] = {
            "candidate": candidate,
            "history_skills": {
                str(season): value for season, value in skills.items()
            },
            "history_worst_skill": float(min(skills.values())),
            "history_mean_skill": float(np.mean(list(skills.values()))),
        }
    selected = max(
        RANK_GRID,
        key=lambda rank: (
            selection_metrics[str(rank)]["history_worst_skill"],
            selection_metrics[str(rank)]["history_mean_skill"],
            -RANK_GRID.index(rank),
        ),
    )
    return selected, history, selection_metrics


def build_strict_rank_paths(
    folds: dict[str, object],
    prediction_cache: dict[int, dict[str, np.ndarray]],
    targets: dict[int, np.ndarray],
) -> dict[str, object]:
    strict_paths: dict[str, object] = {}
    for smoothing in SMOOTHING_GRID:
        path_name = f"strict_rank_s{int(smoothing)}"
        strict_folds: dict[str, object] = {}
        for validation_season in EVALUATED_SEASONS:
            selected_rank, history, selection_metrics = (
                select_rank_from_prior_folds(
                    smoothing, validation_season, folds
                )
            )
            selected_candidate = low_rank_name(
                smoothing, selected_rank
            )
            predictions = prediction_cache[validation_season][
                selected_candidate
            ]
            metrics = calculate_metrics(
                targets[validation_season], predictions
            )
            strict_folds[str(validation_season)] = {
                "validation_season": validation_season,
                "selection_history_seasons": history,
                "selected_rank": selected_rank,
                "selected_candidate": selected_candidate,
                "selection_objective": (
                    "maximum worst prior-fold Skill; then prior-fold mean; "
                    "then lower rank"
                ),
                "selection_metrics": selection_metrics,
                "current_fold_metrics_used_for_selection": False,
                "metrics": metrics,
            }
            np.save(
                ARTIFACT_DIR
                / f"predictions_{path_name}_{validation_season}.npy",
                predictions,
            )

        report_skills = {
            season: float(
                strict_folds[str(season)]["metrics"][
                    "skill_score_unclipped"
                ]
            )
            for season in REPORT_SEASONS
        }
        report_briers = {
            season: float(
                strict_folds[str(season)]["metrics"]["brier_score"]
            )
            for season in REPORT_SEASONS
        }
        next_rank, next_history, next_selection_metrics = (
            select_rank_from_prior_folds(smoothing, 2025, folds)
        )
        strict_paths[path_name] = {
            "fixed_smoothing": smoothing,
            "rank_candidates": list(RANK_GRID),
            "folds": strict_folds,
            "aggregate_2022_2024": {
                "season_briers": {
                    str(season): value
                    for season, value in report_briers.items()
                },
                "season_skills": {
                    str(season): value
                    for season, value in report_skills.items()
                },
                "mean_skill": float(np.mean(list(report_skills.values()))),
                "min_skill": float(np.min(list(report_skills.values()))),
                "latest_2024_skill": report_skills[2024],
                "selection_path": {
                    str(season): strict_folds[str(season)][
                        "selected_rank"
                    ]
                    for season in REPORT_SEASONS
                },
            },
            "next_2025_selection": {
                "selection_history_seasons": next_history,
                "selected_rank": next_rank,
                "selected_candidate": low_rank_name(
                    smoothing, next_rank
                ),
                "selection_metrics": next_selection_metrics,
                "uses_2025_labels": False,
            },
        }
    return strict_paths


def main() -> None:
    started = time.time()
    rows = load_rows()
    targets, base, saturated_reference = load_oof(rows)
    source_models: dict[int, dict[str, object]] = {}
    r_specific_source_models: dict[int, dict[str, object]] = {}

    def get_source_model(source_season: int) -> dict[str, object]:
        if source_season not in source_models:
            source_models[source_season] = fit_source_matrix(
                source_season,
                rows[source_season],
                targets[source_season],
                base[source_season],
            )
        return source_models[source_season]

    def get_r_specific_source_model(
        source_season: int,
    ) -> dict[str, object]:
        if source_season not in r_specific_source_models:
            source_rows = rows[source_season]
            source_is_r = (
                source_rows["game_type"].astype(str).to_numpy() == "R"
            )
            r_specific_source_models[source_season] = fit_source_matrix(
                source_season,
                source_rows.loc[source_is_r].reset_index(drop=True),
                targets[source_season][source_is_r],
                base[source_season][source_is_r],
                smoothing_grid=(300.0,),
                rank_grid=(4,),
            )
        return r_specific_source_models[source_season]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    prediction_cache: dict[int, dict[str, np.ndarray]] = {}
    for validation_season in EVALUATED_SEASONS:
        source_seasons = [
            season
            for season in EVALUATED_SEASONS
            if season < validation_season
        ]
        mapped_sources: dict[int, dict[str, object]] = {}
        for source_season in source_seasons:
            mapped_sources[source_season] = map_source_matrix(
                get_source_model(source_season),
                rows[validation_season],
            )
        validation_is_r = (
            rows[validation_season]["game_type"].astype(str).to_numpy()
            == "R"
        )
        validation_r_rows = rows[validation_season].loc[
            validation_is_r
        ].reset_index(drop=True)
        r_specific_mapped_sources = {
            source_season: map_source_matrix(
                get_r_specific_source_model(source_season),
                validation_r_rows,
            )
            for source_season in source_seasons
        }

        if mapped_sources:
            recomputed_saturated_correction = np.mean(
                np.vstack(
                    [
                        mapped["saturated_values"][600.0]
                        for mapped in mapped_sources.values()
                    ]
                ),
                axis=0,
            )
        else:
            recomputed_saturated_correction = np.zeros(
                len(rows[validation_season]), dtype=np.float64
            )
        recomputed_saturated_prediction = np.clip(
            base[validation_season] + recomputed_saturated_correction,
            0.0,
            1.0,
        )
        saturated_difference = float(
            np.max(
                np.abs(
                    recomputed_saturated_prediction
                    - saturated_reference[validation_season]
                )
            )
        )
        if saturated_difference > 1e-12:
            raise AssertionError(
                "saturated pctx_s600 reconstruction mismatch: "
                f"{validation_season} {saturated_difference}"
            )

        predictions: dict[str, np.ndarray] = {
            BASE_CANDIDATE: base[validation_season].copy(),
            SATURATED_CANDIDATE: saturated_reference[
                validation_season
            ].copy(),
        }
        correction_diagnostics: dict[str, object] = {}
        for smoothing in SMOOTHING_GRID:
            for rank in RANK_GRID:
                candidate = low_rank_name(smoothing, rank)
                if mapped_sources:
                    correction = np.mean(
                        np.vstack(
                            [
                                mapped["low_rank_values"][
                                    (smoothing, rank)
                                ]
                                for mapped in mapped_sources.values()
                            ]
                        ),
                        axis=0,
                    )
                else:
                    correction = np.zeros(
                        len(rows[validation_season]), dtype=np.float64
                    )
                predictions[candidate] = np.clip(
                    base[validation_season] + correction,
                    0.0,
                    1.0,
                )
                correction_diagnostics[candidate] = {
                    "mean": float(correction.mean()),
                    "standard_deviation": float(correction.std()),
                    "mean_absolute": float(np.abs(correction).mean()),
                    "min": float(correction.min()),
                    "max": float(correction.max()),
                }

        r_specific_correction = np.zeros(
            len(rows[validation_season]), dtype=np.float64
        )
        if r_specific_mapped_sources:
            r_specific_correction[validation_is_r] = np.mean(
                np.vstack(
                    [
                        mapped["low_rank_values"][(300.0, 4)]
                        for mapped in r_specific_mapped_sources.values()
                    ]
                ),
                axis=0,
            )
        predictions[R_SPECIFIC_CANDIDATE] = np.clip(
            base[validation_season] + r_specific_correction,
            0.0,
            1.0,
        )
        if not np.array_equal(
            predictions[R_SPECIFIC_CANDIDATE][~validation_is_r],
            base[validation_season][~validation_is_r],
        ):
            raise AssertionError("R-specific F-row base invariant failed")
        correction_diagnostics[R_SPECIFIC_CANDIDATE] = {
            "source_training_rows": "R only",
            "application_rows": "R only; F exact base",
            "mean_all_rows": float(r_specific_correction.mean()),
            "mean_R_rows": float(
                r_specific_correction[validation_is_r].mean()
            ),
            "standard_deviation_R_rows": float(
                r_specific_correction[validation_is_r].std()
            ),
            "mean_absolute_R_rows": float(
                np.abs(r_specific_correction[validation_is_r]).mean()
            ),
        }

        if tuple(predictions) != CANDIDATES:
            raise AssertionError("candidate declaration/order drift")
        prediction_cache[validation_season] = predictions
        metrics = {
            candidate: calculate_metrics(
                targets[validation_season], candidate_predictions
            )
            for candidate, candidate_predictions in predictions.items()
        }
        for candidate, candidate_predictions in predictions.items():
            if not np.isfinite(candidate_predictions).all() or not (
                (candidate_predictions >= 0.0).all()
                and (candidate_predictions <= 1.0).all()
            ):
                raise ValueError(
                    f"invalid prediction {validation_season} {candidate}"
                )
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate}_{validation_season}.npy",
                candidate_predictions,
            )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets[validation_season].astype(np.int8),
        )

        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "rows": int(len(rows[validation_season])),
            "coverage": summarize_coverage(
                mapped_sources, len(rows[validation_season])
            ),
            "r_specific_coverage_R_rows": summarize_coverage(
                r_specific_mapped_sources, int(validation_is_r.sum())
            ),
            "R_rows": int(validation_is_r.sum()),
            "F_rows": int((~validation_is_r).sum()),
            "saturated_reference_reconstruction_max_abs_difference": (
                saturated_difference
            ),
            "correction_diagnostics": correction_diagnostics,
            "candidates": metrics,
            "strict_source_invariants": {
                "all_source_seasons_precede_validation": bool(
                    all(
                        source_season < validation_season
                        for source_season in source_seasons
                    )
                ),
                "current_fold_labels_used_for_effect_fit": False,
                "current_fold_or_test_rows_used_for_aggregation": False,
                "validation_rows_used_only_for_key_mapping": True,
                "source_residuals_centered_within_season": True,
                "static_context_domain_used": True,
                "unseen_pitcher_correction_is_zero": True,
                "R_specific_source_training_R_only": True,
                "R_specific_application_R_only": True,
                "R_specific_F_predictions_equal_base": True,
                "saturated_reference_reproduced": bool(
                    saturated_difference <= 1e-12
                ),
            },
        }
        print(
            f"lowrank_pctx {validation_season}: "
            + " ".join(
                f"{candidate}="
                f"{metrics[candidate]['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    aggregate = aggregate_metrics(folds)
    strict_rank_paths = build_strict_rank_paths(
        folds, prediction_cache, targets
    )
    best_low_rank_mean = max(
        LOW_RANK_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["mean_skill"],
            aggregate[candidate]["min_skill"],
            -LOW_RANK_CANDIDATES.index(candidate),
        ),
    )
    best_low_rank_min = max(
        LOW_RANK_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["min_skill"],
            aggregate[candidate]["latest_2024_skill"],
            aggregate[candidate]["mean_skill"],
            -LOW_RANK_CANDIDATES.index(candidate),
        ),
    )
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": "low_rank_pitcher_context_EB_atop_team_allprior",
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": (
                "EXP-019 team_eb_ensemble all_prior_s1000 OOF"
            ),
            "effect_target": (
                "source-season-centered target minus immutable base OOF"
            ),
            "matrix": (
                "one pitcher_id x static (count_index,batter_hand) EB "
                "matrix per earlier source OOF season"
            ),
            "source_season_combination": (
                "equal mean of source-season corrections; absent pitcher "
                "contributes zero"
            ),
            "current_fold_labels_used_for_effect_fit": False,
            "validation_or_test_row_aggregation": False,
            "candidate_grid_predeclared": False,
            "rank_extension_grid_predeclared_before_extension_run": True,
            "candidate_comparison_nested": False,
            "strict_rank_selection_uses_current_fold": False,
            "R_specific_variant_posthoc_motivated": True,
            "R_specific_variant": (
                "source R rows only, source-R residual centering, apply to "
                "validation R only, F exact immutable base"
            ),
        },
        "predeclared_configuration": {
            "smoothing_grid": list(SMOOTHING_GRID),
            "rank_grid": list(RANK_GRID),
            "reference_ranks": list(REFERENCE_RANKS),
            "bounded_extension_ranks": list(EXTENSION_RANKS),
            "R_specific": {
                "status": (
                    "post-hoc bounded diagnostic motivated by prior "
                    "saturated R-only segment analysis"
                ),
                "smoothing": 300.0,
                "rank": 4,
                "source_game_type": "R",
                "application_game_type": "R",
                "F_prediction": "immutable base exactly",
            },
            "context_domain_source": (
                "static official count domain: balls 0..3, strikes 0..2, "
                "batter_hand in {1,2}"
            ),
            "contexts": [
                {
                    "position": position,
                    "count_index": count_index,
                    "batter_hand": batter_hand,
                }
                for position, (count_index, batter_hand) in enumerate(
                    CONTEXTS
                )
            ],
            "context_count": len(CONTEXTS),
            "matrix_centering": (
                "no matrix-column centering; residual already centered "
                "inside source season"
            ),
            "decomposition": "deterministic full SVD then truncated rank",
            "correction_weight": 1.0,
            "saturated_reference": (
                "saved EXP-020 pitcher_count_eb_atop_team team_pc_all, "
                "smoothing=600"
            ),
        },
        "source_matrix_diagnostics": {
            str(source_season): source_model["diagnostics"]
            for source_season, source_model in source_models.items()
        },
        "R_specific_source_matrix_diagnostics": {
            str(source_season): source_model["diagnostics"]
            for source_season, source_model in (
                r_specific_source_models.items()
            )
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "strict_previous_fold_rank_selection": {
            "selection_scope": (
                "rank selected separately inside each fixed smoothing"
            ),
            "objective": (
                "maximize worst earlier-OOF-fold Skill; then earlier-fold "
                "mean Skill; then lower rank"
            ),
            "current_fold_metrics_used": False,
            "paths": strict_rank_paths,
        },
        "selection": {
            "status": "diagnostic only; candidate comparison is non-nested",
            "posthoc_best_low_rank_mean_candidate": best_low_rank_mean,
            "posthoc_best_low_rank_min_candidate": best_low_rank_min,
            "best_mean_beats_base": bool(
                aggregate[best_low_rank_mean]["mean_skill"]
                > aggregate[BASE_CANDIDATE]["mean_skill"]
            ),
            "best_min_beats_base": bool(
                aggregate[best_low_rank_min]["min_skill"]
                > aggregate[BASE_CANDIDATE]["min_skill"]
            ),
            "best_min_beats_saturated_s600": bool(
                aggregate[best_low_rank_min]["min_skill"]
                > aggregate[SATURATED_CANDIDATE]["min_skill"]
            ),
            "R_specific": {
                "candidate": R_SPECIFIC_CANDIDATE,
                "mean_skill": aggregate[R_SPECIFIC_CANDIDATE][
                    "mean_skill"
                ],
                "min_skill": aggregate[R_SPECIFIC_CANDIDATE][
                    "min_skill"
                ],
                "mean_skill_change_vs_base": aggregate[
                    R_SPECIFIC_CANDIDATE
                ]["mean_skill_change_vs_base"],
                "min_skill_change_vs_base": aggregate[
                    R_SPECIFIC_CANDIDATE
                ]["min_skill_change_vs_base"],
            },
            "adopt_without_nested_confirmation": False,
        },
        "qa": {
            "source_target_and_row_order_alignment_checked": True,
            "saved_saturated_reference_reconstructed": True,
            "source_residual_centering_checked": True,
            "source_season_order_checked": True,
            "static_context_domain_checked": True,
            "prediction_probability_ranges_checked": True,
            "saved_prediction_arrays": True,
            "R_specific_F_base_equality_checked": True,
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
