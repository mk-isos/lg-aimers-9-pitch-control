"""EXP-020: richer low-rank pitcher-context EB atop temporal team OOF.

This bounded diagnostic extends the already evaluated pitcher_id x
(count_index, batter_hand) low-rank matrix along one official current-row
axis at a time:

* ``outs``: pitcher_id x (count_index, batter_hand, outs_before)
* ``runners``: pitcher_id x (count_index, batter_hand, runner-count bucket)

The runner bucket is statically defined as 0, 1, or 2+ occupied bases.  Each
matrix is evaluated independently; richer effects are never added together
or added to the basic low-rank effect.  For every outer validation season,
each earlier OOF season supplies a separate, source-season-centered residual
matrix.  Strong EB smoothing is applied before deterministic rank-4/rank-8
SVD reconstruction, and source-season corrections are averaged equally.
An absent pitcher contributes zero for that source season.

The immutable base and all residual targets are saved temporal OOF arrays.
Current-fold labels, validation/test aggregates, test rows, raw player IDs as
model inputs, and post-result tuning are not used.  Validation rows supply
only their current-row pitcher and statically mapped official context.
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
ARTIFACT_DIR = Path(
    "./artifacts/EXP-020/rich_low_rank_pitcher_context_eb"
)

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
SMOOTHING = 300.0
RANKS = (4, 8)
BATTER_HANDS = (1, 2)
OUT_VALUES = (0, 1, 2)
RUNNER_BUCKETS = (0, 1, 2)
COUNT_INDICES = tuple(
    balls * 4 + strikes
    for balls in range(4)
    for strikes in range(3)
)

OUTS_CONTEXTS = tuple(
    (count_index, batter_hand, outs_before)
    for count_index in COUNT_INDICES
    for batter_hand in BATTER_HANDS
    for outs_before in OUT_VALUES
)
RUNNER_CONTEXTS = tuple(
    (count_index, batter_hand, runner_bucket)
    for count_index in COUNT_INDICES
    for batter_hand in BATTER_HANDS
    for runner_bucket in RUNNER_BUCKETS
)
CONTEXTS = {
    "outs": OUTS_CONTEXTS,
    "runners": RUNNER_CONTEXTS,
}
POSITION_COLUMNS = {
    "outs": "outs_context_position",
    "runners": "runner_context_position",
}
CONTEXT_TO_POSITION = {
    spec: {
        context: position
        for position, context in enumerate(contexts)
    }
    for spec, contexts in CONTEXTS.items()
}

BASE_CANDIDATE = "base_team_all_prior"
BASIC_REFERENCE = "basic_lowrank_s300_r4_reference"
RICH_CANDIDATES = tuple(
    f"{spec}_s300_r{rank}"
    for spec in ("outs", "runners")
    for rank in RANKS
)
CANDIDATES = (BASE_CANDIDATE, BASIC_REFERENCE, *RICH_CANDIDATES)


def candidate_name(spec: str, rank: int) -> str:
    return f"{spec}_s300_r{rank}"


def load_rows() -> dict[int, pd.DataFrame]:
    columns = [
        "season",
        "pitcher_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "outs_before",
        "runner_on_1b",
        "runner_on_2b",
        "runner_on_3b",
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
        "outs_before",
        "runner_on_1b",
        "runner_on_2b",
        "runner_on_3b",
        "control_success",
    ]
    if frame[required].isna().any().any():
        raise ValueError("missing required pitcher-context field")

    frame["count_index"] = (
        frame["balls_before"].to_numpy(dtype=np.int8) * 4
        + frame["strikes_before"].to_numpy(dtype=np.int8)
    ).astype(np.int8)
    runner_columns = ["runner_on_1b", "runner_on_2b", "runner_on_3b"]
    runner_count = frame[runner_columns].to_numpy(dtype=np.int8).sum(axis=1)
    frame["runner_bucket"] = np.minimum(runner_count, 2).astype(np.int8)

    observed_counts = set(frame["count_index"].astype(int).unique())
    observed_hands = set(frame["batter_hand"].astype(int).unique())
    observed_outs = set(frame["outs_before"].astype(int).unique())
    observed_runner_flags = {
        int(value)
        for column in runner_columns
        for value in frame[column].unique()
    }
    observed_runner_buckets = set(
        frame["runner_bucket"].astype(int).unique()
    )
    if not observed_counts.issubset(set(COUNT_INDICES)):
        raise ValueError(f"unexpected count_index: {observed_counts}")
    if not observed_hands.issubset(set(BATTER_HANDS)):
        raise ValueError(f"unexpected batter_hand: {observed_hands}")
    if not observed_outs.issubset(set(OUT_VALUES)):
        raise ValueError(f"unexpected outs_before: {observed_outs}")
    if not observed_runner_flags.issubset({0, 1}):
        raise ValueError(
            f"unexpected runner flag values: {observed_runner_flags}"
        )
    if not observed_runner_buckets.issubset(set(RUNNER_BUCKETS)):
        raise ValueError(
            f"unexpected runner buckets: {observed_runner_buckets}"
        )

    frame[POSITION_COLUMNS["outs"]] = [
        CONTEXT_TO_POSITION["outs"][
            (int(count_index), int(batter_hand), int(outs_before))
        ]
        for count_index, batter_hand, outs_before in zip(
            frame["count_index"],
            frame["batter_hand"],
            frame["outs_before"],
            strict=True,
        )
    ]
    frame[POSITION_COLUMNS["runners"]] = [
        CONTEXT_TO_POSITION["runners"][
            (int(count_index), int(batter_hand), int(runner_bucket))
        ]
        for count_index, batter_hand, runner_bucket in zip(
            frame["count_index"],
            frame["batter_hand"],
            frame["runner_bucket"],
            strict=True,
        )
    ]
    for position_column in POSITION_COLUMNS.values():
        frame[position_column] = frame[position_column].astype(np.int8)

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
    basic_reference: dict[int, np.ndarray] = {}
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
        csv_targets = rows[season]["control_success"].to_numpy(
            dtype=np.float64
        )
        lengths = {
            len(csv_targets),
            len(targets[season]),
            len(base[season]),
            len(basic_reference[season]),
        }
        if len(lengths) != 1 or not np.array_equal(
            csv_targets, targets[season]
        ):
            raise ValueError(f"OOF target/order mismatch for {season}")
        for label, predictions in (
            ("base", base[season]),
            ("basic_reference", basic_reference[season]),
        ):
            if not np.isfinite(predictions).all() or not (
                (predictions >= 0.0).all()
                and (predictions <= 1.0).all()
            ):
                raise ValueError(
                    f"invalid {label} predictions for {season}"
                )
    return targets, base, basic_reference


def fit_source_models(
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

    spec_models: dict[str, object] = {}
    for spec, contexts in CONTEXTS.items():
        context_positions = source_rows[POSITION_COLUMNS[spec]].to_numpy(
            dtype=np.int16
        )
        matrix_shape = (len(pitcher_ids), len(contexts))
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
            raise AssertionError(
                f"source matrix row mismatch {source_season} {spec}"
            )

        smoothed = residual_sums / (
            observation_counts.astype(np.float64) + SMOOTHING
        )
        left, singular_values, right = np.linalg.svd(
            smoothed,
            full_matrices=False,
        )
        total_energy = float(np.square(singular_values).sum())
        reconstructions: dict[int, np.ndarray] = {}
        retained_energy: dict[str, float] = {}
        for rank in RANKS:
            effective_rank = min(rank, len(singular_values))
            reconstruction = (
                left[:, :effective_rank]
                * singular_values[:effective_rank]
            ) @ right[:effective_rank, :]
            reconstructions[rank] = reconstruction
            retained = float(
                np.square(singular_values[:effective_rank]).sum()
            )
            retained_energy[str(rank)] = (
                retained / total_energy if total_energy > 0.0 else 0.0
            )

        if not all(
            np.isfinite(matrix).all()
            for matrix in reconstructions.values()
        ):
            raise ValueError(
                f"non-finite reconstruction {source_season} {spec}"
            )
        if not np.all(np.diff(singular_values) <= 1e-12):
            raise AssertionError("singular values not descending")

        spec_models[spec] = {
            "observation_counts": observation_counts,
            "reconstructions": reconstructions,
            "diagnostics": {
                "source_rows": int(len(source_rows)),
                "source_pitchers": int(len(pitcher_ids)),
                "matrix_shape": [
                    int(matrix_shape[0]), int(matrix_shape[1])
                ],
                "matrix_cells": int(observation_counts.size),
                "observed_pitcher_context_cells": int(
                    (observation_counts > 0).sum()
                ),
                "matrix_observed_density": float(
                    (observation_counts > 0).mean()
                ),
                "singular_values": [
                    float(value) for value in singular_values
                ],
                "top_8_singular_values": [
                    float(value) for value in singular_values[:8]
                ],
                "matrix_frobenius_norm": float(np.sqrt(total_energy)),
                "retained_energy_fraction": retained_energy,
                "smoothed_mean_absolute_effect": float(
                    np.abs(smoothed).mean()
                ),
                "smoothed_max_absolute_effect": float(
                    np.abs(smoothed).max()
                ),
            },
        }

    return {
        "source_season": source_season,
        "pitcher_ids": np.asarray(pitcher_ids),
        "spec_models": spec_models,
        "diagnostics": {
            "source_rows": int(len(source_rows)),
            "source_pitchers": int(len(pitcher_ids)),
            "raw_residual_mean_before_centering": raw_residual_mean,
            "residual_mean_after_centering": float(residual.mean()),
            "specs": {
                spec: spec_models[spec]["diagnostics"]
                for spec in CONTEXTS
            },
        },
    }


def map_source_model(
    source_model: dict[str, object],
    validation_rows: pd.DataFrame,
    spec: str,
) -> dict[str, object]:
    pitcher_ids = source_model["pitcher_ids"]
    source_row_indices = pd.Index(pitcher_ids).get_indexer(
        validation_rows["pitcher_id"]
    )
    pitcher_seen = source_row_indices >= 0
    safe_rows = np.where(pitcher_seen, source_row_indices, 0)
    context_positions = validation_rows[POSITION_COLUMNS[spec]].to_numpy(
        dtype=np.int16
    )
    spec_model = source_model["spec_models"][spec]
    observation_counts = spec_model["observation_counts"]
    exact_context_seen = pitcher_seen & (
        observation_counts[safe_rows, context_positions] > 0
    )

    corrections: dict[int, np.ndarray] = {}
    for rank in RANKS:
        values = np.zeros(len(validation_rows), dtype=np.float64)
        reconstruction = spec_model["reconstructions"][rank]
        values[pitcher_seen] = reconstruction[
            source_row_indices[pitcher_seen],
            context_positions[pitcher_seen],
        ]
        if np.any(values[~pitcher_seen] != 0.0):
            raise AssertionError("unseen pitcher received correction")
        corrections[rank] = values

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
        empty_counts = np.zeros(row_count, dtype=np.int8)
        return (
            {
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
                "pitcher_seen_but_never_exact_rows": 0,
                "pitcher_seen_but_never_exact_rate": 0.0,
                "unseen_pitcher_all_sources_rows": row_count,
                "unseen_pitcher_all_sources_rate": 1.0,
                "per_source": {},
            },
            empty_counts,
            empty_counts.copy(),
        )

    pitcher_matrix = np.vstack(
        [mapped["pitcher_seen"] for mapped in mapped_sources.values()]
    )
    exact_matrix = np.vstack(
        [mapped["exact_context_seen"] for mapped in mapped_sources.values()]
    )
    pitcher_source_count = pitcher_matrix.sum(axis=0).astype(np.int8)
    exact_source_count = exact_matrix.sum(axis=0).astype(np.int8)
    source_count = len(mapped_sources)
    pitcher_any = pitcher_source_count > 0
    exact_any = exact_source_count > 0
    pitcher_every = pitcher_source_count == source_count
    exact_every = exact_source_count == source_count
    pitcher_seen_but_never_exact = pitcher_any & ~exact_any
    unseen_pitcher = ~pitcher_any
    summary = {
        "source_count": int(source_count),
        "rows": row_count,
        "pitcher_seen_any_source_rows": int(pitcher_any.sum()),
        "pitcher_seen_any_source_rate": float(pitcher_any.mean()),
        "pitcher_seen_every_source_rows": int(pitcher_every.sum()),
        "pitcher_seen_every_source_rate": float(pitcher_every.mean()),
        "exact_context_seen_any_source_rows": int(exact_any.sum()),
        "exact_context_seen_any_source_rate": float(exact_any.mean()),
        "exact_context_seen_every_source_rows": int(exact_every.sum()),
        "exact_context_seen_every_source_rate": float(exact_every.mean()),
        "pitcher_seen_but_never_exact_rows": int(
            pitcher_seen_but_never_exact.sum()
        ),
        "pitcher_seen_but_never_exact_rate": float(
            pitcher_seen_but_never_exact.mean()
        ),
        "unseen_pitcher_all_sources_rows": int(unseen_pitcher.sum()),
        "unseen_pitcher_all_sources_rate": float(unseen_pitcher.mean()),
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
                "pitcher_seen_exact_unseen_rows": int(
                    (
                        mapped["pitcher_seen"]
                        & ~mapped["exact_context_seen"]
                    ).sum()
                ),
            }
            for source_season, mapped in mapped_sources.items()
        },
    }
    if np.any(exact_source_count > pitcher_source_count):
        raise AssertionError("exact coverage exceeds pitcher coverage")
    return summary, pitcher_source_count, exact_source_count


def correction_diagnostics(correction: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(correction.mean()),
        "standard_deviation": float(correction.std()),
        "mean_absolute": float(np.abs(correction).mean()),
        "min": float(correction.min()),
        "max": float(correction.max()),
        "nonzero_rows": int(np.count_nonzero(correction)),
        "nonzero_rate": float(np.count_nonzero(correction) / len(correction)),
    }


def calculate_segment_metrics(
    targets: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    pitcher_source_count: np.ndarray,
    exact_source_count: np.ndarray,
) -> dict[str, object]:
    masks = {
        "exact_seen_any_source": exact_source_count > 0,
        "pitcher_seen_but_never_exact": (
            (pitcher_source_count > 0) & (exact_source_count == 0)
        ),
        "unseen_pitcher_all_sources": pitcher_source_count == 0,
    }
    segments: dict[str, object] = {}
    for segment, mask in masks.items():
        if int(mask.sum()) == 0:
            segments[segment] = {"rows": 0, "base": None, "candidate": None}
        else:
            segments[segment] = {
                "rows": int(mask.sum()),
                "base": calculate_metrics(targets[mask], base[mask]),
                "candidate": calculate_metrics(
                    targets[mask], candidate[mask]
                ),
            }
    return segments


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
            "latest_2024_skill": float(skills[2024]),
        }

    base_summary = aggregate[BASE_CANDIDATE]
    basic_summary = aggregate[BASIC_REFERENCE]
    for candidate in RICH_CANDIDATES:
        summary = aggregate[candidate]
        summary["season_skill_change_vs_base"] = {
            str(season): float(
                summary["season_skills"][str(season)]
                - base_summary["season_skills"][str(season)]
            )
            for season in REPORT_SEASONS
        }
        summary["season_skill_change_vs_basic_r4"] = {
            str(season): float(
                summary["season_skills"][str(season)]
                - basic_summary["season_skills"][str(season)]
            )
            for season in REPORT_SEASONS
        }
        summary["mean_skill_change_vs_base"] = float(
            summary["mean_skill"] - base_summary["mean_skill"]
        )
        summary["min_skill_change_vs_base"] = float(
            summary["min_skill"] - base_summary["min_skill"]
        )
        summary["mean_skill_change_vs_basic_r4"] = float(
            summary["mean_skill"] - basic_summary["mean_skill"]
        )
        summary["min_skill_change_vs_basic_r4"] = float(
            summary["min_skill"] - basic_summary["min_skill"]
        )
        summary["beats_base_every_report_season"] = bool(
            all(
                change > 0.0
                for change in summary[
                    "season_skill_change_vs_base"
                ].values()
            )
        )
        summary["beats_basic_r4_every_report_season"] = bool(
            all(
                change > 0.0
                for change in summary[
                    "season_skill_change_vs_basic_r4"
                ].values()
            )
        )
    return aggregate


def main() -> None:
    started = time.time()
    rows = load_rows()
    targets, base, basic_reference = load_oof(rows)
    source_models: dict[int, dict[str, object]] = {}

    def get_source_model(source_season: int) -> dict[str, object]:
        if source_season not in source_models:
            source_models[source_season] = fit_source_models(
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
        predictions: dict[str, np.ndarray] = {
            BASE_CANDIDATE: base[validation_season].copy(),
            BASIC_REFERENCE: basic_reference[validation_season].copy(),
        }
        corrections: dict[str, np.ndarray] = {}
        coverage: dict[str, object] = {}
        segment_metrics: dict[str, object] = {}

        for spec in CONTEXTS:
            mapped_sources: dict[int, dict[str, object]] = {
                source_season: map_source_model(
                    get_source_model(source_season),
                    rows[validation_season],
                    spec,
                )
                for source_season in source_seasons
            }
            (
                coverage[spec],
                pitcher_source_count,
                exact_source_count,
            ) = summarize_coverage(
                mapped_sources, len(rows[validation_season])
            )
            np.save(
                ARTIFACT_DIR
                / f"pitcher_seen_source_count_{spec}_{validation_season}.npy",
                pitcher_source_count,
            )
            np.save(
                ARTIFACT_DIR
                / f"exact_context_source_count_{spec}_{validation_season}.npy",
                exact_source_count,
            )

            for rank in RANKS:
                candidate = candidate_name(spec, rank)
                if mapped_sources:
                    correction = np.mean(
                        np.vstack(
                            [
                                mapped["corrections"][rank]
                                for mapped in mapped_sources.values()
                            ]
                        ),
                        axis=0,
                    )
                else:
                    correction = np.zeros(
                        len(rows[validation_season]), dtype=np.float64
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
                if np.any(correction[unseen] != 0.0):
                    raise AssertionError(
                        f"unseen pitcher correction {validation_season} "
                        f"{candidate}"
                    )
                corrections[candidate] = correction
                predictions[candidate] = prediction
                segment_metrics[candidate] = calculate_segment_metrics(
                    targets[validation_season],
                    base[validation_season],
                    prediction,
                    pitcher_source_count,
                    exact_source_count,
                )

        if tuple(predictions) != CANDIDATES:
            raise AssertionError("candidate declaration/order drift")
        metrics = {
            candidate: calculate_metrics(
                targets[validation_season], candidate_predictions
            )
            for candidate, candidate_predictions in predictions.items()
        }
        for candidate, candidate_predictions in predictions.items():
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate}_{validation_season}.npy",
                candidate_predictions,
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

        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "rows": int(len(rows[validation_season])),
            "coverage": coverage,
            "correction_diagnostics": {
                candidate: correction_diagnostics(correction)
                for candidate, correction in corrections.items()
            },
            "coverage_segment_metrics": segment_metrics,
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
                "static_context_domains_used": True,
                "unseen_pitcher_correction_is_zero": True,
                "richer_matrices_combined_or_added": False,
                "basic_reference_added_to_richer_effect": False,
            },
        }
        print(
            f"rich_lowrank_pctx {validation_season}: "
            + " ".join(
                f"{candidate}="
                f"{metrics[candidate]['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    aggregate = aggregate_metrics(folds)
    best_mean = max(
        RICH_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["mean_skill"],
            aggregate[candidate]["min_skill"],
            -RICH_CANDIDATES.index(candidate),
        ),
    )
    best_min = max(
        RICH_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["min_skill"],
            aggregate[candidate]["latest_2024_skill"],
            aggregate[candidate]["mean_skill"],
            -RICH_CANDIDATES.index(candidate),
        ),
    )
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": (
            "standalone_richer_low_rank_pitcher_context_EB_atop_"
            "team_allprior"
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
                "equal mean of earlier source-season corrections; absent "
                "pitcher contributes zero"
            ),
            "current_fold_labels_used_for_effect_fit": False,
            "validation_or_test_row_aggregation": False,
            "validation_current_row_keys_only": True,
            "test_csv_read": False,
            "candidate_grid_predeclared": True,
            "candidate_comparison_nested": False,
            "richer_effects_evaluated_standalone": True,
        },
        "predeclared_configuration": {
            "smoothing": SMOOTHING,
            "ranks": list(RANKS),
            "richer_candidate_count": len(RICH_CANDIDATES),
            "correction_weight": 1.0,
            "decomposition": "deterministic full SVD then truncated rank",
            "matrix_centering": (
                "no matrix-column centering; residual centered within "
                "source season"
            ),
            "specs": {
                "outs": {
                    "matrix": (
                        "pitcher_id x (count_index,batter_hand,outs_before)"
                    ),
                    "static_axis_values": list(OUT_VALUES),
                    "context_count": len(OUTS_CONTEXTS),
                    "contexts": [
                        {
                            "position": position,
                            "count_index": context[0],
                            "batter_hand": context[1],
                            "outs_before": context[2],
                        }
                        for position, context in enumerate(OUTS_CONTEXTS)
                    ],
                },
                "runners": {
                    "matrix": (
                        "pitcher_id x (count_index,batter_hand,"
                        "coarse_runner_count)"
                    ),
                    "coarse_runner_count_definition": "0, 1, or 2+",
                    "static_axis_values": list(RUNNER_BUCKETS),
                    "context_count": len(RUNNER_CONTEXTS),
                    "contexts": [
                        {
                            "position": position,
                            "count_index": context[0],
                            "batter_hand": context[1],
                            "runner_bucket": context[2],
                        }
                        for position, context in enumerate(RUNNER_CONTEXTS)
                    ],
                },
            },
            "basic_reference": (
                "saved low_rank_pitcher_context_eb lowrank_s300_r4 OOF"
            ),
        },
        "source_matrix_diagnostics": {
            str(source_season): source_model["diagnostics"]
            for source_season, source_model in source_models.items()
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "diagnostic only; comparison is non-nested/post-hoc",
            "posthoc_best_richer_mean_candidate": best_mean,
            "posthoc_best_richer_min_candidate": best_min,
            "best_mean_beats_base_every_report_season": bool(
                aggregate[best_mean]["beats_base_every_report_season"]
            ),
            "best_mean_beats_basic_r4_every_report_season": bool(
                aggregate[best_mean][
                    "beats_basic_r4_every_report_season"
                ]
            ),
            "best_min_beats_basic_r4_min": bool(
                aggregate[best_min]["min_skill"]
                > aggregate[BASIC_REFERENCE]["min_skill"]
            ),
            "adopt_without_nested_confirmation": False,
        },
        "qa": {
            "source_target_and_row_order_alignment_checked": True,
            "source_residual_centering_checked": True,
            "source_season_order_checked": True,
            "official_static_context_domains_checked": True,
            "prediction_probability_ranges_checked": True,
            "unseen_pitcher_zero_correction_checked": True,
            "exact_coverage_subset_of_pitcher_coverage_checked": True,
            "singular_value_order_checked": True,
            "saved_prediction_and_correction_arrays": True,
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
