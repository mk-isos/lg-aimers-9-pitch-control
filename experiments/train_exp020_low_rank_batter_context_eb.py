"""EXP-020: low-rank batter-by-count-context EB atop team OOF.

This is the temporal mirror of the low-rank pitcher-context experiment.  For
each outer validation season, every earlier OOF season independently supplies
a season-centered residual matrix with rows=batter_id and columns=the 24
static (count_index, pitcher_hand) contexts.  Strong EB smoothing precedes a
rank-4 SVD reconstruction, and source-season corrections are averaged equally
with zero for a batter absent from a source season.

The immutable base is the temporal-safe team ``all_prior_s1000`` OOF.  Two
batter-only candidates use smoothing 300 or 600.  Four joint candidates add a
predeclared 25% or 50% batter correction to the saved pitcher lowrank_s300_r4
prediction.  No current-fold label, current-fold/test aggregation, or
post-result weight selection is used for fitting.  Candidate comparison is
still non-nested and therefore diagnostic.
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
PITCHER_REFERENCE_ROOT = Path(
    "./artifacts/EXP-020/low_rank_pitcher_context_eb"
)
ARTIFACT_DIR = Path(
    "./artifacts/EXP-020/low_rank_batter_context_eb"
)

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
SMOOTHING_GRID = (300.0, 600.0)
RANK = 4
BATTER_WEIGHTS = (0.25, 0.50)
PITCHER_HANDS = (1, 2)
COUNT_INDICES = tuple(
    balls * 4 + strikes
    for balls in range(4)
    for strikes in range(3)
)
CONTEXTS = tuple(
    (count_index, pitcher_hand)
    for count_index in COUNT_INDICES
    for pitcher_hand in PITCHER_HANDS
)
CONTEXT_TO_POSITION = {
    context: position for position, context in enumerate(CONTEXTS)
}

BASE_CANDIDATE = "base_team_all_prior"
PITCHER_REFERENCE = "pitcher_lowrank_s300_r4_reference"
BATTER_CANDIDATES = tuple(
    f"batter_lowrank_s{int(smoothing)}_r4"
    for smoothing in SMOOTHING_GRID
)
JOINT_CANDIDATES = tuple(
    f"joint_p300r4_b{int(smoothing)}r4_w{int(weight * 100):03d}"
    for smoothing in SMOOTHING_GRID
    for weight in BATTER_WEIGHTS
)
CANDIDATES = (
    BASE_CANDIDATE,
    PITCHER_REFERENCE,
    *BATTER_CANDIDATES,
    *JOINT_CANDIDATES,
)
EXPERIMENTAL_CANDIDATES = (*BATTER_CANDIDATES, *JOINT_CANDIDATES)


def batter_name(smoothing: float) -> str:
    return f"batter_lowrank_s{int(smoothing)}_r4"


def joint_name(smoothing: float, weight: float) -> str:
    return (
        f"joint_p300r4_b{int(smoothing)}r4_w{int(weight * 100):03d}"
    )


def load_rows() -> dict[int, pd.DataFrame]:
    frame = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=[
            "season",
            "batter_id",
            "balls_before",
            "strikes_before",
            "pitcher_hand",
            "control_success",
        ],
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
        frame["pitcher_hand"].dropna().astype(int).unique().tolist()
    )
    if not observed_counts.issubset(set(COUNT_INDICES)):
        raise ValueError(f"unexpected count_index values: {observed_counts}")
    if not observed_hands.issubset(set(PITCHER_HANDS)):
        raise ValueError(f"unexpected pitcher_hand values: {observed_hands}")
    if frame[
        ["batter_id", "count_index", "pitcher_hand", "control_success"]
    ].isna().any().any():
        raise ValueError("missing required batter-context field")
    frame["context_position"] = [
        CONTEXT_TO_POSITION[(int(count_index), int(pitcher_hand))]
        for count_index, pitcher_hand in zip(
            frame["count_index"],
            frame["pitcher_hand"],
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
    pitcher_reference: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(
            BASE_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        base[season] = np.load(
            BASE_ROOT / f"predictions_all_prior_s1000_{season}.npy"
        ).astype(np.float64)
        pitcher_reference[season] = np.load(
            PITCHER_REFERENCE_ROOT
            / f"predictions_lowrank_s300_r4_{season}.npy"
        ).astype(np.float64)
        csv_targets = rows[season]["control_success"].to_numpy(
            dtype=np.float64
        )
        if not (
            len(csv_targets)
            == len(targets[season])
            == len(base[season])
            == len(pitcher_reference[season])
            and np.array_equal(csv_targets, targets[season])
        ):
            raise ValueError(f"OOF target/order mismatch for {season}")
        for label, predictions in (
            ("base", base[season]),
            ("pitcher_reference", pitcher_reference[season]),
        ):
            if not np.isfinite(predictions).all() or not (
                (predictions >= 0.0).all()
                and (predictions <= 1.0).all()
            ):
                raise ValueError(f"invalid {label} for {season}")
    return targets, base, pitcher_reference


def fit_source_matrix(
    source_season: int,
    source_rows: pd.DataFrame,
    targets: np.ndarray,
    base: np.ndarray,
) -> dict[str, object]:
    raw_residual = targets - base
    raw_mean = float(raw_residual.mean())
    residual = raw_residual - raw_mean
    if abs(float(residual.mean())) > 1e-12:
        raise AssertionError("source residual centering failed")

    batter_codes, batter_ids = pd.factorize(
        source_rows["batter_id"], sort=True
    )
    if (batter_codes < 0).any():
        raise ValueError(f"missing batter ID in source {source_season}")
    context_positions = source_rows["context_position"].to_numpy(
        dtype=np.int16
    )
    shape = (len(batter_ids), len(CONTEXTS))
    residual_sums = np.zeros(shape, dtype=np.float64)
    observation_counts = np.zeros(shape, dtype=np.int64)
    np.add.at(
        residual_sums,
        (batter_codes, context_positions),
        residual,
    )
    np.add.at(
        observation_counts,
        (batter_codes, context_positions),
        1,
    )
    if int(observation_counts.sum()) != len(source_rows):
        raise AssertionError("source matrix row count mismatch")

    reconstructions: dict[float, np.ndarray] = {}
    smoothing_diagnostics: dict[str, object] = {}
    for smoothing in SMOOTHING_GRID:
        smoothed = residual_sums / (
            observation_counts.astype(np.float64) + smoothing
        )
        left, singular_values, right = np.linalg.svd(
            smoothed, full_matrices=False
        )
        effective_rank = min(RANK, len(singular_values))
        reconstruction = (
            left[:, :effective_rank]
            * singular_values[:effective_rank]
        ) @ right[:effective_rank, :]
        reconstructions[smoothing] = reconstruction
        total_energy = float(np.square(singular_values).sum())
        retained_energy = float(
            np.square(singular_values[:effective_rank]).sum()
        )
        smoothing_diagnostics[str(int(smoothing))] = {
            "singular_values": [float(value) for value in singular_values],
            "rank": RANK,
            "retained_energy_fraction": (
                retained_energy / total_energy if total_energy > 0.0 else 0.0
            ),
            "matrix_frobenius_norm": float(np.sqrt(total_energy)),
            "smoothed_mean_absolute_effect": float(
                np.abs(smoothed).mean()
            ),
            "smoothed_max_absolute_effect": float(
                np.abs(smoothed).max()
            ),
        }

    return {
        "source_season": source_season,
        "batter_ids": np.asarray(batter_ids),
        "observation_counts": observation_counts,
        "reconstructions": reconstructions,
        "diagnostics": {
            "source_rows": int(len(source_rows)),
            "source_batters": int(len(batter_ids)),
            "observed_batter_context_cells": int(
                (observation_counts > 0).sum()
            ),
            "matrix_cells": int(observation_counts.size),
            "matrix_observed_density": float(
                (observation_counts > 0).mean()
            ),
            "raw_residual_mean_before_centering": raw_mean,
            "residual_mean_after_centering": float(residual.mean()),
            "smoothing": smoothing_diagnostics,
        },
    }


def map_source_matrix(
    source_model: dict[str, object],
    validation_rows: pd.DataFrame,
) -> dict[str, object]:
    source_rows = pd.Index(source_model["batter_ids"]).get_indexer(
        validation_rows["batter_id"]
    )
    contexts = validation_rows["context_position"].to_numpy(dtype=np.int16)
    batter_seen = source_rows >= 0
    safe_rows = np.where(batter_seen, source_rows, 0)
    counts = source_model["observation_counts"]
    exact_context_seen = batter_seen & (counts[safe_rows, contexts] > 0)
    values: dict[float, np.ndarray] = {}
    for smoothing in SMOOTHING_GRID:
        current = np.zeros(len(validation_rows), dtype=np.float64)
        matrix = source_model["reconstructions"][smoothing]
        current[batter_seen] = matrix[
            source_rows[batter_seen], contexts[batter_seen]
        ]
        values[smoothing] = current
    return {
        "batter_seen": batter_seen,
        "exact_context_seen": exact_context_seen,
        "values": values,
    }


def summarize_coverage(
    mapped_sources: dict[int, dict[str, object]], row_count: int
) -> dict[str, object]:
    if not mapped_sources:
        return {
            "source_count": 0,
            "rows": row_count,
            "batter_seen_any_source_rows": 0,
            "batter_seen_any_source_rate": 0.0,
            "batter_seen_every_source_rows": 0,
            "batter_seen_every_source_rate": 0.0,
            "exact_context_seen_any_source_rows": 0,
            "exact_context_seen_any_source_rate": 0.0,
            "exact_context_seen_every_source_rows": 0,
            "exact_context_seen_every_source_rate": 0.0,
            "shared_only_any_source_rows": 0,
            "shared_only_any_source_rate": 0.0,
            "per_source": {},
        }
    batter_matrix = np.vstack(
        [mapped["batter_seen"] for mapped in mapped_sources.values()]
    )
    exact_matrix = np.vstack(
        [mapped["exact_context_seen"] for mapped in mapped_sources.values()]
    )
    batter_any = batter_matrix.any(axis=0)
    batter_all = batter_matrix.all(axis=0)
    exact_any = exact_matrix.any(axis=0)
    exact_all = exact_matrix.all(axis=0)
    shared_only = np.any(batter_matrix & ~exact_matrix, axis=0)
    return {
        "source_count": int(len(mapped_sources)),
        "rows": row_count,
        "batter_seen_any_source_rows": int(batter_any.sum()),
        "batter_seen_any_source_rate": float(batter_any.mean()),
        "batter_seen_every_source_rows": int(batter_all.sum()),
        "batter_seen_every_source_rate": float(batter_all.mean()),
        "exact_context_seen_any_source_rows": int(exact_any.sum()),
        "exact_context_seen_any_source_rate": float(exact_any.mean()),
        "exact_context_seen_every_source_rows": int(exact_all.sum()),
        "exact_context_seen_every_source_rate": float(exact_all.mean()),
        "shared_only_any_source_rows": int(shared_only.sum()),
        "shared_only_any_source_rate": float(shared_only.mean()),
        "per_source": {
            str(source_season): {
                "batter_seen_rows": int(mapped["batter_seen"].sum()),
                "batter_seen_rate": float(mapped["batter_seen"].mean()),
                "exact_context_seen_rows": int(
                    mapped["exact_context_seen"].sum()
                ),
                "exact_context_seen_rate": float(
                    mapped["exact_context_seen"].mean()
                ),
                "shared_only_rows": int(
                    (
                        mapped["batter_seen"]
                        & ~mapped["exact_context_seen"]
                    ).sum()
                ),
                "shared_only_rate": float(
                    np.mean(
                        mapped["batter_seen"]
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
    pitcher = aggregate[PITCHER_REFERENCE]
    for candidate in EXPERIMENTAL_CANDIDATES:
        current = aggregate[candidate]
        for reference_name, reference in (
            ("base", base),
            ("pitcher_lowrank_s300_r4", pitcher),
        ):
            current[f"season_skill_change_vs_{reference_name}"] = {
                str(season): float(
                    current["season_skills"][str(season)]
                    - reference["season_skills"][str(season)]
                )
                for season in REPORT_SEASONS
            }
            current[f"mean_skill_change_vs_{reference_name}"] = float(
                current["mean_skill"] - reference["mean_skill"]
            )
            current[f"min_skill_change_vs_{reference_name}"] = float(
                current["min_skill"] - reference["min_skill"]
            )
    return aggregate


def main() -> None:
    started = time.time()
    rows = load_rows()
    targets, base, pitcher_reference = load_oof(rows)
    source_models: dict[int, dict[str, object]] = {}

    def get_source_model(source_season: int) -> dict[str, object]:
        if source_season not in source_models:
            source_models[source_season] = fit_source_matrix(
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
            source_season: map_source_matrix(
                get_source_model(source_season), rows[validation_season]
            )
            for source_season in source_seasons
        }
        corrections: dict[float, np.ndarray] = {}
        for smoothing in SMOOTHING_GRID:
            if mapped_sources:
                corrections[smoothing] = np.mean(
                    np.vstack(
                        [
                            mapped["values"][smoothing]
                            for mapped in mapped_sources.values()
                        ]
                    ),
                    axis=0,
                )
            else:
                corrections[smoothing] = np.zeros(
                    len(rows[validation_season]), dtype=np.float64
                )

        predictions: dict[str, np.ndarray] = {
            BASE_CANDIDATE: base[validation_season].copy(),
            PITCHER_REFERENCE: pitcher_reference[
                validation_season
            ].copy(),
        }
        correction_diagnostics: dict[str, object] = {}
        for smoothing in SMOOTHING_GRID:
            correction = corrections[smoothing]
            predictions[batter_name(smoothing)] = np.clip(
                base[validation_season] + correction, 0.0, 1.0
            )
            correction_diagnostics[str(int(smoothing))] = {
                "mean": float(correction.mean()),
                "standard_deviation": float(correction.std()),
                "mean_absolute": float(np.abs(correction).mean()),
                "min": float(correction.min()),
                "max": float(correction.max()),
            }
        for smoothing in SMOOTHING_GRID:
            for weight in BATTER_WEIGHTS:
                predictions[joint_name(smoothing, weight)] = np.clip(
                    pitcher_reference[validation_season]
                    + weight * corrections[smoothing],
                    0.0,
                    1.0,
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
                "unseen_batter_correction_is_zero": True,
            },
        }
        print(
            f"lowrank_bctx {validation_season}: "
            + " ".join(
                f"{candidate}="
                f"{metrics[candidate]['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    aggregate = aggregate_metrics(folds)
    best_mean = max(
        EXPERIMENTAL_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["mean_skill"],
            aggregate[candidate]["min_skill"],
            -EXPERIMENTAL_CANDIDATES.index(candidate),
        ),
    )
    best_min = max(
        EXPERIMENTAL_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["min_skill"],
            aggregate[candidate]["latest_2024_skill"],
            aggregate[candidate]["mean_skill"],
            -EXPERIMENTAL_CANDIDATES.index(candidate),
        ),
    )
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": "low_rank_batter_context_EB_and_fixed_joint",
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": (
                "EXP-019 team_eb_ensemble all_prior_s1000 OOF"
            ),
            "pitcher_reference": (
                "saved EXP-020 lowrank pitcher-context s300 rank4 OOF"
            ),
            "effect_target": (
                "source-season-centered target minus immutable base OOF"
            ),
            "matrix": (
                "one batter_id x static (count_index,pitcher_hand) EB "
                "matrix per earlier source OOF season"
            ),
            "source_season_combination": (
                "equal mean; absent batter contributes zero"
            ),
            "current_fold_labels_used_for_effect_fit": False,
            "validation_or_test_row_aggregation": False,
            "candidate_grid_predeclared": True,
            "candidate_comparison_nested": False,
        },
        "predeclared_configuration": {
            "smoothing_grid": list(SMOOTHING_GRID),
            "rank": RANK,
            "joint_formula": (
                "pitcher_lowrank_s300_r4_prediction + batter_weight * "
                "batter_lowrank_correction"
            ),
            "batter_weight_grid": list(BATTER_WEIGHTS),
            "correction_weight_batter_only": 1.0,
            "context_domain_source": (
                "static balls 0..3, strikes 0..2, pitcher_hand in {1,2}"
            ),
            "context_count": len(CONTEXTS),
            "contexts": [
                {
                    "position": position,
                    "count_index": count_index,
                    "pitcher_hand": pitcher_hand,
                }
                for position, (count_index, pitcher_hand) in enumerate(
                    CONTEXTS
                )
            ],
            "matrix_centering": (
                "none after source-season residual centering"
            ),
            "decomposition": "deterministic full SVD then rank4",
        },
        "source_matrix_diagnostics": {
            str(source_season): source_model["diagnostics"]
            for source_season, source_model in source_models.items()
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "diagnostic only; candidate comparison is non-nested",
            "posthoc_best_mean_candidate": best_mean,
            "posthoc_best_min_candidate": best_min,
            "best_mean_beats_base": bool(
                aggregate[best_mean]["mean_skill"]
                > aggregate[BASE_CANDIDATE]["mean_skill"]
            ),
            "best_min_beats_base": bool(
                aggregate[best_min]["min_skill"]
                > aggregate[BASE_CANDIDATE]["min_skill"]
            ),
            "best_min_beats_pitcher_reference": bool(
                aggregate[best_min]["min_skill"]
                > aggregate[PITCHER_REFERENCE]["min_skill"]
            ),
            "adopt_without_nested_confirmation": False,
        },
        "qa": {
            "source_target_and_row_order_alignment_checked": True,
            "pitcher_reference_alignment_checked": True,
            "source_residual_centering_checked": True,
            "source_season_order_checked": True,
            "static_context_domain_checked": True,
            "prediction_probability_ranges_checked": True,
            "saved_prediction_arrays": True,
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
