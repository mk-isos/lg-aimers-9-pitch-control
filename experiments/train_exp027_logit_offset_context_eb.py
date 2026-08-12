"""EXP-027: temporal-safe logit-offset pitcher-context EB.

EXP-020 learns pitcher x (count, batter hand) corrections as additive
probability residuals.  This bounded experiment estimates the same 24-cell
structure as a penalized odds-ratio around the frozen EXP-019 team base.
Each source OOF season is fitted independently, source-wide calibration is
removed, rank 6 is fixed from the earlier strict EXP-020 selection, and
source effects are averaged with missing pitchers contributing zero.

Only current-row official fields and frozen train OOF predictions are used.
There is no validation/test-row aggregation and no current-fold selection.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics


EXPERIMENT = "EXP-027"
DATA_PATH = Path("./data/train.csv")
BASE_ROOT = Path("./artifacts/EXP-019/team_eb_ensemble")
REFERENCE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_ROOT = Path("./artifacts/EXP-027/logit_offset_context_eb")

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
TARGET_SKILL = 1100.0
RANK = 6
RIDGE = 75.0
EFFECT_WEIGHTS = (0.50, 1.00)
EPS = 1e-6

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


def expit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def load_rows() -> dict[int, pd.DataFrame]:
    columns = [
        "season",
        "pitcher_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "control_success",
        "asof_pitcher_n",
        "asof_batter_n",
    ]
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=columns)
    frame = frame.loc[frame["season"].isin(EVALUATED_SEASONS)].copy()
    frame["count_index"] = (
        frame["balls_before"].to_numpy(dtype=np.int8) * 4
        + frame["strikes_before"].to_numpy(dtype=np.int8)
    ).astype(np.int8)
    if not set(frame["count_index"].unique()).issubset(COUNT_INDICES):
        raise ValueError("unexpected count state")
    if not set(frame["batter_hand"].unique()).issubset(BATTER_HANDS):
        raise ValueError("unexpected batter hand")
    if frame[columns[1:]].isna().any().any():
        raise ValueError("missing required field")
    frame["context_position"] = np.asarray(
        [
            CONTEXT_TO_POSITION[(int(count), int(hand))]
            for count, hand in zip(
                frame["count_index"], frame["batter_hand"], strict=True
            )
        ],
        dtype=np.int8,
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
]:
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    reference: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(
            BASE_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        base[season] = np.load(
            BASE_ROOT / f"predictions_all_prior_s1000_{season}.npy"
        ).astype(np.float64)
        reference[season] = np.load(
            REFERENCE_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
        ).astype(np.float64)
        csv_target = rows[season]["control_success"].to_numpy(dtype=float)
        if not (
            len(csv_target)
            == len(targets[season])
            == len(base[season])
            == len(reference[season])
            and np.array_equal(csv_target, targets[season])
        ):
            raise ValueError(f"target/order mismatch in {season}")
        for name, values in (("base", base[season]), ("reference", reference[season])):
            if not np.isfinite(values).all() or not (
                ((values >= 0.0) & (values <= 1.0)).all()
            ):
                raise ValueError(f"invalid {name} probabilities in {season}")
    return targets, base, reference


def scalar_offset(targets: np.ndarray, base_logits: np.ndarray) -> float:
    offset = 0.0
    for _ in range(20):
        probabilities = expit(base_logits + offset)
        score = float(np.sum(targets - probabilities))
        curvature = float(np.sum(probabilities * (1.0 - probabilities)))
        update = score / max(curvature, 1e-12)
        offset += update
        if abs(update) < 1e-12:
            break
    return float(offset)


def fit_source(
    source_season: int,
    rows: pd.DataFrame,
    targets: np.ndarray,
    base: np.ndarray,
) -> dict[str, object]:
    pitcher_codes, pitcher_ids = pd.factorize(rows["pitcher_id"], sort=True)
    contexts = rows["context_position"].to_numpy(dtype=np.int16)
    shape = (len(pitcher_ids), len(CONTEXTS))
    flat_codes = pitcher_codes * len(CONTEXTS) + contexts
    cell_count = shape[0] * shape[1]
    counts = np.bincount(flat_codes, minlength=cell_count).reshape(shape)

    base_logits = logit(base)
    nuisance_offset = scalar_offset(targets, base_logits)
    offsets = base_logits + nuisance_offset
    effects = np.zeros(cell_count, dtype=np.float64)
    for _ in range(20):
        probabilities = expit(offsets + effects[flat_codes])
        score = np.bincount(
            flat_codes,
            weights=targets - probabilities,
            minlength=cell_count,
        ) - RIDGE * effects
        curvature = np.bincount(
            flat_codes,
            weights=probabilities * (1.0 - probabilities),
            minlength=cell_count,
        ) + RIDGE
        update = score / curvature
        effects += update
        if float(np.max(np.abs(update))) < 1e-10:
            break
    matrix = effects.reshape(shape)
    observed = counts > 0
    weighted_mean = float(
        np.sum(matrix * counts) / max(float(np.sum(counts)), 1.0)
    )
    matrix = matrix - weighted_mean
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    effective_rank = min(RANK, len(singular_values))
    reconstruction = (
        left[:, :effective_rank] * singular_values[:effective_rank]
    ) @ right[:effective_rank, :]
    # Re-center after truncation so source labels cannot transfer a global bias.
    reconstructed_mean = float(
        np.sum(reconstruction * counts) / max(float(np.sum(counts)), 1.0)
    )
    reconstruction -= reconstructed_mean
    total_energy = float(np.square(singular_values).sum())
    retained_energy = float(np.square(singular_values[:effective_rank]).sum())
    return {
        "source_season": source_season,
        "pitcher_ids": np.asarray(pitcher_ids),
        "counts": counts,
        "reconstruction": reconstruction,
        "diagnostics": {
            "rows": int(len(rows)),
            "pitchers": int(len(pitcher_ids)),
            "observed_cells": int(observed.sum()),
            "nuisance_global_logit_offset_not_transferred": nuisance_offset,
            "weighted_effect_mean_before_centering": weighted_mean,
            "weighted_rank_effect_mean_before_centering": reconstructed_mean,
            "rank": RANK,
            "ridge": RIDGE,
            "retained_energy_fraction": (
                retained_energy / total_energy if total_energy > 0.0 else 0.0
            ),
            "mean_absolute_rank_effect": float(np.abs(reconstruction).mean()),
            "max_absolute_rank_effect": float(np.abs(reconstruction).max()),
        },
    }


def map_source(model: dict[str, object], rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    source_indices = pd.Index(model["pitcher_ids"]).get_indexer(rows["pitcher_id"])
    seen = source_indices >= 0
    contexts = rows["context_position"].to_numpy(dtype=np.int16)
    values = np.zeros(len(rows), dtype=np.float64)
    values[seen] = model["reconstruction"][
        source_indices[seen], contexts[seen]
    ]
    return values, seen


def segment_metrics(
    rows: pd.DataFrame, targets: np.ndarray, predictions: np.ndarray
) -> dict[str, object]:
    pitcher_n = rows["asof_pitcher_n"].to_numpy(dtype=float)
    batter_n = rows["asof_batter_n"].to_numpy(dtype=float)
    masks = {
        "pitcher_n_0": pitcher_n == 0,
        "pitcher_n_1_19": (pitcher_n >= 1) & (pitcher_n < 20),
        "pitcher_n_20_99": (pitcher_n >= 20) & (pitcher_n < 100),
        "pitcher_n_100_499": (pitcher_n >= 100) & (pitcher_n < 500),
        "pitcher_n_500_plus": pitcher_n >= 500,
        "both_history": (pitcher_n > 0) & (batter_n > 0),
        "either_cold": (pitcher_n == 0) | (batter_n == 0),
    }
    return {
        name: calculate_metrics(targets[mask], predictions[mask])
        for name, mask in masks.items()
        if int(mask.sum()) > 1
    }


def main() -> None:
    started = time.time()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    targets, base, reference = load_oof(rows)
    source_models: dict[int, dict[str, object]] = {}
    folds: dict[str, object] = {}
    prediction_cache: dict[int, dict[str, np.ndarray]] = {}

    for validation_season in REPORT_SEASONS:
        source_seasons = [
            season for season in EVALUATED_SEASONS if season < validation_season
        ]
        for source_season in source_seasons:
            if source_season not in source_models:
                source_models[source_season] = fit_source(
                    source_season,
                    rows[source_season],
                    targets[source_season],
                    base[source_season],
                )
        mapped: list[np.ndarray] = []
        seen_masks: list[np.ndarray] = []
        for source_season in source_seasons:
            values, seen = map_source(source_models[source_season], rows[validation_season])
            mapped.append(values)
            seen_masks.append(seen)
        mean_effect = np.mean(np.vstack(mapped), axis=0)
        candidates: dict[str, np.ndarray] = {}
        for weight in EFFECT_WEIGHTS:
            name = f"logit_offset_w{int(round(weight * 100)):03d}"
            candidates[name] = expit(logit(base[validation_season]) + weight * mean_effect)
            np.save(
                ARTIFACT_ROOT / f"predictions_{name}_{validation_season}.npy",
                candidates[name],
            )
        prediction_cache[validation_season] = candidates
        fold = {
            "validation_season": validation_season,
            "source_seasons": source_seasons,
            "current_fold_labels_used_for_fit_or_selection": False,
            "base_team_all_prior": calculate_metrics(
                targets[validation_season], base[validation_season]
            ),
            "reference_additive_rank6": calculate_metrics(
                targets[validation_season], reference[validation_season]
            ),
            "candidates": {
                name: calculate_metrics(targets[validation_season], prediction)
                for name, prediction in candidates.items()
            },
            "coverage": {
                "pitcher_seen_any_source_rate": float(
                    np.vstack(seen_masks).any(axis=0).mean()
                ),
                "pitcher_seen_every_source_rate": float(
                    np.vstack(seen_masks).all(axis=0).mean()
                ),
            },
            "segments": {
                name: segment_metrics(rows[validation_season], targets[validation_season], prediction)
                for name, prediction in candidates.items()
            },
        }
        folds[str(validation_season)] = fold
        np.save(ARTIFACT_ROOT / f"targets_{validation_season}.npy", targets[validation_season])

    aggregate_candidates: dict[str, object] = {}
    for name in sorted(prediction_cache[2022]):
        briers = {
            str(season): float(folds[str(season)]["candidates"][name]["brier_score"])
            for season in REPORT_SEASONS
        }
        skills = {
            str(season): float(folds[str(season)]["candidates"][name]["skill_score_unclipped"])
            for season in REPORT_SEASONS
        }
        aggregate_candidates[name] = {
            "season_briers": briers,
            "season_skills": skills,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills["2024"],
            "uniform_1100_passed": all(value >= TARGET_SKILL for value in skills.values()),
        }

    reference_skills = {
        str(season): float(
            folds[str(season)]["reference_additive_rank6"]["skill_score_unclipped"]
        )
        for season in REPORT_SEASONS
    }
    best_name = max(
        aggregate_candidates,
        key=lambda name: (
            aggregate_candidates[name]["min_skill"],
            aggregate_candidates[name]["mean_skill"],
        ),
    )
    uniform_pass = bool(aggregate_candidates[best_name]["uniform_1100_passed"])
    report = {
        "experiment": EXPERIMENT,
        "stage": "bounded rolling validation",
        "hypothesis": (
            "penalized source-season odds ratios around the current as-of-aware "
            "team base transfer more stably than additive probability residuals"
        ),
        "protocol": {
            "report_seasons": list(REPORT_SEASONS),
            "source_season_strictly_prior": True,
            "current_fold_selection": False,
            "test_row_aggregation": False,
            "source_global_calibration_transferred": False,
            "rank": RANK,
            "ridge": RIDGE,
            "effect_weights_predeclared": list(EFFECT_WEIGHTS),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "source_models": {
            str(season): model["diagnostics"]
            for season, model in source_models.items()
        },
        "folds": folds,
        "aggregate_2022_2024": {
            "reference_additive_rank6_season_skills": reference_skills,
            "candidates": aggregate_candidates,
            "posthoc_best_min_candidate": best_name,
            "uniform_1100_passed": uniform_pass,
            "final_fit_authorized": uniform_pass,
            "zip_creation_authorized": uniform_pass,
        },
        "qa": {
            "target_and_row_order_match": True,
            "probabilities_finite_and_in_range": True,
            "missing_source_pitcher_contributes_zero": True,
            "source_effects_equal_weighted_including_zero": True,
            "current_fold_labels_used_for_fit_or_selection": False,
            "test_row_aggregation": False,
            "final_fit_or_zip_created": False,
        },
        "total_seconds": time.time() - started,
    }
    with (ARTIFACT_ROOT / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report["aggregate_2022_2024"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
