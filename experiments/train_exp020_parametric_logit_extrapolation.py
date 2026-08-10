"""EXP-020: parametric clipped-logit extrapolation residual diagnostic.

This bounded experiment tests one specific hypothesis: unlike a tree leaf, a
linear clipped-logit basis can continue changing when 2023/2024 current-season
success posteriors fall outside the range observed by earlier OOF seasons.

The immutable base is the fully temporal ``strict_rank_s300`` low-rank OOF.
The R-specific low-rank OOF is loaded only as a clearly labelled post-hoc
reference and never enters fitting or candidate predictions.  Each outer fold
fits feature scaling and coefficients on earlier evaluated OOF seasons only,
with equal total weight per source season.  Four configurations are fixed:
strong ridge-residual and logistic-offset models, each blended at 0.5 and 1.0.

Inputs are current-row official as-of features or row-independent temporal
reconstructions from fixed prior history: pitcher/batter prior, current-season
posterior and reliability, pitcher recent success/reverse, plus a static
count-by-hands one-hot.  Raw player/team IDs, season, batter raw rate,
pitcher ball/strike rates, current-fold calibration/selection, and validation
or test-row aggregation are excluded.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit

from temporal_residual_features import attach_training_temporal_features
from train_exp017_rolling_residual import calculate_metrics


DATA_PATH = Path("./data/train.csv")
BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path(
    "./artifacts/EXP-020/parametric_logit_extrapolation"
)

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
PROBABILITY_CLIP = 0.02
RIDGE_ALPHA = 100000.0
LOGISTIC_ALPHA = 50000.0
CONTEXT_COUNT = 4 * 3 * 2 * 2

RAW_COLUMNS = (
    "season",
    "balls_before",
    "strikes_before",
    "pitcher_hand",
    "batter_hand",
    "pitcher_id",
    "batter_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "control_success",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    blend_weight: float


CANDIDATES = (
    Candidate("ridge_w050", "ridge_residual", 0.50),
    Candidate("ridge_w100", "ridge_residual", 1.00),
    Candidate("logistic_w050", "logistic_offset", 0.50),
    Candidate("logistic_w100", "logistic_offset", 1.00),
)

CONTINUOUS_SPECS = (
    (
        "pitcher_prior",
        "temporal_pitcher_prior_rate_shrunk_200",
        "logit",
        0.50,
    ),
    (
        "pitcher_season_global30",
        "temporal_pitcher_season_global_30",
        "logit",
        0.50,
    ),
    (
        "pitcher_season_player30",
        "temporal_pitcher_season_player_30",
        "logit",
        0.50,
    ),
    (
        "pitcher_reliability30",
        "temporal_pitcher_reliability_30",
        "linear",
        0.00,
    ),
    (
        "batter_prior",
        "temporal_batter_prior_rate_shrunk_200",
        "logit",
        0.50,
    ),
    (
        "batter_season_global30",
        "temporal_batter_season_global_30",
        "logit",
        0.50,
    ),
    (
        "batter_season_player30",
        "temporal_batter_season_player_30",
        "logit",
        0.50,
    ),
    (
        "batter_reliability30",
        "temporal_batter_reliability_30",
        "linear",
        0.00,
    ),
    (
        "pitcher_recent1_success",
        "asof_pitcher_prev1_game_success_rate",
        "logit",
        0.50,
    ),
    (
        "pitcher_recent3_success",
        "asof_pitcher_prev3_game_success_rate",
        "logit",
        0.50,
    ),
    (
        "pitcher_recent5_success",
        "asof_pitcher_prev5_game_success_rate",
        "logit",
        0.50,
    ),
    (
        "pitcher_reverse",
        "asof_pitcher_reverse_rate",
        "logit",
        0.20,
    ),
)
CONTINUOUS_NAMES = tuple(spec[0] for spec in CONTINUOUS_SPECS)
KEY_LOW_POSTERIORS = (
    "pitcher_season_global30",
    "pitcher_season_player30",
    "batter_season_global30",
    "batter_season_player30",
)
FORBIDDEN_INPUTS = {
    "row_id",
    "control_success",
    "season",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "asof_batter_success_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
}


def prepare_frame() -> pd.DataFrame:
    frame = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=list(RAW_COLUMNS),
    )
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("train rows must remain season sorted")
    frame, _ = attach_training_temporal_features(
        frame, target="control_success"
    )
    required_model_columns = {
        spec[1] for spec in CONTINUOUS_SPECS
    } | {
        "balls_before",
        "strikes_before",
        "pitcher_hand",
        "batter_hand",
    }
    if required_model_columns & FORBIDDEN_INPUTS:
        raise ValueError("forbidden model input configured")
    missing = sorted(required_model_columns - set(frame.columns))
    if missing:
        raise ValueError(f"missing reconstructed feature: {missing}")
    frame = frame.loc[
        frame["season"].isin(EVALUATED_SEASONS)
    ].reset_index(drop=True)
    return frame


def load_oof(
    frame: pd.DataFrame,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    targets: dict[int, np.ndarray] = {}
    strict_base: dict[int, np.ndarray] = {}
    posthoc_r_specific: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        local = frame["season"].eq(season).to_numpy()
        targets[season] = np.load(
            BASE_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        strict_base[season] = np.load(
            BASE_ROOT / f"predictions_strict_rank_s300_{season}.npy"
        ).astype(np.float64)
        posthoc_r_specific[season] = np.load(
            BASE_ROOT
            / f"predictions_lowrank_s300_r4_Rspecific_{season}.npy"
        ).astype(np.float64)
        csv_target = frame.loc[local, "control_success"].to_numpy(
            dtype=np.float64
        )
        if not (
            len(csv_target)
            == len(targets[season])
            == len(strict_base[season])
            == len(posthoc_r_specific[season])
            and np.array_equal(csv_target, targets[season])
        ):
            raise ValueError(f"OOF target/order mismatch for {season}")
        for label, prediction in (
            ("strict", strict_base[season]),
            ("posthoc_r_specific", posthoc_r_specific[season]),
        ):
            if not np.isfinite(prediction).all() or not (
                (prediction >= 0.0).all()
                and (prediction <= 1.0).all()
            ):
                raise ValueError(f"invalid {label} prediction for {season}")
    return targets, strict_base, posthoc_r_specific


def clipped_logit(values: np.ndarray) -> np.ndarray:
    return logit(
        np.clip(values, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    )


def build_continuous(frame: pd.DataFrame) -> np.ndarray:
    output = np.empty(
        (len(frame), len(CONTINUOUS_SPECS)), dtype=np.float64
    )
    league_prior = frame["temporal_prior_league_rate"].to_numpy(
        dtype=float
    )
    for index, (_, column, transform, fill_value) in enumerate(
        CONTINUOUS_SPECS
    ):
        values = frame[column].to_numpy(dtype=float)
        if np.isnan(values).any():
            if transform == "logit" and fill_value == 0.50:
                values = np.where(np.isnan(values), league_prior, values)
            else:
                values = np.nan_to_num(values, nan=fill_value)
        values = np.clip(values, 0.0, 1.0)
        output[:, index] = (
            clipped_logit(values) if transform == "logit" else values
        )
    if not np.isfinite(output).all():
        raise ValueError("non-finite transformed continuous feature")
    return output


def build_context_codes(frame: pd.DataFrame) -> np.ndarray:
    balls = frame["balls_before"].to_numpy(dtype=np.int16)
    strikes = frame["strikes_before"].to_numpy(dtype=np.int16)
    pitcher_hand = frame["pitcher_hand"].to_numpy(dtype=np.int16) - 1
    batter_hand = frame["batter_hand"].to_numpy(dtype=np.int16) - 1
    if not (
        ((balls >= 0) & (balls <= 3)).all()
        and ((strikes >= 0) & (strikes <= 2)).all()
        and ((pitcher_hand >= 0) & (pitcher_hand <= 1)).all()
        and ((batter_hand >= 0) & (batter_hand <= 1)).all()
    ):
        raise ValueError("unexpected static count/hand domain")
    codes = (
        (((balls * 3 + strikes) * 2 + pitcher_hand) * 2 + batter_hand)
    ).astype(np.int16)
    if not ((codes >= 0).all() and (codes < CONTEXT_COUNT).all()):
        raise ValueError("invalid count-by-hands code")
    return codes


def season_equal_weights(seasons: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(seasons), dtype=np.float64)
    unique, counts = np.unique(seasons, return_counts=True)
    for season, count in zip(unique, counts, strict=True):
        weights[seasons == season] = 1.0 / float(count)
    weights *= len(weights) / float(weights.sum())
    return weights


def make_design(
    train_continuous: np.ndarray,
    validation_continuous: np.ndarray,
    train_context: np.ndarray,
    validation_context: np.ndarray,
    train_weights: np.ndarray,
) -> tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    dict[str, Any],
]:
    weight_sum = float(train_weights.sum())
    mean = np.sum(
        train_continuous * train_weights[:, None], axis=0
    ) / weight_sum
    variance = np.sum(
        np.square(train_continuous - mean) * train_weights[:, None],
        axis=0,
    ) / weight_sum
    scale = np.sqrt(np.maximum(variance, 1e-12))
    train_scaled = (
        (train_continuous - mean) / scale
    ).astype(np.float32)
    validation_scaled = (
        (validation_continuous - mean) / scale
    ).astype(np.float32)

    train_rows = np.arange(len(train_context), dtype=np.int64)
    validation_rows = np.arange(len(validation_context), dtype=np.int64)
    train_onehot = sparse.csr_matrix(
        (
            np.ones(len(train_context), dtype=np.float32),
            (train_rows, train_context),
        ),
        shape=(len(train_context), CONTEXT_COUNT),
    )
    validation_onehot = sparse.csr_matrix(
        (
            np.ones(len(validation_context), dtype=np.float32),
            (validation_rows, validation_context),
        ),
        shape=(len(validation_context), CONTEXT_COUNT),
    )
    train_design = sparse.hstack(
        [sparse.csr_matrix(train_scaled), train_onehot], format="csr"
    )
    validation_design = sparse.hstack(
        [
            sparse.csr_matrix(validation_scaled),
            validation_onehot,
        ],
        format="csr",
    )
    return train_design, validation_design, {
        "weighted_training_means": {
            name: float(value)
            for name, value in zip(CONTINUOUS_NAMES, mean, strict=True)
        },
        "weighted_training_scales": {
            name: float(value)
            for name, value in zip(CONTINUOUS_NAMES, scale, strict=True)
        },
    }


def centered_source_residual(
    target: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
) -> np.ndarray:
    residual = target - base
    centered = residual.copy()
    for season in np.unique(seasons):
        mask = seasons == season
        centered[mask] -= float(residual[mask].mean())
    return centered


def fit_ridge(
    design: sparse.csr_matrix,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    weighted = design.multiply(np.sqrt(weights)[:, None])
    gram = (weighted.T @ weighted).toarray().astype(np.float64)
    gram.flat[:: len(gram) + 1] += RIDGE_ALPHA
    cross = np.asarray(
        design.T @ (weights * target), dtype=np.float64
    ).ravel()
    coefficient = np.linalg.solve(gram, cross)
    gradient = gram @ coefficient - cross
    return coefficient, {
        "alpha": RIDGE_ALPHA,
        "solver": "exact weighted normal equations",
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "coefficient_l2": float(np.linalg.norm(coefficient)),
    }


def fit_logistic_offset(
    design: sparse.csr_matrix,
    target: np.ndarray,
    base: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    offset = clipped_logit(base)
    feature_count = design.shape[1]

    def objective_and_gradient(
        coefficient: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        linear = offset + np.asarray(design @ coefficient).ravel()
        loss = float(
            np.sum(
                weights
                * (np.logaddexp(0.0, linear) - target * linear)
            )
            + 0.5 * LOGISTIC_ALPHA * coefficient @ coefficient
        )
        probability = expit(linear)
        gradient = np.asarray(
            design.T @ (weights * (probability - target)),
            dtype=np.float64,
        ).ravel() + LOGISTIC_ALPHA * coefficient
        return loss, gradient

    result = minimize(
        objective_and_gradient,
        np.zeros(feature_count, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={
            "ftol": 1e-13,
            "gtol": 1e-6,
            "maxiter": 300,
            "maxls": 40,
        },
    )
    coefficient = result.x.astype(np.float64)
    _, gradient = objective_and_gradient(coefficient)
    return coefficient, {
        "alpha": LOGISTIC_ALPHA,
        "solver": "L-BFGS-B penalized logistic offset",
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "coefficient_l2": float(np.linalg.norm(coefficient)),
    }


def coefficient_report(coefficient: np.ndarray) -> dict[str, Any]:
    continuous = coefficient[: len(CONTINUOUS_NAMES)]
    context = coefficient[len(CONTINUOUS_NAMES) :]
    return {
        "continuous_coefficients_per_training_sd": {
            name: float(value)
            for name, value in zip(
                CONTINUOUS_NAMES, continuous, strict=True
            )
        },
        "continuous_signs": {
            name: int(np.sign(value))
            for name, value in zip(
                CONTINUOUS_NAMES, continuous, strict=True
            )
        },
        "context_coefficient_min": float(context.min()),
        "context_coefficient_max": float(context.max()),
        "context_coefficient_mean": float(context.mean()),
        "context_coefficient_l2": float(np.linalg.norm(context)),
        "all_coefficients_finite": bool(np.isfinite(coefficient).all()),
    }


def extrapolation_diagnostics(
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    train_min = train.min(axis=0)
    train_max = train.max(axis=0)
    below = validation < train_min[None, :] - 1e-12
    above = validation > train_max[None, :] + 1e-12
    output: dict[str, Any] = {}
    for index, name in enumerate(CONTINUOUS_NAMES):
        output[name] = {
            "training_min": float(train_min[index]),
            "training_max": float(train_max[index]),
            "validation_min": float(validation[:, index].min()),
            "validation_max": float(validation[:, index].max()),
            "below_training_min_rows": int(below[:, index].sum()),
            "below_training_min_fraction": float(
                below[:, index].mean()
            ),
            "above_training_max_rows": int(above[:, index].sum()),
            "above_training_max_fraction": float(
                above[:, index].mean()
            ),
        }
    any_outside = np.any(below | above, axis=1)
    key_indices = [
        CONTINUOUS_NAMES.index(name) for name in KEY_LOW_POSTERIORS
    ]
    any_key_low = np.any(below[:, key_indices], axis=1)
    output["summary"] = {
        "any_continuous_outside_rows": int(any_outside.sum()),
        "any_continuous_outside_fraction": float(any_outside.mean()),
        "any_key_posterior_below_rows": int(any_key_low.sum()),
        "any_key_posterior_below_fraction": float(any_key_low.mean()),
    }
    return output, any_outside, any_key_low


def masked_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if not mask.any():
        return {"rows": 0}
    return calculate_metrics(target[mask], prediction[mask])


def aggregate_metrics(folds: dict[str, Any]) -> dict[str, Any]:
    names = ("strict_temporal_base", "posthoc_Rspecific_reference") + tuple(
        candidate.name for candidate in CANDIDATES
    )
    output: dict[str, Any] = {}
    for name in names:
        metrics = {
            season: folds[str(season)]["candidates"][name]["metrics"]
            for season in REPORT_SEASONS
        }
        skills = {
            season: float(value["skill_score_unclipped"])
            for season, value in metrics.items()
        }
        output[name] = {
            "season_briers": {
                str(season): float(metrics[season]["brier_score"])
                for season in REPORT_SEASONS
            },
            "season_skills": {
                str(season): value for season, value in skills.items()
            },
            "season_mean_gaps": {
                str(season): float(metrics[season]["mean_gap"])
                for season in REPORT_SEASONS
            },
            "season_calibration_slopes": {
                str(season): float(
                    metrics[season]["diagnostic_calibration_slope"]
                )
                for season in REPORT_SEASONS
            },
            "season_calibration_intercepts": {
                str(season): float(
                    metrics[season]["diagnostic_calibration_intercept"]
                )
                for season in REPORT_SEASONS
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": float(skills[2024]),
            "uniform_1100_passed": bool(min(skills.values()) >= 1100.0),
        }
    strict = output["strict_temporal_base"]
    for candidate in CANDIDATES:
        current = output[candidate.name]
        current["season_skill_change_vs_strict_base"] = {
            str(season): float(
                current["season_skills"][str(season)]
                - strict["season_skills"][str(season)]
            )
            for season in REPORT_SEASONS
        }
        current["mean_skill_change_vs_strict_base"] = float(
            current["mean_skill"] - strict["mean_skill"]
        )
        current["min_skill_change_vs_strict_base"] = float(
            current["min_skill"] - strict["min_skill"]
        )
        current["improved_every_reported_season"] = bool(
            all(
                value > 0.0
                for value in current[
                    "season_skill_change_vs_strict_base"
                ].values()
            )
        )
    return output


def coefficient_stability(folds: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in ("ridge_residual", "logistic_offset"):
        values = {
            name: {
                str(season): float(
                    folds[str(season)]["models"][model]["coefficients"][
                        "continuous_coefficients_per_training_sd"
                    ][name]
                )
                for season in REPORT_SEASONS
            }
            for name in CONTINUOUS_NAMES
        }
        output[model] = {
            name: {
                "fold_coefficients": fold_values,
                "same_nonzero_sign_2022_2024": bool(
                    len(
                        {
                            int(np.sign(value))
                            for value in fold_values.values()
                            if abs(value) > 1e-12
                        }
                    )
                    == 1
                    and all(abs(value) > 1e-12 for value in fold_values.values())
                ),
            }
            for name, fold_values in values.items()
        }
    return output


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    frame = prepare_frame()
    targets, strict_base, posthoc_r_specific = load_oof(frame)
    seasons = frame["season"].to_numpy(dtype=np.int16)
    continuous = build_continuous(frame)
    context_codes = build_context_codes(frame)
    target_all = np.concatenate(
        [targets[season] for season in EVALUATED_SEASONS]
    )
    strict_all = np.concatenate(
        [strict_base[season] for season in EVALUATED_SEASONS]
    )

    folds: dict[str, Any] = {}
    qa = {
        "candidate_count": len(CANDIDATES),
        "candidate_count_at_most_4": len(CANDIDATES) <= 4,
        "strict_temporal_base_used_for_all_candidates": True,
        "posthoc_Rspecific_reference_never_used_for_fit": True,
        "current_fold_labels_unused_for_fit_scaling_selection": True,
        "source_seasons_strictly_earlier": True,
        "season_equal_weight_totals": True,
        "all_predictions_finite_and_in_0_1": True,
        "all_target_order_matches": True,
        "all_coefficients_finite": True,
        "all_model_fits_successful": True,
        "test_or_validation_row_aggregation": False,
    }
    for validation_season in EVALUATED_SEASONS:
        validation_mask = seasons == validation_season
        training_mask = seasons < validation_season
        source_seasons = sorted(
            np.unique(seasons[training_mask]).astype(int).tolist()
        )
        qa["source_seasons_strictly_earlier"] &= all(
            season < validation_season for season in source_seasons
        )
        target = targets[validation_season]
        base = strict_base[validation_season]
        reference = posthoc_r_specific[validation_season]
        candidate_results: dict[str, Any] = {
            "strict_temporal_base": {
                "role": "immutable deployable temporal base",
                "metrics": calculate_metrics(target, base),
            },
            "posthoc_Rspecific_reference": {
                "role": "read-only post-hoc reference; not a fit offset",
                "metrics": calculate_metrics(target, reference),
            },
        }
        models: dict[str, Any] = {}
        if not training_mask.any():
            extrapolation = {
                "available": False,
                "reason": "no earlier evaluated OOF season",
            }
            any_outside = np.zeros(len(target), dtype=bool)
            any_key_low = np.zeros(len(target), dtype=bool)
            ridge_correction = np.zeros(len(target), dtype=float)
            logistic_delta = np.zeros(len(target), dtype=float)
            zero = np.zeros(
                len(CONTINUOUS_NAMES) + CONTEXT_COUNT, dtype=float
            )
            for model_name in ("ridge_residual", "logistic_offset"):
                models[model_name] = {
                    "training_seasons": [],
                    "fit": {"status": "identity; no earlier OOF"},
                    "coefficients": coefficient_report(zero),
                }
        else:
            train_seasons = seasons[training_mask]
            weights = season_equal_weights(train_seasons)
            totals = [
                float(weights[train_seasons == season].sum())
                for season in source_seasons
            ]
            qa["season_equal_weight_totals"] &= bool(
                max(totals) - min(totals) < 1e-8
            )
            train_design, validation_design, scaling = make_design(
                continuous[training_mask],
                continuous[validation_mask],
                context_codes[training_mask],
                context_codes[validation_mask],
                weights,
            )
            residual_target = centered_source_residual(
                target_all[training_mask],
                strict_all[training_mask],
                train_seasons,
            )
            ridge_coefficient, ridge_fit = fit_ridge(
                train_design, residual_target, weights
            )
            logistic_coefficient, logistic_fit = fit_logistic_offset(
                train_design,
                target_all[training_mask],
                strict_all[training_mask],
                weights,
            )
            qa["all_model_fits_successful"] &= bool(
                logistic_fit["optimizer_success"]
            )
            qa["all_coefficients_finite"] &= bool(
                np.isfinite(ridge_coefficient).all()
                and np.isfinite(logistic_coefficient).all()
            )
            ridge_correction = np.asarray(
                validation_design @ ridge_coefficient
            ).ravel()
            logistic_probability = expit(
                clipped_logit(base)
                + np.asarray(
                    validation_design @ logistic_coefficient
                ).ravel()
            )
            logistic_delta = logistic_probability - base
            extrapolation, any_outside, any_key_low = (
                extrapolation_diagnostics(
                    continuous[training_mask],
                    continuous[validation_mask],
                )
            )
            extrapolation["available"] = True
            models = {
                "ridge_residual": {
                    "training_seasons": source_seasons,
                    "scaling": scaling,
                    "fit": ridge_fit,
                    "coefficients": coefficient_report(
                        ridge_coefficient
                    ),
                    "validation_raw_correction_mean": float(
                        ridge_correction.mean()
                    ),
                    "validation_raw_correction_std": float(
                        ridge_correction.std()
                    ),
                },
                "logistic_offset": {
                    "training_seasons": source_seasons,
                    "scaling": scaling,
                    "fit": logistic_fit,
                    "coefficients": coefficient_report(
                        logistic_coefficient
                    ),
                    "validation_probability_delta_mean": float(
                        logistic_delta.mean()
                    ),
                    "validation_probability_delta_std": float(
                        logistic_delta.std()
                    ),
                },
            }

        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            target,
        )
        np.save(
            ARTIFACT_DIR
            / f"predictions_strict_base_{validation_season}.npy",
            base,
        )
        np.save(
            ARTIFACT_DIR
            / f"correction_ridge_raw_{validation_season}.npy",
            ridge_correction,
        )
        np.save(
            ARTIFACT_DIR
            / f"correction_logistic_delta_{validation_season}.npy",
            logistic_delta,
        )
        for candidate in CANDIDATES:
            raw_delta = (
                ridge_correction
                if candidate.model == "ridge_residual"
                else logistic_delta
            )
            prediction = np.clip(
                base + candidate.blend_weight * raw_delta, 0.0, 1.0
            )
            valid = bool(
                np.isfinite(prediction).all()
                and (prediction >= 0.0).all()
                and (prediction <= 1.0).all()
            )
            qa["all_predictions_finite_and_in_0_1"] &= valid
            candidate_results[candidate.name] = {
                "configuration": asdict(candidate),
                "training_seasons": source_seasons,
                "current_fold_labels_used_for_fit_scaling_or_selection": False,
                "metrics": calculate_metrics(target, prediction),
                "extrapolation_segments": {
                    "any_continuous_outside_training_range": masked_metrics(
                        target, prediction, any_outside
                    ),
                    "inside_all_continuous_training_ranges": masked_metrics(
                        target, prediction, ~any_outside
                    ),
                    "any_key_posterior_below_training_min": masked_metrics(
                        target, prediction, any_key_low
                    ),
                    "base_same_key_low_mask": masked_metrics(
                        target, base, any_key_low
                    ),
                },
            }
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate.name}_{validation_season}.npy",
                prediction,
            )
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "training_seasons": source_seasons,
            "extrapolation": extrapolation,
            "models": models,
            "candidates": candidate_results,
        }

    aggregate = aggregate_metrics(folds)
    ranking = sorted(
        CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate.name]["min_skill"],
            aggregate[candidate.name]["mean_skill"],
        ),
        reverse=True,
    )
    best = ranking[0]
    result = {
        "experiment": "EXP-020",
        "candidate_family": "parametric_clipped_logit_extrapolation",
        "hypothesis": (
            "Linear/logistic clipped-logit bases can extrapolate below the "
            "past OOF posterior range where tree leaves saturate."
        ),
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": (
                "temporal strict_rank_s300 low-rank pitcher-context OOF"
            ),
            "posthoc_reference": (
                "lowrank_s300_r4_Rspecific; read-only and never used as offset"
            ),
            "outer_fit_scaling": "strictly earlier evaluated OOF seasons only",
            "source_weighting": "equal total weight per source season",
            "current_fold_label_calibration_or_selection": False,
            "validation_or_test_row_aggregation": False,
        },
        "predeclared_configuration": {
            "probability_clip": PROBABILITY_CLIP,
            "ridge_alpha": RIDGE_ALPHA,
            "logistic_alpha": LOGISTIC_ALPHA,
            "continuous_specs": [
                {
                    "name": name,
                    "source_column": column,
                    "transform": transform,
                    "fill_value": fill,
                }
                for name, column, transform, fill in CONTINUOUS_SPECS
            ],
            "context": (
                "static balls(0..3) x strikes(0..2) x "
                "pitcher_hand(1..2) x batter_hand(1..2) one-hot"
            ),
            "context_count": CONTEXT_COUNT,
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
            "forbidden_inputs": sorted(FORBIDDEN_INPUTS),
        },
        "folds": folds,
        "coefficient_stability": coefficient_stability(folds),
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "post-hoc diagnostic ranking only; non-nested",
            "posthoc_best_min_candidate": best.name,
            "posthoc_best_min_skill": float(
                aggregate[best.name]["min_skill"]
            ),
            "posthoc_best_mean_skill": float(
                aggregate[best.name]["mean_skill"]
            ),
            "uniform_1100_gate_passed": bool(
                aggregate[best.name]["min_skill"] >= 1100.0
            ),
            "stop_if_1100_gate_failed": bool(
                aggregate[best.name]["min_skill"] < 1100.0
            ),
            "adopt_without_nested_confirmation": False,
        },
        "qa": qa,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "total_seconds": float(time.time() - started),
    }
    output = ARTIFACT_DIR / "validation_metrics.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "strict_base": aggregate["strict_temporal_base"],
                "posthoc_reference": aggregate[
                    "posthoc_Rspecific_reference"
                ],
                "candidates": {
                    candidate.name: aggregate[candidate.name]
                    for candidate in CANDIDATES
                },
                "extrapolation_summary": {
                    str(season): folds[str(season)]["extrapolation"]
                    for season in REPORT_SEASONS
                },
                "selection": result["selection"],
                "qa": qa,
                "seconds": result["total_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
