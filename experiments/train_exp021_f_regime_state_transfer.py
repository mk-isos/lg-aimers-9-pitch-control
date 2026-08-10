"""EXP-021: strict F-regime state-transfer logistic diagnostic.

The immutable prediction base is the saved EXP-020 strict rank-selected
pitcher-context OOF vector.  Only ``game_type == F`` rows can change.  The
transfer model uses fixed, current-row reliability-shrunk season-versus-prior
state deltas.  It is fitted on earlier OOF F rows with one nuisance intercept
per source season; nuisance intercepts are deliberately discarded at
validation so a historical F-level offset cannot be transferred.

The four candidates (two predeclared L2 strengths by two correction scales)
are fixed before evaluation.  A strict candidate path uses only earlier OOF
fold metrics.  Current-fold labels are used only for evaluation and explicitly
labelled post-hoc segment diagnostics.  No test rows or test-row aggregates
are read.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit

import train_exp019_multirate_residual as multirate
from train_exp017_rolling_residual import calculate_metrics


ARTIFACT_DIR = Path("./artifacts/EXP-021/f_regime_state_transfer")
TEAM_ROOT = Path("./artifacts/EXP-019/team_eb_ensemble")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
TEAM_TEMPLATE = "predictions_all_prior_s1000_{season}.npy"
LOWRANK_TEMPLATE = "predictions_strict_rank_s300_{season}.npy"
TARGET_TEMPLATE = "targets_{season}.npy"
EVALUATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
TARGET_SKILL = 1100.0
LOGIT_EPSILON = 1e-6


@dataclass(frozen=True)
class StateFeature:
    name: str
    delta_column: str
    reliability_column: str
    fixed_scale: float
    coefficient_lower: float
    coefficient_upper: float


STATE_FEATURES = (
    StateFeature(
        "pitcher_success_state",
        "temporal_pitcher_season_minus_prior_rate",
        "temporal_pitcher_reliability_30",
        0.10,
        0.0,
        3.0,
    ),
    StateFeature(
        "pitcher_reverse_state",
        "multirate_pitcher_control_reverse_season_minus_prior",
        "multirate_pitcher_control_reliability_30",
        0.10,
        -3.0,
        0.0,
    ),
    StateFeature(
        "pitcher_middle_state",
        "multirate_pitcher_control_middle_season_minus_prior",
        "multirate_pitcher_control_reliability_30",
        0.10,
        -3.0,
        3.0,
    ),
    StateFeature(
        "batter_success_state",
        "temporal_batter_season_minus_prior_rate",
        "temporal_batter_reliability_30",
        0.10,
        0.0,
        3.0,
    ),
    StateFeature(
        "batter_middle_state",
        "multirate_batter_control_middle_season_minus_prior",
        "multirate_batter_control_reliability_30",
        0.10,
        -3.0,
        3.0,
    ),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    ridge: float
    correction_scale: float


# The conservative default is first so exact historical ties are resolved to
# it without looking at the current fold.
CANDIDATES = (
    Candidate("state_logit_l2_0p10_w050", 0.10, 0.50),
    Candidate("state_logit_l2_0p10_w100", 0.10, 1.00),
    Candidate("state_logit_l2_0p01_w050", 0.01, 0.50),
    Candidate("state_logit_l2_0p01_w100", 0.01, 1.00),
)
DEFAULT_CANDIDATE = CANDIDATES[0].name


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability.astype(float), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def build_state_matrix(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    missing = sorted(
        {
            column
            for feature in STATE_FEATURES
            for column in (feature.delta_column, feature.reliability_column)
        }
        - set(frame.columns)
    )
    if missing:
        raise ValueError(f"missing state columns: {missing}")
    columns: list[np.ndarray] = []
    raw_states: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for feature in STATE_FEATURES:
        delta = frame[feature.delta_column].to_numpy(dtype=float)
        reliability = frame[feature.reliability_column].to_numpy(dtype=float)
        if not (np.isfinite(delta).all() and np.isfinite(reliability).all()):
            raise ValueError(f"non-finite state input: {feature.name}")
        if not ((reliability >= 0.0).all() and (reliability <= 1.0).all()):
            raise ValueError(f"invalid reliability: {feature.name}")
        raw = delta * reliability
        scaled = np.clip(raw / feature.fixed_scale, -3.0, 3.0)
        raw_states[feature.name] = raw
        columns.append(scaled)
        diagnostics[feature.name] = {
            "delta_column": feature.delta_column,
            "reliability_column": feature.reliability_column,
            "fixed_scale": feature.fixed_scale,
            "coefficient_bounds": [
                feature.coefficient_lower,
                feature.coefficient_upper,
            ],
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
            "scaled_clipped_rows": int((np.abs(raw / feature.fixed_scale) > 3.0).sum()),
        }
    matrix = np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)
    return matrix, raw_states, diagnostics


def equal_source_season_weights(source_seasons: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(source_seasons), dtype=float)
    unique = np.unique(source_seasons)
    for season in unique:
        mask = source_seasons == season
        weights[mask] = 1.0 / (len(unique) * int(mask.sum()))
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise AssertionError("source-season weights do not sum to one")
    return weights


def fit_hierarchical_state_logit(
    X: np.ndarray,
    targets: np.ndarray,
    base_predictions: np.ndarray,
    source_seasons: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit shared state coefficients plus discarded source-season intercepts."""

    unique_seasons = np.unique(source_seasons).astype(int)
    if len(unique_seasons) == 0:
        return np.zeros(X.shape[1], dtype=float), {
            "status": "no_prior_oof_source_season",
            "success": True,
            "shared_coefficients": [0.0] * X.shape[1],
            "discarded_source_intercepts": {},
        }
    season_positions = {
        int(season): position for position, season in enumerate(unique_seasons)
    }
    season_index = np.array(
        [season_positions[int(season)] for season in source_seasons],
        dtype=np.int16,
    )
    weights = equal_source_season_weights(source_seasons)
    offset = logit(base_predictions)
    n_features = X.shape[1]

    def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
        beta = parameter[:n_features]
        alpha = parameter[n_features:]
        eta = offset + X @ beta + alpha[season_index]
        probability = expit(eta)
        loss = float(
            np.sum(weights * (np.logaddexp(0.0, eta) - targets * eta))
            + 0.5 * ridge * np.dot(beta, beta)
        )
        error = weights * (probability - targets)
        beta_gradient = X.T @ error + ridge * beta
        alpha_gradient = np.bincount(
            season_index,
            weights=error,
            minlength=len(unique_seasons),
        )
        return loss, np.concatenate([beta_gradient, alpha_gradient])

    bounds = [
        (feature.coefficient_lower, feature.coefficient_upper)
        for feature in STATE_FEATURES
    ] + [(-1.5, 1.5)] * len(unique_seasons)
    initial = np.zeros(n_features + len(unique_seasons), dtype=float)
    optimized = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not optimized.success:
        raise RuntimeError(
            f"hierarchical state logistic failed: {optimized.message}"
        )
    beta = optimized.x[:n_features].astype(float)
    _, gradient = objective(optimized.x)
    return beta, {
        "status": "fit",
        "success": bool(optimized.success),
        "message": str(optimized.message),
        "iterations": int(optimized.nit),
        "function_evaluations": int(optimized.nfev),
        "objective": float(optimized.fun),
        "max_absolute_gradient": float(np.max(np.abs(gradient))),
        "ridge": ridge,
        "source_seasons": unique_seasons.tolist(),
        "source_rows": int(len(targets)),
        "source_rows_by_season": {
            str(season): int((source_seasons == season).sum())
            for season in unique_seasons
        },
        "source_equal_weight_sum_by_season": {
            str(season): float(weights[source_seasons == season].sum())
            for season in unique_seasons
        },
        "shared_coefficients": {
            feature.name: float(value)
            for feature, value in zip(STATE_FEATURES, beta, strict=True)
        },
        "discarded_source_intercepts": {
            str(season): float(optimized.x[n_features + position])
            for position, season in enumerate(unique_seasons)
        },
        "source_intercepts_transferred_to_validation": False,
    }


def load_saved_oof(
    y: np.ndarray,
    seasons: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    team: dict[int, np.ndarray] = {}
    lowrank: dict[int, np.ndarray] = {}
    qa: dict[str, Any] = {}
    for season in EVALUATION_SEASONS:
        mask = seasons == season
        current_targets = y[mask].astype(np.int8)
        team_path = TEAM_ROOT / TEAM_TEMPLATE.format(season=season)
        lowrank_path = LOWRANK_ROOT / LOWRANK_TEMPLATE.format(season=season)
        team_targets_path = TEAM_ROOT / TARGET_TEMPLATE.format(season=season)
        lowrank_targets_path = LOWRANK_ROOT / TARGET_TEMPLATE.format(season=season)
        arrays = {
            "team": np.load(team_path).astype(float),
            "lowrank": np.load(lowrank_path).astype(float),
            "team_targets": np.load(team_targets_path).astype(np.int8),
            "lowrank_targets": np.load(lowrank_targets_path).astype(np.int8),
        }
        if not (
            arrays["team"].shape
            == arrays["lowrank"].shape
            == arrays["team_targets"].shape
            == arrays["lowrank_targets"].shape
            == current_targets.shape
        ):
            raise ValueError(f"saved OOF shape mismatch: {season}")
        target_parity = bool(
            np.array_equal(arrays["team_targets"], current_targets)
            and np.array_equal(arrays["lowrank_targets"], current_targets)
        )
        finite = bool(
            np.isfinite(arrays["team"]).all()
            and np.isfinite(arrays["lowrank"]).all()
        )
        in_range = bool(
            ((arrays["team"] >= 0.0) & (arrays["team"] <= 1.0)).all()
            and ((arrays["lowrank"] >= 0.0) & (arrays["lowrank"] <= 1.0)).all()
        )
        if not (target_parity and finite and in_range):
            raise ValueError(f"invalid saved OOF: {season}")
        team[season] = arrays["team"]
        lowrank[season] = arrays["lowrank"]
        qa[str(season)] = {
            "rows": int(mask.sum()),
            "team_path": str(team_path),
            "lowrank_path": str(lowrank_path),
            "team_target_path": str(team_targets_path),
            "lowrank_target_path": str(lowrank_targets_path),
            "target_and_order_parity": target_parity,
            "finite": finite,
            "probabilities_in_0_1": in_range,
        }
    return team, lowrank, qa


def regime_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    game_types: np.ndarray,
) -> dict[str, dict[str, float]]:
    output = {"full": calculate_metrics(targets, predictions)}
    for regime in ("F", "R"):
        mask = game_types == regime
        output[regime] = calculate_metrics(targets[mask], predictions[mask])
    return output


def brier_gap_decomposition(
    targets: np.ndarray,
    predictions: np.ndarray,
    game_types: np.ndarray,
) -> dict[str, Any]:
    metrics = regime_metrics(targets, predictions, game_types)
    threshold = float(
        metrics["full"]["baseline_brier"]
        * (1.0 - TARGET_SKILL / 100000.0)
    )
    total_gap = float(metrics["full"]["brier_score"] - threshold)
    total_squared_error = float(
        np.sum((predictions.astype(float) - targets.astype(float)) ** 2)
    )
    contributions: dict[str, Any] = {}
    contribution_sum = 0.0
    for regime in ("F", "R"):
        local = game_types == regime
        signed = float(
            local.mean() * (metrics[regime]["brier_score"] - threshold)
        )
        contribution_sum += signed
        squared_error = float(
            np.sum((predictions[local].astype(float) - targets[local]) ** 2)
        )
        contributions[regime] = {
            "row_share": float(local.mean()),
            "squared_error_share": float(squared_error / total_squared_error),
            "signed_full_brier_gap_contribution": signed,
            "share_of_signed_total_gap": (
                float(signed / total_gap) if abs(total_gap) > 1e-15 else None
            ),
        }
    parity_difference = float(contribution_sum - total_gap)
    if abs(parity_difference) > 1e-12:
        raise AssertionError("F/R Brier-gap decomposition failed")
    return {
        "skill_1100_brier_threshold": threshold,
        "full_brier_minus_threshold": total_gap,
        "regime_contributions": contributions,
        "contribution_sum_parity_difference": parity_difference,
    }


def base_change_decomposition(
    targets: np.ndarray,
    team: np.ndarray,
    lowrank: np.ndarray,
    game_types: np.ndarray,
) -> dict[str, Any]:
    full_difference = float(
        np.mean((lowrank - targets) ** 2) - np.mean((team - targets) ** 2)
    )
    contributions: dict[str, float] = {}
    for regime in ("F", "R"):
        local = game_types == regime
        contributions[regime] = float(
            local.mean()
            * (
                np.mean((lowrank[local] - targets[local]) ** 2)
                - np.mean((team[local] - targets[local]) ** 2)
            )
        )
    parity = float(sum(contributions.values()) - full_difference)
    if abs(parity) > 1e-12:
        raise AssertionError("team-to-lowrank decomposition failed")
    return {
        "lowrank_minus_team_full_brier": full_difference,
        "regime_contributions": contributions,
        "contribution_sum_parity_difference": parity,
    }


def fixed_state_bin(values: np.ndarray) -> np.ndarray:
    return np.select(
        [values < -0.02, values > 0.02],
        ["down", "up"],
        default="flat",
    ).astype(str)


def segment_labels(
    frame: pd.DataFrame,
    raw_states: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    pitcher_n = frame["temporal_pitcher_season_n"].to_numpy(dtype=float)
    pitcher_prior = frame["temporal_pitcher_prior_exists"].to_numpy(dtype=np.int8)
    batter_prior = frame["temporal_batter_prior_exists"].to_numpy(dtype=np.int8)
    n_label = np.select(
        [
            pitcher_n == 0,
            (pitcher_n >= 1) & (pitcher_n < 20),
            (pitcher_n >= 20) & (pitcher_n < 100),
            (pitcher_n >= 100) & (pitcher_n < 500),
        ],
        ["n_0", "n_1_19", "n_20_99", "n_100_499"],
        default="n_500_plus",
    ).astype(str)
    new_profile = np.select(
        [
            (pitcher_prior == 1) & (batter_prior == 1),
            (pitcher_prior == 0) & (batter_prior == 1),
            (pitcher_prior == 1) & (batter_prior == 0),
        ],
        ["both_existing", "pitcher_new_only", "batter_new_only"],
        default="both_new",
    ).astype(str)
    pitcher_state = fixed_state_bin(raw_states["pitcher_success_state"])
    batter_state = fixed_state_bin(raw_states["batter_success_state"])
    joint_profile = np.char.add(
        np.char.add("pitcher_", pitcher_state),
        np.char.add("__batter_", batter_state),
    )
    return {
        "month": np.char.add(
            "month_",
            frame["game_month"].to_numpy(dtype=np.int16).astype(str),
        ),
        "pitcher_season_n": n_label,
        "new_player_profile": new_profile,
        "state_profile": joint_profile,
    }


def f_segment_metrics(
    labels: dict[str, np.ndarray],
    f_mask: np.ndarray,
    targets: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dimension, values in labels.items():
        dimension_output: dict[str, Any] = {}
        for label in sorted(np.unique(values[f_mask]).tolist()):
            mask = f_mask & (values == label)
            if not mask.any():
                continue
            model_metrics = {
                name: calculate_metrics(targets[mask], prediction[mask])
                for name, prediction in predictions.items()
            }
            item: dict[str, Any] = {
                "rows": int(mask.sum()),
                "models": model_metrics,
            }
            if "team_base" in model_metrics and "strict_lowrank_base" in model_metrics:
                item["strict_lowrank_minus_team_brier"] = float(
                    model_metrics["strict_lowrank_base"]["brier_score"]
                    - model_metrics["team_base"]["brier_score"]
                )
            if "strict_selected" in model_metrics:
                item["strict_selected_minus_lowrank_brier"] = float(
                    model_metrics["strict_selected"]["brier_score"]
                    - model_metrics["strict_lowrank_base"]["brier_score"]
                )
            dimension_output[label] = item
        output[dimension] = dimension_output
    return output


def choose_strict_candidate(
    candidate_folds: dict[str, dict[int, dict[str, Any]]],
    history: list[int],
) -> tuple[str, dict[str, Any]]:
    if not history:
        return DEFAULT_CANDIDATE, {
            "selection_rule": "predeclared conservative default; no earlier OOF fold",
            "history_seasons": [],
            "candidate_objectives": {},
        }
    objectives: dict[str, Any] = {}
    priority = {candidate.name: index for index, candidate in enumerate(CANDIDATES)}
    for candidate in CANDIDATES:
        normalized_briers = [
            float(
                candidate_folds[candidate.name][season]["full"]["brier_score"]
                / candidate_folds[candidate.name][season]["full"]["baseline_brier"]
            )
            for season in history
        ]
        objectives[candidate.name] = {
            "historical_normalized_briers": normalized_briers,
            "worst_normalized_brier": float(max(normalized_briers)),
            "mean_normalized_brier": float(np.mean(normalized_briers)),
            "latest_normalized_brier": float(normalized_briers[-1]),
        }
    selected = min(
        (candidate.name for candidate in CANDIDATES),
        key=lambda name: (
            objectives[name]["worst_normalized_brier"],
            objectives[name]["mean_normalized_brier"],
            objectives[name]["latest_normalized_brier"],
            priority[name],
        ),
    )
    return selected, {
        "selection_rule": (
            "lexicographic minimum prior-fold worst, mean, then latest "
            "normalized full Brier; predeclared priority breaks exact ties"
        ),
        "history_seasons": history,
        "candidate_objectives": objectives,
    }


def aggregate_metrics(
    folds: dict[int, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for regime in ("full", "F", "R"):
        skills = {
            str(season): float(folds[season][regime]["skill_score_unclipped"])
            for season in REPORT_SEASONS
        }
        briers = {
            str(season): float(folds[season][regime]["brier_score"])
            for season in REPORT_SEASONS
        }
        output[regime] = {
            "season_skills": skills,
            "season_briers": briers,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": float(skills["2024"]),
            "mean_brier": float(np.mean(list(briers.values()))),
        }
    return output


def summarize_f_structure(
    base_diagnostic: dict[str, Any],
    folds: dict[str, Any],
) -> dict[str, Any]:
    direction_by_season: dict[str, Any] = {}
    segment_rankings: dict[str, Any] = {}
    for season in REPORT_SEASONS:
        season_key = str(season)
        residual = float(
            base_diagnostic[season_key]["F_prediction_residual_direction"]
            ["lowrank_actual_minus_prediction_mean"]
        )
        direction_by_season[season_key] = {
            "actual_minus_prediction_mean": residual,
            "direction": (
                "base_underpredicts_F"
                if residual > 0.0
                else "base_overpredicts_F"
            ),
        }
        threshold = float(
            base_diagnostic[season_key]["strict_lowrank_gap_decomposition"]
            ["skill_1100_brier_threshold"]
        )
        total_rows = int(
            folds[season_key]["bases"]["strict_lowrank_s300"]["full"]["rows"]
        )
        season_rankings: dict[str, Any] = {}
        for dimension, bins in base_diagnostic[season_key]["F_segments"].items():
            entries = []
            contribution_sum = 0.0
            for label, item in bins.items():
                metrics = item["models"]["strict_lowrank_base"]
                contribution = float(
                    item["rows"]
                    / total_rows
                    * (metrics["brier_score"] - threshold)
                )
                contribution_sum += contribution
                entries.append(
                    {
                        "segment": label,
                        "rows": int(item["rows"]),
                        "brier_score": float(metrics["brier_score"]),
                        "actual_rate": float(metrics["actual_rate"]),
                        "prediction_mean": float(metrics["prediction_mean"]),
                        "signed_full_brier_gap_contribution": contribution,
                    }
                )
            entries.sort(
                key=lambda item: item["signed_full_brier_gap_contribution"],
                reverse=True,
            )
            f_contribution = float(
                base_diagnostic[season_key]["strict_lowrank_gap_decomposition"]
                ["regime_contributions"]["F"]
                ["signed_full_brier_gap_contribution"]
            )
            parity = float(contribution_sum - f_contribution)
            if abs(parity) > 1e-12:
                raise AssertionError(
                    f"F segment contribution parity failed: {season} {dimension}"
                )
            season_rankings[dimension] = {
                "descending_signed_gap_contribution": entries,
                "contribution_sum": contribution_sum,
                "F_contribution_parity_difference": parity,
            }
        segment_rankings[season_key] = season_rankings
    signs = {
        season: np.sign(
            direction_by_season[str(season)]["actual_minus_prediction_mean"]
        )
        for season in REPORT_SEASONS
    }
    return {
        "lowrank_F_direction_by_season": direction_by_season,
        "2022_direction_reverses_in_both_2023_and_2024": bool(
            signs[2022] != 0
            and signs[2023] != 0
            and signs[2024] != 0
            and signs[2022] == -signs[2023]
            and signs[2022] == -signs[2024]
        ),
        "strict_lowrank_F_segment_gap_rankings": segment_rankings,
    }


def main() -> None:
    started = time.time()
    frame, _, y, _, seasons, reconstruction = multirate.prepare_multirate_data()
    game_types = frame["game_type"].astype(str).to_numpy()
    is_f = game_types == "F"
    is_r = game_types == "R"
    if not np.array_equal(~is_f, is_r):
        raise ValueError("expected exactly F/R game types")
    X, raw_states, state_diagnostics = build_state_matrix(frame)
    labels = segment_labels(frame, raw_states)
    team_oof, lowrank_oof, source_qa = load_saved_oof(y, seasons)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_predictions: dict[str, dict[int, np.ndarray]] = {
        candidate.name: {} for candidate in CANDIDATES
    }
    candidate_folds: dict[str, dict[int, dict[str, Any]]] = {
        candidate.name: {} for candidate in CANDIDATES
    }
    folds: dict[str, Any] = {}
    base_diagnostic: dict[str, Any] = {}

    for validation_season in EVALUATION_SEASONS:
        validation_mask = seasons == validation_season
        validation_f = validation_mask & is_f
        local_f = is_f[validation_mask]
        local_types = game_types[validation_mask]
        targets = y[validation_mask].astype(float)
        team = team_oof[validation_season]
        base = lowrank_oof[validation_season]
        source_season_list = [
            season for season in EVALUATION_SEASONS if season < validation_season
        ]
        source_mask = np.isin(seasons, source_season_list) & is_f
        ridge_fits: dict[float, tuple[np.ndarray, dict[str, Any]]] = {}
        for ridge in sorted({candidate.ridge for candidate in CANDIDATES}):
            if source_mask.any():
                source_base = np.empty(int(source_mask.sum()), dtype=float)
                cursor = 0
                for source_season in source_season_list:
                    source_local_f = is_f[seasons == source_season]
                    values = lowrank_oof[source_season][source_local_f]
                    source_base[cursor : cursor + len(values)] = values
                    cursor += len(values)
                if cursor != len(source_base):
                    raise AssertionError("source OOF concatenation mismatch")
                ridge_fits[ridge] = fit_hierarchical_state_logit(
                    X[source_mask],
                    y[source_mask].astype(float),
                    source_base,
                    seasons[source_mask],
                    ridge,
                )
            else:
                ridge_fits[ridge] = fit_hierarchical_state_logit(
                    np.empty((0, X.shape[1]), dtype=float),
                    np.empty(0, dtype=float),
                    np.empty(0, dtype=float),
                    np.empty(0, dtype=np.int16),
                    ridge,
                )

        fold: dict[str, Any] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_season_list,
            "source_F_rows": int(source_mask.sum()),
            "validation_F_rows": int(validation_f.sum()),
            "validation_R_rows": int((validation_mask & is_r).sum()),
            "current_fold_labels_used_for_model_fit": False,
            "ridge_fits": {
                str(ridge): diagnostics for ridge, (_, diagnostics) in ridge_fits.items()
            },
            "bases": {
                "team_allprior": regime_metrics(targets, team, local_types),
                "strict_lowrank_s300": regime_metrics(targets, base, local_types),
            },
            "candidates": {},
        }
        for candidate in CANDIDATES:
            beta, _ = ridge_fits[candidate.ridge]
            correction = X[validation_f] @ beta
            prediction = base.copy()
            prediction[local_f] = expit(
                logit(base[local_f]) + candidate.correction_scale * correction
            )
            r_exact = bool(np.array_equal(prediction[~local_f], base[~local_f]))
            if not r_exact:
                raise AssertionError("F candidate changed an R row")
            if not (
                np.isfinite(prediction).all()
                and ((prediction >= 0.0) & (prediction <= 1.0)).all()
            ):
                raise ValueError("invalid candidate probability")
            metrics = regime_metrics(targets, prediction, local_types)
            candidate_predictions[candidate.name][validation_season] = prediction
            candidate_folds[candidate.name][validation_season] = metrics
            fold["candidates"][candidate.name] = {
                "ridge": candidate.ridge,
                "correction_scale": candidate.correction_scale,
                "metrics": metrics,
                "F_logit_correction": {
                    "mean": float(correction.mean()) if len(correction) else 0.0,
                    "std": float(correction.std()) if len(correction) else 0.0,
                    "min": float(correction.min()) if len(correction) else 0.0,
                    "max": float(correction.max()) if len(correction) else 0.0,
                },
                "R_exactly_equals_strict_lowrank_base": r_exact,
            }
            np.save(
                ARTIFACT_DIR / f"predictions_{candidate.name}_{validation_season}.npy",
                prediction,
            )
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets)
        folds[str(validation_season)] = fold

        if validation_season in REPORT_SEASONS:
            local_labels = {
                name: values[validation_mask] for name, values in labels.items()
            }
            base_diagnostic[str(validation_season)] = {
                "posthoc_current_fold_label_diagnostic_only": True,
                "F_actual_rate": float(targets[local_f].mean()),
                "F_prediction_residual_direction": {
                    "team_actual_minus_prediction_mean": float(
                        targets[local_f].mean() - team[local_f].mean()
                    ),
                    "lowrank_actual_minus_prediction_mean": float(
                        targets[local_f].mean() - base[local_f].mean()
                    ),
                },
                "team_gap_decomposition": brier_gap_decomposition(
                    targets, team, local_types
                ),
                "strict_lowrank_gap_decomposition": brier_gap_decomposition(
                    targets, base, local_types
                ),
                "strict_lowrank_vs_team_decomposition": base_change_decomposition(
                    targets, team, base, local_types
                ),
                "F_segments": f_segment_metrics(
                    local_labels,
                    local_f,
                    targets,
                    {
                        "team_base": team,
                        "strict_lowrank_base": base,
                    },
                ),
            }
        print(
            f"EXP-021 F-state {validation_season}: base="
            f"{fold['bases']['strict_lowrank_s300']['full']['skill_score_unclipped']:.2f} "
            + " ".join(
                f"{candidate.name}="
                f"{fold['candidates'][candidate.name]['metrics']['full']['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    strict_predictions: dict[int, np.ndarray] = {}
    strict_folds: dict[int, dict[str, Any]] = {}
    strict_selection: dict[str, Any] = {}
    for season in EVALUATION_SEASONS:
        history = [value for value in EVALUATION_SEASONS if value < season]
        selected, selection_diagnostics = choose_strict_candidate(
            candidate_folds, history
        )
        prediction = candidate_predictions[selected][season]
        mask = seasons == season
        targets = y[mask].astype(float)
        local_types = game_types[mask]
        metrics = regime_metrics(targets, prediction, local_types)
        strict_predictions[season] = prediction
        strict_folds[season] = metrics
        strict_selection[str(season)] = {
            "selected_candidate": selected,
            "current_fold_labels_used_for_selection": False,
            **selection_diagnostics,
            "metrics": metrics,
            "R_exactly_equals_strict_lowrank_base": bool(
                np.array_equal(
                    prediction[local_types == "R"],
                    lowrank_oof[season][local_types == "R"],
                )
            ),
        }
        np.save(
            ARTIFACT_DIR / f"predictions_strict_selected_{season}.npy",
            prediction,
        )
        if season in REPORT_SEASONS:
            local_labels = {
                name: values[mask] for name, values in labels.items()
            }
            base_diagnostic[str(season)]["strict_selected_candidate"] = selected
            base_diagnostic[str(season)]["strict_selected_F_segments"] = (
                f_segment_metrics(
                    local_labels,
                    local_types == "F",
                    targets,
                    {
                        "strict_lowrank_base": lowrank_oof[season],
                        "strict_selected": prediction,
                    },
                )
            )

    prospective_candidate, prospective_diagnostics = choose_strict_candidate(
        candidate_folds, list(EVALUATION_SEASONS)
    )
    aggregate_candidates = {
        candidate.name: aggregate_metrics(candidate_folds[candidate.name])
        for candidate in CANDIDATES
    }
    strict_aggregate = aggregate_metrics(strict_folds)
    base_folds = {
        season: regime_metrics(
            y[seasons == season].astype(float),
            lowrank_oof[season],
            game_types[seasons == season],
        )
        for season in EVALUATION_SEASONS
    }
    base_aggregate = aggregate_metrics(base_folds)
    uniform_gate_passed = bool(strict_aggregate["full"]["min_skill"] >= TARGET_SKILL)
    f_structure_summary = summarize_f_structure(base_diagnostic, folds)

    result: dict[str, Any] = {
        "experiment": "EXP-021",
        "candidate_family": "F_hierarchical_state_transfer_logistic",
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATION_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": "EXP-020 strict_rank_s300 OOF",
            "team_reference": "EXP-019 team all_prior_s1000 OOF",
            "model_fit_rows": "earlier evaluated OOF game_type=F rows only",
            "correction_application": "current validation game_type=F rows only",
            "R_prediction": "bitwise exact immutable strict lowrank base",
            "source_season_handling": (
                "equal source-season weights; nuisance source intercepts fit "
                "and discarded; no historical F offset transferred"
            ),
            "current_fold_labels_used_for_model_fit": False,
            "current_fold_labels_used_for_candidate_selection": False,
            "candidate_selection": (
                "strict prior-OOF lexicographic minimax normalized full Brier"
            ),
            "candidate_count": len(CANDIDATES),
            "candidate_grid_predeclared": True,
            "test_rows_read": False,
            "test_row_aggregation": False,
            "current_validation_row_aggregation_for_prediction": False,
            "posthoc_segment_labels_used_for_model_or_selection": False,
        },
        "model": {
            "type": "offset hierarchical logistic state transfer",
            "feature_transform": (
                "fixed clip(delta * reliability / fixed_scale, -3, 3)"
            ),
            "features": state_diagnostics,
            "candidate_definitions": [
                {
                    "name": candidate.name,
                    "ridge": candidate.ridge,
                    "correction_scale": candidate.correction_scale,
                }
                for candidate in CANDIDATES
            ],
            "coefficient_constraints": (
                "pitcher/batter success nonnegative; pitcher reverse "
                "nonpositive; middle-state coefficients bounded unconstrained"
            ),
            "validation_logit": (
                "logit(strict_lowrank_base) + scale * row_state @ shared_beta"
            ),
            "transferred_intercept": 0.0,
        },
        "F_structure_diagnostic": {
            "status": (
                "same-fold label post-hoc diagnostic; never used for model "
                "features, fitting, weights, or candidate selection"
            ),
            "fixed_segment_definitions": {
                "month": "official current-row game_month",
                "pitcher_season_n": "0 / 1-19 / 20-99 / 100-499 / 500+",
                "new_player_profile": (
                    "pitcher/batter prior_exists four-way current-row profile"
                ),
                "state_profile": (
                    "pitcher and batter reliability-shrunk success deltas, "
                    "fixed down<-0.02 / flat / up>0.02"
                ),
            },
            "machine_summary": f_structure_summary,
            "folds": base_diagnostic,
        },
        "folds": folds,
        "aggregate_2022_2024": {
            "strict_lowrank_base": base_aggregate,
            "candidates": aggregate_candidates,
            "strict_selected": strict_aggregate,
        },
        "strict_selection_path": {
            "folds": strict_selection,
            "prospective_2025_candidate": prospective_candidate,
            "prospective_selection_uses_2025_labels": False,
            "prospective_selection_diagnostics": prospective_diagnostics,
        },
        "decision": {
            "target_minimum_skill": TARGET_SKILL,
            "strict_selected_min_skill": strict_aggregate["full"]["min_skill"],
            "uniform_1100_gate_passed": uniform_gate_passed,
            "status": (
                "continue_only_if_uniform_gate_passed"
                if uniform_gate_passed
                else "stopped_below_1100_gate"
            ),
        },
        "qa": {
            "source_oof": source_qa,
            "all_R_predictions_exact_base": bool(
                all(
                    strict_selection[str(season)][
                        "R_exactly_equals_strict_lowrank_base"
                    ]
                    for season in EVALUATION_SEASONS
                )
            ),
            "all_saved_targets_match_training_order": True,
            "all_predictions_finite_and_in_0_1": True,
            "candidate_count_at_most_4": len(CANDIDATES) <= 4,
            "strict_selection_history_excludes_current_fold": bool(
                all(
                    all(
                        source < int(season)
                        for source in item["history_seasons"]
                    )
                    for season, item in strict_selection.items()
                )
            ),
            "no_transferred_source_intercepts": True,
            "no_test_rows_loaded": True,
        },
        "reconstruction_diagnostics": reconstruction,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "total_seconds": time.time() - started,
    }
    output_path = ARTIFACT_DIR / "validation_metrics.json"
    output_path.write_text(
        json.dumps(to_builtin(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "strict_full_skills": strict_aggregate["full"]["season_skills"],
                "strict_F_skills": strict_aggregate["F"]["season_skills"],
                "strict_R_skills": strict_aggregate["R"]["season_skills"],
                "strict_min_skill": strict_aggregate["full"]["min_skill"],
                "prospective_2025_candidate": prospective_candidate,
                "decision": result["decision"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
