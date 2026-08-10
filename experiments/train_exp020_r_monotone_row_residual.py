"""EXP-020: constrained R-only row-context residual atop strict low-rank OOF.

The immutable base is ``strict_rank_s300`` from the temporal low-rank
pitcher-by-count-context experiment.  A read-only same-fold signal audit is
stored for hypothesis documentation only.  The deployable rolling path fits
each source-season model independently and uses only evaluated OOF seasons
strictly earlier than the outer validation season.

The bounded candidate family is fixed at four variants.  All models are
non-negative ridge regressions on direction-coded current-row features.  The
source R residual and source features are centered inside the source season;
source-season corrections are averaged equally.  Corrections are applied to
official ``game_type == R`` rows only, while F remains bitwise equal to the
immutable base.  No raw player/team ID, current-fold aggregate, test-row
aggregate, calibration offset, Trackman feature, or current-fold label enters
training or fold-specific model selection.

The audit that motivated this family makes the overall experiment non-nested;
candidate comparison is diagnostic and cannot itself justify adoption.
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

from train_exp017_rolling_residual import calculate_metrics


DATA_PATH = Path("./data/train.csv")
BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
TEAM_ROOT = Path("./artifacts/EXP-019/team_eb_ensemble")
ARTIFACT_DIR = Path("./artifacts/EXP-020/r_monotone_row_residual")

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
RIDGE_ALPHA = 20000.0

RAW_COLUMNS = (
    "season",
    "game_type",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "li",
    "control_success",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    features: tuple[str, ...]
    correction_weight: float


CORE_FEATURES = ("outs_positive", "earlier_inning")
STABLE_FEATURES = (*CORE_FEATURES, "risp", "log_li")
LATENT_FEATURES = (*STABLE_FEATURES, "latent_late")
CANDIDATES = (
    Candidate("core2_w100", CORE_FEATURES, 1.0),
    Candidate("stable4_w100", STABLE_FEATURES, 1.0),
    Candidate("latent5_w050", LATENT_FEATURES, 0.5),
    Candidate("latent5_w100", LATENT_FEATURES, 1.0),
)
FEATURE_SETS = {
    "core2": CORE_FEATURES,
    "stable4": STABLE_FEATURES,
    "latent5": LATENT_FEATURES,
}


def load_rows() -> pd.DataFrame:
    frame = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=list(RAW_COLUMNS),
    )
    frame = frame.loc[
        frame["season"].isin(EVALUATED_SEASONS)
    ].reset_index(drop=True)
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("evaluation rows must remain season sorted")
    required = [column for column in RAW_COLUMNS if column != "base_state"]
    if frame[required].isna().any().any():
        raise ValueError("missing required current-row field")
    if set(frame["game_type"].astype(str).unique()) != {"F", "R"}:
        raise ValueError("unexpected game_type domain")
    return frame


def load_oof(
    frame: pd.DataFrame,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    team: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        rows = frame["season"].eq(season).to_numpy()
        targets[season] = np.load(
            BASE_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        base[season] = np.load(
            BASE_ROOT / f"predictions_strict_rank_s300_{season}.npy"
        ).astype(np.float64)
        team[season] = np.load(
            TEAM_ROOT / f"predictions_all_prior_s1000_{season}.npy"
        ).astype(np.float64)
        csv_target = frame.loc[rows, "control_success"].to_numpy(
            dtype=np.float64
        )
        if not (
            len(csv_target)
            == len(targets[season])
            == len(base[season])
            == len(team[season])
            and np.array_equal(csv_target, targets[season])
        ):
            raise ValueError(f"OOF target/order mismatch for {season}")
        for label, prediction in (("base", base[season]), ("team", team[season])):
            if not np.isfinite(prediction).all() or not (
                (prediction >= 0.0).all()
                and (prediction <= 1.0).all()
            ):
                raise ValueError(f"invalid {label} prediction for {season}")
    return targets, base, team


def build_features(
    frame: pd.DataFrame,
    base_all: np.ndarray,
    team_all: np.ndarray,
) -> pd.DataFrame:
    latent = base_all - team_all
    inning = frame["inning"].clip(lower=1, upper=12).to_numpy(dtype=float)
    balls = frame["balls_before"].to_numpy(dtype=float)
    strikes = frame["strikes_before"].to_numpy(dtype=float)
    outs = frame["outs_before"].to_numpy(dtype=float)
    score_diff = frame["score_diff_pitcher_team"].to_numpy(dtype=float)
    risp = (
        frame["runner_on_2b"].eq(1) | frame["runner_on_3b"].eq(1)
    ).to_numpy(dtype=float)
    bases_loaded = (
        frame["runner_on_1b"].eq(1)
        & frame["runner_on_2b"].eq(1)
        & frame["runner_on_3b"].eq(1)
    ).to_numpy(dtype=float)
    log_li = np.log1p(
        frame["li"].clip(lower=0.0, upper=100.0).to_numpy(dtype=float)
    )
    late = (inning >= 7.0).astype(float)
    return pd.DataFrame(
        {
            # Positive coefficients encode the audited residual directions.
            "outs_positive": outs,
            "earlier_inning": -inning / 4.0,
            "risp": risp,
            "log_li": log_li,
            "latent_late": latent * late,
            # Diagnostic-only universe below.
            "count_progress": balls - strikes,
            "count_depth": balls + strikes,
            "two_strikes": (strikes == 2.0).astype(float),
            "three_balls": (balls == 3.0).astype(float),
            "full_count": (
                (balls == 3.0) & (strikes == 2.0)
            ).astype(float),
            "score_diff": np.clip(score_diff, -6.0, 6.0),
            "abs_score_diff": np.clip(np.abs(score_diff), 0.0, 8.0),
            "close_game": (np.abs(score_diff) <= 1.0).astype(float),
            "inning": inning,
            "late_inning": late,
            "outs": outs,
            "runners": frame["num_runners_on"].to_numpy(dtype=float),
            "bases_loaded": bases_loaded,
            "latent": latent,
            "latent_count_progress": latent * (balls - strikes),
            "latent_count_depth": latent * (balls + strikes),
            "latent_two_strikes": latent * (strikes == 2.0),
            "latent_score_diff": latent * np.clip(score_diff, -6.0, 6.0),
            "latent_abs_score_diff": latent * np.clip(
                np.abs(score_diff), 0.0, 8.0
            ),
            "latent_inning": latent * inning,
            "latent_outs": latent * outs,
            "latent_risp": latent * risp,
        },
        copy=False,
    )


def samefold_signal_audit(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    targets: dict[int, np.ndarray],
    base: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Diagnostic only: never consumed by the rolling fit."""
    diagnostic_columns = (
        "count_progress",
        "count_depth",
        "two_strikes",
        "three_balls",
        "full_count",
        "score_diff",
        "abs_score_diff",
        "close_game",
        "inning",
        "late_inning",
        "outs",
        "runners",
        "risp",
        "bases_loaded",
        "log_li",
        "latent",
        "latent_count_progress",
        "latent_count_depth",
        "latent_two_strikes",
        "latent_score_diff",
        "latent_abs_score_diff",
        "latent_inning",
        "latent_late",
        "latent_outs",
        "latent_risp",
    )
    seasons = frame["season"].to_numpy(dtype=np.int16)
    game_types = frame["game_type"].astype(str).to_numpy()
    output: dict[str, Any] = {
        "status": "same-fold diagnostic only; not an outer-fold fit input",
        "univariate_slope_per_within_segment_sd": {},
        "game_type_residual_means": {},
        "base_state_residual_means": {},
    }
    for column in diagnostic_columns:
        values = features[column].to_numpy(dtype=float)
        by_regime: dict[str, Any] = {}
        for regime in ("ALL", "R", "F"):
            slopes: dict[str, float] = {}
            for season in EVALUATED_SEASONS:
                season_mask = seasons == season
                local_values = values[season_mask]
                local_types = game_types[season_mask]
                local_residual = targets[season] - base[season]
                if regime != "ALL":
                    regime_mask = local_types == regime
                    local_values = local_values[regime_mask]
                    local_residual = local_residual[regime_mask]
                scale = float(local_values.std())
                if scale <= 1e-15:
                    slope = 0.0
                else:
                    standardized = (
                        local_values - float(local_values.mean())
                    ) / scale
                    centered_residual = (
                        local_residual - float(local_residual.mean())
                    )
                    slope = float(
                        np.mean(standardized * centered_residual)
                    )
                slopes[str(season)] = slope
            reported = [slopes[str(season)] for season in REPORT_SEASONS]
            nonzero = [value for value in reported if abs(value) > 1e-15]
            by_regime[regime] = {
                "season_slopes": slopes,
                "reported_2022_2024_same_positive_sign": bool(
                    len(nonzero) == len(reported)
                    and all(value > 0.0 for value in nonzero)
                ),
                "reported_2022_2024_same_negative_sign": bool(
                    len(nonzero) == len(reported)
                    and all(value < 0.0 for value in nonzero)
                ),
                "reported_mean_absolute_slope": float(
                    np.mean(np.abs(reported))
                ),
            }
        output["univariate_slope_per_within_segment_sd"][column] = by_regime

    for season in EVALUATED_SEASONS:
        season_mask = seasons == season
        residual = targets[season] - base[season]
        local_types = game_types[season_mask]
        output["game_type_residual_means"][str(season)] = {
            regime: {
                "rows": int(np.sum(local_types == regime)),
                "mean_residual": float(np.mean(residual[local_types == regime])),
            }
            for regime in ("R", "F")
        }
        local_states = frame.loc[season_mask, "base_state"].fillna("nan").astype(str)
        state_frame = pd.DataFrame(
            {"base_state": local_states.to_numpy(), "residual": residual}
        )
        state_stats = state_frame.groupby("base_state", sort=True)["residual"].agg(
            ["mean", "count"]
        )
        output["base_state_residual_means"][str(season)] = {
            str(state): {
                "rows": int(row["count"]),
                "mean_residual": float(row["mean"]),
            }
            for state, row in state_stats.iterrows()
        }
    return output


def fit_source_model(
    feature_values: np.ndarray,
    target_residual: np.ndarray,
    feature_names: tuple[str, ...],
    source_season: int,
) -> dict[str, Any]:
    means = feature_values.mean(axis=0)
    scales = feature_values.std(axis=0)
    safe_scales = np.where(scales > 1e-12, scales, 1.0)
    design = (feature_values - means) / safe_scales
    centered_target = target_residual - float(target_residual.mean())
    gram = design.T @ design + RIDGE_ALPHA * np.eye(len(feature_names))
    cross = design.T @ centered_target

    # With at most five features, enumerating every active set gives a
    # deterministic exact solution of the positive-definite non-negative QP
    # and avoids optimizer convergence ambiguity.
    feature_count = len(feature_names)
    feasible: list[tuple[float, np.ndarray]] = []
    kkt_tolerance = 1e-9
    for bit_mask in range(1 << feature_count):
        active_indices = np.asarray(
            [
                index
                for index in range(feature_count)
                if bit_mask & (1 << index)
            ],
            dtype=int,
        )
        candidate = np.zeros(feature_count, dtype=float)
        if len(active_indices):
            active_gram = gram[np.ix_(active_indices, active_indices)]
            active_cross = cross[active_indices]
            active_coefficient = np.linalg.solve(
                active_gram, active_cross
            )
            if float(active_coefficient.min()) < -kkt_tolerance:
                continue
            candidate[active_indices] = np.maximum(
                active_coefficient, 0.0
            )
        gradient = gram @ candidate - cross
        active = candidate > 1e-12
        if active.any() and float(
            np.max(np.abs(gradient[active]))
        ) > kkt_tolerance:
            continue
        if (~active).any() and float(
            np.min(gradient[~active])
        ) < -kkt_tolerance:
            continue
        objective = float(
            0.5 * candidate @ gram @ candidate
            - cross @ candidate
        )
        feasible.append((objective, candidate))
    if not feasible:
        raise RuntimeError(
            f"no KKT-feasible ridge active set for {source_season} "
            f"{feature_names}"
        )
    objective, coefficient = min(feasible, key=lambda item: item[0])
    gradient = gram @ coefficient - cross
    active = coefficient > 1e-12
    kkt_active = (
        float(np.max(np.abs(gradient[active]))) if active.any() else 0.0
    )
    kkt_inactive_violation = (
        float(max(0.0, -np.min(gradient[~active])))
        if (~active).any()
        else 0.0
    )
    return {
        "source_season": source_season,
        "feature_names": feature_names,
        "means": means,
        "scales": safe_scales,
        "coefficients": coefficient,
        "diagnostics": {
            "source_R_rows": int(len(centered_target)),
            "raw_residual_mean_R": float(target_residual.mean()),
            "centered_residual_mean_R": float(centered_target.mean()),
            "feature_raw_scales": {
                name: float(value)
                for name, value in zip(feature_names, scales, strict=True)
            },
            "coefficients_per_source_sd": {
                name: float(value)
                for name, value in zip(feature_names, coefficient, strict=True)
            },
            "optimizer_success": True,
            "optimizer_method": "complete active-set enumeration",
            "optimizer_message": "KKT-feasible global optimum",
            "optimizer_active_sets_evaluated": int(1 << feature_count),
            "optimizer_feasible_active_sets": int(len(feasible)),
            "optimizer_objective": objective,
            "kkt_active_gradient_max_abs": kkt_active,
            "kkt_inactive_negative_gradient_violation": kkt_inactive_violation,
        },
    }


def map_source_model(
    model: dict[str, Any],
    validation_values: np.ndarray,
) -> np.ndarray:
    standardized = (
        validation_values - model["means"]
    ) / model["scales"]
    return standardized @ model["coefficients"]


def regime_metrics(
    game_types: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    return {
        regime: calculate_metrics(
            target[game_types == regime],
            prediction[game_types == regime],
        )
        for regime in ("R", "F")
    }


def aggregate_metrics(
    folds: dict[str, Any],
) -> dict[str, Any]:
    candidate_names = ["base", *(candidate.name for candidate in CANDIDATES)]
    output: dict[str, Any] = {}
    for name in candidate_names:
        metrics = {
            season: folds[str(season)]["candidates"][name]["metrics"]
            for season in REPORT_SEASONS
        }
        skills = {
            season: float(value["skill_score_unclipped"])
            for season, value in metrics.items()
        }
        briers = {
            season: float(value["brier_score"])
            for season, value in metrics.items()
        }
        output[name] = {
            "season_briers": {str(key): value for key, value in briers.items()},
            "season_skills": {str(key): value for key, value in skills.items()},
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
    base = output["base"]
    for candidate in CANDIDATES:
        current = output[candidate.name]
        current["season_skill_change_vs_base"] = {
            str(season): float(
                current["season_skills"][str(season)]
                - base["season_skills"][str(season)]
            )
            for season in REPORT_SEASONS
        }
        current["mean_skill_change_vs_base"] = float(
            current["mean_skill"] - base["mean_skill"]
        )
        current["min_skill_change_vs_base"] = float(
            current["min_skill"] - base["min_skill"]
        )
        current["improved_every_reported_season"] = bool(
            all(
                value > 0.0
                for value in current["season_skill_change_vs_base"].values()
            )
        )
    return output


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_rows()
    targets, base, team = load_oof(frame)
    seasons = frame["season"].to_numpy(dtype=np.int16)
    game_types = frame["game_type"].astype(str).to_numpy()
    base_all = np.concatenate([base[season] for season in EVALUATED_SEASONS])
    team_all = np.concatenate([team[season] for season in EVALUATED_SEASONS])
    features = build_features(frame, base_all, team_all)
    signal_audit = samefold_signal_audit(
        frame, features, targets, base
    )

    source_models: dict[str, dict[int, dict[str, Any]]] = {
        name: {} for name in FEATURE_SETS
    }
    source_model_diagnostics: dict[str, Any] = {
        name: {} for name in FEATURE_SETS
    }
    for set_name, feature_names in FEATURE_SETS.items():
        for source_season in EVALUATED_SEASONS:
            source_mask = (
                (seasons == source_season) & (game_types == "R")
            )
            source_target = targets[source_season]
            source_base = base[source_season]
            local_r = game_types[seasons == source_season] == "R"
            residual = source_target[local_r] - source_base[local_r]
            model = fit_source_model(
                features.loc[
                    source_mask, list(feature_names)
                ].to_numpy(dtype=float),
                residual,
                feature_names,
                source_season,
            )
            source_models[set_name][source_season] = model
            source_model_diagnostics[set_name][str(source_season)] = model[
                "diagnostics"
            ]

    folds: dict[str, Any] = {}
    qa = {
        "candidate_count": len(CANDIDATES),
        "candidate_count_at_most_4": len(CANDIDATES) <= 4,
        "all_current_fold_labels_unused_for_fit": True,
        "all_source_seasons_strictly_earlier": True,
        "all_probabilities_finite_and_in_range": True,
        "all_F_predictions_exactly_base": True,
        "all_target_order_matches": True,
        "all_coefficients_obey_nonnegative_direction_constraints": True,
    }
    for validation_season in EVALUATED_SEASONS:
        validation_mask = seasons == validation_season
        validation_types = game_types[validation_mask]
        validation_r = validation_types == "R"
        source_seasons = [
            season
            for season in EVALUATED_SEASONS
            if season < validation_season
        ]
        qa["all_source_seasons_strictly_earlier"] &= all(
            season < validation_season for season in source_seasons
        )
        corrections: dict[str, np.ndarray] = {}
        correction_diagnostics: dict[str, Any] = {}
        for set_name, feature_names in FEATURE_SETS.items():
            correction = np.zeros(int(validation_mask.sum()), dtype=float)
            if source_seasons and validation_r.any():
                validation_values = features.loc[
                    validation_mask, list(feature_names)
                ].to_numpy(dtype=float)
                mapped = np.vstack(
                    [
                        map_source_model(
                            source_models[set_name][source_season],
                            validation_values,
                        )
                        for source_season in source_seasons
                    ]
                )
                correction[validation_r] = mapped[:, validation_r].mean(axis=0)
            corrections[set_name] = correction
            correction_diagnostics[set_name] = {
                "source_seasons": source_seasons,
                "source_count": len(source_seasons),
                "R_mean": float(correction[validation_r].mean())
                if validation_r.any()
                else 0.0,
                "R_std": float(correction[validation_r].std())
                if validation_r.any()
                else 0.0,
                "R_mean_absolute": float(
                    np.abs(correction[validation_r]).mean()
                )
                if validation_r.any()
                else 0.0,
                "R_min": float(correction[validation_r].min())
                if validation_r.any()
                else 0.0,
                "R_max": float(correction[validation_r].max())
                if validation_r.any()
                else 0.0,
                "F_exactly_zero": bool(
                    np.array_equal(
                        correction[~validation_r],
                        np.zeros((~validation_r).sum(), dtype=float),
                    )
                ),
            }
            np.save(
                ARTIFACT_DIR
                / f"correction_{set_name}_{validation_season}.npy",
                correction,
            )

        target = targets[validation_season]
        base_prediction = base[validation_season]
        candidate_results: dict[str, Any] = {
            "base": {
                "metrics": calculate_metrics(target, base_prediction),
                "regimes": regime_metrics(
                    validation_types, target, base_prediction
                ),
            }
        }
        np.save(
            ARTIFACT_DIR / f"predictions_base_{validation_season}.npy",
            base_prediction,
        )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            target,
        )
        for candidate in CANDIDATES:
            set_name = next(
                name
                for name, values in FEATURE_SETS.items()
                if values == candidate.features
            )
            prediction = np.clip(
                base_prediction
                + candidate.correction_weight * corrections[set_name],
                0.0,
                1.0,
            )
            finite_range = bool(
                np.isfinite(prediction).all()
                and (prediction >= 0.0).all()
                and (prediction <= 1.0).all()
            )
            f_exact = bool(
                np.array_equal(
                    prediction[~validation_r],
                    base_prediction[~validation_r],
                )
            )
            qa["all_probabilities_finite_and_in_range"] &= finite_range
            qa["all_F_predictions_exactly_base"] &= f_exact
            candidate_results[candidate.name] = {
                "configuration": asdict(candidate),
                "feature_set": set_name,
                "source_seasons": source_seasons,
                "current_fold_labels_used_for_fit": False,
                "metrics": calculate_metrics(target, prediction),
                "regimes": regime_metrics(
                    validation_types, target, prediction
                ),
                "F_prediction_exactly_base": f_exact,
            }
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate.name}_{validation_season}.npy",
                prediction,
            )
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "source_seasons": source_seasons,
            "corrections": correction_diagnostics,
            "candidates": candidate_results,
        }

    for set_models in source_models.values():
        for model in set_models.values():
            qa["all_coefficients_obey_nonnegative_direction_constraints"] &= bool(
                (model["coefficients"] >= -1e-15).all()
            )

    aggregate = aggregate_metrics(folds)
    ranked = sorted(
        CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate.name]["min_skill"],
            aggregate[candidate.name]["mean_skill"],
        ),
        reverse=True,
    )
    best = ranked[0]
    result = {
        "experiment": "EXP-020",
        "candidate_family": "R_monotone_row_residual_atop_strict_lowrank",
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": (
                "low_rank_pitcher_context_eb strict_rank_s300 OOF"
            ),
            "outer_fit": (
                "one constrained model per earlier source OOF season; "
                "equal source-season correction average"
            ),
            "source_target": (
                "source R target-minus-base residual centered inside source R"
            ),
            "application": "R only; F bitwise immutable base",
            "current_fold_labels_used_for_fit_or_selection": False,
            "validation_or_test_row_aggregation": False,
            "raw_player_or_team_ids": False,
            "calibration_offset": False,
            "trackman": False,
            "candidate_design_non_nested": True,
        },
        "predeclared_configuration": {
            "ridge_alpha": RIDGE_ALPHA,
            "direction_coding": {
                "outs_positive": "positive coefficient means residual rises with outs",
                "earlier_inning": "-clip(inning,1,12)/4; positive coefficient means residual falls with inning",
                "risp": "positive coefficient",
                "log_li": "positive coefficient",
                "latent_late": "(strict lowrank - team base) * I(inning>=7); positive coefficient",
            },
            "feature_sets": {
                name: list(values) for name, values in FEATURE_SETS.items()
            },
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
            "candidate_count": len(CANDIDATES),
        },
        "signal_audit": signal_audit,
        "signal_interpretation": {
            "retained": (
                "R outs(+), inning(-), RISP(+), log(LI)(+) were sign-stable; "
                "latent x late-inning was the only inspected latent interaction "
                "with the same positive sign in 2022-2024 R diagnostics."
            ),
            "excluded": (
                "Count progression/depth, score difference, base-state categories, "
                "F main effect, and other latent interactions changed sign or regime."
            ),
            "diagnostic_labels_used_by_outer_fit": False,
        },
        "source_model_diagnostics": source_model_diagnostics,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "post-hoc diagnostic ranking only; no same-fold deployment selection",
            "posthoc_best_min_candidate": best.name,
            "posthoc_best_min_skill": float(aggregate[best.name]["min_skill"]),
            "posthoc_best_mean_skill": float(aggregate[best.name]["mean_skill"]),
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
                "base": aggregate["base"],
                "candidates": {
                    candidate.name: aggregate[candidate.name]
                    for candidate in CANDIDATES
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
