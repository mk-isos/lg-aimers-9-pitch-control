"""EXP-020 OOF ensemble audit with strict temporal and oracle paths.

This script does not train a model.  It only reads already generated rolling
OOF predictions, verifies target/order parity, and compares three families:

1. fixed equal-weight additive-correction composites;
2. a non-negative simplex oracle fit on the *same* fold (diagnostic ceiling);
3. a simplex path whose weights use only earlier OOF folds.  The primary
   historical objective is worst-season normalized Brier, with mean normalized
   Brier as a deterministic lexicographic tie-break.

The same-fold oracle is explicitly non-deployable.  The prior-fold weight path
is temporally strict only conditional on the frozen candidate pool: several
candidate configurations were themselves compared post hoc on 2022--2024, so
this audit is not a nested confirmation of the whole selection procedure.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize


ROOT = Path(".")
ARTIFACT_DIR = ROOT / "artifacts/EXP-020/ensemble_ceiling_audit"
TARGET_TEMPLATE = (
    "artifacts/EXP-018/constrained_multiscale/targets_{season}.npy"
)
SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
TARGET_SKILL = 1100.0

# Frozen before the audit optimization.  ``f_transfer_w075`` and
# ``lowrank_s300_r4`` are included because their generating experiments had
# completed when this audit began, but their configuration selection is
# post-hoc/non-nested and is recorded as such in the output.
CANDIDATE_PATHS = {
    "base_team_all_prior": (
        "artifacts/EXP-019/team_eb_ensemble/"
        "predictions_all_prior_s1000_{season}.npy"
    ),
    "team_pc_all": (
        "artifacts/EXP-020/pitcher_count_eb_atop_team/"
        "predictions_team_pc_all_{season}.npy"
    ),
    "r_gated_team_pc_all": (
        "artifacts/EXP-020/pitcher_count_eb_atop_team/"
        "predictions_r_gated_team_pc_all_{season}.npy"
    ),
    "seasonbag_mean_w050": (
        "artifacts/EXP-020/season_bagged_residual/"
        "predictions_mean_w050_{season}.npy"
    ),
    "oof_stack_w025": (
        "artifacts/EXP-020/oof_residual_stack/"
        "predictions_stack_w025_{season}.npy"
    ),
    "pair_all_prior_s2000": (
        "artifacts/EXP-020/player_eb_atop_team/"
        "predictions_all_prior_pair_s2000_{season}.npy"
    ),
    "f_transfer_w075": (
        "artifacts/EXP-020/r_to_f_transfer_residual/"
        "predictions_f_transfer_w075_{season}.npy"
    ),
    "lowrank_s300_r4": (
        "artifacts/EXP-020/low_rank_pitcher_context_eb/"
        "predictions_lowrank_s300_r4_{season}.npy"
    ),
    "joint_lowrank_b300_w025": (
        "artifacts/EXP-020/low_rank_batter_context_eb/"
        "predictions_joint_p300r4_b300r4_w025_{season}.npy"
    ),
    "joint_lowrank_b600_w050": (
        "artifacts/EXP-020/low_rank_batter_context_eb/"
        "predictions_joint_p300r4_b600r4_w050_{season}.npy"
    ),
}

PROMPT_CORE = (
    "team_pc_all",
    "r_gated_team_pc_all",
    "seasonbag_mean_w050",
    "oof_stack_w025",
    "pair_all_prior_s2000",
)
ALL_CORRECTIONS = tuple(
    name for name in CANDIDATE_PATHS if name != "base_team_all_prior"
)


def calculate_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Match the repository's Brier/Skill and calibration diagnostics."""
    target = target.astype(float, copy=False)
    prediction = prediction.astype(float, copy=False)
    actual_rate = float(target.mean())
    brier = float(np.mean((prediction - target) ** 2))
    baseline = actual_rate * (1.0 - actual_rate)
    skill_unclipped = 100000.0 * (1.0 - brier / baseline)
    design = np.column_stack([prediction, np.ones_like(prediction)])
    slope, intercept = np.linalg.lstsq(design, target, rcond=None)[0]
    threshold = baseline * (1.0 - TARGET_SKILL / 100000.0)
    return {
        "rows": int(len(target)),
        "actual_rate": actual_rate,
        "prediction_mean": float(prediction.mean()),
        "mean_gap": float(prediction.mean() - actual_rate),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "brier_score": brier,
        "baseline_brier": baseline,
        "skill_score": float(max(0.0, skill_unclipped)),
        "skill_score_unclipped": float(skill_unclipped),
        "diagnostic_calibration_slope": float(slope),
        "diagnostic_calibration_intercept": float(intercept),
        "brier_threshold_for_skill_1100": float(threshold),
        "brier_margin_vs_skill_1100": float(brier - threshold),
        "reaches_skill_1100": bool(brier <= threshold),
    }


def aggregate(folds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    skills = [float(folds[season]["skill_score_unclipped"]) for season in REPORT_SEASONS]
    briers = [float(folds[season]["brier_score"]) for season in REPORT_SEASONS]
    return {
        "season_skills": {str(s): float(folds[s]["skill_score_unclipped"]) for s in REPORT_SEASONS},
        "season_briers": {str(s): float(folds[s]["brier_score"]) for s in REPORT_SEASONS},
        "mean_skill": float(np.mean(skills)),
        "min_skill": float(np.min(skills)),
        "latest_2024_skill": float(folds[2024]["skill_score_unclipped"]),
        "mean_brier": float(np.mean(briers)),
        "uniform_1100_passed": bool(all(folds[s]["reaches_skill_1100"] for s in REPORT_SEASONS)),
    }


def load_oof() -> tuple[
    dict[int, np.ndarray],
    dict[str, dict[int, np.ndarray]],
    dict[str, Any],
]:
    targets: dict[int, np.ndarray] = {}
    predictions = {name: {} for name in CANDIDATE_PATHS}
    qa: dict[str, Any] = {
        "all_source_files_exist": True,
        "all_shapes_match": True,
        "all_values_finite": True,
        "all_probabilities_in_0_1": True,
        "all_available_source_targets_equal_canonical": True,
        "seasons": {},
    }
    for season in SEASONS:
        target_path = ROOT / TARGET_TEMPLATE.format(season=season)
        if not target_path.exists():
            raise FileNotFoundError(target_path)
        target = np.load(target_path).astype(np.int8, copy=False)
        if not np.isin(target, (0, 1)).all():
            raise ValueError(f"Non-binary target in {target_path}")
        targets[season] = target
        season_qa: dict[str, Any] = {
            "rows": int(len(target)),
            "canonical_target_path": str(target_path),
            "candidate_paths": {},
            "source_target_parity": {},
        }
        for name, template in CANDIDATE_PATHS.items():
            path = ROOT / template.format(season=season)
            if not path.exists():
                qa["all_source_files_exist"] = False
                raise FileNotFoundError(path)
            prediction = np.load(path).astype(np.float64, copy=False)
            shape_ok = prediction.shape == target.shape
            finite = bool(np.isfinite(prediction).all())
            in_range = bool((prediction >= 0.0).all() and (prediction <= 1.0).all())
            qa["all_shapes_match"] &= shape_ok
            qa["all_values_finite"] &= finite
            qa["all_probabilities_in_0_1"] &= in_range
            if not (shape_ok and finite and in_range):
                raise ValueError(f"Invalid prediction array: {name} {season}")
            predictions[name][season] = prediction
            source_target = path.parent / f"targets_{season}.npy"
            parity: bool | None = None
            if source_target.exists():
                parity = bool(np.array_equal(np.load(source_target), target))
                qa["all_available_source_targets_equal_canonical"] &= parity
                if not parity:
                    raise ValueError(f"Target/order mismatch: {source_target}")
            season_qa["candidate_paths"][name] = str(path)
            season_qa["source_target_parity"][name] = parity
        qa["seasons"][str(season)] = season_qa
    return targets, predictions, qa


def gram_matrix(
    target: np.ndarray,
    candidate_matrix: np.ndarray,
) -> np.ndarray:
    error = candidate_matrix - target[:, None]
    return (error.T @ error) / float(len(target))


def solve_simplex_qp(gram: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve min w'Gw on the probability simplex and certify with FW gap."""
    count = gram.shape[0]

    def objective(weight: np.ndarray) -> float:
        return float(weight @ gram @ weight)

    def gradient(weight: np.ndarray) -> np.ndarray:
        return 2.0 * gram @ weight

    constraints = ({
        "type": "eq",
        "fun": lambda weight: float(weight.sum() - 1.0),
        "jac": lambda weight: np.ones(count, dtype=float),
    },)
    starts = [np.full(count, 1.0 / count)]
    starts.extend(np.eye(count, dtype=float))
    results = []
    for start in starts:
        result = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * count,
            constraints=constraints,
            options={"ftol": 1e-15, "maxiter": 3000, "disp": False},
        )
        weight = np.clip(result.x, 0.0, 1.0)
        weight /= weight.sum()
        results.append((objective(weight), weight, result))
    value, weight, result = min(results, key=lambda item: item[0])
    grad = gradient(weight)
    frank_wolfe_gap = max(0.0, float(weight @ grad - grad.min()))
    return weight, {
        "objective_brier": float(value),
        "frank_wolfe_gap": frank_wolfe_gap,
        "certified_brier_lower_bound": float(max(0.0, value - frank_wolfe_gap)),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "simplex_sum_error": float(abs(weight.sum() - 1.0)),
        "minimum_weight": float(weight.min()),
    }


def solve_prior_minimax(
    grams: list[np.ndarray],
    baselines: list[float],
    base_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Lexicographic earlier-fold fit: worst normalized Brier, then mean."""
    normalized = [gram / baseline for gram, baseline in zip(grams, baselines)]
    count = normalized[0].shape[0]

    def losses(weight: np.ndarray) -> np.ndarray:
        return np.asarray([weight @ gram @ weight for gram in normalized])

    constraints = [{
        "type": "eq",
        "fun": lambda value: float(value[:-1].sum() - 1.0),
        "jac": lambda value: np.r_[np.ones(count), 0.0],
    }]
    for gram in normalized:
        constraints.append({
            "type": "ineq",
            "fun": lambda value, current=gram: float(
                value[-1] - value[:-1] @ current @ value[:-1]
            ),
            "jac": lambda value, current=gram: np.r_[
                -2.0 * current @ value[:-1], 1.0
            ],
        })

    starts = [np.full(count, 1.0 / count)]
    starts.extend(np.eye(count, dtype=float))
    primary_results = []
    for weight_start in starts:
        start_losses = losses(weight_start)
        start = np.r_[weight_start, float(start_losses.max() + 1e-8)]
        result = minimize(
            lambda value: float(value[-1]),
            start,
            jac=lambda value: np.r_[np.zeros(count), 1.0],
            method="SLSQP",
            bounds=[(0.0, 1.0)] * count + [(0.0, 5.0)],
            constraints=constraints,
            options={"ftol": 1e-14, "maxiter": 5000, "disp": False},
        )
        weight = np.clip(result.x[:-1], 0.0, 1.0)
        weight /= weight.sum()
        primary_results.append((float(losses(weight).max()), weight, result))
    primary_value, primary_weight, primary_result = min(
        primary_results, key=lambda item: item[0]
    )

    # On a worst-loss plateau (notably when many 2021 predictions are exactly
    # identical), use all earlier folds by minimizing their mean loss while
    # retaining the primary optimum to numerical tolerance.
    primary_tolerance = 2e-11
    secondary_constraints = [{
        "type": "eq",
        "fun": lambda weight: float(weight.sum() - 1.0),
        "jac": lambda weight: np.ones(count, dtype=float),
    }]
    for gram in normalized:
        secondary_constraints.append({
            "type": "ineq",
            "fun": lambda weight, current=gram: float(
                primary_value + primary_tolerance - weight @ current @ weight
            ),
            "jac": lambda weight, current=gram: -2.0 * current @ weight,
        })
    mean_gram = np.mean(normalized, axis=0)
    secondary = minimize(
        lambda weight: float(weight @ mean_gram @ weight),
        primary_weight,
        jac=lambda weight: 2.0 * mean_gram @ weight,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=secondary_constraints,
        options={"ftol": 1e-15, "maxiter": 5000, "disp": False},
    )
    secondary_weight = np.clip(secondary.x, 0.0, 1.0)
    secondary_weight /= secondary_weight.sum()
    secondary_losses = losses(secondary_weight)
    if (
        secondary.success
        and float(secondary_losses.max())
        <= primary_value + primary_tolerance + 2e-10
    ):
        weight = secondary_weight
        used_secondary = True
    else:
        weight = primary_weight
        secondary_losses = losses(weight)
        used_secondary = False

    # A final deterministic tie-break prevents a future-meaningless choice
    # among candidates that had exactly the same prior OOF predictions.  It
    # prefers the base vertex while retaining both the worst-loss and mean-loss
    # optima.  This is especially important for the 2022 path, where all but
    # the F-transfer candidate are identical on 2021 OOF.
    secondary_value = float(weight @ mean_gram @ weight)
    secondary_tolerance = 2e-11
    tertiary_constraints = list(secondary_constraints)
    tertiary_constraints.append({
        "type": "ineq",
        "fun": lambda candidate: float(
            secondary_value
            + secondary_tolerance
            - candidate @ mean_gram @ candidate
        ),
        "jac": lambda candidate: -2.0 * mean_gram @ candidate,
    })
    base_vertex = np.zeros(count, dtype=float)
    base_vertex[base_index] = 1.0
    tertiary = minimize(
        lambda candidate: float(0.5 * np.sum((candidate - base_vertex) ** 2)),
        weight,
        jac=lambda candidate: candidate - base_vertex,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=tertiary_constraints,
        options={"ftol": 1e-15, "maxiter": 5000, "disp": False},
    )
    tertiary_weight = np.clip(tertiary.x, 0.0, 1.0)
    tertiary_weight /= tertiary_weight.sum()
    tertiary_losses = losses(tertiary_weight)
    tertiary_mean = float(tertiary_weight @ mean_gram @ tertiary_weight)
    if (
        tertiary.success
        and float(tertiary_losses.max())
        <= primary_value + primary_tolerance + 2e-10
        and tertiary_mean <= secondary_value + secondary_tolerance + 2e-10
    ):
        weight = tertiary_weight
        used_tertiary = True
    else:
        used_tertiary = False

    final_losses = losses(weight)
    return weight, {
        "primary_worst_normalized_brier": float(primary_value),
        "final_worst_normalized_brier": float(final_losses.max()),
        "final_mean_normalized_brier": float(final_losses.mean()),
        "historical_normalized_briers": [float(value) for value in final_losses],
        "historical_skills": [float((1.0 - value) * 100000.0) for value in final_losses],
        "primary_optimizer_success": bool(primary_result.success),
        "primary_optimizer_message": str(primary_result.message),
        "secondary_tiebreak_used": used_secondary,
        "secondary_optimizer_success": bool(secondary.success),
        "secondary_optimizer_message": str(secondary.message),
        "secondary_optimal_mean_normalized_brier": secondary_value,
        "tertiary_base_distance_tiebreak_used": used_tertiary,
        "tertiary_optimizer_success": bool(tertiary.success),
        "tertiary_optimizer_message": str(tertiary.message),
        "primary_tolerance": primary_tolerance,
        "secondary_tolerance": secondary_tolerance,
        "simplex_sum_error": float(abs(weight.sum() - 1.0)),
        "minimum_weight": float(weight.min()),
    }


def weight_map(names: list[str], weight: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, weight)}


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    targets, predictions, qa = load_oof()
    candidate_names = list(CANDIDATE_PATHS)
    matrices = {
        season: np.column_stack(
            [predictions[name][season] for name in candidate_names]
        )
        for season in SEASONS
    }
    grams = {
        season: gram_matrix(targets[season], matrices[season])
        for season in SEASONS
    }

    individual: dict[str, Any] = {}
    for name in candidate_names:
        folds = {
            season: calculate_metrics(targets[season], predictions[name][season])
            for season in SEASONS
        }
        individual[name] = {"folds": {str(k): v for k, v in folds.items()}, "aggregate_2022_2024": aggregate(folds)}

    composite_groups = {
        "fixed_equal_prompt5_corrections": PROMPT_CORE,
        "fixed_equal_all9_corrections": ALL_CORRECTIONS,
        "fixed_equal_r_gated_plus_lowrank": (
            "r_gated_team_pc_all",
            "lowrank_s300_r4",
        ),
    }
    composites: dict[str, Any] = {}
    for composite_name, members in composite_groups.items():
        folds: dict[int, dict[str, Any]] = {}
        for season in SEASONS:
            # Algebraically base + mean(candidate - base) == mean(candidate).
            prediction = np.mean(
                [predictions[name][season] for name in members], axis=0
            )
            prediction = np.clip(prediction, 0.0, 1.0)
            np.save(
                ARTIFACT_DIR / f"predictions_{composite_name}_{season}.npy",
                prediction,
            )
            folds[season] = calculate_metrics(targets[season], prediction)
        composites[composite_name] = {
            "definition": "base + equal mean(candidate - base); equivalent to equal mean of listed candidates",
            "members": list(members),
            "uses_fold_labels_for_weights": False,
            "candidate_pool_is_non_nested": True,
            "folds": {str(k): v for k, v in folds.items()},
            "aggregate_2022_2024": aggregate(folds),
        }

    samefold_oracle: dict[str, Any] = {}
    oracle_below_1100 = []
    base_index = candidate_names.index("base_team_all_prior")
    for season in REPORT_SEASONS:
        weight, optimizer = solve_simplex_qp(grams[season])
        prediction = matrices[season] @ weight
        metrics = calculate_metrics(targets[season], prediction)
        baseline = metrics["baseline_brier"]
        lower_brier = optimizer["certified_brier_lower_bound"]
        certified_skill_upper = 100000.0 * (1.0 - lower_brier / baseline)
        optimizer["certified_skill_upper_bound"] = float(certified_skill_upper)
        optimizer["certifies_no_convex_blend_reaches_1100"] = bool(
            certified_skill_upper < TARGET_SKILL
        )
        oracle_below_1100.append(certified_skill_upper < TARGET_SKILL)
        np.save(
            ARTIFACT_DIR / f"predictions_diagnostic_samefold_oracle_{season}.npy",
            prediction,
        )
        samefold_oracle[str(season)] = {
            "current_fold_labels_used_for_weights": True,
            "deployable": False,
            "weights": weight_map(candidate_names, weight),
            "metrics": metrics,
            "optimization_certificate": optimizer,
        }

    strict_path: dict[str, Any] = {}
    strict_folds: dict[int, dict[str, Any]] = {}
    for season in REPORT_SEASONS:
        history = [value for value in SEASONS if value < season]
        baselines = [
            float(targets[value].mean() * (1.0 - targets[value].mean()))
            for value in history
        ]
        weight, optimization = solve_prior_minimax(
            [grams[value] for value in history],
            baselines,
            base_index,
        )
        prediction = matrices[season] @ weight
        metrics = calculate_metrics(targets[season], prediction)
        strict_folds[season] = metrics
        np.save(
            ARTIFACT_DIR / f"predictions_strict_prior_minimax_{season}.npy",
            prediction,
        )
        strict_path[str(season)] = {
            "weight_fit_seasons": history,
            "current_fold_labels_used_for_weights": False,
            "weights": weight_map(candidate_names, weight),
            "historical_optimization": optimization,
            "metrics": metrics,
        }

    # Prospective weights use all available OOF folds but no unavailable 2025
    # labels.  They are informative only; this audit has no 2025 predictions.
    all_baselines = [
        float(targets[value].mean() * (1.0 - targets[value].mean()))
        for value in SEASONS
    ]
    prospective_weight, prospective_optimization = solve_prior_minimax(
        [grams[value] for value in SEASONS],
        all_baselines,
        base_index,
    )

    for season in SEASONS:
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", targets[season])

    oracle_metric_folds = {
        int(season): value["metrics"]
        for season, value in samefold_oracle.items()
    }
    certified_below_seasons = [
        int(season)
        for season, value in samefold_oracle.items()
        if value["optimization_certificate"][
            "certifies_no_convex_blend_reaches_1100"
        ]
    ]
    individual_min_ranking = sorted(
        (
            (
                name,
                float(value["aggregate_2022_2024"]["min_skill"]),
                float(value["aggregate_2022_2024"]["mean_skill"]),
            )
            for name, value in individual.items()
        ),
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )

    result: dict[str, Any] = {
        "experiment": "EXP-020",
        "audit": "ensemble_ceiling_and_strict_prior_composite",
        "protocol": {
            "candidate_names": candidate_names,
            "reported_seasons": list(REPORT_SEASONS),
            "simple_composite": "fixed equal additive corrections, no fold-label weight fitting",
            "samefold_oracle": "nonnegative simplex; current-fold labels; diagnostic non-deployable upper bound",
            "strict_weight_path": "only prior OOF folds; lexicographic worst-season then mean normalized Brier",
            "candidate_pool_warning": (
                "Weights are temporal conditional on a frozen pool, but the pool is non-nested: "
                "f_transfer_w075 and lowrank_s300_r4 and other named configurations were compared post hoc."
            ),
            "current_fold_or_test_aggregation": False,
            "models_trained_by_this_audit": False,
        },
        "skill_1100_definition": {
            "target_skill": TARGET_SKILL,
            "brier_threshold_formula": "season_baseline_brier * (1 - 1100 / 100000)",
            "season_thresholds": {
                str(season): float(
                    targets[season].mean()
                    * (1.0 - targets[season].mean())
                    * (1.0 - TARGET_SKILL / 100000.0)
                )
                for season in REPORT_SEASONS
            },
        },
        "individual_candidates": individual,
        "simple_additive_composites": composites,
        "diagnostic_samefold_convex_oracle": {
            "folds": samefold_oracle,
            "aggregate_2022_2024": aggregate(oracle_metric_folds),
            "seasons_certified_below_1100": certified_below_seasons,
            "every_reported_fold_certified_below_1100": bool(all(oracle_below_1100)),
            "uniform_1100_impossible_for_included_convex_hull": bool(
                any(oracle_below_1100)
            ),
            "interpretation": (
                "If any fold's certified oracle ceiling is below 1100, no nonnegative "
                "convex combination of this candidate set can exceed 1100 on every fold."
            ),
        },
        "strict_prior_fold_minimax_path": {
            "folds": strict_path,
            "aggregate_2022_2024": aggregate(strict_folds),
            "prospective_2025_weights": weight_map(candidate_names, prospective_weight),
            "prospective_2025_historical_optimization": prospective_optimization,
            "qualification": (
                "Weight fitting is temporally strict; whole candidate selection is not nested."
            ),
        },
        "audit_ranking": {
            "individual_by_min_skill": [
                {"candidate": name, "min_skill": minimum, "mean_skill": mean}
                for name, minimum, mean in individual_min_ranking
            ],
            "best_individual_min_skill_candidate": individual_min_ranking[0][0],
            "best_individual_min_skill": individual_min_ranking[0][1],
            "strict_path_min_skill": float(aggregate(strict_folds)["min_skill"]),
            "samefold_oracle_min_skill": float(
                aggregate(oracle_metric_folds)["min_skill"]
            ),
        },
        "qa": qa,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    output = ARTIFACT_DIR / "validation_metrics.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(output),
        "candidate_count": len(candidate_names),
        "composites": {
            name: value["aggregate_2022_2024"]
            for name, value in composites.items()
        },
        "samefold_oracle": {
            season: {
                "skill": value["metrics"]["skill_score_unclipped"],
                "certified_skill_upper": value["optimization_certificate"]["certified_skill_upper_bound"],
                "weights": value["weights"],
            }
            for season, value in samefold_oracle.items()
        },
        "strict": {
            "aggregate": aggregate(strict_folds),
            "folds": {
                season: {
                    "skill": value["metrics"]["skill_score_unclipped"],
                    "weights": value["weights"],
                }
                for season, value in strict_path.items()
            },
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
