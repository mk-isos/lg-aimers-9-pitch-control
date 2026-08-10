"""Final read-only EXP-020 convex-hull and strict temporal ensemble audit.

No model is trained and no prediction is tuned here.  The script reads a
frozen, explicitly non-nested pool of saved rolling OOF vectors, verifies
target/order/range parity, and computes:

* individual candidate metrics;
* a same-fold nonnegative simplex oracle (diagnostic/non-deployable), with a
  Frank-Wolfe lower-bound certificate around the SLSQP solution;
* a strict prior-fold minimax path whose primary objective is worst historical
  normalized Brier and whose secondary objective is historical mean Brier.

The candidate pool itself was assembled after prior OOF experiments and is
therefore not a nested confirmation even when weight fitting is temporally
strict.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import audit_exp020_ensemble_ceiling as audit_core


ROOT = Path(".")
ARTIFACT_DIR = ROOT / "artifacts/EXP-020/final_ensemble_ceiling_audit"
TARGET_TEMPLATE = (
    "artifacts/EXP-020/low_rank_pitcher_context_eb/targets_{season}.npy"
)
SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
TARGET_SKILL = 1100.0

# Frozen before optimization in this audit.  Every entry is an already saved
# prediction vector; no generating model is run by this script.
CANDIDATE_PATHS = {
    "team_allprior_base": (
        "artifacts/EXP-019/team_eb_ensemble/"
        "predictions_all_prior_s1000_{season}.npy"
    ),
    "r_gated_team_pc": (
        "artifacts/EXP-020/pitcher_count_eb_atop_team/"
        "predictions_r_gated_team_pc_all_{season}.npy"
    ),
    "f_transfer_w075": (
        "artifacts/EXP-020/r_to_f_transfer_residual/"
        "predictions_f_transfer_w075_{season}.npy"
    ),
    "lowrank_strict_s300": (
        "artifacts/EXP-020/low_rank_pitcher_context_eb/"
        "predictions_strict_rank_s300_{season}.npy"
    ),
    "lowrank_s300_r6": (
        "artifacts/EXP-020/low_rank_pitcher_context_eb/"
        "predictions_lowrank_s300_r6_{season}.npy"
    ),
    "lowrank_s300_r4_Rspecific": (
        "artifacts/EXP-020/low_rank_pitcher_context_eb/"
        "predictions_lowrank_s300_r4_Rspecific_{season}.npy"
    ),
    "binned_core_all_w025": (
        "artifacts/EXP-020/binned_gam_residual/"
        "predictions_core_all_w025_{season}.npy"
    ),
    "monotone_latent5_w100": (
        "artifacts/EXP-020/r_monotone_row_residual/"
        "predictions_latent5_w100_{season}.npy"
    ),
    "seasonbag_mean_w050": (
        "artifacts/EXP-020/season_bagged_residual/"
        "predictions_mean_w050_{season}.npy"
    ),
    "seasonbag_recency124_w050": (
        "artifacts/EXP-020/season_bagged_trend/"
        "predictions_recency124_w050_{season}.npy"
    ),
    "joint_lowrank_b600_w050": (
        "artifacts/EXP-020/low_rank_batter_context_eb/"
        "predictions_joint_p300r4_b600r4_w050_{season}.npy"
    ),
    "parametric_logistic_w050": (
        "artifacts/EXP-020/parametric_logit_extrapolation/"
        "predictions_logistic_w050_{season}.npy"
    ),
    "debiased_bias025_rank4": (
        "artifacts/EXP-020/debiased_two_way_svd/"
        "predictions_debiased_bias025_rank4_{season}.npy"
    ),
}

# Completed families excluded before optimization.  Metrics are read from the
# machine-generated source JSON rather than copied by hand.
EXCLUDED_COMPLETED = {
    "weighted_als_ridge60_rank4": {
        "metrics_path": (
            "artifacts/EXP-020/weighted_als_pitcher_context/"
            "validation_metrics.json"
        ),
        "aggregate_key": "weighted_als_ridge60_rank4",
        "reason": (
            "posthoc family best is below team base and basic SVD on every "
            "reported season; materially worse minimum Skill"
        ),
    },
    "weighted_als_ridge3000_rank4": {
        "metrics_path": (
            "artifacts/EXP-020/weighted_als_strong_ridge_sensitivity/"
            "validation_metrics.json"
        ),
        "aggregate_key": "weighted_als_ridge3000_rank4",
        "reason": (
            "strong-ridge sensitivity best remains below basic SVD r4 on "
            "every reported season and the family stop rule fired"
        ),
    },
    "plain_R_gated_team_EB": {
        "metrics_path": (
            "artifacts/EXP-020/regime_gated_team_eb/validation_metrics.json"
        ),
        "aggregate_key": "R_gated_team_eb",
        "reason": (
            "the included r_gated_team_pc variant has higher Skill on every "
            "reported season"
        ),
    },
}


def calculate_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    return audit_core.calculate_metrics(target, prediction)


def aggregate(folds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return audit_core.aggregate(folds)


def load_excluded_metrics() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, specification in EXCLUDED_COMPLETED.items():
        path = ROOT / str(specification["metrics_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        key = str(specification["aggregate_key"])
        aggregate_metrics = data["aggregate_2022_2024"][key]
        output[name] = {
            "metrics_path": str(path),
            "aggregate_key": key,
            "season_briers": aggregate_metrics.get("season_briers"),
            "season_skills": aggregate_metrics["season_skills"],
            "mean_skill": float(aggregate_metrics["mean_skill"]),
            "min_skill": float(aggregate_metrics["min_skill"]),
            "exclusion_reason": specification["reason"],
            "excluded_before_optimization": True,
        }
    return output


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
        "candidate_count": len(CANDIDATE_PATHS),
        "seasons": {},
    }
    for season in SEASONS:
        target_path = ROOT / TARGET_TEMPLATE.format(season=season)
        if not target_path.exists():
            qa["all_source_files_exist"] = False
            raise FileNotFoundError(target_path)
        target = np.load(target_path).astype(np.int8, copy=False)
        if not np.isin(target, (0, 1)).all():
            raise ValueError(f"non-binary target: {target_path}")
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
            in_range = bool(
                (prediction >= 0.0).all()
                and (prediction <= 1.0).all()
            )
            qa["all_shapes_match"] &= shape_ok
            qa["all_values_finite"] &= finite
            qa["all_probabilities_in_0_1"] &= in_range
            if not (shape_ok and finite and in_range):
                raise ValueError(f"invalid prediction: {name} {season}")
            predictions[name][season] = prediction
            source_target_path = path.parent / f"targets_{season}.npy"
            parity: bool | None = None
            if source_target_path.exists():
                parity = bool(
                    np.array_equal(np.load(source_target_path), target)
                )
                qa["all_available_source_targets_equal_canonical"] &= parity
                if not parity:
                    raise ValueError(
                        f"target/order mismatch: {source_target_path}"
                    )
            season_qa["candidate_paths"][name] = str(path)
            season_qa["source_target_parity"][name] = parity
        qa["seasons"][str(season)] = season_qa
    return targets, predictions, qa


def weight_map(names: list[str], weights: np.ndarray) -> dict[str, float]:
    return {
        name: float(weight)
        for name, weight in zip(names, weights, strict=True)
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    targets, predictions, qa = load_oof()
    excluded = load_excluded_metrics()
    candidate_names = list(CANDIDATE_PATHS)
    matrices = {
        season: np.column_stack(
            [predictions[name][season] for name in candidate_names]
        )
        for season in SEASONS
    }
    grams = {
        season: audit_core.gram_matrix(targets[season], matrices[season])
        for season in SEASONS
    }
    base_index = candidate_names.index("team_allprior_base")

    individual: dict[str, Any] = {}
    gram_diagonal_max_difference = 0.0
    for candidate_index, name in enumerate(candidate_names):
        folds = {
            season: calculate_metrics(
                targets[season], predictions[name][season]
            )
            for season in SEASONS
        }
        for season in SEASONS:
            gram_diagonal_max_difference = max(
                gram_diagonal_max_difference,
                abs(
                    float(grams[season][candidate_index, candidate_index])
                    - float(folds[season]["brier_score"])
                ),
            )
        individual[name] = {
            "folds": {str(season): value for season, value in folds.items()},
            "aggregate_2022_2024": aggregate(folds),
        }
    if gram_diagonal_max_difference > 1e-12:
        raise AssertionError("Gram diagonal Brier parity failed")

    samefold_oracle: dict[str, Any] = {}
    oracle_folds: dict[int, dict[str, Any]] = {}
    conclusive_2023_2024 = True
    max_frank_wolfe_gap = 0.0
    for season in REPORT_SEASONS:
        weights, optimizer = audit_core.solve_simplex_qp(grams[season])
        prediction = matrices[season] @ weights
        metrics = calculate_metrics(targets[season], prediction)
        objective_difference = abs(
            float(optimizer["objective_brier"])
            - float(metrics["brier_score"])
        )
        if objective_difference > 1e-11:
            raise AssertionError("oracle objective/prediction mismatch")
        threshold = float(metrics["brier_threshold_for_skill_1100"])
        lower_bound = float(optimizer["certified_brier_lower_bound"])
        certified_upper_skill = 100000.0 * (
            1.0 - lower_bound / float(metrics["baseline_brier"])
        )
        feasible_reaches = bool(metrics["brier_score"] <= threshold)
        certified_no_reach = bool(lower_bound > threshold)
        conclusive = feasible_reaches or certified_no_reach
        if season in (2023, 2024):
            conclusive_2023_2024 &= conclusive
        max_frank_wolfe_gap = max(
            max_frank_wolfe_gap,
            float(optimizer["frank_wolfe_gap"]),
        )
        certificate_status = (
            "feasible_convex_blend_reaches_1100"
            if feasible_reaches
            else (
                "certified_no_convex_blend_reaches_1100"
                if certified_no_reach
                else "inconclusive_optimizer_gap"
            )
        )
        np.save(
            ARTIFACT_DIR / f"predictions_samefold_oracle_{season}.npy",
            prediction,
        )
        oracle_folds[season] = metrics
        samefold_oracle[str(season)] = {
            "deployable": False,
            "current_fold_labels_used_for_weights": True,
            "weights": weight_map(candidate_names, weights),
            "metrics": metrics,
            "optimization_certificate": {
                **optimizer,
                "objective_prediction_brier_difference": (
                    objective_difference
                ),
                "skill_1100_brier_threshold": threshold,
                "objective_brier_margin_vs_1100": float(
                    metrics["brier_score"] - threshold
                ),
                "certified_lower_brier_margin_vs_1100": float(
                    lower_bound - threshold
                ),
                "certified_skill_upper_bound": float(
                    certified_upper_skill
                ),
                "certificate_conclusive": conclusive,
                "certificate_status": certificate_status,
            },
        }
    if not conclusive_2023_2024:
        raise AssertionError("2023/2024 oracle certificate is inconclusive")

    strict_path: dict[str, Any] = {}
    strict_folds: dict[int, dict[str, Any]] = {}
    for season in REPORT_SEASONS:
        history = [value for value in SEASONS if value < season]
        baselines = [
            float(targets[value].mean() * (1.0 - targets[value].mean()))
            for value in history
        ]
        weights, optimizer = audit_core.solve_prior_minimax(
            [grams[value] for value in history], baselines, base_index
        )
        prediction = matrices[season] @ weights
        metrics = calculate_metrics(targets[season], prediction)
        strict_folds[season] = metrics
        np.save(
            ARTIFACT_DIR
            / f"predictions_strict_prior_minimax_{season}.npy",
            prediction,
        )
        strict_path[str(season)] = {
            "weight_fit_seasons": history,
            "current_fold_labels_used_for_weights": False,
            "weights": weight_map(candidate_names, weights),
            "historical_optimization": optimizer,
            "metrics": metrics,
        }

    all_baselines = [
        float(targets[season].mean() * (1.0 - targets[season].mean()))
        for season in SEASONS
    ]
    prospective_weights, prospective_optimizer = (
        audit_core.solve_prior_minimax(
            [grams[season] for season in SEASONS],
            all_baselines,
            base_index,
        )
    )

    for season in SEASONS:
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", targets[season])

    individual_ranking = sorted(
        (
            {
                "candidate": name,
                "min_skill": float(
                    value["aggregate_2022_2024"]["min_skill"]
                ),
                "mean_skill": float(
                    value["aggregate_2022_2024"]["mean_skill"]
                ),
            }
            for name, value in individual.items()
        ),
        key=lambda value: (value["min_skill"], value["mean_skill"]),
        reverse=True,
    )
    threshold_certificates = {
        season: samefold_oracle[str(season)]["optimization_certificate"]
        for season in (2023, 2024)
    }
    result: dict[str, Any] = {
        "experiment": "EXP-020",
        "audit": "final_strong_diverse_convex_ceiling",
        "protocol": {
            "candidate_names": candidate_names,
            "candidate_count": len(candidate_names),
            "reported_seasons": list(REPORT_SEASONS),
            "models_trained_or_tuned_by_this_audit": False,
            "samefold_oracle": (
                "nonnegative simplex; current-fold labels; diagnostic only"
            ),
            "strict_prior_path": (
                "weights fitted only on earlier OOF folds; lexicographic "
                "worst normalized Brier then mean"
            ),
            "candidate_pool_is_nested": False,
            "candidate_pool_warning": (
                "The pool contains post-hoc strongest configurations from "
                "multiple completed OOF experiments. Strict weights are "
                "temporal only conditional on this non-nested frozen pool."
            ),
            "current_fold_or_test_row_aggregation": False,
        },
        "pool_rationale": {
            "required_families_included": [
                "team base",
                "stronger r-gated team+pitcher-context",
                "F-transfer",
                "lowrank strict/r6/R-specific",
                "binned-GAM best",
                "monotone-row best",
                "seasonbag mean/trend-family best",
            ],
            "additional_diverse_candidates": [
                "joint_lowrank_b600_w050",
                "debiased_bias025_rank4",
            ],
            "parametric_best_included": "parametric_logistic_w050",
            "completed_candidates_excluded_before_optimization": excluded,
        },
        "skill_1100_definition": {
            "target_skill": TARGET_SKILL,
            "brier_threshold_formula": (
                "season baseline_brier * (1 - 1100 / 100000)"
            ),
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
        "diagnostic_samefold_convex_oracle": {
            "folds": samefold_oracle,
            "aggregate_2022_2024": aggregate(oracle_folds),
            "2023_2024_FW_SLSQP_certificates": threshold_certificates,
            "2023_2024_certificates_conclusive": conclusive_2023_2024,
            "interpretation": (
                "A certified lower Brier above the 1100 threshold proves "
                "that no nonnegative convex combination of this pool reaches "
                "1100 on that fold."
            ),
        },
        "strict_prior_fold_minimax_path": {
            "folds": strict_path,
            "aggregate_2022_2024": aggregate(strict_folds),
            "prospective_2025_weights": weight_map(
                candidate_names, prospective_weights
            ),
            "prospective_2025_historical_optimization": (
                prospective_optimizer
            ),
            "uses_2025_labels": False,
            "qualification": (
                "temporal weight fitting conditional on a non-nested pool"
            ),
        },
        "ranking": {
            "individual_by_min_skill": individual_ranking,
            "best_individual": individual_ranking[0],
            "samefold_oracle_min_skill": float(
                aggregate(oracle_folds)["min_skill"]
            ),
            "strict_path_min_skill": float(
                aggregate(strict_folds)["min_skill"]
            ),
        },
        "qa": {
            **qa,
            "gram_diagonal_brier_max_abs_difference": (
                gram_diagonal_max_difference
            ),
            "oracle_objective_prediction_parity_checked": True,
            "maximum_samefold_frank_wolfe_gap": max_frank_wolfe_gap,
            "2023_2024_certificate_conclusive": conclusive_2023_2024,
            "strict_current_fold_labels_unused": True,
            "prospective_2025_labels_unused": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    output_path = ARTIFACT_DIR / "validation_metrics.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "candidate_count": len(candidate_names),
                "best_individual": individual_ranking[0],
                "samefold_oracle": {
                    season: {
                        "skill": samefold_oracle[str(season)]["metrics"][
                            "skill_score_unclipped"
                        ],
                        "brier": samefold_oracle[str(season)]["metrics"][
                            "brier_score"
                        ],
                        "FW_gap": samefold_oracle[str(season)][
                            "optimization_certificate"
                        ]["frank_wolfe_gap"],
                        "certificate": samefold_oracle[str(season)][
                            "optimization_certificate"
                        ]["certificate_status"],
                        "weights": samefold_oracle[str(season)]["weights"],
                    }
                    for season in REPORT_SEASONS
                },
                "strict": {
                    "aggregate": aggregate(strict_folds),
                    "prospective_2025_weights": weight_map(
                        candidate_names, prospective_weights
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
