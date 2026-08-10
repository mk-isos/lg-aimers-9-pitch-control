"""EXP-020: bounded season-by-phase GroupDRO R residual LightGBM.

The temporal hierarchical base, past-only count/hand/reverse group offset, and
non-ID R-full feature matrix are reused without modification.  For every outer
rolling fold, a single shallow LightGBM configuration is refit four times.
After each of the first three fits, training-only residual MSE is measured for
each ``season x phase`` group and exponentiated weights are shifted toward the
worst groups.  The total group-weight ratio is capped at four, so a small or
noisy historical group cannot dominate the fit.  The validation fold is never
used to update the model or the GroupDRO weights.

Only regular-season (R) rows receive the learned correction.  F rows retain
the temporally safe group-only prediction.
"""

from __future__ import annotations

import gc
import json
import math
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import train_exp019_multirate_residual as multirate
from train_exp017_rolling_residual import calculate_metrics
from train_exp019_r_phase_residual import (
    PHASES,
    build_feature_matrix,
    build_group_oof,
    phase_labels,
)


ARTIFACT_DIR = Path("./artifacts/EXP-020/groupdro_residual")
TEAM_EB_REFERENCE = Path(
    "./artifacts/EXP-019/team_eb_ensemble/validation_metrics.json"
)
TEAM_EB_VARIANT = "all_prior_s1000"
SEASON_BALANCED_REFERENCE = Path(
    "./artifacts/EXP-019/r_full_residual/"
    "rfull_l15_m3000_i200/validation_metrics.json"
)

VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
BLEND_WEIGHTS = (0.50, 0.75)

# One predeclared bounded GroupDRO configuration.
DRO_ROUNDS = 4
DRO_ETA = 0.50
DRO_MAX_GROUP_RATIO = 4.0
DRO_LOSS_SCALE_FLOOR = 1.0e-4
ITERATIONS_PER_ROUND = 200
LEARNING_RATE = 0.015
NUM_LEAVES = 15
MIN_CHILD_SAMPLES = 3000


def group_keys(seasons: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Return stable string labels for season-by-phase training groups."""
    return np.char.add(
        np.char.add(seasons.astype(str), "_"),
        phases.astype(str),
    )


def capped_group_update(
    current_weights: np.ndarray,
    losses: np.ndarray,
) -> np.ndarray:
    """Exponentiated GroupDRO update with a hard max/min ratio cap."""
    scale = max(float(np.std(losses)), DRO_LOSS_SCALE_FLOOR)
    standardized = (losses - float(np.mean(losses))) / scale
    log_weights = np.log(current_weights) + DRO_ETA * standardized
    log_weights -= float(np.max(log_weights))
    log_weights = np.maximum(log_weights, -math.log(DRO_MAX_GROUP_RATIO))
    updated = np.exp(log_weights)
    updated /= float(np.mean(updated))
    ratio = float(np.max(updated) / np.min(updated))
    if ratio > DRO_MAX_GROUP_RATIO * (1.0 + 1.0e-12):
        raise AssertionError(f"GroupDRO cap violated: ratio={ratio}")
    return updated


def row_weights_from_groups(
    local_group_codes: np.ndarray,
    group_weights: np.ndarray,
    group_counts: np.ndarray,
) -> np.ndarray:
    """Give each group total mass proportional to its current DRO weight."""
    row_weights = group_weights[local_group_codes] / group_counts[local_group_codes]
    row_weights *= len(row_weights) / float(np.sum(row_weights))
    return row_weights.astype(np.float64, copy=False)


def make_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        metric="l2",
        n_estimators=ITERATIONS_PER_ROUND,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES,
        max_bin=255,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=8.0,
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def team_reference_metrics() -> dict[str, object]:
    metrics = json.loads(TEAM_EB_REFERENCE.read_text(encoding="utf-8"))
    season_metrics = {
        str(season): metrics["folds"][str(season)]["candidates"][
            TEAM_EB_VARIANT
        ]["team_eb"]
        for season in REPORT_SEASONS
    }
    aggregate = metrics["aggregate_2022_2024"][TEAM_EB_VARIANT]
    return {
        "source": str(TEAM_EB_REFERENCE),
        "variant": TEAM_EB_VARIANT,
        "season_skills": {
            season: float(values["skill_score_unclipped"])
            for season, values in season_metrics.items()
        },
        "season_briers": {
            season: float(values["brier_score"])
            for season, values in season_metrics.items()
        },
        "mean_skill": float(aggregate["team_eb_mean_skill"]),
        "min_skill": float(aggregate["team_eb_min_skill"]),
    }


def season_balanced_reference_metrics() -> dict[str, object]:
    metrics = json.loads(
        SEASON_BALANCED_REFERENCE.read_text(encoding="utf-8")
    )
    variants: dict[str, object] = {}
    for weight in BLEND_WEIGHTS:
        variant = f"branch_w{int(weight * 100):03d}"
        variants[variant] = {
            "season_metrics": {
                str(season): metrics["folds"][str(season)][variant]
                for season in REPORT_SEASONS
            },
            "aggregate": metrics["aggregate_2022_2024"][variant],
        }
    return {
        "source": str(SEASON_BALANCED_REFERENCE),
        "training_weighting": "equal total mass per historical season",
        "variants": variants,
    }


def aggregate_folds(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for weight in BLEND_WEIGHTS:
        candidate = f"groupdro_w{int(weight * 100):03d}"
        skills = {
            season: float(
                folds[str(season)][candidate]["skill_score_unclipped"]
            )
            for season in REPORT_SEASONS
        }
        briers = {
            season: float(folds[str(season)][candidate]["brier_score"])
            for season in REPORT_SEASONS
        }
        aggregate[candidate] = {
            "season_skills": {
                str(season): value for season, value in skills.items()
            },
            "season_briers": {
                str(season): value for season, value in briers.items()
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
            "season_calibration": {
                str(season): {
                    "mean_gap": float(
                        folds[str(season)][candidate]["mean_gap"]
                    ),
                    "diagnostic_calibration_slope": float(
                        folds[str(season)][candidate][
                            "diagnostic_calibration_slope"
                        ]
                    ),
                    "diagnostic_calibration_intercept": float(
                        folds[str(season)][candidate][
                            "diagnostic_calibration_intercept"
                        ]
                    ),
                }
                for season in REPORT_SEASONS
            },
        }
    return aggregate


def compare_to_season_balanced(
    aggregate: dict[str, object],
    reference: dict[str, object],
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    for weight in BLEND_WEIGHTS:
        candidate = f"groupdro_w{int(weight * 100):03d}"
        variant = f"branch_w{int(weight * 100):03d}"
        reference_variant = reference["variants"][variant]
        season_changes: dict[str, object] = {}
        for season in REPORT_SEASONS:
            current = aggregate[candidate]
            previous = reference_variant["season_metrics"][str(season)]
            season_changes[str(season)] = {
                "brier_change": float(
                    current["season_briers"][str(season)]
                    - previous["brier_score"]
                ),
                "skill_change": float(
                    current["season_skills"][str(season)]
                    - previous["skill_score_unclipped"]
                ),
                "mean_gap_change": float(
                    current["season_calibration"][str(season)]["mean_gap"]
                    - previous["mean_gap"]
                ),
                "calibration_slope_change": float(
                    current["season_calibration"][str(season)][
                        "diagnostic_calibration_slope"
                    ]
                    - previous["diagnostic_calibration_slope"]
                ),
            }
        comparisons[candidate] = {
            "reference_variant": variant,
            "season_changes": season_changes,
            "mean_skill_change": float(
                aggregate[candidate]["mean_skill"]
                - reference_variant["aggregate"]["mean_skill"]
            ),
            "min_skill_change": float(
                aggregate[candidate]["min_skill"]
                - reference_variant["aggregate"]["min_skill"]
            ),
        }
    return comparisons


def compare_to_team_allprior(
    aggregate: dict[str, object],
    reference: dict[str, object],
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    for candidate, current in aggregate.items():
        comparisons[candidate] = {
            "season_changes": {
                str(season): {
                    "brier_change": float(
                        current["season_briers"][str(season)]
                        - reference["season_briers"][str(season)]
                    ),
                    "skill_change": float(
                        current["season_skills"][str(season)]
                        - reference["season_skills"][str(season)]
                    ),
                }
                for season in REPORT_SEASONS
            },
            "mean_skill_change": float(
                current["mean_skill"] - reference["mean_skill"]
            ),
            "min_skill_change": float(
                current["min_skill"] - reference["min_skill"]
            ),
        }
    return comparisons


def main() -> None:
    started = time.time()
    frame, _, y, base, seasons, reconstruction = (
        multirate.prepare_multirate_data()
    )
    game_types = frame["game_type"].astype(str).to_numpy()
    is_r = game_types == "R"
    phases = phase_labels(frame["game_month"].to_numpy(dtype=np.int8))
    all_group_keys = group_keys(seasons, phases)

    group_all, group_reported = build_group_oof(frame, y, base, seasons)
    residual_target = (y.astype(float) - group_all).astype(np.float32)
    for season in np.unique(seasons):
        mask = (seasons == season) & is_r
        residual_target[mask] -= float(np.mean(residual_target[mask]))

    X, feature_names, feature_diagnostics = build_feature_matrix(frame)
    del frame, base, group_all
    gc.collect()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"features={len(feature_names)} "
        f"matrix_mib={feature_diagnostics['matrix_mib']:.1f}",
        flush=True,
    )

    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        validation_mask = seasons == validation_season
        validation_r = validation_mask & is_r
        validation_types = game_types[validation_mask]
        validation_phases = phases[validation_mask]
        targets = y[validation_mask]

        train_mask = (seasons < validation_season) & is_r
        train_indices = np.flatnonzero(train_mask)
        local_keys = all_group_keys[train_mask]
        unique_groups, local_codes = np.unique(local_keys, return_inverse=True)
        group_counts = np.bincount(
            local_codes, minlength=len(unique_groups)
        ).astype(np.float64)
        dro_weights = np.ones(len(unique_groups), dtype=np.float64)
        round_details: list[dict[str, object]] = []
        model: LGBMRegressor | None = None

        for round_index in range(DRO_ROUNDS):
            sample_weight = row_weights_from_groups(
                local_codes,
                dro_weights,
                group_counts,
            )
            model = make_model()
            fit_started = time.time()
            model.fit(
                X[train_indices],
                residual_target[train_indices],
                sample_weight=sample_weight,
            )
            train_prediction = model.booster_.predict(
                X[train_indices]
            ).astype(float)
            squared_error = (
                residual_target[train_indices].astype(float)
                - train_prediction
            ) ** 2
            losses = np.array(
                [
                    float(np.mean(squared_error[local_codes == code]))
                    for code in range(len(unique_groups))
                ],
                dtype=float,
            )
            detail: dict[str, object] = {
                "round": round_index + 1,
                "fit_seconds": time.time() - fit_started,
                "group_weight_ratio": float(
                    np.max(dro_weights) / np.min(dro_weights)
                ),
                "worst_group": str(unique_groups[int(np.argmax(losses))]),
                "best_group": str(unique_groups[int(np.argmin(losses))]),
                "groups": {
                    str(group): {
                        "rows": int(group_counts[index]),
                        "weight": float(dro_weights[index]),
                        "residual_mse": float(losses[index]),
                    }
                    for index, group in enumerate(unique_groups)
                },
            }
            if round_index < DRO_ROUNDS - 1:
                dro_weights = capped_group_update(dro_weights, losses)
                detail["next_group_weight_ratio"] = float(
                    np.max(dro_weights) / np.min(dro_weights)
                )
            round_details.append(detail)

        if model is None:
            raise AssertionError("GroupDRO did not fit a model")

        correction = np.zeros(int(validation_mask.sum()), dtype=np.float64)
        local_r = validation_types == "R"
        correction[local_r] = model.booster_.predict(X[validation_r]).astype(
            float
        )
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "training_seasons": sorted(
                np.unique(seasons[train_mask]).astype(int).tolist()
            ),
            "training_R_rows": int(train_mask.sum()),
            "validation_R_rows": int(validation_r.sum()),
            "dro_rounds": round_details,
            "final_group_weight_ratio": float(
                np.max(dro_weights) / np.min(dro_weights)
            ),
            "correction_mean_R": float(np.mean(correction[local_r])),
            "correction_std_R": float(np.std(correction[local_r])),
            "feature_importance": {
                name: int(value)
                for name, value in sorted(
                    zip(feature_names, model.feature_importances_, strict=True),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
        }

        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets)
        np.save(
            ARTIFACT_DIR / f"residual_correction_{validation_season}.npy",
            correction,
        )
        for weight in BLEND_WEIGHTS:
            candidate = f"groupdro_w{int(weight * 100):03d}"
            predictions = np.clip(
                group_reported[validation_season] + weight * correction,
                0.0,
                1.0,
            )
            fold[candidate] = calculate_metrics(targets, predictions)
            fold[f"regimes_{candidate}"] = {
                regime: calculate_metrics(
                    targets[validation_types == regime],
                    predictions[validation_types == regime],
                )
                for regime in sorted(np.unique(validation_types))
            }
            fold[f"R_phases_{candidate}"] = {
                phase: calculate_metrics(
                    targets[
                        (validation_types == "R")
                        & (validation_phases == phase)
                    ],
                    predictions[
                        (validation_types == "R")
                        & (validation_phases == phase)
                    ],
                )
                for phase in PHASES
            }
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate}_{validation_season}.npy",
                predictions,
            )
        folds[str(validation_season)] = fold
        print(
            f"groupdro {validation_season}: "
            + " ".join(
                f"w{int(weight * 100):03d}="
                f"{fold[f'groupdro_w{int(weight * 100):03d}']['skill_score_unclipped']:.2f}"
                for weight in BLEND_WEIGHTS
            )
            + f" final_ratio={fold['final_group_weight_ratio']:.3f}",
            flush=True,
        )
        del model
        gc.collect()

    aggregate = aggregate_folds(folds)
    best_candidate = max(
        aggregate,
        key=lambda name: (
            float(aggregate[name]["min_skill"]),
            float(aggregate[name]["latest_2024_skill"]),
            float(aggregate[name]["mean_skill"]),
        ),
    )
    team_reference = team_reference_metrics()
    season_balanced_reference = season_balanced_reference_metrics()
    season_balanced_comparison = compare_to_season_balanced(
        aggregate,
        season_balanced_reference,
    )
    team_comparison = compare_to_team_allprior(
        aggregate,
        team_reference,
    )
    benchmark_min = float(team_reference["min_skill"])
    best_min = float(aggregate[best_candidate]["min_skill"])
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": "bounded_season_phase_groupdro_R_residual",
        "validation_protocol": {
            "outer_folds": list(VALIDATION_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "dro_groups": "training season x phase among R rows",
            "phase_definition": {
                "early": "game_month <= 5",
                "mid": "6 <= game_month <= 7",
                "late": "game_month >= 8",
            },
            "dro_loss": "training-only within-group residual MSE",
            "F_prediction": "past-only all-row group offset; zero residual correction",
            "current_fold_labels_used_for_training_or_weights": False,
            "test_row_aggregation": False,
            "candidate_comparison_status": "one predeclared config and two fixed blend weights",
        },
        "model": {
            "dro_rounds": DRO_ROUNDS,
            "dro_eta": DRO_ETA,
            "dro_max_group_weight_ratio": DRO_MAX_GROUP_RATIO,
            "iterations_per_refit": ITERATIONS_PER_ROUND,
            "refit_policy": "from scratch after each training-only DRO update",
            "learning_rate": LEARNING_RATE,
            "num_leaves": NUM_LEAVES,
            "min_child_samples": MIN_CHILD_SAMPLES,
            "blend_weights": list(BLEND_WEIGHTS),
            "features": feature_names,
            "feature_diagnostics": feature_diagnostics,
            "excluded": [
                "pitcher_id",
                "batter_id",
                "pitcher_team_id",
                "batter_team_id",
                "season",
                "game_type",
            ],
        },
        "reconstruction_diagnostics": reconstruction,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "references": {
            "strict_team_allprior": team_reference,
            "season_balanced_same_capacity": season_balanced_reference,
        },
        "comparison_to_season_balanced": season_balanced_comparison,
        "comparison_to_team_allprior": team_comparison,
        "selection": {
            "best_fixed_weight": best_candidate,
            "best_min_skill": best_min,
            "team_allprior_min_skill": benchmark_min,
            "beats_team_allprior": bool(best_min > benchmark_min),
            "stop_rule_triggered": bool(best_min < benchmark_min),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
        "total_seconds": time.time() - started,
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
