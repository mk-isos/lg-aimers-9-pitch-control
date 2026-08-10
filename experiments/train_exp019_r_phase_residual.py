"""EXP-019: phase-matched R-only full residual LightGBM.

The temporal hierarchical base and past-only all-row count/hand/reverse group
offset are identical to the R-full experiment.  The remaining season-centered
R residual is modeled by three separate shallow LightGBM experts:

* early: ``game_month <= 5``
* mid: ``game_month in {6, 7}``
* late: ``game_month >= 8``

Each expert sees only rows from the same phase in seasons strictly earlier than
the validation season.  Non-regular-season rows keep the group-only prediction.
Raw player IDs, team IDs, season, validation-row aggregation, and test-row
aggregation are excluded.
"""

from __future__ import annotations

import gc
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import train_exp019_multirate_residual as multirate
from train_exp017_rolling_residual import calculate_metrics
from train_exp019_r_full_residual import (
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
    original_group_correction,
)
from train_exp019_stable_monotonic import season_equal_weights


ARTIFACT_DIR = Path("./artifacts/EXP-019/r_phase_residual")
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
BLEND_WEIGHTS = (0.50, 0.75)
PHASES = ("early", "mid", "late")

# One predeclared shallow configuration.  The 3,000-row leaf floor is retained
# from the most regularized R-full model even though each expert sees fewer rows.
ITERATIONS = 200
LEARNING_RATE = 0.015
NUM_LEAVES = 15
MIN_CHILD_SAMPLES = 3000

GLOBAL_REFERENCE = Path(
    "./artifacts/EXP-019/r_full_residual/"
    "rfull_l63_m1000_i300/validation_metrics.json"
)
GLOBAL_REFERENCE_VARIANT = "branch_w075"
HGB_REFERENCE = Path(
    "./artifacts/EXP-019/histgb_residual/"
    "hist_l15_d4_m3000_i160/validation_metrics.json"
)
HGB_REFERENCE_VARIANT = "branch_w100"
TEAM_REFERENCE = Path(
    "./artifacts/EXP-019/r_team_lgbm/"
    "rteam_l31_m2000_i300/validation_metrics.json"
)
TEAM_REFERENCE_VARIANT = "branch_w075"
TEAM_EB_REFERENCE = Path(
    "./artifacts/EXP-019/team_eb_ensemble/validation_metrics.json"
)
TEAM_EB_REFERENCE_VARIANT = "prior1_s500"


def phase_labels(months: np.ndarray) -> np.ndarray:
    return np.where(
        months <= 5,
        "early",
        np.where(months <= 7, "mid", "late"),
    )


def build_group_oof(
    frame: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    initial_residual = multirate.centered_residual(y, base, seasons)
    group_all = np.empty(len(y), dtype=np.float64)
    group_reported: dict[int, np.ndarray] = {}
    for season in sorted(np.unique(seasons).astype(int).tolist()):
        mask = seasons == season
        correction = original_group_correction(
            frame,
            initial_residual,
            seasons,
            season,
        )
        prediction = np.clip(base[mask].astype(float) + correction, 0.0, 1.0)
        group_all[mask] = prediction
        if season in VALIDATION_SEASONS:
            group_reported[season] = prediction
    return group_all, group_reported


def build_feature_matrix(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    model_columns = [
        column
        for column in frame.columns
        if column not in DROP_COLUMNS and column not in CATEGORICAL_COLUMNS
    ]
    numeric_columns = frame[model_columns].select_dtypes(
        include=[np.number]
    ).columns.tolist()
    forbidden = {
        "season",
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "batter_team_id",
    }
    leaked = sorted(forbidden.intersection(numeric_columns))
    if leaked:
        raise ValueError(f"forbidden features selected: {leaked}")
    categorical = pd.get_dummies(
        frame[CATEGORICAL_COLUMNS],
        dummy_na=True,
        dtype=np.int8,
    )
    feature_names = numeric_columns + categorical.columns.tolist()
    X = np.empty((len(frame), len(feature_names)), dtype=np.float32)
    for index, column in enumerate(numeric_columns):
        X[:, index] = frame[column].to_numpy(dtype=np.float32)
    offset = len(numeric_columns)
    for local_index, column in enumerate(categorical.columns):
        X[:, offset + local_index] = categorical[column].to_numpy(
            dtype=np.float32
        )
    diagnostics = {
        "numeric_features": len(numeric_columns),
        "categorical_one_hot_features": int(categorical.shape[1]),
        "total_features": len(feature_names),
        "matrix_mib": float(X.nbytes / 2**20),
        "raw_player_ids_in_matrix": False,
        "raw_team_ids_in_matrix": False,
        "season_in_matrix": False,
    }
    del categorical
    gc.collect()
    return X, feature_names, diagnostics


def reference_metrics() -> dict[str, object]:
    references: dict[str, object] = {}
    for name, path, variant in (
        ("global_r_full", GLOBAL_REFERENCE, GLOBAL_REFERENCE_VARIANT),
        ("histgb", HGB_REFERENCE, HGB_REFERENCE_VARIANT),
        ("r_team_lgbm", TEAM_REFERENCE, TEAM_REFERENCE_VARIANT),
    ):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        references[name] = {
            "source": str(path),
            "variant": variant,
            "season_skills": {
                str(season): metrics["folds"][str(season)][variant][
                    "skill_score_unclipped"
                ]
                for season in REPORT_SEASONS
            },
            "mean_skill": metrics["aggregate_2022_2024"][variant][
                "mean_skill"
            ],
            "min_skill": metrics["aggregate_2022_2024"][variant][
                "min_skill"
            ],
        }
    team_eb = json.loads(TEAM_EB_REFERENCE.read_text(encoding="utf-8"))
    references["hgb_lgb_team_eb"] = {
        "source": str(TEAM_EB_REFERENCE),
        "variant": TEAM_EB_REFERENCE_VARIANT,
        "season_skills": {
            str(season): team_eb["folds"][str(season)]["candidates"][
                TEAM_EB_REFERENCE_VARIANT
            ]["team_eb"]["skill_score_unclipped"]
            for season in REPORT_SEASONS
        },
        "mean_skill": team_eb["aggregate_2022_2024"][
            TEAM_EB_REFERENCE_VARIANT
        ]["team_eb_mean_skill"],
        "min_skill": team_eb["aggregate_2022_2024"][
            TEAM_EB_REFERENCE_VARIANT
        ]["team_eb_min_skill"],
    }
    return references


def aggregate_folds(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for weight in BLEND_WEIGHTS:
        candidate = f"phase_w{int(weight * 100):03d}"
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
        }
    return aggregate


def main() -> None:
    started = time.time()
    frame, _, y, base, seasons, reconstruction = (
        multirate.prepare_multirate_data()
    )
    game_types = frame["game_type"].astype(str).to_numpy()
    is_r = game_types == "R"
    phases = phase_labels(frame["game_month"].to_numpy(dtype=np.int8))
    group_all, group_reported = build_group_oof(frame, y, base, seasons)
    residual_target = (y.astype(float) - group_all).astype(np.float32)
    for season in np.unique(seasons):
        mask = (seasons == season) & is_r
        residual_target[mask] -= residual_target[mask].mean()

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
        validation_types = game_types[validation_mask]
        validation_phases = phases[validation_mask]
        targets = y[validation_mask]
        correction = np.zeros(int(validation_mask.sum()), dtype=np.float64)
        phase_details: dict[str, object] = {}
        for phase in PHASES:
            train_mask = (
                (seasons < validation_season)
                & is_r
                & (phases == phase)
            )
            predict_mask = validation_mask & is_r & (phases == phase)
            local_predict_mask = (
                (validation_types == "R") & (validation_phases == phase)
            )
            if not train_mask.any() or not predict_mask.any():
                raise ValueError(
                    f"missing phase rows: season={validation_season} phase={phase}"
                )
            model = LGBMRegressor(
                objective="regression_l2",
                metric="l2",
                n_estimators=ITERATIONS,
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
            fit_started = time.time()
            model.fit(
                X[train_mask],
                residual_target[train_mask],
                sample_weight=season_equal_weights(seasons[train_mask]),
            )
            phase_prediction = model.booster_.predict(X[predict_mask]).astype(
                float
            )
            correction[local_predict_mask] = phase_prediction
            phase_details[phase] = {
                "training_rows": int(train_mask.sum()),
                "validation_R_rows": int(predict_mask.sum()),
                "training_seasons": sorted(
                    np.unique(seasons[train_mask]).astype(int).tolist()
                ),
                "fit_seconds": time.time() - fit_started,
                "correction_mean": float(phase_prediction.mean()),
                "correction_std": float(phase_prediction.std()),
                "feature_importance": {
                    name: int(value)
                    for name, value in sorted(
                        zip(
                            feature_names,
                            model.feature_importances_,
                            strict=True,
                        ),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
            }

        fold: dict[str, object] = {
            "validation_season": validation_season,
            "phase_experts": phase_details,
            "F_rows_with_zero_correction": int(
                (validation_types != "R").sum()
            ),
            "correction_mean_R": float(
                correction[validation_types == "R"].mean()
            ),
            "correction_std_R": float(
                correction[validation_types == "R"].std()
            ),
        }
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets)
        np.save(
            ARTIFACT_DIR / f"phase_correction_{validation_season}.npy",
            correction,
        )
        for weight in BLEND_WEIGHTS:
            candidate = f"phase_w{int(weight * 100):03d}"
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
            f"r_phase {validation_season}: "
            + " ".join(
                f"w{int(weight * 100):03d}="
                f"{fold[f'phase_w{int(weight * 100):03d}']['skill_score_unclipped']:.2f}"
                for weight in BLEND_WEIGHTS
            ),
            flush=True,
        )

    aggregate = aggregate_folds(folds)
    best_candidate = max(
        aggregate,
        key=lambda name: (
            float(aggregate[name]["min_skill"]),
            float(aggregate[name]["latest_2024_skill"]),
            float(aggregate[name]["mean_skill"]),
        ),
    )
    references = reference_metrics()
    benchmark_min = float(references["hgb_lgb_team_eb"]["min_skill"])
    result: dict[str, object] = {
        "experiment": "EXP-019",
        "candidate_family": "phase_matched_R_only_full_residual_lightgbm",
        "validation_protocol": {
            "outer_folds": list(VALIDATION_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "phase_definition": {
                "early": "game_month <= 5",
                "mid": "6 <= game_month <= 7",
                "late": "game_month >= 8",
            },
            "phase_training": "matching phase R rows from earlier seasons only",
            "residual_centering": "within each past season across R rows",
            "F_prediction": "past-only all-row group offset; zero residual correction",
            "current_fold_labels_used_for_training": False,
            "test_row_aggregation": False,
            "candidate_comparison_status": "single predeclared config and weights",
        },
        "model": {
            "iterations": ITERATIONS,
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
        "references": references,
        "selection": {
            "best_fixed_weight": best_candidate,
            "best_min_skill": aggregate[best_candidate]["min_skill"],
            "team_eb_benchmark_min_skill": benchmark_min,
            "beats_team_eb_benchmark": bool(
                float(aggregate[best_candidate]["min_skill"])
                > benchmark_min
            ),
            "stop_rule_triggered": bool(
                float(aggregate[best_candidate]["min_skill"])
                < benchmark_min
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
        "total_seconds": time.time() - started,
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f"selection={result['selection']}", flush=True)
    print(f"saved={ARTIFACT_DIR / 'validation_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
