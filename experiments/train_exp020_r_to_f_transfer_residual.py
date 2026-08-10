"""EXP-020: transfer a shallow regular-season residual model to F rows.

This experiment reuses the EXP-019 R-full temporal group residual protocol.
A single LightGBM is trained only on past regular-season rows and their
source-season-centered residuals.  Unlike R-full, its correction is evaluated
only on validation ``game_type == F`` rows.  R predictions remain immutable
at the saved EXP-020 team-plus-pitcher-count OOF core, making the F transfer
effect directly attributable.

Raw player IDs, team IDs, season, game type, current-fold labels, and any
validation/test-row aggregate are excluded from model features.  Transfer
weights 0.25, 0.50, and 0.75 are fixed before evaluation.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import train_exp019_multirate_residual as multirate
import train_exp019_r_full_residual as rfull
from train_exp017_rolling_residual import calculate_metrics, segment_metrics
from train_exp019_stable_monotonic import season_equal_weights


ARTIFACT_DIR = Path("./artifacts/EXP-020/r_to_f_transfer_residual")
CORE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
CORE_VARIANT = "team_pc_all"
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
TRANSFER_WEIGHTS = (0.25, 0.50, 0.75)
ITERATIONS = 200
LEARNING_RATE = 0.015
NUM_LEAVES = 15
MIN_CHILD_SAMPLES = 3000


def candidate_name(weight: float) -> str:
    return f"f_transfer_w{int(round(weight * 100)):03d}"


def build_feature_matrix(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    model_columns = [
        column
        for column in frame.columns
        if column not in rfull.DROP_COLUMNS
        and column not in rfull.CATEGORICAL_COLUMNS
    ]
    numeric = frame[model_columns].select_dtypes(include=[np.number]).astype(
        np.float32
    )
    categorical = pd.get_dummies(
        frame[rfull.CATEGORICAL_COLUMNS],
        dummy_na=True,
        dtype=np.int8,
    )
    feature_names = numeric.columns.tolist() + categorical.columns.tolist()
    forbidden_tokens = ("pitcher_id", "batter_id", "team_id")
    forbidden_exact = {
        "season",
        "row_id",
        "control_success",
        "game_type",
    }
    invalid = [
        name
        for name in feature_names
        if name in forbidden_exact
        or any(token in name for token in forbidden_tokens)
    ]
    if invalid:
        raise ValueError(f"forbidden model features: {invalid}")
    X = np.column_stack(
        [
            numeric.to_numpy(dtype=np.float32),
            categorical.to_numpy(dtype=np.float32),
        ]
    )
    return np.ascontiguousarray(X), feature_names


def build_group_predictions(
    frame: pd.DataFrame,
    y: np.ndarray,
    temporal_base: np.ndarray,
    seasons: np.ndarray,
) -> np.ndarray:
    initial_residual = multirate.centered_residual(
        y,
        temporal_base,
        seasons,
    )
    group_predictions = np.empty(len(y), dtype=np.float64)
    for season in sorted(np.unique(seasons).astype(int).tolist()):
        mask = seasons == season
        correction = rfull.original_group_correction(
            frame,
            initial_residual,
            seasons,
            season,
        )
        group_predictions[mask] = np.clip(
            temporal_base[mask].astype(float) + correction,
            0.0,
            1.0,
        )
    return group_predictions


def load_core_oof(
    y: np.ndarray,
    seasons: np.ndarray,
) -> dict[int, np.ndarray]:
    core: dict[int, np.ndarray] = {}
    for season in VALIDATION_SEASONS:
        mask = seasons == season
        predictions = np.load(
            CORE_ROOT / f"predictions_{CORE_VARIANT}_{season}.npy"
        ).astype(float)
        saved_targets = np.load(
            CORE_ROOT / f"targets_{season}.npy"
        ).astype(np.int8)
        current_targets = y[mask].astype(np.int8)
        if not (
            len(predictions) == int(mask.sum()) == len(saved_targets)
            and np.array_equal(current_targets, saved_targets)
            and np.isfinite(predictions).all()
        ):
            raise ValueError(f"core OOF alignment mismatch for {season}")
        core[season] = np.clip(predictions, 0.0, 1.0)
    return core


def make_model() -> LGBMRegressor:
    return LGBMRegressor(
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


def regime_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    validation_types: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        game_type: calculate_metrics(
            targets[validation_types == game_type],
            predictions[validation_types == game_type],
        )
        for game_type in sorted(np.unique(validation_types))
    }


def aggregate_folds(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for weight in TRANSFER_WEIGHTS:
        candidate = candidate_name(weight)
        skills = {
            season: float(
                folds[str(season)]["candidates"][candidate]["metrics"][
                    "skill_score_unclipped"
                ]
            )
            for season in REPORT_SEASONS
        }
        briers = {
            season: float(
                folds[str(season)]["candidates"][candidate]["metrics"][
                    "brier_score"
                ]
            )
            for season in REPORT_SEASONS
        }
        f_skills = {
            season: float(
                folds[str(season)]["candidates"][candidate]["regimes"][
                    "F"
                ]["skill_score_unclipped"]
            )
            for season in REPORT_SEASONS
        }
        aggregate[candidate] = {
            "season_skills": {
                str(season): value for season, value in skills.items()
            },
            "season_briers": {
                str(season): value for season, value in briers.items()
            },
            "season_F_skills": {
                str(season): value for season, value in f_skills.items()
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
            "mean_F_skill": float(np.mean(list(f_skills.values()))),
            "min_F_skill": float(np.min(list(f_skills.values()))),
        }
    return aggregate


def main() -> None:
    started = time.time()
    frame, diagnostics, y, temporal_base, seasons, reconstruction = (
        multirate.prepare_multirate_data()
    )
    X, feature_names = build_feature_matrix(frame)
    group_predictions = build_group_predictions(
        frame,
        y,
        temporal_base,
        seasons,
    )
    core_oof = load_core_oof(y, seasons)
    game_types = frame["game_type"].astype(str).to_numpy()
    is_r = game_types == "R"
    is_f = game_types == "F"
    if not np.array_equal(~is_r, is_f):
        raise ValueError("expected exactly R and F game_type values")
    residual_target = (y.astype(float) - group_predictions).astype(
        np.float32
    )
    for season in np.unique(seasons):
        mask = (seasons == season) & is_r
        residual_target[mask] -= residual_target[mask].mean()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        train_mask = (seasons < validation_season) & is_r
        validation_mask = seasons == validation_season
        validation_f = validation_mask & is_f
        targets = y[validation_mask].astype(float)
        validation_types = game_types[validation_mask]
        core = core_oof[validation_season]
        model = make_model()
        fit_started = time.time()
        model.fit(
            X[train_mask],
            residual_target[train_mask],
            sample_weight=season_equal_weights(seasons[train_mask]),
        )
        correction = np.zeros(int(validation_mask.sum()), dtype=float)
        local_f = is_f[validation_mask]
        correction[local_f] = model.booster_.predict(
            X[validation_f]
        ).astype(float)
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "training_seasons": sorted(
                np.unique(seasons[train_mask]).astype(int).tolist()
            ),
            "training_R_rows": int(train_mask.sum()),
            "validation_F_rows": int(validation_f.sum()),
            "fit_seconds": time.time() - fit_started,
            "core_team_pc_all": {
                "metrics": calculate_metrics(targets, core),
                "regimes": regime_metrics(
                    targets,
                    core,
                    validation_types,
                ),
            },
            "correction": {
                "applied_only_to": "F",
                "F_mean": float(correction[local_f].mean()),
                "F_std": float(correction[local_f].std()),
                "F_min": float(correction[local_f].min()),
                "F_max": float(correction[local_f].max()),
                "R_is_exactly_zero": bool(
                    np.all(correction[~local_f] == 0.0)
                ),
            },
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
            "candidates": {},
        }
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets)
        np.save(
            ARTIFACT_DIR / f"f_correction_{validation_season}.npy",
            correction,
        )
        for weight in TRANSFER_WEIGHTS:
            candidate = candidate_name(weight)
            predictions = np.clip(
                core + weight * correction,
                0.0,
                1.0,
            )
            fold["candidates"][candidate] = {
                "metrics": calculate_metrics(targets, predictions),
                "regimes": regime_metrics(
                    targets,
                    predictions,
                    validation_types,
                ),
                "segments": segment_metrics(
                    diagnostics,
                    validation_mask,
                    targets,
                    predictions,
                ),
            }
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate}_{validation_season}.npy",
                predictions,
            )
        folds[str(validation_season)] = fold
        print(
            f"r_to_f {validation_season}: core="
            f"{fold['core_team_pc_all']['metrics']['skill_score_unclipped']:.2f} "
            + " ".join(
                f"w{int(weight * 100):03d}="
                f"{fold['candidates'][candidate_name(weight)]['metrics']['skill_score_unclipped']:.2f}"
                for weight in TRANSFER_WEIGHTS
            ),
            flush=True,
        )

    aggregate = aggregate_folds(folds)
    posthoc_best = max(
        aggregate,
        key=lambda candidate: (
            float(aggregate[candidate]["min_skill"]),
            float(aggregate[candidate]["latest_2024_skill"]),
            float(aggregate[candidate]["mean_skill"]),
        ),
    )
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": "R_to_F_transfer_residual",
        "validation_protocol": {
            "outer_folds": list(VALIDATION_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "residual_base": (
                "EXP-019 temporal global30 plus past-only count/hand/reverse group"
            ),
            "model_training": "prior-season R rows only",
            "residual_centering": "inside each source season and R",
            "immutable_prediction_base": "saved team_pc_all OOF",
            "R_prediction": "immutable team_pc_all core for every candidate",
            "correction_application": "F rows only",
            "current_fold_labels_used_for_training": False,
            "test_row_aggregation": False,
            "candidate_selection": "post-hoc diagnostic ranking only",
        },
        "model": {
            "features": feature_names,
            "feature_count": len(feature_names),
            "iterations": ITERATIONS,
            "learning_rate": LEARNING_RATE,
            "num_leaves": NUM_LEAVES,
            "min_child_samples": MIN_CHILD_SAMPLES,
            "transfer_weights": list(TRANSFER_WEIGHTS),
            "excluded": [
                "raw player IDs",
                "team IDs",
                "season",
                "game_type",
                "validation/test-row aggregates",
            ],
        },
        "reconstruction_diagnostics": reconstruction,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "post-hoc diagnostic; not a nested selection",
            "posthoc_best_candidate": posthoc_best,
            "posthoc_best_min_skill": aggregate[posthoc_best]["min_skill"],
            "posthoc_best_latest_2024_skill": aggregate[posthoc_best][
                "latest_2024_skill"
            ],
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
