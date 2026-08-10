"""EXP-019: R-only full residual LightGBM with categorical team indicators.

This is a focused derivative of ``train_exp019_r_full_residual.py``.  The
temporal base and all-row count/hand/reverse group offset are unchanged.  A
residual model is fitted only on earlier regular-season (``game_type == R``)
rows, while non-R rows retain the group-only prediction.

Unlike the earlier full residual model, anonymous team IDs never enter as
ordered numeric values.  Pitcher team, batter team, and explicit
team-by-pitcher-hand-by-batter-hand interactions are one-hot encoded.  Raw
player IDs and season are excluded.  The encoding uses only official current-
row values and never aggregates validation/test rows.
"""

from __future__ import annotations

import gc
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import train_exp019_multirate_residual as multirate
from train_exp017_rolling_residual import calculate_metrics
from train_exp019_r_full_residual import original_group_correction
from train_exp019_stable_monotonic import season_equal_weights


ARTIFACT_ROOT = Path("./artifacts/EXP-019/r_team_lgbm")
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
BLEND_WEIGHTS = (0.25, 0.50, 0.75, 1.00)
REFERENCE_HGB_LGB_TEAM_EB_MIN = 850.4

DROP_COLUMNS = {
    "row_id",
    "control_success",
    "season",
    "game_type",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
}
BASE_CATEGORICAL_COLUMNS = ("top_bottom", "base_state")


@dataclass(frozen=True)
class Config:
    name: str
    iterations: int
    num_leaves: int
    min_child_samples: int


CONFIGS = (
    Config("rteam_l15_m3000_i200", 200, 15, 3000),
    Config("rteam_l31_m2000_i300", 300, 31, 2000),
)


def string_code(series: pd.Series) -> pd.Series:
    """Stable string representation for one-hot keys, including missing."""
    return series.astype("Int64").astype("string").fillna("NA")


def build_one_hot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    pitcher_team = string_code(frame["pitcher_team_id"])
    batter_team = string_code(frame["batter_team_id"])
    pitcher_hand = string_code(frame["pitcher_hand"])
    batter_hand = string_code(frame["batter_hand"])
    categories = pd.DataFrame(
        {
            "top_bottom": frame["top_bottom"].astype("string").fillna("NA"),
            "base_state": frame["base_state"].astype("string").fillna("NA"),
            "pitcher_team": pitcher_team,
            "batter_team": batter_team,
            "pitcher_team_hand_context": (
                pitcher_team + "|PH" + pitcher_hand + "|BH" + batter_hand
            ),
            "batter_team_hand_context": (
                batter_team + "|PH" + pitcher_hand + "|BH" + batter_hand
            ),
        }
    )
    return pd.get_dummies(categories, dummy_na=False, dtype=np.int8)


def build_feature_matrix(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    model_columns = [
        column
        for column in frame.columns
        if column not in DROP_COLUMNS
        and column not in BASE_CATEGORICAL_COLUMNS
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
        raise ValueError(f"forbidden numeric columns selected: {leaked}")

    one_hot = build_one_hot_frame(frame)
    feature_names = numeric_columns + one_hot.columns.tolist()
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("duplicate encoded feature names")
    X = np.empty((len(frame), len(feature_names)), dtype=np.float32)
    for index, column in enumerate(numeric_columns):
        X[:, index] = frame[column].to_numpy(dtype=np.float32)
    offset = len(numeric_columns)
    for local_index, column in enumerate(one_hot.columns):
        X[:, offset + local_index] = one_hot[column].to_numpy(dtype=np.float32)

    prefix_counts = {
        "pitcher_team": int(
            sum(
                name.startswith("pitcher_team_")
                and not name.startswith("pitcher_team_hand_context_")
                for name in one_hot.columns
            )
        ),
        "batter_team": int(
            sum(
                name.startswith("batter_team_")
                and not name.startswith("batter_team_hand_context_")
                for name in one_hot.columns
            )
        ),
        "pitcher_team_hand_context": int(
            sum(
                name.startswith("pitcher_team_hand_context_")
                for name in one_hot.columns
            )
        ),
        "batter_team_hand_context": int(
            sum(
                name.startswith("batter_team_hand_context_")
                for name in one_hot.columns
            )
        ),
    }
    diagnostics = {
        "numeric_features": len(numeric_columns),
        "one_hot_features": int(one_hot.shape[1]),
        "total_features": len(feature_names),
        "one_hot_prefix_counts": prefix_counts,
        "raw_team_ids_in_numeric_matrix": False,
        "raw_player_ids_in_matrix": False,
        "season_in_matrix": False,
        "matrix_mib": float(X.nbytes / 2**20),
    }
    del one_hot
    gc.collect()
    return X, feature_names, diagnostics


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


def aggregate_result(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for weight in BLEND_WEIGHTS:
        candidate = f"branch_w{int(weight * 100):03d}"
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


def run_config(
    config: Config,
    X: np.ndarray,
    feature_names: list[str],
    y: np.ndarray,
    seasons: np.ndarray,
    is_r: np.ndarray,
    game_types: np.ndarray,
    residual_target: np.ndarray,
    group_reported: dict[int, np.ndarray],
    feature_diagnostics: dict[str, object],
    reconstruction: dict[str, object],
) -> dict[str, object]:
    artifact_dir = ARTIFACT_ROOT / config.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        train_mask = (seasons < validation_season) & is_r
        validation_mask = seasons == validation_season
        validation_r = validation_mask & is_r
        targets = y[validation_mask]
        model = LGBMRegressor(
            objective="regression_l2",
            metric="l2",
            n_estimators=config.iterations,
            learning_rate=0.015,
            num_leaves=config.num_leaves,
            min_child_samples=config.min_child_samples,
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
        correction = np.zeros(int(validation_mask.sum()), dtype=float)
        local_r = is_r[validation_mask]
        correction[local_r] = model.predict(X[validation_r]).astype(float)
        validation_types = game_types[validation_mask]
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "training_seasons": sorted(
                np.unique(seasons[train_mask]).astype(int).tolist()
            ),
            "fit_seconds": time.time() - fit_started,
            "correction_mean_R": float(correction[local_r].mean()),
            "correction_std_R": float(correction[local_r].std()),
            "feature_importance": {
                name: int(value)
                for name, value in sorted(
                    zip(feature_names, model.feature_importances_, strict=True),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
        }
        np.save(artifact_dir / f"targets_{validation_season}.npy", targets)
        np.save(
            artifact_dir / f"residual_correction_{validation_season}.npy",
            correction,
        )
        for weight in BLEND_WEIGHTS:
            candidate = f"branch_w{int(weight * 100):03d}"
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
            np.save(
                artifact_dir
                / f"predictions_{candidate}_{validation_season}.npy",
                predictions,
            )
        folds[str(validation_season)] = fold
        print(
            f"{config.name} {validation_season}: "
            + " ".join(
                f"w{int(weight * 100):03d}="
                f"{fold[f'branch_w{int(weight * 100):03d}']['skill_score_unclipped']:.2f}"
                for weight in BLEND_WEIGHTS
            ),
            flush=True,
        )

    aggregate = aggregate_result(folds)
    best_candidate = max(
        aggregate,
        key=lambda name: (
            float(aggregate[name]["min_skill"]),
            float(aggregate[name]["latest_2024_skill"]),
            float(aggregate[name]["mean_skill"]),
        ),
    )
    result: dict[str, object] = {
        "experiment": "EXP-019",
        "candidate_family": "R_only_team_one_hot_residual_lightgbm",
        "candidate": config.name,
        "validation_protocol": {
            "outer_folds": list(VALIDATION_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "offset": "past-only all-row count/hand/reverse group",
            "R_model_training": "past R rows only with equal season total weight",
            "F_prediction": "all-row temporal group offset only",
            "current_fold_labels_used_for_training": False,
            "test_row_aggregation": False,
            "category_vocabulary": (
                "label-free official train schema; unknown future categories "
                "map to all-zero indicators"
            ),
            "candidate_comparison_status": "fixed diagnostic grid",
        },
        "model": {
            **asdict(config),
            "learning_rate": 0.015,
            "blend_weights": list(BLEND_WEIGHTS),
            "raw_team_encoding": "excluded",
            "team_encoding": "one-hot",
            "explicit_interactions": [
                "pitcher_team_id x pitcher_hand x batter_hand",
                "batter_team_id x pitcher_hand x batter_hand",
            ],
            "features": feature_names,
            "feature_diagnostics": feature_diagnostics,
        },
        "reconstruction_diagnostics": reconstruction,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_fixed_weight_by_min": best_candidate,
            "best_min_skill": aggregate[best_candidate]["min_skill"],
            "reference_hgb_lgb_team_eb_min_approx": (
                REFERENCE_HGB_LGB_TEAM_EB_MIN
            ),
            "beats_reference_min": bool(
                float(aggregate[best_candidate]["min_skill"])
                > REFERENCE_HGB_LGB_TEAM_EB_MIN
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    with (artifact_dir / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    started = time.time()
    frame, _, y, base, seasons, reconstruction = (
        multirate.prepare_multirate_data()
    )
    game_types = frame["game_type"].astype(str).to_numpy()
    is_r = game_types == "R"
    group_all, group_reported = build_group_oof(frame, y, base, seasons)
    residual_target = (y.astype(float) - group_all).astype(np.float32)
    for season in np.unique(seasons):
        mask = (seasons == season) & is_r
        residual_target[mask] -= residual_target[mask].mean()

    X, feature_names, feature_diagnostics = build_feature_matrix(frame)
    del frame, base, group_all
    gc.collect()
    print(
        f"features={len(feature_names)} "
        f"matrix_mib={feature_diagnostics['matrix_mib']:.1f}",
        flush=True,
    )
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    config_results: dict[str, object] = {}
    for config in CONFIGS:
        result = run_config(
            config,
            X,
            feature_names,
            y,
            seasons,
            is_r,
            game_types,
            residual_target,
            group_reported,
            feature_diagnostics,
            reconstruction,
        )
        config_results[config.name] = {
            "aggregate_2022_2024": result["aggregate_2022_2024"],
            "selection": result["selection"],
        }

    best_pair: tuple[str, str] | None = None
    best_key: tuple[float, float, float] | None = None
    for config_name, result in config_results.items():
        aggregate = result["aggregate_2022_2024"]
        for candidate_name, metrics in aggregate.items():
            key = (
                float(metrics["min_skill"]),
                float(metrics["latest_2024_skill"]),
                float(metrics["mean_skill"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_pair = (config_name, candidate_name)
    if best_pair is None or best_key is None:
        raise RuntimeError("no team LightGBM candidate was evaluated")

    root_result = {
        "experiment": "EXP-019",
        "stage": "R_only_team_one_hot_residual_lightgbm",
        "predetermined_configs": [asdict(config) for config in CONFIGS],
        "predetermined_blend_weights": list(BLEND_WEIGHTS),
        "config_results": config_results,
        "selection": {
            "best_config": best_pair[0],
            "best_candidate": best_pair[1],
            "best_min_skill": best_key[0],
            "best_latest_2024_skill": best_key[1],
            "best_mean_skill": best_key[2],
            "reference_hgb_lgb_team_eb_min_approx": (
                REFERENCE_HGB_LGB_TEAM_EB_MIN
            ),
            "beats_reference_min": bool(
                best_key[0] > REFERENCE_HGB_LGB_TEAM_EB_MIN
            ),
        },
        "total_seconds": time.time() - started,
    }
    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(root_result, file, ensure_ascii=False, indent=2)
    print(f"selection={root_result['selection']}", flush=True)
    print(f"saved={ARTIFACT_ROOT / 'validation_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
