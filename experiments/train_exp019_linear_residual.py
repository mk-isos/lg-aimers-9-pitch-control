"""EXP-019: 안정 피처 Ridge residual 비교.

temporal group OOF를 offset으로 사용하고, 고정 one-hot context와 현재 시즌
multirate의 선형 correction만 학습한다. 모든 전처리는 outer fold 학습 시즌에
fit하며 테스트 행끼리 집계하지 않는다.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import train_exp019_multirate_residual as multirate
from train_exp017_rolling_residual import calculate_metrics
from train_exp019_stable_monotonic import season_equal_weights


ARTIFACT_ROOT = Path("./artifacts/EXP-019/linear_residual")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
ALPHAS = [100.0, 1000.0, 10000.0, 100000.0]
BLEND_WEIGHTS = [0.5, 1.0]

CATEGORICAL_COLUMNS = [
    "count_index",
    "pitcher_hand",
    "batter_hand",
    "outs_before",
    "num_runners_on",
]
CONTINUOUS_COLUMNS = [
    "inning",
    "li",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "multirate_pitcher_control_log_season_n",
    "multirate_pitcher_control_reliability_30",
    "multirate_pitcher_control_success_prior_shrunk_200",
    "multirate_pitcher_control_success_season_global_30",
    "multirate_pitcher_control_reverse_prior_shrunk_200",
    "multirate_pitcher_control_reverse_season_global_30",
    "multirate_pitcher_pitchmix_breaking_prior_shrunk_200",
    "multirate_pitcher_pitchmix_breaking_season_global_30",
]


def build_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    categorical = frame[CATEGORICAL_COLUMNS].astype("Int64").astype(str)
    categorical["count_hand"] = (
        categorical["count_index"]
        + "_"
        + categorical["pitcher_hand"]
        + "_"
        + categorical["batter_hand"]
    )
    encoded = pd.get_dummies(categorical, dtype=np.float32)
    continuous = frame[CONTINUOUS_COLUMNS].astype(np.float32)
    matrix = np.column_stack(
        [continuous.to_numpy(dtype=np.float32), encoded.to_numpy(dtype=np.float32)]
    )
    return matrix, CONTINUOUS_COLUMNS + encoded.columns.tolist()


def group_oof_predictions(
    frame: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    initial_residual = multirate.centered_residual(y, base, seasons)
    all_predictions = np.empty(len(y), dtype=np.float64)
    reported: dict[int, np.ndarray] = {}
    for season in sorted(np.unique(seasons).astype(int).tolist()):
        mask = seasons == season
        correction = multirate.multirate_group_correction(
            frame, initial_residual, seasons, season
        )
        prediction = np.clip(base[mask].astype(float) + correction, 0.0, 1.0)
        all_predictions[mask] = prediction
        if season in VALIDATION_SEASONS:
            reported[season] = prediction
    return all_predictions, reported


def main() -> None:
    started = time.time()
    frame, _, y, base, seasons, reconstruction = multirate.prepare_multirate_data()
    X, feature_names = build_matrix(frame)
    group_all, group_reported = group_oof_predictions(frame, y, base, seasons)
    residual_target = (y.astype(float) - group_all).astype(np.float32)
    for season in np.unique(seasons):
        mask = seasons == season
        residual_target[mask] -= residual_target[mask].mean()

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        train_mask = seasons < validation_season
        validation_mask = seasons == validation_season
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_X = imputer.fit_transform(X[train_mask])
        validation_X = imputer.transform(X[validation_mask])
        train_X = scaler.fit_transform(train_X)
        validation_X = scaler.transform(validation_X)
        targets = y[validation_mask]
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "training_seasons": sorted(
                np.unique(seasons[train_mask]).astype(int).tolist()
            ),
        }
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
            fit_started = time.time()
            model.fit(
                train_X,
                residual_target[train_mask],
                sample_weight=season_equal_weights(seasons[train_mask]),
            )
            correction = model.predict(validation_X).astype(float)
            name = f"ridge_a{int(alpha)}"
            fold[name] = {
                "fit_seconds": time.time() - fit_started,
                "correction_mean": float(correction.mean()),
                "coefficient_l2": float(np.linalg.norm(model.coef_)),
            }
            for weight in BLEND_WEIGHTS:
                candidate = f"{name}_w{int(weight * 100):03d}"
                predictions = np.clip(
                    group_reported[validation_season] + weight * correction,
                    0.0,
                    1.0,
                )
                fold[candidate] = calculate_metrics(targets, predictions)
                np.save(
                    ARTIFACT_ROOT
                    / f"predictions_{candidate}_{validation_season}.npy",
                    predictions,
                )
        np.save(ARTIFACT_ROOT / f"targets_{validation_season}.npy", targets)
        folds[str(validation_season)] = fold
        print(
            f"ridge {validation_season}: "
            + " ".join(
                f"a{int(alpha)}="
                f"{fold[f'ridge_a{int(alpha)}_w100']['skill_score_unclipped']:.2f}"
                for alpha in ALPHAS
            )
        )

    aggregate: dict[str, object] = {}
    for alpha in ALPHAS:
        for weight in BLEND_WEIGHTS:
            candidate = f"ridge_a{int(alpha)}_w{int(weight * 100):03d}"
            scores = [
                folds[str(season)][candidate]["skill_score_unclipped"]
                for season in REPORT_SEASONS
            ]
            aggregate[candidate] = {
                "mean_skill": float(np.mean(scores)),
                "min_skill": float(np.min(scores)),
                "latest_2024_skill": float(scores[-1]),
            }
    result = {
        "experiment": "EXP-019",
        "stage": "stable_linear_group_oof_residual",
        "validation_protocol": {
            "outer_folds": VALIDATION_SEASONS,
            "reported_folds": REPORT_SEASONS,
            "preprocessing_fit_on_training_fold_only": True,
            "current_fold_labels_used_for_training": False,
            "candidate_comparison_status": "diagnostic; nested selection required",
        },
        "features": feature_names,
        "alphas": ALPHAS,
        "blend_weights": BLEND_WEIGHTS,
        "reconstruction_diagnostics": reconstruction,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "total_seconds": time.time() - started,
    }
    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
