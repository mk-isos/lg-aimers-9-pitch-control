"""EXP-018: 계층적 기준값 + 안정적 그룹 효과 + 최근 시즌 residual.

최종 확률은 세 시간축을 결합한다.

1. 현재 행의 공식 as-of 값에서 복원한 현재 시즌 계층적 기준 확률
2. 과거 3시즌의 count × 투타 손 조합별 안정적 잔차 평균
   (투수 reverse rate 구간 조건부 효과를 30% 혼합)
3. 직전 1시즌으로 학습한 LightGBM residual의 15%만 반영

검증 시즌 정답은 모델, 그룹 효과, 보정값 어디에도 사용하지 않는다.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_exp017_rolling_residual import (
    calculate_metrics,
    fit_and_predict,
    prepare_data,
    segment_metrics,
)


ARTIFACT_DIR = Path("./artifacts/EXP-018/constrained_multiscale")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
GROUP_WINDOW = 3
GROUP_SMOOTHING = 100.0
REVERSE_GROUP_SMOOTHING = 300.0
REVERSE_GROUP_WEIGHT = 0.30
RECENT_WINDOW = 1
RECENT_WEIGHT = 0.15
RECENT_ITERATIONS = 200
NUM_LEAVES = 15
MIN_CHILD_SAMPLES = 2000


def centered_residual(
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
) -> np.ndarray:
    residual = (y - base).astype(np.float32, copy=True)
    for season in np.unique(seasons):
        mask = seasons == season
        residual[mask] -= residual[mask].mean()
    return residual


def build_group_keys(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    indices = {name: index for index, name in enumerate(feature_names)}
    reverse_rate = X[:, indices["asof_pitcher_reverse_rate"]]
    return pd.DataFrame(
        {
            "count_index": X[:, indices["count_index"]].astype(np.int8),
            "pitcher_hand": X[:, indices["pitcher_hand"]].astype(np.int8),
            "batter_hand": X[:, indices["batter_hand"]].astype(np.int8),
            "reverse_rate_bin": np.where(
                np.isfinite(reverse_rate),
                np.floor(reverse_rate / 0.05),
                -1,
            ).astype(np.int16),
        }
    )


def group_correction(
    group_keys: pd.DataFrame,
    residual: np.ndarray,
    seasons: np.ndarray,
    validation_season: int,
) -> tuple[np.ndarray, dict[str, int]]:
    train_mask = (
        (seasons < validation_season)
        & (seasons >= validation_season - GROUP_WINDOW)
    )
    validation_mask = seasons == validation_season
    base_columns = ["count_index", "pitcher_hand", "batter_hand"]
    reverse_columns = base_columns + ["reverse_rate_bin"]

    def map_effect(columns: list[str], smoothing: float) -> tuple[np.ndarray, int]:
        grouped = group_keys.loc[train_mask, columns].copy()
        grouped["residual"] = residual[train_mask]
        statistics = grouped.groupby(columns, sort=False)["residual"].agg(
            ["sum", "count"]
        )
        effects = statistics["sum"] / (statistics["count"] + smoothing)
        validation_keys = pd.MultiIndex.from_frame(
            group_keys.loc[validation_mask, columns]
        )
        mapped = effects.reindex(validation_keys).fillna(0.0).to_numpy()
        return mapped, int(len(effects))

    base_effect, base_groups = map_effect(base_columns, GROUP_SMOOTHING)
    reverse_effect, reverse_groups = map_effect(
        reverse_columns,
        REVERSE_GROUP_SMOOTHING,
    )
    correction = (
        (1.0 - REVERSE_GROUP_WEIGHT) * base_effect
        + REVERSE_GROUP_WEIGHT * reverse_effect
    )
    return correction.astype(np.float32), {
        "base_groups": base_groups,
        "reverse_groups": reverse_groups,
    }


def main() -> None:
    started_at = time.time()
    diagnostics, X, y, base, seasons, feature_names = prepare_data()
    residual = centered_residual(y, base, seasons)
    group_keys = build_group_keys(X, feature_names)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    folds: dict[str, object] = {}
    predictions_by_season: dict[int, np.ndarray] = {}
    targets_by_season: dict[int, np.ndarray] = {}
    previous_mean_gap: float | None = None
    for validation_season in VALIDATION_SEASONS:
        validation_mask = seasons == validation_season
        targets = y[validation_mask]
        correction, group_counts = group_correction(
            group_keys,
            residual,
            seasons,
            validation_season,
        )
        recent_predictions, fit_seconds, inference_seconds = fit_and_predict(
            X,
            y,
            base,
            seasons,
            validation_season=validation_season,
            objective="residual_centered",
            window=RECENT_WINDOW,
            best_iteration=RECENT_ITERATIONS,
            num_leaves=NUM_LEAVES,
            min_child_samples=MIN_CHILD_SAMPLES,
        )
        base_predictions = base[validation_mask]
        group_predictions = np.clip(
            base_predictions + correction,
            0.0,
            1.0,
        )
        final_predictions = np.clip(
            base_predictions
            + correction
            + RECENT_WEIGHT * (recent_predictions - base_predictions),
            0.0,
            1.0,
        )
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "training_seasons": sorted(
                np.unique(seasons[seasons < validation_season])
                .astype(int)
                .tolist()
            ),
            "group_training_seasons": list(
                range(validation_season - GROUP_WINDOW, validation_season)
            ),
            "recent_model_training_seasons": [validation_season - 1],
            "group_counts": group_counts,
            "recent_model_fit_seconds": fit_seconds,
            "recent_model_inference_seconds": inference_seconds,
            "hierarchical_base": calculate_metrics(targets, base_predictions),
            "base_plus_group": calculate_metrics(targets, group_predictions),
            "recent_model_full_weight_diagnostic": calculate_metrics(
                targets, recent_predictions
            ),
            "final": calculate_metrics(targets, final_predictions),
            "segments_final": segment_metrics(
                diagnostics,
                validation_mask,
                targets,
                final_predictions,
            ),
        }
        if previous_mean_gap is None:
            fold["prior_fold_mean_bias_calibration"] = {
                "applied": False,
                "reason": "no earlier evaluated fold",
                **calculate_metrics(targets, final_predictions),
            }
        else:
            bias_adjusted = np.clip(
                final_predictions - previous_mean_gap,
                0.0,
                1.0,
            )
            fold["prior_fold_mean_bias_calibration"] = {
                "applied": True,
                "subtracted_previous_fold_mean_gap": previous_mean_gap,
                **calculate_metrics(targets, bias_adjusted),
            }
        previous_mean_gap = float(
            final_predictions.mean() - targets.mean()
        )
        folds[str(validation_season)] = fold
        predictions_by_season[validation_season] = final_predictions
        targets_by_season[validation_season] = targets
        np.save(
            ARTIFACT_DIR / f"predictions_{validation_season}.npy",
            final_predictions,
        )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets.astype(np.int8),
        )
        print(
            f"EXP-018 {validation_season}: "
            f"base={fold['hierarchical_base']['skill_score_unclipped']:.2f} "
            f"group={fold['base_plus_group']['skill_score_unclipped']:.2f} "
            f"final={fold['final']['skill_score_unclipped']:.2f} "
            f"gap={fold['final']['mean_gap']:+.6f}"
        )

    raw_skills = [
        folds[str(season)]["final"]["skill_score_unclipped"]
        for season in REPORT_SEASONS
    ]
    calibrated_skills = [
        folds[str(season)]["prior_fold_mean_bias_calibration"][
            "skill_score_unclipped"
        ]
        for season in REPORT_SEASONS
    ]
    result: dict[str, object] = {
        "experiment": "EXP-018",
        "candidate": "constrained_multiscale",
        "validation_protocol": {
            "reported_seasons": REPORT_SEASONS,
            "warmup_diagnostic_season": 2021,
            "validation_labels_used_for_current_fold_training": False,
            "group_effect": "previous 3 seasons only",
            "recent_model": "previous 1 season only; fixed iterations",
            "probability_calibration": (
                "identity selected; prior-fold mean-bias correction reported "
                "but rejected because it did not improve all seasons"
            ),
        },
        "components": {
            "base": "row-independent current-season hierarchical as-of probability",
            "group": {
                "keys": ["count_index", "pitcher_hand", "batter_hand"],
                "window": GROUP_WINDOW,
                "smoothing": GROUP_SMOOTHING,
                "reverse_rate_bin_width": 0.05,
                "reverse_conditional_weight": REVERSE_GROUP_WEIGHT,
                "reverse_conditional_smoothing": REVERSE_GROUP_SMOOTHING,
            },
            "recent_residual_lightgbm": {
                "window": RECENT_WINDOW,
                "weight": RECENT_WEIGHT,
                "iterations": RECENT_ITERATIONS,
                "num_leaves": NUM_LEAVES,
                "min_child_samples": MIN_CHILD_SAMPLES,
                "learning_rate": 0.015,
                "target": "season-centered y minus hierarchical base",
            },
        },
        "folds": folds,
        "aggregate_2022_2024": {
            "mean_skill": float(np.mean(raw_skills)),
            "min_skill": float(np.min(raw_skills)),
            "prior_bias_calibrated_mean_skill": float(
                np.mean(calibrated_skills)
            ),
            "prior_bias_calibrated_min_skill": float(
                np.min(calibrated_skills)
            ),
        },
        "selection": {
            "selected_probability_calibration": "identity",
            "reason": (
                "raw candidate improves hierarchical base in every reported "
                "season; prior-fold mean correction lowers the 2023 minimum"
            ),
            "exp013_2024_reference": {
                "brier_score": 0.247862497,
                "skill_score": 778.37,
                "note": "EXP-013 affine was fitted on the same 2024 labels",
            },
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
        "total_seconds": time.time() - started_at,
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    with (ARTIFACT_DIR / "feature_names.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(feature_names, file, ensure_ascii=False, indent=2)
    print(f"saved={ARTIFACT_DIR / 'validation_metrics.json'}")


if __name__ == "__main__":
    main()
