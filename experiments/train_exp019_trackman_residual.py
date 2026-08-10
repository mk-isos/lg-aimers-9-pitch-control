"""EXP-019: 시점별 고신뢰 Trackman 매핑을 이용한 보완 residual.

매 validation season의 ID 매핑은 직전 시즌까지의 공식 데이터만 사용한다.
Trackman 값도 예측 시즌보다 이전 로그만 집계한다. 매핑된 투수에서만
physical/repertoire residual을 학습·적용하며 미매핑 행의 correction은 0이다.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import train_exp019_multirate_residual as multirate
from trackman_features import (
    attach_trackman_features,
    build_pitcher_mapping,
    build_prior_season_trackman_features,
)
from train_exp017_rolling_residual import calculate_metrics
from train_exp019_stable_monotonic import season_equal_weights


TRACKMAN_PATH = Path("./data/trackman_history.csv")
ARTIFACT_ROOT = Path("./artifacts/EXP-019/trackman_residual")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]


@dataclass(frozen=True)
class Config:
    name: str
    max_mapping_cost: float
    include_last_season: bool
    iterations: int
    num_leaves: int
    min_child_samples: int


CONFIGS = [
    Config("hist_c002_l7_i200", 0.02, False, 200, 7, 2000),
    Config("hist_c010_l7_i200", 0.10, False, 200, 7, 2000),
    Config("histlast_c002_l7_i200", 0.02, True, 200, 7, 2000),
    Config("histlast_c010_l7_i200", 0.10, True, 200, 7, 2000),
    Config("histlast_c010_l15_i200", 0.10, True, 200, 15, 2000),
]


def build_temporal_trackman_frame(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    seasons = sorted(main["season"].astype(int).unique().tolist())
    summaries = build_prior_season_trackman_features(trackman, seasons)
    feature_columns = [
        column
        for column in summaries.columns
        if column.startswith("tm_hist_") or column.startswith("tm_last_")
    ]
    output = pd.DataFrame(
        np.nan,
        index=main.index,
        columns=feature_columns,
        dtype=np.float32,
    )
    output["trackman_mapping_cost"] = np.nan
    output["has_trackman_mapping"] = np.int8(0)
    audit: dict[str, object] = {}

    for season in seasons:
        if season <= min(seasons):
            continue
        mapping_result = build_pitcher_mapping(
            main,
            trackman,
            cutoff_season=season - 1,
            max_cost=0.10,
        )
        mask = main["season"].to_numpy(dtype=int) == season
        indices = main.index[mask]
        mapped_ids = main.loc[indices, "pitcher_id"].map(mapping_result.mapping)
        season_summary = summaries.loc[summaries["season"] == season].set_index(
            "pitcher_trackman_id"
        )
        matched = season_summary.reindex(mapped_ids.to_numpy())
        matched.index = indices
        output.loc[indices, feature_columns] = matched[feature_columns].to_numpy(
            dtype=np.float32
        )
        costs = main.loc[indices, "pitcher_id"].map(mapping_result.costs)
        output.loc[indices, "trackman_mapping_cost"] = costs.to_numpy(dtype=float)
        output.loc[indices, "has_trackman_mapping"] = costs.notna().to_numpy(
            dtype=np.int8
        )
        audit[str(season)] = {
            "mapping_cutoff_season": season - 1,
            "mapped_pitchers": len(mapping_result.mapping),
            "candidate_main_ids": mapping_result.candidate_main_ids,
            "candidate_trackman_ids": mapping_result.candidate_trackman_ids,
            "row_coverage_cost_0_10": float(costs.notna().mean()),
            "row_coverage_cost_0_02": float((costs <= 0.02).mean()),
        }
    return output, audit


def temporal_group_predictions(
    frame: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    initial_residual = multirate.centered_residual(y, base, seasons)
    all_predictions = np.empty(len(y), dtype=np.float64)
    report_predictions: dict[int, np.ndarray] = {}
    for season in sorted(np.unique(seasons).astype(int).tolist()):
        mask = seasons == season
        correction = multirate.multirate_group_correction(
            frame, initial_residual, seasons, season
        )
        prediction = np.clip(base[mask].astype(float) + correction, 0.0, 1.0)
        all_predictions[mask] = prediction
        if season in VALIDATION_SEASONS:
            report_predictions[season] = prediction
    return all_predictions, report_predictions


def run_config(
    config: Config,
    trackman_frame: pd.DataFrame,
    y: np.ndarray,
    seasons: np.ndarray,
    residual_target: np.ndarray,
    group_predictions: dict[int, np.ndarray],
) -> dict[str, object]:
    feature_names = [
        column
        for column in trackman_frame.columns
        if column.startswith("tm_hist_")
        or (config.include_last_season and column.startswith("tm_last_"))
    ]
    X = trackman_frame[feature_names].to_numpy(dtype=np.float32)
    costs = trackman_frame["trackman_mapping_cost"].to_numpy(dtype=float)
    eligible = np.isfinite(costs) & (costs <= config.max_mapping_cost)
    artifact_dir = ARTIFACT_ROOT / config.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}

    for validation_season in VALIDATION_SEASONS:
        train_mask = (seasons < validation_season) & eligible
        validation_mask = seasons == validation_season
        validation_eligible = validation_mask & eligible
        model = LGBMRegressor(
            objective="regression_l2",
            metric="l2",
            n_estimators=config.iterations,
            learning_rate=0.015,
            num_leaves=config.num_leaves,
            min_child_samples=config.min_child_samples,
            max_bin=127,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=1.0,
            reg_lambda=12.0,
            random_state=42,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        started = time.time()
        model.fit(
            X[train_mask],
            residual_target[train_mask],
            sample_weight=season_equal_weights(seasons[train_mask]),
        )
        fit_seconds = time.time() - started
        correction = np.zeros(int(validation_mask.sum()), dtype=np.float64)
        local_eligible = eligible[validation_mask]
        correction[local_eligible] = model.predict(
            X[validation_eligible]
        ).astype(float)
        targets = y[validation_mask]
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "training_rows_mapped": int(train_mask.sum()),
            "validation_rows_mapped": int(validation_eligible.sum()),
            "validation_row_coverage": float(local_eligible.mean()),
            "fit_seconds": fit_seconds,
            "correction_mean_all_rows": float(correction.mean()),
            "feature_importance": {
                name: int(value)
                for name, value in sorted(
                    zip(feature_names, model.feature_importances_, strict=True),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
        }
        for weight in BLEND_WEIGHTS:
            candidate_name = f"group_plus_trackman_w{int(weight * 100):03d}"
            predictions = np.clip(
                group_predictions[validation_season] + weight * correction,
                0.0,
                1.0,
            )
            fold[candidate_name] = calculate_metrics(targets, predictions)
            np.save(
                artifact_dir
                / f"predictions_{candidate_name}_{validation_season}.npy",
                predictions,
            )
        np.save(artifact_dir / f"targets_{validation_season}.npy", targets)
        folds[str(validation_season)] = fold
        print(
            f"{config.name} {validation_season} coverage={local_eligible.mean():.3f}: "
            + " ".join(
                f"w{int(weight * 100):03d}="
                f"{fold[f'group_plus_trackman_w{int(weight * 100):03d}']['skill_score_unclipped']:.2f}"
                for weight in BLEND_WEIGHTS
            )
        )

    aggregate: dict[str, object] = {}
    for weight in BLEND_WEIGHTS:
        candidate_name = f"group_plus_trackman_w{int(weight * 100):03d}"
        scores = [
            folds[str(season)][candidate_name]["skill_score_unclipped"]
            for season in REPORT_SEASONS
        ]
        aggregate[candidate_name] = {
            "mean_skill": float(np.mean(scores)),
            "min_skill": float(np.min(scores)),
            "latest_2024_skill": float(scores[-1]),
        }
    result = {
        "experiment": "EXP-019",
        "candidate": config.name,
        "validation_protocol": {
            "outer_folds": VALIDATION_SEASONS,
            "mapping": "each prediction season uses main data through season-1",
            "trackman_aggregation": "Trackman rows strictly before prediction season",
            "current_fold_labels_used_for_training": False,
            "candidate_comparison_status": "diagnostic; nested selection required",
        },
        "model": {
            "max_mapping_cost": config.max_mapping_cost,
            "include_last_season": config.include_last_season,
            "features": feature_names,
            "iterations": config.iterations,
            "num_leaves": config.num_leaves,
            "min_child_samples": config.min_child_samples,
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    with (artifact_dir / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    started = time.time()
    frame, _, y, base, seasons, _ = multirate.prepare_multirate_data()
    trackman = pd.read_csv(TRACKMAN_PATH, encoding="utf-8-sig")
    trackman_frame, mapping_audit = build_temporal_trackman_frame(frame, trackman)
    group_all, group_predictions = temporal_group_predictions(
        frame, y, base, seasons
    )
    residual_target = (y.astype(float) - group_all).astype(np.float32)
    for season in np.unique(seasons):
        mask = seasons == season
        residual_target[mask] -= residual_target[mask].mean()

    summaries: dict[str, object] = {}
    for config in CONFIGS:
        result = run_config(
            config,
            trackman_frame,
            y,
            seasons,
            residual_target,
            group_predictions,
        )
        summaries[config.name] = result["aggregate_2022_2024"]
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "experiment": "EXP-019",
                "stage": "temporal_trackman_residual_candidate_search",
                "selection_status": "not selected; nested selection required",
                "mapping_audit": mapping_audit,
                "summaries": summaries,
                "total_seconds": time.time() - started,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
