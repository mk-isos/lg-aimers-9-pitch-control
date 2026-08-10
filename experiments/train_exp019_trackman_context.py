"""EXP-019: 투수×카운트×타자 손 Trackman context residual.

기존 Trackman 후보는 투수별 상수 요약만 사용했다. 이 실험은 예측 시즌보다
이전 로그에서 현재 count와 batter hand에 대응하는 투수의 repertoire/physical
요약을 만들고, 투수 전체 평균으로 shrink한다. ID 매핑도 season-1까지만 쓴다.
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
    PITCH_GROUPS,
    TRACKMAN_NUMERIC_COLUMNS,
    build_pitcher_mapping,
)
from train_exp017_rolling_residual import calculate_metrics
from train_exp019_r_full_residual import original_group_correction
from train_exp019_stable_monotonic import season_equal_weights


TRACKMAN_PATH = Path("./data/trackman_history.csv")
ARTIFACT_ROOT = Path("./artifacts/EXP-019/trackman_context")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
CONTEXT_SMOOTHING = 50.0


@dataclass(frozen=True)
class Config:
    name: str
    max_mapping_cost: float
    iterations: int
    num_leaves: int
    min_child_samples: int


CONFIGS = [
    Config("ctx_c002_l7_i200", 0.02, 200, 7, 2000),
    Config("ctx_c010_l7_i200", 0.10, 200, 7, 2000),
    Config("ctx_c010_l15_i300", 0.10, 300, 15, 2000),
]


def aggregate_context(history: pd.DataFrame) -> pd.DataFrame:
    work = history.copy()
    work["count_index"] = (
        work["balls_before"] * 4 + work["strikes_before"]
    ).astype(np.int8)
    group_columns = ["pitcher_trackman_id", "count_index", "batter_hand"]
    grouped = work.groupby(group_columns, sort=False)
    overall = work.groupby("pitcher_trackman_id", sort=False)
    output = grouped.size().rename("tm_ctx_n").to_frame()
    counts = output["tm_ctx_n"].to_numpy(dtype=float)

    pitcher_ids = output.index.get_level_values("pitcher_trackman_id")
    for column in TRACKMAN_NUMERIC_COLUMNS:
        context_mean = grouped[column].mean()
        overall_mean = overall[column].mean().reindex(pitcher_ids).to_numpy(dtype=float)
        values = (
            counts * context_mean.to_numpy(dtype=float)
            + CONTEXT_SMOOTHING * overall_mean
        ) / (counts + CONTEXT_SMOOTHING)
        output[f"tm_ctx_{column}_shrunk"] = values.astype(np.float32)

    pitch_counts = pd.crosstab(
        [
            work["pitcher_trackman_id"],
            work["count_index"],
            work["batter_hand"],
        ],
        work["pitch_type_group"],
    ).reindex(index=output.index, columns=PITCH_GROUPS, fill_value=0)
    overall_pitch_counts = pd.crosstab(
        work["pitcher_trackman_id"], work["pitch_type_group"]
    ).reindex(columns=PITCH_GROUPS, fill_value=0)
    overall_rates = overall_pitch_counts.div(
        overall_pitch_counts.sum(axis=1).replace(0, np.nan), axis=0
    )
    for pitch_group in PITCH_GROUPS:
        prior = overall_rates[pitch_group].reindex(pitcher_ids).to_numpy(dtype=float)
        values = (
            pitch_counts[pitch_group].to_numpy(dtype=float)
            + CONTEXT_SMOOTHING * prior
        ) / (counts + CONTEXT_SMOOTHING)
        output[f"tm_ctx_{pitch_group}_rate_shrunk"] = values.astype(np.float32)
    output["tm_ctx_log_n"] = np.log1p(counts).astype(np.float32)
    return output


def build_context_frame(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_names = [
        *[f"tm_ctx_{column}_shrunk" for column in TRACKMAN_NUMERIC_COLUMNS],
        *[f"tm_ctx_{group}_rate_shrunk" for group in PITCH_GROUPS],
        "tm_ctx_log_n",
    ]
    output = pd.DataFrame(
        np.nan, index=main.index, columns=feature_names, dtype=np.float32
    )
    output["trackman_mapping_cost"] = np.nan
    audit: dict[str, object] = {}
    main_batter_hand = main["batter_hand"].map({1: "Left", 2: "Right"})
    seasons = sorted(main["season"].astype(int).unique().tolist())

    for season in seasons:
        if season <= min(seasons):
            continue
        historical = trackman.loc[trackman["season"] < season]
        summary = aggregate_context(historical)
        mapping = build_pitcher_mapping(
            main,
            trackman,
            cutoff_season=season - 1,
            max_cost=0.10,
        )
        mask = main["season"].to_numpy(dtype=int) == season
        indices = main.index[mask]
        mapped_pitchers = main.loc[indices, "pitcher_id"].map(mapping.mapping)
        query_index = pd.MultiIndex.from_arrays(
            [
                mapped_pitchers.to_numpy(),
                main.loc[indices, "count_index"].to_numpy(dtype=np.int8),
                main_batter_hand.loc[indices].to_numpy(),
            ],
            names=["pitcher_trackman_id", "count_index", "batter_hand"],
        )
        matched = summary.reindex(query_index)
        output.loc[indices, feature_names] = matched[feature_names].to_numpy(
            dtype=np.float32
        )
        costs = main.loc[indices, "pitcher_id"].map(mapping.costs)
        output.loc[indices, "trackman_mapping_cost"] = costs.to_numpy(dtype=float)
        audit[str(season)] = {
            "mapping_cutoff": season - 1,
            "mapped_pitchers": len(mapping.mapping),
            "row_coverage_cost_0_10": float(costs.notna().mean()),
            "context_feature_coverage": float(matched["tm_ctx_log_n"].notna().mean()),
        }
    return output, audit


def main() -> None:
    started = time.time()
    frame, _, y, base, seasons, reconstruction = multirate.prepare_multirate_data()
    trackman = pd.read_csv(TRACKMAN_PATH, encoding="utf-8-sig")
    context, mapping_audit = build_context_frame(frame, trackman)
    feature_names = [column for column in context if column.startswith("tm_ctx_")]
    X = context[feature_names].to_numpy(dtype=np.float32)
    costs = context["trackman_mapping_cost"].to_numpy(dtype=float)

    initial_residual = multirate.centered_residual(y, base, seasons)
    group_all = np.empty(len(y), dtype=np.float64)
    group_reported: dict[int, np.ndarray] = {}
    for season in sorted(np.unique(seasons).astype(int).tolist()):
        mask = seasons == season
        correction = original_group_correction(frame, initial_residual, seasons, season)
        prediction = np.clip(base[mask].astype(float) + correction, 0.0, 1.0)
        group_all[mask] = prediction
        if season in VALIDATION_SEASONS:
            group_reported[season] = prediction
    residual_target = (y.astype(float) - group_all).astype(np.float32)
    for season in np.unique(seasons):
        mask = seasons == season
        residual_target[mask] -= residual_target[mask].mean()

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    for config in CONFIGS:
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
            fit_started = time.time()
            model.fit(
                X[train_mask],
                residual_target[train_mask],
                sample_weight=season_equal_weights(seasons[train_mask]),
            )
            correction = np.zeros(int(validation_mask.sum()), dtype=float)
            local_eligible = eligible[validation_mask]
            correction[local_eligible] = model.predict(
                X[validation_eligible]
            ).astype(float)
            targets = y[validation_mask]
            fold: dict[str, object] = {
                "validation_season": validation_season,
                "fit_seconds": time.time() - fit_started,
                "validation_coverage": float(local_eligible.mean()),
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
                candidate = f"group_plus_tmctx_w{int(weight * 100):03d}"
                predictions = np.clip(
                    group_reported[validation_season] + weight * correction,
                    0.0,
                    1.0,
                )
                fold[candidate] = calculate_metrics(targets, predictions)
                np.save(
                    artifact_dir
                    / f"predictions_{candidate}_{validation_season}.npy",
                    predictions,
                )
            np.save(artifact_dir / f"targets_{validation_season}.npy", targets)
            folds[str(validation_season)] = fold
            print(
                f"{config.name} {validation_season}: "
                + " ".join(
                    f"w{int(weight * 100):03d}="
                    f"{fold[f'group_plus_tmctx_w{int(weight * 100):03d}']['skill_score_unclipped']:.2f}"
                    for weight in BLEND_WEIGHTS
                )
            )

        aggregate: dict[str, object] = {}
        for weight in BLEND_WEIGHTS:
            candidate = f"group_plus_tmctx_w{int(weight * 100):03d}"
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
            "candidate": config.name,
            "validation_protocol": {
                "outer_folds": VALIDATION_SEASONS,
                "reported_folds": REPORT_SEASONS,
                "mapping": "main data through prediction season-1",
                "trackman": "logs strictly before prediction season",
                "context": "pitcher x count x batter hand, shrunk to pitcher mean",
                "current_fold_labels_used_for_training": False,
                "test_row_aggregation": False,
                "candidate_comparison_status": "diagnostic; nested selection required",
            },
            "model": {
                "features": feature_names,
                "max_mapping_cost": config.max_mapping_cost,
                "iterations": config.iterations,
                "num_leaves": config.num_leaves,
                "min_child_samples": config.min_child_samples,
            },
            "mapping_audit": mapping_audit,
            "reconstruction_diagnostics": reconstruction,
            "folds": folds,
            "aggregate_2022_2024": aggregate,
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
        summaries[config.name] = aggregate

    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "experiment": "EXP-019",
                "stage": "trackman_context_residual",
                "selection_status": "not selected; nested selection required",
                "summaries": summaries,
                "total_seconds": time.time() - started,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
