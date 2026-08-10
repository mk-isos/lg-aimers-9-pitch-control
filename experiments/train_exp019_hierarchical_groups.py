"""EXP-019: 계층적 empirical-Bayes residual group 후보.

coarse count×투타 손 효과를 prior로 두고, 현재 행의 context·현재 시즌
success/reverse·최근 성공률 구간별 fine effect를 과거 시즌에서만 추정한다.
테스트 데이터 내부 집계는 사용하지 않는다.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics, segment_metrics
from train_exp018_constrained_multiscale import centered_residual
from train_exp019_multirate_residual import prepare_multirate_data


ARTIFACT_ROOT = Path("./artifacts/EXP-019/hierarchical_groups")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
WINDOW = 3
COARSE_SMOOTHING = 100.0

COARSE_KEYS = ["count_index", "pitcher_hand", "batter_hand"]
FINE_SPECS = {
    "context": {
        "keys": COARSE_KEYS
        + ["outs_before", "num_runners_on", "inning_bin"],
        "smoothing": 500.0,
    },
    "base_state": {
        "keys": COARSE_KEYS + ["outs_before", "base_state_code"],
        "smoothing": 400.0,
    },
    "multirate": {
        "keys": COARSE_KEYS + ["season_success_bin", "season_reverse_bin"],
        "smoothing": 500.0,
    },
    "recent": {
        "keys": COARSE_KEYS + ["recent3_success_bin", "season_n_bin"],
        "smoothing": 400.0,
    },
}

CANDIDATE_WEIGHTS = {
    "coarse": {"coarse": 1.0},
    "context": {"context": 1.0},
    "base_state": {"base_state": 1.0},
    "multirate": {"multirate": 1.0},
    "recent": {"recent": 1.0},
    "context_multirate": {"context": 0.5, "multirate": 0.5},
    "state_multirate": {"base_state": 0.5, "multirate": 0.5},
    "context_multirate_recent": {
        "context": 0.4,
        "multirate": 0.4,
        "recent": 0.2,
    },
    "all_fine": {
        "context": 0.3,
        "base_state": 0.2,
        "multirate": 0.3,
        "recent": 0.2,
    },
}


def build_keys(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[
        [
            "count_index",
            "pitcher_hand",
            "batter_hand",
            "outs_before",
            "num_runners_on",
        ]
    ].copy()
    keys["inning_bin"] = np.select(
        [frame["inning"] <= 3, frame["inning"] <= 6], [0, 1], default=2
    ).astype(np.int8)
    base_state_values = sorted(frame["base_state"].astype(str).unique().tolist())
    base_state_mapping = {value: index for index, value in enumerate(base_state_values)}
    keys["base_state_code"] = (
        frame["base_state"].astype(str).map(base_state_mapping).astype(np.int8)
    )
    keys["season_success_bin"] = np.floor(
        frame[
            "multirate_pitcher_control_success_season_global_30"
        ].to_numpy(dtype=float)
        / 0.025
    ).astype(np.int16)
    keys["season_reverse_bin"] = np.floor(
        frame[
            "multirate_pitcher_control_reverse_season_global_30"
        ].to_numpy(dtype=float)
        / 0.025
    ).astype(np.int16)
    keys["recent3_success_bin"] = np.floor(
        frame["asof_pitcher_prev3_game_success_rate"]
        .fillna(0.5)
        .to_numpy(dtype=float)
        / 0.05
    ).astype(np.int16)
    season_n = frame["temporal_pitcher_season_n"].to_numpy(dtype=float)
    keys["season_n_bin"] = np.select(
        [season_n == 0, season_n < 20, season_n < 100, season_n < 500],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int8)
    return keys


def fit_effect(
    keys: pd.DataFrame,
    residual: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    columns: list[str],
    smoothing: float,
    parent_effects: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series]:
    grouped = keys.loc[train_mask, columns].copy()
    grouped["residual"] = residual[train_mask]
    statistics = grouped.groupby(columns, sort=False)["residual"].agg(
        ["sum", "count"]
    )
    if parent_effects is None:
        prior = np.zeros(len(statistics), dtype=float)
    else:
        parent_index = pd.MultiIndex.from_frame(
            statistics.reset_index()[COARSE_KEYS]
        )
        prior = parent_effects.reindex(parent_index).fillna(0.0).to_numpy()
    effects = pd.Series(
        (statistics["sum"].to_numpy() + smoothing * prior)
        / (statistics["count"].to_numpy() + smoothing),
        index=statistics.index,
    )
    validation_index = pd.MultiIndex.from_frame(keys.loc[validation_mask, columns])
    mapped = effects.reindex(validation_index).to_numpy(dtype=float)
    if parent_effects is None:
        return np.nan_to_num(mapped, nan=0.0), effects
    parent_validation_index = pd.MultiIndex.from_frame(
        keys.loc[validation_mask, COARSE_KEYS]
    )
    parent_fallback = (
        parent_effects.reindex(parent_validation_index).fillna(0.0).to_numpy()
    )
    return np.where(np.isfinite(mapped), mapped, parent_fallback), effects


def game_type_metrics(
    diagnostics: pd.DataFrame,
    validation_mask: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    values = diagnostics.loc[validation_mask, "game_type"].astype(str).to_numpy()
    return {
        value: calculate_metrics(targets[values == value], predictions[values == value])
        for value in sorted(np.unique(values))
    }


def main() -> None:
    started = time.time()
    frame, diagnostics, y, base, seasons, reconstruction = prepare_multirate_data()
    keys = build_keys(frame)
    residual = centered_residual(y, base, seasons)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}

    for validation_season in VALIDATION_SEASONS:
        train_mask = (
            (seasons < validation_season)
            & (seasons >= validation_season - WINDOW)
        )
        validation_mask = seasons == validation_season
        targets = y[validation_mask]
        coarse, coarse_series = fit_effect(
            keys,
            residual,
            train_mask,
            validation_mask,
            COARSE_KEYS,
            COARSE_SMOOTHING,
        )
        corrections = {"coarse": coarse}
        group_counts = {"coarse": int(len(coarse_series))}
        for name, spec in FINE_SPECS.items():
            correction, series = fit_effect(
                keys,
                residual,
                train_mask,
                validation_mask,
                spec["keys"],
                float(spec["smoothing"]),
                coarse_series,
            )
            corrections[name] = correction
            group_counts[name] = int(len(series))

        fold: dict[str, object] = {
            "validation_season": validation_season,
            "training_seasons": sorted(
                np.unique(seasons[train_mask]).astype(int).tolist()
            ),
            "group_counts": group_counts,
        }
        for candidate, weights in CANDIDATE_WEIGHTS.items():
            correction = sum(
                weight * corrections[name] for name, weight in weights.items()
            )
            predictions = np.clip(
                base[validation_mask].astype(float) + correction, 0.0, 1.0
            )
            fold[candidate] = calculate_metrics(targets, predictions)
            fold[f"segments_{candidate}"] = segment_metrics(
                diagnostics, validation_mask, targets, predictions
            )
            fold[f"game_type_{candidate}"] = game_type_metrics(
                diagnostics, validation_mask, targets, predictions
            )
            np.save(
                ARTIFACT_ROOT / f"predictions_{candidate}_{validation_season}.npy",
                predictions,
            )
        np.save(ARTIFACT_ROOT / f"targets_{validation_season}.npy", targets)
        folds[str(validation_season)] = fold
        print(
            f"hierarchical {validation_season}: "
            + " ".join(
                f"{candidate}={fold[candidate]['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATE_WEIGHTS
            )
        )

    aggregate: dict[str, object] = {}
    for candidate in CANDIDATE_WEIGHTS:
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
        "stage": "hierarchical_empirical_bayes_groups",
        "validation_protocol": {
            "outer_folds": VALIDATION_SEASONS,
            "reported_folds": REPORT_SEASONS,
            "group_window": WINDOW,
            "current_fold_labels_used_for_training": False,
            "test_row_aggregation": False,
            "candidate_comparison_status": "diagnostic; nested selection required",
        },
        "coarse": {"keys": COARSE_KEYS, "smoothing": COARSE_SMOOTHING},
        "fine": FINE_SPECS,
        "candidate_weights": CANDIDATE_WEIGHTS,
        "reconstruction_diagnostics": reconstruction,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "total_seconds": time.time() - started,
    }
    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
