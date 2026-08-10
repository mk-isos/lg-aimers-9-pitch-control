"""EXP-019: season-equal hierarchical player-context OOF residual effects.

The official row-level as-of values are first converted into the same temporal
hierarchical base used by EXP-018.  The stable count/hand group correction for
every season is then generated strictly out of earlier seasons.  Player effects
therefore learn only the residual left by an out-of-fold base/group prediction.

For each validation season, every player component is estimated independently
inside each earlier season and the season-specific maps are averaged with equal
season weight.  Missing players/groups contribute zero for that season, which
also shrinks effects that do not persist.  No validation-season label and no
other evaluation row is used to construct a feature or effect.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_residual_features import (
    TARGET,
    attach_training_temporal_features,
)
from train_exp017_rolling_residual import calculate_metrics, segment_metrics


DATA_DIR = Path("./data")
ARTIFACT_DIR = Path("./artifacts/EXP-019/player_context_residual")
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
LATEST_SEASONS = (2023, 2024)
GROUP_WINDOW = 3
GROUP_SMOOTHING = 100.0
REVERSE_GROUP_SMOOTHING = 300.0
REVERSE_GROUP_WEIGHT = 0.30


@dataclass(frozen=True)
class Candidate:
    """Predetermined strong-shrinkage hierarchy configuration."""

    window: int | None
    pitcher_smoothing: float
    pitcher_context_smoothing: float
    batter_smoothing: float
    batter_context_smoothing: float


CANDIDATES: dict[str, Candidate | None] = {
    "base_group_only": None,
    "hier_strong_3y": Candidate(
        window=3,
        pitcher_smoothing=800.0,
        pitcher_context_smoothing=250.0,
        batter_smoothing=1000.0,
        batter_context_smoothing=300.0,
    ),
    "hier_stronger_3y": Candidate(
        window=3,
        pitcher_smoothing=1500.0,
        pitcher_context_smoothing=500.0,
        batter_smoothing=1800.0,
        batter_context_smoothing=600.0,
    ),
    "hier_pitcher_focus_3y": Candidate(
        window=3,
        pitcher_smoothing=600.0,
        pitcher_context_smoothing=200.0,
        batter_smoothing=1800.0,
        batter_context_smoothing=600.0,
    ),
    "hier_strong_2y": Candidate(
        window=2,
        pitcher_smoothing=800.0,
        pitcher_context_smoothing=250.0,
        batter_smoothing=1000.0,
        batter_context_smoothing=300.0,
    ),
    "hier_persistent_all": Candidate(
        window=None,
        pitcher_smoothing=1000.0,
        pitcher_context_smoothing=350.0,
        batter_smoothing=1400.0,
        batter_context_smoothing=450.0,
    ),
}


COMPONENTS = (
    ("pitcher", ("pitcher_id",), "pitcher_smoothing"),
    (
        "pitcher_context",
        ("pitcher_id", "count_index", "batter_hand"),
        "pitcher_context_smoothing",
    ),
    ("batter", ("batter_id",), "batter_smoothing"),
    (
        "batter_context",
        ("batter_id", "count_index", "pitcher_hand"),
        "batter_context_smoothing",
    ),
)


def load_temporal_data() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        TARGET,
    ]
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=columns,
    )
    train, _ = attach_training_temporal_features(train, target=TARGET)
    train["count_index"] = (
        train["balls_before"] * 4 + train["strikes_before"]
    ).astype("int8")
    return train


def centered_by_season(values: np.ndarray, seasons: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float32, copy=True)
    for season in np.unique(seasons):
        mask = seasons == season
        centered[mask] -= centered[mask].mean()
    return centered


def build_group_keys(train: pd.DataFrame) -> pd.DataFrame:
    reverse_rate = train["asof_pitcher_reverse_rate"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "count_index": train["count_index"].to_numpy(dtype=np.int8),
            "pitcher_hand": train["pitcher_hand"].to_numpy(dtype=np.int8),
            "batter_hand": train["batter_hand"].to_numpy(dtype=np.int8),
            "reverse_rate_bin": np.where(
                np.isfinite(reverse_rate),
                np.floor(reverse_rate / 0.05),
                -1,
            ).astype(np.int16),
        }
    )


def map_smoothed_effect(
    training_keys: pd.DataFrame,
    training_values: np.ndarray,
    validation_keys: pd.DataFrame,
    columns: tuple[str, ...],
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    grouped = training_keys.loc[:, list(columns)].copy()
    grouped["value"] = training_values
    statistics = grouped.groupby(list(columns), sort=False)["value"].agg(
        ["sum", "count"]
    )
    effects = statistics["sum"] / (statistics["count"] + smoothing)
    validation_index = pd.MultiIndex.from_frame(
        validation_keys.loc[:, list(columns)]
    )
    mapped = effects.reindex(validation_index)
    matched = mapped.notna().to_numpy()
    return (
        mapped.fillna(0.0).to_numpy(dtype=np.float32),
        matched,
        int(len(effects)),
    )


def group_oof_predictions(
    train: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate EXP-018 group predictions with each season held out in time."""
    group_keys = build_group_keys(train)
    residual = centered_by_season(y - base, seasons)
    predictions = base.astype(np.float32, copy=True)
    metadata: dict[str, object] = {}
    base_columns = ("count_index", "pitcher_hand", "batter_hand")
    reverse_columns = base_columns + ("reverse_rate_bin",)
    for validation_season in np.unique(seasons):
        validation_mask = seasons == validation_season
        training_mask = (
            (seasons < validation_season)
            & (seasons >= validation_season - GROUP_WINDOW)
        )
        if not training_mask.any():
            metadata[str(int(validation_season))] = {
                "training_seasons": [],
                "base_groups": 0,
                "reverse_groups": 0,
            }
            continue
        base_effect, _, base_groups = map_smoothed_effect(
            group_keys.loc[training_mask],
            residual[training_mask],
            group_keys.loc[validation_mask],
            base_columns,
            GROUP_SMOOTHING,
        )
        reverse_effect, _, reverse_groups = map_smoothed_effect(
            group_keys.loc[training_mask],
            residual[training_mask],
            group_keys.loc[validation_mask],
            reverse_columns,
            REVERSE_GROUP_SMOOTHING,
        )
        correction = (
            (1.0 - REVERSE_GROUP_WEIGHT) * base_effect
            + REVERSE_GROUP_WEIGHT * reverse_effect
        )
        predictions[validation_mask] = np.clip(
            base[validation_mask] + correction,
            0.0,
            1.0,
        )
        metadata[str(int(validation_season))] = {
            "training_seasons": sorted(
                np.unique(seasons[training_mask]).astype(int).tolist()
            ),
            "base_groups": base_groups,
            "reverse_groups": reverse_groups,
        }
    return predictions, metadata


def candidate_effect(
    train: pd.DataFrame,
    residual_target: np.ndarray,
    seasons: np.ndarray,
    validation_season: int,
    candidate: Candidate,
) -> tuple[np.ndarray, dict[str, object]]:
    validation_mask = seasons == validation_season
    validation_rows = train.loc[validation_mask]
    training_seasons = sorted(
        np.unique(seasons[seasons < validation_season]).astype(int).tolist()
    )
    if candidate.window is not None:
        training_seasons = [
            season
            for season in training_seasons
            if season >= validation_season - candidate.window
        ]
    if not training_seasons:
        return np.zeros(int(validation_mask.sum()), dtype=np.float32), {
            "training_seasons": [],
            "components": {},
        }

    total_effect = np.zeros(int(validation_mask.sum()), dtype=np.float32)
    matched_counts = {
        name: np.zeros(int(validation_mask.sum()), dtype=np.int16)
        for name, _, _ in COMPONENTS
    }
    group_counts = {name: 0 for name, _, _ in COMPONENTS}

    for training_season in training_seasons:
        season_mask = seasons == training_season
        season_rows = train.loc[season_mask]
        working_residual = residual_target[season_mask].astype(
            np.float32, copy=True
        )
        for name, columns, smoothing_name in COMPONENTS:
            smoothing = float(getattr(candidate, smoothing_name))
            mapped_validation, matched, number_of_groups = map_smoothed_effect(
                season_rows,
                working_residual,
                validation_rows,
                columns,
                smoothing,
            )
            mapped_training, _, _ = map_smoothed_effect(
                season_rows,
                working_residual,
                season_rows,
                columns,
                smoothing,
            )
            total_effect += mapped_validation / len(training_seasons)
            matched_counts[name] += matched.astype(np.int16)
            group_counts[name] += number_of_groups
            working_residual -= mapped_training

    component_metadata: dict[str, object] = {}
    for name, _, _ in COMPONENTS:
        counts = matched_counts[name]
        component_metadata[name] = {
            "total_season_groups": int(group_counts[name]),
            "validation_coverage_any": float((counts > 0).mean()),
            "mean_training_seasons_matched": float(counts.mean()),
        }
    return total_effect, {
        "training_seasons": training_seasons,
        "components": component_metadata,
        "correction_mean": float(total_effect.mean()),
        "correction_std": float(total_effect.std()),
        "correction_min": float(total_effect.min()),
        "correction_max": float(total_effect.max()),
    }


def aggregate_candidate(
    folds: dict[str, object],
    candidate_name: str,
) -> dict[str, object]:
    skills = {
        season: float(
            folds[str(season)]["candidates"][candidate_name]["metrics"][
                "skill_score_unclipped"
            ]
        )
        for season in REPORT_SEASONS
    }
    briers = {
        season: float(
            folds[str(season)]["candidates"][candidate_name]["metrics"][
                "brier_score"
            ]
        )
        for season in REPORT_SEASONS
    }
    latest_skills = [skills[season] for season in LATEST_SEASONS]
    return {
        "season_skills": {str(key): value for key, value in skills.items()},
        "season_briers": {str(key): value for key, value in briers.items()},
        "mean_skill": float(np.mean(list(skills.values()))),
        "min_skill": float(np.min(list(skills.values()))),
        "latest_mean_skill": float(np.mean(latest_skills)),
        "latest_min_skill": float(np.min(latest_skills)),
    }


def select_candidate(
    aggregate: dict[str, dict[str, object]],
    primary_metric: str,
) -> str:
    return max(
        aggregate,
        key=lambda name: (
            float(aggregate[name][primary_metric]),
            float(aggregate[name]["min_skill"]),
            float(aggregate[name]["latest_mean_skill"]),
            float(aggregate[name]["mean_skill"]),
        ),
    )


def main() -> None:
    started_at = time.time()
    train = load_temporal_data()
    seasons = train["season"].to_numpy(dtype=np.int16)
    y = train[TARGET].to_numpy(dtype=np.float32)
    base = train["temporal_base_global_30"].to_numpy(dtype=np.float32)
    base_group_oof, group_metadata = group_oof_predictions(
        train, y, base, seasons
    )
    residual_target = centered_by_season(y - base_group_oof, seasons)
    diagnostics = train[
        [
            "season",
            "temporal_pitcher_season_n",
            "temporal_pitcher_prior_exists",
            "temporal_batter_prior_exists",
        ]
    ].copy()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        validation_mask = seasons == validation_season
        targets = y[validation_mask]
        base_group_predictions = base_group_oof[validation_mask]
        fold_candidates: dict[str, object] = {}
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets)
        np.save(
            ARTIFACT_DIR / f"predictions_base_group_{validation_season}.npy",
            base_group_predictions,
        )
        np.save(
            ARTIFACT_DIR / f"oof_residual_targets_{validation_season}.npy",
            residual_target[validation_mask],
        )
        for candidate_name, candidate in CANDIDATES.items():
            if candidate is None:
                correction = np.zeros(len(targets), dtype=np.float32)
                effect_metadata: dict[str, object] = {
                    "training_seasons": [],
                    "components": {},
                    "correction_mean": 0.0,
                    "correction_std": 0.0,
                    "correction_min": 0.0,
                    "correction_max": 0.0,
                }
            else:
                correction, effect_metadata = candidate_effect(
                    train,
                    residual_target,
                    seasons,
                    validation_season,
                    candidate,
                )
            predictions = np.clip(
                base_group_predictions + correction,
                0.0,
                1.0,
            )
            metrics = calculate_metrics(targets, predictions)
            fold_candidates[candidate_name] = {
                "metrics": metrics,
                "effect": effect_metadata,
                "segments": segment_metrics(
                    diagnostics,
                    validation_mask,
                    targets,
                    predictions,
                ),
            }
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate_name}_{validation_season}.npy",
                predictions,
            )
            print(
                f"EXP-019 {candidate_name} {validation_season}: "
                f"brier={metrics['brier_score']:.12f} "
                f"skill={metrics['skill_score_unclipped']:.2f} "
                f"gap={metrics['mean_gap']:+.6f}"
            )
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "rows": int(validation_mask.sum()),
            "base_group_oof": calculate_metrics(
                targets, base_group_predictions
            ),
            "group_oof": group_metadata[str(validation_season)],
            "candidates": fold_candidates,
        }

    aggregate = {
        candidate_name: aggregate_candidate(folds, candidate_name)
        for candidate_name in CANDIDATES
    }
    best_min = select_candidate(aggregate, "min_skill")
    best_latest = select_candidate(aggregate, "latest_min_skill")
    result: dict[str, object] = {
        "experiment": "EXP-019",
        "candidate_family": "season_equal_hierarchical_player_context_residual",
        "validation_protocol": {
            "validation_seasons": list(VALIDATION_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "latest_seasons": list(LATEST_SEASONS),
            "base_group_residual_target": (
                "each season predicted from earlier seasons only; residual "
                "centered independently inside each training season"
            ),
            "player_effect_training": (
                "past seasons only; season-specific strongly shrunk maps "
                "averaged with equal season weight; absent groups contribute zero"
            ),
            "validation_labels_used_for_current_fold_training": False,
            "test_row_independence": (
                "stored train effects plus current-row official IDs/count/hands only"
            ),
        },
        "predetermined_grid": {
            name: None if config is None else asdict(config)
            for name, config in CANDIDATES.items()
        },
        "component_order": [name for name, _, _ in COMPONENTS],
        "component_keys": {
            name: list(columns) for name, columns, _ in COMPONENTS
        },
        "base_group": {
            "base": "temporal_base_global_30",
            "group_window": GROUP_WINDOW,
            "group_smoothing": GROUP_SMOOTHING,
            "reverse_group_smoothing": REVERSE_GROUP_SMOOTHING,
            "reverse_group_weight": REVERSE_GROUP_WEIGHT,
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_min_2022_2024": best_min,
            "best_latest_min_2023_2024": best_latest,
            "selection_metrics_are_diagnostic": (
                "the fixed grid is reported in full; no candidate is packaged "
                "without a separate temporal selection decision"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "total_seconds": time.time() - started_at,
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f"best_min={best_min} {aggregate[best_min]}")
    print(f"best_latest={best_latest} {aggregate[best_latest]}")
    print(f"saved={ARTIFACT_DIR / 'validation_metrics.json'}")


if __name__ == "__main__":
    main()
