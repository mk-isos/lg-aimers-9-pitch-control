"""EXP-019: strongly regularized HistGradientBoosting residual comparison.

This experiment keeps the temporally safe hierarchical base plus the original
three-season count/hand group correction from EXP-018.  A shallow
HistGradientBoosting regressor learns only the remaining season-centered
residual on past regular-season (``game_type == "R"``) rows.  Non-regular
season rows retain the base-plus-group prediction.

The feature allow-list contains only current-row official inputs and
row-independent features derived from the stored train history.  Raw player
IDs, team IDs, season, and any aggregation across validation/test rows are not
used.  Configurations, residual blend weights, and cross-model 50/50 blending
are fixed before the reported folds are evaluated.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from train_exp017_rolling_residual import calculate_metrics, prepare_data
from train_exp018_constrained_multiscale import (
    build_group_keys,
    centered_residual,
    group_correction,
)


ARTIFACT_ROOT = Path("./artifacts/EXP-019/histgb_residual")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
RESIDUAL_BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
REFERENCE_ROOT = Path(
    "./artifacts/EXP-019/r_full_residual/rfull_l63_m1000_i300"
)
REFERENCE_VARIANT = "branch_w075"


# Explicitly stable, row-local numeric features.  The encoded prefixes below
# are also row-local categorical indicators generated from official columns.
STABLE_FEATURES = {
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_hand",
    "batter_hand",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "count_index",
    "count_out_index",
    "is_full_count",
    "has_two_strikes",
    "has_three_balls",
    "count_advantage",
    "runner_in_scoring_position",
    "bases_loaded",
    "same_hand",
    "late_inning",
    "close_game",
    "log_li",
    "score_pressure",
    "win_expectancy_gap",
    "pitcher_batter_success_gap",
    "pitcher_recent_success_delta_1_5",
    "pitcher_recent_success_delta_3_5",
    "pitcher_recent_middle_delta_1_5",
    "log_pitcher_n",
    "log_batter_n",
    "log_pitchmix_n",
    "temporal_prior_league_rate",
    "temporal_pitcher_prior_exists",
    "temporal_pitcher_log_prior_n",
    "temporal_pitcher_prior_rate_shrunk_200",
    "temporal_pitcher_log_season_n",
    "temporal_pitcher_season_global_30",
    "temporal_pitcher_season_player_30",
    "temporal_pitcher_reliability_30",
    "temporal_batter_prior_exists",
    "temporal_batter_log_prior_n",
    "temporal_batter_prior_rate_shrunk_200",
    "temporal_batter_log_season_n",
    "temporal_batter_season_global_30",
    "temporal_batter_season_player_30",
    "temporal_batter_reliability_30",
    "temporal_base_global_30",
    "temporal_base_player_30",
}
ENCODED_PREFIXES = ("top_bottom_", "base_state_")


@dataclass(frozen=True)
class Config:
    name: str
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    max_depth: int
    min_samples_leaf: int
    l2_regularization: float
    max_bins: int
    max_features: float


CONFIGS = [
    Config(
        name="hist_l7_d3_m5000_i120",
        max_iter=120,
        learning_rate=0.035,
        max_leaf_nodes=7,
        max_depth=3,
        min_samples_leaf=5000,
        l2_regularization=20.0,
        max_bins=63,
        max_features=0.70,
    ),
    Config(
        name="hist_l15_d4_m3000_i160",
        max_iter=160,
        learning_rate=0.025,
        max_leaf_nodes=15,
        max_depth=4,
        min_samples_leaf=3000,
        l2_regularization=30.0,
        max_bins=127,
        max_features=0.70,
    ),
]


def season_equal_weights(seasons: np.ndarray) -> np.ndarray:
    """Give each past season equal total weight without using future rows."""
    weights = np.zeros(len(seasons), dtype=np.float32)
    unique, counts = np.unique(seasons, return_counts=True)
    for season, count in zip(unique, counts, strict=True):
        weights[seasons == season] = 1.0 / float(count)
    weights *= len(weights) / weights.sum()
    return weights


def select_stable_features(
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    selected_indices = [
        index
        for index, name in enumerate(feature_names)
        if name in STABLE_FEATURES
        or any(name.startswith(prefix) for prefix in ENCODED_PREFIXES)
    ]
    selected_names = [feature_names[index] for index in selected_indices]
    forbidden_tokens = ("pitcher_id", "batter_id", "team_id")
    forbidden_exact = {"season", "row_id", "control_success"}
    invalid = [
        name
        for name in selected_names
        if name in forbidden_exact
        or any(token in name for token in forbidden_tokens)
    ]
    if invalid:
        raise ValueError(f"forbidden features selected: {invalid}")
    missing = sorted(STABLE_FEATURES.difference(selected_names))
    if missing:
        raise ValueError(f"stable features missing from matrix: {missing}")
    return np.ascontiguousarray(X[:, selected_indices]), selected_names


def build_temporal_group_predictions(
    X: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[int, dict[str, int]]]:
    """Build base-plus-group OOF predictions for every train season."""
    residual = centered_residual(y, base, seasons)
    keys = build_group_keys(X, feature_names)
    predictions = np.empty(len(y), dtype=np.float64)
    counts: dict[int, dict[str, int]] = {}
    for season in sorted(np.unique(seasons).astype(int).tolist()):
        mask = seasons == season
        correction, group_counts = group_correction(
            keys,
            residual,
            seasons,
            season,
        )
        predictions[mask] = np.clip(
            base[mask].astype(float) + correction.astype(float),
            0.0,
            1.0,
        )
        counts[season] = group_counts
    return predictions, counts


def centered_group_residual(
    y: np.ndarray,
    group_predictions: np.ndarray,
    seasons: np.ndarray,
    is_regular: np.ndarray,
) -> np.ndarray:
    residual = (y.astype(float) - group_predictions).astype(np.float32)
    for season in np.unique(seasons):
        mask = (seasons == season) & is_regular
        if mask.any():
            residual[mask] -= residual[mask].mean()
    return residual


def optimal_constrained_weight(
    reference: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
) -> float:
    """Same-fold diagnostic weight for candidate in a convex blend."""
    delta = candidate - reference
    denominator = float(np.mean(delta * delta))
    if denominator <= 0.0:
        return 0.0
    numerator = -float(np.mean((reference - target) * delta))
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def variant_name(weight: float) -> str:
    return f"branch_w{int(round(weight * 100)):03d}"


def compare_with_reference(
    prediction_paths: dict[str, dict[int, Path]],
) -> dict[str, object]:
    """Measure fixed and clearly labelled diagnostic error complementarity."""
    required = [
        REFERENCE_ROOT / f"predictions_{REFERENCE_VARIANT}_{season}.npy"
        for season in VALIDATION_SEASONS
    ]
    required += [
        REFERENCE_ROOT / f"targets_{season}.npy"
        for season in VALIDATION_SEASONS
    ]
    if not all(path.exists() for path in required):
        return {
            "available": False,
            "reason": "r_full reference OOF arrays are missing",
            "reference_root": str(REFERENCE_ROOT),
        }

    reference_predictions = {
        season: np.load(
            REFERENCE_ROOT
            / f"predictions_{REFERENCE_VARIANT}_{season}.npy"
        ).astype(float)
        for season in VALIDATION_SEASONS
    }
    targets = {
        season: np.load(
            REFERENCE_ROOT / f"targets_{season}.npy"
        ).astype(float)
        for season in VALIDATION_SEASONS
    }
    variants: dict[str, object] = {}
    for name, paths in prediction_paths.items():
        candidate_predictions = {
            season: np.load(paths[season]).astype(float)
            for season in VALIDATION_SEASONS
        }
        folds: dict[str, object] = {}
        fixed_skills: list[float] = []
        prior_skills: list[float] = []
        for season in VALIDATION_SEASONS:
            target = targets[season]
            reference = reference_predictions[season]
            candidate = candidate_predictions[season]
            candidate_target = np.load(
                paths[season].parent / f"targets_{season}.npy"
            ).astype(float)
            if not (
                len(target) == len(reference) == len(candidate)
                and np.array_equal(target, candidate_target)
            ):
                raise ValueError(f"OOF alignment mismatch for {name} {season}")
            fixed_half = 0.5 * reference + 0.5 * candidate
            oracle_weight = optimal_constrained_weight(
                reference,
                candidate,
                target,
            )
            oracle = (
                (1.0 - oracle_weight) * reference
                + oracle_weight * candidate
            )
            previous_season = season - 1
            if previous_season in candidate_predictions:
                prior_weight = optimal_constrained_weight(
                    reference_predictions[previous_season],
                    candidate_predictions[previous_season],
                    targets[previous_season],
                )
            else:
                prior_weight = 0.0
            prior_blend = (
                (1.0 - prior_weight) * reference
                + prior_weight * candidate
            )
            error_correlation = float(
                np.corrcoef(reference - target, candidate - target)[0, 1]
            )
            correction_correlation = float(
                np.corrcoef(
                    reference - reference.mean(),
                    candidate - candidate.mean(),
                )[0, 1]
            )
            folds[str(season)] = {
                "error_correlation": error_correlation,
                "centered_prediction_correlation": correction_correlation,
                "reference": calculate_metrics(target, reference),
                "candidate": calculate_metrics(target, candidate),
                "fixed_50_50": calculate_metrics(target, fixed_half),
                "same_fold_oracle_diagnostic": {
                    "candidate_weight": oracle_weight,
                    **calculate_metrics(target, oracle),
                },
                "previous_fold_weight_diagnostic": {
                    "candidate_weight": prior_weight,
                    "weight_source_season": previous_season,
                    **calculate_metrics(target, prior_blend),
                },
            }
            if season in REPORT_SEASONS:
                fixed_skills.append(
                    folds[str(season)]["fixed_50_50"][
                        "skill_score_unclipped"
                    ]
                )
                prior_skills.append(
                    folds[str(season)]["previous_fold_weight_diagnostic"][
                        "skill_score_unclipped"
                    ]
                )
        variants[name] = {
            "folds": folds,
            "aggregate_2022_2024": {
                "fixed_50_50_mean_skill": float(np.mean(fixed_skills)),
                "fixed_50_50_min_skill": float(np.min(fixed_skills)),
                "previous_fold_weight_mean_skill": float(
                    np.mean(prior_skills)
                ),
                "previous_fold_weight_min_skill": float(
                    np.min(prior_skills)
                ),
            },
        }
    return {
        "available": True,
        "reference_root": str(REFERENCE_ROOT),
        "reference_variant": REFERENCE_VARIANT,
        "notes": {
            "fixed_50_50": "predefined, no current-fold fitting",
            "same_fold_oracle": "diagnostic only; uses current-fold labels",
            "previous_fold_weight": (
                "weight fitted on immediately previous OOF fold; candidate "
                "config itself remains a multi-candidate diagnostic"
            ),
        },
        "variants": variants,
    }


def main() -> None:
    started = time.time()
    diagnostics, full_X, y, base, seasons, feature_names = prepare_data()
    del diagnostics
    game_type_r_name = "game_type_R"
    if game_type_r_name not in feature_names:
        raise ValueError(f"missing encoded feature: {game_type_r_name}")
    is_regular = full_X[:, feature_names.index(game_type_r_name)] > 0.5
    group_predictions, group_counts = build_temporal_group_predictions(
        full_X,
        y,
        base,
        seasons,
        feature_names,
    )
    X, selected_features = select_stable_features(full_X, feature_names)
    del full_X
    residual_target = centered_group_residual(
        y,
        group_predictions,
        seasons,
        is_regular,
    )
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    prediction_paths: dict[str, dict[int, Path]] = {}

    print(
        f"rows={len(y)} stable_features={len(selected_features)} "
        f"regular_rows={int(is_regular.sum())}"
    )
    for config in CONFIGS:
        artifact_dir = ARTIFACT_ROOT / config.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        folds: dict[str, object] = {}
        for validation_season in VALIDATION_SEASONS:
            train_mask = (seasons < validation_season) & is_regular
            validation_mask = seasons == validation_season
            validation_regular = validation_mask & is_regular
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=config.learning_rate,
                max_iter=config.max_iter,
                max_leaf_nodes=config.max_leaf_nodes,
                max_depth=config.max_depth,
                min_samples_leaf=config.min_samples_leaf,
                l2_regularization=config.l2_regularization,
                max_features=config.max_features,
                max_bins=config.max_bins,
                early_stopping=False,
                random_state=42,
            )
            fit_started = time.time()
            model.fit(
                X[train_mask],
                residual_target[train_mask],
                sample_weight=season_equal_weights(seasons[train_mask]),
            )
            fit_seconds = time.time() - fit_started
            prediction_started = time.time()
            correction = np.zeros(int(validation_mask.sum()), dtype=float)
            local_regular = is_regular[validation_mask]
            correction[local_regular] = model.predict(
                X[validation_regular]
            ).astype(float)
            inference_seconds = time.time() - prediction_started
            targets = y[validation_mask]
            fold: dict[str, object] = {
                "validation_season": validation_season,
                "training_seasons": sorted(
                    np.unique(seasons[train_mask]).astype(int).tolist()
                ),
                "training_rows": int(train_mask.sum()),
                "validation_regular_rows": int(validation_regular.sum()),
                "fit_seconds": fit_seconds,
                "inference_seconds": inference_seconds,
                "model_iterations_completed": int(model.n_iter_),
                "correction_mean_regular": float(
                    correction[local_regular].mean()
                ),
                "base_plus_group": calculate_metrics(
                    targets,
                    group_predictions[validation_mask],
                ),
            }
            for weight in RESIDUAL_BLEND_WEIGHTS:
                candidate = variant_name(weight)
                predictions = np.clip(
                    group_predictions[validation_mask]
                    + weight * correction,
                    0.0,
                    1.0,
                )
                fold[candidate] = calculate_metrics(targets, predictions)
                prediction_path = (
                    artifact_dir
                    / f"predictions_{candidate}_{validation_season}.npy"
                )
                np.save(prediction_path, predictions)
                key = f"{config.name}/{candidate}"
                prediction_paths.setdefault(key, {})[
                    validation_season
                ] = prediction_path
            np.save(
                artifact_dir / f"targets_{validation_season}.npy",
                targets.astype(np.int8),
            )
            folds[str(validation_season)] = fold
            print(
                f"{config.name} {validation_season}: "
                + " ".join(
                    f"w{int(weight * 100):03d}="
                    f"{fold[variant_name(weight)]['skill_score_unclipped']:.2f}"
                    for weight in RESIDUAL_BLEND_WEIGHTS
                )
                + f" fit={fit_seconds:.1f}s"
            )

        aggregate: dict[str, object] = {}
        for weight in RESIDUAL_BLEND_WEIGHTS:
            candidate = variant_name(weight)
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
                "base": "EXP-018 temporal hierarchical base",
                "group": (
                    "past-only three-season count/hand/reverse-bin correction"
                ),
                "residual_training": "past regular-season rows only",
                "non_regular_prediction": "base plus group only",
                "current_fold_labels_used_for_training": False,
                "test_row_aggregation": False,
                "candidate_status": (
                    "predetermined diagnostic configs and weights; nested "
                    "selection required before adoption"
                ),
            },
            "model": {
                **asdict(config),
                "features": selected_features,
                "feature_count": len(selected_features),
                "dropped": [
                    "season",
                    "pitcher_id",
                    "batter_id",
                    "pitcher_team_id",
                    "batter_team_id",
                    "game_type",
                ],
            },
            "group_counts": group_counts,
            "folds": folds,
            "aggregate_2022_2024": aggregate,
            "environment": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
        with (artifact_dir / "validation_metrics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        summaries[config.name] = aggregate

    comparison = compare_with_reference(prediction_paths)
    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "experiment": "EXP-019",
                "stage": "histgradientboosting_regular_residual",
                "selection_status": (
                    "comparison only; nested temporal selection required"
                ),
                "summaries": summaries,
                "reference_comparison": comparison,
                "total_seconds": time.time() - started,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    with (ARTIFACT_ROOT / "feature_names.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(selected_features, file, ensure_ascii=False, indent=2)
    print(f"saved={ARTIFACT_ROOT / 'validation_metrics.json'}")


if __name__ == "__main__":
    main()
