"""EXP-020: bounded season-invariant binned-GAM/EB residual diagnostic.

The immutable base is the saved temporal-safe team all-prior OOF plus the
R-specific low-rank pitcher-context correction.  Each outer fold learns only
from earlier OOF seasons.  A separate source-season map is fitted after
removing that source season's residual mean, and validation corrections are
the equal mean of the eligible source maps.

Only stable current-row pitcher state is used: career success, reverse rate,
recent-three-game success, temporally reconstructed current-season success,
and reliability as an interaction axis.  Raw IDs, team IDs, season, batter
rates, ball/strike rates, current-fold labels, validation/test aggregates, and
test-row distributions are never model inputs.  Fixed bin edges, EB
smoothing, monotone directions, pair definitions, and four candidate formulas
are declared below before evaluation.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression

from temporal_residual_features import attach_training_temporal_features
from train_exp017_rolling_residual import calculate_metrics


DATA_PATH = Path("./data/train.csv")
BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-020/binned_gam_residual")

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
UNIVARIATE_SMOOTHING = 5000.0
PAIR_SMOOTHING = 15000.0
COMPONENT_CLIP = 0.020
PAIR_COMPONENT_CLIP = 0.010
CORE_RAW_CLIP = 0.035
CORE_PAIR_RAW_CLIP = 0.045


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    column: str
    edges: tuple[float, ...]
    fill_value: float
    increasing: bool


SUCCESS_EDGES = (-0.001, 0.40, 0.45, 0.475, 0.50, 0.525, 0.55, 0.60, 1.001)
REVERSE_EDGES = (-0.001, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 1.001)
RELIABILITY_EDGES = (-0.001, 0.10, 0.30, 0.50, 0.70, 0.85, 0.95, 1.001)

FEATURES = (
    FeatureSpec(
        "career_success",
        "asof_pitcher_success_rate",
        SUCCESS_EDGES,
        0.50,
        True,
    ),
    FeatureSpec(
        "career_reverse",
        "asof_pitcher_reverse_rate",
        REVERSE_EDGES,
        0.20,
        False,
    ),
    FeatureSpec(
        "recent3_success",
        "asof_pitcher_prev3_game_success_rate",
        SUCCESS_EDGES,
        0.50,
        True,
    ),
    FeatureSpec(
        "season_success_global30",
        "temporal_pitcher_season_global_30",
        SUCCESS_EDGES,
        0.50,
        True,
    ),
)
FEATURE_BY_NAME = {feature.name: feature for feature in FEATURES}
RELIABILITY = FeatureSpec(
    "season_reliability30",
    "temporal_pitcher_reliability_30",
    RELIABILITY_EDGES,
    0.0,
    True,
)
PAIR_INTERACTIONS = (
    ("season_success_global30", RELIABILITY.name),
    ("recent3_success", RELIABILITY.name),
)

BASE_CANDIDATE = "fixed_Rspecific_lowrank_base"
CANDIDATE_CONFIGS = {
    "core_all_w025": {
        "source_regime": "all",
        "application_regime": "all",
        "include_pairs": False,
        "scale": 0.25,
    },
    "core_R_w025": {
        "source_regime": "R",
        "application_regime": "R",
        "include_pairs": False,
        "scale": 0.25,
    },
    "core_pairs_R_w025": {
        "source_regime": "R",
        "application_regime": "R",
        "include_pairs": True,
        "scale": 0.25,
    },
    "core_pairs_R_w050": {
        "source_regime": "R",
        "application_regime": "R",
        "include_pairs": True,
        "scale": 0.50,
    },
}
CANDIDATES = (BASE_CANDIDATE, *CANDIDATE_CONFIGS)
FORBIDDEN_MODEL_INPUTS = {
    "row_id",
    "control_success",
    "season",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "asof_batter_success_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
}


def prepare_rows() -> dict[int, pd.DataFrame]:
    columns = [
        "season",
        "game_type",
        "pitcher_id",
        "batter_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "control_success",
    ]
    frame = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=columns,
    )
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("train rows must be ordered by season")
    frame, _ = attach_training_temporal_features(
        frame, target="control_success"
    )
    model_columns = {feature.column for feature in FEATURES} | {
        RELIABILITY.column
    }
    if model_columns & FORBIDDEN_MODEL_INPUTS:
        raise ValueError("forbidden model feature configured")
    missing = sorted(model_columns - set(frame.columns))
    if missing:
        raise ValueError(f"missing model features: {missing}")
    if set(frame["game_type"].astype(str).unique()) != {"F", "R"}:
        raise ValueError("unexpected game_type domain")
    frame = frame.loc[frame["season"].isin(EVALUATED_SEASONS)].copy()
    for feature in (*FEATURES, RELIABILITY):
        values = (
            frame[feature.column]
            .fillna(feature.fill_value)
            .to_numpy(dtype=np.float64)
        )
        values = np.clip(values, 0.0, 1.0)
        bins = np.digitize(values, feature.edges[1:-1], right=False)
        if not (
            (bins >= 0).all()
            and (bins < len(feature.edges) - 1).all()
        ):
            raise ValueError(f"invalid bin for {feature.name}")
        frame[f"bin_{feature.name}"] = bins.astype(np.int8)
    return {
        season: frame.loc[frame["season"].eq(season)].reset_index(drop=True)
        for season in EVALUATED_SEASONS
    }


def load_oof(
    rows: dict[int, pd.DataFrame],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(
            BASE_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        base[season] = np.load(
            BASE_ROOT
            / f"predictions_lowrank_s300_r4_Rspecific_{season}.npy"
        ).astype(np.float64)
        csv_target = rows[season]["control_success"].to_numpy(
            dtype=np.float64
        )
        if not (
            len(csv_target) == len(targets[season]) == len(base[season])
            and np.array_equal(csv_target, targets[season])
        ):
            raise ValueError(f"OOF alignment mismatch for {season}")
        if not np.isfinite(base[season]).all() or not (
            (base[season] >= 0.0).all()
            and (base[season] <= 1.0).all()
        ):
            raise ValueError(f"invalid fixed base for {season}")
    return targets, base


def centered_and_clipped(
    effects: np.ndarray,
    source_bins: np.ndarray,
    clip: float,
) -> np.ndarray:
    centered = effects - float(np.mean(effects[source_bins]))
    centered = np.clip(centered, -clip, clip)
    centered -= float(np.mean(centered[source_bins]))
    return centered


def fit_univariate_effect(
    bins: np.ndarray,
    residual: np.ndarray,
    feature: FeatureSpec,
) -> tuple[np.ndarray, dict[str, object]]:
    bin_count = len(feature.edges) - 1
    counts = np.bincount(bins, minlength=bin_count).astype(np.float64)
    sums = np.bincount(
        bins, weights=residual, minlength=bin_count
    ).astype(np.float64)
    raw = sums / (counts + UNIVARIATE_SMOOTHING)
    isotonic = IsotonicRegression(
        increasing=feature.increasing,
        out_of_bounds="clip",
    )
    monotone = isotonic.fit_transform(
        np.arange(bin_count, dtype=np.float64),
        raw,
        sample_weight=counts + UNIVARIATE_SMOOTHING,
    )
    effect = centered_and_clipped(
        monotone.astype(np.float64), bins, COMPONENT_CLIP
    )
    return effect, {
        "counts": [int(value) for value in counts],
        "raw_EB_effects": [float(value) for value in raw],
        "monotone_centered_effects": [float(value) for value in effect],
        "increasing": feature.increasing,
        "observed_bins": int((counts > 0).sum()),
    }


def fit_pair_effect(
    first_bins: np.ndarray,
    second_bins: np.ndarray,
    residual_after_core: np.ndarray,
    first_count: int,
    second_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    cell = first_bins.astype(np.int16) * second_count + second_bins
    cell_count = first_count * second_count
    counts = np.bincount(cell, minlength=cell_count).astype(np.float64)
    sums = np.bincount(
        cell, weights=residual_after_core, minlength=cell_count
    ).astype(np.float64)
    raw = sums / (counts + PAIR_SMOOTHING)
    effect = centered_and_clipped(raw, cell, PAIR_COMPONENT_CLIP)
    return effect.reshape(first_count, second_count), {
        "shape": [first_count, second_count],
        "observed_cells": int((counts > 0).sum()),
        "total_cells": int(cell_count),
        "min_cell_rows": int(counts.min()),
        "median_cell_rows": float(np.median(counts)),
        "max_cell_rows": int(counts.max()),
        "mean_absolute_effect": float(np.abs(effect).mean()),
        "max_absolute_effect": float(np.abs(effect).max()),
    }


def fit_source_map(
    source_season: int,
    source_rows: pd.DataFrame,
    targets: np.ndarray,
    base: np.ndarray,
    regime: str,
) -> dict[str, object]:
    if regime not in {"all", "R"}:
        raise ValueError(f"unknown source regime: {regime}")
    source_mask = np.ones(len(source_rows), dtype=bool)
    if regime == "R":
        source_mask = (
            source_rows["game_type"].astype(str).to_numpy() == "R"
        )
    fitting_rows = source_rows.loc[source_mask].reset_index(drop=True)
    raw_residual = targets[source_mask] - base[source_mask]
    raw_mean = float(raw_residual.mean())
    residual = raw_residual - raw_mean
    if abs(float(residual.mean())) > 1e-12:
        raise AssertionError("source residual centering failed")

    bins = {
        feature.name: fitting_rows[f"bin_{feature.name}"].to_numpy(
            dtype=np.int16
        )
        for feature in (*FEATURES, RELIABILITY)
    }
    univariate_effects: dict[str, np.ndarray] = {}
    univariate_diagnostics: dict[str, object] = {}
    core_train = np.zeros(len(fitting_rows), dtype=np.float64)
    for feature in FEATURES:
        effect, diagnostics = fit_univariate_effect(
            bins[feature.name], residual, feature
        )
        univariate_effects[feature.name] = effect
        univariate_diagnostics[feature.name] = diagnostics
        core_train += effect[bins[feature.name]]
    core_train = np.clip(core_train, -CORE_RAW_CLIP, CORE_RAW_CLIP)

    residual_after_core = residual - core_train
    pair_effects: dict[str, np.ndarray] = {}
    pair_diagnostics: dict[str, object] = {}
    for first, second in PAIR_INTERACTIONS:
        first_spec = FEATURE_BY_NAME[first]
        effect, diagnostics = fit_pair_effect(
            bins[first],
            bins[second],
            residual_after_core,
            len(first_spec.edges) - 1,
            len(RELIABILITY.edges) - 1,
        )
        pair_effects[f"{first}__{second}"] = effect
        pair_diagnostics[f"{first}__{second}"] = diagnostics

    return {
        "source_season": source_season,
        "regime": regime,
        "univariate_effects": univariate_effects,
        "pair_effects": pair_effects,
        "diagnostics": {
            "source_rows_total": int(len(source_rows)),
            "source_rows_used": int(source_mask.sum()),
            "raw_residual_mean_before_centering": raw_mean,
            "residual_mean_after_centering": float(residual.mean()),
            "univariate": univariate_diagnostics,
            "pairs": pair_diagnostics,
            "core_train_mean": float(core_train.mean()),
            "core_train_mean_absolute": float(np.abs(core_train).mean()),
        },
    }


def map_source(
    source_map: dict[str, object],
    validation_rows: pd.DataFrame,
    include_pairs: bool,
) -> np.ndarray:
    core = np.zeros(len(validation_rows), dtype=np.float64)
    for feature in FEATURES:
        bins = validation_rows[f"bin_{feature.name}"].to_numpy(
            dtype=np.int16
        )
        core += source_map["univariate_effects"][feature.name][bins]
    core = np.clip(core, -CORE_RAW_CLIP, CORE_RAW_CLIP)
    if not include_pairs:
        return core

    pair_sum = np.zeros(len(validation_rows), dtype=np.float64)
    for first, second in PAIR_INTERACTIONS:
        first_bins = validation_rows[f"bin_{first}"].to_numpy(
            dtype=np.int16
        )
        second_bins = validation_rows[f"bin_{second}"].to_numpy(
            dtype=np.int16
        )
        pair_sum += source_map["pair_effects"][
            f"{first}__{second}"
        ][first_bins, second_bins]
    return np.clip(core + pair_sum, -CORE_PAIR_RAW_CLIP, CORE_PAIR_RAW_CLIP)


def regime_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    game_types: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        game_type: calculate_metrics(
            targets[game_types == game_type],
            predictions[game_types == game_type],
        )
        for game_type in sorted(np.unique(game_types))
    }


def aggregate_metrics(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for candidate in CANDIDATES:
        metrics = {
            season: folds[str(season)]["candidates"][candidate]["metrics"]
            for season in REPORT_SEASONS
        }
        skills = {
            season: float(value["skill_score_unclipped"])
            for season, value in metrics.items()
        }
        aggregate[candidate] = {
            "season_briers": {
                str(season): float(value["brier_score"])
                for season, value in metrics.items()
            },
            "season_skills": {
                str(season): value for season, value in skills.items()
            },
            "season_mean_gaps": {
                str(season): float(value["mean_gap"])
                for season, value in metrics.items()
            },
            "season_calibration_slopes": {
                str(season): float(
                    value["diagnostic_calibration_slope"]
                )
                for season, value in metrics.items()
            },
            "season_calibration_intercepts": {
                str(season): float(
                    value["diagnostic_calibration_intercept"]
                )
                for season, value in metrics.items()
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
            "uniform_1100_passed": bool(
                all(value >= 1100.0 for value in skills.values())
            ),
        }
    base = aggregate[BASE_CANDIDATE]
    for candidate in CANDIDATE_CONFIGS:
        current = aggregate[candidate]
        current["season_skill_change_vs_base"] = {
            str(season): float(
                current["season_skills"][str(season)]
                - base["season_skills"][str(season)]
            )
            for season in REPORT_SEASONS
        }
        current["mean_skill_change_vs_base"] = float(
            current["mean_skill"] - base["mean_skill"]
        )
        current["min_skill_change_vs_base"] = float(
            current["min_skill"] - base["min_skill"]
        )
    return aggregate


def main() -> None:
    started = time.time()
    rows = prepare_rows()
    targets, base = load_oof(rows)
    source_maps: dict[tuple[int, str], dict[str, object]] = {}

    def get_source_map(source_season: int, regime: str) -> dict[str, object]:
        key = (source_season, regime)
        if key not in source_maps:
            source_maps[key] = fit_source_map(
                source_season,
                rows[source_season],
                targets[source_season],
                base[source_season],
                regime,
            )
        return source_maps[key]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        source_seasons = [
            season
            for season in EVALUATED_SEASONS
            if season < validation_season
        ]
        validation_rows = rows[validation_season]
        game_types = validation_rows["game_type"].astype(str).to_numpy()
        is_r = game_types == "R"
        predictions: dict[str, np.ndarray] = {
            BASE_CANDIDATE: base[validation_season].copy()
        }
        correction_diagnostics: dict[str, object] = {}
        for candidate, config in CANDIDATE_CONFIGS.items():
            source_regime = str(config["source_regime"])
            apply_r_only = config["application_regime"] == "R"
            application_rows = (
                validation_rows.loc[is_r].reset_index(drop=True)
                if apply_r_only
                else validation_rows
            )
            mapped = [
                map_source(
                    get_source_map(source_season, source_regime),
                    application_rows,
                    bool(config["include_pairs"]),
                )
                for source_season in source_seasons
            ]
            local = np.mean(np.vstack(mapped), axis=0) if mapped else np.zeros(
                len(application_rows), dtype=np.float64
            )
            correction = np.zeros(len(validation_rows), dtype=np.float64)
            if apply_r_only:
                correction[is_r] = float(config["scale"]) * local
            else:
                correction = float(config["scale"]) * local
            candidate_predictions = np.clip(
                base[validation_season] + correction, 0.0, 1.0
            )
            if apply_r_only and not np.array_equal(
                candidate_predictions[~is_r],
                base[validation_season][~is_r],
            ):
                raise AssertionError(f"F base invariant failed: {candidate}")
            predictions[candidate] = candidate_predictions
            correction_diagnostics[candidate] = {
                "source_regime": source_regime,
                "application_regime": config["application_regime"],
                "source_seasons": source_seasons,
                "source_season_equal_weighting": True,
                "mean": float(correction.mean()),
                "standard_deviation": float(correction.std()),
                "mean_absolute": float(np.abs(correction).mean()),
                "min": float(correction.min()),
                "max": float(correction.max()),
            }
        if tuple(predictions) != CANDIDATES:
            raise AssertionError("candidate declaration/order drift")

        candidate_metrics: dict[str, object] = {}
        for candidate, candidate_predictions in predictions.items():
            if not np.isfinite(candidate_predictions).all() or not (
                (candidate_predictions >= 0.0).all()
                and (candidate_predictions <= 1.0).all()
            ):
                raise ValueError(
                    f"invalid predictions {validation_season} {candidate}"
                )
            candidate_metrics[candidate] = {
                "metrics": calculate_metrics(
                    targets[validation_season], candidate_predictions
                ),
                "regime_metrics": regime_metrics(
                    targets[validation_season],
                    candidate_predictions,
                    game_types,
                ),
            }
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate}_{validation_season}.npy",
                candidate_predictions,
            )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets[validation_season].astype(np.int8),
        )
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "rows": int(len(validation_rows)),
            "R_rows": int(is_r.sum()),
            "F_rows": int((~is_r).sum()),
            "correction_diagnostics": correction_diagnostics,
            "candidates": candidate_metrics,
            "strict_invariants": {
                "all_source_seasons_precede_validation": bool(
                    all(
                        source_season < validation_season
                        for source_season in source_seasons
                    )
                ),
                "current_fold_labels_used_for_maps": False,
                "current_fold_used_for_bins_or_weights": False,
                "foldwise_candidate_selection_performed": False,
                "validation_or_test_row_aggregation": False,
                "fixed_bin_edges": True,
                "source_residual_centered_per_season": True,
                "source_season_equal_weighting": True,
                "R_candidates_F_predictions_equal_base": True,
            },
        }
        print(
            f"binned_gam {validation_season}: "
            + " ".join(
                f"{candidate}="
                f"{candidate_metrics[candidate]['metrics']['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    aggregate = aggregate_metrics(folds)
    best_mean = max(
        CANDIDATE_CONFIGS,
        key=lambda candidate: (
            aggregate[candidate]["mean_skill"],
            aggregate[candidate]["min_skill"],
            -list(CANDIDATE_CONFIGS).index(candidate),
        ),
    )
    best_min = max(
        CANDIDATE_CONFIGS,
        key=lambda candidate: (
            aggregate[candidate]["min_skill"],
            aggregate[candidate]["latest_2024_skill"],
            aggregate[candidate]["mean_skill"],
            -list(CANDIDATE_CONFIGS).index(candidate),
        ),
    )
    best_observed_min = float(aggregate[best_min]["min_skill"])
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": "season_invariant_binned_GAM_EB_residual",
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": (
                "EXP-020 team-allprior plus R-specific lowrank_s300_r4 "
                "pitcher-context OOF"
            ),
            "effect_training": (
                "independent source-season maps on source-season-centered "
                "OOF residual; equal mean across earlier sources"
            ),
            "current_fold_labels_used_for_maps": False,
            "current_fold_used_for_bins_or_weights": False,
            "foldwise_candidate_selection_performed": False,
            "candidate_ranking_is_posthoc_reporting_only": True,
            "validation_or_test_row_aggregation": False,
            "test_rows_read": False,
            "candidate_grid_predeclared": True,
            "candidate_comparison_nested": False,
        },
        "predeclared_configuration": {
            "candidate_count_excluding_base": len(CANDIDATE_CONFIGS),
            "candidates": CANDIDATE_CONFIGS,
            "univariate_smoothing": UNIVARIATE_SMOOTHING,
            "pair_smoothing": PAIR_SMOOTHING,
            "component_clip": COMPONENT_CLIP,
            "pair_component_clip": PAIR_COMPONENT_CLIP,
            "core_raw_clip": CORE_RAW_CLIP,
            "core_pair_raw_clip": CORE_PAIR_RAW_CLIP,
            "features": {
                feature.name: {
                    "column": feature.column,
                    "edges": list(feature.edges),
                    "fill_value": feature.fill_value,
                    "monotone_direction": (
                        "increasing" if feature.increasing else "decreasing"
                    ),
                }
                for feature in FEATURES
            },
            "reliability_interaction_axis": {
                "column": RELIABILITY.column,
                "edges": list(RELIABILITY.edges),
                "fill_value": RELIABILITY.fill_value,
            },
            "pair_interactions": [
                list(interaction) for interaction in PAIR_INTERACTIONS
            ],
            "forbidden_inputs_absent": True,
            "explicitly_excluded_unstable_inputs": [
                "raw IDs and team IDs",
                "season",
                "batter success rate",
                "pitcher ball rate",
                "pitcher strike rate",
            ],
        },
        "source_map_diagnostics": {
            f"{season}_{regime}": source_map["diagnostics"]
            for (season, regime), source_map in source_maps.items()
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "diagnostic only; candidate comparison is non-nested",
            "posthoc_best_mean_candidate": best_mean,
            "posthoc_best_min_candidate": best_min,
            "best_observed_min_skill": best_observed_min,
            "uniform_1100_gate_passed": bool(best_observed_min >= 1100.0),
            "stop_rule_triggered": bool(best_observed_min < 1100.0),
            "stop_reason": (
                "best minimum Skill remains below the 1100 robustness gate"
                if best_observed_min < 1100.0
                else "gate passed"
            ),
            "adopt_without_nested_confirmation": False,
        },
        "qa": {
            "source_target_and_base_alignment_checked": True,
            "source_residual_centering_checked": True,
            "source_season_order_checked": True,
            "fixed_bin_domain_checked": True,
            "forbidden_feature_check": True,
            "R_candidate_F_base_equality_checked": True,
            "prediction_probability_ranges_checked": True,
            "saved_prediction_arrays": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "total_seconds": float(time.time() - started),
    }
    metrics_path = ARTIFACT_DIR / "validation_metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"selection={result['selection']}", flush=True)
    print(f"saved={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
