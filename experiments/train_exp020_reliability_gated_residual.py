"""EXP-020: reliability-gated regular-season residual LightGBM.

This bounded comparison keeps the EXP-019 temporal base plus the past-only
count/hand/reverse group prediction.  Only regular-season rows receive a
learned residual correction; non-regular-season rows always retain the group
prediction.  Three predeclared reliability schemes are compared:

1. train and apply only when pitcher current-season n >= 100 and both players
   existed before the current season;
2. train on every regular-season row with season-equal weights multiplied by
   sqrt(pitcher reliability30 * batter reliability30), floored at 0.1, and
   apply to every regular-season row;
3. separate experts for pitcher current-season n >= 100 and n < 100.

All features are current-row official inputs or row-independent temporal
features reconstructed from prior train seasons.  Raw IDs, team IDs, season,
validation/test-row aggregates, and current-fold labels are excluded.
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


ARTIFACT_DIR = Path("./artifacts/EXP-020/reliability_gated_residual")
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
SCHEMES = ("high_both", "reliability_weighted", "two_experts")
BLEND_WEIGHTS = (0.50, 0.75)
HIGH_PITCHER_N = 100.0
RELIABILITY_FLOOR = 0.10
ITERATIONS = 300
LEARNING_RATE = 0.015
NUM_LEAVES = 31
MIN_CHILD_SAMPLES = 2000
TEAM_EB_METRICS = Path(
    "./artifacts/EXP-019/team_eb_ensemble/validation_metrics.json"
)
TEAM_EB_REFERENCE = "all_prior_s1000"


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


def build_full_feature_matrix(
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
    forbidden_exact = {"season", "row_id", "control_success"}
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
    base: np.ndarray,
    seasons: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    initial_residual = multirate.centered_residual(y, base, seasons)
    group_all = np.empty(len(y), dtype=np.float64)
    reported: dict[int, np.ndarray] = {}
    for season in sorted(np.unique(seasons).astype(int).tolist()):
        mask = seasons == season
        correction = rfull.original_group_correction(
            frame,
            initial_residual,
            seasons,
            season,
        )
        prediction = np.clip(
            base[mask].astype(float) + correction,
            0.0,
            1.0,
        )
        group_all[mask] = prediction
        if season in VALIDATION_SEASONS:
            reported[season] = prediction
    return group_all, reported


def season_equal_reliability_weights(
    train_seasons: np.ndarray,
    reliability: np.ndarray,
) -> np.ndarray:
    """Keep equal season totals and reliability-proportional row weights."""
    weights = np.zeros(len(train_seasons), dtype=np.float64)
    unique_seasons = np.unique(train_seasons)
    for season in unique_seasons:
        mask = train_seasons == season
        values = reliability[mask].astype(float)
        weights[mask] = values / values.sum()
    weights *= len(weights) / float(len(unique_seasons))
    return weights.astype(np.float32)


def importance(
    model: LGBMRegressor,
    feature_names: list[str],
) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in sorted(
            zip(feature_names, model.feature_importances_, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    }


def fit_one_expert(
    X: np.ndarray,
    residual_target: np.ndarray,
    seasons: np.ndarray,
    train_mask: np.ndarray,
    apply_mask: np.ndarray,
    validation_mask: np.ndarray,
    feature_names: list[str],
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    correction = np.zeros(int(validation_mask.sum()), dtype=np.float64)
    local_apply = apply_mask[validation_mask]
    diagnostic: dict[str, object] = {
        "training_rows": int(train_mask.sum()),
        "application_rows": int(apply_mask.sum()),
    }
    if not train_mask.any() or not apply_mask.any():
        diagnostic["fit_seconds"] = 0.0
        diagnostic["feature_importance"] = {}
        diagnostic["no_fit_reason"] = "empty training or application segment"
        return correction, diagnostic
    model = make_model()
    started = time.time()
    model.fit(
        X[train_mask],
        residual_target[train_mask],
        sample_weight=sample_weight,
    )
    correction[local_apply] = model.booster_.predict(X[apply_mask]).astype(
        float
    )
    diagnostic["fit_seconds"] = time.time() - started
    diagnostic["feature_importance"] = importance(model, feature_names)
    if sample_weight is not None:
        diagnostic["sample_weight_min"] = float(sample_weight.min())
        diagnostic["sample_weight_mean"] = float(sample_weight.mean())
        diagnostic["sample_weight_max"] = float(sample_weight.max())
    return correction, diagnostic


def fit_scheme(
    scheme: str,
    X: np.ndarray,
    residual_target: np.ndarray,
    seasons: np.ndarray,
    is_r: np.ndarray,
    high: np.ndarray,
    both_existing: np.ndarray,
    reliability_weight: np.ndarray,
    validation_season: int,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    validation_mask = seasons == validation_season
    past_r = (seasons < validation_season) & is_r
    validation_r = validation_mask & is_r
    if scheme == "high_both":
        train_mask = past_r & high & both_existing
        apply_mask = validation_r & high & both_existing
        weights = season_equal_weights(seasons[train_mask])
        return fit_one_expert(
            X,
            residual_target,
            seasons,
            train_mask,
            apply_mask,
            validation_mask,
            feature_names,
            weights,
        )
    if scheme == "reliability_weighted":
        train_mask = past_r
        apply_mask = validation_r
        weights = season_equal_reliability_weights(
            seasons[train_mask],
            reliability_weight[train_mask],
        )
        return fit_one_expert(
            X,
            residual_target,
            seasons,
            train_mask,
            apply_mask,
            validation_mask,
            feature_names,
            weights,
        )
    if scheme == "two_experts":
        correction = np.zeros(int(validation_mask.sum()), dtype=np.float64)
        diagnostics: dict[str, object] = {"experts": {}}
        for expert, segment in (("high", high), ("low", ~high)):
            train_mask = past_r & segment
            apply_mask = validation_r & segment
            weights = season_equal_weights(seasons[train_mask])
            current, current_diagnostic = fit_one_expert(
                X,
                residual_target,
                seasons,
                train_mask,
                apply_mask,
                validation_mask,
                feature_names,
                weights,
            )
            correction += current
            diagnostics["experts"][expert] = current_diagnostic
        diagnostics["training_rows"] = int(past_r.sum())
        diagnostics["application_rows"] = int(validation_r.sum())
        return correction, diagnostics
    raise ValueError(f"unknown scheme: {scheme}")


def reliability_segment_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    validation_types: np.ndarray,
    validation_high: np.ndarray,
    validation_both: np.ndarray,
) -> dict[str, dict[str, float]]:
    is_r = validation_types == "R"
    masks = {
        "R_all": is_r,
        "F_all": ~is_r,
        "R_high": is_r & validation_high,
        "R_low": is_r & ~validation_high,
        "R_both_existing": is_r & validation_both,
        "R_either_new": is_r & ~validation_both,
        "R_high_both_existing": is_r & validation_high & validation_both,
        "R_high_either_new": is_r & validation_high & ~validation_both,
        "R_low_both_existing": is_r & ~validation_high & validation_both,
        "R_low_either_new": is_r & ~validation_high & ~validation_both,
    }
    return {
        name: calculate_metrics(targets[mask], predictions[mask])
        for name, mask in masks.items()
        if mask.any()
    }


def candidate_name(scheme: str, weight: float) -> str:
    return f"{scheme}_w{int(round(weight * 100)):03d}"


def aggregate_folds(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for scheme in SCHEMES:
        for weight in BLEND_WEIGHTS:
            candidate = candidate_name(scheme, weight)
            skills = {
                season: float(
                    folds[str(season)]["candidates"][candidate][
                        "metrics"
                    ]["skill_score_unclipped"]
                )
                for season in REPORT_SEASONS
            }
            briers = {
                season: float(
                    folds[str(season)]["candidates"][candidate][
                        "metrics"
                    ]["brier_score"]
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
                "mean_skill": float(np.mean(list(skills.values()))),
                "min_skill": float(np.min(list(skills.values()))),
                "latest_2024_skill": skills[2024],
            }
    return aggregate


def main() -> None:
    started = time.time()
    frame, diagnostics, y, base, seasons, reconstruction = (
        multirate.prepare_multirate_data()
    )
    X, feature_names = build_full_feature_matrix(frame)
    group_all, group_reported = build_group_predictions(
        frame,
        y,
        base,
        seasons,
    )
    game_types = frame["game_type"].astype(str).to_numpy()
    is_r = game_types == "R"
    high = (
        frame["temporal_pitcher_season_n"].to_numpy(dtype=float)
        >= HIGH_PITCHER_N
    )
    both_existing = (
        frame["temporal_pitcher_prior_exists"].to_numpy(dtype=bool)
        & frame["temporal_batter_prior_exists"].to_numpy(dtype=bool)
    )
    pitcher_reliability = np.clip(
        frame["temporal_pitcher_reliability_30"].to_numpy(dtype=float),
        0.0,
        1.0,
    )
    batter_reliability = np.clip(
        frame["temporal_batter_reliability_30"].to_numpy(dtype=float),
        0.0,
        1.0,
    )
    reliability_weight = np.maximum(
        RELIABILITY_FLOOR,
        np.sqrt(pitcher_reliability * batter_reliability),
    )
    residual_target = (y.astype(float) - group_all).astype(np.float32)
    for season in np.unique(seasons):
        mask = (seasons == season) & is_r
        residual_target[mask] -= residual_target[mask].mean()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        validation_mask = seasons == validation_season
        targets = y[validation_mask].astype(float)
        validation_types = game_types[validation_mask]
        validation_high = high[validation_mask]
        validation_both = both_existing[validation_mask]
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "group_only": calculate_metrics(
                targets,
                group_reported[validation_season],
            ),
            "segment_rows": {
                "all": int(validation_mask.sum()),
                "R": int((validation_types == "R").sum()),
                "F": int((validation_types != "R").sum()),
                "R_high": int(
                    ((validation_types == "R") & validation_high).sum()
                ),
                "R_low": int(
                    ((validation_types == "R") & ~validation_high).sum()
                ),
                "R_high_both_existing": int(
                    (
                        (validation_types == "R")
                        & validation_high
                        & validation_both
                    ).sum()
                ),
            },
            "schemes": {},
            "candidates": {},
        }
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets)
        for scheme in SCHEMES:
            correction, scheme_diagnostic = fit_scheme(
                scheme,
                X,
                residual_target,
                seasons,
                is_r,
                high,
                both_existing,
                reliability_weight,
                validation_season,
                feature_names,
            )
            fold["schemes"][scheme] = scheme_diagnostic
            np.save(
                ARTIFACT_DIR
                / f"correction_{scheme}_{validation_season}.npy",
                correction,
            )
            for weight in BLEND_WEIGHTS:
                candidate = candidate_name(scheme, weight)
                predictions = np.clip(
                    group_reported[validation_season]
                    + weight * correction,
                    0.0,
                    1.0,
                )
                fold["candidates"][candidate] = {
                    "metrics": calculate_metrics(targets, predictions),
                    "segments": segment_metrics(
                        diagnostics,
                        validation_mask,
                        targets,
                        predictions,
                    ),
                    "reliability_segments": reliability_segment_metrics(
                        targets,
                        predictions,
                        validation_types,
                        validation_high,
                        validation_both,
                    ),
                }
                np.save(
                    ARTIFACT_DIR
                    / f"predictions_{candidate}_{validation_season}.npy",
                    predictions,
                )
            print(
                f"reliability_gated {validation_season} {scheme} done",
                flush=True,
            )
        folds[str(validation_season)] = fold
        print(
            f"reliability_gated {validation_season}: "
            + " ".join(
                f"{candidate}="
                f"{fold['candidates'][candidate]['metrics']['skill_score_unclipped']:.2f}"
                for candidate in fold["candidates"]
            ),
            flush=True,
        )

    aggregate = aggregate_folds(folds)
    best_candidate = max(
        aggregate,
        key=lambda candidate: (
            float(aggregate[candidate]["min_skill"]),
            float(aggregate[candidate]["latest_2024_skill"]),
            float(aggregate[candidate]["mean_skill"]),
        ),
    )
    team_eb = json.loads(TEAM_EB_METRICS.read_text(encoding="utf-8"))
    benchmark_min = float(
        team_eb["aggregate_2022_2024"][TEAM_EB_REFERENCE][
            "team_eb_min_skill"
        ]
    )
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": "reliability_gated_R_residual",
        "validation_protocol": {
            "outer_folds": list(VALIDATION_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "base": "temporal global30 plus past-only count/hand/reverse group",
            "R_model_training": "only prior-season R rows",
            "F_prediction": "group-only",
            "residual_centering": "inside source season and R regime",
            "season_weighting": "equal total weight per source season",
            "current_fold_labels_used_for_training": False,
            "test_row_aggregation": False,
            "candidate_comparison_status": "bounded predeclared diagnostic",
        },
        "schemes": {
            "high_both": {
                "train_apply": (
                    "R and temporal_pitcher_season_n >= 100 and both prior-exist"
                )
            },
            "reliability_weighted": {
                "train_apply": "all R",
                "row_weight": (
                    "max(0.1, sqrt(pitcher reliability30 * batter reliability30)); "
                    "renormalized to equal source-season totals"
                ),
            },
            "two_experts": {
                "train_apply": (
                    "separate R experts at temporal_pitcher_season_n >= 100"
                )
            },
        },
        "model": {
            "features": feature_names,
            "feature_count": len(feature_names),
            "iterations": ITERATIONS,
            "learning_rate": LEARNING_RATE,
            "num_leaves": NUM_LEAVES,
            "min_child_samples": MIN_CHILD_SAMPLES,
            "blend_weights": list(BLEND_WEIGHTS),
            "excluded": [
                "raw player IDs",
                "team IDs",
                "season",
                "validation/test-row aggregates",
            ],
        },
        "reconstruction_diagnostics": reconstruction,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "reference": {
            "team_eb_candidate": TEAM_EB_REFERENCE,
            "team_eb_min_skill": benchmark_min,
        },
        "selection": {
            "best_candidate": best_candidate,
            "best_min_skill": aggregate[best_candidate]["min_skill"],
            "best_latest_2024_skill": aggregate[best_candidate][
                "latest_2024_skill"
            ],
            "beats_team_eb_allprior_min": bool(
                float(aggregate[best_candidate]["min_skill"])
                > benchmark_min
            ),
            "status": "candidate comparison is non-nested",
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
