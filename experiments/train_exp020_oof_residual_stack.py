"""EXP-020: pooled temporal OOF residual stack on team all-prior.

The meta-model for validation season ``v`` sees only regular-season OOF rows
from evaluated seasons ``2021..v-1``.  Its target is ``y - team_allprior``
centered within each source season's R rows.  It uses a fixed stable numeric
official/temporal whitelist with no raw player IDs, team IDs, game type, or
season feature.  The correction is applied to R rows only; F rows retain the
team all-prior base exactly.

The configuration and three correction weights are predeclared.  Their OOF
comparison is still non-nested because this stack was proposed after earlier
experiment inspection.
"""

from __future__ import annotations

import gc
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from temporal_residual_features import attach_training_temporal_features
from train_exp017_rolling_residual import calculate_metrics
from train_exp019_stable_monotonic import (
    MONOTONE_DIRECTIONS,
    STRICT_FEATURES,
    season_equal_weights,
)


DATA_PATH = Path("./data/train.csv")
BASE_ROOT = Path("./artifacts/EXP-019/team_eb_ensemble")
BASE_METRICS_PATH = BASE_ROOT / "validation_metrics.json"
ARTIFACT_DIR = Path("./artifacts/EXP-020/oof_residual_stack")

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
CORRECTION_WEIGHTS = (0.25, 0.50, 0.75)

ITERATIONS = 200
LEARNING_RATE = 0.015
NUM_LEAVES = 7
MIN_CHILD_SAMPLES = 5000

RAW_COLUMNS = [
    "season",
    "game_type",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "control_success",
]


def add_required_static_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["count_index"] = (
        out["balls_before"] * 4 + out["strikes_before"]
    ).astype(np.int8)
    out["count_out_index"] = (
        out["count_index"] * 3 + out["outs_before"]
    ).astype(np.int8)
    out["is_full_count"] = (
        out["balls_before"].eq(3) & out["strikes_before"].eq(2)
    ).astype(np.int8)
    out["has_two_strikes"] = out["strikes_before"].eq(2).astype(np.int8)
    out["has_three_balls"] = out["balls_before"].eq(3).astype(np.int8)
    out["count_advantage"] = (
        out["strikes_before"] - out["balls_before"]
    ).astype(np.int8)
    out["runner_in_scoring_position"] = (
        out["runner_on_2b"].eq(1) | out["runner_on_3b"].eq(1)
    ).astype(np.int8)
    out["bases_loaded"] = (
        out["runner_on_1b"].eq(1)
        & out["runner_on_2b"].eq(1)
        & out["runner_on_3b"].eq(1)
    ).astype(np.int8)
    out["same_hand"] = out["pitcher_hand"].eq(out["batter_hand"]).astype(
        np.int8
    )
    return out


def prepare_features() -> tuple[
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    frame = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=RAW_COLUMNS,
    )
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("train rows must remain season sorted")
    frame = add_required_static_features(frame)
    frame, _ = attach_training_temporal_features(
        frame,
        target="control_success",
    )
    feature_names = sorted(STRICT_FEATURES)
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"missing stable features: {missing}")
    forbidden = {
        "season",
        "game_type",
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "batter_team_id",
    }
    leaked = sorted(forbidden.intersection(feature_names))
    if leaked:
        raise ValueError(f"forbidden meta features: {leaked}")

    X = frame[feature_names].to_numpy(dtype=np.float32)
    targets = frame["control_success"].to_numpy(dtype=np.float64)
    seasons = frame["season"].to_numpy(dtype=np.int16)
    game_types = frame["game_type"].astype(str).to_numpy()
    del frame
    gc.collect()
    return X, feature_names, targets, seasons, game_types


def load_base_predictions(
    targets: np.ndarray,
    seasons: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    target_oof: dict[int, np.ndarray] = {}
    base_oof: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        mask = seasons == season
        target_oof[season] = np.load(
            BASE_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        base_oof[season] = np.load(
            BASE_ROOT / f"predictions_all_prior_s1000_{season}.npy"
        ).astype(np.float64)
        if not np.array_equal(target_oof[season], targets[mask]):
            raise ValueError(f"target/order mismatch for season {season}")
        if len(base_oof[season]) != int(mask.sum()):
            raise ValueError(f"base length mismatch for season {season}")
        if not np.all(np.isfinite(base_oof[season])):
            raise ValueError(f"non-finite base for season {season}")
    return target_oof, base_oof


def build_centered_residuals(
    target_oof: dict[int, np.ndarray],
    base_oof: dict[int, np.ndarray],
    seasons: np.ndarray,
    game_types: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    outputs: dict[int, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        mask = seasons == season
        local_r = game_types[mask] == "R"
        residual = target_oof[season] - base_oof[season]
        centered = residual.copy()
        centered[local_r] -= float(np.mean(residual[local_r]))
        outputs[season] = centered
        diagnostics[str(season)] = {
            "R_rows": int(local_r.sum()),
            "raw_residual_mean_R": float(np.mean(residual[local_r])),
            "centered_residual_mean_R": float(
                np.mean(centered[local_r])
            ),
        }
    return outputs, diagnostics


def make_model(constraints: list[int]) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        metric="l2",
        n_estimators=ITERATIONS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        min_child_samples=MIN_CHILD_SAMPLES,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=12.0,
        monotone_constraints=constraints,
        monotone_constraints_method="advanced",
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def summarize_folds(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    base_skills = {
        season: float(folds[str(season)]["base"]["skill_score_unclipped"])
        for season in REPORT_SEASONS
    }
    base_briers = {
        season: float(folds[str(season)]["base"]["brier_score"])
        for season in REPORT_SEASONS
    }
    for weight in CORRECTION_WEIGHTS:
        candidate = f"stack_w{int(weight * 100):03d}"
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
            "season_skill_change_vs_base": {
                str(season): float(skills[season] - base_skills[season])
                for season in REPORT_SEASONS
            },
            "season_brier_change_vs_base": {
                str(season): float(briers[season] - base_briers[season])
                for season in REPORT_SEASONS
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
            "mean_skill_change_vs_base": float(
                np.mean(list(skills.values()))
                - np.mean(list(base_skills.values()))
            ),
            "min_skill_change_vs_base": float(
                np.min(list(skills.values()))
                - np.min(list(base_skills.values()))
            ),
            "season_calibration": {
                str(season): {
                    "mean_gap": float(folds[str(season)][candidate]["mean_gap"]),
                    "diagnostic_calibration_slope": float(
                        folds[str(season)][candidate][
                            "diagnostic_calibration_slope"
                        ]
                    ),
                    "diagnostic_calibration_intercept": float(
                        folds[str(season)][candidate][
                            "diagnostic_calibration_intercept"
                        ]
                    ),
                }
                for season in REPORT_SEASONS
            },
        }
    return aggregate


def main() -> None:
    started = time.time()
    X, feature_names, targets, seasons, game_types = prepare_features()
    target_oof, base_oof = load_base_predictions(targets, seasons)
    residual_oof, residual_centering_diagnostics = build_centered_residuals(
        target_oof,
        base_oof,
        seasons,
        game_types,
    )
    constraints = [MONOTONE_DIRECTIONS.get(name, 0) for name in feature_names]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    all_sources_strictly_prior = True
    f_correction_exactly_zero = True
    f_predictions_equal_base = True
    probabilities_bounded = True
    for validation_season in EVALUATED_SEASONS:
        validation_mask = seasons == validation_season
        validation_types = game_types[validation_mask]
        validation_r = validation_mask & (game_types == "R")
        local_r = validation_types == "R"
        correction = np.zeros(int(validation_mask.sum()), dtype=np.float64)
        source_seasons = [
            season
            for season in EVALUATED_SEASONS
            if season < validation_season
        ]
        all_sources_strictly_prior &= all(
            source < validation_season for source in source_seasons
        )
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
        }

        if source_seasons:
            train_mask = (
                np.isin(seasons, source_seasons) & (game_types == "R")
            )
            train_target = np.empty(int(train_mask.sum()), dtype=np.float64)
            train_seasons = seasons[train_mask]
            offset = 0
            for source_season in source_seasons:
                source_mask = seasons == source_season
                source_r = game_types[source_mask] == "R"
                values = residual_oof[source_season][source_r]
                local = train_seasons == source_season
                if int(local.sum()) != len(values):
                    raise AssertionError("source OOF row alignment failed")
                train_target[local] = values
                offset += len(values)
            if offset != len(train_target):
                raise AssertionError("source OOF residual count mismatch")

            model = make_model(constraints)
            fit_started = time.time()
            model.fit(
                X[train_mask],
                train_target,
                sample_weight=season_equal_weights(train_seasons),
            )
            fit_seconds = time.time() - fit_started
            correction[local_r] = model.booster_.predict(
                X[validation_r]
            ).astype(np.float64)
            train_prediction = model.booster_.predict(X[train_mask]).astype(
                np.float64
            )
            fold["meta_model"] = {
                "training_R_rows": int(train_mask.sum()),
                "validation_R_rows": int(validation_r.sum()),
                "fit_seconds": fit_seconds,
                "training_residual_mse_by_source": {
                    str(source_season): float(
                        np.mean(
                            (
                                train_target[train_seasons == source_season]
                                - train_prediction[
                                    train_seasons == source_season
                                ]
                            )
                            ** 2
                        )
                    )
                    for source_season in source_seasons
                },
                "feature_importance": {
                    name: int(value)
                    for name, value in sorted(
                        zip(feature_names, model.feature_importances_, strict=True),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
            }
            del model, train_prediction, train_target
            gc.collect()
        else:
            fold["meta_model"] = {
                "training_R_rows": 0,
                "validation_R_rows": int(validation_r.sum()),
                "fit_seconds": 0.0,
                "training_residual_mse_by_source": {},
                "feature_importance": {},
            }

        local_targets = target_oof[validation_season]
        local_base = base_oof[validation_season]
        fold["correction"] = {
            "mean_R": float(np.mean(correction[local_r])),
            "std_R": float(np.std(correction[local_r])),
            "min_R": float(np.min(correction[local_r])),
            "max_R": float(np.max(correction[local_r])),
            "F_rows_with_zero_correction": int(
                np.sum(correction[~local_r] == 0.0)
            ),
            "F_rows": int(np.sum(~local_r)),
        }
        f_correction_exactly_zero &= bool(
            np.all(correction[~local_r] == 0.0)
        )
        fold["base"] = calculate_metrics(local_targets, local_base)
        fold["regimes_base"] = {
            regime: calculate_metrics(
                local_targets[validation_types == regime],
                local_base[validation_types == regime],
            )
            for regime in sorted(np.unique(validation_types))
        }
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            local_targets,
        )
        np.save(
            ARTIFACT_DIR / f"predictions_base_{validation_season}.npy",
            local_base,
        )
        np.save(
            ARTIFACT_DIR / f"residual_correction_{validation_season}.npy",
            correction,
        )
        for weight in CORRECTION_WEIGHTS:
            candidate = f"stack_w{int(weight * 100):03d}"
            predictions = np.clip(
                local_base + weight * correction,
                0.0,
                1.0,
            )
            f_predictions_equal_base &= bool(
                np.array_equal(
                    predictions[~local_r],
                    local_base[~local_r],
                )
            )
            probabilities_bounded &= bool(
                np.all(np.isfinite(predictions))
                and float(np.min(predictions)) >= 0.0
                and float(np.max(predictions)) <= 1.0
            )
            fold[candidate] = calculate_metrics(local_targets, predictions)
            fold[f"regimes_{candidate}"] = {
                regime: calculate_metrics(
                    local_targets[validation_types == regime],
                    predictions[validation_types == regime],
                )
                for regime in sorted(np.unique(validation_types))
            }
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate}_{validation_season}.npy",
                predictions,
            )
        folds[str(validation_season)] = fold
        print(
            f"oof_stack {validation_season}: base="
            f"{fold['base']['skill_score_unclipped']:.2f} "
            + " ".join(
                f"w{int(weight * 100):03d}="
                f"{fold[f'stack_w{int(weight * 100):03d}']['skill_score_unclipped']:.2f}"
                for weight in CORRECTION_WEIGHTS
            ),
            flush=True,
        )

    aggregate = summarize_folds(folds)
    best_candidate = max(
        aggregate,
        key=lambda candidate: (
            aggregate[candidate]["min_skill"],
            aggregate[candidate]["latest_2024_skill"],
            aggregate[candidate]["mean_skill"],
        ),
    )
    base_skills = [
        float(folds[str(season)]["base"]["skill_score_unclipped"])
        for season in REPORT_SEASONS
    ]
    base_metrics = json.loads(BASE_METRICS_PATH.read_text(encoding="utf-8"))
    base_reference = base_metrics["aggregate_2022_2024"]["all_prior_s1000"]
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": "pooled_temporal_OOF_R_residual_stack",
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "base": "EXP-019 team all_prior_s1000 OOF",
            "meta_training": (
                "pooled R-only rows from evaluated OOF seasons 2021..v-1"
            ),
            "residual_target": (
                "y - team_allprior, centered within each source OOF season's R rows"
            ),
            "F_prediction": "team all-prior base unchanged",
            "current_fold_labels_used_for_meta_training": False,
            "test_row_aggregation": False,
            "candidate_comparison_non_nested": True,
        },
        "model": {
            "iterations": ITERATIONS,
            "learning_rate": LEARNING_RATE,
            "num_leaves": NUM_LEAVES,
            "min_child_samples": MIN_CHILD_SAMPLES,
            "correction_weights": list(CORRECTION_WEIGHTS),
            "features": feature_names,
            "feature_count": len(feature_names),
            "monotone_constraints": {
                name: constraint
                for name, constraint in zip(
                    feature_names,
                    constraints,
                    strict=True,
                )
                if constraint != 0
            },
            "excluded": [
                "season",
                "game_type",
                "pitcher_id",
                "batter_id",
                "pitcher_team_id",
                "batter_team_id",
            ],
        },
        "residual_centering_diagnostics": residual_centering_diagnostics,
        "oof_invariants": {
            "base_target_order_match_checked": True,
            "all_source_seasons_strictly_before_validation": bool(
                all_sources_strictly_prior
            ),
            "F_correction_exactly_zero": bool(f_correction_exactly_zero),
            "F_predictions_exactly_equal_base": bool(
                f_predictions_equal_base
            ),
            "all_candidate_probabilities_finite_and_bounded": bool(
                probabilities_bounded
            ),
            "raw_player_team_ids_and_season_excluded": True,
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "base_reference": {
            "source": str(BASE_METRICS_PATH),
            "variant": "all_prior_s1000",
            "season_skills": {
                str(season): float(
                    folds[str(season)]["base"]["skill_score_unclipped"]
                )
                for season in REPORT_SEASONS
            },
            "season_briers": {
                str(season): float(folds[str(season)]["base"]["brier_score"])
                for season in REPORT_SEASONS
            },
            "mean_skill": float(np.mean(base_skills)),
            "min_skill": float(np.min(base_skills)),
            "stored_reference_mean_skill": float(
                base_reference["team_eb_mean_skill"]
            ),
            "stored_reference_min_skill": float(
                base_reference["team_eb_min_skill"]
            ),
        },
        "selection": {
            "best_predeclared_weight": best_candidate,
            "best_mean_skill": float(aggregate[best_candidate]["mean_skill"]),
            "best_min_skill": float(aggregate[best_candidate]["min_skill"]),
            "base_min_skill": float(np.min(base_skills)),
            "beats_base_min": bool(
                aggregate[best_candidate]["min_skill"] > np.min(base_skills)
            ),
            "stop_rule_triggered": bool(
                aggregate[best_candidate]["min_skill"] <= np.min(base_skills)
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
        "total_seconds": time.time() - started,
    }
    metrics_path = ARTIFACT_DIR / "validation_metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"selection={result['selection']}", flush=True)
    print(f"saved={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
