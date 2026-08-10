"""EXP-021: row-independent recent-window denominator residual models.

The official recent-game success and middle rates are converted to a
current-row-only conservative denominator lower bound.  This experiment first
audits the denominator helper against a train-only game-boundary proxy, then
fits one shallow residual LightGBM per earlier OOF source season.  Source-model
corrections are averaged equally; current-fold labels, calibration, candidate
selection, and test-row aggregates are never used.

The four candidates are predeclared: a raw-rates control, denominator features
at two correction weights, and an R-only denominator correction.  The
immutable base is the saved strict-rank-s300 OOF prediction.  Rows outside a
candidate's application regime remain bitwise equal to that base.
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import recent_denominator_features as recent_den
from train_exp017_rolling_residual import calculate_metrics


DATA_PATH = Path("./data/train.csv")
BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
BASE_TEMPLATE = "predictions_strict_rank_s300_{season}.npy"
TARGET_TEMPLATE = "targets_{season}.npy"
ARTIFACT_DIR = Path("./artifacts/EXP-021/recent_denominator_residual")
EVALUATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
SOURCE_MODEL_SEASONS = (2021, 2022, 2023)
TARGET_SKILL = 1100.0
RAW_ROUNDING_TOLERANCE = 0.500001e-6

RATE_COLUMNS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
)
RECENT_RATE_COLUMNS = tuple(
    column
    for window in recent_den.WINDOWS
    for column in (window.success_column, window.middle_column)
)
TRAIN_COLUMNS = (
    "season",
    "game_month",
    "game_dayofweek",
    "game_type",
    "pitcher_id",
    "asof_pitcher_n",
    "asof_batter_n",
    *RATE_COLUMNS,
    "control_success",
)

MODEL_PARAMETERS = {
    "objective": "regression_l2",
    "metric": "l2",
    "n_estimators": 200,
    "learning_rate": 0.015,
    "num_leaves": 7,
    "min_child_samples": 3000,
    "max_bin": 127,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 12.0,
    "random_state": 42,
    "n_jobs": -1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    correction_family: str
    weight: float
    apply_to: str


CANDIDATES = (
    Candidate("rates_only_all_w025", "rates_all", 0.25, "all"),
    Candidate("denominator_all_w025", "denominator_all", 0.25, "all"),
    Candidate("denominator_all_w050", "denominator_all", 0.50, "all"),
    Candidate("denominator_R_only_w025", "denominator_R", 0.25, "R"),
)
CORRECTION_FAMILIES = ("rates_all", "denominator_all", "denominator_R")


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def denominator_feature_names() -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return recent_den.recent_denominator_feature_names()


def infer_raw_tolerance_minimum(
    success_rate: np.ndarray,
    middle_rate: np.ndarray,
    max_denominator: int,
) -> np.ndarray:
    """Independent comparator using raw CSV floats, not deployed semantics."""
    success = np.asarray(success_rate, dtype=float)
    middle = np.asarray(middle_rate, dtype=float)
    result = np.zeros(len(success), dtype=np.int16)
    for denominator in range(1, max_denominator + 1):
        success_error = np.abs(
            np.rint(success * denominator) / denominator - success
        )
        middle_error = np.abs(
            np.rint(middle * denominator) / denominator - middle
        )
        valid = (
            (result == 0)
            & (success_error <= RAW_ROUNDING_TOLERANCE)
            & (middle_error <= RAW_ROUNDING_TOLERANCE)
        )
        result[valid] = denominator
    return result


def denominator_boundary_proxy_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Train-only proxy audit; never used by the prediction model."""
    columns = [
        "pitcher_id",
        "asof_pitcher_n",
        "game_month",
        "game_dayofweek",
        *RECENT_RATE_COLUMNS,
    ]
    ordered = frame[list(dict.fromkeys(columns))].sort_values(
        ["pitcher_id", "asof_pitcher_n"], kind="stable"
    ).reset_index(drop=True)
    new_pitcher = ordered["pitcher_id"].ne(
        ordered["pitcher_id"].shift()
    )
    calendar_change = ordered[["game_month", "game_dayofweek"]].ne(
        ordered[["game_month", "game_dayofweek"]].shift()
    ).any(axis=1)
    recent = ordered[list(RECENT_RATE_COLUMNS)]
    recent_equal = (
        recent.eq(recent.shift())
        | (recent.isna() & recent.shift().isna())
    ).all(axis=1)
    boundary = new_pitcher | calendar_change | ~recent_equal
    boundary_frame = ordered.loc[boundary].copy()
    boundary_frame["game_n_proxy"] = boundary_frame.groupby(
        "pitcher_id", sort=False
    )["asof_pitcher_n"].diff()

    actual_windows: dict[str, pd.Series] = {}
    for window, width in zip(recent_den.WINDOWS, (1, 3, 5), strict=True):
        if width == 1:
            actual = boundary_frame["game_n_proxy"]
        else:
            actual = (
                boundary_frame.groupby("pitcher_id", sort=False)[
                    "game_n_proxy"
                ]
                .rolling(width, min_periods=width)
                .sum()
                .reset_index(level=0, drop=True)
                .sort_index()
            )
        actual_windows[window.name] = actual

    individual: dict[str, Any] = {}
    for window in recent_den.WINDOWS:
        actual = actual_windows[window.name]
        valid = (
            actual.between(1, window.max_denominator)
            & boundary_frame[window.success_column].notna()
            & boundary_frame[window.middle_column].notna()
        )
        actual_n = actual.loc[valid].to_numpy(dtype=int)
        success = boundary_frame.loc[valid, window.success_column].to_numpy(
            dtype=float
        )
        middle = boundary_frame.loc[valid, window.middle_column].to_numpy(
            dtype=float
        )
        deployed = recent_den.infer_minimum_common_denominator(
            success, middle, window.max_denominator
        )
        deployed_n = deployed["denominator"].astype(int)
        found = deployed["denominator_found"]
        raw_n = infer_raw_tolerance_minimum(
            success, middle, window.max_denominator
        ).astype(int)
        raw_actual_reconstructs = (
            np.abs(np.rint(success * actual_n) / actual_n - success)
            <= RAW_ROUNDING_TOLERANCE
        ) & (
            np.abs(np.rint(middle * actual_n) / actual_n - middle)
            <= RAW_ROUNDING_TOLERANCE
        )
        multiple = np.zeros(len(actual_n), dtype=bool)
        multiple[found] = actual_n[found] % deployed_n[found] == 0
        candidate_count = deployed["candidate_count"].astype(float)
        individual[window.name] = {
            "proxy_rows": int(len(actual_n)),
            "deployed_six_significant_digit_semantics": {
                "found_rows": int(found.sum()),
                "found_rate": float(found.mean()),
                "exact_rate_all_proxy_rows": float(
                    np.mean(deployed_n == actual_n)
                ),
                "exact_rate_conditioned_on_found": float(
                    np.mean(deployed_n[found] == actual_n[found])
                ),
                "actual_is_integer_multiple_conditioned_on_found": float(
                    multiple[found].mean()
                ),
                "mean_absolute_error_conditioned_on_found": float(
                    np.mean(np.abs(deployed_n[found] - actual_n[found]))
                ),
                "median_candidate_count_conditioned_on_found": float(
                    np.median(candidate_count[found])
                ),
                "unique_candidate_rate_conditioned_on_found": float(
                    np.mean(candidate_count[found] == 1)
                ),
            },
            "raw_float_tolerance_comparator": {
                "tolerance": RAW_ROUNDING_TOLERANCE,
                "actual_denominator_reconstructs_rate_pair": float(
                    raw_actual_reconstructs.mean()
                ),
                "minimum_exact_rate": float(np.mean(raw_n == actual_n)),
                "minimum_mean_absolute_error": float(
                    np.mean(np.abs(raw_n - actual_n))
                ),
            },
        }

    common_valid = np.ones(len(boundary_frame), dtype=bool)
    for window in recent_den.WINDOWS:
        actual = actual_windows[window.name]
        common_valid &= (
            actual.between(1, window.max_denominator).to_numpy()
            & boundary_frame[window.success_column].notna().to_numpy()
            & boundary_frame[window.middle_column].notna().to_numpy()
        )
    actual_matrix = np.column_stack(
        [
            actual_windows[window.name].loc[common_valid].to_numpy(dtype=int)
            for window in recent_den.WINDOWS
        ]
    )
    success_matrix = np.column_stack(
        [
            boundary_frame.loc[common_valid, window.success_column].to_numpy(
                dtype=float
            )
            for window in recent_den.WINDOWS
        ]
    )
    middle_matrix = np.column_stack(
        [
            boundary_frame.loc[common_valid, window.middle_column].to_numpy(
                dtype=float
            )
            for window in recent_den.WINDOWS
        ]
    )
    joint = recent_den.infer_joint_window_denominators(
        success_matrix, middle_matrix
    )
    joint_n = joint["denominators"].astype(int)
    joint_found = joint["joint_found"]
    joint_exact = joint_n[joint_found] == actual_matrix[joint_found]
    constraints = (
        (joint_n[joint_found, 1] >= joint_n[joint_found, 0] + 2)
        & (joint_n[joint_found, 2] >= joint_n[joint_found, 1] + 2)
    )
    success_counts = joint["success_counts"][joint_found]
    middle_counts = joint["middle_counts"][joint_found]
    count_constraints = (
        (success_counts[:, 1] >= success_counts[:, 0])
        & (success_counts[:, 2] >= success_counts[:, 1])
        & (middle_counts[:, 1] >= middle_counts[:, 0])
        & (middle_counts[:, 2] >= middle_counts[:, 1])
    )
    joint_output = {
        "common_proxy_rows": int(len(actual_matrix)),
        "joint_found_rows": int(joint_found.sum()),
        "joint_found_rate": float(joint_found.mean()),
        "constraints_hold_conditioned_on_found": bool(constraints.all()),
        "success_and_middle_count_nesting_conditioned_on_found": bool(
            count_constraints.all()
        ),
        "exact_rate_all_proxy_rows": {
            window.name: float(
                np.mean(joint_n[:, index] == actual_matrix[:, index])
            )
            for index, window in enumerate(recent_den.WINDOWS)
        },
        "exact_rate_conditioned_on_joint_found": {
            window.name: float(joint_exact[:, index].mean())
            for index, window in enumerate(recent_den.WINDOWS)
        },
        "all_three_exact_rate_conditioned_on_joint_found": float(
            joint_exact.all(axis=1).mean()
        ),
        "mean_absolute_error_conditioned_on_joint_found": {
            window.name: float(
                np.mean(
                    np.abs(
                        joint_n[joint_found, index]
                        - actual_matrix[joint_found, index]
                    )
                )
            )
            for index, window in enumerate(recent_den.WINDOWS)
        },
    }
    return {
        "status": "posthoc train-only helper audit; not used by model fitting",
        "proxy_caveat": (
            "train has no game_id/date; a proxy boundary is a new pitcher, "
            "a NaN-safe change in the six recent-rate values, or a change in "
            "the (game_month, game_dayofweek) pair after stable sorting by "
            "(pitcher_id, asof_pitcher_n)"
        ),
        "boundary_rows": int(len(boundary_frame)),
        "individual_windows": individual,
        "joint_windows": joint_output,
    }


def build_recent_features(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    recent_input = frame[list(RATE_COLUMNS)].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        attached, diagnostics = recent_den.attach_recent_denominator_features(
            recent_input
        )
    denominator_names = denominator_feature_names()
    individual_names = [
        name for name in denominator_names if "recent_den_joint" not in name
    ]
    joint_names = [
        name for name in denominator_names if "recent_den_joint" in name
    ]
    if not (
        len(denominator_names) == 114
        and len(individual_names) == 62
        and len(joint_names) == 52
    ):
        raise AssertionError("unexpected denominator feature schema")
    denominator_model_names = list(RATE_COLUMNS) + denominator_names
    X_rates = np.ascontiguousarray(
        attached[list(RATE_COLUMNS)].to_numpy(dtype=np.float32)
    )
    X_denominator = np.ascontiguousarray(
        attached[denominator_model_names].to_numpy(dtype=np.float32)
    )

    sample_positions = np.linspace(
        0, len(recent_input) - 1, num=16, dtype=int
    )
    sample = recent_input.iloc[sample_positions].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        batch, _ = recent_den.attach_recent_denominator_features(sample)
        singleton_rows = [
            recent_den.attach_recent_denominator_features(
                sample.iloc[[index]].reset_index(drop=True)
            )[0][denominator_names]
            for index in range(len(sample))
        ]
        singleton = pd.concat(singleton_rows, ignore_index=True)
        permutation = np.random.default_rng(20210810).permutation(len(sample))
        permuted, _ = recent_den.attach_recent_denominator_features(
            sample.iloc[permutation].reset_index(drop=True)
        )
    inverse_permutation = np.argsort(permutation)
    batch_values = batch[denominator_names].to_numpy(dtype=float)
    singleton_values = singleton[denominator_names].to_numpy(dtype=float)
    permuted_values = permuted[denominator_names].to_numpy(dtype=float)[
        inverse_permutation
    ]
    batch_singleton_equal = bool(
        np.allclose(batch_values, singleton_values, atol=0.0, rtol=0.0)
    )
    permutation_equal = bool(
        np.allclose(batch_values, permuted_values, atol=0.0, rtol=0.0)
    )
    denominator_values = attached[denominator_names].to_numpy(dtype=float)
    all_finite = bool(np.isfinite(denominator_values).all())
    reliability_columns = [
        index
        for index, name in enumerate(denominator_names)
        if "_reliability_" in name or name.endswith("_inverse_ambiguity")
    ]
    reliability_in_range = bool(
        (
            (denominator_values[:, reliability_columns] >= 0.0)
            & (denominator_values[:, reliability_columns] <= 1.0)
        ).all()
    )
    joint_found = attached["recent_den_joint_found"].to_numpy(dtype=bool)
    joint_n = np.column_stack(
        [
            attached[f"recent_den_joint_{window.name}_n"].to_numpy(dtype=float)
            for window in recent_den.WINDOWS
        ]
    )
    joint_constraints = bool(
        (
            (joint_n[joint_found, 1] >= joint_n[joint_found, 0] + 2)
            & (joint_n[joint_found, 2] >= joint_n[joint_found, 1] + 2)
        ).all()
    )
    joint_success = np.column_stack(
        [
            attached[
                f"recent_den_joint_{window.name}_success_count"
            ].to_numpy(dtype=float)
            for window in recent_den.WINDOWS
        ]
    )
    joint_middle = np.column_stack(
        [
            attached[
                f"recent_den_joint_{window.name}_middle_count"
            ].to_numpy(dtype=float)
            for window in recent_den.WINDOWS
        ]
    )
    joint_count_constraints = bool(
        (
            (joint_success[joint_found, 1] >= joint_success[joint_found, 0])
            & (joint_success[joint_found, 2] >= joint_success[joint_found, 1])
            & (joint_middle[joint_found, 1] >= joint_middle[joint_found, 0])
            & (joint_middle[joint_found, 2] >= joint_middle[joint_found, 1])
        ).all()
    )
    synthetic_success = np.array([[0.55, 0.55, 0.55]], dtype=float)
    synthetic_middle = np.array([[0.10, 0.10, 0.10]], dtype=float)
    synthetic_individual = recent_den.infer_minimum_common_denominator(
        synthetic_success[:, 0], synthetic_middle[:, 0], 180
    )["denominator"]
    synthetic_joint = recent_den.infer_joint_window_denominators(
        synthetic_success, synthetic_middle
    )["denominators"]
    synthetic_expected = bool(
        synthetic_individual.tolist() == [20]
        and synthetic_joint.tolist() == [[20, 40, 60]]
    )
    audit = {
        "helper_module_path": str(Path(recent_den.__file__).resolve()),
        "helper_module_sha256": hashlib.sha256(
            Path(recent_den.__file__).read_bytes()
        ).hexdigest(),
        "authoritative_rounding_semantics": (
            "exact float equality after formatting reconstructed rates to "
            "six significant digits"
        ),
        "feature_count": len(denominator_names),
        "original_individual_feature_count": len(individual_names),
        "joint_extension_feature_count": len(joint_names),
        "feature_names": denominator_names,
        "module_diagnostics": diagnostics,
        "batch_equals_singleton_exactly": batch_singleton_equal,
        "permutation_invariant_exactly": permutation_equal,
        "all_generated_features_finite": all_finite,
        "reliability_and_inverse_ambiguity_in_0_1": reliability_in_range,
        "joint_constraints_hold": joint_constraints,
        "joint_success_middle_count_nesting_holds": (
            joint_count_constraints
        ),
        "synthetic_0p55_0p10_individual20_joint20_40_60": synthetic_expected,
        "row_independent_invariant_passed": bool(
            batch_singleton_equal
            and permutation_equal
            and all_finite
            and reliability_in_range
            and joint_constraints
            and joint_count_constraints
            and synthetic_expected
        ),
    }
    if not audit["row_independent_invariant_passed"]:
        raise AssertionError("recent denominator row-independent audit failed")
    del attached, denominator_values
    gc.collect()
    return X_rates, X_denominator, denominator_model_names, audit


def load_base_oof(
    y: np.ndarray,
    seasons: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    output: dict[int, np.ndarray] = {}
    qa: dict[str, Any] = {}
    for season in EVALUATION_SEASONS:
        mask = seasons == season
        predictions_path = BASE_ROOT / BASE_TEMPLATE.format(season=season)
        targets_path = BASE_ROOT / TARGET_TEMPLATE.format(season=season)
        prediction = np.load(predictions_path).astype(float)
        targets = np.load(targets_path).astype(np.int8)
        current_targets = y[mask].astype(np.int8)
        valid = bool(
            prediction.shape == targets.shape == current_targets.shape
            and np.array_equal(targets, current_targets)
            and np.isfinite(prediction).all()
            and ((prediction >= 0.0) & (prediction <= 1.0)).all()
        )
        if not valid:
            raise ValueError(f"base OOF parity failure: {season}")
        output[season] = prediction
        qa[str(season)] = {
            "rows": int(len(prediction)),
            "prediction_path": str(predictions_path),
            "target_path": str(targets_path),
            "target_and_order_parity": True,
            "finite_and_in_0_1": True,
        }
    return output, qa


def make_model() -> LGBMRegressor:
    return LGBMRegressor(**MODEL_PARAMETERS)


def train_source_models_and_accumulate(
    X_rates: np.ndarray,
    X_denominator: np.ndarray,
    y: np.ndarray,
    seasons: np.ndarray,
    game_types: np.ndarray,
    base_oof: dict[int, np.ndarray],
    denominator_feature_names: list[str],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    corrections = {
        family: {
            season: np.zeros(int((seasons == season).sum()), dtype=float)
            for season in EVALUATION_SEASONS
        }
        for family in CORRECTION_FAMILIES
    }
    source_counts = {
        family: {season: 0 for season in EVALUATION_SEASONS}
        for family in CORRECTION_FAMILIES
    }
    diagnostics: dict[str, Any] = {}
    feature_names = {
        "rates_all": list(RATE_COLUMNS),
        "denominator_all": denominator_feature_names,
        "denominator_R": denominator_feature_names,
    }
    matrices = {
        "rates_all": X_rates,
        "denominator_all": X_denominator,
        "denominator_R": X_denominator,
    }
    for source_season in SOURCE_MODEL_SEASONS:
        source_global = seasons == source_season
        source_types = game_types[source_global]
        residual = y[source_global].astype(float) - base_oof[source_season]
        source_output: dict[str, Any] = {}
        for family in CORRECTION_FAMILIES:
            family_started = time.time()
            source_apply = (
                np.ones(int(source_global.sum()), dtype=bool)
                if family != "denominator_R"
                else source_types == "R"
            )
            centered_target = residual[source_apply].copy()
            residual_mean_before = float(centered_target.mean())
            centered_target -= residual_mean_before
            source_rows = np.flatnonzero(source_global)[source_apply]
            model = make_model()
            model.fit(matrices[family][source_rows], centered_target)
            training_prediction = model.booster_.predict(
                matrices[family][source_rows]
            ).astype(float)
            prediction_center = float(training_prediction.mean())
            future_diagnostics: dict[str, Any] = {}
            for validation_season in EVALUATION_SEASONS:
                if validation_season <= source_season:
                    continue
                validation_global = seasons == validation_season
                validation_types = game_types[validation_global]
                validation_apply = (
                    np.ones(int(validation_global.sum()), dtype=bool)
                    if family != "denominator_R"
                    else validation_types == "R"
                )
                validation_rows = np.flatnonzero(validation_global)[
                    validation_apply
                ]
                source_correction = model.booster_.predict(
                    matrices[family][validation_rows]
                ).astype(float) - prediction_center
                corrections[family][validation_season][
                    validation_apply
                ] += source_correction
                source_counts[family][validation_season] += 1
                future_diagnostics[str(validation_season)] = {
                    "applied_rows": int(validation_apply.sum()),
                    "correction_mean": float(source_correction.mean()),
                    "correction_std": float(source_correction.std()),
                    "correction_min": float(source_correction.min()),
                    "correction_max": float(source_correction.max()),
                }
            source_output[family] = {
                "fit_rows": int(len(source_rows)),
                "fit_regime": "R" if family == "denominator_R" else "all",
                "residual_mean_before_source_centering": residual_mean_before,
                "residual_mean_after_source_centering": float(
                    centered_target.mean()
                ),
                "training_prediction_mean_removed": prediction_center,
                "model_parameters": MODEL_PARAMETERS,
                "feature_count": len(feature_names[family]),
                "feature_importance": {
                    name: int(value)
                    for name, value in sorted(
                        zip(
                            feature_names[family],
                            model.feature_importances_,
                            strict=True,
                        ),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
                "future_predictions": future_diagnostics,
                "fit_seconds": time.time() - family_started,
            }
            del model, training_prediction, centered_target
            gc.collect()
        diagnostics[str(source_season)] = source_output
    for family in CORRECTION_FAMILIES:
        for season in EVALUATION_SEASONS:
            count = source_counts[family][season]
            expected = len(
                [source for source in SOURCE_MODEL_SEASONS if source < season]
            )
            if count != expected:
                raise AssertionError(
                    f"source-model count mismatch: {family} {season}"
                )
            if count > 0:
                corrections[family][season] /= count
    return corrections, {
        "source_models": diagnostics,
        "equal_average_source_model_counts": source_counts,
    }


def regime_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    game_types: np.ndarray,
) -> dict[str, dict[str, float]]:
    output = {"full": calculate_metrics(targets, predictions)}
    for regime in ("F", "R"):
        local = game_types == regime
        output[regime] = calculate_metrics(targets[local], predictions[local])
    threshold = float(
        output["full"]["baseline_brier"]
        * (1.0 - TARGET_SKILL / 100000.0)
    )
    output["full"]["skill_1100_brier_threshold"] = threshold
    output["full"]["brier_minus_skill_1100_threshold"] = float(
        output["full"]["brier_score"] - threshold
    )
    return output


def build_segment_labels(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    pitcher_n = frame["asof_pitcher_n"].to_numpy(dtype=float)
    batter_n = frame["asof_batter_n"].to_numpy(dtype=float)
    n_label = np.select(
        [
            pitcher_n == 0,
            (pitcher_n >= 1) & (pitcher_n < 20),
            (pitcher_n >= 20) & (pitcher_n < 100),
            (pitcher_n >= 100) & (pitcher_n < 500),
        ],
        ["n_0", "n_1_19", "n_20_99", "n_100_499"],
        default="n_500_plus",
    ).astype(str)
    new_label = np.select(
        [
            (pitcher_n > 0) & (batter_n > 0),
            (pitcher_n == 0) & (batter_n > 0),
            (pitcher_n > 0) & (batter_n == 0),
        ],
        ["both_seen", "pitcher_new_only", "batter_new_only"],
        default="both_new",
    ).astype(str)
    return {"pitcher_asof_n": n_label, "new_player": new_label}


def safe_segment_metric(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    if len(targets) == 0:
        return {"rows": 0, "status": "empty_segment"}
    actual_rate = float(targets.mean())
    if 0.0 < actual_rate < 1.0:
        return calculate_metrics(targets, predictions)
    brier = float(np.mean((predictions.astype(float) - targets) ** 2))
    design = np.column_stack([predictions, np.ones_like(predictions)])
    slope, intercept = np.linalg.lstsq(design, targets, rcond=None)[0]
    return {
        "rows": int(len(targets)),
        "actual_rate": actual_rate,
        "prediction_mean": float(predictions.mean()),
        "mean_gap": float(predictions.mean() - actual_rate),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "brier_score": brier,
        "baseline_brier": 0.0,
        "skill_score": None,
        "skill_score_unclipped": None,
        "diagnostic_calibration_slope": float(slope),
        "diagnostic_calibration_intercept": float(intercept),
        "degenerate_all_same_target_segment": True,
    }


def segment_metrics(
    labels: dict[str, np.ndarray],
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dimension, values in labels.items():
        output[dimension] = {
            label: safe_segment_metric(
                targets[values == label], predictions[values == label]
            )
            for label in sorted(np.unique(values).tolist())
        }
    return output


def denominator_ambiguity_by_season(
    denominator_features: np.ndarray,
    denominator_feature_names: list[str],
    seasons: np.ndarray,
) -> dict[str, Any]:
    index = {
        name: position
        for position, name in enumerate(denominator_feature_names)
    }
    output: dict[str, Any] = {}
    for season in EVALUATION_SEASONS:
        local = seasons == season
        season_output: dict[str, Any] = {}
        for window in recent_den.WINDOWS:
            found = denominator_features[
                local, index[f"recent_den_{window.name}_found"]
            ].astype(bool)
            denominator = denominator_features[
                local, index[f"recent_den_{window.name}_min_n"]
            ]
            candidates = denominator_features[
                local, index[f"recent_den_{window.name}_candidate_count"]
            ]
            season_output[window.name] = {
                "rows": int(local.sum()),
                "found_rate": float(found.mean()),
                "median_min_n_conditioned_on_found": float(
                    np.median(denominator[found]) if found.any() else 0.0
                ),
                "median_candidate_count_conditioned_on_found": float(
                    np.median(candidates[found]) if found.any() else 0.0
                ),
                "unique_candidate_rate_conditioned_on_found": float(
                    np.mean(candidates[found] == 1) if found.any() else 0.0
                ),
            }
        joint_found = denominator_features[
            local, index["recent_den_joint_found"]
        ].astype(bool)
        season_output["joint"] = {
            "rows": int(local.sum()),
            "joint_found_rate": float(joint_found.mean()),
            "prev3_adjusted_rate_among_joint_found": float(
                denominator_features[
                    local,
                    index["recent_den_joint_prev3_adjusted_from_individual"],
                ][joint_found].mean()
                if joint_found.any()
                else 0.0
            ),
            "prev5_adjusted_rate_among_joint_found": float(
                denominator_features[
                    local,
                    index["recent_den_joint_prev5_adjusted_from_individual"],
                ][joint_found].mean()
                if joint_found.any()
                else 0.0
            ),
        }
        output[str(season)] = season_output
    return output


def aggregate_metrics(
    folds: dict[int, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for regime in ("full", "F", "R"):
        skills = {
            str(season): float(folds[season][regime]["skill_score_unclipped"])
            for season in REPORT_SEASONS
        }
        briers = {
            str(season): float(folds[season][regime]["brier_score"])
            for season in REPORT_SEASONS
        }
        output[regime] = {
            "season_skills": skills,
            "season_briers": briers,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": float(skills["2024"]),
            "mean_brier": float(np.mean(list(briers.values()))),
        }
    return output


def main() -> None:
    started = time.time()
    frame = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=list(dict.fromkeys(TRAIN_COLUMNS)),
    )
    y = frame["control_success"].to_numpy(dtype=np.float32)
    seasons = frame["season"].to_numpy(dtype=np.int16)
    game_types = frame["game_type"].astype(str).to_numpy()
    if not set(np.unique(game_types)).issubset({"F", "R"}):
        raise ValueError("unexpected game_type")
    boundary_audit = denominator_boundary_proxy_audit(frame)
    X_rates, X_denominator, denominator_model_names, helper_audit = (
        build_recent_features(frame)
    )
    denominator_names = denominator_feature_names()
    ambiguity = denominator_ambiguity_by_season(
        X_denominator[:, len(RATE_COLUMNS) :],
        denominator_names,
        seasons,
    )
    labels = build_segment_labels(frame)
    base_oof, base_qa = load_base_oof(y, seasons)
    corrections, source_model_diagnostics = train_source_models_and_accumulate(
        X_rates,
        X_denominator,
        y,
        seasons,
        game_types,
        base_oof,
        denominator_model_names,
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, Any] = {}
    candidate_fold_metrics: dict[str, dict[int, dict[str, Any]]] = {
        candidate.name: {} for candidate in CANDIDATES
    }
    base_fold_metrics: dict[int, dict[str, Any]] = {}
    all_nonapplied_exact = True

    for season in EVALUATION_SEASONS:
        mask = seasons == season
        targets = y[mask].astype(float)
        local_types = game_types[mask]
        local_labels = {
            name: values[mask] for name, values in labels.items()
        }
        base = base_oof[season]
        base_metrics = regime_metrics(targets, base, local_types)
        base_fold_metrics[season] = base_metrics
        fold: dict[str, Any] = {
            "validation_season": season,
            "source_model_seasons": [
                source for source in SOURCE_MODEL_SEASONS if source < season
            ],
            "base": {
                "metrics": base_metrics,
                "segments": segment_metrics(
                    local_labels, targets, base
                ),
            },
            "corrections": {},
            "candidates": {},
        }
        for family in CORRECTION_FAMILIES:
            correction = corrections[family][season]
            fold["corrections"][family] = {
                "mean": float(correction.mean()),
                "std": float(correction.std()),
                "min": float(correction.min()),
                "max": float(correction.max()),
                "source_model_count": len(fold["source_model_seasons"]),
            }
            np.save(
                ARTIFACT_DIR / f"correction_{family}_{season}.npy",
                correction,
            )
        for candidate in CANDIDATES:
            correction = corrections[candidate.correction_family][season]
            apply_mask = (
                np.ones(len(targets), dtype=bool)
                if candidate.apply_to == "all"
                else local_types == candidate.apply_to
            )
            prediction = base.copy()
            prediction[apply_mask] = np.clip(
                base[apply_mask] + candidate.weight * correction[apply_mask],
                0.0,
                1.0,
            )
            nonapplied_exact = bool(
                np.array_equal(prediction[~apply_mask], base[~apply_mask])
            )
            all_nonapplied_exact &= nonapplied_exact
            if not (
                np.isfinite(prediction).all()
                and ((prediction >= 0.0) & (prediction <= 1.0)).all()
            ):
                raise ValueError("invalid candidate prediction")
            metrics = regime_metrics(targets, prediction, local_types)
            candidate_fold_metrics[candidate.name][season] = metrics
            fold["candidates"][candidate.name] = {
                "correction_family": candidate.correction_family,
                "weight": candidate.weight,
                "apply_to": candidate.apply_to,
                "applied_rows": int(apply_mask.sum()),
                "nonapplied_rows": int((~apply_mask).sum()),
                "nonapplied_rows_bitwise_exact_base": nonapplied_exact,
                "metrics": metrics,
                "segments": segment_metrics(
                    local_labels, targets, prediction
                ),
            }
            np.save(
                ARTIFACT_DIR / f"predictions_{candidate.name}_{season}.npy",
                prediction,
            )
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", targets)
        folds[str(season)] = fold
        print(
            f"EXP-021 denominator {season}: base="
            f"{base_metrics['full']['skill_score_unclipped']:.2f} "
            + " ".join(
                f"{candidate.name}="
                f"{fold['candidates'][candidate.name]['metrics']['full']['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    aggregate_candidates = {
        candidate.name: aggregate_metrics(
            candidate_fold_metrics[candidate.name]
        )
        for candidate in CANDIDATES
    }
    base_aggregate = aggregate_metrics(base_fold_metrics)
    best_min = max(
        float(value["full"]["min_skill"])
        for value in aggregate_candidates.values()
    )
    uniform_gate_passed = bool(best_min >= TARGET_SKILL)
    result: dict[str, Any] = {
        "experiment": "EXP-021",
        "candidate_family": "recent_minimum_denominator_residual",
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATION_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": "EXP-020 strict_rank_s300 OOF",
            "source_models": (
                "one shallow residual LightGBM per earlier OOF source season"
            ),
            "source_residual_centering": (
                "inside each source season and source model regime"
            ),
            "source_model_prediction_centering": (
                "source-training prediction mean removed before transfer"
            ),
            "source_model_combination": "equal mean across earlier sources",
            "current_fold_labels_used_for_fit": False,
            "current_fold_labels_used_for_selection": False,
            "candidate_selection_or_calibration": False,
            "candidate_count": len(CANDIDATES),
            "candidate_grid_predeclared": True,
            "test_csv_read": False,
            "test_row_aggregation": False,
            "current_validation_row_aggregation_for_prediction": False,
            "raw_player_or_team_ID_model_features": False,
        },
        "model": {
            "parameters": MODEL_PARAMETERS,
            "rates_only_features": list(RATE_COLUMNS),
            "denominator_model_features": denominator_model_names,
            "denominator_feature_count": len(denominator_names),
            "denominator_schema_note": (
                "62 original independent-window features plus 52 jointly "
                "constrained current-row features"
            ),
            "candidate_definitions": [
                {
                    "name": candidate.name,
                    "correction_family": candidate.correction_family,
                    "weight": candidate.weight,
                    "apply_to": candidate.apply_to,
                }
                for candidate in CANDIDATES
            ],
        },
        "denominator_diagnostics": {
            "helper_invariant_audit": helper_audit,
            "train_game_boundary_proxy_recovery": boundary_audit,
            "season_ambiguity": ambiguity,
        },
        "source_model_diagnostics": source_model_diagnostics,
        "folds": folds,
        "aggregate_2022_2024": {
            "strict_rank_s300_base": base_aggregate,
            "candidates": aggregate_candidates,
        },
        "decision": {
            "target_minimum_skill": TARGET_SKILL,
            "best_candidate_min_skill": best_min,
            "uniform_1100_gate_passed": uniform_gate_passed,
            "status": (
                "continue_only_if_uniform_gate_passed"
                if uniform_gate_passed
                else "stopped_below_1100_gate"
            ),
            "selection_status": (
                "no candidate selected; all metrics are bounded diagnostics"
            ),
        },
        "qa": {
            "base_oof": base_qa,
            "candidate_count_at_most_4": len(CANDIDATES) <= 4,
            "all_nonapplied_rows_bitwise_exact_base": all_nonapplied_exact,
            "all_predictions_finite_and_in_0_1": True,
            "all_saved_targets_match_training_order": True,
            "source_seasons_strictly_before_validation": bool(
                all(
                    all(source < int(season) for source in fold["source_model_seasons"])
                    for season, fold in folds.items()
                )
            ),
            "row_independent_feature_invariant": helper_audit[
                "row_independent_invariant_passed"
            ],
            "no_test_rows_loaded": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
        "total_seconds": time.time() - started,
    }
    output_path = ARTIFACT_DIR / "validation_metrics.json"
    output_path.write_text(
        json.dumps(to_builtin(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "base_full_skills": base_aggregate["full"]["season_skills"],
                "candidate_full_skills": {
                    name: value["full"]["season_skills"]
                    for name, value in aggregate_candidates.items()
                },
                "candidate_min_skills": {
                    name: value["full"]["min_skill"]
                    for name, value in aggregate_candidates.items()
                },
                "decision": result["decision"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
