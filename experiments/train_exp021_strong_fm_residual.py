"""EXP-021: strongly regularized sparse residual FM on regular-season rows.

PyTorch is intentionally not a dependency of this repository.  This script
therefore implements the requested small embedding model with deterministic
NumPy full-batch Adam.  The deployable correction model is

    pitcher_bias + batter_bias + context_bias + official_state @ beta
    + <pitcher_embedding, pitcher_context_embedding>
    + <batter_embedding, batter_context_embedding>.

The immutable base is the temporal ``strict_rank_s300`` OOF prediction.  Raw
team IDs and season are excluded.  Context is a fixed current-row domain of
count, pitcher/batter hand, outs, and coarse runner state.  Official state
terms are row-local and source-only standardized.  A validation ID absent
from all source seasons receives zero ID bias/interaction while still using
the fixed context and official continuous state terms.

Only past regular-season (game_type == ``R``) rows train the residual model.
Each source-season R residual is centered separately and every source season
has equal total optimization weight.  Corrections are applied only to current
row R observations; F predictions remain bitwise identical to the base.

Official sample-size reliability gates player biases by ``n/(n+300)`` and
player factors by its square root, making every player-ID contribution zero
when the official current-row sample size is zero.

Exactly four candidates are predeclared: embedding dimension 2/4 crossed
with correction weights 0.25/0.50.  All share one deliberately strong L2
configuration and a +/-0.03 correction clip.  Epoch selection uses only the
latest available *past* inner season: for outer 2023, 2021 trains and 2022
selects the epoch; for outer 2024, 2021-2022 train and 2023 selects it.  Outer
2022 has only one source season and uses the fixed default epoch count.
Current-fold labels never fit or select a model.  test.csv is never read and
no validation/test-row aggregate is used by prediction.
"""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics


DATA_PATH = Path("./data/train.csv")
BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
TARGET_ROOT = Path("./artifacts/EXP-019/team_eb_ensemble")
ARTIFACT_DIR = Path("./artifacts/EXP-021/strong_fm_residual")

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
EMBEDDING_DIMS = (2, 4)
CORRECTION_WEIGHTS = (0.25, 0.50)
CORRECTION_CLIP = 0.03

DEFAULT_EPOCHS = 12
MAX_INNER_EPOCHS = 25
INNER_PATIENCE = 5
LEARNING_RATE = 0.001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
BIAS_L2 = 0.02
LINEAR_L2 = 0.05
EMBEDDING_L2 = 0.05
PARAMETER_CLIP = 0.10
RANDOM_SEED = 42

CONTEXT_COUNT = 12 * 2 * 2 * 3 * 3

RAW_COLUMNS = (
    "season",
    "game_type",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "balls_before",
    "strikes_before",
    "outs_before",
    "num_runners_on",
    "inning",
    "score_diff_pitcher_team",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
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
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "control_success",
)

CONTINUOUS_FEATURES = (
    "log1p_pitcher_n",
    "pitcher_reliability30",
    "asof_pitcher_success_rate",
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
    "log1p_batter_n",
    "batter_reliability30",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "inning_scaled",
    "score_diff_clipped",
    "log1p_li",
)

BASE_CANDIDATE = "strict_rank_s300_base"


@dataclass(frozen=True)
class Candidate:
    name: str
    embedding_dim: int
    correction_weight: float


CANDIDATES = tuple(
    Candidate(
        name=f"fm_d{dimension}_w{int(weight * 100):03d}",
        embedding_dim=dimension,
        correction_weight=weight,
    )
    for dimension in EMBEDDING_DIMS
    for weight in CORRECTION_WEIGHTS
)


@dataclass
class EncodedData:
    pitcher: np.ndarray
    batter: np.ndarray
    context: np.ndarray
    continuous: np.ndarray
    pitcher_reliability: np.ndarray
    batter_reliability: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    seasons: np.ndarray


def load_rows() -> dict[int, pd.DataFrame]:
    frame = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=list(RAW_COLUMNS),
    )
    frame = frame.loc[
        frame["season"].isin(EVALUATED_SEASONS)
    ].reset_index(drop=True)
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("evaluated rows must remain season sorted")
    required = [
        column
        for column in RAW_COLUMNS
        if column
        not in {
            "asof_pitcher_success_rate",
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
            "asof_batter_success_rate",
            "asof_batter_middle_rate",
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
        }
    ]
    if frame[required].isna().any().any():
        raise ValueError("missing required current-row field")
    if set(frame["game_type"].astype(str).unique()) != {"F", "R"}:
        raise ValueError("unexpected game_type domain")
    if not set(frame["pitcher_hand"].astype(int).unique()).issubset({1, 2}):
        raise ValueError("unexpected pitcher hand")
    if not set(frame["batter_hand"].astype(int).unique()).issubset({1, 2}):
        raise ValueError("unexpected batter hand")
    if not set(frame["balls_before"].astype(int).unique()).issubset(range(4)):
        raise ValueError("unexpected balls")
    if not set(frame["strikes_before"].astype(int).unique()).issubset(range(3)):
        raise ValueError("unexpected strikes")
    if not set(frame["outs_before"].astype(int).unique()).issubset(range(3)):
        raise ValueError("unexpected outs")

    count_position = (
        frame["balls_before"].to_numpy(dtype=np.int16) * 3
        + frame["strikes_before"].to_numpy(dtype=np.int16)
    )
    pitcher_hand = frame["pitcher_hand"].to_numpy(dtype=np.int16) - 1
    batter_hand = frame["batter_hand"].to_numpy(dtype=np.int16) - 1
    outs = frame["outs_before"].to_numpy(dtype=np.int16)
    runner_coarse = np.minimum(
        frame["num_runners_on"].to_numpy(dtype=np.int16), 2
    )
    context = count_position
    context = context * 2 + pitcher_hand
    context = context * 2 + batter_hand
    context = context * 3 + outs
    context = context * 3 + runner_coarse
    if context.min() < 0 or context.max() >= CONTEXT_COUNT:
        raise ValueError("context outside fixed domain")
    frame["context_position"] = context.astype(np.int16)
    return {
        season: frame.loc[frame["season"].eq(season)].reset_index(
            drop=True
        )
        for season in EVALUATED_SEASONS
    }


def load_oof(
    rows: dict[int, pd.DataFrame],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(
            TARGET_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        base[season] = np.load(
            BASE_ROOT / f"predictions_strict_rank_s300_{season}.npy"
        ).astype(np.float64)
        csv_target = rows[season]["control_success"].to_numpy(
            dtype=np.float64
        )
        if not (
            len(csv_target) == len(targets[season]) == len(base[season])
            and np.array_equal(csv_target, targets[season])
        ):
            raise ValueError(f"OOF target/order mismatch {season}")
        if not np.isfinite(base[season]).all() or not (
            (base[season] >= 0.0).all() and (base[season] <= 1.0).all()
        ):
            raise ValueError(f"invalid base prediction {season}")
    return targets, base


def continuous_matrix(frame: pd.DataFrame) -> np.ndarray:
    pitcher_n = frame["asof_pitcher_n"].to_numpy(dtype=np.float64)
    batter_n = frame["asof_batter_n"].to_numpy(dtype=np.float64)
    values: dict[str, np.ndarray] = {
        "log1p_pitcher_n": np.log1p(pitcher_n),
        "pitcher_reliability30": pitcher_n / (pitcher_n + 30.0),
        "log1p_batter_n": np.log1p(batter_n),
        "batter_reliability30": batter_n / (batter_n + 30.0),
        "inning_scaled": np.clip(
            frame["inning"].to_numpy(dtype=np.float64), 1.0, 12.0
        )
        / 9.0,
        "score_diff_clipped": np.clip(
            frame["score_diff_pitcher_team"].to_numpy(dtype=np.float64),
            -8.0,
            8.0,
        )
        / 8.0,
        "log1p_li": np.log1p(
            np.clip(frame["li"].to_numpy(dtype=np.float64), 0.0, 100.0)
        ),
    }
    for column in (
        "asof_pitcher_success_rate",
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
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
    ):
        values[column] = frame[column].to_numpy(dtype=np.float64)
    matrix = np.column_stack([values[name] for name in CONTINUOUS_FEATURES])
    return matrix


def season_equal_weights(seasons: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(seasons), dtype=np.float64)
    unique = np.unique(seasons)
    for season in unique:
        mask = seasons == season
        weights[mask] = 1.0 / float(mask.sum())
    weights *= len(weights) / float(len(unique))
    totals = [float(weights[seasons == season].sum()) for season in unique]
    if max(totals) - min(totals) > 1e-8:
        raise AssertionError("source season weights differ")
    return weights.astype(np.float32)


def centered_residuals(
    rows: dict[int, pd.DataFrame],
    targets: dict[int, np.ndarray],
    base: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    result: dict[int, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        is_r = rows[season]["game_type"].astype(str).to_numpy() == "R"
        raw = targets[season] - base[season]
        centered = raw.copy()
        center = float(raw[is_r].mean())
        centered[is_r] -= center
        result[season] = centered
        diagnostics[str(season)] = {
            "R_rows": int(is_r.sum()),
            "F_rows": int((~is_r).sum()),
            "raw_R_residual_mean": center,
            "centered_R_residual_mean": float(centered[is_r].mean()),
        }
    return result, diagnostics


def fit_preprocessor(
    raw: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    finite = np.isfinite(raw)
    weighted_finite = finite * weights[:, None]
    denominators = weighted_finite.sum(axis=0)
    if np.any(denominators <= 0.0):
        raise ValueError("continuous feature has no finite source value")
    means = np.nansum(raw * weights[:, None], axis=0) / denominators
    filled = np.where(finite, raw, means[None, :])
    variance = np.sum(
        weights[:, None] * np.square(filled - means[None, :]), axis=0
    ) / float(weights.sum())
    scales = np.sqrt(np.maximum(variance, 1e-8))
    return {
        "mean": means.astype(np.float64),
        "scale": scales.astype(np.float64),
    }


def transform_continuous(
    raw: np.ndarray, preprocessor: dict[str, np.ndarray]
) -> np.ndarray:
    mean = preprocessor["mean"]
    scale = preprocessor["scale"]
    filled = np.where(np.isfinite(raw), raw, mean[None, :])
    transformed = np.clip((filled - mean[None, :]) / scale[None, :], -6, 6)
    if not np.isfinite(transformed).all():
        raise ValueError("non-finite transformed continuous state")
    return transformed.astype(np.float32)


def id_mapping(values: pd.Series) -> tuple[np.ndarray, pd.Index]:
    unique = pd.Index(pd.unique(values))
    encoded = unique.get_indexer(values).astype(np.int32)
    if (encoded < 0).any():
        raise AssertionError("source ID encoding failed")
    return encoded, unique


def build_training_data(
    source_seasons: list[int],
    rows: dict[int, pd.DataFrame],
    centered: dict[int, np.ndarray],
) -> tuple[EncodedData, dict[str, Any]]:
    source_frames: list[pd.DataFrame] = []
    source_targets: list[np.ndarray] = []
    for season in source_seasons:
        mask = rows[season]["game_type"].astype(str).to_numpy() == "R"
        source_frames.append(rows[season].loc[mask].reset_index(drop=True))
        source_targets.append(centered[season][mask])
    frame = pd.concat(source_frames, ignore_index=True)
    target = np.concatenate(source_targets).astype(np.float32)
    seasons = frame["season"].to_numpy(dtype=np.int16)
    weights = season_equal_weights(seasons)
    pitcher, pitcher_index = id_mapping(frame["pitcher_id"])
    batter, batter_index = id_mapping(frame["batter_id"])
    preprocessor = fit_preprocessor(continuous_matrix(frame), weights)
    continuous = transform_continuous(continuous_matrix(frame), preprocessor)
    data = EncodedData(
        pitcher=pitcher,
        batter=batter,
        context=frame["context_position"].to_numpy(dtype=np.int32),
        continuous=continuous,
        pitcher_reliability=(
            frame["asof_pitcher_n"].to_numpy(dtype=np.float64)
            / (
                frame["asof_pitcher_n"].to_numpy(dtype=np.float64)
                + 300.0
            )
        ).astype(np.float32),
        batter_reliability=(
            frame["asof_batter_n"].to_numpy(dtype=np.float64)
            / (
                frame["asof_batter_n"].to_numpy(dtype=np.float64)
                + 300.0
            )
        ).astype(np.float32),
        target=target,
        weight=weights,
        seasons=seasons,
    )
    metadata: dict[str, Any] = {
        "source_seasons": [int(value) for value in source_seasons],
        "source_R_rows": int(len(frame)),
        "pitcher_index": pitcher_index,
        "batter_index": batter_index,
        "preprocessor": preprocessor,
        "season_weight_totals": {
            str(season): float(weights[seasons == season].sum())
            for season in np.unique(seasons)
        },
    }
    return data, metadata


def encode_application(
    frame: pd.DataFrame,
    target: np.ndarray | None,
    metadata: dict[str, Any],
) -> EncodedData:
    is_r = frame["game_type"].astype(str).to_numpy() == "R"
    local = frame.loc[is_r].reset_index(drop=True)
    pitcher = metadata["pitcher_index"].get_indexer(
        local["pitcher_id"]
    ).astype(np.int32)
    batter = metadata["batter_index"].get_indexer(
        local["batter_id"]
    ).astype(np.int32)
    continuous = transform_continuous(
        continuous_matrix(local), metadata["preprocessor"]
    )
    seasons = local["season"].to_numpy(dtype=np.int16)
    return EncodedData(
        pitcher=pitcher,
        batter=batter,
        context=local["context_position"].to_numpy(dtype=np.int32),
        continuous=continuous,
        pitcher_reliability=(
            local["asof_pitcher_n"].to_numpy(dtype=np.float64)
            / (
                local["asof_pitcher_n"].to_numpy(dtype=np.float64)
                + 300.0
            )
        ).astype(np.float32),
        batter_reliability=(
            local["asof_batter_n"].to_numpy(dtype=np.float64)
            / (
                local["asof_batter_n"].to_numpy(dtype=np.float64)
                + 300.0
            )
        ).astype(np.float32),
        target=(
            np.zeros(len(local), dtype=np.float32)
            if target is None
            else target[is_r].astype(np.float32)
        ),
        weight=np.ones(len(local), dtype=np.float32),
        seasons=seasons,
    )


def initialize_parameters(
    data: EncodedData, embedding_dim: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED + embedding_dim)
    pitcher_count = int(data.pitcher.max()) + 1
    batter_count = int(data.batter.max()) + 1
    feature_count = data.continuous.shape[1]
    parameters = {
        "pitcher_bias": np.zeros(pitcher_count, dtype=np.float64),
        "batter_bias": np.zeros(batter_count, dtype=np.float64),
        "context_bias": np.zeros(CONTEXT_COUNT, dtype=np.float64),
        "linear": np.zeros(feature_count, dtype=np.float64),
        "pitcher_embedding": rng.normal(
            0.0, 0.003, size=(pitcher_count, embedding_dim)
        ),
        "pitcher_context_embedding": rng.normal(
            0.0, 0.003, size=(CONTEXT_COUNT, embedding_dim)
        ),
        "batter_embedding": rng.normal(
            0.0, 0.003, size=(batter_count, embedding_dim)
        ),
        "batter_context_embedding": rng.normal(
            0.0, 0.003, size=(CONTEXT_COUNT, embedding_dim)
        ),
    }
    return parameters


def predict_encoded(
    parameters: dict[str, np.ndarray], data: EncodedData
) -> np.ndarray:
    result = (
        parameters["context_bias"][data.context]
        + data.continuous.astype(np.float64) @ parameters["linear"]
    )
    known_pitcher = data.pitcher >= 0
    if known_pitcher.any():
        p = data.pitcher[known_pitcher]
        c = data.context[known_pitcher]
        reliability = data.pitcher_reliability[known_pitcher].astype(
            np.float64
        )
        result[known_pitcher] += (
            reliability * parameters["pitcher_bias"][p]
        )
        result[known_pitcher] += np.sum(
            np.sqrt(reliability)[:, None]
            * parameters["pitcher_embedding"][p]
            * parameters["pitcher_context_embedding"][c],
            axis=1,
        )
    known_batter = data.batter >= 0
    if known_batter.any():
        b = data.batter[known_batter]
        c = data.context[known_batter]
        reliability = data.batter_reliability[known_batter].astype(
            np.float64
        )
        result[known_batter] += (
            reliability * parameters["batter_bias"][b]
        )
        result[known_batter] += np.sum(
            np.sqrt(reliability)[:, None]
            * parameters["batter_embedding"][b]
            * parameters["batter_context_embedding"][c],
            axis=1,
        )
    return result


def parameter_penalty(parameters: dict[str, np.ndarray]) -> float:
    bias = sum(
        float(np.square(parameters[name]).sum())
        for name in ("pitcher_bias", "batter_bias", "context_bias")
    )
    linear = float(np.square(parameters["linear"]).sum())
    embedding = sum(
        float(np.square(parameters[name]).sum())
        for name in (
            "pitcher_embedding",
            "pitcher_context_embedding",
            "batter_embedding",
            "batter_context_embedding",
        )
    )
    return BIAS_L2 * bias + LINEAR_L2 * linear + EMBEDDING_L2 * embedding


def gradients(
    parameters: dict[str, np.ndarray], data: EncodedData
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    prediction = predict_encoded(parameters, data)
    error = prediction - data.target.astype(np.float64)
    weight = data.weight.astype(np.float64)
    denominator = float(weight.sum())
    row_gradient = 2.0 * weight * error / denominator
    p = data.pitcher
    b = data.batter
    c = data.context
    pitcher_reliability = data.pitcher_reliability.astype(np.float64)
    batter_reliability = data.batter_reliability.astype(np.float64)
    pitcher_factor_reliability = np.sqrt(pitcher_reliability)
    batter_factor_reliability = np.sqrt(batter_reliability)
    dimension = parameters["pitcher_embedding"].shape[1]
    output: dict[str, np.ndarray] = {}
    output["pitcher_bias"] = np.bincount(
        p,
        weights=row_gradient * pitcher_reliability,
        minlength=len(parameters["pitcher_bias"]),
    )
    output["batter_bias"] = np.bincount(
        b,
        weights=row_gradient * batter_reliability,
        minlength=len(parameters["batter_bias"]),
    )
    output["context_bias"] = np.bincount(
        c, weights=row_gradient, minlength=CONTEXT_COUNT
    )
    output["linear"] = (
        data.continuous.astype(np.float64).T @ row_gradient
    )
    output["pitcher_embedding"] = np.empty_like(
        parameters["pitcher_embedding"]
    )
    output["pitcher_context_embedding"] = np.empty_like(
        parameters["pitcher_context_embedding"]
    )
    output["batter_embedding"] = np.empty_like(
        parameters["batter_embedding"]
    )
    output["batter_context_embedding"] = np.empty_like(
        parameters["batter_context_embedding"]
    )
    for component in range(dimension):
        output["pitcher_embedding"][:, component] = np.bincount(
            p,
            weights=(
                row_gradient
                * pitcher_factor_reliability
                * parameters["pitcher_context_embedding"][c, component]
            ),
            minlength=len(parameters["pitcher_bias"]),
        )
        output["pitcher_context_embedding"][:, component] = np.bincount(
            c,
            weights=(
                row_gradient
                * pitcher_factor_reliability
                * parameters["pitcher_embedding"][p, component]
            ),
            minlength=CONTEXT_COUNT,
        )
        output["batter_embedding"][:, component] = np.bincount(
            b,
            weights=(
                row_gradient
                * batter_factor_reliability
                * parameters["batter_context_embedding"][c, component]
            ),
            minlength=len(parameters["batter_bias"]),
        )
        output["batter_context_embedding"][:, component] = np.bincount(
            c,
            weights=(
                row_gradient
                * batter_factor_reliability
                * parameters["batter_embedding"][b, component]
            ),
            minlength=CONTEXT_COUNT,
        )
    for name in ("pitcher_bias", "batter_bias", "context_bias"):
        output[name] += 2.0 * BIAS_L2 * parameters[name]
    output["linear"] += 2.0 * LINEAR_L2 * parameters["linear"]
    for name in (
        "pitcher_embedding",
        "pitcher_context_embedding",
        "batter_embedding",
        "batter_context_embedding",
    ):
        output[name] += 2.0 * EMBEDDING_L2 * parameters[name]
    weighted_mse = float(np.sum(weight * np.square(error)) / denominator)
    return output, {
        "weighted_mse": weighted_mse,
        "penalty": parameter_penalty(parameters),
        "objective": weighted_mse + parameter_penalty(parameters),
    }


def adam_update(
    parameters: dict[str, np.ndarray],
    gradients_now: dict[str, np.ndarray],
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    epoch: int,
) -> None:
    for name in parameters:
        gradient = gradients_now[name]
        first[name] = ADAM_BETA1 * first[name] + (1.0 - ADAM_BETA1) * gradient
        second[name] = (
            ADAM_BETA2 * second[name]
            + (1.0 - ADAM_BETA2) * np.square(gradient)
        )
        first_hat = first[name] / (1.0 - ADAM_BETA1**epoch)
        second_hat = second[name] / (1.0 - ADAM_BETA2**epoch)
        parameters[name] -= (
            LEARNING_RATE
            * first_hat
            / (np.sqrt(second_hat) + ADAM_EPSILON)
        )
        np.clip(
            parameters[name],
            -PARAMETER_CLIP,
            PARAMETER_CLIP,
            out=parameters[name],
        )


def embedding_diagnostics(
    parameters: dict[str, np.ndarray]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, values in parameters.items():
        result[name] = {
            "shape": [int(value) for value in values.shape],
            "frobenius_or_l2_norm": float(np.linalg.norm(values)),
            "rms": float(np.sqrt(np.mean(np.square(values)))),
            "max_absolute": float(np.max(np.abs(values))),
        }
    return result


def fit_model(
    data: EncodedData,
    embedding_dim: int,
    epochs: int,
    validation: EncodedData | None = None,
    use_early_stopping: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    parameters = initialize_parameters(data, embedding_dim)
    first = {name: np.zeros_like(value) for name, value in parameters.items()}
    second = {name: np.zeros_like(value) for name, value in parameters.items()}
    history: list[dict[str, float | int]] = []
    best_epoch = epochs
    best_validation = math.inf
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        gradient, before = gradients(parameters, data)
        adam_update(parameters, gradient, first, second, epoch)
        train_prediction = predict_encoded(parameters, data)
        train_offset = float(
            np.average(train_prediction, weights=data.weight)
        )
        train_error = train_prediction - train_offset - data.target
        train_mse = float(
            np.average(np.square(train_error), weights=data.weight)
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "objective_before_update": before["objective"],
            "train_centered_weighted_mse": train_mse,
            "source_prediction_offset": train_offset,
        }
        if validation is not None:
            validation_prediction = (
                predict_encoded(parameters, validation) - train_offset
            )
            validation_mse = float(
                np.mean(
                    np.square(validation_prediction - validation.target)
                )
            )
            record["inner_validation_centered_residual_mse"] = validation_mse
            if validation_mse < best_validation - 1e-10:
                best_validation = validation_mse
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        history.append(record)
        if (
            use_early_stopping
            and validation is not None
            and epochs_without_improvement >= INNER_PATIENCE
        ):
            break
    final_prediction = predict_encoded(parameters, data)
    output_offset = float(np.average(final_prediction, weights=data.weight))
    diagnostics = {
        "requested_epochs": int(epochs),
        "completed_epochs": int(len(history)),
        "selected_best_epoch": int(best_epoch),
        "best_inner_validation_centered_residual_mse": (
            None if validation is None else float(best_validation)
        ),
        "source_prediction_offset": output_offset,
        "history": history,
        "parameter_norms": embedding_diagnostics(parameters),
    }
    parameters["source_prediction_offset"] = np.array(
        output_offset, dtype=np.float64
    )
    return parameters, diagnostics


def model_predict(
    parameters: dict[str, np.ndarray], data: EncodedData
) -> np.ndarray:
    model_parameters = {
        key: value
        for key, value in parameters.items()
        if key != "source_prediction_offset"
    }
    return (
        predict_encoded(model_parameters, data)
        - float(parameters["source_prediction_offset"])
    )


def segment_masks(
    frame: pd.DataFrame,
    source_pitchers: pd.Index,
    source_batters: pd.Index,
) -> dict[str, np.ndarray]:
    game_types = frame["game_type"].astype(str).to_numpy()
    pitcher_seen = source_pitchers.get_indexer(frame["pitcher_id"]) >= 0
    batter_seen = source_batters.get_indexer(frame["batter_id"]) >= 0
    n = frame["asof_pitcher_n"].to_numpy(dtype=np.float64)
    return {
        "all": np.ones(len(frame), dtype=bool),
        "game_type_R": game_types == "R",
        "game_type_F": game_types == "F",
        "source_pitcher_seen": pitcher_seen,
        "source_pitcher_unseen": ~pitcher_seen,
        "source_batter_seen": batter_seen,
        "source_batter_unseen": ~batter_seen,
        "both_ids_seen": pitcher_seen & batter_seen,
        "any_id_unseen": ~(pitcher_seen & batter_seen),
        "pitcher_history_n0": n == 0.0,
        "pitcher_history_n1_19": (n >= 1.0) & (n < 20.0),
        "pitcher_history_n20_99": (n >= 20.0) & (n < 100.0),
        "pitcher_history_n100_499": (n >= 100.0) & (n < 500.0),
        "pitcher_history_n500_plus": n >= 500.0,
    }


def segment_metrics(
    frame: pd.DataFrame,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    source_pitchers: pd.Index,
    source_batters: pd.Index,
) -> dict[str, object]:
    def safe_metrics(
        local_target: np.ndarray, local_prediction: np.ndarray
    ) -> dict[str, object]:
        actual_rate = float(local_target.mean())
        baseline_brier = actual_rate * (1.0 - actual_rate)
        if baseline_brier > 0.0:
            return calculate_metrics(local_target, local_prediction)
        design = np.column_stack(
            [local_prediction, np.ones_like(local_prediction)]
        )
        slope, intercept = np.linalg.lstsq(
            design, local_target, rcond=None
        )[0]
        brier = float(np.mean(np.square(local_prediction - local_target)))
        return {
            "rows": int(len(local_target)),
            "actual_rate": actual_rate,
            "prediction_mean": float(local_prediction.mean()),
            "mean_gap": float(local_prediction.mean() - actual_rate),
            "prediction_min": float(local_prediction.min()),
            "prediction_max": float(local_prediction.max()),
            "brier_score": brier,
            "baseline_brier": baseline_brier,
            "skill_score": None,
            "skill_score_unclipped": None,
            "diagnostic_calibration_slope": float(slope),
            "diagnostic_calibration_intercept": float(intercept),
            "constant_target_skill_undefined": True,
        }

    result: dict[str, object] = {}
    for name, mask in segment_masks(
        frame, source_pitchers, source_batters
    ).items():
        count = int(mask.sum())
        result[name] = {
            "rows": count,
            "actual_rate": float(target[mask].mean()) if count else None,
            "candidates": {
                candidate: (
                    safe_metrics(target[mask], prediction[mask])
                    if count
                    else None
                )
                for candidate, prediction in predictions.items()
            },
        }
    return result


def aggregate_metrics(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    names = [BASE_CANDIDATE, *[candidate.name for candidate in CANDIDATES]]
    for candidate in names:
        briers = {
            str(season): float(
                folds[str(season)]["candidates"][candidate]["brier_score"]
            )
            for season in REPORT_SEASONS
        }
        skills = {
            str(season): float(
                folds[str(season)]["candidates"][candidate][
                    "skill_score_unclipped"
                ]
            )
            for season in REPORT_SEASONS
        }
        summary: dict[str, Any] = {
            "season_briers": briers,
            "season_skills": skills,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills["2024"],
        }
        if candidate != BASE_CANDIDATE:
            summary["season_skill_change_vs_base"] = {
                str(season): float(
                    skills[str(season)]
                    - folds[str(season)]["candidates"][BASE_CANDIDATE][
                        "skill_score_unclipped"
                    ]
                )
                for season in REPORT_SEASONS
            }
            summary["beats_base_every_report_season"] = bool(
                all(
                    value > 0.0
                    for value in summary[
                        "season_skill_change_vs_base"
                    ].values()
                )
            )
        aggregate[candidate] = summary
    return aggregate


def json_preprocessor(metadata: dict[str, Any]) -> dict[str, object]:
    return {
        "source_seasons": metadata["source_seasons"],
        "source_R_rows": metadata["source_R_rows"],
        "source_pitcher_count": int(len(metadata["pitcher_index"])),
        "source_batter_count": int(len(metadata["batter_index"])),
        "season_weight_totals": metadata["season_weight_totals"],
        "continuous_feature_names": list(CONTINUOUS_FEATURES),
        "continuous_source_mean": [
            float(value) for value in metadata["preprocessor"]["mean"]
        ],
        "continuous_source_scale": [
            float(value) for value in metadata["preprocessor"]["scale"]
        ],
    }


def main() -> None:
    started = time.time()
    rows = load_rows()
    targets, base = load_oof(rows)
    centered, residual_diagnostics = centered_residuals(
        rows, targets, base
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        fold_started = time.time()
        source_seasons = [
            season
            for season in EVALUATED_SEASONS
            if season < validation_season
        ]
        validation_frame = rows[validation_season]
        validation_target = targets[validation_season]
        validation_base = base[validation_season]
        validation_is_r = (
            validation_frame["game_type"].astype(str).to_numpy() == "R"
        )
        predictions: dict[str, np.ndarray] = {
            BASE_CANDIDATE: validation_base.copy()
        }
        corrections: dict[str, np.ndarray] = {}
        model_diagnostics: dict[str, object] = {}
        source_pitchers = pd.Index([], dtype=object)
        source_batters = pd.Index([], dtype=object)
        source_metadata_json: dict[str, object] | None = None

        if source_seasons:
            final_data, final_metadata = build_training_data(
                source_seasons, rows, centered
            )
            source_pitchers = final_metadata["pitcher_index"]
            source_batters = final_metadata["batter_index"]
            source_metadata_json = json_preprocessor(final_metadata)
            validation_encoded = encode_application(
                validation_frame,
                None,
                final_metadata,
            )

            selected_epochs: dict[int, int] = {}
            inner_diagnostics: dict[str, object] = {}
            if len(source_seasons) >= 2:
                inner_validation_season = source_seasons[-1]
                inner_train_seasons = source_seasons[:-1]
                inner_data, inner_metadata = build_training_data(
                    inner_train_seasons, rows, centered
                )
                inner_validation = encode_application(
                    rows[inner_validation_season],
                    centered[inner_validation_season],
                    inner_metadata,
                )
                for dimension in EMBEDDING_DIMS:
                    _, diagnostics = fit_model(
                        inner_data,
                        dimension,
                        MAX_INNER_EPOCHS,
                        validation=inner_validation,
                        use_early_stopping=True,
                    )
                    selected_epochs[dimension] = int(
                        diagnostics["selected_best_epoch"]
                    )
                    inner_diagnostics[str(dimension)] = {
                        "inner_train_seasons": inner_train_seasons,
                        "inner_validation_season": inner_validation_season,
                        "selected_epoch": selected_epochs[dimension],
                        "completed_epochs": diagnostics[
                            "completed_epochs"
                        ],
                        "best_inner_validation_centered_residual_mse": (
                            diagnostics[
                                "best_inner_validation_centered_residual_mse"
                            ]
                        ),
                        "history": diagnostics["history"],
                    }
            else:
                selected_epochs = {
                    dimension: DEFAULT_EPOCHS
                    for dimension in EMBEDDING_DIMS
                }
                inner_diagnostics = {
                    str(dimension): {
                        "inner_train_seasons": [],
                        "inner_validation_season": None,
                        "selected_epoch": DEFAULT_EPOCHS,
                        "reason": "single source season; fixed default",
                    }
                    for dimension in EMBEDDING_DIMS
                }

            raw_by_dimension: dict[int, np.ndarray] = {}
            for dimension in EMBEDDING_DIMS:
                model, diagnostics = fit_model(
                    final_data,
                    dimension,
                    selected_epochs[dimension],
                )
                raw_r = model_predict(model, validation_encoded)
                raw = np.zeros(len(validation_frame), dtype=np.float64)
                raw[validation_is_r] = raw_r
                raw_by_dimension[dimension] = raw
                model_diagnostics[str(dimension)] = {
                    "selected_epoch_from_past_inner_season": int(
                        selected_epochs[dimension]
                    ),
                    "inner_selection": inner_diagnostics[str(dimension)],
                    "final_fit": diagnostics,
                    "validation_R_raw_correction_mean": float(raw_r.mean()),
                    "validation_R_raw_correction_std": float(raw_r.std()),
                    "validation_R_raw_correction_min": float(raw_r.min()),
                    "validation_R_raw_correction_max": float(raw_r.max()),
                    "validation_R_known_pitcher_rate": float(
                        (validation_encoded.pitcher >= 0).mean()
                    ),
                    "validation_R_known_batter_rate": float(
                        (validation_encoded.batter >= 0).mean()
                    ),
                }

            for candidate in CANDIDATES:
                correction = np.zeros(
                    len(validation_frame), dtype=np.float64
                )
                correction[validation_is_r] = np.clip(
                    candidate.correction_weight
                    * raw_by_dimension[candidate.embedding_dim][
                        validation_is_r
                    ],
                    -CORRECTION_CLIP,
                    CORRECTION_CLIP,
                )
                prediction = np.clip(
                    validation_base + correction, 0.0, 1.0
                )
                if not np.array_equal(
                    prediction[~validation_is_r],
                    validation_base[~validation_is_r],
                ):
                    raise AssertionError("F prediction changed")
                predictions[candidate.name] = prediction
                corrections[candidate.name] = correction
        else:
            for candidate in CANDIDATES:
                predictions[candidate.name] = validation_base.copy()
                corrections[candidate.name] = np.zeros(
                    len(validation_frame), dtype=np.float64
                )

        metrics = {
            candidate: calculate_metrics(validation_target, prediction)
            for candidate, prediction in predictions.items()
        }
        for candidate, prediction in predictions.items():
            if not np.isfinite(prediction).all() or not (
                (prediction >= 0.0).all() and (prediction <= 1.0).all()
            ):
                raise ValueError(f"invalid prediction {candidate}")
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate}_{validation_season}.npy",
                prediction,
            )
        for candidate, correction in corrections.items():
            np.save(
                ARTIFACT_DIR
                / f"correction_{candidate}_{validation_season}.npy",
                correction,
            )
        pitcher_seen = source_pitchers.get_indexer(
            validation_frame["pitcher_id"]
        ) >= 0
        batter_seen = source_batters.get_indexer(
            validation_frame["batter_id"]
        ) >= 0
        np.save(
            ARTIFACT_DIR / f"source_pitcher_seen_{validation_season}.npy",
            pitcher_seen.astype(np.int8),
        )
        np.save(
            ARTIFACT_DIR / f"source_batter_seen_{validation_season}.npy",
            batter_seen.astype(np.int8),
        )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            validation_target.astype(np.int8),
        )
        np.save(
            ARTIFACT_DIR / f"predictions_{BASE_CANDIDATE}_{validation_season}.npy",
            validation_base,
        )
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "rows": int(len(validation_frame)),
            "source_preprocessing_and_mapping": source_metadata_json,
            "model_diagnostics": model_diagnostics,
            "candidates": metrics,
            "segments": segment_metrics(
                validation_frame,
                validation_target,
                predictions,
                source_pitchers,
                source_batters,
            ),
            "correction_diagnostics": {
                candidate: {
                    "mean": float(correction.mean()),
                    "std": float(correction.std()),
                    "mean_absolute": float(np.abs(correction).mean()),
                    "min": float(correction.min()),
                    "max": float(correction.max()),
                    "nonzero_rows": int((correction != 0.0).sum()),
                    "F_nonzero_rows": int(
                        (correction[~validation_is_r] != 0.0).sum()
                    ),
                }
                for candidate, correction in corrections.items()
            },
            "strict_temporal_invariants": {
                "all_source_seasons_precede_validation": bool(
                    all(
                        source_season < validation_season
                        for source_season in source_seasons
                    )
                ),
                "inner_selection_season_precedes_outer_validation": bool(
                    len(source_seasons) < 2
                    or source_seasons[-1] < validation_season
                ),
                "current_fold_labels_used_for_fit_or_selection": False,
                "validation_or_test_row_aggregation_used_for_prediction": False,
                "source_season_equal_weighting": True,
                "raw_team_or_season_model_feature": False,
                "F_prediction_bitwise_base": bool(
                    all(
                        np.array_equal(
                            prediction[~validation_is_r],
                            validation_base[~validation_is_r],
                        )
                        for prediction in predictions.values()
                    )
                ),
            },
            "fit_seconds": float(time.time() - fold_started),
        }
        print(
            f"strong FM {validation_season}: "
            + " ".join(
                f"{candidate}="
                f"{metrics[candidate]['skill_score_unclipped']:.2f}"
                for candidate in predictions
            ),
            flush=True,
        )

    aggregate = aggregate_metrics(folds)
    candidate_names = [candidate.name for candidate in CANDIDATES]
    best_mean = max(
        candidate_names,
        key=lambda name: (
            aggregate[name]["mean_skill"],
            aggregate[name]["min_skill"],
            -candidate_names.index(name),
        ),
    )
    best_min = max(
        candidate_names,
        key=lambda name: (
            aggregate[name]["min_skill"],
            aggregate[name]["latest_2024_skill"],
            aggregate[name]["mean_skill"],
            -candidate_names.index(name),
        ),
    )
    output = {
        "experiment": "EXP-021",
        "candidate_family": "strongly_regularized_sparse_residual_FM_R_only",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch_available": False,
            "fallback": "deterministic NumPy full-batch Adam sparse FM",
        },
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": "strict_rank_s300 temporal OOF",
            "training_game_type": "R only",
            "application_game_type": "R only; F exact base",
            "source_residual": "target minus base, centered per source R season",
            "source_season_weighting": "equal total weight",
            "epoch_selection": (
                "latest past inner season only; outer 2022 fixed epoch"
            ),
            "current_fold_labels_used_for_fit_or_selection": False,
            "validation_or_test_row_aggregation_for_prediction": False,
            "test_csv_read": False,
            "candidate_grid_predeclared": True,
            "candidate_comparison_nested": False,
        },
        "predeclared_configuration": {
            "candidate_count": len(CANDIDATES),
            "candidates": [candidate.__dict__ for candidate in CANDIDATES],
            "context": (
                "count x pitcher_hand x batter_hand x outs x runner_count_0_1_2plus"
            ),
            "context_count": CONTEXT_COUNT,
            "continuous_features": list(CONTINUOUS_FEATURES),
            "raw_player_ids": "embedding keys only",
            "raw_team_ids_in_model": False,
            "raw_season_in_model": False,
            "correction_clip": [-CORRECTION_CLIP, CORRECTION_CLIP],
            "default_epochs": DEFAULT_EPOCHS,
            "max_inner_epochs": MAX_INNER_EPOCHS,
            "inner_patience": INNER_PATIENCE,
            "learning_rate": LEARNING_RATE,
            "bias_l2": BIAS_L2,
            "linear_l2": LINEAR_L2,
            "embedding_l2": EMBEDDING_L2,
            "parameter_clip": PARAMETER_CLIP,
            "random_seed": RANDOM_SEED,
            "unseen_id_rule": "zero ID bias and interaction",
            "player_reliability_gate": (
                "bias*n/(n+300), embedding*sqrt(n/(n+300)); n0 ID effect zero"
            ),
            "continuous_preprocessing": "source-only weighted mean/std",
        },
        "residual_centering_diagnostics": residual_diagnostics,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "diagnostic only; candidate comparison is post-hoc",
            "posthoc_best_mean_candidate": best_mean,
            "posthoc_best_min_candidate": best_min,
            "best_mean_beats_base_every_report_season": bool(
                aggregate[best_mean]["beats_base_every_report_season"]
            ),
            "best_min_exceeds_1100": bool(
                aggregate[best_min]["min_skill"] >= 1100.0
            ),
            "stop_factorization_family": bool(
                aggregate[best_min]["min_skill"] < 1100.0
            ),
            "adopt_without_nested_confirmation": False,
        },
        "qa": {
            "target_and_row_order_checked": True,
            "base_probability_range_checked": True,
            "source_R_residual_centering_checked": True,
            "source_season_equal_weight_totals_checked": True,
            "source_only_ID_mapping_and_continuous_preprocessing_checked": True,
            "all_source_and_inner_seasons_precede_outer_validation": True,
            "raw_team_and_season_exclusion_checked": True,
            "F_prediction_bitwise_base_checked": True,
            "prediction_probability_ranges_checked": True,
            "saved_prediction_correction_and_segment_arrays": True,
        },
        "total_seconds": float(time.time() - started),
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            output,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
