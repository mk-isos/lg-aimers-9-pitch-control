"""EXP-115: variational correlated pitcher random slopes on EXP-071 residuals.

The two configurations in this module are the frozen cheap screen from
``docs/MODEL_DISCOVERY_EXP112_ULTRA.md``:

* ``V1-diagonal`` uses a diagonal seven-dimensional pitcher population prior.
* ``V2-rank2`` uses a rank-2-plus-diagonal population prior with bounded
  marginal standard deviations.

Both configurations fit a Gaussian likelihood for the strict source-season
EXP-071 OOF residual.  The model contains no fixed intercept and no
intercept-only candidate: its pitcher term is a seven-dimensional random
slope, accompanied by crossed random intercepts for batter, pitcher team,
batter team, and count/hand context.  Empirical-Bayes prior scales are learned
only through the source ELBO and are constrained to the preregistered
``[1e-4, 0.05]`` interval.

Inference exports posterior means into a deterministic scalar row path.  An
unknown entity is represented by code ``-1`` and contributes exactly zero;
validation rows never update a posterior, vocabulary, normalization statistic,
or any other row's state.
"""

from __future__ import annotations

import argparse
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ultra_model_common import (
    CORRECTION_CLIP,
    MODEL_SEED,
    bounded_candidate,
    diagnostic_metrics,
    exp071_fold,
    fold_rows,
    json_dump,
    load_official,
    peak_rss_mb,
    pooled_metrics,
    promotion_gate,
    reconstructed_game_ids,
    row_independence_audit,
    season_equal_weights,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "EXP-115" / "variational_random_slopes"
EXPERIMENT = "EXP-115"
TARGET = "control_success"
PROTOCOL_PATH = ROOT / "docs" / "MODEL_DISCOVERY_EXP112_ULTRA.md"
PROTOCOL_SHA256 = "1bc9dd6384a721d93205521f6058ff4c0d368d1a2efbb5fa44a32942723184d0"

CONFIGURATIONS = ("V1-diagonal", "V2-rank2")
CONFIG_SLUG = {
    "V1-diagonal": "v1_diagonal",
    "V2-rank2": "v2_rank2",
}
OUTER_SEASONS = (2023, 2024)
SOURCE_SEASONS = {
    2023: (2022,),
    2024: (2022, 2023),
}

SLOPE_NAMES = (
    "intercept",
    "centered_balls",
    "centered_strikes",
    "batter_hand_sign",
    "centered_outs",
    "centered_runner_count",
    "standardized_log1p_li",
)
SLOPE_DIMENSION = len(SLOPE_NAMES)
RANK2_DIMENSION = 2
GROUP_NAMES = (
    "batter",
    "pitcher_team",
    "batter_team",
    "count_pitcher_hand_batter_hand",
)

LEARNING_RATE = 3e-3
BATCH_SIZE = 16_384
EPOCHS = 12
PRIOR_SCALE_MIN = 1e-4
PRIOR_SCALE_MAX = 0.05
INITIAL_SCALE = math.sqrt(PRIOR_SCALE_MIN * PRIOR_SCALE_MAX)
POSTERIOR_STD_EPS = 1e-8
LIKELIHOOD_STD_FLOOR = 1e-6
NOVELTY_RMS_THRESHOLD = 1e-4
TORCH_DTYPE = torch.float64

MODEL_COLUMNS = (
    "balls_before",
    "strikes_before",
    "outs_before",
    "num_runners_on",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
)

REQUIRED_COLUMNS = (
    TARGET,
    *MODEL_COLUMNS,
    # Required by the shared reconstructed-game block bootstrap.
    "inning",
    "top_bottom",
    "run_top_before",
    "run_bot_before",
)


def set_deterministic(seed: int = MODEL_SEED) -> None:
    """Fix all RNGs and use the deterministic CPU implementation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], name: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1 or not len(values):
        raise ValueError("weighted mean requires aligned nonempty vectors")
    return float(np.sum(weights * values) / np.sum(weights))


def _source_vocabulary(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=np.int64)
    if not len(array):
        raise ValueError("cannot fit an empty source vocabulary")
    return np.unique(array)


def _vocabulary_codes(values: pd.Series, vocabulary: np.ndarray) -> np.ndarray:
    return pd.Index(vocabulary).get_indexer(
        values.to_numpy(dtype=np.int64)
    ).astype(np.int64)


def _context_key(frame: pd.DataFrame) -> np.ndarray:
    balls = frame["balls_before"].to_numpy(dtype=np.int64)
    strikes = frame["strikes_before"].to_numpy(dtype=np.int64)
    pitcher_hand = frame["pitcher_hand"].to_numpy(dtype=np.int64)
    batter_hand = frame["batter_hand"].to_numpy(dtype=np.int64)
    if (
        np.any((balls < 0) | (balls > 3))
        or np.any((strikes < 0) | (strikes > 2))
        or np.any((pitcher_hand < 1) | (pitcher_hand > 2))
        or np.any((batter_hand < 1) | (batter_hand > 2))
    ):
        raise ValueError("unexpected count or handedness value")
    count_index = 3 * balls + strikes
    return (
        (count_index * 2 + (pitcher_hand - 1)) * 2 + (batter_hand - 1)
    ).astype(np.int64)


@dataclass(frozen=True)
class EncodedRows:
    slopes: np.ndarray
    pitcher: np.ndarray
    batter: np.ndarray
    pitcher_team: np.ndarray
    batter_team: np.ndarray
    context: np.ndarray

    def __len__(self) -> int:
        return int(len(self.slopes))

    def take(self, positions: np.ndarray) -> "EncodedRows":
        index = np.asarray(positions, dtype=np.int64)
        return EncodedRows(
            slopes=self.slopes[index],
            pitcher=self.pitcher[index],
            batter=self.batter[index],
            pitcher_team=self.pitcher_team[index],
            batter_team=self.batter_team[index],
            context=self.context[index],
        )

    def unknown_counts(self) -> dict[str, int]:
        return {
            "pitcher": int(np.sum(self.pitcher < 0)),
            "batter": int(np.sum(self.batter < 0)),
            "pitcher_team": int(np.sum(self.pitcher_team < 0)),
            "batter_team": int(np.sum(self.batter_team < 0)),
            "count_pitcher_hand_batter_hand": int(
                np.sum(self.context < 0)
            ),
        }


@dataclass(frozen=True)
class SourcePreprocessor:
    balls_center: float
    strikes_center: float
    outs_center: float
    runner_count_center: float
    log1p_li_center: float
    log1p_li_scale: float
    pitcher_vocabulary: np.ndarray
    batter_vocabulary: np.ndarray
    pitcher_team_vocabulary: np.ndarray
    batter_team_vocabulary: np.ndarray
    context_vocabulary: np.ndarray

    @classmethod
    def fit(
        cls, frame: pd.DataFrame, source_weights: np.ndarray
    ) -> "SourcePreprocessor":
        _require_columns(frame, MODEL_COLUMNS, "source rows")
        weights = np.asarray(source_weights, dtype=np.float64)
        if len(frame) != len(weights):
            raise ValueError("source weights are not row-aligned")
        required = [
            "balls_before",
            "strikes_before",
            "outs_before",
            "num_runners_on",
            "li",
            "pitcher_id",
            "batter_id",
            "pitcher_hand",
            "batter_hand",
            "pitcher_team_id",
            "batter_team_id",
        ]
        if frame[required].isna().any().any():
            raise ValueError("source model fields contain missing values")
        li = frame["li"].to_numpy(dtype=np.float64)
        if np.any(li < 0.0) or not np.isfinite(li).all():
            raise ValueError("li must be finite and nonnegative")
        log_li = np.log1p(li)
        log_li_center = _weighted_mean(log_li, weights)
        variance = _weighted_mean(
            np.square(log_li - log_li_center), weights
        )
        log_li_scale = max(float(math.sqrt(variance)), 1e-8)
        context = pd.Series(_context_key(frame), index=frame.index)
        return cls(
            balls_center=_weighted_mean(
                frame["balls_before"].to_numpy(dtype=np.float64), weights
            ),
            strikes_center=_weighted_mean(
                frame["strikes_before"].to_numpy(dtype=np.float64), weights
            ),
            outs_center=_weighted_mean(
                frame["outs_before"].to_numpy(dtype=np.float64), weights
            ),
            runner_count_center=_weighted_mean(
                frame["num_runners_on"].to_numpy(dtype=np.float64), weights
            ),
            log1p_li_center=log_li_center,
            log1p_li_scale=log_li_scale,
            pitcher_vocabulary=_source_vocabulary(frame["pitcher_id"]),
            batter_vocabulary=_source_vocabulary(frame["batter_id"]),
            pitcher_team_vocabulary=_source_vocabulary(
                frame["pitcher_team_id"]
            ),
            batter_team_vocabulary=_source_vocabulary(
                frame["batter_team_id"]
            ),
            context_vocabulary=np.unique(context.to_numpy(dtype=np.int64)),
        )

    def encode(self, frame: pd.DataFrame) -> EncodedRows:
        _require_columns(frame, MODEL_COLUMNS, "model rows")
        li = frame["li"].to_numpy(dtype=np.float64)
        if np.any(li < 0.0) or not np.isfinite(li).all():
            raise ValueError("li must be finite and nonnegative")
        batter_hand = frame["batter_hand"].to_numpy(dtype=np.int64)
        if np.any((batter_hand < 1) | (batter_hand > 2)):
            raise ValueError("unexpected batter hand")
        # Official code 1 is Left and code 2 is Right.  The sign convention is
        # fixed and is not centered or estimated from validation rows.
        batter_hand_sign = np.where(batter_hand == 1, -1.0, 1.0)
        slopes = np.column_stack(
            [
                np.ones(len(frame), dtype=np.float64),
                frame["balls_before"].to_numpy(dtype=np.float64)
                - self.balls_center,
                frame["strikes_before"].to_numpy(dtype=np.float64)
                - self.strikes_center,
                batter_hand_sign,
                frame["outs_before"].to_numpy(dtype=np.float64)
                - self.outs_center,
                frame["num_runners_on"].to_numpy(dtype=np.float64)
                - self.runner_count_center,
                (np.log1p(li) - self.log1p_li_center)
                / self.log1p_li_scale,
            ]
        )
        encoded = EncodedRows(
            slopes=slopes,
            pitcher=_vocabulary_codes(
                frame["pitcher_id"], self.pitcher_vocabulary
            ),
            batter=_vocabulary_codes(
                frame["batter_id"], self.batter_vocabulary
            ),
            pitcher_team=_vocabulary_codes(
                frame["pitcher_team_id"], self.pitcher_team_vocabulary
            ),
            batter_team=_vocabulary_codes(
                frame["batter_team_id"], self.batter_team_vocabulary
            ),
            context=pd.Index(self.context_vocabulary)
            .get_indexer(_context_key(frame))
            .astype(np.int64),
        )
        if not np.isfinite(encoded.slopes).all():
            raise ValueError("nonfinite pitcher slope input")
        return encoded

    def metadata(self) -> dict[str, object]:
        return {
            "slope_names": list(SLOPE_NAMES),
            "centers": {
                "balls_before": self.balls_center,
                "strikes_before": self.strikes_center,
                "outs_before": self.outs_center,
                "num_runners_on": self.runner_count_center,
                "log1p_li": self.log1p_li_center,
            },
            "scales": {"log1p_li": self.log1p_li_scale},
            "batter_hand_sign": {"Left_1": -1.0, "Right_2": 1.0},
            "vocabulary_sizes": {
                "pitcher": int(len(self.pitcher_vocabulary)),
                "batter": int(len(self.batter_vocabulary)),
                "pitcher_team": int(len(self.pitcher_team_vocabulary)),
                "batter_team": int(len(self.batter_team_vocabulary)),
                "count_pitcher_hand_batter_hand": int(
                    len(self.context_vocabulary)
                ),
            },
            "fit_scope": "strict source EXP-071 OOF seasons only",
            "season_is_predictive_feature": False,
        }


class VariationalTable(nn.Module):
    """Independent diagonal Gaussian variational table."""

    def __init__(self, rows: int, width: int = 1) -> None:
        super().__init__()
        if rows <= 0 or width <= 0:
            raise ValueError("variational table dimensions must be positive")
        shape = (rows, width) if width > 1 else (rows,)
        self.mean = nn.Parameter(torch.zeros(shape, dtype=TORCH_DTYPE))
        initial_rho = _inverse_softplus(INITIAL_SCALE)
        self.rho = nn.Parameter(
            torch.full(shape, initial_rho, dtype=TORCH_DTYPE)
        )

    def std(self) -> torch.Tensor:
        return F.softplus(self.rho) + POSTERIOR_STD_EPS

    def sample(self) -> torch.Tensor:
        return self.mean + self.std() * torch.randn_like(self.mean)


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse softplus requires a positive value")
    return float(math.log(math.expm1(value)))


def _bounded_scale(raw: torch.Tensor) -> torch.Tensor:
    return PRIOR_SCALE_MIN + (
        PRIOR_SCALE_MAX - PRIOR_SCALE_MIN
    ) * torch.sigmoid(raw)


def _initial_bounded_raw(value: float) -> float:
    fraction = (value - PRIOR_SCALE_MIN) / (
        PRIOR_SCALE_MAX - PRIOR_SCALE_MIN
    )
    if not 0.0 < fraction < 1.0:
        raise ValueError("initial prior scale must be strictly inside bounds")
    return float(math.log(fraction / (1.0 - fraction)))


class VariationalRandomSlopes(nn.Module):
    """Variational random slopes with learned bounded population priors."""

    def __init__(self, preprocessor: SourcePreprocessor, config: str) -> None:
        super().__init__()
        if config not in CONFIGURATIONS:
            raise ValueError(f"unknown configuration {config}")
        self.config = config
        self.pitcher = VariationalTable(
            len(preprocessor.pitcher_vocabulary), SLOPE_DIMENSION
        )
        self.batter = VariationalTable(len(preprocessor.batter_vocabulary))
        self.pitcher_team = VariationalTable(
            len(preprocessor.pitcher_team_vocabulary)
        )
        self.batter_team = VariationalTable(
            len(preprocessor.batter_team_vocabulary)
        )
        self.context = VariationalTable(len(preprocessor.context_vocabulary))

        prior_raw = _initial_bounded_raw(INITIAL_SCALE)
        self.pitcher_prior_raw = nn.Parameter(
            torch.full(
                (SLOPE_DIMENSION,), prior_raw, dtype=TORCH_DTYPE
            )
        )
        self.group_prior_raw = nn.Parameter(
            torch.full((len(GROUP_NAMES),), prior_raw, dtype=TORCH_DTYPE)
        )
        if config == "V2-rank2":
            # A nonzero deterministic initialization avoids the zero-factor
            # stationary point while preserving the fixed model seed.
            self.rank_factor_raw = nn.Parameter(
                0.05
                * torch.randn(
                    SLOPE_DIMENSION,
                    RANK2_DIMENSION,
                    dtype=TORCH_DTYPE,
                )
            )
        else:
            self.register_parameter("rank_factor_raw", None)

    def pitcher_prior_scales(self) -> torch.Tensor:
        return _bounded_scale(self.pitcher_prior_raw)

    def group_prior_scales(self) -> torch.Tensor:
        return _bounded_scale(self.group_prior_raw)

    def population_covariance(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return covariance, diagonal std component, and rank loadings.

        In V2 the marginal standard deviation of every slope is exactly the
        corresponding bounded empirical-Bayes scale.  The normalization below
        expresses the covariance as ``diag(d**2) + U @ U.T`` while preventing
        the rank component from evading the prior-scale bound.
        """

        marginal = self.pitcher_prior_scales()
        if self.config == "V1-diagonal":
            diagonal_std = marginal
            loading = torch.zeros(
                SLOPE_DIMENSION, 0, dtype=TORCH_DTYPE
            )
        else:
            assert self.rank_factor_raw is not None
            squared_norm = torch.sum(
                torch.square(self.rank_factor_raw), dim=1
            )
            denominator = torch.sqrt(1.0 + squared_norm)
            diagonal_std = marginal / denominator
            loading = (
                marginal[:, None]
                * self.rank_factor_raw
                / denominator[:, None]
            )
        covariance = torch.diag(torch.square(diagonal_std))
        if loading.shape[1]:
            covariance = covariance + loading @ loading.T
        return covariance, diagonal_std, loading

    @staticmethod
    def _univariate_kl(
        table: VariationalTable, prior_scale: torch.Tensor
    ) -> torch.Tensor:
        variance = torch.square(table.std())
        prior_variance = torch.square(prior_scale)
        return 0.5 * torch.sum(
            (variance + torch.square(table.mean)) / prior_variance
            - 1.0
            + torch.log(prior_variance)
            - torch.log(variance)
        )

    def pitcher_kl(self) -> torch.Tensor:
        covariance, _, _ = self.population_covariance()
        cholesky = torch.linalg.cholesky(covariance)
        inverse = torch.cholesky_inverse(cholesky)
        logdet_prior = 2.0 * torch.sum(
            torch.log(torch.diagonal(cholesky))
        )
        variance = torch.square(self.pitcher.std())
        trace = torch.sum(variance * torch.diagonal(inverse)[None, :])
        quadratic = torch.sum((self.pitcher.mean @ inverse) * self.pitcher.mean)
        logdet_q = torch.sum(torch.log(variance))
        rows = self.pitcher.mean.shape[0]
        return 0.5 * (
            trace
            + quadratic
            - rows * SLOPE_DIMENSION
            + rows * logdet_prior
            - logdet_q
        )

    def total_kl(self) -> torch.Tensor:
        group_scales = self.group_prior_scales()
        group_kl = (
            self._univariate_kl(self.batter, group_scales[0])
            + self._univariate_kl(self.pitcher_team, group_scales[1])
            + self._univariate_kl(self.batter_team, group_scales[2])
            + self._univariate_kl(self.context, group_scales[3])
        )
        return self.pitcher_kl() + group_kl

    def sampled_raw(
        self,
        slopes: torch.Tensor,
        pitcher: torch.Tensor,
        batter: torch.Tensor,
        pitcher_team: torch.Tensor,
        batter_team: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        pitcher_draw = self.pitcher.sample()
        batter_draw = self.batter.sample()
        pitcher_team_draw = self.pitcher_team.sample()
        batter_team_draw = self.batter_team.sample()
        context_draw = self.context.sample()
        return (
            torch.sum(pitcher_draw[pitcher] * slopes, dim=1)
            + batter_draw[batter]
            + pitcher_team_draw[pitcher_team]
            + batter_team_draw[batter_team]
            + context_draw[context]
        )

    def parameter_diagnostics(self) -> dict[str, object]:
        with torch.no_grad():
            covariance, diagonal_std, loading = self.population_covariance()
            prior_scales = self.pitcher_prior_scales()
            group_scales = self.group_prior_scales()
            covariance_array = covariance.cpu().numpy()
            marginal = np.sqrt(np.diag(covariance_array))
            correlation = covariance_array / np.outer(marginal, marginal)
            nonintercept = self.pitcher.mean[:, 1:].cpu().numpy()
            return {
                "pitcher_prior_scales": prior_scales.cpu().numpy().tolist(),
                "group_prior_scales": {
                    name: float(value)
                    for name, value in zip(
                        GROUP_NAMES,
                        group_scales.cpu().numpy(),
                        strict=True,
                    )
                },
                "population_covariance": covariance_array.tolist(),
                "population_correlation": correlation.tolist(),
                "population_offdiagonal_correlation_rms": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                correlation
                                - np.diag(np.diag(correlation))
                            )
                        )
                    )
                ),
                "diagonal_component_std": diagonal_std.cpu()
                .numpy()
                .tolist(),
                "rank2_loading": loading.cpu().numpy().tolist(),
                "pitcher_posterior_mean_rms": float(
                    torch.sqrt(torch.mean(torch.square(self.pitcher.mean)))
                ),
                "pitcher_nonintercept_mean_rms": float(
                    np.sqrt(np.mean(np.square(nonintercept)))
                ),
                "pitcher_posterior_std_mean": float(
                    torch.mean(self.pitcher.std())
                ),
            }


def _as_tensor(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(values), dtype=dtype)


def fit_variational_model(
    model: VariationalRandomSlopes,
    encoded: EncodedRows,
    residual: np.ndarray,
    source_weights: np.ndarray,
    source_season: np.ndarray,
    *,
    seed: int = MODEL_SEED,
) -> dict[str, object]:
    """Fit the fixed 12-epoch source-season-equal Monte Carlo ELBO."""

    residual = np.asarray(residual, dtype=np.float64)
    weights = np.asarray(source_weights, dtype=np.float64)
    seasons = np.asarray(source_season, dtype=np.int16)
    if not (
        len(encoded) == len(residual) == len(weights) == len(seasons)
    ):
        raise ValueError("ELBO arrays are not row-aligned")
    if np.any(encoded.pitcher < 0) or any(
        count > 0 for count in encoded.unknown_counts().values()
    ):
        raise ValueError("source encoding unexpectedly contains an unknown entity")
    if not np.isfinite(residual).all() or not np.isfinite(weights).all():
        raise ValueError("ELBO contains nonfinite residuals or weights")
    likelihood_std = max(
        math.sqrt(_weighted_mean(np.square(residual), weights)),
        LIKELIHOOD_STD_FLOOR,
    )

    slope_tensor = _as_tensor(encoded.slopes, TORCH_DTYPE)
    pitcher_tensor = _as_tensor(encoded.pitcher, torch.int64)
    batter_tensor = _as_tensor(encoded.batter, torch.int64)
    pitcher_team_tensor = _as_tensor(encoded.pitcher_team, torch.int64)
    batter_team_tensor = _as_tensor(encoded.batter_team, torch.int64)
    context_tensor = _as_tensor(encoded.context, torch.int64)
    residual_tensor = _as_tensor(residual, TORCH_DTYPE)
    weight_tensor = _as_tensor(weights, TORCH_DTYPE)

    set_deterministic(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history: list[dict[str, object]] = []
    row_count = len(residual)
    gaussian_constant = math.log(likelihood_std) + 0.5 * math.log(
        2.0 * math.pi
    )
    for epoch in range(EPOCHS):
        order = np.arange(row_count, dtype=np.int64)
        np.random.default_rng(seed + epoch).shuffle(order)
        epoch_nll = 0.0
        epoch_loss = 0.0
        observed_rows = 0
        optimizer_steps = 0
        for start in range(0, row_count, BATCH_SIZE):
            batch_array = order[start : start + BATCH_SIZE]
            batch = torch.as_tensor(batch_array, dtype=torch.int64)
            raw = model.sampled_raw(
                slope_tensor[batch],
                pitcher_tensor[batch],
                batter_tensor[batch],
                pitcher_team_tensor[batch],
                batter_team_tensor[batch],
                context_tensor[batch],
            )
            standardized_error = (
                residual_tensor[batch] - raw
            ) / likelihood_std
            row_nll = 0.5 * torch.square(standardized_error)
            row_nll = row_nll + gaussian_constant
            weighted_nll = torch.mean(weight_tensor[batch] * row_nll)
            normalized_kl = model.total_kl() / float(row_count)
            loss = weighted_nll + normalized_kl
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite ELBO in epoch {epoch + 1}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_rows = len(batch_array)
            epoch_nll += float(weighted_nll.detach()) * batch_rows
            epoch_loss += float(loss.detach()) * batch_rows
            observed_rows += batch_rows
            optimizer_steps += 1
        diagnostics = model.parameter_diagnostics()
        history.append(
            {
                "epoch": epoch + 1,
                "optimizer_steps": optimizer_steps,
                "weighted_gaussian_nll": epoch_nll / observed_rows,
                "negative_elbo_per_row": epoch_loss / observed_rows,
                "normalized_kl": float(model.total_kl().detach())
                / float(row_count),
                "pitcher_prior_scales": diagnostics[
                    "pitcher_prior_scales"
                ],
                "group_prior_scales": diagnostics["group_prior_scales"],
            }
        )
    model.eval()

    season_weight_totals = {
        str(int(season)): float(np.sum(weights[seasons == season]))
        for season in np.unique(seasons)
    }
    totals = np.asarray(list(season_weight_totals.values()), dtype=np.float64)
    return {
        "rows": row_count,
        "source_seasons": sorted(np.unique(seasons).astype(int).tolist()),
        "source_season_equal_weight": True,
        "source_season_weight_totals": season_weight_totals,
        "source_season_weight_max_min_ratio": float(
            totals.max() / totals.min()
        ),
        "gaussian_likelihood": {
            "target": "EXP-071 OOF residual y - p071",
            "std": likelihood_std,
            "std_estimation": "source-season-equal weighted residual RMS",
            "mean_function": "random-effects linear predictor",
        },
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seed": seed,
        "prior_scale_bounds": [PRIOR_SCALE_MIN, PRIOR_SCALE_MAX],
        "history": history,
    }


@dataclass(frozen=True)
class FrozenRandomSlopesState:
    """Posterior-mean state with a bit-stable scalar inference path."""

    preprocessor: SourcePreprocessor
    pitcher_mean: np.ndarray
    batter_mean: np.ndarray
    pitcher_team_mean: np.ndarray
    batter_team_mean: np.ndarray
    context_mean: np.ndarray

    @classmethod
    def from_model(
        cls,
        model: VariationalRandomSlopes,
        preprocessor: SourcePreprocessor,
    ) -> "FrozenRandomSlopesState":
        with torch.no_grad():
            return cls(
                preprocessor=preprocessor,
                pitcher_mean=model.pitcher.mean.cpu().numpy().copy(),
                batter_mean=model.batter.mean.cpu().numpy().copy(),
                pitcher_team_mean=model.pitcher_team.mean.cpu()
                .numpy()
                .copy(),
                batter_team_mean=model.batter_team.mean.cpu()
                .numpy()
                .copy(),
                context_mean=model.context.mean.cpu().numpy().copy(),
            )

    def effect(
        self, frame: pd.DataFrame, *, remove_nonintercept_slopes: bool = False
    ) -> np.ndarray:
        encoded = self.preprocessor.encode(frame)
        output = np.zeros(len(encoded), dtype=np.float64)
        # The same scalar addition sequence is used for every query, regardless
        # of surrounding batch shape or order.  This is the exported inference
        # path audited by ultra_model_common.row_independence_audit.
        for row in range(len(encoded)):
            value = 0.0
            pitcher = int(encoded.pitcher[row])
            if pitcher >= 0:
                value += float(self.pitcher_mean[pitcher, 0])
                if not remove_nonintercept_slopes:
                    for slope in range(1, SLOPE_DIMENSION):
                        value += float(
                            self.pitcher_mean[pitcher, slope]
                            * encoded.slopes[row, slope]
                        )
            batter = int(encoded.batter[row])
            if batter >= 0:
                value += float(self.batter_mean[batter])
            pitcher_team = int(encoded.pitcher_team[row])
            if pitcher_team >= 0:
                value += float(self.pitcher_team_mean[pitcher_team])
            batter_team = int(encoded.batter_team[row])
            if batter_team >= 0:
                value += float(self.batter_team_mean[batter_team])
            context = int(encoded.context[row])
            if context >= 0:
                value += float(self.context_mean[context])
            output[row] = value
        return output

    def raw(
        self, frame: pd.DataFrame, *, remove_nonintercept_slopes: bool = False
    ) -> np.ndarray:
        """Map the Gaussian residual mean into the common bounded raw score."""

        effect = self.effect(
            frame, remove_nonintercept_slopes=remove_nonintercept_slopes
        )
        bounded = np.clip(
            effect,
            -CORRECTION_CLIP * (1.0 - 1e-12),
            CORRECTION_CLIP * (1.0 - 1e-12),
        )
        return np.arctanh(bounded / CORRECTION_CLIP)

    def predict(self, frame: pd.DataFrame, baseline: np.ndarray) -> np.ndarray:
        return bounded_candidate(baseline, self.raw(frame))


def _parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _parameter_bytes(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
    )


def save_frozen_state(
    path: Path,
    state: FrozenRandomSlopesState,
    model: VariationalRandomSlopes,
    likelihood_std: float,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        covariance, diagonal_std, loading = model.population_covariance()
        np.savez_compressed(
            path,
            pitcher_mean=state.pitcher_mean,
            pitcher_std=model.pitcher.std().cpu().numpy(),
            batter_mean=state.batter_mean,
            batter_std=model.batter.std().cpu().numpy(),
            pitcher_team_mean=state.pitcher_team_mean,
            pitcher_team_std=model.pitcher_team.std().cpu().numpy(),
            batter_team_mean=state.batter_team_mean,
            batter_team_std=model.batter_team.std().cpu().numpy(),
            context_mean=state.context_mean,
            context_std=model.context.std().cpu().numpy(),
            pitcher_prior_scales=model.pitcher_prior_scales().cpu().numpy(),
            group_prior_scales=model.group_prior_scales().cpu().numpy(),
            population_covariance=covariance.cpu().numpy(),
            population_diagonal_std=diagonal_std.cpu().numpy(),
            population_rank2_loading=loading.cpu().numpy(),
            pitcher_vocabulary=state.preprocessor.pitcher_vocabulary,
            batter_vocabulary=state.preprocessor.batter_vocabulary,
            pitcher_team_vocabulary=state.preprocessor.pitcher_team_vocabulary,
            batter_team_vocabulary=state.preprocessor.batter_team_vocabulary,
            context_vocabulary=state.preprocessor.context_vocabulary,
            transform=np.asarray(
                [
                    state.preprocessor.balls_center,
                    state.preprocessor.strikes_center,
                    state.preprocessor.outs_center,
                    state.preprocessor.runner_count_center,
                    state.preprocessor.log1p_li_center,
                    state.preprocessor.log1p_li_scale,
                ],
                dtype=np.float64,
            ),
            likelihood_std=np.asarray([likelihood_std], dtype=np.float64),
        )
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "format": "compressed NumPy posterior state",
    }


def _source_fold(
    official: pd.DataFrame, outer_season: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    parts: list[pd.DataFrame] = []
    residuals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    for season in SOURCE_SEASONS[outer_season]:
        rows = fold_rows(official, season)
        target, baseline = exp071_fold(season)
        parts.append(rows)
        targets.append(target)
        baselines.append(baseline)
        residuals.append(target - baseline)
    source = pd.concat(parts, ignore_index=True)
    target = np.concatenate(targets)
    baseline = np.concatenate(baselines)
    residual = np.concatenate(residuals)
    if not source["season"].lt(outer_season).all():
        raise AssertionError("outer-fold source cutoff failed")
    if not np.array_equal(
        source[TARGET].to_numpy(dtype=np.float64), target
    ):
        raise ValueError("source target/order mismatch")
    return source, target, baseline, residual


def run_fold(
    official: pd.DataFrame, config: str, outer_season: int
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if config not in CONFIGURATIONS or outer_season not in OUTER_SEASONS:
        raise ValueError("unregistered EXP-115 fold")
    total_started = time.perf_counter()
    source, _, _, residual = _source_fold(official, outer_season)
    validation = fold_rows(official, outer_season)
    validation_target, validation_baseline = exp071_fold(outer_season)
    source_season = source["season"].to_numpy(dtype=np.int16)
    source_weights = season_equal_weights(source_season)

    preprocess_started = time.perf_counter()
    preprocessor = SourcePreprocessor.fit(source, source_weights)
    source_encoded = preprocessor.encode(source)
    preprocessing_seconds = time.perf_counter() - preprocess_started

    set_deterministic(MODEL_SEED)
    model = VariationalRandomSlopes(preprocessor, config)
    train_started = time.perf_counter()
    fit_report = fit_variational_model(
        model,
        source_encoded,
        residual,
        source_weights,
        source_season,
        seed=MODEL_SEED,
    )
    training_seconds = time.perf_counter() - train_started
    state = FrozenRandomSlopesState.from_model(model, preprocessor)

    fold_dir = ARTIFACT_ROOT / CONFIG_SLUG[config] / str(outer_season)
    fold_dir.mkdir(parents=True, exist_ok=True)
    state_report = save_frozen_state(
        fold_dir / "posterior_state.npz",
        state,
        model,
        float(fit_report["gaussian_likelihood"]["std"]),
    )

    inference_started = time.perf_counter()
    raw = state.raw(validation)
    posterior_mean_effect = state.effect(validation)
    candidate = bounded_candidate(validation_baseline, raw)
    inference_seconds = time.perf_counter() - inference_started
    no_slope_raw = state.raw(
        validation, remove_nonintercept_slopes=True
    )
    no_slope_effect = state.effect(
        validation, remove_nonintercept_slopes=True
    )
    no_slope_candidate = bounded_candidate(
        validation_baseline, no_slope_raw
    )
    slope_removal_rms = float(
        np.sqrt(np.mean(np.square(candidate - no_slope_candidate)))
    )
    slope_removal_raw_rms = float(
        np.sqrt(
            np.mean(np.square(posterior_mean_effect - no_slope_effect))
        )
    )

    games = reconstructed_game_ids(validation)
    metrics = diagnostic_metrics(
        validation_target,
        candidate,
        validation_baseline,
        games,
        season=outer_season,
    )
    audit = row_independence_audit(
        state.predict,
        validation,
        validation_baseline,
        seed=MODEL_SEED,
    )
    validation_encoded = preprocessor.encode(validation)
    parameter_diagnostics = model.parameter_diagnostics()
    prior_scales = np.asarray(
        parameter_diagnostics["pitcher_prior_scales"], dtype=np.float64
    )
    group_scales = np.asarray(
        list(parameter_diagnostics["group_prior_scales"].values()),
        dtype=np.float64,
    )
    if (
        np.min(np.concatenate([prior_scales, group_scales]))
        < PRIOR_SCALE_MIN
        or np.max(np.concatenate([prior_scales, group_scales]))
        > PRIOR_SCALE_MAX
    ):
        raise AssertionError("learned prior scale escaped its frozen bounds")

    np.save(fold_dir / "validation_predictions.npy", candidate)
    np.save(fold_dir / "validation_targets.npy", validation_target)
    np.save(fold_dir / "exp071_baseline.npy", validation_baseline)
    np.save(fold_dir / "posterior_mean_raw.npy", raw)
    np.save(fold_dir / "posterior_mean_residual_effect.npy", posterior_mean_effect)
    np.save(
        fold_dir / "slope_removed_predictions_diagnostic.npy",
        no_slope_candidate,
    )
    total_seconds = time.perf_counter() - total_started
    report: dict[str, object] = {
        "experiment": EXPERIMENT,
        "configuration": config,
        "configuration_role": (
            "primary" if config == "V1-diagonal" else "secondary"
        ),
        "outer_validation_season": outer_season,
        "source_seasons": list(SOURCE_SEASONS[outer_season]),
        "source_rows": int(len(source)),
        "validation_rows": int(len(validation)),
        "model_definition": {
            "likelihood": "Gaussian EXP-071 residual",
            "pitcher_slope_inputs": list(SLOPE_NAMES),
            "crossed_random_intercepts": list(GROUP_NAMES),
            "population_covariance": (
                "diagonal"
                if config == "V1-diagonal"
                else "rank-2-plus-diagonal"
            ),
            "fixed_intercept": False,
            "intercept_only_candidate": False,
            "unseen_entity_posterior_mean": 0.0,
            "inference": "deterministic scalar posterior-mean path",
            "correction": (
                "Gaussian residual effect is clipped to +/-0.03, mapped by "
                "raw=atanh(effect/0.03), then "
                "p=clip(p071+0.25*0.03*tanh(raw),0,1)"
            ),
        },
        "preprocessor": preprocessor.metadata(),
        "fit": fit_report,
        "parameters": {
            "trainable_parameter_count": _parameter_count(model),
            "trainable_parameter_tensor_bytes": _parameter_bytes(model),
            "learned": parameter_diagnostics,
            "serialized_state": state_report,
        },
        "novelty_gate": {
            "diagnostic_only_comparator": True,
            "slope_removed_model_is_candidate": False,
            "removed": "all six non-intercept pitcher slopes",
            "prediction_rms_change": slope_removal_rms,
            "raw_linear_predictor_rms_change": slope_removal_raw_rms,
            "threshold": NOVELTY_RMS_THRESHOLD,
            "passed": slope_removal_rms >= NOVELTY_RMS_THRESHOLD,
        },
        "validation_unknown_entity_counts": validation_encoded.unknown_counts(),
        "row_independence": audit,
        "runtime": {
            "source_preprocessing_seconds": preprocessing_seconds,
            "training_seconds": training_seconds,
            "validation_inference_seconds": inference_seconds,
            "total_fold_seconds": total_seconds,
            "peak_rss_mb": peak_rss_mb(),
        },
        "metrics": metrics,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": str(TORCH_DTYPE),
        },
    }
    json_dump(fold_dir / "fold_report.json", report)
    if not bool(audit["literal_exact_identity_passed"]):
        raise RuntimeError(
            f"{config} {outer_season} failed literal row independence"
        )
    return report, (validation_target, candidate, validation_baseline)


def summarize_configuration(
    fold_reports: Mapping[int, dict[str, object]],
    pooled_arrays: Mapping[
        int, tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
) -> dict[str, object]:
    metrics = {
        int(season): report["metrics"]
        for season, report in fold_reports.items()
    }
    gate = promotion_gate(metrics)
    novelty = {
        str(int(season)): bool(report["novelty_gate"]["passed"])
        for season, report in fold_reports.items()
    }
    novelty_all = all(novelty.values())
    return {
        "fold_metrics": {str(season): value for season, value in metrics.items()},
        "pooled_metrics": pooled_metrics(dict(pooled_arrays)),
        "promotion_gate": gate,
        "novelty_gate_by_fold": novelty,
        "novelty_gate_all_folds": novelty_all,
        "cheap_survivor": bool(gate["metric_survivor"] and novelty_all),
    }


def run(configurations: Iterable[str] = CONFIGURATIONS) -> dict[str, object]:
    actual_protocol_sha256 = sha256_file(PROTOCOL_PATH)
    if actual_protocol_sha256 != PROTOCOL_SHA256:
        raise RuntimeError(
            "preregistration changed after lock: "
            f"{actual_protocol_sha256} != {PROTOCOL_SHA256}"
        )
    selected = tuple(configurations)
    if not selected or any(value not in CONFIGURATIONS for value in selected):
        raise ValueError("requested configuration is not preregistered")
    official = load_official(
        REQUIRED_COLUMNS, seasons=(2022, 2023, 2024)
    )
    summaries: dict[str, object] = {}
    for config in selected:
        fold_reports: dict[int, dict[str, object]] = {}
        pooled_arrays: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        for season in OUTER_SEASONS:
            print(
                f"{EXPERIMENT} {config} outer={season}: fitting",
                flush=True,
            )
            report, arrays = run_fold(official, config, season)
            fold_reports[season] = report
            pooled_arrays[season] = arrays
            delta = float(report["metrics"]["delta_brier_vs_exp071"])
            novelty = float(
                report["novelty_gate"]["prediction_rms_change"]
            )
            print(
                f"  delta_brier={delta:+.9f} slope_removal_rms={novelty:.9f}",
                flush=True,
            )
        summaries[config] = summarize_configuration(
            fold_reports, pooled_arrays
        )

    survivors = [
        config
        for config in CONFIGURATIONS
        if config in summaries and bool(summaries[config]["cheap_survivor"])
    ]
    both_qualified = len(survivors) == 2
    selected_survivor = survivors[0] if survivors else None
    summary = {
        "experiment": EXPERIMENT,
        "stage": "cheap_2023_2024_only",
        "research_lock": "docs/MODEL_DISCOVERY_EXP112_ULTRA.md",
        "research_lock_sha256": PROTOCOL_SHA256,
        "configurations_run": list(selected),
        "configuration_summaries": summaries,
        "family_selection": {
            "eligible_configurations": survivors,
            "selected_configuration": selected_survivor,
            "both_configurations_qualified": both_qualified,
            "primary_wins_if_both_qualify": True,
            "at_most_one_family_survivor_slot": True,
        },
        "full_rolling_or_submission_created": False,
    }
    json_dump(ARTIFACT_ROOT / "validation_metrics.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered EXP-115 cheap variational screen."
    )
    parser.add_argument(
        "--config",
        choices=("all", *CONFIGURATIONS),
        default="all",
        help="Run both frozen configurations or one exact configuration.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    selected = (
        CONFIGURATIONS
        if arguments.config == "all"
        else (arguments.config,)
    )
    summary = run(selected)
    print(
        f"saved {ARTIFACT_ROOT / 'validation_metrics.json'}; "
        f"selected={summary['family_selection']['selected_configuration']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
