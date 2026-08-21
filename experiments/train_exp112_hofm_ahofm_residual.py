"""EXP-112: preregistered all-field order-3 HOFM/AHOFM residual models.

This runner implements only the two configurations frozen in
``docs/MODEL_DISCOVERY_EXP112_ULTRA.md``.  It fits strict EXP-071 OOF
residuals from 2022 for outer 2023 and from equally weighted 2022/2023 for
outer 2024.  Validation labels are used only after every source-fitted model
and prediction has been frozen.

The training implementation uses PyTorch, while scored inference is exported
to a deterministic NumPy path whose reduction order is fixed across rows.
That path is intentionally used for singleton/batch/reverse/permutation/split/
duplicate audits and for the saved validation predictions.
"""

from __future__ import annotations

import argparse
import gc
import math
import platform
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.interpolate import BSpline
from torch import nn

from modern_tabular_nn import (
    CATEGORICAL_COLUMNS,
    EXPECTANCY_COLUMNS,
    NONNEGATIVE_LOG_COLUMNS,
    OFFICIAL_FEATURE_COLUMNS,
    RATE_COLUMNS,
    SIGNED_LOG_COLUMNS,
)
from ultra_model_common import (
    CORRECTION_CLIP,
    DIAGNOSTIC_SEASONS,
    EXP071_ROOT,
    GAME_COLUMNS,
    INTEGRATION_WEIGHT,
    MODEL_SEED,
    bounded_candidate,
    diagnostic_metrics,
    exp051_fold,
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
PREREGISTRATION_PATH = ROOT / "docs" / "MODEL_DISCOVERY_EXP112_ULTRA.md"
PREREGISTRATION_SHA256 = (
    "1bc9dd6384a721d93205521f6058ff4c0d368d1a2efbb5fa44a32942723184d0"
)
ARTIFACT_ROOT = ROOT / "artifacts" / "EXP-112" / "hofm_ahofm_residual"
REPORT_PATH = ARTIFACT_ROOT / "validation_metrics.json"
TARGET_COLUMN = "control_success"

SOURCE_SEASONS_BY_OUTER = {2023: (2022,), 2024: (2022, 2023)}
CONFIGURATION_ORDER = ("F1-hofm3", "F2-ahofm3")
CONFIGURATIONS: dict[str, dict[str, object]] = {
    "F1-hofm3": {
        "kind": "hofm",
        "order_2_rank": 16,
        "order_3_rank": 16,
        "primary": True,
    },
    "F2-ahofm3": {
        "kind": "ahofm",
        "order_2_rank": 8,
        "order_3_rank": 8,
        "spline_basis_count": 6,
        "spline_degree": 3,
        "homogeneous_marginal_df": 4.0,
        "primary": False,
    },
}

EPOCHS = 8
EFFECTIVE_BATCH_SIZE = 8_192
MICRO_BATCH_SIZE = 1_024
LEARNING_RATE = 1e-3
LINEAR_WEIGHT_DECAY = 1e-4
FACTOR_WEIGHT_DECAY = 1e-3
NOVELTY_RMS_THRESHOLD = 1e-4

FEATURE_COLUMNS = tuple(
    column for column in OFFICIAL_FEATURE_COLUMNS if column != "season"
)
CATEGORICAL_FEATURES = tuple(CATEGORICAL_COLUMNS)
CONTINUOUS_FEATURES = tuple(
    column
    for column in FEATURE_COLUMNS
    if column not in set(CATEGORICAL_FEATURES)
)

FROZEN_EXP071_HASHES = {
    "predictions_playerphys_resid_w025_2022.npy": (
        "7794481d1f45cb987e104cc3593e8124747bc7aabc8d6bf239ea93cb3675e18c"
    ),
    "predictions_playerphys_resid_w025_2023.npy": (
        "e302d5cc4d2dd8a16ca4205df24b8763f39f9c381c3344edbe333ba9b0dbb1b3"
    ),
    "predictions_playerphys_resid_w025_2024.npy": (
        "c98a550d83e4b311d7da13bd59074e977c5f87f408712c40069426a09040ee1d"
    ),
    "targets_2022.npy": (
        "40ec827616f5192fb49034ba9299528c3a7c70e31fdb64f1291794839d05a4e8"
    ),
    "targets_2023.npy": (
        "7f2117cb4614e23e43bbfef3f52fb8aff65340e4a2b3f90253fd3f23874e1417"
    ),
    "targets_2024.npy": (
        "32a5a12d22d5e171e4227ed46fe02a9cd4fac20b0ca4b7618977333df97b238f"
    ),
    "validation_metrics.json": (
        "e275db2ac39af70bce5f2dcf4c40ca13f7b75162d0c50619049ffc85666860b9"
    ),
}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def set_seeds(seed: int = MODEL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.ones(1, device=device)
    return device


def verify_locked_inputs() -> dict[str, object]:
    observed_lock = sha256_file(PREREGISTRATION_PATH)
    if observed_lock != PREREGISTRATION_SHA256:
        raise ValueError(
            "EXP-112 preregistration hash changed: "
            f"{observed_lock} != {PREREGISTRATION_SHA256}"
        )
    observed_assets: dict[str, str] = {}
    for filename, expected in FROZEN_EXP071_HASHES.items():
        path = EXP071_ROOT / filename
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"frozen EXP-071 asset changed: {filename} {observed} != {expected}"
            )
        observed_assets[filename] = observed
    return {
        "preregistration_path": str(PREREGISTRATION_PATH),
        "preregistration_sha256": observed_lock,
        "exp071_assets": observed_assets,
    }


def semantic_continuous_matrix(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    missing: list[np.ndarray] = []
    for column in CONTINUOUS_FEATURES:
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        missing.append(~np.isfinite(raw))
        if column in NONNEGATIVE_LOG_COLUMNS:
            transformed = np.log1p(np.clip(raw, 0.0, None))
        elif column in SIGNED_LOG_COLUMNS:
            transformed = np.sign(raw) * np.log1p(np.abs(raw))
        elif column in EXPECTANCY_COLUMNS:
            transformed = np.clip(raw / 100.0, 0.0, 1.0)
        elif column == "li":
            transformed = np.log1p(np.clip(raw, 0.0, None))
        elif column in RATE_COLUMNS:
            transformed = np.clip(raw, 0.0, 1.0)
        else:
            transformed = raw
        values.append(transformed)
    return (
        np.column_stack(values).astype(np.float64, copy=False),
        np.column_stack(missing).astype(bool, copy=False),
    )


def _open_quantile_knot_vector(values: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(values, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    lower = float(quantiles[0])
    upper = float(quantiles[-1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("nonfinite source quantile knot")
    if upper - lower < 1e-6:
        center = 0.5 * (lower + upper)
        lower = center - 5e-4
        upper = center + 5e-4
    internal = np.asarray(quantiles[1:3], dtype=np.float64)
    if not (lower < internal[0] < internal[1] < upper):
        internal = np.linspace(lower, upper, 4, dtype=np.float64)[1:3]
    return np.asarray(
        [lower, lower, lower, lower, *internal, upper, upper, upper, upper],
        dtype=np.float64,
    )


def _spline_basis(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(values, dtype=np.float64), float(knots[3]), float(knots[-4])
    )
    spline = BSpline(
        np.asarray(knots, dtype=np.float64),
        np.eye(6, dtype=np.float64),
        3,
        extrapolate=False,
    )
    basis = np.asarray(spline(clipped), dtype=np.float64)
    if basis.shape != (len(clipped), 6) or not np.isfinite(basis).all():
        raise ValueError("invalid cubic B-spline basis")
    if not np.allclose(basis.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("cubic B-spline basis does not form a partition")
    return basis


SECOND_DIFFERENCE_PENALTY = np.diff(np.eye(6, dtype=np.float64), n=2, axis=0)
SECOND_DIFFERENCE_PENALTY = (
    SECOND_DIFFERENCE_PENALTY.T @ SECOND_DIFFERENCE_PENALTY
)


def _effective_df(gram: np.ndarray, smoothing: float) -> float:
    penalized = gram + float(smoothing) * SECOND_DIFFERENCE_PENALTY
    mapping = np.linalg.pinv(penalized, rcond=1e-12) @ gram
    return float(2.0 * np.trace(mapping) - np.trace(mapping @ mapping))


def _smoothing_for_df(
    basis: np.ndarray,
    sample_weight: np.ndarray,
    target_df: float,
) -> tuple[float, float]:
    weight = np.asarray(sample_weight, dtype=np.float64)
    weight = weight / weight.sum()
    gram = np.einsum("n,ni,nj->ij", weight, basis, basis, optimize=True)
    unsmoothed = _effective_df(gram, 0.0)
    if unsmoothed <= target_df:
        return 0.0, unsmoothed
    upper = 1e-8
    while _effective_df(gram, upper) > target_df and upper < 1e8:
        upper *= 10.0
    if upper >= 1e8 and _effective_df(gram, upper) > target_df:
        raise ValueError("could not solve homogeneous spline smoothing")
    lower = 0.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if _effective_df(gram, midpoint) > target_df:
            lower = midpoint
        else:
            upper = midpoint
    smoothing = float(upper)
    return smoothing, _effective_df(gram, smoothing)


class AllFieldPreprocessor:
    """Source-only vocabularies, semantic scaling, and optional spline state."""

    def __init__(self, kind: str) -> None:
        if kind not in {"hofm", "ahofm"}:
            raise ValueError(f"unknown preprocessor kind: {kind}")
        self.kind = kind
        self.vocabularies: dict[str, list[Any]] = {}
        self.medians = np.empty(0, dtype=np.float64)
        self.means = np.empty(0, dtype=np.float64)
        self.scales = np.empty(0, dtype=np.float64)
        self.knot_vectors = np.empty((0, 10), dtype=np.float64)
        self.smoothing_lambdas = np.empty(0, dtype=np.float64)
        self.achieved_df = np.empty(0, dtype=np.float64)
        self.fit_rows = 0
        self.fit_seasons: list[int] = []
        self._fitted = False

    @property
    def categorical_cardinalities(self) -> list[int]:
        self._require_fitted()
        return [len(self.vocabularies[name]) + 1 for name in CATEGORICAL_FEATURES]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("preprocessor is not fitted")

    def fit(
        self,
        frame: pd.DataFrame,
        sample_weight: np.ndarray,
    ) -> "AllFieldPreprocessor":
        if not len(frame):
            raise ValueError("cannot fit an empty source frame")
        weight = np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != (len(frame),) or np.min(weight) <= 0.0:
            raise ValueError("invalid source sample weights")
        weight = weight / weight.sum()
        self.vocabularies = {}
        for column in CATEGORICAL_FEATURES:
            values = pd.Series(frame[column].dropna().unique()).tolist()
            values = sorted(
                values, key=lambda value: (type(value).__name__, str(value))
            )
            self.vocabularies[column] = [_json_scalar(value) for value in values]

        semantic, _ = semantic_continuous_matrix(frame)
        medians = np.nanmedian(semantic, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(semantic), semantic, medians)
        means = np.sum(filled * weight[:, None], axis=0)
        variances = np.sum(np.square(filled - means) * weight[:, None], axis=0)
        scales = np.sqrt(np.maximum(variances, 0.0))
        scales = np.where(scales > 1e-8, scales, 1.0)
        self.medians = medians.astype(np.float64)
        self.means = means.astype(np.float64)
        self.scales = scales.astype(np.float64)

        if self.kind == "ahofm":
            knots: list[np.ndarray] = []
            lambdas: list[float] = []
            achieved: list[float] = []
            for feature in range(filled.shape[1]):
                knot_vector = _open_quantile_knot_vector(filled[:, feature])
                basis = _spline_basis(filled[:, feature], knot_vector)
                smoothing, marginal_df = _smoothing_for_df(
                    basis, weight, target_df=4.0
                )
                knots.append(knot_vector)
                lambdas.append(smoothing)
                achieved.append(marginal_df)
            self.knot_vectors = np.stack(knots)
            self.smoothing_lambdas = np.asarray(lambdas, dtype=np.float64)
            self.achieved_df = np.asarray(achieved, dtype=np.float64)

        self.fit_rows = int(len(frame))
        self.fit_seasons = sorted(frame["season"].astype(int).unique().tolist())
        self._fitted = True
        return self

    def _categorical(self, frame: pd.DataFrame) -> np.ndarray:
        categorical = np.empty(
            (len(frame), len(CATEGORICAL_FEATURES)), dtype=np.int32
        )
        for index, column in enumerate(CATEGORICAL_FEATURES):
            vocabulary = self.vocabularies[column]
            codes = pd.Index(vocabulary).get_indexer(frame[column].to_numpy())
            categorical[:, index] = codes.astype(np.int32) + 1
        return categorical

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        self._require_fitted()
        semantic, missing = semantic_continuous_matrix(frame)
        filled = np.where(np.isfinite(semantic), semantic, self.medians)
        categorical = self._categorical(frame)
        if self.kind == "hofm":
            standardized = (filled - self.means) / self.scales
            numeric = np.concatenate(
                [standardized, missing.astype(np.float64)], axis=1
            ).astype(np.float32)
            return {"numeric": numeric, "categorical": categorical}

        basis = np.empty(
            (len(frame), len(CONTINUOUS_FEATURES), 6), dtype=np.float32
        )
        for feature, knot_vector in enumerate(self.knot_vectors):
            basis[:, feature, :] = _spline_basis(
                filled[:, feature], knot_vector
            ).astype(np.float32)
        return {
            "basis": basis,
            "missing": missing.astype(np.float32),
            "categorical": categorical,
        }

    def to_dict(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "version": 1,
            "kind": self.kind,
            "official_feature_columns": list(OFFICIAL_FEATURE_COLUMNS),
            "season_is_split_only": True,
            "categorical_features": list(CATEGORICAL_FEATURES),
            "continuous_features": list(CONTINUOUS_FEATURES),
            "vocabularies": self.vocabularies,
            "semantic_transform": {
                "nonnegative_log1p": list(NONNEGATIVE_LOG_COLUMNS),
                "signed_log1p": list(SIGNED_LOG_COLUMNS),
                "expectancy_divisor": 100.0,
                "li_log1p": True,
                "rates_clipped_0_1": list(RATE_COLUMNS),
            },
            "missing_strategy": "source_median_plus_per-field_missing_indicator",
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "spline": {
                "basis_count": 6,
                "degree": 3,
                "source_quantile_probabilities": [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
                "knot_vectors": self.knot_vectors.tolist(),
                "second_difference_penalty": SECOND_DIFFERENCE_PENALTY.tolist(),
                "homogeneous_target_df": 4.0,
                "smoothing_lambdas": self.smoothing_lambdas.tolist(),
                "achieved_df": self.achieved_df.tolist(),
            }
            if self.kind == "ahofm"
            else None,
            "fit_rows": self.fit_rows,
            "fit_seasons": self.fit_seasons,
        }


def _init_factor(parameter: torch.Tensor) -> None:
    nn.init.normal_(parameter, mean=0.0, std=0.02)


def _categorical_tables(
    cardinalities: Sequence[int], rank: int, *, linear: bool
) -> nn.ModuleList:
    output = nn.ModuleList()
    width = 1 if linear else rank
    for cardinality in cardinalities:
        embedding = nn.Embedding(int(cardinality), width, padding_idx=0)
        if linear:
            nn.init.zeros_(embedding.weight)
        else:
            _init_factor(embedding.weight)
            with torch.no_grad():
                embedding.weight[0].zero_()
        output.append(embedding)
    return output


def _anova_term(contributions: torch.Tensor, order: int) -> torch.Tensor:
    first = contributions.sum(dim=1)
    second = torch.square(contributions).sum(dim=1)
    if order == 2:
        per_factor = 0.5 * (torch.square(first) - second)
    elif order == 3:
        third = torch.pow(contributions, 3).sum(dim=1)
        per_factor = (
            torch.pow(first, 3) - 3.0 * first * second + 2.0 * third
        ) / 6.0
    else:
        raise ValueError(f"unsupported ANOVA order: {order}")
    return per_factor.sum(dim=1)


class HOFMModel(nn.Module):
    def __init__(
        self,
        cardinalities: Sequence[int],
        numeric_count: int,
        rank_2: int,
        rank_3: int,
    ) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear_categorical = _categorical_tables(
            cardinalities, 1, linear=True
        )
        self.factor_2_categorical = _categorical_tables(
            cardinalities, rank_2, linear=False
        )
        self.factor_3_categorical = _categorical_tables(
            cardinalities, rank_3, linear=False
        )
        self.linear_numeric = nn.Parameter(torch.zeros(numeric_count))
        self.factor_2_numeric = nn.Parameter(torch.empty(numeric_count, rank_2))
        self.factor_3_numeric = nn.Parameter(torch.empty(numeric_count, rank_3))
        _init_factor(self.factor_2_numeric)
        _init_factor(self.factor_3_numeric)

    def forward_components(
        self, numeric: torch.Tensor, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        linear = self.bias.expand(len(numeric))
        for index, embedding in enumerate(self.linear_categorical):
            linear = linear + embedding(categorical[:, index]).squeeze(1)
        linear = linear + (numeric * self.linear_numeric).sum(dim=1)

        categorical_2 = torch.stack(
            [
                embedding(categorical[:, index])
                for index, embedding in enumerate(self.factor_2_categorical)
            ],
            dim=1,
        )
        numeric_2 = numeric.unsqueeze(2) * self.factor_2_numeric.unsqueeze(0)
        order_2 = _anova_term(
            torch.cat([categorical_2, numeric_2], dim=1), 2
        )

        categorical_3 = torch.stack(
            [
                embedding(categorical[:, index])
                for index, embedding in enumerate(self.factor_3_categorical)
            ],
            dim=1,
        )
        numeric_3 = numeric.unsqueeze(2) * self.factor_3_numeric.unsqueeze(0)
        order_3 = _anova_term(
            torch.cat([categorical_3, numeric_3], dim=1), 3
        )
        return linear, order_2, order_3

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        linear, order_2, order_3 = self.forward_components(numeric, categorical)
        return linear + order_2 + order_3

    def linear_parameters(self) -> list[nn.Parameter]:
        return [
            self.linear_numeric,
            *(embedding.weight for embedding in self.linear_categorical),
        ]

    def factor_parameters(self) -> list[nn.Parameter]:
        return [
            self.factor_2_numeric,
            self.factor_3_numeric,
            *(embedding.weight for embedding in self.factor_2_categorical),
            *(embedding.weight for embedding in self.factor_3_categorical),
        ]

    def smoothness_penalty(self) -> torch.Tensor:
        return 0.0 * self.bias

    def zero_unknown_parameters(self) -> None:
        with torch.no_grad():
            for tables in (
                self.linear_categorical,
                self.factor_2_categorical,
                self.factor_3_categorical,
            ):
                for embedding in tables:
                    embedding.weight[0].zero_()


class AHOFMModel(nn.Module):
    def __init__(
        self,
        cardinalities: Sequence[int],
        continuous_count: int,
        smoothing_lambdas: np.ndarray,
        rank_2: int,
        rank_3: int,
    ) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear_categorical = _categorical_tables(
            cardinalities, 1, linear=True
        )
        self.factor_2_categorical = _categorical_tables(
            cardinalities, rank_2, linear=False
        )
        self.factor_3_categorical = _categorical_tables(
            cardinalities, rank_3, linear=False
        )
        self.linear_spline = nn.Parameter(torch.zeros(continuous_count, 6))
        self.linear_missing = nn.Parameter(torch.zeros(continuous_count))
        self.factor_2_spline = nn.Parameter(
            torch.empty(continuous_count, 6, rank_2)
        )
        self.factor_3_spline = nn.Parameter(
            torch.empty(continuous_count, 6, rank_3)
        )
        self.factor_2_missing = nn.Parameter(
            torch.empty(continuous_count, rank_2)
        )
        self.factor_3_missing = nn.Parameter(
            torch.empty(continuous_count, rank_3)
        )
        for parameter in (
            self.factor_2_spline,
            self.factor_3_spline,
            self.factor_2_missing,
            self.factor_3_missing,
        ):
            _init_factor(parameter)
        self.register_buffer(
            "smoothing_lambdas",
            torch.as_tensor(smoothing_lambdas, dtype=torch.float32),
        )

    def forward_components(
        self,
        basis: torch.Tensor,
        missing: torch.Tensor,
        categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        linear = self.bias.expand(len(basis))
        for index, embedding in enumerate(self.linear_categorical):
            linear = linear + embedding(categorical[:, index]).squeeze(1)
        linear = linear + torch.einsum("bjm,jm->b", basis, self.linear_spline)
        linear = linear + (missing * self.linear_missing).sum(dim=1)

        categorical_2 = torch.stack(
            [
                embedding(categorical[:, index])
                for index, embedding in enumerate(self.factor_2_categorical)
            ],
            dim=1,
        )
        spline_2 = torch.einsum("bjm,jmr->bjr", basis, self.factor_2_spline)
        missing_2 = missing.unsqueeze(2) * self.factor_2_missing.unsqueeze(0)
        order_2 = _anova_term(
            torch.cat([categorical_2, spline_2, missing_2], dim=1), 2
        )

        categorical_3 = torch.stack(
            [
                embedding(categorical[:, index])
                for index, embedding in enumerate(self.factor_3_categorical)
            ],
            dim=1,
        )
        spline_3 = torch.einsum("bjm,jmr->bjr", basis, self.factor_3_spline)
        missing_3 = missing.unsqueeze(2) * self.factor_3_missing.unsqueeze(0)
        order_3 = _anova_term(
            torch.cat([categorical_3, spline_3, missing_3], dim=1), 3
        )
        return linear, order_2, order_3

    def forward(
        self,
        basis: torch.Tensor,
        missing: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        linear, order_2, order_3 = self.forward_components(
            basis, missing, categorical
        )
        return linear + order_2 + order_3

    def linear_parameters(self) -> list[nn.Parameter]:
        return [
            self.linear_spline,
            self.linear_missing,
            *(embedding.weight for embedding in self.linear_categorical),
        ]

    def factor_parameters(self) -> list[nn.Parameter]:
        return [
            self.factor_2_spline,
            self.factor_3_spline,
            self.factor_2_missing,
            self.factor_3_missing,
            *(embedding.weight for embedding in self.factor_2_categorical),
            *(embedding.weight for embedding in self.factor_3_categorical),
        ]

    def _smooth_parameter(self, parameter: torch.Tensor) -> torch.Tensor:
        difference = parameter[:, 2:] - 2.0 * parameter[:, 1:-1] + parameter[:, :-2]
        reduce_dimensions = tuple(range(1, difference.ndim))
        per_feature = torch.square(difference).sum(dim=reduce_dimensions)
        return (self.smoothing_lambdas * per_feature).sum()

    def smoothness_penalty(self) -> torch.Tensor:
        return 0.5 * (
            self._smooth_parameter(self.linear_spline)
            + self._smooth_parameter(self.factor_2_spline)
            + self._smooth_parameter(self.factor_3_spline)
        )

    def zero_unknown_parameters(self) -> None:
        with torch.no_grad():
            for tables in (
                self.linear_categorical,
                self.factor_2_categorical,
                self.factor_3_categorical,
            ):
                for embedding in tables:
                    embedding.weight[0].zero_()


FactorModel = HOFMModel | AHOFMModel


def build_model(
    configuration: Mapping[str, object], preprocessor: AllFieldPreprocessor
) -> FactorModel:
    set_seeds(MODEL_SEED)
    if configuration["kind"] == "hofm":
        return HOFMModel(
            preprocessor.categorical_cardinalities,
            numeric_count=2 * len(CONTINUOUS_FEATURES),
            rank_2=int(configuration["order_2_rank"]),
            rank_3=int(configuration["order_3_rank"]),
        )
    return AHOFMModel(
        preprocessor.categorical_cardinalities,
        continuous_count=len(CONTINUOUS_FEATURES),
        smoothing_lambdas=preprocessor.smoothing_lambdas,
        rank_2=int(configuration["order_2_rank"]),
        rank_3=int(configuration["order_3_rank"]),
    )


def _tensor_batch(
    arrays: Mapping[str, np.ndarray],
    positions: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if "numeric" in arrays:
        return (
            torch.as_tensor(
                arrays["numeric"][positions], dtype=torch.float32, device=device
            ),
            torch.as_tensor(
                arrays["categorical"][positions], dtype=torch.long, device=device
            ),
        )
    return (
        torch.as_tensor(
            arrays["basis"][positions], dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            arrays["missing"][positions], dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            arrays["categorical"][positions], dtype=torch.long, device=device
        ),
    )


def train_model(
    model: FactorModel,
    arrays: Mapping[str, np.ndarray],
    target_residual: np.ndarray,
    sample_weight: np.ndarray,
    *,
    device: torch.device,
    configuration_name: str,
    outer: int,
) -> dict[str, object]:
    set_seeds(MODEL_SEED)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [model.bias],
                "weight_decay": 0.0,
                "group_name": "intercept",
            },
            {
                "params": model.linear_parameters(),
                "weight_decay": LINEAR_WEIGHT_DECAY,
                "group_name": "linear",
            },
            {
                "params": model.factor_parameters(),
                "weight_decay": FACTOR_WEIGHT_DECAY,
                "group_name": "factor",
            },
        ],
        lr=LEARNING_RATE,
    )
    target = np.asarray(target_residual, dtype=np.float32)
    weight = np.asarray(sample_weight, dtype=np.float32)
    if target.shape != weight.shape or len(target) != len(arrays["categorical"]):
        raise ValueError("training arrays are not aligned")
    generator = np.random.default_rng(MODEL_SEED)
    history: list[dict[str, object]] = []
    optimizer_steps = 0
    started = time.time()
    for epoch in range(EPOCHS):
        epoch_started = time.time()
        order = generator.permutation(len(target))
        weighted_data_loss = 0.0
        total_weight = 0.0
        smoothness_sum = 0.0
        model.train()
        for group_start in range(0, len(order), EFFECTIVE_BATCH_SIZE):
            group = order[group_start : group_start + EFFECTIVE_BATCH_SIZE]
            group_weight = float(weight[group].sum())
            optimizer.zero_grad(set_to_none=True)
            group_data_loss = 0.0
            for micro_start in range(0, len(group), MICRO_BATCH_SIZE):
                positions = group[micro_start : micro_start + MICRO_BATCH_SIZE]
                inputs = _tensor_batch(arrays, positions, device)
                residual = torch.as_tensor(
                    target[positions], dtype=torch.float32, device=device
                )
                current_weight = torch.as_tensor(
                    weight[positions], dtype=torch.float32, device=device
                )
                raw = model(*inputs)
                predicted_residual = (
                    INTEGRATION_WEIGHT * CORRECTION_CLIP * torch.tanh(raw)
                )
                squared = torch.square(predicted_residual - residual)
                weighted_sum = (squared * current_weight).sum()
                (weighted_sum / group_weight).backward()
                group_data_loss += float(weighted_sum.detach().cpu())
            smoothness = model.smoothness_penalty()
            smoothness.backward()
            optimizer.step()
            model.zero_unknown_parameters()
            optimizer_steps += 1
            weighted_data_loss += group_data_loss
            total_weight += group_weight
            smoothness_sum += float(smoothness.detach().cpu())
        history.append(
            {
                "epoch": epoch + 1,
                "weighted_data_mse": weighted_data_loss / total_weight,
                "mean_smoothness_penalty_per_step": (
                    smoothness_sum
                    / max(1, math.ceil(len(target) / EFFECTIVE_BATCH_SIZE))
                ),
                "seconds": time.time() - epoch_started,
            }
        )
        print(
            f"EXP-112 outer={outer} config={configuration_name} "
            f"epoch={epoch + 1}/{EPOCHS} "
            f"mse={history[-1]['weighted_data_mse']:.8f} "
            f"seconds={history[-1]['seconds']:.1f}",
            flush=True,
        )
    return {
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "linear_weight_decay": LINEAR_WEIGHT_DECAY,
        "factor_weight_decay": FACTOR_WEIGHT_DECAY,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "optimizer_steps": optimizer_steps,
        "history": history,
        "total_seconds": time.time() - started,
    }


def _cpu_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64, copy=True)


def freeze_numpy_state(model: FactorModel, kind: str) -> dict[str, object]:
    state: dict[str, object] = {
        "kind": kind,
        "bias": float(model.bias.detach().cpu()),
        "linear_categorical": [
            _cpu_array(embedding.weight).reshape(-1)
            for embedding in model.linear_categorical
        ],
        "factor_2_categorical": [
            _cpu_array(embedding.weight) for embedding in model.factor_2_categorical
        ],
        "factor_3_categorical": [
            _cpu_array(embedding.weight) for embedding in model.factor_3_categorical
        ],
    }
    if isinstance(model, HOFMModel):
        state.update(
            {
                "linear_numeric": _cpu_array(model.linear_numeric),
                "factor_2_numeric": _cpu_array(model.factor_2_numeric),
                "factor_3_numeric": _cpu_array(model.factor_3_numeric),
            }
        )
    else:
        state.update(
            {
                "linear_spline": _cpu_array(model.linear_spline),
                "linear_missing": _cpu_array(model.linear_missing),
                "factor_2_spline": _cpu_array(model.factor_2_spline),
                "factor_3_spline": _cpu_array(model.factor_3_spline),
                "factor_2_missing": _cpu_array(model.factor_2_missing),
                "factor_3_missing": _cpu_array(model.factor_3_missing),
            }
        )
    return state


def _state_arrays(value: object) -> Iterable[np.ndarray]:
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _state_arrays(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _state_arrays(nested)


def numpy_state_bytes(state: Mapping[str, object]) -> int:
    return int(sum(array.nbytes for array in _state_arrays(state)))


def _numpy_anova(
    contributions: Iterable[np.ndarray],
    *,
    rows: int,
    rank: int,
    order: int,
) -> np.ndarray:
    first = np.zeros((rows, rank), dtype=np.float64)
    second = np.zeros((rows, rank), dtype=np.float64)
    third = np.zeros((rows, rank), dtype=np.float64) if order == 3 else None
    count = 0
    for contribution in contributions:
        value = np.asarray(contribution, dtype=np.float64)
        if value.shape != (rows, rank):
            raise ValueError("invalid frozen factor contribution")
        first += value
        second += value * value
        if third is not None:
            third += value * value * value
        count += 1
    if count < order:
        raise ValueError("insufficient fields for ANOVA interaction")
    if order == 2:
        per_factor = 0.5 * (first * first - second)
    else:
        assert third is not None
        per_factor = (first * first * first - 3.0 * first * second + 2.0 * third) / 6.0
    output = np.zeros(rows, dtype=np.float64)
    for factor in range(rank):
        output += per_factor[:, factor]
    return output


def _hofm_contributions(
    arrays: Mapping[str, np.ndarray],
    state: Mapping[str, object],
    order: int,
) -> Iterable[np.ndarray]:
    categorical = arrays["categorical"]
    tables = state[f"factor_{order}_categorical"]
    assert isinstance(tables, list)
    for index, table in enumerate(tables):
        yield np.asarray(table)[categorical[:, index]]
    numeric = arrays["numeric"].astype(np.float64, copy=False)
    factors = np.asarray(state[f"factor_{order}_numeric"], dtype=np.float64)
    for feature in range(numeric.shape[1]):
        yield numeric[:, feature, None] * factors[feature]


def _ahofm_contributions(
    arrays: Mapping[str, np.ndarray],
    state: Mapping[str, object],
    order: int,
) -> Iterable[np.ndarray]:
    categorical = arrays["categorical"]
    tables = state[f"factor_{order}_categorical"]
    assert isinstance(tables, list)
    for index, table in enumerate(tables):
        yield np.asarray(table)[categorical[:, index]]
    basis = arrays["basis"].astype(np.float64, copy=False)
    spline_factors = np.asarray(
        state[f"factor_{order}_spline"], dtype=np.float64
    )
    for feature in range(basis.shape[1]):
        contribution = np.zeros(
            (len(basis), spline_factors.shape[2]), dtype=np.float64
        )
        for basis_index in range(6):
            contribution += (
                basis[:, feature, basis_index, None]
                * spline_factors[feature, basis_index]
            )
        yield contribution
    missing = arrays["missing"].astype(np.float64, copy=False)
    missing_factors = np.asarray(
        state[f"factor_{order}_missing"], dtype=np.float64
    )
    for feature in range(missing.shape[1]):
        yield missing[:, feature, None] * missing_factors[feature]


def numpy_raw_components(
    arrays: Mapping[str, np.ndarray], state: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categorical = arrays["categorical"]
    rows = len(categorical)
    linear = np.full(rows, float(state["bias"]), dtype=np.float64)
    linear_tables = state["linear_categorical"]
    assert isinstance(linear_tables, list)
    for index, table in enumerate(linear_tables):
        linear += np.asarray(table, dtype=np.float64)[categorical[:, index]]

    if state["kind"] == "hofm":
        numeric = arrays["numeric"].astype(np.float64, copy=False)
        linear_numeric = np.asarray(state["linear_numeric"], dtype=np.float64)
        for feature in range(numeric.shape[1]):
            linear += numeric[:, feature] * linear_numeric[feature]
        factor_2 = np.asarray(state["factor_2_numeric"])
        factor_3 = np.asarray(state["factor_3_numeric"])
        order_2 = _numpy_anova(
            _hofm_contributions(arrays, state, 2),
            rows=rows,
            rank=factor_2.shape[1],
            order=2,
        )
        order_3 = _numpy_anova(
            _hofm_contributions(arrays, state, 3),
            rows=rows,
            rank=factor_3.shape[1],
            order=3,
        )
    else:
        basis = arrays["basis"].astype(np.float64, copy=False)
        linear_spline = np.asarray(state["linear_spline"], dtype=np.float64)
        for feature in range(basis.shape[1]):
            for basis_index in range(6):
                linear += (
                    basis[:, feature, basis_index]
                    * linear_spline[feature, basis_index]
                )
        missing = arrays["missing"].astype(np.float64, copy=False)
        linear_missing = np.asarray(state["linear_missing"], dtype=np.float64)
        for feature in range(missing.shape[1]):
            linear += missing[:, feature] * linear_missing[feature]
        factor_2 = np.asarray(state["factor_2_spline"])
        factor_3 = np.asarray(state["factor_3_spline"])
        order_2 = _numpy_anova(
            _ahofm_contributions(arrays, state, 2),
            rows=rows,
            rank=factor_2.shape[2],
            order=2,
        )
        order_3 = _numpy_anova(
            _ahofm_contributions(arrays, state, 3),
            rows=rows,
            rank=factor_3.shape[2],
            order=3,
        )
    if not (
        np.isfinite(linear).all()
        and np.isfinite(order_2).all()
        and np.isfinite(order_3).all()
    ):
        raise ValueError("nonfinite frozen NumPy model output")
    return linear, order_2, order_3


@dataclass
class FrozenPredictor:
    preprocessor: AllFieldPreprocessor
    state: Mapping[str, object]

    def raw_components(
        self, rows: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return numpy_raw_components(self.preprocessor.transform(rows), self.state)

    def __call__(self, rows: pd.DataFrame, baseline: np.ndarray) -> np.ndarray:
        linear, order_2, order_3 = self.raw_components(rows)
        return bounded_candidate(baseline, linear + order_2 + order_3)


def source_payload(
    frame: pd.DataFrame, source_seasons: Sequence[int]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    residuals: list[np.ndarray] = []
    audit: dict[str, object] = {}
    for season in source_seasons:
        rows = fold_rows(frame, season)
        target, baseline = exp071_fold(season)
        observed = rows[TARGET_COLUMN].to_numpy(dtype=np.float64)
        if not np.array_equal(observed, target):
            raise ValueError(f"source target mismatch in {season}")
        frames.append(rows)
        residuals.append(target - baseline)
        audit[str(season)] = {
            "rows": int(len(rows)),
            "residual_mean": float(np.mean(target - baseline)),
            "residual_std": float(np.std(target - baseline)),
        }
    source = pd.concat(frames, ignore_index=True)
    residual = np.concatenate(residuals).astype(np.float64)
    weights = season_equal_weights(source["season"].to_numpy(dtype=np.int16))
    if len(source) != len(residual) or len(weights) != len(residual):
        raise ValueError("source payload alignment failure")
    audit["season_equal_weight_sums"] = {
        str(season): float(weights[source["season"].eq(season)].sum())
        for season in source_seasons
    }
    return source, residual, weights, audit


def save_model_bundle(
    model: FactorModel,
    preprocessor: AllFieldPreprocessor,
    *,
    configuration_name: str,
    configuration: Mapping[str, object],
    outer: int,
    training_report: Mapping[str, object],
) -> dict[str, object]:
    model_path = ARTIFACT_ROOT / f"model_{configuration_name}_{outer}.pt"
    preprocessor_path = (
        ARTIFACT_ROOT / f"preprocessor_{configuration_name}_{outer}.json"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {
        name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "experiment": "EXP-112",
            "configuration_name": configuration_name,
            "configuration": dict(configuration),
            "outer": outer,
            "state_dict": cpu_state,
            "training": dict(training_report),
            "preregistration_sha256": PREREGISTRATION_SHA256,
        },
        model_path,
    )
    json_dump(preprocessor_path, preprocessor.to_dict())
    return {
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": sha256_file(model_path),
        "preprocessor_path": str(preprocessor_path),
        "preprocessor_bytes": preprocessor_path.stat().st_size,
        "preprocessor_sha256": sha256_file(preprocessor_path),
    }


def parameter_audit(model: FactorModel) -> dict[str, int]:
    parameters = list(model.parameters())
    return {
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        "trainable_bytes": int(
            sum(
                parameter.numel() * parameter.element_size()
                for parameter in parameters
                if parameter.requires_grad
            )
        ),
    }


def secondary_exp051_control(
    target: np.ndarray, candidate: np.ndarray, season: int
) -> dict[str, float]:
    control = exp051_fold(season)
    candidate_brier = float(np.mean(np.square(target - candidate)))
    control_brier = float(np.mean(np.square(target - control)))
    return {
        "exp051_brier": control_brier,
        "candidate_brier": candidate_brier,
        "delta_brier_candidate_minus_exp051": candidate_brier - control_brier,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="training device only; scored inference always uses deterministic NumPy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    lock_audit = verify_locked_inputs()
    set_seeds(MODEL_SEED)
    device = select_device(args.device)

    requested_columns = list(
        dict.fromkeys([*OFFICIAL_FEATURE_COLUMNS, TARGET_COLUMN, *GAME_COLUMNS])
    )
    frame = load_official(requested_columns, seasons=(2022, 2023, 2024))
    if "row_id" in frame.columns:
        raise ValueError("row_id must not enter EXP-112")

    fold_reports: dict[str, dict[str, object]] = {
        name: {} for name in CONFIGURATION_ORDER
    }
    scored: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        name: {} for name in CONFIGURATION_ORDER
    }
    training_input_audit: dict[str, object] = {}
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    for outer in DIAGNOSTIC_SEASONS:
        source_seasons = SOURCE_SEASONS_BY_OUTER[outer]
        source, source_residual, weights, source_audit = source_payload(
            frame, source_seasons
        )
        training_input_audit[str(outer)] = source_audit
        validation = fold_rows(frame, outer)
        target, baseline = exp071_fold(outer)
        games = reconstructed_game_ids(validation)
        np.save(ARTIFACT_ROOT / f"targets_{outer}.npy", target)
        np.save(ARTIFACT_ROOT / f"predictions_exp071_{outer}.npy", baseline)

        for configuration_name in CONFIGURATION_ORDER:
            configuration = CONFIGURATIONS[configuration_name]
            fold_started = time.time()
            rss_before = peak_rss_mb()
            preprocessing_started = time.time()
            preprocessor = AllFieldPreprocessor(str(configuration["kind"]))
            preprocessor.fit(source, weights)
            source_arrays = preprocessor.transform(source)
            preprocessing_seconds = time.time() - preprocessing_started

            model = build_model(configuration, preprocessor)
            parameters = parameter_audit(model)
            training_report = train_model(
                model,
                source_arrays,
                source_residual,
                weights,
                device=device,
                configuration_name=configuration_name,
                outer=outer,
            )
            model.cpu()
            bundle = save_model_bundle(
                model,
                preprocessor,
                configuration_name=configuration_name,
                configuration=configuration,
                outer=outer,
                training_report=training_report,
            )
            frozen_state = freeze_numpy_state(model, str(configuration["kind"]))
            predictor = FrozenPredictor(preprocessor, frozen_state)

            inference_started = time.time()
            validation_arrays = preprocessor.transform(validation)
            linear, order_2, order_3 = numpy_raw_components(
                validation_arrays, frozen_state
            )
            raw = linear + order_2 + order_3
            candidate = bounded_candidate(baseline, raw)
            order_3_zero = bounded_candidate(baseline, linear + order_2)
            inference_seconds = time.time() - inference_started
            order_3_prediction_rms = float(
                np.sqrt(np.mean(np.square(candidate - order_3_zero)))
            )
            novelty = {
                "ablation": "order_3_contribution_zeroed",
                "threshold_prediction_rms": NOVELTY_RMS_THRESHOLD,
                "prediction_rms_change": order_3_prediction_rms,
                "raw_order_3_rms": float(np.sqrt(np.mean(np.square(order_3)))),
                "passed": order_3_prediction_rms >= NOVELTY_RMS_THRESHOLD,
            }
            independence = row_independence_audit(
                predictor, validation, baseline
            )
            metrics = diagnostic_metrics(
                target, candidate, baseline, games, season=outer
            )
            metrics["secondary_exp051_control"] = secondary_exp051_control(
                target, candidate, outer
            )

            prediction_path = (
                ARTIFACT_ROOT
                / f"predictions_{configuration_name}_{outer}.npy"
            )
            ablation_path = (
                ARTIFACT_ROOT
                / f"predictions_{configuration_name}_order3_zero_{outer}.npy"
            )
            order_3_path = (
                ARTIFACT_ROOT
                / f"raw_order3_{configuration_name}_{outer}.npy"
            )
            np.save(prediction_path, candidate)
            np.save(ablation_path, order_3_zero)
            np.save(order_3_path, order_3)
            fold_reports[configuration_name][str(outer)] = {
                "outer": outer,
                "source_seasons": list(source_seasons),
                "source_rows": int(len(source)),
                "validation_rows": int(len(validation)),
                "validation_labels_used_for_fit_or_selection": False,
                "configuration": dict(configuration),
                "preprocessing_seconds": preprocessing_seconds,
                "training": training_report,
                "inference_seconds": inference_seconds,
                "runtime_seconds": time.time() - fold_started,
                "peak_rss_mb_before": rss_before,
                "peak_rss_mb_after": peak_rss_mb(),
                "parameter_audit": parameters,
                "frozen_numpy_state_bytes": numpy_state_bytes(frozen_state),
                "bundle": bundle,
                "novelty_gate": novelty,
                "row_independence": independence,
                "metrics": metrics,
                "arrays": {
                    "prediction_path": str(prediction_path),
                    "prediction_sha256": sha256_file(prediction_path),
                    "order_3_zero_path": str(ablation_path),
                    "order_3_zero_sha256": sha256_file(ablation_path),
                    "raw_order_3_path": str(order_3_path),
                    "raw_order_3_sha256": sha256_file(order_3_path),
                },
            }
            scored[configuration_name][outer] = (target, candidate, baseline)
            del (
                source_arrays,
                validation_arrays,
                model,
                frozen_state,
                predictor,
                linear,
                order_2,
                order_3,
                raw,
                candidate,
                order_3_zero,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        del source, source_residual, weights, validation
        gc.collect()

    configuration_summary: dict[str, object] = {}
    survivor_flags: dict[str, bool] = {}
    for configuration_name in CONFIGURATION_ORDER:
        metrics_by_season = {
            season: fold_reports[configuration_name][str(season)]["metrics"]
            for season in DIAGNOSTIC_SEASONS
        }
        metric_gate = promotion_gate(metrics_by_season)
        novelty_passed = all(
            bool(
                fold_reports[configuration_name][str(season)]["novelty_gate"][
                    "passed"
                ]
            )
            for season in DIAGNOSTIC_SEASONS
        )
        exact_independence = all(
            bool(
                fold_reports[configuration_name][str(season)][
                    "row_independence"
                ]["literal_exact_identity_passed"]
            )
            for season in DIAGNOSTIC_SEASONS
        )
        cheap_survivor = bool(
            metric_gate["metric_survivor"]
            and novelty_passed
            and exact_independence
        )
        survivor_flags[configuration_name] = cheap_survivor
        configuration_summary[configuration_name] = {
            "pooled_2023_2024": pooled_metrics(scored[configuration_name]),
            "promotion_gate": metric_gate,
            "novelty_gate_passed_both_folds": novelty_passed,
            "literal_row_independence_passed_both_folds": exact_independence,
            "cheap_survivor": cheap_survivor,
        }

    selected = next(
        (name for name in CONFIGURATION_ORDER if survivor_flags[name]), None
    )
    report = {
        "experiment": "EXP-112",
        "candidate_family": "all-field order-3 HOFM/AHOFM residual",
        "stage": "cheap_outer_2023_2024",
        "lock": lock_audit,
        "protocol": {
            "primary_baseline": "immutable EXP-071 playerphys_resid_w025",
            "secondary_control": "EXP-051",
            "outer_source_seasons": {
                str(key): list(value)
                for key, value in SOURCE_SEASONS_BY_OUTER.items()
            },
            "source_season_equal_weight": True,
            "validation_rows": "full",
            "validation_labels_used_for_fit_or_selection": False,
            "season_used_as_predictive_feature": False,
            "row_id_used": False,
            "test_csv_opened": False,
            "current_or_query_trackman_used": False,
            "correction": "0.03*tanh(raw)",
            "integration_weight": INTEGRATION_WEIGHT,
            "epochs": EPOCHS,
            "seed": MODEL_SEED,
        },
        "feature_schema": {
            "official_feature_columns": list(OFFICIAL_FEATURE_COLUMNS),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "continuous_features": list(CONTINUOUS_FEATURES),
            "missing_indicators": list(CONTINUOUS_FEATURES),
        },
        "training_input_audit": training_input_audit,
        "configurations": fold_reports,
        "configuration_summary": configuration_summary,
        "selection": {
            "family_rule": (
                "F1-hofm3 primary wins when both configurations qualify; "
                "at most one family survivor slot"
            ),
            "selected_configuration": selected,
            "any_cheap_survivor": selected is not None,
            "full_training_authorized": selected is not None,
            "no_post_result_tuning": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "training_device": str(device),
            "scored_inference": "deterministic sequential NumPy float64",
        },
        "total_seconds": time.time() - started,
        "peak_rss_mb": peak_rss_mb(),
    }
    json_dump(REPORT_PATH, report)
    print(
        f"EXP-112 complete report={REPORT_PATH} selected={selected} "
        f"seconds={report['total_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
