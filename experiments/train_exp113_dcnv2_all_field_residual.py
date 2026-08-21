"""EXP-113: preregistered DCNv2 all-field residual cheap falsification.

This file implements the single frozen ``D1-mix`` configuration from
``docs/MODEL_DISCOVERY_EXP112_ULTRA.md``.  It is intentionally not a tuning
script: outer 2023 trains on strict 2022 EXP-071 OOF residual supervision and
outer 2024 trains on equal-total-weight 2022/2023 supervision.  The candidate
is always ``clip(p071 + .25 * .03 * tanh(raw), 0, 1)``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ultra_model_common import (
    CORRECTION_CLIP,
    DIAGNOSTIC_SEASONS,
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
ARTIFACT_DIR = ROOT / "artifacts" / "EXP-113" / "dcnv2_all_field_residual"
PROTOCOL_PATH = ROOT / "docs" / "MODEL_DISCOVERY_EXP112_ULTRA.md"
PROTOCOL_SHA256 = "1bc9dd6384a721d93205521f6058ff4c0d368d1a2efbb5fa44a32942723184d0"

CATEGORICAL_FIELDS = (
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
)
NUMERIC_FIELDS = (
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
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)
FEATURE_FIELDS = (*CATEGORICAL_FIELDS, *NUMERIC_FIELDS)
READ_COLUMNS = tuple(
    dict.fromkeys(["season", *FEATURE_FIELDS, *GAME_COLUMNS, "control_success"])
)

PWL_KNOTS = 16
PWL_WIDTH = 8
PLAYER_WIDTH = 12
STRUCTURE_WIDTH = 4
OTHER_CATEGORY_WIDTH = 2
CROSS_LAYERS = 2
CROSS_EXPERTS = 4
CROSS_RANK = 32
DEEP_WIDTHS = (256, 128)
ID_DROPOUT = 0.15
DROPOUT = 0.10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 3e-4
BATCH_SIZE = 8_192
CHEAP_EPOCHS = 6
GRADIENT_CLIP = 5.0


def set_determinism(seed: int = MODEL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)


def category_width(name: str) -> int:
    if name in {"pitcher_id", "batter_id"}:
        return PLAYER_WIDTH
    if name in {
        "pitcher_team_id",
        "batter_team_id",
        "base_state",
        "game_month",
        "game_dayofweek",
        "inning",
    }:
        return STRUCTURE_WIDTH
    return OTHER_CATEGORY_WIDTH


def _category_key(value: object) -> str:
    if pd.isna(value):
        return "<MISSING>"
    return str(value)


@dataclass
class EncodedRows:
    categorical: np.ndarray
    numeric_left: np.ndarray
    numeric_fraction: np.ndarray
    numeric_missing: np.ndarray

    def subset(self, indices: np.ndarray | slice) -> "EncodedRows":
        return EncodedRows(
            categorical=self.categorical[indices],
            numeric_left=self.numeric_left[indices],
            numeric_fraction=self.numeric_fraction[indices],
            numeric_missing=self.numeric_missing[indices],
        )


class SourcePreprocessor:
    def __init__(self) -> None:
        self.category_maps: dict[str, dict[str, int]] = {}
        self.numeric_medians: dict[str, float] = {}
        self.numeric_knots: dict[str, np.ndarray] = {}

    def fit(self, frame: pd.DataFrame) -> "SourcePreprocessor":
        quantiles = np.linspace(0.0, 1.0, PWL_KNOTS)
        for column in CATEGORICAL_FIELDS:
            values = sorted({_category_key(value) for value in frame[column]})
            self.category_maps[column] = {
                value: index + 1 for index, value in enumerate(values)
            }
        for column in NUMERIC_FIELDS:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            filled = np.where(np.isfinite(values), values, median)
            knots = np.quantile(filled, quantiles).astype(np.float64)
            # Preserve the preregistered source quantiles while making every
            # interpolation interval numerically well-defined.
            scale = max(1.0, float(np.max(np.abs(knots))))
            minimum_step = np.finfo(np.float64).eps * scale * 16.0
            for index in range(1, len(knots)):
                if knots[index] <= knots[index - 1]:
                    knots[index] = knots[index - 1] + minimum_step
            self.numeric_medians[column] = median
            self.numeric_knots[column] = knots
        return self

    def transform(self, frame: pd.DataFrame) -> EncodedRows:
        count = len(frame)
        categorical = np.zeros(
            (count, len(CATEGORICAL_FIELDS)), dtype=np.int32
        )
        for field_index, column in enumerate(CATEGORICAL_FIELDS):
            mapping = self.category_maps[column]
            categorical[:, field_index] = np.fromiter(
                (mapping.get(_category_key(value), 0) for value in frame[column]),
                dtype=np.int32,
                count=count,
            )
        left = np.zeros((count, len(NUMERIC_FIELDS)), dtype=np.uint8)
        fraction = np.zeros((count, len(NUMERIC_FIELDS)), dtype=np.float32)
        missing = np.zeros((count, len(NUMERIC_FIELDS)), dtype=np.bool_)
        for field_index, column in enumerate(NUMERIC_FIELDS):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            invalid = ~np.isfinite(values)
            filled = np.where(invalid, self.numeric_medians[column], values)
            knots = self.numeric_knots[column]
            positions = np.searchsorted(knots, filled, side="right") - 1
            positions = np.clip(positions, 0, PWL_KNOTS - 2)
            low = knots[positions]
            high = knots[positions + 1]
            alpha = np.clip((filled - low) / (high - low), 0.0, 1.0)
            left[:, field_index] = positions.astype(np.uint8)
            fraction[:, field_index] = alpha.astype(np.float32)
            missing[:, field_index] = invalid
        return EncodedRows(categorical, left, fraction, missing)

    def state(self) -> dict[str, object]:
        return {
            "categorical_fields": list(CATEGORICAL_FIELDS),
            "numeric_fields": list(NUMERIC_FIELDS),
            "category_maps": self.category_maps,
            "numeric_medians": self.numeric_medians,
            "numeric_knots": {
                name: values.tolist() for name, values in self.numeric_knots.items()
            },
            "unknown_category_index": 0,
            "pwl_knots": PWL_KNOTS,
            "pwl_width": PWL_WIDTH,
            "source_only_fit": True,
        }


class PWLNumericEmbedding(nn.Module):
    def __init__(self, field_count: int) -> None:
        super().__init__()
        self.field_count = field_count
        self.weight = nn.Parameter(
            torch.empty(field_count, PWL_KNOTS, PWL_WIDTH)
        )
        self.missing = nn.Parameter(torch.zeros(field_count, PWL_WIDTH))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(
        self,
        left: torch.Tensor,
        fraction: torch.Tensor,
        missing: torch.Tensor,
    ) -> torch.Tensor:
        batch = left.shape[0]
        fields = torch.arange(self.field_count, device=left.device).view(1, -1)
        fields = fields.expand(batch, -1)
        low = self.weight[fields, left]
        high = self.weight[fields, left + 1]
        alpha = fraction.unsqueeze(-1)
        output = low + alpha * (high - low)
        return torch.where(
            missing.unsqueeze(-1),
            self.missing.unsqueeze(0).expand(batch, -1, -1),
            output,
        )


class CrossNetMixLayer(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.v = nn.Parameter(
            torch.empty(CROSS_EXPERTS, dimension, CROSS_RANK)
        )
        self.c = nn.Parameter(
            torch.empty(CROSS_EXPERTS, CROSS_RANK, CROSS_RANK)
        )
        self.u = nn.Parameter(
            torch.empty(CROSS_EXPERTS, CROSS_RANK, dimension)
        )
        self.bias = nn.Parameter(torch.zeros(CROSS_EXPERTS, dimension))
        self.gate = nn.Linear(dimension, CROSS_EXPERTS, bias=False)
        for parameter in (self.v, self.c, self.u):
            nn.init.xavier_uniform_(parameter)
        nn.init.zeros_(self.gate.weight)

    def forward(self, x0: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        projected = torch.tanh(torch.einsum("bd,edr->ber", current, self.v))
        projected = torch.tanh(torch.einsum("ber,erk->bek", projected, self.c))
        projected = torch.einsum("ber,erd->bed", projected, self.u)
        expert = x0.unsqueeze(1) * projected + self.bias.unsqueeze(0)
        gates = torch.softmax(self.gate(current), dim=1).unsqueeze(-1)
        return current + torch.sum(gates * expert, dim=1)


class DCNv2Residual(nn.Module):
    def __init__(self, category_sizes: list[int]) -> None:
        super().__init__()
        self.category_embeddings = nn.ModuleList(
            [
                nn.Embedding(size, category_width(name), padding_idx=0)
                for size, name in zip(
                    category_sizes, CATEGORICAL_FIELDS, strict=True
                )
            ]
        )
        for embedding in self.category_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.numeric_embedding = PWLNumericEmbedding(len(NUMERIC_FIELDS))
        dimension = sum(category_width(name) for name in CATEGORICAL_FIELDS)
        dimension += len(NUMERIC_FIELDS) * PWL_WIDTH
        self.input_dimension = dimension
        self.cross_layers = nn.ModuleList(
            [CrossNetMixLayer(dimension) for _ in range(CROSS_LAYERS)]
        )
        deep: list[nn.Module] = []
        current = dimension
        for width in DEEP_WIDTHS:
            deep.extend(
                [
                    nn.Linear(current, width),
                    nn.SiLU(),
                    nn.LayerNorm(width),
                    nn.Dropout(DROPOUT),
                ]
            )
            current = width
        self.deep = nn.Sequential(*deep)
        self.head = nn.Linear(dimension + DEEP_WIDTHS[-1], 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def representation(
        self,
        categorical: torch.Tensor,
        numeric_left: torch.Tensor,
        numeric_fraction: torch.Tensor,
        numeric_missing: torch.Tensor,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for index, embedding in enumerate(self.category_embeddings):
            codes = categorical[:, index]
            if self.training and CATEGORICAL_FIELDS[index] in {
                "pitcher_id",
                "batter_id",
            }:
                drop = torch.rand(codes.shape, device=codes.device) < ID_DROPOUT
                codes = torch.where(drop, torch.zeros_like(codes), codes)
            pieces.append(embedding(codes))
        numeric = self.numeric_embedding(
            numeric_left, numeric_fraction, numeric_missing
        )
        pieces.append(numeric.flatten(start_dim=1))
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        categorical: torch.Tensor,
        numeric_left: torch.Tensor,
        numeric_fraction: torch.Tensor,
        numeric_missing: torch.Tensor,
        *,
        zero_cross: bool = False,
    ) -> torch.Tensor:
        x0 = self.representation(
            categorical, numeric_left, numeric_fraction, numeric_missing
        )
        cross = x0
        for layer in self.cross_layers:
            cross = layer(x0, cross)
        if zero_cross:
            cross = torch.zeros_like(cross)
        deep = self.deep(x0)
        return self.head(torch.cat([cross, deep], dim=1)).squeeze(1)


def tensors(rows: EncodedRows, indices: np.ndarray | slice) -> tuple[torch.Tensor, ...]:
    selected = rows.subset(indices)
    return (
        torch.from_numpy(selected.categorical.astype(np.int64, copy=False)),
        torch.from_numpy(selected.numeric_left.astype(np.int64, copy=False)),
        torch.from_numpy(selected.numeric_fraction),
        torch.from_numpy(selected.numeric_missing),
    )


def train_model(
    model: DCNv2Residual,
    encoded: EncodedRows,
    target: np.ndarray,
    baseline: np.ndarray,
    weights: np.ndarray,
    *,
    fold_seed: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scale = CORRECTION_CLIP * INTEGRATION_WEIGHT
    history: list[dict[str, float]] = []
    generator = np.random.default_rng(fold_seed)
    for epoch in range(CHEAP_EPOCHS):
        order = generator.permutation(len(target))
        model.train()
        weighted_sum = 0.0
        weight_sum = 0.0
        started = time.perf_counter()
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            inputs = tensors(encoded, indices)
            y = torch.from_numpy(target[indices].astype(np.float32, copy=False))
            p0 = torch.from_numpy(baseline[indices].astype(np.float32, copy=False))
            weight = torch.from_numpy(weights[indices].astype(np.float32, copy=False))
            optimizer.zero_grad(set_to_none=True)
            raw = model(*inputs)
            prediction = p0 + scale * torch.tanh(raw)
            row_loss = torch.square((prediction - y) / scale)
            loss = torch.sum(weight * row_loss) / torch.sum(weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            batch_weight = float(torch.sum(weight).detach())
            weighted_sum += float(loss.detach()) * batch_weight
            weight_sum += batch_weight
        history.append(
            {
                "epoch": float(epoch + 1),
                "scaled_weighted_mse": weighted_sum / weight_sum,
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"epoch={epoch + 1}/{CHEAP_EPOCHS} "
            f"scaled_mse={history[-1]['scaled_weighted_mse']:.6f} "
            f"seconds={history[-1]['seconds']:.1f}",
            flush=True,
        )
    return history


def predict_encoded(
    model: DCNv2Residual,
    encoded: EncodedRows,
    baseline: np.ndarray,
    *,
    batch_size: int = BATCH_SIZE,
    zero_cross: bool = False,
    force_scalar: bool = False,
) -> np.ndarray:
    model.eval()
    size = 1 if force_scalar else batch_size
    raw_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(baseline), size):
            stop = min(len(baseline), start + size)
            raw = model(
                *tensors(encoded, slice(start, stop)), zero_cross=zero_cross
            )
            raw_parts.append(raw.detach().cpu().numpy().astype(np.float64))
    raw = np.concatenate(raw_parts)
    return bounded_candidate(baseline, raw)


def category_sizes(preprocessor: SourcePreprocessor) -> list[int]:
    return [
        len(preprocessor.category_maps[name]) + 1 for name in CATEGORICAL_FIELDS
    ]


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def source_bundle(
    frame: pd.DataFrame, seasons: tuple[int, ...]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    season_values: list[np.ndarray] = []
    for season in seasons:
        current = fold_rows(frame, season)
        target, baseline = exp071_fold(season)
        rows.append(current)
        targets.append(target)
        baselines.append(baseline)
        season_values.append(np.full(len(current), season, dtype=np.int16))
    all_seasons = np.concatenate(season_values)
    return (
        pd.concat(rows, ignore_index=True),
        np.concatenate(targets),
        np.concatenate(baselines),
        season_equal_weights(all_seasons),
    )


def validate_protocol_hash() -> None:
    actual = sha256_file(PROTOCOL_PATH)
    if actual != PROTOCOL_SHA256:
        raise RuntimeError(
            "preregistration changed after lock: " f"{actual} != {PROTOCOL_SHA256}"
        )


def run_fold(frame: pd.DataFrame, season: int) -> dict[str, Any]:
    fold_started = time.perf_counter()
    rss_before = peak_rss_mb()
    source_seasons = (2022,) if season == 2023 else (2022, 2023)
    source, source_target, source_baseline, source_weight = source_bundle(
        frame, source_seasons
    )
    validation = fold_rows(frame, season)
    target, baseline = exp071_fold(season)
    games = reconstructed_game_ids(validation)

    preprocessing_started = time.perf_counter()
    preprocessor = SourcePreprocessor().fit(source)
    source_encoded = preprocessor.transform(source)
    validation_encoded = preprocessor.transform(validation)
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    set_determinism(MODEL_SEED + season)
    model = DCNv2Residual(category_sizes(preprocessor))
    training_started = time.perf_counter()
    history = train_model(
        model,
        source_encoded,
        source_target,
        source_baseline,
        source_weight,
        fold_seed=MODEL_SEED + season,
    )
    training_seconds = time.perf_counter() - training_started

    inference_started = time.perf_counter()
    prediction = predict_encoded(model, validation_encoded, baseline)
    inference_seconds = time.perf_counter() - inference_started
    ablated = predict_encoded(
        model, validation_encoded, baseline, zero_cross=True
    )
    cross_ablation_rms = float(
        np.sqrt(np.mean(np.square(prediction - ablated)))
    )
    metrics = diagnostic_metrics(
        target, prediction, baseline, games, season=season
    )
    metrics_exp051 = diagnostic_metrics(
        target, prediction, exp051_fold(season), games, season=season
    )

    def audited_predict(rows: pd.DataFrame, p0: np.ndarray) -> np.ndarray:
        encoded = preprocessor.transform(rows)
        return predict_encoded(
            model, encoded, p0, force_scalar=True, batch_size=1
        )

    independence = row_independence_audit(
        audited_predict, validation, baseline, sample_rows=64
    )
    audit_indices = np.linspace(0, len(validation) - 1, 64, dtype=np.int64)
    scalar_audit = audited_predict(
        validation.iloc[audit_indices].reset_index(drop=True), baseline[audit_indices]
    )
    vector_audit = predict_encoded(
        model,
        preprocessor.transform(
            validation.iloc[audit_indices].reset_index(drop=True)
        ),
        baseline[audit_indices],
    )
    independence["vectorized_vs_scalar_max_difference"] = float(
        np.max(np.abs(vector_audit - scalar_audit))
    )

    fold_dir = ARTIFACT_DIR / f"fold_{season}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    json_dump(fold_dir / "preprocessor.json", preprocessor.state())
    np.save(fold_dir / "predictions.npy", prediction)
    np.save(fold_dir / "predictions_exp071.npy", baseline)
    np.save(fold_dir / "targets.npy", target.astype(np.int8))

    return {
        "validation_season": season,
        "source_seasons": list(source_seasons),
        "source_rows": int(len(source)),
        "source_season_equal_weight": True,
        "validation_rows": int(len(validation)),
        "metrics_vs_exp071": metrics,
        "metrics_vs_exp051": metrics_exp051,
        "cross_ablation_prediction_rms": cross_ablation_rms,
        "cross_novelty_gate_rms_at_least_1e_4": cross_ablation_rms >= 1e-4,
        "row_independence": independence,
        "training_history": history,
        "runtime": {
            "preprocessing_seconds": preprocessing_seconds,
            "training_seconds": training_seconds,
            "validation_inference_seconds": inference_seconds,
            "validation_rows_per_second": len(validation) / inference_seconds,
            "fold_total_seconds": time.perf_counter() - fold_started,
        },
        "memory": {
            "peak_rss_before_mb": rss_before,
            "peak_rss_after_mb": peak_rss_mb(),
        },
        "model": {
            "parameters": model_parameter_count(model),
            "bytes": model_path.stat().st_size,
            "input_dimension": model.input_dimension,
            "path": str(model_path.relative_to(ROOT)),
            "sha256": sha256_file(model_path),
        },
        "preprocessor_path": str(
            (fold_dir / "preprocessor.json").relative_to(ROOT)
        ),
    }


def main() -> None:
    validate_protocol_hash()
    set_determinism()
    started = time.perf_counter()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_official(READ_COLUMNS, seasons=(2022, 2023, 2024))
    folds: dict[int, dict[str, Any]] = {}
    pooled_arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for season in DIAGNOSTIC_SEASONS:
        print(f"EXP-113 cheap outer fold {season}", flush=True)
        folds[season] = run_fold(frame, season)
        fold_dir = ARTIFACT_DIR / f"fold_{season}"
        pooled_arrays[season] = (
            np.load(fold_dir / "targets.npy").astype(np.float64),
            np.load(fold_dir / "predictions.npy").astype(np.float64),
            np.load(fold_dir / "predictions_exp071.npy").astype(np.float64),
        )
        metrics = folds[season]["metrics_vs_exp071"]
        print(
            f"fold={season} brier={metrics['candidate_brier']:.12f} "
            f"delta={metrics['delta_brier_vs_exp071']:+.12f} "
            f"skill={metrics['candidate_skill']:.3f}",
            flush=True,
        )

    metric_map = {
        season: folds[season]["metrics_vs_exp071"]
        for season in DIAGNOSTIC_SEASONS
    }
    gate = promotion_gate(metric_map)
    novelty = all(
        bool(folds[season]["cross_novelty_gate_rms_at_least_1e_4"])
        for season in DIAGNOSTIC_SEASONS
    )
    exact_independence = all(
        bool(
            folds[season]["row_independence"][
                "literal_exact_identity_passed"
            ]
        )
        for season in DIAGNOSTIC_SEASONS
    )
    report = {
        "experiment": "EXP-113",
        "candidate": "D1-mix",
        "family": "dcnv2_all_field_exp071_residual",
        "status": "cheap_falsification_complete",
        "preregistration": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "sha256": PROTOCOL_SHA256,
            "validated_before_run": True,
        },
        "configuration": {
            "categorical_fields": list(CATEGORICAL_FIELDS),
            "numeric_fields": list(NUMERIC_FIELDS),
            "excluded_predictors": ["row_id", "season", "control_success"],
            "pwl_knots": PWL_KNOTS,
            "pwl_width": PWL_WIDTH,
            "player_embedding_width": PLAYER_WIDTH,
            "structure_embedding_width": STRUCTURE_WIDTH,
            "other_category_width": OTHER_CATEGORY_WIDTH,
            "id_dropout": ID_DROPOUT,
            "cross_layers": CROSS_LAYERS,
            "cross_experts": CROSS_EXPERTS,
            "cross_rank": CROSS_RANK,
            "deep_widths": list(DEEP_WIDTHS),
            "dropout": DROPOUT,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": CHEAP_EPOCHS,
            "seed": MODEL_SEED,
            "correction": (
                "clip(p071 + 0.25 * 0.03 * tanh(raw), 0, 1)"
            ),
        },
        "validation_contract": {
            "cheap_folds": list(DIAGNOSTIC_SEASONS),
            "source_exp071_oof_strictly_prior": True,
            "source_season_equal_weight": True,
            "validation_labels_used_for_fit_or_selection": False,
            "query_peer_features": False,
            "public_score_used": False,
        },
        "folds": {str(season): value for season, value in folds.items()},
        "pooled_2023_2024": pooled_metrics(pooled_arrays),
        "promotion_gate": gate,
        "novelty_gate_all_folds": novelty,
        "literal_row_independence_all_folds": exact_independence,
        "survivor": bool(
            gate["metric_survivor"] and novelty and exact_independence
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "total_seconds": time.perf_counter() - started,
        "peak_rss_mb": peak_rss_mb(),
    }
    json_dump(ARTIFACT_DIR / "validation_metrics.json", report)
    print(
        f"EXP-113 survivor={report['survivor']} "
        f"pooled_delta={report['pooled_2023_2024']['delta_brier_vs_exp071']:+.12f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered EXP-113 cheap DCNv2 screen."
    )
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    main()
