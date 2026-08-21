"""Shared immutable evaluation utilities for the EXP-112+ model reset.

The module deliberately contains no model-selection logic.  It aligns every
candidate to the frozen EXP-071 OOF arrays, reconstructs target-free game
blocks, and reports the exact diagnostic set frozen in
``docs/MODEL_DISCOVERY_EXP112_ULTRA.md``.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
EXP071_ROOT = ROOT / "artifacts" / "EXP-071" / "partial_player_physics_residual"
EXP051_ROOT = ROOT / "artifacts" / "EXP-071" / "partial_player_physics_residual"
DIAGNOSTIC_SEASONS = (2023, 2024)
FULL_ROLLING_SEASONS = (2022, 2023, 2024)
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_BASE_SEED = 20_260_821
MODEL_SEED = 20_260_821
CORRECTION_CLIP = 0.03
INTEGRATION_WEIGHT = 0.25

GAME_COLUMNS = (
    "season",
    "inning",
    "top_bottom",
    "run_top_before",
    "run_bot_before",
    "pitcher_team_id",
    "batter_team_id",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def exp071_fold(season: int) -> tuple[np.ndarray, np.ndarray]:
    prediction_path = (
        EXP071_ROOT / f"predictions_playerphys_resid_w025_{season}.npy"
    )
    target_path = EXP071_ROOT / f"targets_{season}.npy"
    prediction = np.load(prediction_path).astype(np.float64)
    target = np.load(target_path).astype(np.float64)
    if prediction.ndim != 1 or target.ndim != 1 or len(prediction) != len(target):
        raise ValueError(f"invalid EXP-071 arrays for {season}")
    if not np.isfinite(prediction).all() or not np.isin(target, [0.0, 1.0]).all():
        raise ValueError(f"invalid EXP-071 values for {season}")
    return target, prediction


def exp051_fold(season: int) -> np.ndarray:
    prediction = np.load(EXP051_ROOT / f"predictions_base_{season}.npy").astype(
        np.float64
    )
    target, _ = exp071_fold(season)
    if prediction.shape != target.shape:
        raise ValueError(f"invalid EXP-051 array for {season}")
    return prediction


def load_official(
    columns: Iterable[str],
    *,
    seasons: Iterable[int] = FULL_ROLLING_SEASONS,
) -> pd.DataFrame:
    requested = list(dict.fromkeys([*columns, "season"]))
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=requested)
    allowed = set(int(value) for value in seasons)
    frame = frame.loc[frame["season"].isin(allowed)].reset_index(drop=True)
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("official rows must remain season-sorted")
    return frame


def fold_rows(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    rows = frame.loc[frame["season"].eq(season)].reset_index(drop=True)
    target, _ = exp071_fold(season)
    if len(rows) != len(target):
        raise ValueError(
            f"official/EXP-071 row mismatch in {season}: {len(rows)} != {len(target)}"
        )
    if "control_success" in rows:
        observed = rows["control_success"].to_numpy(dtype=np.float64)
        if not np.array_equal(observed, target):
            raise ValueError(f"official target order mismatch in {season}")
    return rows


def season_equal_weights(seasons: np.ndarray) -> np.ndarray:
    values = np.asarray(seasons)
    output = np.empty(len(values), dtype=np.float64)
    unique, counts = np.unique(values, return_counts=True)
    for season, count in zip(unique, counts, strict=True):
        output[values == season] = 1.0 / float(count)
    output *= len(output) / output.sum()
    return output


def bounded_candidate(baseline: np.ndarray, raw: np.ndarray) -> np.ndarray:
    reference = np.asarray(baseline, dtype=np.float64)
    score = np.asarray(raw, dtype=np.float64)
    if reference.shape != score.shape:
        raise ValueError("baseline/raw shape mismatch")
    correction = CORRECTION_CLIP * np.tanh(score)
    return np.clip(reference + INTEGRATION_WEIGHT * correction, 0.0, 1.0)


def reconstructed_game_ids(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(GAME_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"missing game columns: {missing}")
    season = frame["season"].to_numpy(dtype=np.int16)
    phase = 2 * (frame["inning"].to_numpy(dtype=np.int16) - 1)
    phase += frame["top_bottom"].astype(str).eq("B").to_numpy(dtype=np.int16)
    pitcher_team = frame["pitcher_team_id"].to_numpy(dtype=np.int16)
    batter_team = frame["batter_team_id"].to_numpy(dtype=np.int16)
    team_low = np.minimum(pitcher_team, batter_team)
    team_high = np.maximum(pitcher_team, batter_team)
    run_top = frame["run_top_before"].to_numpy(dtype=np.int16)
    run_bottom = frame["run_bot_before"].to_numpy(dtype=np.int16)
    boundary = np.zeros(len(frame), dtype=bool)
    if not len(frame):
        return np.empty(0, dtype=np.int64)
    boundary[0] = True
    boundary[1:] = (
        (season[1:] != season[:-1])
        | (team_low[1:] != team_low[:-1])
        | (team_high[1:] != team_high[:-1])
        | (phase[1:] < phase[:-1])
    )
    reset = np.zeros(len(frame), dtype=bool)
    reset[1:] = (
        (phase[1:] == 0)
        & (run_top[1:] == 0)
        & (run_bottom[1:] == 0)
        & ((phase[:-1] > 0) | (run_top[:-1] > 0) | (run_bottom[:-1] > 0))
    )
    boundary |= reset
    return np.cumsum(boundary, dtype=np.int64)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("correlation requires aligned vectors")
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def game_block_bootstrap(
    paired_loss: np.ndarray,
    games: np.ndarray,
    *,
    season: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    paired = np.asarray(paired_loss, dtype=np.float64)
    groups = np.asarray(games)
    if paired.ndim != 1 or groups.shape != paired.shape or not len(paired):
        raise ValueError("bootstrap requires aligned nonempty row vectors")
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=paired)
    counts = np.bincount(inverse)
    seed = BOOTSTRAP_BASE_SEED + int(season)
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        stop = min(replicates, start + 256)
        sampled = generator.integers(0, len(unique), size=(stop - start, len(unique)))
        estimates[start:stop] = sums[sampled].sum(axis=1) / counts[sampled].sum(
            axis=1
        )
    return {
        "games": int(len(unique)),
        "replicates": int(replicates),
        "seed": seed,
        "observed_delta_brier": float(paired.mean()),
        "ci95": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "probability_delta_brier_negative": float(np.mean(estimates < 0.0)),
    }


def diagnostic_metrics(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    games: np.ndarray,
    *,
    season: int,
) -> dict[str, object]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(candidate, dtype=np.float64)
    p0 = np.asarray(baseline, dtype=np.float64)
    if y.shape != p.shape or y.shape != p0.shape or y.ndim != 1:
        raise ValueError("metric vectors are not aligned")
    if not np.isfinite(p).all() or np.min(p) < 0.0 or np.max(p) > 1.0:
        raise ValueError("candidate probabilities are invalid")
    candidate_loss = np.square(y - p)
    baseline_loss = np.square(y - p0)
    paired = candidate_loss - baseline_loss
    rate = float(y.mean())
    uncertainty = rate * (1.0 - rate)
    brier = float(candidate_loss.mean())
    baseline_brier = float(baseline_loss.mean())
    correction = p - p0
    target_residual = y - p0
    denominator = float(np.dot(correction, correction))
    oracle_coefficient = (
        float(np.dot(correction, target_residual) / denominator)
        if denominator > 0.0
        else 0.0
    )
    oracle_coefficient_clipped = float(np.clip(oracle_coefficient, -4.0, 4.0))
    oracle_prediction = np.clip(
        p0 + oracle_coefficient_clipped * correction, 0.0, 1.0
    )
    oracle_brier = float(np.mean(np.square(y - oracle_prediction)))
    return {
        "rows": int(len(y)),
        "actual_rate": rate,
        "candidate_mean": float(p.mean()),
        "candidate_brier": brier,
        "baseline_brier": baseline_brier,
        "delta_brier_vs_exp071": float(paired.mean()),
        "candidate_skill": 100_000.0 * (1.0 - brier / uncertainty),
        "baseline_skill": 100_000.0 * (1.0 - baseline_brier / uncertainty),
        "delta_skill": 100_000.0 * (baseline_brier - brier) / uncertainty,
        "paired_loss_mean": float(paired.mean()),
        "paired_loss_std": float(paired.std(ddof=1)) if len(paired) > 1 else 0.0,
        "prediction_correlation": correlation(p, p0),
        "target_error_correlation": correlation(y - p, y - p0),
        "correction_target_residual_correlation": correlation(
            correction, target_residual
        ),
        "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
        "oracle": {
            "deployable": False,
            "coefficient_unclipped": oracle_coefficient,
            "coefficient_used": oracle_coefficient_clipped,
            "brier": oracle_brier,
            "gain_vs_exp071": baseline_brier - oracle_brier,
        },
        "game_block_bootstrap": game_block_bootstrap(
            paired, games, season=season
        ),
    }


def pooled_metrics(
    folds: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, object]:
    seasons = sorted(folds)
    target = np.concatenate([folds[s][0] for s in seasons])
    candidate = np.concatenate([folds[s][1] for s in seasons])
    baseline = np.concatenate([folds[s][2] for s in seasons])
    candidate_loss = np.square(target - candidate)
    baseline_loss = np.square(target - baseline)
    correction = candidate - baseline
    residual = target - baseline
    denominator = float(np.dot(correction, correction))
    coefficient = (
        float(np.dot(correction, residual) / denominator)
        if denominator > 0.0
        else 0.0
    )
    coefficient_used = float(np.clip(coefficient, -4.0, 4.0))
    oracle = np.clip(baseline + coefficient_used * correction, 0.0, 1.0)
    return {
        "seasons": seasons,
        "rows": int(len(target)),
        "candidate_brier": float(candidate_loss.mean()),
        "baseline_brier": float(baseline_loss.mean()),
        "delta_brier_vs_exp071": float((candidate_loss - baseline_loss).mean()),
        "prediction_correlation": correlation(candidate, baseline),
        "target_error_correlation": correlation(
            target - candidate, target - baseline
        ),
        "correction_target_residual_correlation": correlation(
            correction, residual
        ),
        "oracle": {
            "deployable": False,
            "coefficient_unclipped": coefficient,
            "coefficient_used": coefficient_used,
            "brier": float(np.mean(np.square(target - oracle))),
            "gain_vs_exp071": float(
                baseline_loss.mean() - np.mean(np.square(target - oracle))
            ),
        },
    }


def row_independence_audit(
    predict: Callable[[pd.DataFrame, np.ndarray], np.ndarray],
    rows: pd.DataFrame,
    baseline: np.ndarray,
    *,
    sample_rows: int = 64,
    seed: int = MODEL_SEED,
) -> dict[str, object]:
    count = min(sample_rows, len(rows))
    if count < 2:
        raise ValueError("row-independence audit needs at least two rows")
    indices = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    audit_rows = rows.iloc[indices].reset_index(drop=True)
    audit_base = np.asarray(baseline, dtype=np.float64)[indices]
    canonical = np.asarray(predict(audit_rows, audit_base), dtype=np.float64)
    singleton = np.concatenate(
        [
            np.asarray(
                predict(audit_rows.iloc[[index]], audit_base[[index]]),
                dtype=np.float64,
            )
            for index in range(count)
        ]
    )
    reverse_index = np.arange(count - 1, -1, -1)
    reversed_prediction = np.asarray(
        predict(audit_rows.iloc[reverse_index], audit_base[reverse_index]),
        dtype=np.float64,
    )[reverse_index]
    permutation = np.random.default_rng(seed).permutation(count)
    inverse = np.argsort(permutation)
    permuted = np.asarray(
        predict(audit_rows.iloc[permutation], audit_base[permutation]),
        dtype=np.float64,
    )[inverse]
    midpoint = count // 2
    split = np.concatenate(
        [
            np.asarray(
                predict(audit_rows.iloc[:midpoint], audit_base[:midpoint]),
                dtype=np.float64,
            ),
            np.asarray(
                predict(audit_rows.iloc[midpoint:], audit_base[midpoint:]),
                dtype=np.float64,
            ),
        ]
    )
    duplicated_rows = pd.concat([audit_rows, audit_rows], ignore_index=True)
    duplicated_base = np.concatenate([audit_base, audit_base])
    duplicated = np.asarray(
        predict(duplicated_rows, duplicated_base), dtype=np.float64
    )
    differences = {
        "singleton": float(np.max(np.abs(canonical - singleton))),
        "reverse": float(np.max(np.abs(canonical - reversed_prediction))),
        "permutation": float(np.max(np.abs(canonical - permuted))),
        "split": float(np.max(np.abs(canonical - split))),
        "duplicate_first": float(
            np.max(np.abs(canonical - duplicated[:count]))
        ),
        "duplicate_second": float(
            np.max(np.abs(canonical - duplicated[count:]))
        ),
    }
    maximum = max(differences.values())
    return {
        "rows": count,
        "seed": seed,
        "maximum_absolute_differences": differences,
        "maximum_absolute_difference": maximum,
        "literal_exact_identity_passed": maximum == 0.0,
        "semantic_tolerance_1e_12_passed": maximum <= 1e-12,
    }


def promotion_gate(fold_metrics: dict[int, dict[str, object]]) -> dict[str, object]:
    if sorted(fold_metrics) != [2023, 2024]:
        raise ValueError("promotion gate requires 2023 and 2024")
    deltas = [
        float(fold_metrics[season]["delta_brier_vs_exp071"])
        for season in (2023, 2024)
    ]
    rows = [int(fold_metrics[season]["rows"]) for season in (2023, 2024)]
    pooled_delta = float(np.average(deltas, weights=rows))
    probabilities = [
        float(
            fold_metrics[season]["game_block_bootstrap"][
                "probability_delta_brier_negative"
            ]
        )
        for season in (2023, 2024)
    ]
    tier_a = bool(
        all(delta < 0.0 for delta in deltas)
        and pooled_delta <= -1e-5
        and all(probability >= 0.60 for probability in probabilities)
    )
    tier_b = bool(
        pooled_delta <= -5e-5
        and max(deltas) <= 2e-5
        and min(deltas) <= -7.5e-5
    )
    errors = [
        float(fold_metrics[season]["target_error_correlation"])
        for season in (2023, 2024)
    ]
    oracle_gains = [
        float(fold_metrics[season]["oracle"]["gain_vs_exp071"])
        for season in (2023, 2024)
    ]
    pooled_oracle = float(np.average(oracle_gains, weights=rows))
    diversity = bool(
        max(deltas) <= 1e-4
        and max(errors) <= 0.995
        and pooled_oracle >= 2e-5
    )
    fast_kill = bool(all(delta > 1e-4 for delta in deltas))
    return {
        "season_delta_brier": {"2023": deltas[0], "2024": deltas[1]},
        "pooled_delta_brier": pooled_delta,
        "tier_a_route": tier_a,
        "tier_b_route": tier_b,
        "diversity_route": diversity,
        "fast_kill_both_above_1e_4": fast_kill,
        "metric_survivor": bool((tier_a or tier_b or diversity) and not fast_kill),
    }
