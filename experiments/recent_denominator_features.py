"""Row-independent reliability features for rounded recent-game rates.

The official ``prev1/3/5_game`` success and middle rates for one window are
computed from the same pitches.  Their smallest common rational denominator
therefore supplies a conservative lower bound on the number of observations
behind the two rounded rates.  This module recovers that denominator using
only values from the current row.

The unique-pair cache below is only a deterministic speed optimization.  A
row's output is identical when it is transformed alone; no frequency, order,
distribution, target, or other evaluation-row value affects the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SIGNIFICANT_DIGITS = 6
SHRINKAGE_STRENGTHS = (10.0, 30.0, 100.0)


@dataclass(frozen=True)
class RecentWindow:
    name: str
    success_column: str
    middle_column: str
    max_denominator: int


WINDOWS = (
    RecentWindow(
        "prev1",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        180,
    ),
    RecentWindow(
        "prev3",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev3_game_middle_rate",
        450,
    ),
    RecentWindow(
        "prev5",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev5_game_middle_rate",
        700,
    ),
)


def _rate_matches(
    count: int,
    denominator: int,
    observed_rate: float,
) -> bool:
    """Match the six-significant-digit formatting used by the CSV rates."""
    reconstructed = float(
        format(count / denominator, f".{SIGNIFICANT_DIGITS}g")
    )
    return reconstructed == float(observed_rate)


def _pair_candidate_values(
    success_rate: float,
    middle_rate: float,
    max_denominator: int,
) -> list[tuple[int, int, int]]:
    """Return denominators consistent with the official rounded rates."""
    candidates: list[tuple[int, int, int]] = []
    for denominator in range(1, max_denominator + 1):
        success_count = int(np.rint(success_rate * denominator))
        middle_count = int(np.rint(middle_rate * denominator))
        if (
            _rate_matches(success_count, denominator, success_rate)
            and _rate_matches(middle_count, denominator, middle_rate)
        ):
            candidates.append(
                (denominator, success_count, middle_count)
            )
    return candidates


def _pair_candidates(
    success_rate: float,
    middle_rate: float,
    max_denominator: int,
) -> tuple[int, int, int, int]:
    """Return smallest denominator, candidate count, and its numerators."""
    candidates = _pair_candidate_values(
        success_rate,
        middle_rate,
        max_denominator,
    )
    if not candidates:
        return 0, 0, 0, 0
    first, first_success_count, first_middle_count = candidates[0]
    return (
        first,
        len(candidates),
        first_success_count,
        first_middle_count,
    )


def infer_minimum_common_denominator(
    success_rate: np.ndarray,
    middle_rate: np.ndarray,
    max_denominator: int,
) -> dict[str, np.ndarray]:
    """Infer a deterministic lower-bound denominator for every row."""
    success = np.asarray(success_rate, dtype=np.float64)
    middle = np.asarray(middle_rate, dtype=np.float64)
    if success.shape != middle.shape:
        raise ValueError("success and middle rate shapes differ")
    valid = (
        np.isfinite(success)
        & np.isfinite(middle)
        & (success >= 0.0)
        & (success <= 1.0)
        & (middle >= 0.0)
        & (middle <= 1.0)
    )
    denominator = np.zeros(success.shape, dtype=np.int16)
    candidate_count = np.zeros(success.shape, dtype=np.int16)
    success_count = np.zeros(success.shape, dtype=np.int16)
    middle_count = np.zeros(success.shape, dtype=np.int16)
    if valid.any():
        pairs = np.stack([success[valid], middle[valid]], axis=1)
        unique_pairs, inverse = np.unique(
            pairs, axis=0, return_inverse=True
        )
        lookup = np.asarray(
            [
                _pair_candidates(
                    float(success_rate),
                    float(middle_rate),
                    max_denominator,
                )
                for success_rate, middle_rate in unique_pairs
            ],
            dtype=np.int32,
        )
        denominator[valid] = lookup[inverse, 0].astype(np.int16)
        candidate_count[valid] = lookup[inverse, 1].astype(np.int16)
        success_count[valid] = lookup[inverse, 2].astype(np.int16)
        middle_count[valid] = lookup[inverse, 3].astype(np.int16)
    found = denominator > 0
    return {
        "denominator": denominator,
        "candidate_count": candidate_count,
        "success_count": success_count,
        "middle_count": middle_count,
        "valid_rate_pair": valid,
        "denominator_found": found,
    }


def infer_joint_window_denominators(
    success_rates: np.ndarray,
    middle_rates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Infer the smallest jointly feasible prev1/prev3/prev5 triplet.

    A three-game window contains the one-game window plus at least one pitch
    in each of two additional games.  The analogous constraint holds between
    the three- and five-game windows.  These deterministic constraints often
    resolve a reduced fraction in the longer window without looking at any
    other row.
    """
    success = np.asarray(success_rates, dtype=np.float64)
    middle = np.asarray(middle_rates, dtype=np.float64)
    if success.shape != middle.shape or success.ndim != 2:
        raise ValueError("joint rate arrays must have identical 2D shapes")
    if success.shape[1] != len(WINDOWS):
        raise ValueError("joint rate arrays must contain prev1/prev3/prev5")
    valid = (
        np.isfinite(success).all(axis=1)
        & np.isfinite(middle).all(axis=1)
        & (success >= 0.0).all(axis=1)
        & (success <= 1.0).all(axis=1)
        & (middle >= 0.0).all(axis=1)
        & (middle <= 1.0).all(axis=1)
    )
    denominators = np.zeros(success.shape, dtype=np.int16)
    success_counts = np.zeros(success.shape, dtype=np.int16)
    middle_counts = np.zeros(success.shape, dtype=np.int16)
    found = np.zeros(success.shape[0], dtype=bool)
    if not valid.any():
        return {
            "denominators": denominators,
            "success_counts": success_counts,
            "middle_counts": middle_counts,
            "valid_all_rate_pairs": valid,
            "joint_found": found,
        }

    vectors = np.concatenate([success[valid], middle[valid]], axis=1)
    unique_vectors, inverse = np.unique(
        vectors, axis=0, return_inverse=True
    )
    pair_cache: dict[
        tuple[int, float, float], list[tuple[int, int, int]]
    ] = {}
    unique_results = np.zeros((len(unique_vectors), 10), dtype=np.int32)
    for row_index, keys in enumerate(unique_vectors):
        candidate_lists: list[list[tuple[int, int, int]]] = []
        for window_index, window in enumerate(WINDOWS):
            cache_key = (
                window_index,
                float(keys[window_index]),
                float(keys[window_index + 3]),
            )
            candidates = pair_cache.get(cache_key)
            if candidates is None:
                candidates = _pair_candidate_values(
                    cache_key[1],
                    cache_key[2],
                    window.max_denominator,
                )
                pair_cache[cache_key] = candidates
            candidate_lists.append(candidates)
        if any(not candidates for candidates in candidate_lists):
            continue
        selected = None
        for prev1 in candidate_lists[0]:
            for prev3 in candidate_lists[1]:
                if not (
                    prev3[0] >= prev1[0] + 2
                    and prev3[1] >= prev1[1]
                    and prev3[2] >= prev1[2]
                ):
                    continue
                prev5 = next(
                    (
                        candidate
                        for candidate in candidate_lists[2]
                        if candidate[0] >= prev3[0] + 2
                        and candidate[1] >= prev3[1]
                        and candidate[2] >= prev3[2]
                    ),
                    None,
                )
                if prev5 is not None:
                    selected = (prev1, prev3, prev5)
                    break
            if selected is not None:
                break
        if selected is None:
            continue
        unique_results[row_index, 0] = 1
        for window_index, candidate in enumerate(selected):
            unique_results[row_index, 1 + window_index] = candidate[0]
            unique_results[row_index, 4 + window_index] = candidate[1]
            unique_results[row_index, 7 + window_index] = candidate[2]

    mapped = unique_results[inverse]
    valid_positions = np.flatnonzero(valid)
    found[valid_positions] = mapped[:, 0].astype(bool)
    denominators[valid_positions] = mapped[:, 1:4].astype(np.int16)
    success_counts[valid_positions] = mapped[:, 4:7].astype(np.int16)
    middle_counts[valid_positions] = mapped[:, 7:10].astype(np.int16)
    return {
        "denominators": denominators,
        "success_counts": success_counts,
        "middle_counts": middle_counts,
        "valid_all_rate_pairs": valid,
        "joint_found": found,
    }


def attach_recent_denominator_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    """Attach row-only recent-window reliability and posterior features."""
    required = {
        "asof_pitcher_success_rate",
        "asof_pitcher_middle_rate",
        *(
            column
            for window in WINDOWS
            for column in (window.success_column, window.middle_column)
        ),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing recent denominator inputs: {missing}")

    out = frame.copy()
    long_success = (
        out["asof_pitcher_success_rate"]
        .fillna(0.5)
        .clip(0.0, 1.0)
        .to_numpy(dtype=np.float64)
    )
    long_middle = (
        out["asof_pitcher_middle_rate"]
        .fillna(0.15)
        .clip(0.0, 1.0)
        .to_numpy(dtype=np.float64)
    )
    diagnostics: dict[str, dict[str, float | int]] = {}
    lower_denominators: list[np.ndarray] = []
    recent_success_rates: list[np.ndarray] = []
    recent_middle_rates: list[np.ndarray] = []

    for window in WINDOWS:
        success_rate = out[window.success_column].to_numpy(
            dtype=np.float64
        )
        middle_rate = out[window.middle_column].to_numpy(
            dtype=np.float64
        )
        recent_success_rates.append(success_rate)
        recent_middle_rates.append(middle_rate)
        inferred = infer_minimum_common_denominator(
            success_rate,
            middle_rate,
            window.max_denominator,
        )
        denominator = inferred["denominator"].astype(np.float64)
        lower_denominators.append(denominator)
        found = inferred["denominator_found"]
        candidate_count = inferred["candidate_count"].astype(np.float64)
        prefix = f"recent_den_{window.name}"
        out[f"{prefix}_found"] = found.astype(np.int8)
        out[f"{prefix}_min_n"] = denominator.astype(np.float32)
        out[f"{prefix}_log_min_n"] = np.log1p(denominator).astype(
            np.float32
        )
        out[f"{prefix}_candidate_count"] = candidate_count.astype(
            np.float32
        )
        out[f"{prefix}_inverse_ambiguity"] = np.divide(
            1.0,
            candidate_count,
            out=np.zeros(len(out), dtype=np.float64),
            where=candidate_count > 0,
        ).astype(np.float32)
        out[f"{prefix}_success_count_lower"] = inferred[
            "success_count"
        ].astype(np.float32)
        out[f"{prefix}_middle_count_lower"] = inferred[
            "middle_count"
        ].astype(np.float32)

        safe_success = np.where(
            np.isfinite(success_rate), success_rate, long_success
        )
        safe_middle = np.where(
            np.isfinite(middle_rate), middle_rate, long_middle
        )
        for strength in SHRINKAGE_STRENGTHS:
            suffix = int(strength)
            reliability = denominator / (denominator + strength)
            reliability = np.where(found, reliability, 0.0)
            success_posterior = (
                safe_success * denominator + strength * long_success
            ) / (denominator + strength)
            middle_posterior = (
                safe_middle * denominator + strength * long_middle
            ) / (denominator + strength)
            success_posterior = np.where(
                found, success_posterior, long_success
            )
            middle_posterior = np.where(
                found, middle_posterior, long_middle
            )
            out[f"{prefix}_reliability_{suffix}"] = (
                reliability.astype(np.float32)
            )
            out[f"{prefix}_success_posterior_{suffix}"] = (
                success_posterior.astype(np.float32)
            )
            out[f"{prefix}_success_delta_{suffix}"] = (
                success_posterior - long_success
            ).astype(np.float32)
            out[f"{prefix}_middle_posterior_{suffix}"] = (
                middle_posterior.astype(np.float32)
            )

        diagnostics[window.name] = {
            "rows": int(len(out)),
            "valid_rate_pair_rows": int(
                inferred["valid_rate_pair"].sum()
            ),
            "denominator_found_rows": int(found.sum()),
            "found_rate_among_valid": float(
                found.sum()
                / max(int(inferred["valid_rate_pair"].sum()), 1)
            ),
            "median_min_denominator_found": float(
                np.median(denominator[found]) if found.any() else 0.0
            ),
            "median_candidate_count_found": float(
                np.median(candidate_count[found]) if found.any() else 0.0
            ),
            "max_denominator_searched": window.max_denominator,
        }

    joint = infer_joint_window_denominators(
        np.stack(recent_success_rates, axis=1),
        np.stack(recent_middle_rates, axis=1),
    )
    joint_found = joint["joint_found"]
    for window_index, window in enumerate(WINDOWS):
        prefix = f"recent_den_joint_{window.name}"
        denominator = joint["denominators"][:, window_index].astype(
            np.float64
        )
        success_rate = recent_success_rates[window_index]
        middle_rate = recent_middle_rates[window_index]
        safe_success = np.where(
            np.isfinite(success_rate), success_rate, long_success
        )
        safe_middle = np.where(
            np.isfinite(middle_rate), middle_rate, long_middle
        )
        out[f"{prefix}_n"] = denominator.astype(np.float32)
        out[f"{prefix}_log_n"] = np.log1p(denominator).astype(
            np.float32
        )
        out[f"{prefix}_adjusted_from_individual"] = (
            joint_found
            & (denominator != lower_denominators[window_index])
        ).astype(np.int8)
        out[f"{prefix}_success_count"] = joint[
            "success_counts"
        ][:, window_index].astype(np.float32)
        out[f"{prefix}_middle_count"] = joint[
            "middle_counts"
        ][:, window_index].astype(np.float32)
        for strength in SHRINKAGE_STRENGTHS:
            suffix = int(strength)
            reliability = np.where(
                joint_found,
                denominator / (denominator + strength),
                0.0,
            )
            success_posterior = (
                safe_success * denominator + strength * long_success
            ) / (denominator + strength)
            middle_posterior = (
                safe_middle * denominator + strength * long_middle
            ) / (denominator + strength)
            success_posterior = np.where(
                joint_found, success_posterior, long_success
            )
            middle_posterior = np.where(
                joint_found, middle_posterior, long_middle
            )
            out[f"{prefix}_reliability_{suffix}"] = (
                reliability.astype(np.float32)
            )
            out[f"{prefix}_success_posterior_{suffix}"] = (
                success_posterior.astype(np.float32)
            )
            out[f"{prefix}_success_delta_{suffix}"] = (
                success_posterior - long_success
            ).astype(np.float32)
            out[f"{prefix}_middle_posterior_{suffix}"] = (
                middle_posterior.astype(np.float32)
            )
    out["recent_den_joint_found"] = joint_found.astype(np.int8)
    diagnostics["joint"] = {
        "rows": int(len(out)),
        "valid_all_rate_pairs_rows": int(
            joint["valid_all_rate_pairs"].sum()
        ),
        "joint_found_rows": int(joint_found.sum()),
        "joint_found_rate_among_valid": float(
            joint_found.sum()
            / max(int(joint["valid_all_rate_pairs"].sum()), 1)
        ),
        "prev3_adjusted_rows": int(
            (
                joint_found
                & (
                    joint["denominators"][:, 1]
                    != lower_denominators[1]
                )
            ).sum()
        ),
        "prev5_adjusted_rows": int(
            (
                joint_found
                & (
                    joint["denominators"][:, 2]
                    != lower_denominators[2]
                )
            ).sum()
        ),
        "minimum_additional_games_constraint": 2,
    }

    monotone_lower = np.maximum.accumulate(
        np.stack(lower_denominators, axis=1), axis=1
    )
    for index, window in enumerate(WINDOWS):
        out[f"recent_den_{window.name}_monotone_lower_n"] = (
            monotone_lower[:, index].astype(np.float32)
        )
    out["recent_den_prev1_over_prev3_lower"] = np.divide(
        lower_denominators[0],
        lower_denominators[1],
        out=np.zeros(len(out), dtype=np.float64),
        where=lower_denominators[1] > 0,
    ).astype(np.float32)
    out["recent_den_prev3_over_prev5_lower"] = np.divide(
        lower_denominators[1],
        lower_denominators[2],
        out=np.zeros(len(out), dtype=np.float64),
        where=lower_denominators[2] > 0,
    ).astype(np.float32)
    return out, diagnostics


def recent_denominator_feature_names() -> list[str]:
    """Return the deterministic feature schema without touching any rows."""
    empty = pd.DataFrame(
        {
            "asof_pitcher_success_rate": pd.Series(dtype=float),
            "asof_pitcher_middle_rate": pd.Series(dtype=float),
            **{
                column: pd.Series(dtype=float)
                for window in WINDOWS
                for column in (window.success_column, window.middle_column)
            },
        }
    )
    attached, _ = attach_recent_denominator_features(empty)
    return sorted(set(attached.columns) - set(empty.columns))
