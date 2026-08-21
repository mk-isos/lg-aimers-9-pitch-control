"""EXP-114: target-free sticky switching pitcher-regime residual.

The hidden Markov model is fitted only to source official covariates.  One
observation is retained for each reconstructed pitcher-game and consists of
the six legal previous-1/3/5-game success/middle rates plus missingness.  The
current-pitch target, current-pitch result, and validation/test peers never
enter the HMM, its scaler, a query prior, or a query posterior.

For a row in season ``s``, the prior is the pitcher's source-frozen terminal
posterior from the latest season strictly before ``s``, propagated through
the learned transition once per season gap.  The posterior then consumes only
that row's emission.  Source residual-map rows follow the identical contract;
they do not receive a state filtered through other rows in their own season.

Exactly two preregistered configurations are implemented: H1-sticky3 and
H2-sticky4.  This runner performs only the cheap 2023/2024 folds when invoked.
It is intentionally not imported or executed by repository tooling.
"""

from __future__ import annotations

import argparse
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from ultra_model_common import (
    CORRECTION_CLIP,
    DATA_PATH,
    DIAGNOSTIC_SEASONS,
    MODEL_SEED,
    bounded_candidate,
    diagnostic_metrics,
    exp051_fold,
    exp071_fold,
    json_dump,
    peak_rss_mb,
    pooled_metrics,
    promotion_gate,
    reconstructed_game_ids,
    row_independence_audit,
    season_equal_weights,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "EXP-114" / "sticky_switching_regime"
REPORT_PATH = ARTIFACT_ROOT / "validation_metrics.json"
PROTOCOL_PATH = ROOT / "docs" / "MODEL_DISCOVERY_EXP112_ULTRA.md"
PROTOCOL_SHA256 = "1bc9dd6384a721d93205521f6058ff4c0d368d1a2efbb5fa44a32942723184d0"

ALL_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
RESIDUAL_SOURCE_FIRST_SEASON = 2022
CONFIGURATIONS = {
    "H1-sticky3": 3,
    "H2-sticky4": 4,
}
PRIMARY_CONFIGURATION = "H1-sticky3"

RATE_COLUMNS = (
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
)
MISSING_FEATURE_NAMES = tuple(f"missing__{name}" for name in RATE_COLUMNS)
EMISSION_FEATURE_NAMES = (*RATE_COLUMNS, *MISSING_FEATURE_NAMES)
GAME_COLUMNS = (
    "season",
    "inning",
    "top_bottom",
    "run_top_before",
    "run_bot_before",
    "pitcher_team_id",
    "batter_team_id",
)
OFFICIAL_COLUMNS = tuple(
    dict.fromkeys(
        (
            *GAME_COLUMNS,
            "pitcher_id",
            "balls_before",
            "strikes_before",
            "batter_hand",
            *RATE_COLUMNS,
            "control_success",
        )
    )
)

RATE_EPSILON = 1e-4
VARIANCE_FLOOR = 0.05**2
TRANSITION_DIAGONAL_PSEUDOCOUNT = 20.0
TRANSITION_OFF_DIAGONAL_PSEUDOCOUNT = 1.0
INITIALIZATION_COUNT = 5
MAX_EM_ITERATIONS = 75
MIN_EM_ITERATIONS = 5
EM_RELATIVE_TOLERANCE = 1e-6
KMEANS_MAX_ITERATIONS = 100
MAP_SMOOTHING = 300.0
COUNT_CARDINALITY = 12
BATTER_HAND_CARDINALITY = 2
STATE_OCCUPANCY_MINIMUM = 0.05
POSTERIOR_CONFIDENCE_MINIMUM = 0.60
TRANSITION_CORRECTION_RMS_MINIMUM = 1e-4


@dataclass(frozen=True)
class EmissionTransform:
    """Source-only imputation and standardization for the 12 HMM features."""

    rate_logit_medians: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    source_rows: int
    source_seasons: tuple[int, ...]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        logit_values, missing = rate_logit_matrix(frame)
        filled = np.where(
            np.isfinite(logit_values), logit_values, self.rate_logit_medians
        )
        features = np.column_stack([filled, missing]).astype(np.float64)
        transformed = (features - self.centers) / self.scales
        if not np.isfinite(transformed).all():
            raise ValueError("non-finite standardized HMM emission")
        return transformed

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(EMISSION_FEATURE_NAMES),
            "rate_logit_medians": self.rate_logit_medians.tolist(),
            "centers": self.centers.tolist(),
            "scales": self.scales.tolist(),
            "source_rows": self.source_rows,
            "source_seasons": list(self.source_seasons),
            "rate_epsilon": RATE_EPSILON,
        }


@dataclass(frozen=True)
class HMMParameters:
    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    log_likelihood: float
    iterations: int
    initialization_seed: int
    converged: bool

    @property
    def state_count(self) -> int:
        return int(len(self.initial))


@dataclass(frozen=True)
class HMMFitResult:
    parameters: HMMParameters
    source_occupancy: np.ndarray
    initialization_records: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class GameObservations:
    metadata: pd.DataFrame
    matrix: np.ndarray
    sequences: tuple[np.ndarray, ...]
    sequence_pitchers: tuple[int, ...]


@dataclass(frozen=True)
class FrozenRegimePredictor:
    """All state needed for deterministic independent-row prediction."""

    transform: EmissionTransform
    parameters: HMMParameters
    terminal_posteriors: dict[tuple[int, int], np.ndarray]
    transition_effects: np.ndarray
    emission_only_effects: np.ndarray
    prediction_season: int

    def posteriors(
        self, rows: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = self.transform.transform(rows)
        transition_prior, prior_diagnostics = query_priors(
            rows,
            self.prediction_season,
            self.parameters,
            self.terminal_posteriors,
        )
        log_emission = diagonal_gaussian_log_emission(
            matrix, self.parameters.means, self.parameters.variances
        )
        transition_posterior = posterior_from_prior_and_log_emission(
            transition_prior, log_emission
        )
        stationary = stationary_distribution(self.parameters.transition)
        emission_prior = np.broadcast_to(
            stationary, (len(rows), len(stationary))
        )
        emission_posterior = posterior_from_prior_and_log_emission(
            emission_prior, log_emission
        )
        return transition_posterior, emission_posterior, prior_diagnostics

    def raw_scores(
        self, rows: pd.DataFrame, *, emission_only: bool = False
    ) -> np.ndarray:
        transition, emission, _ = self.posteriors(rows)
        if emission_only:
            effects = self.emission_only_effects
            posterior = emission
        else:
            effects = self.transition_effects
            posterior = transition
        residual_effect = posterior_weighted_effect(rows, posterior, effects)
        return bounded_effect_to_raw(residual_effect)

    def predict(self, rows: pd.DataFrame, baseline: np.ndarray) -> np.ndarray:
        return bounded_candidate(baseline, self.raw_scores(rows))

    def predict_emission_only(
        self, rows: pd.DataFrame, baseline: np.ndarray
    ) -> np.ndarray:
        return bounded_candidate(
            baseline, self.raw_scores(rows, emission_only=True)
        )


def rate_logit_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing_columns = sorted(set(RATE_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"missing HMM rate columns: {missing_columns}")
    rates = np.column_stack(
        [
            pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            for column in RATE_COLUMNS
        ]
    )
    missing = ~np.isfinite(rates)
    finite = np.clip(rates, RATE_EPSILON, 1.0 - RATE_EPSILON)
    logits = np.log(finite / (1.0 - finite))
    logits[missing] = np.nan
    return logits, missing.astype(np.float64)


def fit_emission_transform(
    observations: pd.DataFrame, source_seasons: tuple[int, ...]
) -> EmissionTransform:
    logits, missing = rate_logit_matrix(observations)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(logits, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(logits), logits, medians)
    features = np.column_stack([filled, missing]).astype(np.float64)
    centers = features.mean(axis=0)
    scales = features.std(axis=0, ddof=0)
    scales = np.where(scales > 1e-6, scales, 1.0)
    return EmissionTransform(
        rate_logit_medians=medians.astype(np.float64),
        centers=centers.astype(np.float64),
        scales=scales.astype(np.float64),
        source_rows=int(len(observations)),
        source_seasons=source_seasons,
    )


def load_official_frame() -> pd.DataFrame:
    frame = pd.read_csv(
        DATA_PATH, encoding="utf-8-sig", usecols=list(OFFICIAL_COLUMNS)
    )
    frame = frame.loc[frame["season"].isin(ALL_SEASONS)].reset_index(drop=True)
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("official rows must remain season-monotone")
    if frame[["season", "pitcher_id", "balls_before", "strikes_before"]].isna().any().any():
        raise ValueError("missing required official HMM field")
    observed_target = frame["control_success"].to_numpy(dtype=np.float64)
    if not np.isin(observed_target, [0.0, 1.0]).all():
        raise ValueError("invalid official binary target")
    frame["official_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame["official_game_id"] = reconstructed_game_ids(frame)
    return frame


def build_game_observations(
    frame: pd.DataFrame, source_seasons: tuple[int, ...]
) -> tuple[GameObservations, EmissionTransform, dict[str, object]]:
    source = frame.loc[frame["season"].isin(source_seasons)].copy()
    if source.empty:
        raise ValueError("empty HMM source")
    observation_columns = (
        "season",
        "official_game_id",
        "pitcher_id",
        "official_row_index",
        *RATE_COLUMNS,
    )
    observations = (
        source.sort_values("official_row_index")
        .drop_duplicates(
            ["season", "official_game_id", "pitcher_id"], keep="first"
        )
        .loc[:, list(observation_columns)]
        .reset_index(drop=True)
    )
    duplicate = observations.duplicated(
        ["season", "official_game_id", "pitcher_id"]
    )
    if duplicate.any():
        raise AssertionError("duplicate reconstructed pitcher-game observation")

    transform = fit_emission_transform(observations, source_seasons)
    matrix = transform.transform(observations)
    sequences: list[np.ndarray] = []
    sequence_pitchers: list[int] = []
    grouped = observations.groupby("pitcher_id", sort=True, observed=True).indices
    for pitcher_id, positions in grouped.items():
        ordered = np.asarray(positions, dtype=np.int64)
        ordered = ordered[np.argsort(
            observations.iloc[ordered]["official_row_index"].to_numpy()
        )]
        sequences.append(ordered)
        sequence_pitchers.append(int(pitcher_id))
    if sum(len(sequence) for sequence in sequences) != len(observations):
        raise AssertionError("HMM sequences do not partition observations")

    audit = {
        "source_seasons": list(source_seasons),
        "official_source_rows": int(len(source)),
        "reconstructed_games": int(source["official_game_id"].nunique()),
        "pitcher_game_observations": int(len(observations)),
        "pitcher_sequences": int(len(sequences)),
        "minimum_sequence_length": int(min(map(len, sequences))),
        "maximum_sequence_length": int(max(map(len, sequences))),
        "median_sequence_length": float(np.median([len(value) for value in sequences])),
        "target_free_hmm_fit": True,
        "hmm_fit_columns": list(EMISSION_FEATURE_NAMES),
        "control_success_used_by_hmm": False,
        "observation_unit": "first row of target-free reconstructed pitcher-game",
    }
    return (
        GameObservations(
            metadata=observations,
            matrix=matrix,
            sequences=tuple(sequences),
            sequence_pitchers=tuple(sequence_pitchers),
        ),
        transform,
        audit,
    )


def diagonal_gaussian_log_emission(
    matrix: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if values.ndim != 2 or means.ndim != 2 or variances.shape != means.shape:
        raise ValueError("invalid diagonal Gaussian shapes")
    if values.shape[1] != means.shape[1]:
        raise ValueError("emission feature dimension mismatch")
    if np.any(variances < VARIANCE_FLOOR) or not np.isfinite(variances).all():
        raise ValueError("invalid HMM variances")
    difference = values[:, None, :] - means[None, :, :]
    return -0.5 * np.sum(
        math.log(2.0 * math.pi)
        + np.log(variances)[None, :, :]
        + np.square(difference) / variances[None, :, :],
        axis=2,
    )


def sticky_pseudocounts(state_count: int) -> np.ndarray:
    output = np.full(
        (state_count, state_count),
        TRANSITION_OFF_DIAGONAL_PSEUDOCOUNT,
        dtype=np.float64,
    )
    np.fill_diagonal(output, TRANSITION_DIAGONAL_PSEUDOCOUNT)
    return output


def initialize_parameters(
    observations: GameObservations,
    state_count: int,
    seed: int,
) -> HMMParameters:
    cluster = KMeans(
        n_clusters=state_count,
        init="k-means++",
        n_init=1,
        max_iter=KMEANS_MAX_ITERATIONS,
        random_state=seed,
        algorithm="lloyd",
    ).fit(observations.matrix)
    labels = cluster.labels_.astype(np.int16)
    means = cluster.cluster_centers_.astype(np.float64)
    global_variance = np.maximum(
        observations.matrix.var(axis=0, ddof=0), VARIANCE_FLOOR
    )
    variances = np.empty_like(means)
    for state in range(state_count):
        members = observations.matrix[labels == state]
        variances[state] = (
            np.maximum(members.var(axis=0, ddof=0), VARIANCE_FLOOR)
            if len(members) >= 2
            else global_variance
        )

    initial_counts = np.ones(state_count, dtype=np.float64)
    transition_counts = sticky_pseudocounts(state_count)
    for sequence in observations.sequences:
        local = labels[sequence]
        initial_counts[local[0]] += 1.0
        if len(local) > 1:
            np.add.at(transition_counts, (local[:-1], local[1:]), 1.0)
    initial = initial_counts / initial_counts.sum()
    transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    return HMMParameters(
        initial=initial,
        transition=transition,
        means=means,
        variances=variances,
        log_likelihood=-np.inf,
        iterations=0,
        initialization_seed=seed,
        converged=False,
    )


def expectation_statistics(
    observations: GameObservations, parameters: HMMParameters
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_count = parameters.state_count
    log_initial = np.log(np.clip(parameters.initial, 1e-300, None))
    log_transition = np.log(np.clip(parameters.transition, 1e-300, None))
    log_emission = diagonal_gaussian_log_emission(
        observations.matrix, parameters.means, parameters.variances
    )
    initial_sum = np.zeros(state_count, dtype=np.float64)
    transition_sum = np.zeros((state_count, state_count), dtype=np.float64)
    occupancy = np.zeros(state_count, dtype=np.float64)
    weighted_values = np.zeros_like(parameters.means)
    weighted_squares = np.zeros_like(parameters.means)
    total_log_likelihood = 0.0

    for sequence in observations.sequences:
        local_emission = log_emission[sequence]
        length = len(sequence)
        alpha = np.empty((length, state_count), dtype=np.float64)
        alpha[0] = log_initial + local_emission[0]
        for index in range(1, length):
            alpha[index] = local_emission[index] + logsumexp(
                alpha[index - 1][:, None] + log_transition, axis=0
            )
        sequence_log_likelihood = float(logsumexp(alpha[-1]))
        total_log_likelihood += sequence_log_likelihood

        beta = np.zeros((length, state_count), dtype=np.float64)
        for index in range(length - 2, -1, -1):
            beta[index] = logsumexp(
                log_transition
                + local_emission[index + 1][None, :]
                + beta[index + 1][None, :],
                axis=1,
            )
        log_gamma = alpha + beta - sequence_log_likelihood
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=1, keepdims=True)
        initial_sum += gamma[0]
        occupancy += gamma.sum(axis=0)
        values = observations.matrix[sequence]
        weighted_values += gamma.T @ values
        weighted_squares += gamma.T @ np.square(values)
        for index in range(length - 1):
            log_xi = (
                alpha[index][:, None]
                + log_transition
                + local_emission[index + 1][None, :]
                + beta[index + 1][None, :]
                - sequence_log_likelihood
            )
            xi = np.exp(log_xi)
            xi /= xi.sum()
            transition_sum += xi

    return (
        total_log_likelihood,
        initial_sum,
        transition_sum,
        occupancy,
        weighted_values,
        weighted_squares,
    )


def fit_one_initialization(
    observations: GameObservations,
    state_count: int,
    seed: int,
) -> tuple[HMMParameters, np.ndarray, list[float]]:
    parameters = initialize_parameters(observations, state_count, seed)
    history: list[float] = []
    converged = False
    occupancy = np.full(state_count, 1.0 / state_count, dtype=np.float64)
    for iteration in range(1, MAX_EM_ITERATIONS + 1):
        (
            log_likelihood,
            initial_sum,
            transition_sum,
            gamma_sum,
            weighted_values,
            weighted_squares,
        ) = expectation_statistics(observations, parameters)
        history.append(log_likelihood)
        initial = (initial_sum + 1.0) / (initial_sum.sum() + state_count)
        transition = transition_sum + sticky_pseudocounts(state_count)
        transition /= transition.sum(axis=1, keepdims=True)
        safe_gamma = np.maximum(gamma_sum, 1e-12)
        means = weighted_values / safe_gamma[:, None]
        second_moment = weighted_squares / safe_gamma[:, None]
        variances = np.maximum(
            second_moment - np.square(means), VARIANCE_FLOOR
        )
        parameters = HMMParameters(
            initial=initial,
            transition=transition,
            means=means,
            variances=variances,
            log_likelihood=log_likelihood,
            iterations=iteration,
            initialization_seed=seed,
            converged=False,
        )
        occupancy = gamma_sum / gamma_sum.sum()
        if iteration >= MIN_EM_ITERATIONS and len(history) >= 2:
            improvement = history[-1] - history[-2]
            relative = abs(improvement) / max(1.0, abs(history[-2]))
            if improvement >= -1e-8 and relative <= EM_RELATIVE_TOLERANCE:
                converged = True
                break

    final_log_likelihood, *final_statistics = expectation_statistics(
        observations, parameters
    )
    final_occupancy = final_statistics[2]
    final_occupancy = final_occupancy / final_occupancy.sum()
    parameters = HMMParameters(
        initial=parameters.initial,
        transition=parameters.transition,
        means=parameters.means,
        variances=parameters.variances,
        log_likelihood=float(final_log_likelihood),
        iterations=parameters.iterations,
        initialization_seed=seed,
        converged=converged,
    )
    return parameters, final_occupancy, history


def reorder_states_by_success(
    parameters: HMMParameters,
    occupancy: np.ndarray,
    transform: EmissionTransform,
) -> tuple[HMMParameters, np.ndarray, np.ndarray]:
    unscaled_means = parameters.means * transform.scales + transform.centers
    mean_success_logit = unscaled_means[:, :3].mean(axis=1)
    order = np.argsort(mean_success_logit, kind="stable")
    reordered = HMMParameters(
        initial=parameters.initial[order],
        transition=parameters.transition[np.ix_(order, order)],
        means=parameters.means[order],
        variances=parameters.variances[order],
        log_likelihood=parameters.log_likelihood,
        iterations=parameters.iterations,
        initialization_seed=parameters.initialization_seed,
        converged=parameters.converged,
    )
    return reordered, occupancy[order], mean_success_logit[order]


def fit_sticky_hmm(
    observations: GameObservations,
    transform: EmissionTransform,
    state_count: int,
) -> HMMFitResult:
    records: list[dict[str, object]] = []
    fitted: list[tuple[HMMParameters, np.ndarray]] = []
    for initialization in range(INITIALIZATION_COUNT):
        seed = MODEL_SEED + state_count * 10_007 + initialization * 1_009
        started = time.perf_counter()
        parameters, occupancy, history = fit_one_initialization(
            observations, state_count, seed
        )
        records.append(
            {
                "initialization": initialization,
                "seed": seed,
                "source_log_likelihood": parameters.log_likelihood,
                "iterations": parameters.iterations,
                "converged": parameters.converged,
                "occupancy": occupancy.tolist(),
                "likelihood_history": history,
                "seconds": time.perf_counter() - started,
            }
        )
        fitted.append((parameters, occupancy))
    best_index = max(
        range(len(fitted)), key=lambda index: fitted[index][0].log_likelihood
    )
    parameters, occupancy = fitted[best_index]
    parameters, occupancy, ordered_success = reorder_states_by_success(
        parameters, occupancy, transform
    )
    records[best_index]["selected_by_source_likelihood"] = True
    records[best_index]["ordered_mean_success_logit"] = ordered_success.tolist()
    for index, record in enumerate(records):
        if index != best_index:
            record["selected_by_source_likelihood"] = False
    return HMMFitResult(
        parameters=parameters,
        source_occupancy=occupancy,
        initialization_records=tuple(records),
    )


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transition, dtype=np.float64)
    probability = np.full(len(matrix), 1.0 / len(matrix), dtype=np.float64)
    for _ in range(10_000):
        updated = probability @ matrix
        if np.max(np.abs(updated - probability)) <= 1e-14:
            probability = updated
            break
        probability = updated
    probability = np.clip(probability, 0.0, None)
    probability /= probability.sum()
    return probability


def posterior_from_prior_and_log_emission(
    prior: np.ndarray, log_emission: np.ndarray
) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float64)
    log_emission = np.asarray(log_emission, dtype=np.float64)
    if prior.shape != log_emission.shape:
        raise ValueError("prior/emission posterior shape mismatch")
    log_joint = np.log(np.clip(prior, 1e-300, None)) + log_emission
    posterior = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    return posterior


def terminal_posteriors_by_pitcher_season(
    observations: GameObservations, parameters: HMMParameters
) -> dict[tuple[int, int], np.ndarray]:
    log_emission = diagonal_gaussian_log_emission(
        observations.matrix, parameters.means, parameters.variances
    )
    terminals: dict[tuple[int, int], np.ndarray] = {}
    for pitcher_id, sequence in zip(
        observations.sequence_pitchers, observations.sequences, strict=True
    ):
        posterior: np.ndarray | None = None
        previous_season: int | None = None
        for position in sequence:
            season = int(observations.metadata.iloc[int(position)]["season"])
            if posterior is None:
                prior = parameters.initial
            else:
                gap = 1 if previous_season == season else max(
                    1, season - int(previous_season)
                )
                prior = posterior @ np.linalg.matrix_power(
                    parameters.transition, gap
                )
            posterior = posterior_from_prior_and_log_emission(
                prior[None, :], log_emission[[int(position)]]
            )[0]
            terminals[(pitcher_id, season)] = posterior.copy()
            previous_season = season
    return terminals


def query_priors(
    rows: pd.DataFrame,
    prediction_season: int,
    parameters: HMMParameters,
    terminal_posteriors: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not rows["season"].eq(prediction_season).all():
        raise ValueError("query rows must belong to one declared season")
    stationary = stationary_distribution(parameters.transition)
    latest: dict[int, tuple[int, np.ndarray]] = {}
    for (pitcher_id, season), posterior in terminal_posteriors.items():
        if season >= prediction_season:
            continue
        current = latest.get(pitcher_id)
        if current is None or season > current[0]:
            latest[pitcher_id] = (season, posterior)
    pitcher_ids = rows["pitcher_id"].to_numpy(dtype=np.int64)
    priors = np.empty((len(rows), parameters.state_count), dtype=np.float64)
    seen = np.zeros(len(rows), dtype=np.float64)
    gaps = np.zeros(len(rows), dtype=np.float64)
    cache: dict[int, np.ndarray] = {}
    for index, pitcher_id_value in enumerate(pitcher_ids):
        pitcher_id = int(pitcher_id_value)
        if pitcher_id not in latest:
            priors[index] = stationary
            continue
        source_season, posterior = latest[pitcher_id]
        gap = max(1, prediction_season - source_season)
        if pitcher_id not in cache:
            cache[pitcher_id] = posterior @ np.linalg.matrix_power(
                parameters.transition, gap
            )
        priors[index] = cache[pitcher_id]
        seen[index] = 1.0
        gaps[index] = float(gap)
    priors /= priors.sum(axis=1, keepdims=True)
    diagnostics = np.column_stack([seen, gaps])
    return priors, diagnostics


def count_and_hand_indices(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    balls = rows["balls_before"].to_numpy(dtype=np.int16)
    strikes = rows["strikes_before"].to_numpy(dtype=np.int16)
    count = 3 * balls + strikes
    if np.any(count < 0) or np.any(count >= COUNT_CARDINALITY):
        raise ValueError("invalid count index")
    batter_hand = rows["batter_hand"].to_numpy(dtype=np.int16) - 1
    if np.any(batter_hand < 0) or np.any(
        batter_hand >= BATTER_HAND_CARDINALITY
    ):
        raise ValueError("invalid batter hand")
    return count.astype(np.int16), batter_hand.astype(np.int8)


def fit_residual_effects(
    posterior: np.ndarray,
    rows: pd.DataFrame,
    residual: np.ndarray,
    weights: np.ndarray,
    state_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    count, batter_hand = count_and_hand_indices(rows)
    posterior = np.asarray(posterior, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if posterior.shape != (len(rows), state_count):
        raise ValueError("invalid residual-map posterior")
    if residual.shape != (len(rows),) or weights.shape != (len(rows),):
        raise ValueError("invalid residual-map vectors")
    numerator = np.zeros(
        (state_count, COUNT_CARDINALITY, BATTER_HAND_CARDINALITY),
        dtype=np.float64,
    )
    mass = np.zeros_like(numerator)
    for state in range(state_count):
        local_weight = weights * posterior[:, state]
        np.add.at(mass[state], (count, batter_hand), local_weight)
        np.add.at(
            numerator[state],
            (count, batter_hand),
            local_weight * residual,
        )
    raw_effect = numerator / (mass + MAP_SMOOTHING)
    effects = np.clip(raw_effect, -CORRECTION_CLIP, CORRECTION_CLIP)
    return effects, {
        "smoothing": MAP_SMOOTHING,
        "positive_cells": int(np.sum(mass > 0.0)),
        "total_cells": int(mass.size),
        "effective_mass_mean_positive": float(mass[mass > 0.0].mean()),
        "effective_mass_max": float(mass.max()),
        "effect_rms": float(np.sqrt(np.mean(np.square(effects)))),
        "effect_max_abs": float(np.max(np.abs(effects))),
        "effect_clip_count": int(np.sum(np.abs(raw_effect) > CORRECTION_CLIP)),
    }


def posterior_weighted_effect(
    rows: pd.DataFrame, posterior: np.ndarray, effects: np.ndarray
) -> np.ndarray:
    count, batter_hand = count_and_hand_indices(rows)
    selected = effects[:, count, batter_hand].T
    if selected.shape != posterior.shape:
        raise ValueError("posterior/effect prediction shape mismatch")
    return np.sum(posterior * selected, axis=1)


def bounded_effect_to_raw(effect: np.ndarray) -> np.ndarray:
    bounded = np.clip(
        np.asarray(effect, dtype=np.float64),
        -CORRECTION_CLIP * (1.0 - 1e-12),
        CORRECTION_CLIP * (1.0 - 1e-12),
    )
    return np.arctanh(bounded / CORRECTION_CLIP)


def build_residual_maps(
    frame: pd.DataFrame,
    validation_season: int,
    transform: EmissionTransform,
    parameters: HMMParameters,
    terminal_posteriors: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    source_seasons = tuple(
        season
        for season in range(RESIDUAL_SOURCE_FIRST_SEASON, validation_season)
        if season in ALL_SEASONS
    )
    if not source_seasons:
        raise ValueError("no strict EXP-071 residual source season")
    row_parts: list[pd.DataFrame] = []
    transition_parts: list[np.ndarray] = []
    emission_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    season_parts: list[np.ndarray] = []
    prior_diagnostics: dict[str, object] = {}
    stationary = stationary_distribution(parameters.transition)

    for season in source_seasons:
        rows = frame.loc[frame["season"].eq(season)].reset_index(drop=True)
        target, baseline = exp071_fold(season)
        if len(rows) != len(target):
            raise ValueError(f"source row/EXP-071 mismatch for {season}")
        if not np.array_equal(
            rows["control_success"].to_numpy(dtype=np.float64), target
        ):
            raise ValueError(f"source target order mismatch for {season}")
        matrix = transform.transform(rows)
        prior, prior_detail = query_priors(
            rows, season, parameters, terminal_posteriors
        )
        log_emission = diagonal_gaussian_log_emission(
            matrix, parameters.means, parameters.variances
        )
        transition_posterior = posterior_from_prior_and_log_emission(
            prior, log_emission
        )
        emission_posterior = posterior_from_prior_and_log_emission(
            np.broadcast_to(stationary, prior.shape), log_emission
        )
        row_parts.append(rows)
        transition_parts.append(transition_posterior)
        emission_parts.append(emission_posterior)
        residual_parts.append(target - baseline)
        season_parts.append(np.full(len(rows), season, dtype=np.int16))
        prior_diagnostics[str(season)] = {
            "rows": int(len(rows)),
            "source_terminal_prior_rate": float(prior_detail[:, 0].mean()),
            "season_gap_mean_seen": float(
                prior_detail[prior_detail[:, 0] > 0.0, 1].mean()
            )
            if np.any(prior_detail[:, 0] > 0.0)
            else None,
        }

    combined_rows = pd.concat(row_parts, ignore_index=True)
    transition_posterior = np.concatenate(transition_parts, axis=0)
    emission_posterior = np.concatenate(emission_parts, axis=0)
    residual = np.concatenate(residual_parts)
    seasons = np.concatenate(season_parts)
    weights = season_equal_weights(seasons)
    transition_effects, transition_audit = fit_residual_effects(
        transition_posterior,
        combined_rows,
        residual,
        weights,
        parameters.state_count,
    )
    emission_effects, emission_audit = fit_residual_effects(
        emission_posterior,
        combined_rows,
        residual,
        weights,
        parameters.state_count,
    )
    audit = {
        "source_seasons": list(source_seasons),
        "rows": int(len(combined_rows)),
        "source_season_equal_weight_totals": {
            str(season): float(weights[seasons == season].sum())
            for season in source_seasons
        },
        "raw_residual_mean_by_season": {
            str(season): float(residual[seasons == season].mean())
            for season in source_seasons
        },
        "query_contract_used_for_source_rows": (
            "prior-season terminal posterior plus this row emission only"
        ),
        "source_prior_diagnostics": prior_diagnostics,
        "transition_map": transition_audit,
        "emission_only_map": emission_audit,
    }
    return transition_effects, emission_effects, audit


def parameter_state_size_bytes(
    parameters: HMMParameters,
    terminal_posteriors: dict[tuple[int, int], np.ndarray],
    transition_effects: np.ndarray,
    emission_effects: np.ndarray,
    transform: EmissionTransform,
) -> int:
    arrays = (
        parameters.initial,
        parameters.transition,
        parameters.means,
        parameters.variances,
        transform.rate_logit_medians,
        transform.centers,
        transform.scales,
        transition_effects,
        emission_effects,
    )
    return int(
        sum(value.nbytes for value in arrays)
        + sum(value.nbytes for value in terminal_posteriors.values())
    )


def save_fold_artifacts(
    configuration: str,
    season: int,
    target: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    emission_only_prediction: np.ndarray,
    predictor: FrozenRegimePredictor,
    fold_report: dict[str, object],
) -> None:
    output = ARTIFACT_ROOT / configuration / str(season)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "target.npy", np.asarray(target, dtype=np.float64))
    np.save(output / "exp071.npy", np.asarray(baseline, dtype=np.float64))
    np.save(output / "prediction.npy", np.asarray(prediction, dtype=np.float64))
    np.save(
        output / "prediction_emission_only.npy",
        np.asarray(emission_only_prediction, dtype=np.float64),
    )
    terminal_keys = sorted(predictor.terminal_posteriors)
    terminal_pitcher = np.asarray([key[0] for key in terminal_keys], dtype=np.int64)
    terminal_season = np.asarray([key[1] for key in terminal_keys], dtype=np.int16)
    terminal_values = np.vstack(
        [predictor.terminal_posteriors[key] for key in terminal_keys]
    )
    np.savez_compressed(
        output / "model_state.npz",
        initial=predictor.parameters.initial,
        transition=predictor.parameters.transition,
        means=predictor.parameters.means,
        variances=predictor.parameters.variances,
        rate_logit_medians=predictor.transform.rate_logit_medians,
        emission_centers=predictor.transform.centers,
        emission_scales=predictor.transform.scales,
        transition_effects=predictor.transition_effects,
        emission_only_effects=predictor.emission_only_effects,
        terminal_pitcher=terminal_pitcher,
        terminal_season=terminal_season,
        terminal_posterior=terminal_values,
    )
    json_dump(output / "metrics.json", fold_report)


def secondary_exp051_metrics(
    target: np.ndarray, candidate: np.ndarray, season: int
) -> dict[str, float]:
    control = exp051_fold(season)
    if control.shape != target.shape:
        raise ValueError("EXP-051 secondary control shape mismatch")
    candidate_loss = np.square(target - candidate)
    control_loss = np.square(target - control)
    return {
        "exp051_brier": float(control_loss.mean()),
        "candidate_delta_brier_vs_exp051": float(
            (candidate_loss - control_loss).mean()
        ),
        "prediction_correlation_with_exp051": float(
            np.corrcoef(candidate, control)[0, 1]
        ),
    }


def run_configuration_fold(
    frame: pd.DataFrame,
    observations: GameObservations,
    transform: EmissionTransform,
    configuration: str,
    state_count: int,
    validation_season: int,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    started = time.perf_counter()
    train_started = time.perf_counter()
    fitted = fit_sticky_hmm(observations, transform, state_count)
    terminal_posteriors = terminal_posteriors_by_pitcher_season(
        observations, fitted.parameters
    )
    transition_effects, emission_effects, map_audit = build_residual_maps(
        frame,
        validation_season,
        transform,
        fitted.parameters,
        terminal_posteriors,
    )
    train_seconds = time.perf_counter() - train_started
    predictor = FrozenRegimePredictor(
        transform=transform,
        parameters=fitted.parameters,
        terminal_posteriors=terminal_posteriors,
        transition_effects=transition_effects,
        emission_only_effects=emission_effects,
        prediction_season=validation_season,
    )

    rows = frame.loc[frame["season"].eq(validation_season)].reset_index(drop=True)
    target, baseline = exp071_fold(validation_season)
    if len(rows) != len(target):
        raise ValueError("validation row/EXP-071 mismatch")
    if not np.array_equal(
        rows["control_success"].to_numpy(dtype=np.float64), target
    ):
        raise ValueError("validation target order mismatch")
    inference_started = time.perf_counter()
    prediction = predictor.predict(rows, baseline)
    inference_seconds = time.perf_counter() - inference_started
    emission_only_prediction = predictor.predict_emission_only(rows, baseline)
    transition_posterior, emission_posterior, prior_detail = predictor.posteriors(rows)
    transition_correction = prediction - baseline
    emission_correction = emission_only_prediction - baseline
    transition_vs_emission_rms = float(
        np.sqrt(np.mean(np.square(transition_correction - emission_correction)))
    )
    posterior_rms = float(
        np.sqrt(np.mean(np.square(transition_posterior - emission_posterior)))
    )
    posterior_confidence = np.max(transition_posterior, axis=1)
    game_ids = rows["official_game_id"].to_numpy(dtype=np.int64)
    metrics = diagnostic_metrics(
        target,
        prediction,
        baseline,
        game_ids,
        season=validation_season,
    )
    independence = row_independence_audit(
        predictor.predict, rows, baseline, sample_rows=64, seed=MODEL_SEED
    )
    novelty = {
        "source_state_occupancy": fitted.source_occupancy.tolist(),
        "minimum_source_state_occupancy": float(fitted.source_occupancy.min()),
        "occupancy_gate_0_05_passed": bool(
            fitted.source_occupancy.min() >= STATE_OCCUPANCY_MINIMUM
        ),
        "query_median_maximum_posterior": float(np.median(posterior_confidence)),
        "posterior_confidence_gate_0_60_passed": bool(
            np.median(posterior_confidence) >= POSTERIOR_CONFIDENCE_MINIMUM
        ),
        "transition_vs_emission_only_correction_rms": transition_vs_emission_rms,
        "transition_correction_rms_gate_1e_4_passed": bool(
            transition_vs_emission_rms >= TRANSITION_CORRECTION_RMS_MINIMUM
        ),
        "transition_vs_emission_only_posterior_rms": posterior_rms,
        "source_terminal_prior_query_rate": float(prior_detail[:, 0].mean()),
        "season_gap_mean_seen_query": float(
            prior_detail[prior_detail[:, 0] > 0.0, 1].mean()
        )
        if np.any(prior_detail[:, 0] > 0.0)
        else None,
        "query_peer_state_updates": False,
        "query_emission_fields": list(RATE_COLUMNS),
    }
    novelty["fold_novelty_gate_passed"] = bool(
        novelty["occupancy_gate_0_05_passed"]
        and novelty["posterior_confidence_gate_0_60_passed"]
        and novelty["transition_correction_rms_gate_1e_4_passed"]
    )
    state_bytes = parameter_state_size_bytes(
        fitted.parameters,
        terminal_posteriors,
        transition_effects,
        emission_effects,
        transform,
    )
    fold_report: dict[str, object] = {
        "configuration": configuration,
        "state_count": state_count,
        "validation_season": validation_season,
        "source_seasons": sorted(
            int(value)
            for value in observations.metadata["season"].unique().tolist()
        ),
        "metrics": metrics,
        "secondary_exp051_control": secondary_exp051_metrics(
            target, prediction, validation_season
        ),
        "novelty": novelty,
        "row_independence": independence,
        "residual_map": map_audit,
        "hmm": {
            "selected_source_log_likelihood": fitted.parameters.log_likelihood,
            "selected_initialization_seed": fitted.parameters.initialization_seed,
            "iterations": fitted.parameters.iterations,
            "converged": fitted.parameters.converged,
            "initial_probability": fitted.parameters.initial.tolist(),
            "transition": fitted.parameters.transition.tolist(),
            "stationary_distribution": stationary_distribution(
                fitted.parameters.transition
            ).tolist(),
            "variance_floor": VARIANCE_FLOOR,
            "transition_pseudocount_diagonal": (
                TRANSITION_DIAGONAL_PSEUDOCOUNT
            ),
            "transition_pseudocount_off_diagonal": (
                TRANSITION_OFF_DIAGONAL_PSEUDOCOUNT
            ),
            "initializations": list(fitted.initialization_records),
            "terminal_posterior_entries": len(terminal_posteriors),
        },
        "runtime": {
            "train_seconds": train_seconds,
            "inference_seconds": inference_seconds,
            "inference_rows_per_second": len(rows) / max(inference_seconds, 1e-12),
            "total_seconds": time.perf_counter() - started,
            "peak_rss_mb": peak_rss_mb(),
        },
        "state_size": {
            "bytes": state_bytes,
            "mib": state_bytes / (1024.0**2),
            "terminal_posterior_entries": len(terminal_posteriors),
            "residual_effect_cells_per_map": int(transition_effects.size),
        },
        "legality": {
            "hmm_target_free": True,
            "current_pitch_outcome_used_by_hmm": False,
            "validation_or_test_peer_used": False,
            "query_prior_source_frozen": True,
            "one_transition_per_season_gap": True,
            "query_posterior_row_emission_only": True,
            "source_residual_rows_use_prior_season_terminal_only": True,
        },
    }
    save_fold_artifacts(
        configuration,
        validation_season,
        target,
        baseline,
        prediction,
        emission_only_prediction,
        predictor,
        fold_report,
    )
    return fold_report, (target, prediction, baseline)


def main() -> None:
    actual_protocol_sha256 = sha256_file(PROTOCOL_PATH)
    if actual_protocol_sha256 != PROTOCOL_SHA256:
        raise RuntimeError(
            "preregistration changed after lock: "
            f"{actual_protocol_sha256} != {PROTOCOL_SHA256}"
        )
    overall_started = time.perf_counter()
    frame = load_official_frame()
    configuration_reports: dict[str, dict[str, object]] = {
        name: {"folds": {}} for name in CONFIGURATIONS
    }
    pooled_inputs: dict[
        str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = {name: {} for name in CONFIGURATIONS}
    source_audits: dict[str, object] = {}

    for validation_season in DIAGNOSTIC_SEASONS:
        source_seasons = tuple(
            season for season in ALL_SEASONS if season < validation_season
        )
        preparation_started = time.perf_counter()
        observations, transform, source_audit = build_game_observations(
            frame, source_seasons
        )
        source_audit["preparation_seconds"] = (
            time.perf_counter() - preparation_started
        )
        source_audit["emission_transform"] = transform.to_dict()
        source_audits[str(validation_season)] = source_audit
        for configuration, state_count in CONFIGURATIONS.items():
            fold_report, arrays = run_configuration_fold(
                frame,
                observations,
                transform,
                configuration,
                state_count,
                validation_season,
            )
            configuration_reports[configuration]["folds"][
                str(validation_season)
            ] = fold_report
            pooled_inputs[configuration][validation_season] = arrays

    selected_configurations: list[str] = []
    for configuration in CONFIGURATIONS:
        folds = configuration_reports[configuration]["folds"]
        metric_by_season = {
            season: dict(folds[str(season)]["metrics"])
            for season in DIAGNOSTIC_SEASONS
        }
        gate = promotion_gate(metric_by_season)
        novelty_passed = all(
            bool(folds[str(season)]["novelty"]["fold_novelty_gate_passed"])
            for season in DIAGNOSTIC_SEASONS
        )
        pooled = pooled_metrics(pooled_inputs[configuration])
        survivor = bool(gate["metric_survivor"] and novelty_passed)
        configuration_reports[configuration]["pooled"] = pooled
        configuration_reports[configuration]["promotion_gate"] = gate
        configuration_reports[configuration]["family_novelty_gate_passed"] = (
            novelty_passed
        )
        configuration_reports[configuration]["cheap_survivor"] = survivor
        if survivor:
            selected_configurations.append(configuration)

    promoted: str | None = None
    if PRIMARY_CONFIGURATION in selected_configurations:
        promoted = PRIMARY_CONFIGURATION
    elif selected_configurations:
        promoted = selected_configurations[0]

    report = {
        "experiment": "EXP-114",
        "stage": "cheap_2023_2024",
        "family": "sticky_switching_pitcher_regime",
        "preregistration": "docs/MODEL_DISCOVERY_EXP112_ULTRA.md",
        "preregistration_sha256": PROTOCOL_SHA256,
        "baseline": "EXP-071 playerphys_resid_w025",
        "configurations": configuration_reports,
        "source_preparation": source_audits,
        "selection": {
            "primary_configuration": PRIMARY_CONFIGURATION,
            "qualifying_configurations": selected_configurations,
            "promoted_configuration": promoted,
            "primary_wins_if_both_qualify": True,
            "full_rolling_authorized": promoted is not None,
        },
        "fixed_definition": {
            "diagnostic_seasons": list(DIAGNOSTIC_SEASONS),
            "configuration_state_counts": CONFIGURATIONS,
            "rate_columns": list(RATE_COLUMNS),
            "emission_features": list(EMISSION_FEATURE_NAMES),
            "variance_floor": VARIANCE_FLOOR,
            "sticky_transition_pseudocounts": {
                "diagonal": TRANSITION_DIAGONAL_PSEUDOCOUNT,
                "off_diagonal": TRANSITION_OFF_DIAGONAL_PSEUDOCOUNT,
            },
            "initializations": INITIALIZATION_COUNT,
            "selection": "maximum source HMM likelihood only",
            "state_order": "ascending mean of three success-rate logits",
            "residual_map": "posterior state x count_index x batter_hand",
            "residual_map_smoothing": MAP_SMOOTHING,
            "correction": "0.03*tanh(raw), integrated at weight 0.25",
            "model_seed": MODEL_SEED,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "data_path": str(DATA_PATH),
        "artifact_root": str(ARTIFACT_ROOT),
        "total_seconds": time.perf_counter() - overall_started,
        "peak_rss_mb": peak_rss_mb(),
    }
    json_dump(REPORT_PATH, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered EXP-114 cheap sticky-HMM screen."
    )
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    main()
