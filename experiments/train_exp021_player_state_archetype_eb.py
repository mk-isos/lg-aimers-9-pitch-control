"""EXP-021: source-season player-state archetype x context EB.

Identity low-rank corrections are zero for a pitcher absent from an earlier
source season.  This bounded diagnostic transfers residual context effects
through an archetype learned solely from the official current-row pitcher
profile: control rates, recent 1/3/5-game rates, pitch mix, and row-local
prior/current-season posterior reliability features.

For each source OOF season independently:

1. take one equal-weight final snapshot per pitcher with official career
   sample size >=20;
2. fit a source-only imputer, scaler, and deterministic KMeans;
3. center that source season's residual against the immutable temporal
   ``strict_rank_s300`` base;
4. estimate archetype x static (count, pitcher hand, batter hand) EB maps.

At validation, every current row is independently transformed by each earlier
source model.  Corrections from all earlier source seasons are averaged
equally, including zero.  A row with official pitcher n==0 always receives
zero.  Raw player/team IDs and season are never cluster inputs.  Exactly four
candidates are predeclared: K=8/16 crossed with hard/soft assignment.  Map
smoothing is 300, effects are clipped to +/-0.02, and soft temperature is the
source-only median nearest-centroid squared distance.

No validation/test-row aggregate is used for fitting or prediction, no
current-fold label is used for selection, and test.csv is never read.
"""

from __future__ import annotations

import itertools
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from train_exp017_rolling_residual import calculate_metrics


DATA_PATH = Path("./data/train.csv")
TARGET_ROOT = Path("./artifacts/EXP-019/team_eb_ensemble")
IDENTITY_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-021/player_state_archetype_eb")

ALL_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
CLUSTER_COUNTS = (8, 16)
ASSIGNMENT_MODES = ("hard", "soft")
SNAPSHOT_MIN_N = 20.0
CAREER_STABILIZATION = 200.0
MAP_SMOOTHING = 300.0
EFFECT_CLIP = 0.02
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 300
KMEANS_RANDOM_STATE = 42
STABILITY_SEEDS = (11, 29, 42, 71, 101)

COUNT_INDICES = tuple(
    balls * 4 + strikes
    for balls in range(4)
    for strikes in range(3)
)
PITCHER_HANDS = (1, 2)
BATTER_HANDS = (1, 2)
CONTEXTS = tuple(
    (count_index, pitcher_hand, batter_hand)
    for count_index in COUNT_INDICES
    for pitcher_hand in PITCHER_HANDS
    for batter_hand in BATTER_HANDS
)
CONTEXT_TO_POSITION = {
    context: position for position, context in enumerate(CONTEXTS)
}

CONTROL_METRICS = (
    ("success", "asof_pitcher_success_rate", 0.50, True),
    ("reverse", "asof_pitcher_reverse_rate", 0.20, False),
    ("middle", "asof_pitcher_middle_rate", 0.15, False),
    ("ball", "asof_pitcher_ball_rate", 0.35, False),
    ("strike", "asof_pitcher_strike_rate", 0.45, False),
)
PITCHMIX_METRICS = (
    ("fastball", "asof_pitcher_fastball_rate", 0.50),
    ("breaking", "asof_pitcher_breaking_rate", 0.35),
    ("offspeed", "asof_pitcher_offspeed_rate", 0.15),
)
CAREER_PROFILE_COLUMNS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
)
RECENT_PROFILE_COLUMNS = (
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
)
TEMPORAL_PROFILE_COLUMNS = (
    "temporal_control_prior_exists",
    "temporal_control_reliability30",
    "temporal_control_success_prior_shrunk200",
    "temporal_control_success_season_player30",
    "temporal_control_reverse_prior_shrunk200",
    "temporal_control_reverse_season_player30",
    "temporal_control_middle_prior_shrunk200",
    "temporal_control_middle_season_player30",
    "temporal_pitchmix_fastball_prior_shrunk200",
    "temporal_pitchmix_fastball_season_player30",
    "temporal_pitchmix_breaking_prior_shrunk200",
    "temporal_pitchmix_breaking_season_player30",
)
PROFILE_FEATURES = tuple(
    [f"stabilized_{column}" for column in CAREER_PROFILE_COLUMNS]
    + [f"filled_{column}" for column in RECENT_PROFILE_COLUMNS]
    + ["recent_missing_indicator"]
    + list(TEMPORAL_PROFILE_COLUMNS)
)

BASE_CANDIDATE = "strict_rank_s300_base"
FIXED_IDENTITY_REFERENCE = "fixed_identity_s300_r4_reference"
ARCHETYPE_CANDIDATES = tuple(
    f"archetype_k{cluster_count}_{mode}"
    for cluster_count in CLUSTER_COUNTS
    for mode in ASSIGNMENT_MODES
)
CANDIDATES = (
    BASE_CANDIDATE,
    FIXED_IDENTITY_REFERENCE,
    *ARCHETYPE_CANDIDATES,
)


def archetype_name(cluster_count: int, mode: str) -> str:
    return f"archetype_k{cluster_count}_{mode}"


def _empty_state(metric_names: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "prior_n",
            *[f"prior_{name}_count" for name in metric_names],
        ],
        dtype=np.float64,
    )


def _global_rates(
    state: pd.DataFrame,
    metrics: tuple[tuple[object, ...], ...],
) -> dict[str, float]:
    defaults = {str(metric[0]): float(metric[2]) for metric in metrics}
    if state.empty:
        return defaults
    total_n = float(state["prior_n"].sum())
    if total_n <= 0.0:
        return defaults
    return {
        str(metric[0]): float(
            state[f"prior_{metric[0]}_count"].sum() / total_n
        )
        for metric in metrics
    }


def _state_values(
    ids: pd.Series,
    state: pd.DataFrame,
    column: str,
) -> np.ndarray:
    if state.empty:
        return np.zeros(len(ids), dtype=np.float64)
    return ids.map(state[column]).fillna(0.0).to_numpy(dtype=np.float64)


def _attach_state_group(
    rows: pd.DataFrame,
    state: pd.DataFrame,
    n_column: str,
    metrics: tuple[tuple[object, ...], ...],
    prefix: str,
    global_rates: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = rows.copy()
    ids = out["pitcher_id"]
    prior_n = _state_values(ids, state, "prior_n")
    prior_exists = (
        ids.isin(state.index).to_numpy(dtype=np.int8)
        if not state.empty
        else np.zeros(len(out), dtype=np.int8)
    )
    career_n = out[n_column].fillna(0.0).to_numpy(dtype=np.float64)
    season_n_raw = career_n - prior_n
    if np.any(season_n_raw < -1e-6):
        raise ValueError(f"{prefix}: official n below prior state")
    season_n = np.maximum(season_n_raw, 0.0)
    out[f"{prefix}_prior_exists"] = prior_exists
    out[f"{prefix}_prior_n"] = prior_n.astype(np.float32)
    out[f"{prefix}_season_n"] = season_n.astype(np.float32)
    out[f"{prefix}_reliability30"] = (
        season_n / (season_n + 30.0)
    ).astype(np.float32)

    negative = 0
    above_n = 0
    for metric in metrics:
        name = str(metric[0])
        rate_column = str(metric[1])
        global_rate = float(global_rates[name])
        prior_count = _state_values(
            ids, state, f"prior_{name}_count"
        )
        rate = out[rate_column].fillna(global_rate).to_numpy(dtype=np.float64)
        career_count = np.rint(career_n * rate)
        season_count_raw = career_count - prior_count
        negative += int((season_count_raw < -0.01).sum())
        above_n += int((season_count_raw - season_n > 0.01).sum())
        season_count = np.clip(season_count_raw, 0.0, season_n)
        prior_shrunk = (prior_count + 200.0 * global_rate) / (
            prior_n + 200.0
        )
        season_player30 = (
            season_count + 30.0 * prior_shrunk
        ) / (season_n + 30.0)
        out[f"{prefix}_{name}_prior_shrunk200"] = (
            prior_shrunk.astype(np.float32)
        )
        out[f"{prefix}_{name}_season_player30"] = (
            season_player30.astype(np.float32)
        )
    return out, {
        "negative_metric_count_before_clip": negative,
        "above_n_metric_count_before_clip": above_n,
    }


def _updated_state(
    season_rows: pd.DataFrame,
    n_column: str,
    metrics: tuple[tuple[object, ...], ...],
    global_rates: dict[str, float],
) -> pd.DataFrame:
    end_indices = season_rows.groupby("pitcher_id", sort=False)[
        n_column
    ].idxmax()
    columns = ["pitcher_id", n_column, "control_success"] + [
        str(metric[1]) for metric in metrics
    ]
    last = season_rows.loc[end_indices, columns]
    n_before = last[n_column].fillna(0.0).to_numpy(dtype=np.float64)
    result: dict[str, np.ndarray] = {
        "pitcher_id": last["pitcher_id"].to_numpy(),
        "prior_n": n_before + 1.0,
    }
    for metric in metrics:
        name = str(metric[0])
        rate_column = str(metric[1])
        count = np.rint(
            n_before
            * last[rate_column]
            .fillna(global_rates[name])
            .to_numpy(dtype=np.float64)
        )
        exact_target = bool(metric[3]) if len(metric) == 4 else False
        if exact_target:
            count += last["control_success"].to_numpy(dtype=np.float64)
        result[f"prior_{name}_count"] = count
    return pd.DataFrame(result).set_index("pitcher_id")


def _merge_state(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        return new
    combined = pd.concat([old, new])
    return combined[~combined.index.duplicated(keep="last")]


def attach_temporal_profile_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("train rows must be season ordered")
    control_state = _empty_state(
        tuple(str(metric[0]) for metric in CONTROL_METRICS)
    )
    pitchmix_state = _empty_state(
        tuple(str(metric[0]) for metric in PITCHMIX_METRICS)
    )
    outputs: list[pd.DataFrame] = []
    diagnostics: dict[str, object] = {}
    for season in ALL_SEASONS:
        rows = frame.loc[frame["season"].eq(season)].copy()
        control_globals = _global_rates(control_state, CONTROL_METRICS)
        pitchmix_globals = _global_rates(
            pitchmix_state, PITCHMIX_METRICS
        )
        rows, control_diagnostics = _attach_state_group(
            rows,
            control_state,
            "asof_pitcher_n",
            CONTROL_METRICS,
            "temporal_control",
            control_globals,
        )
        rows, pitchmix_diagnostics = _attach_state_group(
            rows,
            pitchmix_state,
            "asof_pitcher_pitchmix_n",
            PITCHMIX_METRICS,
            "temporal_pitchmix",
            pitchmix_globals,
        )
        outputs.append(rows)
        control_state = _merge_state(
            control_state,
            _updated_state(
                rows,
                "asof_pitcher_n",
                CONTROL_METRICS,
                control_globals,
            ),
        )
        pitchmix_state = _merge_state(
            pitchmix_state,
            _updated_state(
                rows,
                "asof_pitcher_pitchmix_n",
                PITCHMIX_METRICS,
                pitchmix_globals,
            ),
        )
        diagnostics[str(season)] = {
            "rows": int(len(rows)),
            "control_state_pitchers_after": int(len(control_state)),
            "pitchmix_state_pitchers_after": int(len(pitchmix_state)),
            "control_global_rates_before": control_globals,
            "pitchmix_global_rates_before": pitchmix_globals,
            "control_reconstruction": control_diagnostics,
            "pitchmix_reconstruction": pitchmix_diagnostics,
        }
    output = pd.concat(outputs).sort_index()
    if len(output) != len(frame):
        raise AssertionError("temporal profile row count mismatch")
    return output, diagnostics


def load_rows() -> tuple[dict[int, pd.DataFrame], dict[str, object]]:
    columns = [
        "season",
        "pitcher_id",
        "pitcher_hand",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "control_success",
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
        "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    frame = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        usecols=columns,
    )
    observed_seasons = tuple(
        sorted(frame["season"].astype(int).unique().tolist())
    )
    if observed_seasons != ALL_SEASONS:
        raise ValueError(f"unexpected seasons: {observed_seasons}")
    if not np.array_equal(
        frame["asof_pitcher_n"].to_numpy(),
        frame["asof_pitcher_pitchmix_n"].to_numpy(),
    ):
        raise ValueError("pitchmix n differs from official pitcher n")
    frame, temporal_diagnostics = attach_temporal_profile_features(frame)
    frame = frame.loc[frame["season"].isin(EVALUATED_SEASONS)].copy()
    frame["count_index"] = (
        frame["balls_before"].to_numpy(dtype=np.int8) * 4
        + frame["strikes_before"].to_numpy(dtype=np.int8)
    ).astype(np.int8)
    if not set(frame["count_index"].astype(int).unique()).issubset(
        set(COUNT_INDICES)
    ):
        raise ValueError("unexpected count domain")
    if not set(frame["pitcher_hand"].astype(int).unique()).issubset(
        set(PITCHER_HANDS)
    ):
        raise ValueError("unexpected pitcher hand domain")
    if not set(frame["batter_hand"].astype(int).unique()).issubset(
        set(BATTER_HANDS)
    ):
        raise ValueError("unexpected batter hand domain")
    frame["context_position"] = [
        CONTEXT_TO_POSITION[
            (int(count_index), int(pitcher_hand), int(batter_hand))
        ]
        for count_index, pitcher_hand, batter_hand in zip(
            frame["count_index"],
            frame["pitcher_hand"],
            frame["batter_hand"],
            strict=True,
        )
    ]
    frame["context_position"] = frame["context_position"].astype(np.int8)
    for column in TEMPORAL_PROFILE_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"missing temporal profile feature: {column}")
    return (
        {
            season: frame.loc[frame["season"].eq(season)].reset_index(
                drop=True
            )
            for season in EVALUATED_SEASONS
        },
        temporal_diagnostics,
    )


def load_oof(
    rows: dict[int, pd.DataFrame],
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    targets: dict[int, np.ndarray] = {}
    strict_base: dict[int, np.ndarray] = {}
    identity_reference: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(
            TARGET_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        strict_base[season] = np.load(
            IDENTITY_ROOT / f"predictions_strict_rank_s300_{season}.npy"
        ).astype(np.float64)
        identity_reference[season] = np.load(
            IDENTITY_ROOT / f"predictions_lowrank_s300_r4_{season}.npy"
        ).astype(np.float64)
        csv_target = rows[season]["control_success"].to_numpy(
            dtype=np.float64
        )
        if not (
            len(csv_target)
            == len(targets[season])
            == len(strict_base[season])
            == len(identity_reference[season])
            and np.array_equal(csv_target, targets[season])
        ):
            raise ValueError(f"OOF target/order mismatch {season}")
        for label, prediction in (
            ("strict", strict_base[season]),
            ("identity", identity_reference[season]),
        ):
            if not np.isfinite(prediction).all() or not (
                (prediction >= 0.0).all() and (prediction <= 1.0).all()
            ):
                raise ValueError(f"invalid {label} prediction {season}")
    return targets, strict_base, identity_reference


def source_snapshots(rows: pd.DataFrame) -> pd.DataFrame:
    indices = rows.groupby("pitcher_id", sort=False)[
        "asof_pitcher_n"
    ].idxmax()
    snapshots = rows.loc[indices].copy()
    snapshots = snapshots.loc[
        snapshots["asof_pitcher_n"].ge(SNAPSHOT_MIN_N)
    ].reset_index(drop=True)
    if snapshots["pitcher_id"].duplicated().any():
        raise AssertionError("snapshot pitcher duplicated")
    return snapshots


def fit_profile_preprocessor(snapshots: pd.DataFrame) -> dict[str, object]:
    impute_columns = (*CAREER_PROFILE_COLUMNS, *RECENT_PROFILE_COLUMNS)
    impute_means = {
        column: float(snapshots[column].mean(skipna=True))
        for column in impute_columns
    }
    if not all(np.isfinite(value) for value in impute_means.values()):
        raise ValueError("non-finite source imputation mean")
    raw = transform_profile(snapshots, impute_means)
    mean = raw.mean(axis=0, dtype=np.float64)
    variance = np.mean(np.square(raw - mean), axis=0, dtype=np.float64)
    scale = np.sqrt(np.maximum(variance, 1e-8))
    return {
        "impute_means": impute_means,
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
    }


def transform_profile(
    rows: pd.DataFrame,
    impute_means: dict[str, float],
) -> np.ndarray:
    n = rows["asof_pitcher_n"].to_numpy(dtype=np.float64)
    reliability = n / (n + CAREER_STABILIZATION)
    columns: list[np.ndarray] = []
    for column in CAREER_PROFILE_COLUMNS:
        mean = float(impute_means[column])
        value = rows[column].fillna(mean).to_numpy(dtype=np.float64)
        columns.append(mean + reliability * (value - mean))
    for column in RECENT_PROFILE_COLUMNS:
        mean = float(impute_means[column])
        columns.append(
            rows[column].fillna(mean).to_numpy(dtype=np.float64)
        )
    recent_missing = rows[RECENT_PROFILE_COLUMNS[0]].isna().to_numpy(
        dtype=np.float64
    )
    columns.append(recent_missing)
    for column in TEMPORAL_PROFILE_COLUMNS:
        columns.append(rows[column].to_numpy(dtype=np.float64))
    matrix = np.column_stack(columns).astype(np.float32)
    if matrix.shape[1] != len(PROFILE_FEATURES):
        raise AssertionError("profile feature count mismatch")
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite transformed profile")
    return matrix


def squared_distances(
    standardized: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        np.sum(np.square(standardized), axis=1)[:, None]
        + np.sum(np.square(centers), axis=1)[None, :]
        - 2.0 * standardized @ centers.T,
        0.0,
    )


def fit_source_cluster_model(
    source_season: int,
    rows: pd.DataFrame,
    cluster_count: int,
) -> dict[str, object]:
    snapshots = source_snapshots(rows)
    if len(snapshots) < cluster_count:
        raise ValueError("too few eligible pitcher snapshots")
    preprocessor = fit_profile_preprocessor(snapshots)
    raw = transform_profile(snapshots, preprocessor["impute_means"])
    standardized = (
        (raw - preprocessor["mean"]) / preprocessor["scale"]
    ).astype(np.float32)
    kmeans = KMeans(
        n_clusters=cluster_count,
        init="k-means++",
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        random_state=KMEANS_RANDOM_STATE,
        algorithm="lloyd",
    ).fit(standardized)
    centers = kmeans.cluster_centers_.astype(np.float32)
    distances = squared_distances(standardized, centers)
    hard = np.argmin(distances, axis=1).astype(np.int16)
    nearest_squared = distances[np.arange(len(hard)), hard]
    tau_squared = float(max(np.median(nearest_squared), 1e-6))

    seed_labels: dict[int, np.ndarray] = {}
    for seed in STABILITY_SEEDS:
        seed_labels[seed] = KMeans(
            n_clusters=cluster_count,
            init="k-means++",
            n_init=1,
            max_iter=KMEANS_MAX_ITER,
            random_state=seed,
            algorithm="lloyd",
        ).fit_predict(standardized)
    pairwise_ari = [
        float(adjusted_rand_score(seed_labels[left], seed_labels[right]))
        for left, right in itertools.combinations(STABILITY_SEEDS, 2)
    ]
    cluster_rows = np.bincount(hard, minlength=cluster_count).astype(np.int64)
    pairwise_center_distance = np.sqrt(
        np.maximum(
            np.sum(
                np.square(centers[:, None, :] - centers[None, :, :]),
                axis=2,
            ),
            0.0,
        )
    )
    pairwise_center_distance[
        pairwise_center_distance == 0.0
    ] = np.inf
    return {
        "source_season": source_season,
        "cluster_count": cluster_count,
        "snapshot_pitcher_ids": snapshots["pitcher_id"].to_numpy(),
        "snapshot_hard": hard,
        "impute_means": preprocessor["impute_means"],
        "mean": preprocessor["mean"],
        "scale": preprocessor["scale"],
        "centers": centers,
        "raw_centers": centers * preprocessor["scale"] + preprocessor["mean"],
        "tau_squared": tau_squared,
        "diagnostics": {
            "source_season": source_season,
            "source_rows": int(len(rows)),
            "source_pitchers": int(rows["pitcher_id"].nunique()),
            "eligible_equal_pitcher_snapshots": int(len(snapshots)),
            "snapshot_min_n": SNAPSHOT_MIN_N,
            "impute_means": preprocessor["impute_means"],
            "scaler_mean": [float(value) for value in preprocessor["mean"]],
            "scaler_scale": [float(value) for value in preprocessor["scale"]],
            "inertia_per_snapshot": float(kmeans.inertia_ / len(snapshots)),
            "iterations": int(kmeans.n_iter_),
            "cluster_snapshot_rows": [int(value) for value in cluster_rows],
            "min_cluster_snapshot_rows": int(cluster_rows.min()),
            "min_cluster_snapshot_share": float(
                cluster_rows.min() / len(snapshots)
            ),
            "empty_cluster_count": int((cluster_rows == 0).sum()),
            "tau_squared_source_median_nearest": tau_squared,
            "nearest_squared_distance_median": float(
                np.median(nearest_squared)
            ),
            "nearest_squared_distance_p95": float(
                np.quantile(nearest_squared, 0.95)
            ),
            "minimum_center_pairwise_distance": float(
                pairwise_center_distance.min()
            ),
            "seed_stability_pairwise_ari": pairwise_ari,
            "seed_stability_mean_ari": float(np.mean(pairwise_ari)),
            "seed_stability_min_ari": float(np.min(pairwise_ari)),
            "standardized_centers": [
                [float(value) for value in center] for center in centers
            ],
        },
    }


def assign_rows(
    rows: pd.DataFrame,
    model: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = transform_profile(rows, model["impute_means"])
    standardized = ((raw - model["mean"]) / model["scale"]).astype(
        np.float32
    )
    distances = squared_distances(standardized, model["centers"])
    hard = np.argmin(distances, axis=1).astype(np.int16)
    shifted = distances - distances.min(axis=1, keepdims=True)
    logits = -shifted / (2.0 * float(model["tau_squared"]))
    soft = np.exp(np.clip(logits, -60.0, 0.0)).astype(np.float32)
    soft /= soft.sum(axis=1, keepdims=True)
    if not np.allclose(soft.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("soft assignment normalization failed")
    return hard, soft, distances


def fit_source_residual_maps(
    source_season: int,
    rows: pd.DataFrame,
    targets: np.ndarray,
    strict_base: np.ndarray,
    model: dict[str, object],
) -> dict[str, object]:
    raw_residual = targets - strict_base
    raw_residual_mean = float(raw_residual.mean())
    residual = raw_residual - raw_residual_mean
    if abs(float(residual.mean())) > 1e-12:
        raise AssertionError("source residual centering failed")
    hard, soft, distances = assign_rows(rows, model)
    context = rows["context_position"].to_numpy(dtype=np.int16)
    history_known = rows["asof_pitcher_n"].to_numpy(dtype=np.float64) > 0.0
    cluster_count = int(model["cluster_count"])
    shape = (cluster_count, len(CONTEXTS))
    hard_sum = np.zeros(shape, dtype=np.float64)
    hard_count = np.zeros(shape, dtype=np.float64)
    np.add.at(
        hard_sum,
        (hard[history_known], context[history_known]),
        residual[history_known],
    )
    np.add.at(
        hard_count,
        (hard[history_known], context[history_known]),
        1.0,
    )
    soft_sum = np.zeros(shape, dtype=np.float64)
    soft_count = np.zeros(shape, dtype=np.float64)
    soft_square_sum = np.zeros(shape, dtype=np.float64)
    for context_position in range(len(CONTEXTS)):
        mask = history_known & (context == context_position)
        if not mask.any():
            continue
        membership = soft[mask].astype(np.float64)
        soft_count[:, context_position] = membership.sum(axis=0)
        soft_square_sum[:, context_position] = np.square(membership).sum(axis=0)
        soft_sum[:, context_position] = membership.T @ residual[mask]
    hard_effect_unclipped = hard_sum / (hard_count + MAP_SMOOTHING)
    soft_effect_unclipped = soft_sum / (soft_count + MAP_SMOOTHING)
    hard_effect = np.clip(hard_effect_unclipped, -EFFECT_CLIP, EFFECT_CLIP)
    soft_effect = np.clip(soft_effect_unclipped, -EFFECT_CLIP, EFFECT_CLIP)
    soft_effective_n = np.divide(
        np.square(soft_count),
        soft_square_sum,
        out=np.zeros_like(soft_count),
        where=soft_square_sum > 0.0,
    )
    nearest = np.sqrt(
        distances[np.arange(len(hard)), hard]
    )
    return {
        "source_season": source_season,
        "source_pitcher_ids": set(rows["pitcher_id"].unique().tolist()),
        "hard_count": hard_count,
        "soft_count": soft_count,
        "soft_effective_n": soft_effective_n,
        "hard_effect": hard_effect,
        "soft_effect": soft_effect,
        "diagnostics": {
            "source_season": source_season,
            "source_rows": int(len(rows)),
            "history_known_rows_used": int(history_known.sum()),
            "history_new_rows_forced_zero": int((~history_known).sum()),
            "raw_residual_mean_before_centering": raw_residual_mean,
            "residual_mean_after_centering": float(residual.mean()),
            "hard_observed_cells": int((hard_count > 0.0).sum()),
            "hard_cells_below_smoothing_mass": int(
                ((hard_count > 0.0) & (hard_count < MAP_SMOOTHING)).sum()
            ),
            "soft_positive_cells": int((soft_count > 0.0).sum()),
            "soft_cells_below_smoothing_mass": int(
                ((soft_count > 0.0) & (soft_count < MAP_SMOOTHING)).sum()
            ),
            "soft_effective_n_mean_positive": float(
                soft_effective_n[soft_effective_n > 0.0].mean()
            ),
            "hard_effect_mean_absolute": float(np.abs(hard_effect).mean()),
            "hard_effect_max_absolute": float(np.abs(hard_effect).max()),
            "soft_effect_mean_absolute": float(np.abs(soft_effect).mean()),
            "soft_effect_max_absolute": float(np.abs(soft_effect).max()),
            "hard_effect_clip_count": int(
                (np.abs(hard_effect_unclipped) > EFFECT_CLIP).sum()
            ),
            "soft_effect_clip_count": int(
                (np.abs(soft_effect_unclipped) > EFFECT_CLIP).sum()
            ),
            "nearest_center_distance_mean": float(nearest.mean()),
            "nearest_center_distance_p95": float(
                np.quantile(nearest, 0.95)
            ),
        },
    }


def map_source_correction(
    rows: pd.DataFrame,
    model: dict[str, object],
    residual_map: dict[str, object],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    hard, soft, distances = assign_rows(rows, model)
    context = rows["context_position"].to_numpy(dtype=np.int16)
    history_known = rows["asof_pitcher_n"].to_numpy(dtype=np.float64) > 0.0
    if mode == "hard":
        correction = residual_map["hard_effect"][hard, context]
        mass = residual_map["hard_count"][hard, context]
    elif mode == "soft":
        effects = residual_map["soft_effect"][:, context].T
        masses = residual_map["soft_count"][:, context].T
        correction = np.sum(soft * effects, axis=1)
        mass = np.sum(soft * masses, axis=1)
    else:
        raise ValueError(f"unexpected assignment mode {mode}")
    correction = correction.astype(np.float64)
    correction[~history_known] = 0.0
    seen = history_known & (mass > 0.0)
    nearest = np.sqrt(distances[np.arange(len(hard)), hard])
    sorted_distance = np.partition(distances, 1, axis=1)[:, :2]
    margin = sorted_distance[:, 1] - sorted_distance[:, 0]
    entropy = -np.sum(
        soft * np.log(np.maximum(soft, 1e-15)), axis=1
    ) / math.log(int(model["cluster_count"]))
    assignment_diagnostics = {
        "hard_cluster_rows": [
            int(value)
            for value in np.bincount(
                hard, minlength=int(model["cluster_count"])
            )
        ],
        "nearest_center_distance_mean": float(nearest.mean()),
        "nearest_center_distance_p95": float(np.quantile(nearest, 0.95)),
        "hard_margin_p05": float(np.quantile(margin, 0.05)),
        "hard_margin_median": float(np.median(margin)),
        "soft_max_membership_mean": float(soft.max(axis=1).mean()),
        "soft_normalized_entropy_mean": float(entropy.mean()),
        "soft_normalized_entropy_p95": float(np.quantile(entropy, 0.95)),
    }
    return correction, seen, mass, assignment_diagnostics


def matched_source_stability(
    previous_model: dict[str, object] | None,
    current_model: dict[str, object],
    previous_map: dict[str, object] | None,
    current_map: dict[str, object],
) -> dict[str, object]:
    if previous_model is None or previous_map is None:
        return {"previous_source_available": False}
    previous_ids = pd.Index(previous_model["snapshot_pitcher_ids"])
    current_ids = pd.Index(current_model["snapshot_pitcher_ids"])
    common = previous_ids.intersection(current_ids)
    previous_positions = previous_ids.get_indexer(common)
    current_positions = current_ids.get_indexer(common)
    previous_labels = previous_model["snapshot_hard"][previous_positions]
    current_labels = current_model["snapshot_hard"][current_positions]

    previous_centers_in_current_scale = (
        previous_model["raw_centers"] - current_model["mean"]
    ) / current_model["scale"]
    current_centers = current_model["centers"]
    cost = np.sqrt(
        np.maximum(
            np.sum(
                np.square(
                    current_centers[:, None, :]
                    - previous_centers_in_current_scale[None, :, :]
                ),
                axis=2,
            ),
            0.0,
        )
    )
    current_indices, previous_indices = linear_sum_assignment(cost)
    previous_for_current = np.full(
        len(current_indices), -1, dtype=np.int16
    )
    previous_for_current[current_indices] = previous_indices
    matched_distance = cost[current_indices, previous_indices]

    effect_stability: dict[str, object] = {}
    for mode in ASSIGNMENT_MODES:
        previous_effect = previous_map[f"{mode}_effect"]
        current_effect = current_map[f"{mode}_effect"]
        aligned_previous = previous_effect[previous_for_current]
        flat_previous = aligned_previous.ravel()
        flat_current = current_effect.ravel()
        correlation = float(
            np.corrcoef(flat_previous, flat_current)[0, 1]
        )
        sign_mask = (flat_previous != 0.0) & (flat_current != 0.0)
        effect_stability[mode] = {
            "aligned_effect_correlation": correlation,
            "aligned_nonzero_sign_agreement": (
                float(
                    np.mean(
                        np.sign(flat_previous[sign_mask])
                        == np.sign(flat_current[sign_mask])
                    )
                )
                if sign_mask.any()
                else None
            ),
        }
    return {
        "previous_source_available": True,
        "previous_source_season": int(previous_model["source_season"]),
        "common_eligible_pitchers": int(len(common)),
        "common_pitcher_adjusted_rand": float(
            adjusted_rand_score(previous_labels, current_labels)
        ),
        "common_pitcher_normalized_mutual_info": float(
            normalized_mutual_info_score(previous_labels, current_labels)
        ),
        "matched_center_distance_mean": float(matched_distance.mean()),
        "matched_center_distance_max": float(matched_distance.max()),
        "matched_center_distance_per_feature_mean": float(
            matched_distance.mean() / math.sqrt(len(PROFILE_FEATURES))
        ),
        "current_to_previous_center_matching": [
            int(value) for value in previous_for_current
        ],
        "effect_stability": effect_stability,
    }


def segment_masks(
    rows: pd.DataFrame,
    source_pitcher_ids: set[object],
) -> dict[str, np.ndarray]:
    n = rows["asof_pitcher_n"].to_numpy(dtype=np.float64)
    source_seen = rows["pitcher_id"].isin(source_pitcher_ids).to_numpy()
    return {
        "history_new_n0": n == 0.0,
        "history_known_n_positive": n > 0.0,
        "profile_reliable_n20_plus": n >= SNAPSHOT_MIN_N,
        "profile_unreliable_n1_19": (n > 0.0) & (n < SNAPSHOT_MIN_N),
        "source_id_seen_any": source_seen,
        "transfer_cold_start": (n > 0.0) & ~source_seen,
        "career_n_0": n == 0.0,
        "career_n_1_19": (n >= 1.0) & (n < 20.0),
        "career_n_20_99": (n >= 20.0) & (n < 100.0),
        "career_n_100_499": (n >= 100.0) & (n < 500.0),
        "career_n_500_plus": n >= 500.0,
    }


def segment_metrics(
    rows: pd.DataFrame,
    targets: np.ndarray,
    predictions: dict[str, np.ndarray],
    source_pitcher_ids: set[object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, mask in segment_masks(rows, source_pitcher_ids).items():
        count = int(mask.sum())
        result[name] = {
            "rows": count,
            "actual_rate": float(targets[mask].mean()) if count else None,
            "candidates": {
                candidate: (
                    calculate_metrics(targets[mask], prediction[mask])
                    if count
                    else None
                )
                for candidate, prediction in predictions.items()
            },
        }
    return result


def aggregate_metrics(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for candidate in CANDIDATES:
        briers = {
            season: float(
                folds[str(season)]["candidates"][candidate]["brier_score"]
            )
            for season in REPORT_SEASONS
        }
        skills = {
            season: float(
                folds[str(season)]["candidates"][candidate][
                    "skill_score_unclipped"
                ]
            )
            for season in REPORT_SEASONS
        }
        aggregate[candidate] = {
            "season_briers": {
                str(season): value for season, value in briers.items()
            },
            "season_skills": {
                str(season): value for season, value in skills.items()
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
        }
    base = aggregate[BASE_CANDIDATE]
    identity = aggregate[FIXED_IDENTITY_REFERENCE]
    for candidate in ARCHETYPE_CANDIDATES:
        summary = aggregate[candidate]
        for label, reference in (
            ("strict_base", base),
            ("fixed_identity_r4", identity),
        ):
            summary[f"season_skill_change_vs_{label}"] = {
                str(season): float(
                    summary["season_skills"][str(season)]
                    - reference["season_skills"][str(season)]
                )
                for season in REPORT_SEASONS
            }
            summary[f"mean_skill_change_vs_{label}"] = float(
                summary["mean_skill"] - reference["mean_skill"]
            )
            summary[f"min_skill_change_vs_{label}"] = float(
                summary["min_skill"] - reference["min_skill"]
            )
        summary["beats_strict_base_every_report_season"] = bool(
            all(
                value > 0.0
                for value in summary[
                    "season_skill_change_vs_strict_base"
                ].values()
            )
        )
    return aggregate


def main() -> None:
    started = time.time()
    rows, temporal_diagnostics = load_rows()
    targets, strict_base, identity_reference = load_oof(rows)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    source_models: dict[tuple[int, int], dict[str, object]] = {}
    source_maps: dict[tuple[int, int], dict[str, object]] = {}

    def get_source_model(
        source_season: int, cluster_count: int
    ) -> dict[str, object]:
        key = (source_season, cluster_count)
        if key not in source_models:
            source_models[key] = fit_source_cluster_model(
                source_season, rows[source_season], cluster_count
            )
        return source_models[key]

    def get_source_map(
        source_season: int, cluster_count: int
    ) -> dict[str, object]:
        key = (source_season, cluster_count)
        if key not in source_maps:
            source_maps[key] = fit_source_residual_maps(
                source_season,
                rows[source_season],
                targets[source_season],
                strict_base[source_season],
                get_source_model(source_season, cluster_count),
            )
        return source_maps[key]

    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        source_seasons = [
            season
            for season in EVALUATED_SEASONS
            if season < validation_season
        ]
        predictions: dict[str, np.ndarray] = {
            BASE_CANDIDATE: strict_base[validation_season].copy(),
            FIXED_IDENTITY_REFERENCE: identity_reference[
                validation_season
            ].copy(),
        }
        corrections: dict[str, np.ndarray] = {}
        coverage: dict[str, object] = {}
        fold_source_model_diagnostics: dict[str, object] = {}
        fold_source_map_diagnostics: dict[str, object] = {}
        stability: dict[str, object] = {}
        assignment_diagnostics: dict[str, object] = {}
        seen_count_arrays: dict[str, np.ndarray] = {}

        all_source_pitcher_ids: set[object] = set()
        for source_season in source_seasons:
            all_source_pitcher_ids.update(
                rows[source_season]["pitcher_id"].unique().tolist()
            )

        if source_seasons:
            for cluster_count in CLUSTER_COUNTS:
                fold_source_model_diagnostics[str(cluster_count)] = {}
                fold_source_map_diagnostics[str(cluster_count)] = {}
                stability[str(cluster_count)] = {}
                models = {
                    source_season: get_source_model(
                        source_season, cluster_count
                    )
                    for source_season in source_seasons
                }
                maps = {
                    source_season: get_source_map(
                        source_season, cluster_count
                    )
                    for source_season in source_seasons
                }
                previous_model: dict[str, object] | None = None
                previous_map: dict[str, object] | None = None
                for source_season in source_seasons:
                    fold_source_model_diagnostics[str(cluster_count)][
                        str(source_season)
                    ] = models[source_season]["diagnostics"]
                    fold_source_map_diagnostics[str(cluster_count)][
                        str(source_season)
                    ] = maps[source_season]["diagnostics"]
                    stability[str(cluster_count)][str(source_season)] = (
                        matched_source_stability(
                            previous_model,
                            models[source_season],
                            previous_map,
                            maps[source_season],
                        )
                    )
                    previous_model = models[source_season]
                    previous_map = maps[source_season]

                for mode in ASSIGNMENT_MODES:
                    candidate = archetype_name(cluster_count, mode)
                    source_corrections: list[np.ndarray] = []
                    source_seen: list[np.ndarray] = []
                    source_mass: list[np.ndarray] = []
                    assignment_diagnostics[candidate] = {}
                    for source_season in source_seasons:
                        correction, seen, mass, diagnostics = (
                            map_source_correction(
                                rows[validation_season],
                                models[source_season],
                                maps[source_season],
                                mode,
                            )
                        )
                        source_corrections.append(correction)
                        source_seen.append(seen)
                        source_mass.append(mass)
                        assignment_diagnostics[candidate][
                            str(source_season)
                        ] = diagnostics
                    correction = np.mean(
                        np.vstack(source_corrections), axis=0
                    )
                    seen_matrix = np.vstack(source_seen)
                    mass_matrix = np.vstack(source_mass)
                    seen_count = seen_matrix.sum(axis=0).astype(np.int8)
                    history_new = rows[validation_season][
                        "asof_pitcher_n"
                    ].to_numpy(dtype=np.float64) == 0.0
                    if np.any(correction[history_new] != 0.0):
                        raise AssertionError("history-new correction nonzero")
                    prediction = np.clip(
                        strict_base[validation_season] + correction,
                        0.0,
                        1.0,
                    )
                    predictions[candidate] = prediction
                    corrections[candidate] = correction
                    seen_count_arrays[candidate] = seen_count
                    coverage[candidate] = {
                        "source_count": len(source_seasons),
                        "rows": int(len(prediction)),
                        "map_seen_any_source_rows": int(
                            (seen_count > 0).sum()
                        ),
                        "map_seen_any_source_rate": float(
                            (seen_count > 0).mean()
                        ),
                        "map_seen_every_source_rows": int(
                            (seen_count == len(source_seasons)).sum()
                        ),
                        "map_seen_every_source_rate": float(
                            (seen_count == len(source_seasons)).mean()
                        ),
                        "effective_mass_mean_by_source": [
                            float(value.mean()) for value in mass_matrix
                        ],
                        "effective_mass_p05_by_source": [
                            float(np.quantile(value, 0.05))
                            for value in mass_matrix
                        ],
                        "correction_mean": float(correction.mean()),
                        "correction_standard_deviation": float(
                            correction.std()
                        ),
                        "correction_mean_absolute": float(
                            np.abs(correction).mean()
                        ),
                        "correction_min": float(correction.min()),
                        "correction_max": float(correction.max()),
                    }
        else:
            zero = np.zeros(
                len(rows[validation_season]), dtype=np.float64
            )
            for candidate in ARCHETYPE_CANDIDATES:
                predictions[candidate] = strict_base[
                    validation_season
                ].copy()
                corrections[candidate] = zero.copy()
                seen_count_arrays[candidate] = np.zeros(
                    len(zero), dtype=np.int8
                )
                coverage[candidate] = {
                    "source_count": 0,
                    "rows": int(len(zero)),
                    "map_seen_any_source_rows": 0,
                    "map_seen_any_source_rate": 0.0,
                    "map_seen_every_source_rows": 0,
                    "map_seen_every_source_rate": 0.0,
                    "effective_mass_mean_by_source": [],
                    "effective_mass_p05_by_source": [],
                    "correction_mean": 0.0,
                    "correction_standard_deviation": 0.0,
                    "correction_mean_absolute": 0.0,
                    "correction_min": 0.0,
                    "correction_max": 0.0,
                }

        if tuple(predictions) != CANDIDATES:
            raise AssertionError("candidate declaration/order drift")
        metrics = {
            candidate: calculate_metrics(
                targets[validation_season], prediction
            )
            for candidate, prediction in predictions.items()
        }
        source_seen_array = rows[validation_season]["pitcher_id"].isin(
            all_source_pitcher_ids
        ).to_numpy(dtype=np.int8)
        history_known_array = rows[validation_season][
            "asof_pitcher_n"
        ].to_numpy(dtype=np.float64) > 0.0
        np.save(
            ARTIFACT_DIR
            / f"source_id_seen_any_{validation_season}.npy",
            source_seen_array,
        )
        np.save(
            ARTIFACT_DIR
            / f"history_known_{validation_season}.npy",
            history_known_array.astype(np.int8),
        )
        for candidate, prediction in predictions.items():
            if not np.isfinite(prediction).all() or not (
                (prediction >= 0.0).all() and (prediction <= 1.0).all()
            ):
                raise ValueError(
                    f"invalid prediction {validation_season} {candidate}"
                )
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
            np.save(
                ARTIFACT_DIR
                / f"map_seen_source_count_{candidate}_{validation_season}.npy",
                seen_count_arrays[candidate],
            )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets[validation_season].astype(np.int8),
        )
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "rows": int(len(rows[validation_season])),
            "source_cluster_models": fold_source_model_diagnostics,
            "source_residual_maps": fold_source_map_diagnostics,
            "source_cluster_stability": stability,
            "validation_assignment_diagnostics_not_used_for_fit": (
                assignment_diagnostics
            ),
            "cluster_map_coverage": coverage,
            "segment_metrics": segment_metrics(
                rows[validation_season],
                targets[validation_season],
                predictions,
                all_source_pitcher_ids,
            ),
            "candidates": metrics,
            "strict_source_invariants": {
                "all_source_seasons_precede_validation": bool(
                    all(
                        source_season < validation_season
                        for source_season in source_seasons
                    )
                ),
                "current_fold_labels_used_for_fit_or_selection": False,
                "validation_or_test_rows_used_for_fit_aggregation": False,
                "validation_rows_transformed_independently": True,
                "source_models_and_maps_independent_by_season": True,
                "source_residuals_centered_within_season": True,
                "source_corrections_combined_with_equal_weight": True,
                "raw_id_team_season_in_cluster_input": False,
                "history_new_n0_correction_zero": True,
            },
        }
        print(
            f"archetype {validation_season}: "
            + " ".join(
                f"{candidate}="
                f"{metrics[candidate]['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    aggregate = aggregate_metrics(folds)
    best_mean = max(
        ARCHETYPE_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["mean_skill"],
            aggregate[candidate]["min_skill"],
            -ARCHETYPE_CANDIDATES.index(candidate),
        ),
    )
    best_min = max(
        ARCHETYPE_CANDIDATES,
        key=lambda candidate: (
            aggregate[candidate]["min_skill"],
            aggregate[candidate]["latest_2024_skill"],
            aggregate[candidate]["mean_skill"],
            -ARCHETYPE_CANDIDATES.index(candidate),
        ),
    )
    result: dict[str, object] = {
        "experiment": "EXP-021",
        "candidate_family": (
            "official_current_row_player_state_archetype_context_EB"
        ),
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": (
                "saved low_rank_pitcher_context_eb strict_rank_s300 OOF"
            ),
            "base_rank_selection_uses_current_fold": False,
            "effect_target": (
                "source-season-centered target minus strict base OOF"
            ),
            "cluster_and_map_fit": (
                "independent inside each earlier OOF source season"
            ),
            "source_map_combination": (
                "equal mean across all earlier sources including zero"
            ),
            "current_fold_labels_used_for_fit_or_selection": False,
            "validation_or_test_row_aggregation_for_prediction": False,
            "test_csv_read": False,
            "candidate_grid_predeclared": True,
            "candidate_comparison_nested": False,
            "post_result_expansion": False,
        },
        "predeclared_configuration": {
            "cluster_counts": list(CLUSTER_COUNTS),
            "assignment_modes": list(ASSIGNMENT_MODES),
            "candidate_count": len(ARCHETYPE_CANDIDATES),
            "snapshot_min_official_n": SNAPSHOT_MIN_N,
            "equal_pitcher_source_snapshots": True,
            "career_rate_stabilization": CAREER_STABILIZATION,
            "context": (
                "static official count_index x pitcher_hand x batter_hand"
            ),
            "context_count": len(CONTEXTS),
            "map_smoothing": MAP_SMOOTHING,
            "effect_clip": [-EFFECT_CLIP, EFFECT_CLIP],
            "soft_temperature": (
                "source-only median nearest-centroid squared distance"
            ),
            "kmeans": {
                "implementation": "sklearn KMeans",
                "n_init": KMEANS_N_INIT,
                "max_iter": KMEANS_MAX_ITER,
                "random_state": KMEANS_RANDOM_STATE,
                "algorithm": "lloyd",
            },
            "profile_features": list(PROFILE_FEATURES),
            "profile_feature_count": len(PROFILE_FEATURES),
            "raw_player_ids_in_cluster_input": False,
            "raw_team_ids_in_cluster_input": False,
            "raw_season_in_cluster_input": False,
            "raw_sample_count_in_cluster_distance": False,
            "official_n_zero_rule": "correction exactly zero",
            "source_only_missing_rule": (
                "source eligible-pitcher snapshot mean imputation"
            ),
        },
        "temporal_profile_reconstruction_diagnostics": temporal_diagnostics,
        "source_model_cache_diagnostics": {
            f"{season}_k{cluster_count}": {
                "cluster_model": model["diagnostics"],
                "residual_map": source_maps[(season, cluster_count)][
                    "diagnostics"
                ],
            }
            for (season, cluster_count), model in source_models.items()
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "diagnostic only; candidate comparison is post-hoc",
            "posthoc_best_mean_candidate": best_mean,
            "posthoc_best_min_candidate": best_min,
            "best_mean_beats_strict_base_every_report_season": bool(
                aggregate[best_mean][
                    "beats_strict_base_every_report_season"
                ]
            ),
            "best_min_exceeds_1100": bool(
                aggregate[best_min]["min_skill"] >= 1100.0
            ),
            "stop_archetype_family": bool(
                aggregate[best_min]["min_skill"] < 1100.0
            ),
            "adopt_without_nested_confirmation": False,
        },
        "qa": {
            "source_target_and_row_order_alignment_checked": True,
            "strict_base_and_identity_reference_ranges_checked": True,
            "official_pitcher_and_pitchmix_n_equality_checked": True,
            "temporal_profile_previous_season_state_checked": True,
            "source_snapshot_equal_pitcher_weight_checked": True,
            "source_only_imputer_scaler_centers_temperature_checked": True,
            "cluster_feature_exclusions_checked": True,
            "source_season_order_checked": True,
            "source_residual_centering_checked": True,
            "soft_assignment_normalization_checked": True,
            "history_new_zero_correction_checked": True,
            "prediction_probability_ranges_checked": True,
            "saved_prediction_correction_coverage_segment_arrays": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn_version,
        },
        "total_seconds": float(time.time() - started),
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
