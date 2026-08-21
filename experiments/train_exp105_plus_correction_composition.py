"""EXP-105--111: compose Public-positive historical correction mechanisms.

The common baseline is EXP-051.  EXP-063/064/071/072 are loaded as additive
OOF correction fields rather than averaged as complete probabilities.  Every
coefficient is selected and fitted only on seasons strictly before the outer
validation season.  The 2021 EXP-071 field is exactly zero because that
mechanism requires saved EXP-051 OOF residuals beginning in 2021 and therefore
has no admissible prior residual season for a 2021 prediction.

No competition test row and no Public score is read by this program.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from train_exp017_rolling_residual import calculate_metrics
from train_exp041_exact_game_trackman_sequence import mapping_from_aligned
from train_exp064_invariant_uncertainty_group_eb import (
    SMOOTHING as EXP064_SMOOTHING,
    row_keys as exp064_row_keys,
    season_map as exp064_season_map,
)
from train_exp066_partial_sequence_alignment_control import partial_aligned_rows
from train_exp070_partial_player_physics_integration import (
    CONTEXT_SMOOTHING,
    load_trackman,
)
from train_exp072_dynamic_pitcher_state import (
    STATE_SMOOTHING,
    latest_latent_before,
    prior_career_states,
    season_latent_states,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
ARTIFACT_ROOT = ROOT / "artifacts"
OUTPUT_ROOT = ARTIFACT_ROOT / "EXP-105" / "correction_geometry"
REPORT_PATH = OUTPUT_ROOT / "report.json"

SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
COMPONENTS = ("c063", "c064", "c071", "c072")
RIDGE_GRID = (1.0, 10.0, 100.0, 1000.0)
SVD_RANKS = (1, 2, 3, 4)
FALLBACK_LAMBDA = 100.0
FALLBACK_RANK = 2
COEFFICIENT_SUM_BOUND = 2.0
TOLERANCE = 1e-12


OOF_PATHS = {
    "p063": ARTIFACT_ROOT
    / "EXP-063/uncertain_region_residual/predictions_close060_last_w025_{season}.npy",
    "base063": ARTIFACT_ROOT
    / "EXP-063/uncertain_region_residual/predictions_base_{season}.npy",
    "p064": ARTIFACT_ROOT
    / "EXP-064/invariant_uncertainty_group_eb/predictions_stable_count_runners_pbin_w050_{season}.npy",
    "base064": ARTIFACT_ROOT
    / "EXP-064/invariant_uncertainty_group_eb/predictions_base_{season}.npy",
    "p071": ARTIFACT_ROOT
    / "EXP-071/partial_player_physics_residual/predictions_playerphys_resid_w025_{season}.npy",
    "base071": ARTIFACT_ROOT
    / "EXP-071/partial_player_physics_residual/predictions_base_{season}.npy",
    "p072": ARTIFACT_ROOT
    / "EXP-072/dynamic_pitcher_state/predictions_ar_k30_w050_{season}.npy",
    "base072": ARTIFACT_ROOT
    / "EXP-072/dynamic_pitcher_state/predictions_base_exp051_{season}.npy",
    "target": ARTIFACT_ROOT
    / "EXP-072/dynamic_pitcher_state/targets_{season}.npy",
}


@dataclass(frozen=True)
class SeasonBundle:
    season: int
    rows: pd.DataFrame
    target: np.ndarray
    p0: np.ndarray
    corrections: np.ndarray
    reliability: np.ndarray
    fourier: np.ndarray


def _load(name: str, season: int) -> np.ndarray:
    return np.load(Path(str(OOF_PATHS[name]).format(season=season))).astype(float)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def season_equal_weights(season_ids: np.ndarray) -> np.ndarray:
    values, counts = np.unique(season_ids, return_counts=True)
    lookup = {int(value): int(count) for value, count in zip(values, counts, strict=True)}
    weights = np.asarray([1.0 / lookup[int(value)] for value in season_ids])
    weights *= len(weights) / weights.sum()
    return weights


def _exp064_reliability(
    season: int,
    rows_by_season: dict[int, pd.DataFrame],
    p0_by_season: dict[int, np.ndarray],
    targets_by_season: dict[int, np.ndarray],
) -> np.ndarray:
    sources = [value for value in SEASONS if value < season]
    if not sources:
        return np.zeros(len(rows_by_season[season]), dtype=float)
    columns = ("count", "runners", "pbin")
    source_maps = []
    for source in sources:
        keys = exp064_row_keys(rows_by_season[source], p0_by_season[source])
        residual = targets_by_season[source] - p0_by_season[source]
        source_maps.append(exp064_season_map(keys, residual, columns))
    union = source_maps[0].index
    for source_map in source_maps[1:]:
        union = union.union(source_map.index)
    effects = np.column_stack(
        [source_map["effect"].reindex(union).fillna(0.0).to_numpy() for source_map in source_maps]
    )
    counts = np.column_stack(
        [source_map["count"].reindex(union).fillna(0.0).to_numpy() for source_map in source_maps]
    )
    nonzero = counts > 0
    stable = nonzero.all(axis=1) & (
        (effects > 0).all(axis=1) | (effects < 0).all(axis=1)
    )
    total = counts.sum(axis=1)
    reliability = np.where(
        stable,
        total / (total + EXP064_SMOOTHING * len(sources)),
        0.0,
    )
    query_keys = exp064_row_keys(rows_by_season[season], p0_by_season[season])
    query = pd.MultiIndex.from_frame(query_keys.loc[:, list(columns)])
    return (
        pd.Series(reliability, index=union)
        .reindex(query)
        .fillna(0.0)
        .to_numpy(float)
    )


def _exp071_reliability(
    season: int,
    rows: pd.DataFrame,
    trackman: pd.DataFrame,
    aligned: pd.DataFrame,
) -> np.ndarray:
    if season == 2021:
        return np.zeros(len(rows), dtype=float)
    cutoff = season - 1
    mapping, _ = mapping_from_aligned(aligned, cutoff)
    history = trackman.loc[trackman["season"].le(cutoff)]
    overall = history.groupby("pitcher_trackman_id").size()
    context = history.groupby(
        ["pitcher_trackman_id", "count_index", "batter_hand_code"]
    ).size()
    mapped = rows["pitcher_id"].map(mapping.mapping)
    count_index = 4 * rows["balls_before"] + rows["strikes_before"]
    query = pd.MultiIndex.from_arrays(
        [mapped, count_index, rows["batter_hand"]],
        names=["pitcher_trackman_id", "count_index", "batter_hand_code"],
    )
    support = context.reindex(query).to_numpy(float)
    fallback = overall.reindex(mapped).to_numpy(float)
    support = np.where(np.isfinite(support), support, fallback)
    support = np.where(np.isfinite(support), support, 0.0)
    return support / (support + CONTEXT_SMOOTHING)


def _exp072_reliability(
    season: int,
    rows: pd.DataFrame,
    states: pd.DataFrame,
    careers: dict[int, object],
) -> np.ndarray:
    latest = latest_latent_before(states, season)
    pitcher_ids = rows["pitcher_id"]
    last_count = pitcher_ids.map(latest["count"]).fillna(0.0).to_numpy(float)
    career = careers[season]
    prior_n = pitcher_ids.map(career.n).fillna(0.0).to_numpy(float)
    current_n = np.maximum(rows["asof_pitcher_n"].to_numpy(float) - prior_n, 0.0)
    historical = last_count / (last_count + STATE_SMOOTHING)
    current = 30.0 / (current_n + 30.0)
    return historical * current


def fourier_features(rows: pd.DataFrame) -> np.ndarray:
    """Return the only row-local calendar harmonics supported by official data.

    The competition table has no exact date/timestamp.  These proxies are kept
    for the EXP-111 feasibility audit, but are not executed as a candidate:
    using them would reduce the paper's continuous timestamp embedding to a
    smooth reparameterization of EXP-057's month/weekday calendar cells.
    """

    month = rows["game_month"].to_numpy(float)
    weekday = rows["game_dayofweek"].to_numpy(float)
    trend = rows["season"].to_numpy(float) + (month - 0.5) / 12.0
    annual = 2.0 * np.pi * (month - 0.5) / 12.0
    weekly = 2.0 * np.pi * weekday / 7.0
    return np.column_stack(
        [trend, np.sin(annual), np.cos(annual), np.sin(weekly), np.cos(weekly)]
    )


def load_bundles() -> tuple[dict[int, SeasonBundle], dict[str, object]]:
    raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    rows_by_season = {
        season: raw.loc[raw["season"].eq(season)].reset_index(drop=True)
        for season in SEASONS
    }
    p0_by_season: dict[int, np.ndarray] = {}
    targets_by_season: dict[int, np.ndarray] = {}
    integrity: dict[str, object] = {"base_max_abs_diff": {}, "target_exact": {}}
    for season in SEASONS:
        p0 = _load("base072", season)
        target = _load("target", season)
        p0_by_season[season] = p0
        targets_by_season[season] = target
        bases = [_load("base063", season), _load("base064", season)]
        if season >= 2022:
            bases.append(_load("base071", season))
        diffs = [float(np.max(np.abs(base - p0))) for base in bases]
        integrity["base_max_abs_diff"][str(season)] = max(diffs)
        row_target = rows_by_season[season]["control_success"].to_numpy(float)
        integrity["target_exact"][str(season)] = bool(np.array_equal(target, row_target))
        if max(diffs) > TOLERANCE or not np.array_equal(target, row_target):
            raise ValueError(f"OOF alignment failure in {season}")

    trackman = load_trackman()
    aligned, _ = partial_aligned_rows()
    states, _ = season_latent_states(raw)
    careers = prior_career_states(raw)
    bundles: dict[int, SeasonBundle] = {}
    for season in SEASONS:
        p0 = p0_by_season[season]
        c063 = _load("p063", season) - p0
        c064 = _load("p064", season) - p0
        c071 = (
            _load("p071", season) - p0
            if season >= 2022
            else np.zeros(len(p0), dtype=float)
        )
        c072 = _load("p072", season) - p0
        corrections = np.column_stack([c063, c064, c071, c072])
        rows = rows_by_season[season]
        rel063 = (np.abs(p0 - 0.5) < 0.06).astype(float)
        rel064 = _exp064_reliability(
            season, rows_by_season, p0_by_season, targets_by_season
        )
        rel071 = _exp071_reliability(season, rows, trackman, aligned)
        rel072 = _exp072_reliability(season, rows, states, careers)
        reliability = np.column_stack([rel063, rel064, rel071, rel072])
        bundles[season] = SeasonBundle(
            season=season,
            rows=rows,
            target=targets_by_season[season],
            p0=p0,
            corrections=corrections,
            reliability=reliability,
            fourier=fourier_features(rows),
        )
    integrity["exp071_2021_policy"] = (
        "zero correction: no prior admissible EXP-051 OOF residual season"
    )
    return bundles, integrity


def concatenate(
    bundles: dict[int, SeasonBundle],
    seasons: Iterable[int],
    matrix: Callable[[SeasonBundle], np.ndarray],
    target: Callable[[SeasonBundle], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    season_list = list(seasons)
    x = np.concatenate([matrix(bundles[value]) for value in season_list])
    y = np.concatenate([target(bundles[value]) for value in season_list])
    ids = np.concatenate(
        [np.full(len(bundles[value].target), value, dtype=int) for value in season_list]
    )
    return x, y, season_equal_weights(ids)


def ridge_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    gram = x.T @ (weights[:, None] * x)
    rhs = x.T @ (weights * y)
    return np.linalg.solve(gram + alpha * np.eye(x.shape[1]), rhs)


def constrained_fit(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float
) -> np.ndarray:
    weighted_x = np.sqrt(weights)[:, None] * x
    weighted_y = np.sqrt(weights) * y
    augmented_x = np.vstack([weighted_x, math.sqrt(alpha) * np.eye(x.shape[1])])
    augmented_y = np.concatenate([weighted_y, np.zeros(x.shape[1])])
    nonnegative, _ = nnls(augmented_x, augmented_y)
    if nonnegative.sum() <= COEFFICIENT_SUM_BOUND + 1e-12:
        return nonnegative

    def objective(coef: np.ndarray) -> float:
        error = y - x @ coef
        return float(np.dot(weights, error * error) + alpha * np.dot(coef, coef))

    def gradient(coef: np.ndarray) -> np.ndarray:
        return -2.0 * x.T @ (weights * (y - x @ coef)) + 2.0 * alpha * coef

    initial = nonnegative * (COEFFICIENT_SUM_BOUND / nonnegative.sum())
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, COEFFICIENT_SUM_BOUND)] * x.shape[1],
        constraints=[{"type": "ineq", "fun": lambda coef: COEFFICIENT_SUM_BOUND - coef.sum()}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    feasible = (
        np.isfinite(result.x).all()
        and np.min(result.x) >= -1e-10
        and result.x.sum() <= COEFFICIENT_SUM_BOUND + 1e-9
    )
    if not result.success and not feasible:
        raise RuntimeError(f"constrained correction fit failed: {result.message}")
    return np.clip(np.asarray(result.x, dtype=float), 0.0, None)


def choose_lambda(
    bundles: dict[int, SeasonBundle],
    history: list[int],
    matrix: Callable[[SeasonBundle], np.ndarray],
    target: Callable[[SeasonBundle], np.ndarray],
    fitter: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
) -> tuple[float, dict[str, float]]:
    if len(history) < 2:
        return FALLBACK_LAMBDA, {"fallback": FALLBACK_LAMBDA}
    scores: dict[str, float] = {}
    for alpha in RIDGE_GRID:
        fold_scores = []
        for heldout in history:
            train_seasons = [value for value in history if value != heldout]
            x, y, weights = concatenate(bundles, train_seasons, matrix, target)
            coef = fitter(x, y, weights, alpha)
            valid_x = matrix(bundles[heldout])
            valid_y = target(bundles[heldout])
            fold_scores.append(float(np.mean((valid_y - valid_x @ coef) ** 2)))
        scores[str(int(alpha))] = float(np.mean(fold_scores))
    selected = min(RIDGE_GRID, key=lambda value: (scores[str(int(value))], -value))
    return selected, scores


def evaluate_linear_candidate(
    bundles: dict[int, SeasonBundle],
    *,
    experiment: str,
    matrix: Callable[[SeasonBundle], np.ndarray],
    fitter: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
) -> dict[str, object]:
    folds: dict[str, object] = {}
    artifact_dir = ARTIFACT_ROOT / experiment / "correction_stack"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for season in REPORT_SEASONS:
        history = [value for value in SEASONS if value < season]
        alpha, cv = choose_lambda(
            bundles,
            history,
            matrix,
            lambda bundle: bundle.target - bundle.p0,
            fitter,
        )
        x, y, weights = concatenate(
            bundles,
            history,
            matrix,
            lambda bundle: bundle.target - bundle.p0,
        )
        coef = fitter(x, y, weights, alpha)
        prediction = np.clip(
            bundles[season].p0 + matrix(bundles[season]) @ coef, 0.0, 1.0
        )
        np.save(artifact_dir / f"predictions_{season}.npy", prediction)
        folds[str(season)] = {
            "history": history,
            "lambda": alpha,
            "inner_loso_brier": cv,
            "coefficient": coef.tolist(),
            "metrics": calculate_metrics(bundles[season].target, prediction),
        }
    alpha, cv = choose_lambda(
        bundles,
        list(SEASONS),
        matrix,
        lambda bundle: bundle.target - bundle.p0,
        fitter,
    )
    x, y, weights = concatenate(
        bundles,
        SEASONS,
        matrix,
        lambda bundle: bundle.target - bundle.p0,
    )
    coef = fitter(x, y, weights, alpha)
    return {
        "experiment": experiment,
        "folds": folds,
        "final_2025": {
            "history": list(SEASONS),
            "lambda": alpha,
            "inner_loso_brier": cv,
            "coefficient": coef.tolist(),
        },
    }


def _svd_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    rank: int,
) -> dict[str, np.ndarray | float | int]:
    mean = np.average(x, axis=0, weights=weights)
    centered = x - mean
    covariance = centered.T @ (weights[:, None] * centered)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    actual_rank = min(rank, int(np.sum(eigenvalues > 1e-18)))
    basis = vectors[:, :actual_rank]
    z = centered @ basis
    coef = ridge_fit(z, y, weights, alpha)
    return {
        "mean": mean,
        "basis": basis,
        "coefficient": coef,
        "singular_values": np.sqrt(np.maximum(eigenvalues, 0.0)),
        "rank": actual_rank,
        "lambda": alpha,
    }


def _svd_predict(model: dict[str, object], x: np.ndarray) -> np.ndarray:
    return (x - model["mean"]) @ model["basis"] @ model["coefficient"]


def evaluate_svd_candidate(bundles: dict[int, SeasonBundle]) -> dict[str, object]:
    artifact_dir = ARTIFACT_ROOT / "EXP-109" / "orthogonal_correction_basis"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for season in REPORT_SEASONS:
        history = [value for value in SEASONS if value < season]
        if len(history) < 2:
            selected = (FALLBACK_RANK, FALLBACK_LAMBDA)
            scores = {"fallback": [FALLBACK_RANK, FALLBACK_LAMBDA]}
        else:
            scores = {}
            for rank in SVD_RANKS:
                for alpha in RIDGE_GRID:
                    values = []
                    for heldout in history:
                        train = [value for value in history if value != heldout]
                        x, y, weights = concatenate(
                            bundles,
                            train,
                            lambda bundle: bundle.corrections,
                            lambda bundle: bundle.target - bundle.p0,
                        )
                        model = _svd_fit(x, y, weights, alpha, rank)
                        pred = _svd_predict(model, bundles[heldout].corrections)
                        values.append(
                            float(np.mean((bundles[heldout].target - bundles[heldout].p0 - pred) ** 2))
                        )
                    scores[f"r{rank}_l{int(alpha)}"] = float(np.mean(values))
            selected = min(
                ((rank, alpha) for rank in SVD_RANKS for alpha in RIDGE_GRID),
                key=lambda pair: (scores[f"r{pair[0]}_l{int(pair[1])}"], pair[0], -pair[1]),
            )
        x, y, weights = concatenate(
            bundles,
            history,
            lambda bundle: bundle.corrections,
            lambda bundle: bundle.target - bundle.p0,
        )
        model = _svd_fit(x, y, weights, selected[1], selected[0])
        correction = _svd_predict(model, bundles[season].corrections)
        prediction = np.clip(bundles[season].p0 + correction, 0.0, 1.0)
        np.save(artifact_dir / f"predictions_{season}.npy", prediction)
        folds[str(season)] = {
            "history": history,
            "selected_rank": int(model["rank"]),
            "lambda": selected[1],
            "inner_loso_brier": scores,
            "mean": model["mean"].tolist(),
            "basis": model["basis"].tolist(),
            "basis_coefficient": model["coefficient"].tolist(),
            "singular_values": model["singular_values"].tolist(),
            "metrics": calculate_metrics(bundles[season].target, prediction),
        }
    # Final 2025 selection uses the same inner LOSO rule on all historical OOF.
    scores = {}
    for rank in SVD_RANKS:
        for alpha in RIDGE_GRID:
            values = []
            for heldout in SEASONS:
                train = [value for value in SEASONS if value != heldout]
                x, y, weights = concatenate(
                    bundles,
                    train,
                    lambda bundle: bundle.corrections,
                    lambda bundle: bundle.target - bundle.p0,
                )
                model = _svd_fit(x, y, weights, alpha, rank)
                pred = _svd_predict(model, bundles[heldout].corrections)
                values.append(
                    float(np.mean((bundles[heldout].target - bundles[heldout].p0 - pred) ** 2))
                )
            scores[f"r{rank}_l{int(alpha)}"] = float(np.mean(values))
    selected = min(
        ((rank, alpha) for rank in SVD_RANKS for alpha in RIDGE_GRID),
        key=lambda pair: (scores[f"r{pair[0]}_l{int(pair[1])}"], pair[0], -pair[1]),
    )
    x, y, weights = concatenate(
        bundles,
        SEASONS,
        lambda bundle: bundle.corrections,
        lambda bundle: bundle.target - bundle.p0,
    )
    final = _svd_fit(x, y, weights, selected[1], selected[0])
    return {
        "experiment": "EXP-109",
        "folds": folds,
        "final_2025": {
            "history": list(SEASONS),
            "selected_rank": int(final["rank"]),
            "lambda": selected[1],
            "inner_loso_brier": scores,
            "mean": final["mean"].tolist(),
            "basis": final["basis"].tolist(),
            "basis_coefficient": final["coefficient"].tolist(),
            "singular_values": final["singular_values"].tolist(),
        },
    }


def rule_auxiliary(bundle: SeasonBundle) -> tuple[np.ndarray, np.ndarray]:
    c063, c064, c071, c072 = bundle.corrections.T
    physical_available = bundle.reliability[:, 2] > 0.0
    auxiliary_fields = np.column_stack(
        [c063, c064, np.where(physical_available, 0.0, c072)]
    )
    active = np.sum(np.abs(auxiliary_fields) > 0.0, axis=1)
    auxiliary = auxiliary_fields.sum(axis=1) / np.maximum(active, 1)
    return c071, auxiliary


def _rule_alpha(bundles: dict[int, SeasonBundle], seasons: list[int]) -> float:
    aux_parts = []
    target_parts = []
    ids = []
    for season in seasons:
        c071, auxiliary = rule_auxiliary(bundles[season])
        aux_parts.append(auxiliary)
        target_parts.append(bundles[season].target - bundles[season].p0 - c071)
        ids.append(np.full(len(auxiliary), season, dtype=int))
    auxiliary = np.concatenate(aux_parts)
    target = np.concatenate(target_parts)
    weights = season_equal_weights(np.concatenate(ids))
    denominator = float(np.dot(weights, auxiliary * auxiliary))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(weights, auxiliary * target) / denominator, 0.0, 1.0))


def evaluate_rule_candidate(bundles: dict[int, SeasonBundle]) -> dict[str, object]:
    artifact_dir = ARTIFACT_ROOT / "EXP-110" / "mechanism_rule_composition"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for season in REPORT_SEASONS:
        history = [value for value in SEASONS if value < season]
        alpha = _rule_alpha(bundles, history)
        c071, auxiliary = rule_auxiliary(bundles[season])
        prediction = np.clip(bundles[season].p0 + c071 + alpha * auxiliary, 0.0, 1.0)
        np.save(artifact_dir / f"predictions_{season}.npy", prediction)
        folds[str(season)] = {
            "history": history,
            "auxiliary_shrinkage": alpha,
            "metrics": calculate_metrics(bundles[season].target, prediction),
        }
    return {
        "experiment": "EXP-110",
        "formula": (
            "p051 + c071 + alpha*mean(active c063, c064, "
            "c072 only where physical lookup unavailable)"
        ),
        "folds": folds,
        "final_2025": {
            "history": list(SEASONS),
            "auxiliary_shrinkage": _rule_alpha(bundles, list(SEASONS)),
        },
    }


def _fourier_matrix(bundle: SeasonBundle) -> np.ndarray:
    return bundle.fourier


def evaluate_fourier_candidate(bundles: dict[int, SeasonBundle]) -> dict[str, object]:
    # Ridge is fit against the EXP-071 residual, not EXP-051 residual.
    artifact_dir = ARTIFACT_ROOT / "EXP-111" / "fixed_fourier_temporal_control"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for season in REPORT_SEASONS:
        history = [value for value in SEASONS if value < season]
        target_fn = lambda bundle: bundle.target - bundle.p0 - bundle.corrections[:, 2]
        alpha, cv = choose_lambda(
            bundles, history, _fourier_matrix, target_fn, ridge_fit
        )
        x, y, weights = concatenate(bundles, history, _fourier_matrix, target_fn)
        mean = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
        scale = np.where(scale > 1e-12, scale, 1.0)
        coef = ridge_fit((x - mean) / scale, y, weights, alpha)
        valid_x = (bundles[season].fourier - mean) / scale
        prediction = np.clip(
            bundles[season].p0 + bundles[season].corrections[:, 2] + valid_x @ coef,
            0.0,
            1.0,
        )
        np.save(artifact_dir / f"predictions_{season}.npy", prediction)
        folds[str(season)] = {
            "history": history,
            "lambda": alpha,
            "inner_loso_brier_unstandardized_screen": cv,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "coefficient": coef.tolist(),
            "metrics": calculate_metrics(bundles[season].target, prediction),
        }
    # Final fit: select alpha with the same fixed raw-feature screen, then fit standardized.
    target_fn = lambda bundle: bundle.target - bundle.p0 - bundle.corrections[:, 2]
    alpha, cv = choose_lambda(
        bundles, list(SEASONS), _fourier_matrix, target_fn, ridge_fit
    )
    x, y, weights = concatenate(bundles, SEASONS, _fourier_matrix, target_fn)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.0)
    coef = ridge_fit((x - mean) / scale, y, weights, alpha)
    return {
        "experiment": "EXP-111",
        "period_days": [365.25, 30.4375, 7.0],
        "trend": "UTC day number, train-standardized",
        "period_search": False,
        "folds": folds,
        "final_2025": {
            "history": list(SEASONS),
            "lambda": alpha,
            "inner_loso_brier_unstandardized_screen": cv,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "coefficient": coef.tolist(),
        },
    }


def correction_geometry(bundles: dict[int, SeasonBundle]) -> dict[str, object]:
    pooled_c = np.concatenate([bundles[season].corrections for season in SEASONS])
    pooled_r = np.concatenate(
        [bundles[season].target - bundles[season].p0 for season in SEASONS]
    )
    pooled_complete_error = np.concatenate(
        [
            bundles[season].p0[:, None]
            + bundles[season].corrections
            - bundles[season].target[:, None]
            for season in SEASONS
        ]
    )
    pairwise = {}
    complete_error_pairwise = {}
    for left_index, left in enumerate(COMPONENTS):
        for right_index in range(left_index + 1, len(COMPONENTS)):
            right = COMPONENTS[right_index]
            pairwise[f"{left}__{right}"] = pearson(
                pooled_c[:, left_index], pooled_c[:, right_index]
            )
            complete_error_pairwise[f"{left}__{right}"] = pearson(
                pooled_complete_error[:, left_index],
                pooled_complete_error[:, right_index],
            )
    residual = {
        name: {
            "correlation": pearson(pooled_c[:, index], pooled_r),
            "covariance": float(np.cov(pooled_c[:, index], pooled_r, ddof=0)[0, 1]),
            "mean_abs": float(np.mean(np.abs(pooled_c[:, index]))),
            "nonzero_rate": float(np.mean(pooled_c[:, index] != 0.0)),
        }
        for index, name in enumerate(COMPONENTS)
    }
    segments: dict[str, object] = {}
    for season, bundle in bundles.items():
        rows = bundle.rows
        definitions = {
            "uncertainty": pd.cut(
                np.abs(bundle.p0 - 0.5),
                [-np.inf, 0.02, 0.04, 0.06, np.inf],
                labels=["lt020", "020_040", "040_060", "ge060"],
            ).astype(str),
            "pitcher_history_n": pd.cut(
                rows["asof_pitcher_n"],
                [-np.inf, 0, 19, 99, 499, np.inf],
                labels=["0", "1_19", "20_99", "100_499", "500_plus"],
            ).astype(str),
            "batter_history_n": pd.cut(
                rows["asof_batter_n"],
                [-np.inf, 0, 19, 99, 499, np.inf],
                labels=["0", "1_19", "20_99", "100_499", "500_plus"],
            ).astype(str),
            "trackman_available": np.where(bundle.reliability[:, 2] > 0, "yes", "no"),
            "physical_reliability": pd.cut(
                bundle.reliability[:, 2],
                [-np.inf, 0, 0.25, 0.5, 0.75, np.inf],
                labels=["0", "0_025", "025_050", "050_075", "075_1"],
            ).astype(str),
            "ar_reliability": pd.cut(
                bundle.reliability[:, 3],
                [-np.inf, 0, 0.1, 0.25, 0.5, np.inf],
                labels=["0", "0_010", "010_025", "025_050", "050_1"],
            ).astype(str),
            "count": (
                4 * rows["balls_before"] + rows["strikes_before"]
            ).astype(str).to_numpy(),
            "runners": rows["num_runners_on"].astype(str).to_numpy(),
            "game_type": rows["game_type"].astype(str).to_numpy(),
        }
        season_segments = {}
        r0 = bundle.target - bundle.p0
        for segment_name, labels in definitions.items():
            groups = {}
            for label in sorted(pd.unique(labels)):
                mask = np.asarray(labels) == label
                if mask.sum() < 2:
                    continue
                groups[str(label)] = {
                    "rows": int(mask.sum()),
                    "residual_correlation": {
                        name: pearson(bundle.corrections[mask, index], r0[mask])
                        for index, name in enumerate(COMPONENTS)
                    },
                    "mean_abs_correction": {
                        name: float(np.mean(np.abs(bundle.corrections[mask, index])))
                        for index, name in enumerate(COMPONENTS)
                    },
                }
            season_segments[segment_name] = groups
        segments[str(season)] = season_segments
    return {
        "components": list(COMPONENTS),
        "pooled_2021_2024_pairwise_correlation": pairwise,
        "pooled_2021_2024_complete_prediction_error_correlation": complete_error_pairwise,
        "pooled_2021_2024_residual_relation": residual,
        "segments": segments,
    }


def exp071_metrics(bundles: dict[int, SeasonBundle]) -> dict[str, object]:
    return {
        str(season): calculate_metrics(
            bundles[season].target,
            np.clip(bundles[season].p0 + bundles[season].corrections[:, 2], 0, 1),
        )
        for season in REPORT_SEASONS
    }


def classify_candidates(result: dict[str, object], reference: dict[str, object]) -> str:
    deltas = []
    for season in (2023, 2024):
        candidate = float(result["folds"][str(season)]["metrics"]["brier_score"])
        baseline = float(reference[str(season)]["brier_score"])
        deltas.append(candidate - baseline)
    pooled = float(np.mean(deltas))
    if all(value < 0 for value in deltas) and pooled < 0:
        return "A"
    if min(deltas) < 0 and max(deltas) <= 2e-5 and pooled <= 1e-5:
        return "B"
    return "C"


def write_result(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    started = time.time()
    bundles, integrity = load_bundles()
    geometry = correction_geometry(bundles)
    reference = exp071_metrics(bundles)
    results = {
        "EXP-106": evaluate_linear_candidate(
            bundles,
            experiment="EXP-106",
            matrix=lambda bundle: bundle.corrections,
            fitter=ridge_fit,
        ),
        "EXP-107": evaluate_linear_candidate(
            bundles,
            experiment="EXP-107",
            matrix=lambda bundle: bundle.corrections,
            fitter=constrained_fit,
        ),
        "EXP-108": evaluate_linear_candidate(
            bundles,
            experiment="EXP-108",
            matrix=lambda bundle: bundle.corrections * bundle.reliability,
            fitter=ridge_fit,
        ),
        "EXP-109": evaluate_svd_candidate(bundles),
        "EXP-110": evaluate_rule_candidate(bundles),
    }
    for name, result in results.items():
        result["candidate_tier"] = classify_candidates(result, reference)
        write_result(
            ARTIFACT_ROOT / name / result["experiment"].lower().replace("-", "_") / "report.json",
            result,
        )
    output = {
        "experiment": "EXP-105",
        "objective": "correction geometry and strict historical correction composition",
        "integrity": integrity,
        "exp071_reference": reference,
        "geometry": geometry,
        "results": results,
        "EXP-111": {
            "status": "skipped_after_feasibility_audit",
            "reason": (
                "Official rows expose season, month, and weekday but no exact "
                "timestamp. The ICML method consumes a continuous timestamp "
                "with trend plus year/month/week/day Fourier periods. Under the "
                "mandatory row-independence rule, row order cannot reconstruct "
                "the missing date. Available month/weekday harmonics would be a "
                "smooth reparameterization of EXP-057 rather than a faithful "
                "new temporal-shift control."
            ),
            "training_lag": (
                "strict rolling components use source seasons ending at least "
                "one season before each validation season; exact within-season "
                "lag is not identifiable from the released row schema"
            ),
            "official_method_period_seconds": [31557600.0, 2629800.0, 604800.0, 86400.0],
            "executed": False,
        },
        "protocol": {
            "public_score_used_for_numeric_fit": False,
            "outer_validation_labels_used_for_fit": False,
            "intercept": 0,
            "ridge_grid": list(RIDGE_GRID),
            "ridge_objective": "season-equal weighted SSE + lambda*L2",
            "exp107_nonnegative": True,
            "exp107_sum_bound": COEFFICIENT_SUM_BOUND,
            "exp111_source": "Cai and Ye, ICML 2025; fixed calendar periods, no period search",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "runtime_seconds": time.time() - started,
    }
    write_result(REPORT_PATH, output)
    print(json.dumps({
        name: {
            "tier": value["candidate_tier"],
            "final": value["final_2025"],
            "brier": {
                season: value["folds"][season]["metrics"]["brier_score"]
                for season in map(str, REPORT_SEASONS)
            },
        }
        for name, value in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
