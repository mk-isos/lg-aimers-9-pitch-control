"""EXP-072: temporal-safe dynamic pitcher latent-state correction.

The official ``asof_pitcher_*`` fields are career cumulative.  For each row,
the current-season sample and success count are reconstructed by subtracting
the frozen end-of-prior-season state learned from ``train.csv``.  A pitcher's
latest prior-season success state is represented as a league-centered,
empirical-Bayes log-odds effect.  Its year-to-year persistence is estimated
only from seasons strictly before the prediction season with a bounded AR(1)
transition.  The forecast state becomes the prior for the current row and is
automatically down-weighted as current-season observations accumulate.

The correction is evaluated on the frozen EXP-051 OOF prediction.  No
validation/test-row aggregation, current-fold fitting, or current-fold
calibration is used.  Same-fold one-dimensional oracle values are emitted as
diagnostic ceilings only and are never candidates for deployment.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics
from train_exp066_partial_sequence_alignment_control import base_components


EXPERIMENT = "EXP-072"
DATA_PATH = Path("./data/train.csv")
TARGET_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-072/dynamic_pitcher_state")

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
STATE_SMOOTHING = 200.0
TRANSITION_RIDGE = 1.0
EPS = 1e-6

# Fixed before looking at EXP-072 validation labels.
CANDIDATES = {
    "ar_k30_w025": ("ar", 30.0, 0.25),
    "ar_k30_w050": ("ar", 30.0, 0.50),
    "ar_k100_w025": ("ar", 100.0, 0.25),
    "last_k30_w025": ("last", 30.0, 0.25),
}


def expit(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logit(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def exp051_base(season: int) -> np.ndarray:
    recent, exact_correction = base_components(season)
    return np.clip(recent + 0.10 * exact_correction, 0.0, 1.0)


def load_frame() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_id",
        "batter_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "control_success",
    ]
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=columns)
    required = [column for column in columns if column != "asof_pitcher_success_rate"]
    if frame[required].isna().any().any():
        raise ValueError("missing required official field")
    missing_rate = frame["asof_pitcher_success_rate"].isna()
    if not frame.loc[missing_rate, "asof_pitcher_n"].eq(0).all():
        raise ValueError("pitcher success rate is missing at positive sample size")
    frame["asof_pitcher_success_rate"] = frame[
        "asof_pitcher_success_rate"
    ].fillna(0.0)
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("train rows must be season-monotone")
    return frame


def season_latent_states(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    stats = frame.groupby(["season", "pitcher_id"], sort=False)[
        "control_success"
    ].agg(["sum", "count"])
    league = frame.groupby("season", sort=False)["control_success"].mean()
    league_rates = {int(key): float(value) for key, value in league.items()}
    season_values = stats.index.get_level_values("season")
    priors = np.asarray([league_rates[int(value)] for value in season_values])
    posterior = (
        stats["sum"].to_numpy(float) + STATE_SMOOTHING * priors
    ) / (stats["count"].to_numpy(float) + STATE_SMOOTHING)
    states = stats.copy()
    states["league_rate"] = priors
    states["posterior_rate"] = posterior
    states["latent_logit"] = logit(posterior) - logit(priors)
    states["reliability"] = stats["count"].to_numpy(float) / (
        stats["count"].to_numpy(float) + STATE_SMOOTHING
    )
    return states, league_rates


def fit_transition(
    states: pd.DataFrame, prediction_season: int
) -> tuple[float, dict[str, float | int]]:
    history = states.loc[
        states.index.get_level_values("season") < prediction_season
    ].reset_index()
    previous = history.rename(
        columns={
            "season": "previous_season",
            "latent_logit": "previous_latent",
            "reliability": "previous_reliability",
        }
    )[
        [
            "pitcher_id",
            "previous_season",
            "previous_latent",
            "previous_reliability",
        ]
    ]
    current = history.rename(
        columns={
            "season": "current_season",
            "latent_logit": "current_latent",
            "reliability": "current_reliability",
        }
    )[
        [
            "pitcher_id",
            "current_season",
            "current_latent",
            "current_reliability",
        ]
    ]
    pairs = current.merge(previous, on="pitcher_id", how="inner")
    pairs = pairs.loc[
        pairs["current_season"].eq(pairs["previous_season"] + 1)
    ]
    if pairs.empty:
        rho = 0.0
        numerator = 0.0
        denominator = 0.0
    else:
        weights = np.sqrt(
            pairs["current_reliability"].to_numpy(float)
            * pairs["previous_reliability"].to_numpy(float)
        )
        x = pairs["previous_latent"].to_numpy(float)
        y = pairs["current_latent"].to_numpy(float)
        numerator = float(np.sum(weights * x * y))
        denominator = float(np.sum(weights * x * x))
        rho = float(
            np.clip(
                numerator / (denominator + TRANSITION_RIDGE),
                0.0,
                1.0,
            )
        )
    return rho, {
        "transition_pairs": int(len(pairs)),
        "weighted_cross_product": numerator,
        "weighted_square": denominator,
        "ridge": TRANSITION_RIDGE,
        "rho": rho,
    }


@dataclass(frozen=True)
class PriorCareerState:
    n: pd.Series
    successes: pd.Series


def end_state(season_rows: pd.DataFrame) -> PriorCareerState:
    end_indices = season_rows.groupby("pitcher_id", sort=False)[
        "asof_pitcher_n"
    ].idxmax()
    last = season_rows.loc[
        end_indices,
        [
            "pitcher_id",
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
            "control_success",
        ],
    ]
    n_before = last["asof_pitcher_n"].to_numpy(float)
    successes_before = np.rint(
        n_before * last["asof_pitcher_success_rate"].to_numpy(float)
    )
    return PriorCareerState(
        n=pd.Series(n_before + 1.0, index=last["pitcher_id"].to_numpy()),
        successes=pd.Series(
            successes_before + last["control_success"].to_numpy(float),
            index=last["pitcher_id"].to_numpy(),
        ),
    )


def prior_career_states(frame: pd.DataFrame) -> dict[int, PriorCareerState]:
    output: dict[int, PriorCareerState] = {}
    latest_n = pd.Series(dtype=float)
    latest_successes = pd.Series(dtype=float)
    for season in sorted(frame["season"].astype(int).unique()):
        output[int(season)] = PriorCareerState(
            n=latest_n.copy(), successes=latest_successes.copy()
        )
        state = end_state(frame.loc[frame["season"].eq(season)])
        latest_n = pd.concat([latest_n, state.n])
        latest_n = latest_n[~latest_n.index.duplicated(keep="last")]
        latest_successes = pd.concat([latest_successes, state.successes])
        latest_successes = latest_successes[
            ~latest_successes.index.duplicated(keep="last")
        ]
    return output


def latest_latent_before(
    states: pd.DataFrame, prediction_season: int
) -> pd.DataFrame:
    history = states.loc[
        states.index.get_level_values("season") < prediction_season
    ].reset_index()
    latest_index = history.groupby("pitcher_id", sort=False)["season"].idxmax()
    return history.loc[
        latest_index,
        ["pitcher_id", "season", "latent_logit", "count"],
    ].set_index("pitcher_id")


def dynamic_deltas(
    rows: pd.DataFrame,
    prediction_season: int,
    states: pd.DataFrame,
    league_rates: dict[int, float],
    career: PriorCareerState,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    rho, transition_audit = fit_transition(states, prediction_season)
    latest = latest_latent_before(states, prediction_season)
    pitcher_ids = rows["pitcher_id"]
    prior_n = pitcher_ids.map(career.n).fillna(0.0).to_numpy(float)
    prior_success = pitcher_ids.map(career.successes).fillna(0.0).to_numpy(float)
    career_n = rows["asof_pitcher_n"].to_numpy(float)
    career_success = np.rint(
        career_n * rows["asof_pitcher_success_rate"].to_numpy(float)
    )
    season_n = career_n - prior_n
    season_success = career_success - prior_success
    if np.any(season_n < -1e-6):
        raise ValueError(f"negative current-season n in {prediction_season}")
    if np.any(season_success < -0.01) or np.any(
        season_success - season_n > 0.01
    ):
        raise ValueError(f"invalid current-season success in {prediction_season}")
    season_n = np.maximum(season_n, 0.0)
    season_success = np.clip(season_success, 0.0, season_n)

    league_prior = float(league_rates[prediction_season - 1])
    last_latent = pitcher_ids.map(latest["latent_logit"]).fillna(0.0).to_numpy(float)
    last_season = pitcher_ids.map(latest["season"]).to_numpy(float)
    known = np.isfinite(last_season)
    gap = np.where(known, prediction_season - last_season, 0.0)
    ar_latent = np.where(known, last_latent * np.power(rho, gap), 0.0)
    last_probability = expit(logit(league_prior) + last_latent)
    ar_probability = expit(logit(league_prior) + ar_latent)

    deltas: dict[str, np.ndarray] = {}
    for method, probability in (
        ("ar", ar_probability),
        ("last", last_probability),
    ):
        for strength in (30.0, 100.0):
            dynamic = (season_success + strength * probability) / (
                season_n + strength
            )
            global_posterior = (
                season_success + strength * league_prior
            ) / (season_n + strength)
            deltas[f"{method}_k{int(strength)}"] = dynamic - global_posterior

    return deltas, {
        **transition_audit,
        "prediction_season": prediction_season,
        "league_prior": league_prior,
        "known_latest_state_rows": int(known.sum()),
        "known_latest_state_rate": float(known.mean()),
        "unique_known_pitchers": int(pitcher_ids.loc[known].nunique()),
        "mean_state_gap_known": float(gap[known].mean()) if known.any() else 0.0,
        "season_n_mean": float(season_n.mean()),
        "season_n_max": float(season_n.max()),
        "correction_mean_abs": {
            key: float(np.mean(np.abs(value))) for key, value in deltas.items()
        },
    }


def candidate_predictions(
    base: np.ndarray, deltas: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {"base_exp051": base}
    for name, (method, strength, weight) in CANDIDATES.items():
        delta = deltas[f"{method}_k{int(strength)}"]
        predictions[name] = np.clip(base + weight * delta, 0.0, 1.0)
    return predictions


def same_fold_oracle(
    target: np.ndarray, base: np.ndarray, delta: np.ndarray
) -> dict[str, float]:
    denominator = float(np.dot(delta, delta))
    unconstrained = (
        float(np.dot(delta, target - base)) / denominator
        if denominator > 0.0
        else 0.0
    )
    alpha = float(np.clip(unconstrained, -2.0, 2.0))
    prediction = np.clip(base + alpha * delta, 0.0, 1.0)
    return {
        "unconstrained_alpha": unconstrained,
        "bounded_alpha": alpha,
        **calculate_metrics(target, prediction),
    }


def segment_metrics(
    rows: pd.DataFrame,
    prediction: np.ndarray,
    target: np.ndarray,
    career: PriorCareerState,
) -> dict[str, object]:
    ids = rows["pitcher_id"]
    prior_n = ids.map(career.n).fillna(0.0).to_numpy(float)
    current_n = rows["asof_pitcher_n"].to_numpy(float) - prior_n
    known = ids.isin(career.n.index).to_numpy()
    masks = {
        "n_0": current_n == 0,
        "n_1_19": (current_n >= 1) & (current_n < 20),
        "n_20_99": (current_n >= 20) & (current_n < 100),
        "n_100_499": (current_n >= 100) & (current_n < 500),
        "n_500_plus": current_n >= 500,
        "pitcher_existing": known,
        "pitcher_new": ~known,
    }
    return {
        name: calculate_metrics(target[mask], prediction[mask])
        for name, mask in masks.items()
        if int(mask.sum()) > 0
    }


def choose_from_history(
    folds: dict[str, object], seasons: list[int]
) -> str:
    if not seasons:
        return "ar_k30_w025"
    return max(
        CANDIDATES,
        key=lambda name: (
            min(
                float(folds[str(season)][name]["skill_score_unclipped"])
                for season in seasons
            ),
            np.mean(
                [
                    float(folds[str(season)][name]["skill_score_unclipped"])
                    for season in seasons
                ]
            ),
            -CANDIDATES[name][2],
        ),
    )


def main() -> None:
    started = time.time()
    frame = load_frame()
    states, league_rates = season_latent_states(frame)
    career_states = prior_career_states(frame)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    folds: dict[str, object] = {}
    dynamic_cache: dict[int, dict[str, np.ndarray]] = {}
    target_cache: dict[int, np.ndarray] = {}
    base_cache: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        rows = frame.loc[frame["season"].eq(season)].reset_index(drop=True)
        target = np.load(TARGET_ROOT / f"targets_{season}.npy").astype(float)
        if not np.array_equal(target, rows["control_success"].to_numpy(float)):
            raise ValueError(f"target/order mismatch in {season}")
        base = exp051_base(season)
        if len(base) != len(target):
            raise ValueError(f"base length mismatch in {season}")
        deltas, audit = dynamic_deltas(
            rows,
            season,
            states,
            league_rates,
            career_states[season],
        )
        predictions = candidate_predictions(base, deltas)
        fold: dict[str, object] = {
            "validation_season": season,
            "history_cutoff": season - 1,
            "state_audit": audit,
            "same_fold_oracle_diagnostic_only": {
                name: same_fold_oracle(target, base, delta)
                for name, delta in deltas.items()
            },
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(target, prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{season}.npy", prediction)
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", target.astype(np.int8))
        folds[str(season)] = fold
        dynamic_cache[season] = predictions
        target_cache[season] = target
        base_cache[season] = base
        print(
            f"fold {season}: rho={audit['rho']:.4f} "
            + " ".join(
                f"{name}={fold[name]['skill_score_unclipped']:.2f}"
                for name in CANDIDATES
            ),
            flush=True,
        )

    strict_path: dict[str, object] = {}
    strict_skills: list[float] = []
    strict_briers: list[float] = []
    for season in REPORT_SEASONS:
        history = [value for value in EVALUATED_SEASONS if value < season]
        selected = choose_from_history(folds, history)
        metric = folds[str(season)][selected]
        strict_path[str(season)] = {
            "selected_using_seasons": history,
            "candidate": selected,
            "metrics": metric,
            "segments": segment_metrics(
                frame.loc[frame["season"].eq(season)].reset_index(drop=True),
                dynamic_cache[season][selected],
                target_cache[season],
                career_states[season],
            ),
        }
        strict_skills.append(float(metric["skill_score_unclipped"]))
        strict_briers.append(float(metric["brier_score"]))

    aggregate: dict[str, object] = {}
    for name in ("base_exp051", *CANDIDATES):
        skills = {
            str(season): float(folds[str(season)][name]["skill_score_unclipped"])
            for season in REPORT_SEASONS
        }
        briers = {
            str(season): float(folds[str(season)][name]["brier_score"])
            for season in REPORT_SEASONS
        }
        aggregate[name] = {
            "season_skills": skills,
            "season_briers": briers,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
        }

    next_candidate = choose_from_history(folds, list(EVALUATED_SEASONS))
    strict_mean = float(np.mean(strict_skills))
    strict_min = float(np.min(strict_skills))
    latest_oracle = max(
        folds["2024"]["same_fold_oracle_diagnostic_only"].values(),
        key=lambda value: value["skill_score_unclipped"],
    )
    result = {
        "experiment": EXPERIMENT,
        "candidate_family": "dynamic_pitcher_latent_AR1_current_row_posterior",
        "validation_protocol": {
            "evaluated_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "base": "EXP-051 OOF trackman_direct_recent_w010",
            "state_fit_seasons_strictly_prior": True,
            "transition_fit_seasons_strictly_prior": True,
            "current_row_official_asof_only": True,
            "current_fold_labels_used_for_fit_or_selection": False,
            "test_or_validation_row_aggregation": False,
            "candidate_grid_predeclared": True,
            "same_fold_oracle_is_diagnostic_only": True,
        },
        "configuration": {
            "state_smoothing": STATE_SMOOTHING,
            "transition_ridge": TRANSITION_RIDGE,
            "transition": "bounded zero-intercept AR(1), rho in [0,1]",
            "missing_or_new_pitcher_effect": 0.0,
            "gap_transition": "rho ** seasons_since_last_observed",
            "candidates": {
                name: {
                    "state_method": value[0],
                    "current_season_prior_strength": value[1],
                    "additive_delta_weight": value[2],
                }
                for name, value in CANDIDATES.items()
            },
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "strict_prior_fold_selection": {
            "path": strict_path,
            "mean_skill": strict_mean,
            "min_skill": strict_min,
            "mean_brier": float(np.mean(strict_briers)),
            "gate_each_season_1000": bool(strict_min >= 1000.0),
            "gate_mean_1100": bool(strict_mean >= 1100.0),
        },
        "prospective_2025_selection": {
            "candidate": next_candidate,
            "selected_using_seasons": list(EVALUATED_SEASONS),
            "uses_2025_labels": False,
        },
        "family_ceiling_diagnostic": {
            "best_same_fold_2024": latest_oracle,
            "latest_oracle_reaches_1000": bool(
                latest_oracle["skill_score_unclipped"] >= 1000.0
            ),
        },
        "selection": {
            "adopt": bool(strict_min >= 1000.0 and strict_mean >= 1100.0),
            "build_submission_zip": bool(
                strict_min >= 1000.0 and strict_mean >= 1100.0
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "runtime_seconds": float(time.time() - started),
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(
        f"strict mean={strict_mean:.2f} min={strict_min:.2f} "
        f"next={next_candidate} adopt={result['selection']['adopt']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
