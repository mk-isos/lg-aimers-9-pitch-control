"""EXP-074: prior-only career-stage residual model.

The model uses a player's frozen pre-season career trajectory rather than a
single cumulative rate: tenure, season gap, prior workload, regular-season
share, last/mean/three-season trend and volatility of EB latent control, and
whether the current official team differs from the last observed team.  The
only within-season quantity is reconstructed from the current row's official
career as-of count and the frozen prior-season endpoint.

Each outer fold fits a small Ridge model on earlier OOF seasons only.  Source
residuals are centered per season and source seasons receive equal total
weight.  There is no validation/test-row aggregation or current-fold model
selection.  A same-fold fit is saved only as a nondeployable family ceiling.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from train_exp017_rolling_residual import calculate_metrics
from train_exp072_dynamic_pitcher_state import (
    EVALUATED_SEASONS,
    REPORT_SEASONS,
    exp051_base,
    prior_career_states,
    same_fold_oracle,
    season_latent_states,
)


EXPERIMENT = "EXP-074"
DATA_PATH = Path("./data/train.csv")
TARGET_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-074/career_stage_residual")
CORRECTION_CLIP = 0.03

# Fixed before the run.  Alpha changes regularization only; the row features
# and source protocol stay identical.
CANDIDATES = {
    "career_a1000_w025": (1000.0, 0.25),
    "career_a10000_w025": (10000.0, 0.25),
    "career_a1000_w050": (1000.0, 0.50),
}

FEATURE_NAMES = (
    "prior_exists",
    "seasons_played",
    "years_since_debut",
    "seasons_since_last",
    "log_last_workload",
    "log_mean_workload",
    "last_latent",
    "mean_latent",
    "trend_latent_last3",
    "volatility_latent",
    "last_vs_mean_workload",
    "last_regular_share",
    "log_current_season_n",
    "current_reliability_30",
    "forecast_latent_trend",
    "team_changed",
)


def load_frame() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_id",
        "pitcher_team_id",
        "game_type",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "control_success",
    ]
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=columns)
    required = [column for column in columns if column != "asof_pitcher_success_rate"]
    if frame[required].isna().any().any():
        raise ValueError("missing required career-stage field")
    missing_rate = frame["asof_pitcher_success_rate"].isna()
    if not frame.loc[missing_rate, "asof_pitcher_n"].eq(0).all():
        raise ValueError("success rate missing at positive n")
    frame["asof_pitcher_success_rate"] = frame[
        "asof_pitcher_success_rate"
    ].fillna(0.0)
    if set(frame["game_type"].astype(str).unique()) != {"F", "R"}:
        raise ValueError("unexpected game_type")
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("train rows must be season-monotone")
    return frame


def season_summary(frame: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby(["season", "pitcher_id"], sort=False).agg(
        workload=("control_success", "size"),
        regular_share=("game_type", lambda value: float(value.astype(str).eq("R").mean())),
    )
    summary = summary.join(states[["latent_logit"]])
    end_indices = frame.groupby(["season", "pitcher_id"], sort=False)[
        "asof_pitcher_n"
    ].idxmax()
    last_team = frame.loc[
        end_indices, ["season", "pitcher_id", "pitcher_team_id"]
    ].set_index(["season", "pitcher_id"])["pitcher_team_id"]
    summary["last_team"] = last_team.reindex(summary.index).to_numpy()
    if summary.isna().any().any():
        raise ValueError("invalid season summary")
    return summary


def trajectory_table(summary: pd.DataFrame, prediction_season: int) -> pd.DataFrame:
    history = summary.loc[
        summary.index.get_level_values("season") < prediction_season
    ].reset_index()
    records: list[dict[str, float | int]] = []
    for pitcher_id, values in history.groupby("pitcher_id", sort=False):
        values = values.sort_values("season")
        seasons = values["season"].to_numpy(float)
        latent = values["latent_logit"].to_numpy(float)
        workload = values["workload"].to_numpy(float)
        recent_count = min(3, len(values))
        recent_seasons = seasons[-recent_count:]
        recent_latent = latent[-recent_count:]
        if recent_count >= 2 and np.ptp(recent_seasons) > 0:
            centered_x = recent_seasons - recent_seasons.mean()
            trend = float(
                np.dot(centered_x, recent_latent - recent_latent.mean())
                / np.dot(centered_x, centered_x)
            )
        else:
            trend = 0.0
        records.append(
            {
                "pitcher_id": int(pitcher_id),
                "seasons_played": float(len(values)),
                "debut_season": float(seasons[0]),
                "last_season": float(seasons[-1]),
                "last_workload": float(workload[-1]),
                "mean_workload": float(workload.mean()),
                "last_latent": float(latent[-1]),
                "mean_latent": float(latent.mean()),
                "trend_latent_last3": trend,
                "volatility_latent": float(latent.std(ddof=0)),
                "last_regular_share": float(values["regular_share"].iloc[-1]),
                "last_team": int(values["last_team"].iloc[-1]),
            }
        )
    if not records:
        return pd.DataFrame().set_index(pd.Index([], name="pitcher_id"))
    return pd.DataFrame.from_records(records).set_index("pitcher_id")


def feature_matrix(
    rows: pd.DataFrame,
    prediction_season: int,
    summary: pd.DataFrame,
    career_state: object,
) -> tuple[np.ndarray, dict[str, object]]:
    trajectory = trajectory_table(summary, prediction_season)
    ids = rows["pitcher_id"]
    known = ids.isin(trajectory.index).to_numpy()

    def mapped(column: str, default: float = 0.0) -> np.ndarray:
        if trajectory.empty:
            return np.full(len(rows), default, dtype=float)
        return ids.map(trajectory[column]).fillna(default).to_numpy(float)

    seasons_played = mapped("seasons_played")
    debut = mapped("debut_season", float(prediction_season))
    last_season = mapped("last_season", float(prediction_season))
    last_workload = mapped("last_workload")
    mean_workload = mapped("mean_workload")
    last_latent = mapped("last_latent")
    mean_latent = mapped("mean_latent")
    trend = mapped("trend_latent_last3")
    volatility = mapped("volatility_latent")
    last_regular_share = mapped("last_regular_share")
    last_team = mapped("last_team", -1.0)

    prior_n = ids.map(career_state.n).fillna(0.0).to_numpy(float)
    current_n = rows["asof_pitcher_n"].to_numpy(float) - prior_n
    if np.any(current_n < -1e-6):
        raise ValueError(f"negative current-season n in {prediction_season}")
    current_n = np.maximum(current_n, 0.0)
    gap = np.where(known, prediction_season - last_season, 0.0)
    workload_ratio = np.divide(
        last_workload - mean_workload,
        np.maximum(mean_workload, 1.0),
        out=np.zeros(len(rows), dtype=float),
        where=known,
    )
    current_team = rows["pitcher_team_id"].to_numpy(float)
    team_changed = known & (current_team != last_team)
    matrix = np.column_stack(
        [
            known.astype(float),
            seasons_played,
            np.where(known, prediction_season - debut, 0.0),
            gap,
            np.log1p(last_workload),
            np.log1p(mean_workload),
            last_latent,
            mean_latent,
            trend,
            volatility,
            workload_ratio,
            last_regular_share,
            np.log1p(current_n),
            current_n / (current_n + 30.0),
            last_latent + gap * trend,
            team_changed.astype(float),
        ]
    ).astype(np.float64)
    if matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
        raise ValueError("invalid career-stage feature matrix")
    return matrix, {
        "rows": int(len(rows)),
        "features": len(FEATURE_NAMES),
        "known_rows": int(known.sum()),
        "known_rate": float(known.mean()),
        "unique_known_pitchers": int(ids.loc[known].nunique()),
        "team_changed_rows": int(team_changed.sum()),
        "mean_current_season_n": float(current_n.mean()),
        "mean_seasons_played_known": (
            float(seasons_played[known].mean()) if known.any() else 0.0
        ),
        "mean_gap_known": float(gap[known].mean()) if known.any() else 0.0,
    }


def fit_correction(
    source_seasons: list[int],
    features: dict[int, np.ndarray],
    targets: dict[int, np.ndarray],
    bases: dict[int, np.ndarray],
    validation_features: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if not source_seasons:
        return np.zeros(len(validation_features)), {
            "source_seasons": [],
            "fit_rows": 0,
            "alpha": alpha,
            "coefficients": [0.0] * len(FEATURE_NAMES),
        }
    source_x = np.concatenate([features[season] for season in source_seasons])
    centered_targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for season in source_seasons:
        residual = targets[season] - bases[season]
        centered_targets.append(residual - residual.mean())
        weights.append(
            np.full(
                len(residual),
                1.0 / (len(source_seasons) * len(residual)),
                dtype=float,
            )
        )
    source_y = np.concatenate(centered_targets)
    sample_weight = np.concatenate(weights)
    sample_weight *= len(sample_weight)
    mean = np.average(source_x, axis=0, weights=sample_weight)
    variance = np.average(np.square(source_x - mean), axis=0, weights=sample_weight)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    train_x = (source_x - mean) / scale
    valid_x = (validation_features - mean) / scale
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(train_x, source_y, sample_weight=sample_weight)
    correction = np.clip(
        model.predict(valid_x), -CORRECTION_CLIP, CORRECTION_CLIP
    )
    return correction, {
        "source_seasons": source_seasons,
        "fit_rows": int(len(source_x)),
        "alpha": alpha,
        "source_season_equal_total_weight": True,
        "source_residual_center_max_abs": float(
            max(abs(float(value.mean())) for value in centered_targets)
        ),
        "coefficients": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, model.coef_, strict=True)
        },
        "correction_mean_abs": float(np.mean(np.abs(correction))),
        "correction_max_abs": float(np.max(np.abs(correction))),
    }


def choose_from_history(folds: dict[str, object], seasons: list[int]) -> str:
    if not seasons:
        return "career_a10000_w025"
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
            -CANDIDATES[name][1],
        ),
    )


def main() -> None:
    started = time.time()
    frame = load_frame()
    states, _ = season_latent_states(frame)
    summary = season_summary(frame, states)
    career = prior_career_states(frame)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    features: dict[int, np.ndarray] = {}
    feature_audits: dict[str, object] = {}
    targets: dict[int, np.ndarray] = {}
    bases: dict[int, np.ndarray] = {}
    rows_by_season: dict[int, pd.DataFrame] = {}
    for season in EVALUATED_SEASONS:
        rows = frame.loc[frame["season"].eq(season)].reset_index(drop=True)
        target = np.load(TARGET_ROOT / f"targets_{season}.npy").astype(float)
        if not np.array_equal(target, rows["control_success"].to_numpy(float)):
            raise ValueError(f"target/order mismatch in {season}")
        rows_by_season[season] = rows
        targets[season] = target
        bases[season] = exp051_base(season)
        features[season], feature_audits[str(season)] = feature_matrix(
            rows, season, summary, career[season]
        )

    folds: dict[str, object] = {}
    prediction_cache: dict[int, dict[str, np.ndarray]] = {}
    for season in EVALUATED_SEASONS:
        source_seasons = [value for value in EVALUATED_SEASONS if value < season]
        correction_by_alpha: dict[float, np.ndarray] = {}
        fit_audit: dict[str, object] = {}
        for alpha in sorted({value[0] for value in CANDIDATES.values()}):
            correction_by_alpha[alpha], fit_audit[str(int(alpha))] = fit_correction(
                source_seasons,
                features,
                targets,
                bases,
                features[season],
                alpha,
            )
        predictions = {"base_exp051": bases[season]}
        for name, (alpha, weight) in CANDIDATES.items():
            predictions[name] = np.clip(
                bases[season] + weight * correction_by_alpha[alpha], 0.0, 1.0
            )

        # Diagnostic ceiling: fit and score the same fold.  Never deploy it.
        same_correction, same_audit = fit_correction(
            [season], features, targets, bases, features[season], 1000.0
        )
        fold: dict[str, object] = {
            "validation_season": season,
            "history_cutoff": season - 1,
            "feature_audit": feature_audits[str(season)],
            "prior_fit_audit": fit_audit,
            "same_fold_fit_diagnostic_only": {
                "fit_audit": same_audit,
                "unit_weight_metrics": calculate_metrics(
                    targets[season],
                    np.clip(bases[season] + same_correction, 0.0, 1.0),
                ),
                "scalar_oracle": same_fold_oracle(
                    targets[season], bases[season], same_correction
                ),
            },
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(targets[season], prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{season}.npy", prediction)
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", targets[season].astype(np.int8))
        folds[str(season)] = fold
        prediction_cache[season] = predictions
        print(
            f"fold {season}: "
            + " ".join(
                f"{name}={fold[name]['skill_score_unclipped']:.2f}"
                for name in CANDIDATES
            )
            + f" samefold={fold['same_fold_fit_diagnostic_only']['scalar_oracle']['skill_score_unclipped']:.2f}",
            flush=True,
        )

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

    strict_path: dict[str, object] = {}
    strict_skills: list[float] = []
    strict_briers: list[float] = []
    for season in REPORT_SEASONS:
        selection_history = [value for value in EVALUATED_SEASONS if value < season]
        selected = choose_from_history(folds, selection_history)
        metric = folds[str(season)][selected]
        strict_path[str(season)] = {
            "selected_using_seasons": selection_history,
            "candidate": selected,
            "metrics": metric,
        }
        strict_skills.append(float(metric["skill_score_unclipped"]))
        strict_briers.append(float(metric["brier_score"]))

    strict_mean = float(np.mean(strict_skills))
    strict_min = float(np.min(strict_skills))
    next_candidate = choose_from_history(folds, list(EVALUATED_SEASONS))
    latest_ceiling = folds["2024"]["same_fold_fit_diagnostic_only"]["scalar_oracle"]
    result = {
        "experiment": EXPERIMENT,
        "candidate_family": "prior_only_pitcher_career_stage_ridge_residual",
        "validation_protocol": {
            "evaluated_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "base": "EXP-051 OOF trackman_direct_recent_w010",
            "source_seasons_strictly_prior": True,
            "source_residual_season_centered": True,
            "source_season_equal_total_weight": True,
            "current_row_official_asof_only": True,
            "current_fold_labels_used_for_candidate_fit_or_selection": False,
            "test_or_validation_row_aggregation": False,
            "candidate_grid_predeclared": True,
            "same_fold_fit_is_diagnostic_only": True,
        },
        "configuration": {
            "feature_names": list(FEATURE_NAMES),
            "correction_clip": CORRECTION_CLIP,
            "candidates": {
                name: {"ridge_alpha": alpha, "correction_weight": weight}
                for name, (alpha, weight) in CANDIDATES.items()
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
            "same_fold_2024": latest_ceiling,
            "latest_ceiling_reaches_1000": bool(
                latest_ceiling["skill_score_unclipped"] >= 1000.0
            ),
        },
        "selection": {
            "adopt": bool(strict_min >= 1000.0 and strict_mean >= 1100.0),
            "build_submission_zip": bool(
                strict_min >= 1000.0 and strict_mean >= 1100.0
            ),
            "stop_career_stage_family": bool(
                latest_ceiling["skill_score_unclipped"] < 1000.0
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
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        f"strict mean={strict_mean:.2f} min={strict_min:.2f} "
        f"next={next_candidate} stop={result['selection']['stop_career_stage_family']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
