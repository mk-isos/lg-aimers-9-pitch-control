"""EXP-073: joint pitcher/batter dynamic latent-state correction.

This is the bounded companion to EXP-072.  The same prior-season-only AR(1)
state forecast is built independently for pitchers and batters.  Current-row
career as-of counts are converted to current-season counts with frozen train
end states.  The dynamic prior replaces only the league-prior part of the
row-local posterior; no evaluation-row aggregation is performed.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics
from train_exp072_dynamic_pitcher_state import (
    ARTIFACT_DIR as PITCHER_ARTIFACT_DIR,
    EVALUATED_SEASONS,
    REPORT_SEASONS,
    dynamic_deltas,
    exp051_base,
    prior_career_states,
    season_latent_states,
    same_fold_oracle,
)


EXPERIMENT = "EXP-073"
DATA_PATH = Path("./data/train.csv")
TARGET_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-073/dynamic_batter_state")

# Fixed before the EXP-073 run.  0.7/0.3 matches the long-standing temporal
# pitcher/batter decomposition; the batter-only candidate isolates novelty.
CANDIDATES = {
    "joint_ar30_w025": ("joint", 0.25),
    "joint_ar30_w050": ("joint", 0.50),
    "batter_ar30_w025": ("batter", 0.25),
}


def load_frame() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_id",
        "batter_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "control_success",
    ]
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=columns)
    required = [
        column
        for column in columns
        if column not in {"asof_pitcher_success_rate", "asof_batter_success_rate"}
    ]
    if frame[required].isna().any().any():
        raise ValueError("missing required official field")
    for entity in ("pitcher", "batter"):
        rate = f"asof_{entity}_success_rate"
        count = f"asof_{entity}_n"
        missing = frame[rate].isna()
        if not frame.loc[missing, count].eq(0).all():
            raise ValueError(f"{entity} rate missing at positive n")
        frame[rate] = frame[rate].fillna(0.0)
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("train rows must be season-monotone")
    return frame


def entity_view(frame: pd.DataFrame, entity: str) -> pd.DataFrame:
    columns = [
        "season",
        f"{entity}_id",
        f"asof_{entity}_n",
        f"asof_{entity}_success_rate",
        "control_success",
    ]
    return frame[columns].rename(
        columns={
            f"{entity}_id": "pitcher_id",
            f"asof_{entity}_n": "asof_pitcher_n",
            f"asof_{entity}_success_rate": "asof_pitcher_success_rate",
        }
    )


def choose_from_history(folds: dict[str, object], seasons: list[int]) -> str:
    if not seasons:
        return "joint_ar30_w025"
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
    views = {entity: entity_view(frame, entity) for entity in ("pitcher", "batter")}
    states: dict[str, pd.DataFrame] = {}
    league_rates: dict[str, dict[int, float]] = {}
    career: dict[str, object] = {}
    for entity, view in views.items():
        states[entity], league_rates[entity] = season_latent_states(view)
        career[entity] = prior_career_states(view)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    prediction_cache: dict[int, dict[str, np.ndarray]] = {}
    targets: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        target = np.load(TARGET_ROOT / f"targets_{season}.npy").astype(float)
        base = exp051_base(season)
        if not np.array_equal(
            target,
            frame.loc[frame["season"].eq(season), "control_success"].to_numpy(float),
        ):
            raise ValueError(f"target/order mismatch in {season}")
        entity_delta: dict[str, dict[str, np.ndarray]] = {}
        audits: dict[str, object] = {}
        for entity in ("pitcher", "batter"):
            rows = views[entity].loc[
                views[entity]["season"].eq(season)
            ].reset_index(drop=True)
            entity_delta[entity], audits[entity] = dynamic_deltas(
                rows,
                season,
                states[entity],
                league_rates[entity],
                career[entity][season],
            )
        pitcher_delta = entity_delta["pitcher"]["ar_k30"]
        batter_delta = entity_delta["batter"]["ar_k30"]
        corrections = {
            "joint": 0.7 * pitcher_delta + 0.3 * batter_delta,
            "batter": batter_delta,
        }
        predictions = {"base_exp051": base}
        for name, (kind, weight) in CANDIDATES.items():
            predictions[name] = np.clip(base + weight * corrections[kind], 0.0, 1.0)
        fold: dict[str, object] = {
            "validation_season": season,
            "history_cutoff": season - 1,
            "entity_state_audit": audits,
            "same_fold_oracle_diagnostic_only": {
                name: same_fold_oracle(target, base, correction)
                for name, correction in corrections.items()
            },
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(target, prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{season}.npy", prediction)
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", target.astype(np.int8))
        folds[str(season)] = fold
        prediction_cache[season] = predictions
        targets[season] = target
        print(
            f"fold {season}: pitcher_rho={audits['pitcher']['rho']:.4f} "
            f"batter_rho={audits['batter']['rho']:.4f} "
            + " ".join(
                f"{name}={fold[name]['skill_score_unclipped']:.2f}"
                for name in CANDIDATES
            ),
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
        history = [value for value in EVALUATED_SEASONS if value < season]
        selected = choose_from_history(folds, history)
        metric = folds[str(season)][selected]
        strict_path[str(season)] = {
            "selected_using_seasons": history,
            "candidate": selected,
            "metrics": metric,
        }
        strict_skills.append(float(metric["skill_score_unclipped"]))
        strict_briers.append(float(metric["brier_score"]))

    next_candidate = choose_from_history(folds, list(EVALUATED_SEASONS))
    strict_mean = float(np.mean(strict_skills))
    strict_min = float(np.min(strict_skills))
    latest_oracle = max(
        folds["2024"]["same_fold_oracle_diagnostic_only"].values(),
        key=lambda value: value["skill_score_unclipped"],
    )
    result = {
        "experiment": EXPERIMENT,
        "candidate_family": "joint_pitcher_batter_dynamic_AR1_state",
        "validation_protocol": {
            "evaluated_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "base": "EXP-051 OOF trackman_direct_recent_w010",
            "entity_state_and_transition_fit_strictly_prior": True,
            "current_row_official_asof_only": True,
            "current_fold_labels_used_for_fit_or_selection": False,
            "test_or_validation_row_aggregation": False,
            "candidate_grid_predeclared": True,
            "same_fold_oracle_is_diagnostic_only": True,
        },
        "configuration": {
            "joint_weights": {"pitcher": 0.7, "batter": 0.3},
            "current_season_prior_strength": 30.0,
            "candidates": {
                name: {"correction": kind, "weight": weight}
                for name, (kind, weight) in CANDIDATES.items()
            },
            "pitcher_reference_artifact": str(PITCHER_ARTIFACT_DIR),
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
            "stop_dynamic_state_family": bool(
                latest_oracle["skill_score_unclipped"] < 1000.0
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
        f"next={next_candidate} stop={result['selection']['stop_dynamic_state_family']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
