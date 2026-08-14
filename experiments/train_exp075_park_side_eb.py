"""EXP-075: temporal-safe home-park and defensive-side residual EB.

The home team/park is reconstructed from current-row official fields only:
the pitcher team in the top half and the batter team in the bottom half.  For
each earlier OOF season, centered EXP-051 residual effects are estimated for
park, park x batting half, and park x month.  Source-season effects are
averaged equally with missing keys contributing zero.  No validation/test-row
aggregate or current-fold selection is used.
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
    EVALUATED_SEASONS,
    REPORT_SEASONS,
    exp051_base,
    same_fold_oracle,
)


EXPERIMENT = "EXP-075"
DATA_PATH = Path("./data/train.csv")
TARGET_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-075/park_side_eb")

EFFECTS = {
    "park_s2000": (("home_team",), 2000.0),
    "park_side_s1000": (("home_team", "top_bottom"), 1000.0),
    "park_month_s1000": (("home_team", "game_month"), 1000.0),
}
CANDIDATES = {
    "park_main": (("park_s2000", 1.0),),
    "park_side": (("park_side_s1000", 1.0),),
    "park_main_side_equal": (
        ("park_s2000", 0.5),
        ("park_side_s1000", 0.5),
    ),
    "park_main_month_equal": (
        ("park_s2000", 0.5),
        ("park_month_s1000", 0.5),
    ),
}


def load_rows() -> dict[int, pd.DataFrame]:
    columns = [
        "season",
        "game_month",
        "top_bottom",
        "pitcher_team_id",
        "batter_team_id",
        "control_success",
    ]
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=columns)
    frame = frame.loc[frame["season"].isin(EVALUATED_SEASONS)].copy()
    if frame[columns].isna().any().any():
        raise ValueError("missing park field")
    if set(frame["top_bottom"].astype(str).unique()) != {"T", "B"}:
        raise ValueError("unexpected top_bottom")
    top = frame["top_bottom"].astype(str).eq("T")
    frame["home_team"] = np.where(
        top, frame["pitcher_team_id"], frame["batter_team_id"]
    ).astype(np.int16)
    frame["away_team"] = np.where(
        top, frame["batter_team_id"], frame["pitcher_team_id"]
    ).astype(np.int16)
    if (frame["home_team"] == frame["away_team"]).any():
        raise ValueError("same home and away team")
    return {
        season: frame.loc[frame["season"].eq(season)].reset_index(drop=True)
        for season in EVALUATED_SEASONS
    }


def query_effect(
    source_rows: pd.DataFrame,
    residual: np.ndarray,
    validation_rows: pd.DataFrame,
    keys: tuple[str, ...],
    smoothing: float,
) -> tuple[np.ndarray, dict[str, object]]:
    work = source_rows.loc[:, list(keys)].copy()
    work["residual"] = residual
    stats = work.groupby(list(keys), sort=False)["residual"].agg(["sum", "count"])
    effects = stats["sum"] / (stats["count"] + smoothing)
    if len(keys) == 1:
        query = validation_rows[keys[0]]
    else:
        query = pd.MultiIndex.from_frame(validation_rows.loc[:, list(keys)])
    correction = effects.reindex(query).fillna(0.0).to_numpy(float)
    return correction, {
        "keys": list(keys),
        "smoothing": smoothing,
        "groups": int(len(stats)),
        "source_rows": int(len(source_rows)),
        "seen_validation_rows": int(np.count_nonzero(correction)),
        "mean_absolute_effect_seen": (
            float(np.mean(np.abs(correction[correction != 0.0])))
            if np.count_nonzero(correction)
            else 0.0
        ),
    }


def prior_corrections(
    validation_season: int,
    rows: dict[int, pd.DataFrame],
    targets: dict[int, np.ndarray],
    bases: dict[int, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    source_seasons = [value for value in EVALUATED_SEASONS if value < validation_season]
    validation_rows = rows[validation_season]
    per_effect: dict[str, list[np.ndarray]] = {name: [] for name in EFFECTS}
    audits: dict[str, object] = {}
    for source_season in source_seasons:
        raw = targets[source_season] - bases[source_season]
        residual = raw - raw.mean()
        season_audit: dict[str, object] = {
            "raw_residual_mean_removed": float(raw.mean()),
            "centered_residual_mean": float(residual.mean()),
        }
        for name, (keys, smoothing) in EFFECTS.items():
            correction, audit = query_effect(
                rows[source_season],
                residual,
                validation_rows,
                keys,
                smoothing,
            )
            per_effect[name].append(correction)
            season_audit[name] = audit
        audits[str(source_season)] = season_audit
    output = {
        name: (
            np.mean(values, axis=0)
            if values
            else np.zeros(len(validation_rows), dtype=float)
        )
        for name, values in per_effect.items()
    }
    return output, {
        "source_seasons": source_seasons,
        "source_effects": audits,
        "source_season_equal_mean_including_missing_zero": True,
        "correction_mean_abs": {
            name: float(np.mean(np.abs(value))) for name, value in output.items()
        },
    }


def same_fold_effects(
    season: int,
    rows: dict[int, pd.DataFrame],
    targets: dict[int, np.ndarray],
    bases: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    raw = targets[season] - bases[season]
    residual = raw - raw.mean()
    return {
        name: query_effect(rows[season], residual, rows[season], keys, smoothing)[0]
        for name, (keys, smoothing) in EFFECTS.items()
    }


def combine(effect: dict[str, np.ndarray], specification: tuple[tuple[str, float], ...]) -> np.ndarray:
    return sum(weight * effect[name] for name, weight in specification)


def choose_from_history(folds: dict[str, object], seasons: list[int]) -> str:
    if not seasons:
        return "park_main"
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
            -len(CANDIDATES[name]),
        ),
    )


def main() -> None:
    started = time.time()
    rows = load_rows()
    targets: dict[int, np.ndarray] = {}
    bases: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(TARGET_ROOT / f"targets_{season}.npy").astype(float)
        bases[season] = exp051_base(season)
        if not np.array_equal(
            targets[season], rows[season]["control_success"].to_numpy(float)
        ):
            raise ValueError(f"target/order mismatch in {season}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    folds: dict[str, object] = {}
    prediction_cache: dict[int, dict[str, np.ndarray]] = {}
    for season in EVALUATED_SEASONS:
        effects, audit = prior_corrections(season, rows, targets, bases)
        predictions = {"base_exp051": bases[season]}
        for name, specification in CANDIDATES.items():
            predictions[name] = np.clip(
                bases[season] + combine(effects, specification), 0.0, 1.0
            )
        oracle_effects = same_fold_effects(season, rows, targets, bases)
        oracle: dict[str, object] = {}
        for name, specification in CANDIDATES.items():
            correction = combine(oracle_effects, specification)
            oracle[name] = same_fold_oracle(
                targets[season], bases[season], correction
            )
        fold: dict[str, object] = {
            "validation_season": season,
            "history_cutoff": season - 1,
            "effect_audit": audit,
            "same_fold_oracle_diagnostic_only": oracle,
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
            + " oracle="
            + f"{max(value['skill_score_unclipped'] for value in oracle.values()):.2f}",
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

    strict_mean = float(np.mean(strict_skills))
    strict_min = float(np.min(strict_skills))
    latest_oracle = max(
        folds["2024"]["same_fold_oracle_diagnostic_only"].values(),
        key=lambda value: value["skill_score_unclipped"],
    )
    next_candidate = choose_from_history(folds, list(EVALUATED_SEASONS))
    result = {
        "experiment": EXPERIMENT,
        "candidate_family": "home_park_and_defensive_side_residual_EB",
        "validation_protocol": {
            "evaluated_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "base": "EXP-051 OOF trackman_direct_recent_w010",
            "home_team_formula": "pitcher_team_id if top_bottom=T else batter_team_id",
            "source_seasons_strictly_prior": True,
            "source_residual_season_centered": True,
            "source_season_equal_mean_missing_zero": True,
            "current_fold_labels_used_for_candidate_fit_or_selection": False,
            "test_or_validation_row_aggregation": False,
            "candidate_grid_predeclared": True,
            "same_fold_oracle_is_diagnostic_only": True,
        },
        "configuration": {
            "effects": {
                name: {"keys": list(keys), "smoothing": smoothing}
                for name, (keys, smoothing) in EFFECTS.items()
            },
            "candidates": {
                name: {effect: weight for effect, weight in specification}
                for name, specification in CANDIDATES.items()
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
            "stop_park_family": bool(
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
        f"next={next_candidate} stop={result['selection']['stop_park_family']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
