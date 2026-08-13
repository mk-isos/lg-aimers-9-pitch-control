"""EXP-057: source-season equal calendar/progression residual EB.

Each earlier OOF season estimates centered residual effects for the current
row's month, weekday and inning phase.  Source maps are averaged with missing
effects equal to zero before applying to the next outer fold.  No validation
labels or evaluation-row aggregates are used.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics


DATA_DIR = Path("./data")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
DIRECT_ROOT = Path("./artifacts/EXP-050/exact_dual_propensity_control")
ARTIFACT_DIR = Path("./artifacts/EXP-057/temporal_calendar_eb")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
CORRECTION_CLIP = 0.02
CANDIDATES = (
    "month_s5000",
    "month_game_s3000",
    "month_phase_s2500",
    "calendar_additive",
)


def load_rows() -> pd.DataFrame:
    columns = [
        "season",
        "game_month",
        "game_dayofweek",
        "game_type",
        "inning",
        "control_success",
    ]
    frame = pd.read_csv(
        DATA_DIR / "train.csv", encoding="utf-8-sig", usecols=columns
    )
    frame["inning_phase"] = np.select(
        [frame["inning"].le(3), frame["inning"].le(6)], [0, 1], default=2
    ).astype(np.int8)
    return frame


def base_prediction(season: int) -> np.ndarray:
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    base = 0.5 * recency + 0.5 * aggressive
    direct = (
        np.load(DIRECT_ROOT / f"predictions_pitcher_prop_w025_{season}.npy")
        - np.load(DIRECT_ROOT / f"predictions_base_{season}.npy")
    ) / 0.25
    return np.clip(base + 0.10 * direct, 0.0, 1.0)


def fit_effect(
    rows: pd.DataFrame,
    residual: np.ndarray,
    columns: list[str],
    smoothing: float,
) -> pd.Series:
    work = rows.loc[:, columns].copy()
    work["residual"] = residual - residual.mean()
    stats = work.groupby(columns, sort=True)["residual"].agg(["sum", "count"])
    return stats["sum"] / (stats["count"] + smoothing)


def map_effect(
    rows: pd.DataFrame, effect: pd.Series, columns: list[str]
) -> np.ndarray:
    if len(columns) == 1:
        return rows[columns[0]].map(effect).fillna(0.0).to_numpy(float)
    index = pd.MultiIndex.from_frame(rows[columns])
    return effect.reindex(index).fillna(0.0).to_numpy(float)


def source_average(
    validation_rows: pd.DataFrame,
    source_maps: list[pd.Series],
    columns: list[str],
) -> np.ndarray:
    return np.mean(
        np.vstack(
            [map_effect(validation_rows, effect, columns) for effect in source_maps]
        ),
        axis=0,
    )


def main() -> None:
    started = time.time()
    rows = load_rows()
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    residual: dict[int, np.ndarray] = {}
    season_rows: dict[int, pd.DataFrame] = {}
    for season in EVALUATED_SEASONS:
        season_rows[season] = rows.loc[rows["season"].eq(season)].reset_index(drop=True)
        targets[season] = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        if not np.array_equal(
            season_rows[season]["control_success"].to_numpy(float), targets[season]
        ):
            raise ValueError(f"target/order mismatch {season}")
        base[season] = base_prediction(season)
        residual[season] = targets[season] - base[season]

    specs = {
        "month": (["game_month"], 5000.0),
        "month_game": (["game_month", "game_type"], 3000.0),
        "month_phase": (["game_month", "inning_phase"], 2500.0),
        "weekday": (["game_dayofweek"], 5000.0),
    }
    maps: dict[str, dict[int, pd.Series]] = {name: {} for name in specs}
    for name, (columns, smoothing) in specs.items():
        for season in EVALUATED_SEASONS:
            maps[name][season] = fit_effect(
                season_rows[season], residual[season], columns, smoothing
            )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        sources = [season for season in EVALUATED_SEASONS if season < validation_season]
        correction: dict[str, np.ndarray] = {}
        for name, (columns, _) in specs.items():
            if sources:
                correction[name] = source_average(
                    season_rows[validation_season],
                    [maps[name][source] for source in sources],
                    columns,
                )
            else:
                correction[name] = np.zeros(len(targets[validation_season]))
        predictions = {
            "base": base[validation_season],
            "month_s5000": np.clip(
                base[validation_season]
                + np.clip(correction["month"], -CORRECTION_CLIP, CORRECTION_CLIP),
                0.0,
                1.0,
            ),
            "month_game_s3000": np.clip(
                base[validation_season]
                + np.clip(correction["month_game"], -CORRECTION_CLIP, CORRECTION_CLIP),
                0.0,
                1.0,
            ),
            "month_phase_s2500": np.clip(
                base[validation_season]
                + np.clip(correction["month_phase"], -CORRECTION_CLIP, CORRECTION_CLIP),
                0.0,
                1.0,
            ),
            "calendar_additive": np.clip(
                base[validation_season]
                + np.clip(
                    0.5 * correction["month"]
                    + 0.25 * correction["month_phase"]
                    + 0.25 * correction["weekday"],
                    -CORRECTION_CLIP,
                    CORRECTION_CLIP,
                ),
                0.0,
                1.0,
            ),
        }
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_oof_seasons": sources,
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(targets[validation_season], prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{validation_season}.npy", prediction)
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets[validation_season].astype(np.int8))
        folds[str(validation_season)] = fold
        print(
            f"fold {validation_season}: "
            + " ".join(
                f"{name}={fold[name]['skill_score_unclipped']:.2f}"
                for name in CANDIDATES
            ),
            flush=True,
        )

    aggregate: dict[str, object] = {}
    for name in ("base", *CANDIDATES):
        skills = {
            str(season): float(folds[str(season)][name]["skill_score_unclipped"])
            for season in REPORT_SEASONS
        }
        briers = {
            str(season): float(folds[str(season)][name]["brier_score"])
            for season in REPORT_SEASONS
        }
        aggregate[name] = {
            "season_briers": briers,
            "season_skills": skills,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills["2024"],
        }
    best = max(CANDIDATES, key=lambda name: (aggregate[name]["min_skill"], aggregate[name]["mean_skill"]))
    result = {
        "experiment": "EXP-057",
        "candidate_family": "source_season_equal_calendar_progression_eb",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "source_residuals": "earlier OOF seasons, each centered, equal source mean",
            "current_fold_labels_used_for_fit_or_selection": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "model": {
            "specs": {
                name: {"columns": columns, "smoothing": smoothing}
                for name, (columns, smoothing) in specs.items()
            },
            "correction_clip": CORRECTION_CLIP,
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_fixed_candidate": best,
            "best_min_skill": aggregate[best]["min_skill"],
            "gate_each_season_1000": bool(min(aggregate[best]["season_skills"].values()) >= 1000.0),
            "gate_mean_1050": bool(aggregate[best]["mean_skill"] >= 1050.0),
            "adopt": bool(min(aggregate[best]["season_skills"].values()) >= 1000.0),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "total_seconds": time.time() - started,
    }
    (ARTIFACT_DIR / "validation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"best={best} mean={aggregate[best]['mean_skill']:.2f} "
        f"min={aggregate[best]['min_skill']:.2f} adopt={result['selection']['adopt']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
