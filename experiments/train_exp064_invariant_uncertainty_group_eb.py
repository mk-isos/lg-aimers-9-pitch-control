"""EXP-064: invariant uncertainty-group empirical Bayes corrections.

Same-fold diagnostics show useful residual structure by count, hand and base
prediction bin, but naive previous-season transfer reverses.  For each outer
fold this experiment estimates one map per prior OOF season, retains only
cells whose source effects agree in sign, and shrinks heterogeneous cells.
Validation labels and test-row aggregates are never used for construction.
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
TRACKMAN_ROOT = Path("./artifacts/EXP-043/exact_pitchtype_control_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-064/invariant_uncertainty_group_eb")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
SMOOTHING = 500.0
CORRECTION_CLIP = 0.02
CANDIDATES = (
    "stable_count_hand_pbin_w050",
    "stable_count_runners_pbin_w050",
    "stable_blend_w050",
    "heterogeneity_blend_w050",
)


def base_prediction(season: int) -> np.ndarray:
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    strict = np.load(
        LOWRANK_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
    ).astype(float)
    trackman = np.load(
        TRACKMAN_ROOT / f"predictions_fine_direct_w025_{season}.npy"
    ).astype(float)
    return np.clip(
        0.5 * recency + 0.5 * aggressive + 0.10 * (trackman - strict) / 0.25,
        0.0,
        1.0,
    )


def row_keys(rows: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    output = pd.DataFrame(index=rows.index)
    output["count"] = 4 * rows["balls_before"] + rows["strikes_before"]
    output["hand"] = (
        rows["pitcher_hand"].astype(str) + "_" + rows["batter_hand"].astype(str)
    )
    output["runners"] = rows["num_runners_on"].astype(int)
    output["pbin"] = np.clip(((prediction - 0.35) / 0.025).astype(int), 0, 12)
    return output.reset_index(drop=True)


def season_map(
    keys: pd.DataFrame,
    residual: np.ndarray,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    work = keys.loc[:, list(columns)].copy()
    work["residual"] = residual - residual.mean()
    stats = work.groupby(list(columns), dropna=False)["residual"].agg(
        ["sum", "count"]
    )
    stats["effect"] = stats["sum"] / (stats["count"] + SMOOTHING)
    return stats[["effect", "count"]]


def invariant_correction(
    source_maps: list[pd.DataFrame],
    validation_keys: pd.DataFrame,
    columns: tuple[str, ...],
    heterogeneity: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    union = source_maps[0].index
    for source in source_maps[1:]:
        union = union.union(source.index)
    effects = np.column_stack(
        [source["effect"].reindex(union).fillna(0.0).to_numpy() for source in source_maps]
    )
    counts = np.column_stack(
        [source["count"].reindex(union).fillna(0.0).to_numpy() for source in source_maps]
    )
    nonzero = counts > 0
    positive = effects > 0
    negative = effects < 0
    if len(source_maps) == 1:
        stable = nonzero[:, 0]
    else:
        stable = nonzero.all(axis=1) & (positive.all(axis=1) | negative.all(axis=1))
    mean = effects.mean(axis=1)
    spread = effects.std(axis=1)
    if heterogeneity:
        reliability = np.abs(mean) / (np.abs(mean) + spread + 0.002)
    else:
        reliability = np.ones(len(mean))
    effect = np.where(stable, mean * reliability, 0.0)
    series = pd.Series(effect, index=union)
    if len(columns) == 1:
        query = pd.Index(validation_keys[columns[0]])
    else:
        query = pd.MultiIndex.from_frame(validation_keys.loc[:, list(columns)])
    correction = series.reindex(query).fillna(0.0).to_numpy(float)
    correction = np.clip(correction, -CORRECTION_CLIP, CORRECTION_CLIP)
    return correction, {
        "union_cells": len(union),
        "stable_cells": int(stable.sum()),
        "stable_fraction": float(stable.mean()),
        "mapped_validation_fraction": float((correction != 0).mean()),
        "mean_abs_correction": float(np.mean(np.abs(correction))),
    }


def main() -> None:
    started = time.time()
    raw = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=[
            "season",
            "balls_before",
            "strikes_before",
            "pitcher_hand",
            "batter_hand",
            "num_runners_on",
            "control_success",
        ],
    )
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    keys: dict[int, pd.DataFrame] = {}
    residual: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        rows = raw.loc[raw["season"].eq(season)].reset_index(drop=True)
        targets[season] = np.load(
            LOWRANK_ROOT / f"targets_{season}.npy"
        ).astype(float)
        base[season] = base_prediction(season)
        keys[season] = row_keys(rows, base[season])
        residual[season] = targets[season] - base[season]
        if not np.array_equal(targets[season], rows["control_success"].to_numpy(float)):
            raise ValueError(f"target/order mismatch {season}")

    specs = {
        "count_hand_pbin": ("count", "hand", "pbin"),
        "count_runners_pbin": ("count", "runners", "pbin"),
    }
    maps = {
        name: {
            season: season_map(keys[season], residual[season], columns)
            for season in EVALUATED_SEASONS
        }
        for name, columns in specs.items()
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        sources = [season for season in EVALUATED_SEASONS if season < validation_season]
        corrections: dict[str, np.ndarray] = {
            name: np.zeros(len(targets[validation_season])) for name in specs
        }
        hetero: dict[str, np.ndarray] = {
            name: np.zeros(len(targets[validation_season])) for name in specs
        }
        audit: dict[str, object] = {}
        if sources:
            for name, columns in specs.items():
                source_maps = [maps[name][season] for season in sources]
                corrections[name], audit[f"{name}_stable"] = invariant_correction(
                    source_maps, keys[validation_season], columns, False
                )
                hetero[name], audit[f"{name}_heterogeneity"] = invariant_correction(
                    source_maps, keys[validation_season], columns, True
                )
        stable_blend = 0.5 * (
            corrections["count_hand_pbin"] + corrections["count_runners_pbin"]
        )
        hetero_blend = 0.5 * (
            hetero["count_hand_pbin"] + hetero["count_runners_pbin"]
        )
        predictions = {
            "base": base[validation_season],
            "stable_count_hand_pbin_w050": np.clip(
                base[validation_season] + 0.50 * corrections["count_hand_pbin"], 0, 1
            ),
            "stable_count_runners_pbin_w050": np.clip(
                base[validation_season] + 0.50 * corrections["count_runners_pbin"], 0, 1
            ),
            "stable_blend_w050": np.clip(
                base[validation_season] + 0.50 * stable_blend, 0, 1
            ),
            "heterogeneity_blend_w050": np.clip(
                base[validation_season] + 0.50 * hetero_blend, 0, 1
            ),
        }
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_seasons": sources,
            "map_audit": audit,
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(targets[validation_season], prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{validation_season}.npy", prediction)
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets[validation_season].astype(np.int8),
        )
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
    best = max(
        CANDIDATES,
        key=lambda name: (aggregate[name]["min_skill"], aggregate[name]["mean_skill"]),
    )
    result = {
        "experiment": "EXP-064",
        "candidate_family": "sign_consistent_uncertainty_group_eb",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "source_maps": "prior OOF seasons only",
            "current_fold_labels_used_for_fit_or_selection": False,
            "source_season_equal_weight": True,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "model": {
            "group_specs": {name: list(columns) for name, columns in specs.items()},
            "pbin": "clip(floor((EXP051 prediction-.35)/.025),0,12)",
            "smoothing": SMOOTHING,
            "correction_clip": CORRECTION_CLIP,
            "stability": "nonzero every source season and common sign",
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_fixed_candidate": best,
            "best_min_skill": aggregate[best]["min_skill"],
            "best_mean_skill": aggregate[best]["mean_skill"],
            "gate_each_season_1000": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
            ),
            "gate_mean_1100": bool(aggregate[best]["mean_skill"] >= 1100.0),
            "adopt": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
                and aggregate[best]["mean_skill"] >= 1100.0
            ),
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
