"""EXP-076: prior-season sign-stable park residual effects.

EXP-075 showed useful same-season park resolution but poor adjacent-season
correlation.  This bounded diagnostic transfers an effect only when the same
key has the same non-zero sign in at least the two most recent prior OOF
seasons.  The correction is their simple mean.  With fewer than two prior OOF
seasons the effect is exactly zero, so 2022 remains the immutable EXP-051
base.  No current-fold label or validation/test-row aggregate is used.
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
)
from train_exp075_park_side_eb import EFFECTS, load_rows


EXPERIMENT = "EXP-076"
TARGET_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-076/stable_park_eb")
MIN_SOURCE_SEASONS = 2

CANDIDATES = {
    "stable_main": (("park_s2000", 1.0),),
    "stable_side": (("park_side_s1000", 1.0),),
    "stable_month": (("park_month_s1000", 1.0),),
    "stable_main_side_equal": (
        ("park_s2000", 0.5),
        ("park_side_s1000", 0.5),
    ),
}


def source_map(
    rows: pd.DataFrame,
    residual: np.ndarray,
    keys: tuple[str, ...],
    smoothing: float,
) -> pd.Series:
    work = rows.loc[:, list(keys)].copy()
    work["residual"] = residual
    stats = work.groupby(list(keys), sort=False)["residual"].agg(["sum", "count"])
    return stats["sum"] / (stats["count"] + smoothing)


def query_index(rows: pd.DataFrame, keys: tuple[str, ...]) -> pd.Index:
    if len(keys) == 1:
        return pd.Index(rows[keys[0]].to_numpy(), name=keys[0])
    return pd.MultiIndex.from_frame(rows.loc[:, list(keys)])


def stable_effects(
    validation_season: int,
    rows: dict[int, pd.DataFrame],
    targets: dict[int, np.ndarray],
    bases: dict[int, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    all_prior = [season for season in EVALUATED_SEASONS if season < validation_season]
    selected_sources = all_prior[-2:]
    outputs: dict[str, np.ndarray] = {}
    audit: dict[str, object] = {
        "all_prior_sources": all_prior,
        "selected_last_two_sources": selected_sources,
        "minimum_sources": MIN_SOURCE_SEASONS,
    }
    for effect_name, (keys, smoothing) in EFFECTS.items():
        if len(selected_sources) < MIN_SOURCE_SEASONS:
            outputs[effect_name] = np.zeros(len(rows[validation_season]), dtype=float)
            audit[effect_name] = {
                "transferred": False,
                "reason": "fewer than two prior OOF seasons",
                "stable_keys": 0,
                "validation_rows_with_effect": 0,
            }
            continue
        maps: list[pd.Series] = []
        for source_season in selected_sources:
            raw = targets[source_season] - bases[source_season]
            residual = raw - raw.mean()
            maps.append(source_map(rows[source_season], residual, keys, smoothing))
        aligned = pd.concat(maps, axis=1, join="inner")
        aligned.columns = [str(season) for season in selected_sources]
        signs = np.sign(aligned.to_numpy(float))
        stable = np.all(signs == signs[:, [0]], axis=1) & np.all(signs != 0.0, axis=1)
        stable_map = aligned.loc[stable].mean(axis=1)
        query = query_index(rows[validation_season], keys)
        correction = stable_map.reindex(query).fillna(0.0).to_numpy(float)
        outputs[effect_name] = correction
        audit[effect_name] = {
            "transferred": True,
            "common_keys": int(len(aligned)),
            "stable_keys": int(stable.sum()),
            "stable_key_rate": float(stable.mean()) if len(stable) else 0.0,
            "validation_rows_with_effect": int(np.count_nonzero(correction)),
            "validation_row_coverage": float(np.count_nonzero(correction) / len(correction)),
            "mean_absolute_effect": float(np.mean(np.abs(correction))),
        }
    return outputs, audit


def combine(effects: dict[str, np.ndarray], specification: tuple[tuple[str, float], ...]) -> np.ndarray:
    return sum(weight * effects[name] for name, weight in specification)


def choose_from_history(folds: dict[str, object], seasons: list[int]) -> str:
    if not seasons:
        return "stable_main"
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
    for season in EVALUATED_SEASONS:
        effects, audit = stable_effects(season, rows, targets, bases)
        predictions = {"base_exp051": bases[season]}
        for name, specification in CANDIDATES.items():
            predictions[name] = np.clip(
                bases[season] + combine(effects, specification), 0.0, 1.0
            )
        fold: dict[str, object] = {
            "validation_season": season,
            "history_cutoff": season - 1,
            "stability_audit": audit,
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(targets[season], prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{season}.npy", prediction)
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", targets[season].astype(np.int8))
        folds[str(season)] = fold
        print(
            f"fold {season}: "
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

    strict_mean = float(np.mean(strict_skills))
    strict_min = float(np.min(strict_skills))
    next_candidate = choose_from_history(folds, list(EVALUATED_SEASONS))
    result = {
        "experiment": EXPERIMENT,
        "candidate_family": "last_two_source_sign_stable_park_EB",
        "validation_protocol": {
            "evaluated_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "base": "EXP-051 OOF trackman_direct_recent_w010",
            "source_seasons_strictly_prior": True,
            "last_two_prior_sources_only": True,
            "current_fold_labels_used_for_fit_or_selection": False,
            "test_or_validation_row_aggregation": False,
            "candidate_family_posthoc_motivated_by_EXP075_correlation_audit": True,
        },
        "configuration": {
            "minimum_source_seasons": MIN_SOURCE_SEASONS,
            "stability_rule": "same non-zero sign in both last-two prior OOF seasons",
            "stable_value": "simple mean of last-two source effects",
            "effects_reused_from_EXP075": {
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
        "selection": {
            "adopt": bool(strict_min >= 1000.0 and strict_mean >= 1100.0),
            "build_submission_zip": bool(
                strict_min >= 1000.0 and strict_mean >= 1100.0
            ),
            "park_resolution_from_EXP075_is_not_temporally_transferable": True,
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
        f"next={next_candidate} adopt={result['selection']['adopt']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
