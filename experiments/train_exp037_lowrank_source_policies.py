"""EXP-037: temporal source policies for low-rank pitcher-context effects.

The existing strict candidate averages every prior OOF source-season low-rank
matrix equally.  This bounded experiment keeps smoothing=300 and rank=6 fixed
and changes only the deployable combination of already prior-only source
corrections: last, recency-weighted, median, and capped linear trend.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from train_exp017_rolling_residual import calculate_metrics
from train_exp020_low_rank_pitcher_context_eb import (
    EVALUATED_SEASONS,
    REPORT_SEASONS,
    fit_source_matrix,
    load_oof,
    load_rows,
    map_source_matrix,
)


ARTIFACT_DIR = Path("./artifacts/EXP-037/lowrank_source_policies")
SMOOTHING = 300.0
RANK = 6
POLICIES = (
    "equal",
    "last",
    "recency2",
    "recency4",
    "median",
    "trend025",
    "direction_shrink",
    "sign_consensus",
    "adaptive_recency2",
    "last_guarded",
)


def combine_source_corrections(
    values: list[np.ndarray], policy: str
) -> np.ndarray:
    matrix = np.vstack(values)
    if policy == "equal":
        return matrix.mean(axis=0)
    if policy == "last":
        return matrix[-1]
    if policy in {"recency2", "recency4"}:
        base = 2.0 if policy == "recency2" else 4.0
        weights = np.power(base, np.arange(len(values), dtype=float))
        weights /= weights.sum()
        return np.average(matrix, axis=0, weights=weights)
    if policy == "median":
        return np.median(matrix, axis=0)
    if policy == "trend025":
        if len(values) < 2:
            return matrix[-1]
        delta = np.clip(matrix[-1] - matrix[-2], -0.03, 0.03)
        return matrix[-1] + 0.25 * delta
    signs = np.sign(matrix)
    directional_agreement = np.abs(signs.mean(axis=0))
    equal = matrix.mean(axis=0)
    if policy == "direction_shrink":
        return equal * directional_agreement
    if policy == "sign_consensus":
        required = 1.0 if len(values) <= 2 else (2.0 / 3.0)
        return equal * (directional_agreement >= required)
    if policy == "adaptive_recency2":
        weights = np.power(2.0, np.arange(len(values), dtype=float))
        weights /= weights.sum()
        recent = np.average(matrix, axis=0, weights=weights)
        return equal + directional_agreement * (recent - equal)
    if policy == "last_guarded":
        agreement = np.sign(matrix[-1]) == np.sign(equal)
        guarded = 0.5 * equal
        guarded[agreement] = 0.5 * (
            equal[agreement] + matrix[-1, agreement]
        )
        return guarded
    raise ValueError(f"unknown source policy: {policy}")


def main() -> None:
    started = time.time()
    rows = load_rows()
    targets, base, _ = load_oof(rows)
    source_models: dict[int, dict[str, object]] = {}

    def source_model(season: int) -> dict[str, object]:
        if season not in source_models:
            source_models[season] = fit_source_matrix(
                season,
                rows[season],
                targets[season],
                base[season],
                smoothing_grid=(SMOOTHING,),
                rank_grid=(RANK,),
            )
        return source_models[season]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        source_seasons = [
            season for season in EVALUATED_SEASONS if season < validation_season
        ]
        corrections: list[np.ndarray] = []
        for season in source_seasons:
            mapped = map_source_matrix(source_model(season), rows[validation_season])
            corrections.append(mapped["low_rank_values"][(SMOOTHING, RANK)])
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
        }
        for policy in POLICIES:
            if corrections:
                correction = combine_source_corrections(corrections, policy)
            else:
                correction = np.zeros(len(targets[validation_season]), dtype=float)
            prediction = np.clip(base[validation_season] + correction, 0.0, 1.0)
            if not np.isfinite(prediction).all() or not (
                (prediction >= 0.0).all() and (prediction <= 1.0).all()
            ):
                raise ValueError(f"invalid prediction {validation_season} {policy}")
            fold[policy] = calculate_metrics(targets[validation_season], prediction)
            np.save(
                ARTIFACT_DIR / f"predictions_{policy}_{validation_season}.npy",
                prediction,
            )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets[validation_season].astype(np.int8),
        )
        folds[str(validation_season)] = fold
        print(
            f"fold {validation_season}: "
            + " ".join(
                f"{policy}={fold[policy]['skill_score_unclipped']:.2f}"
                for policy in POLICIES
            ),
            flush=True,
        )

    aggregate: dict[str, object] = {}
    for policy in POLICIES:
        skills = {
            str(season): float(
                folds[str(season)][policy]["skill_score_unclipped"]
            )
            for season in REPORT_SEASONS
        }
        briers = {
            str(season): float(folds[str(season)][policy]["brier_score"])
            for season in REPORT_SEASONS
        }
        aggregate[policy] = {
            "season_briers": briers,
            "season_skills": skills,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills["2024"],
        }
    best = max(
        POLICIES,
        key=lambda policy: (
            aggregate[policy]["min_skill"],
            aggregate[policy]["latest_2024_skill"],
            aggregate[policy]["mean_skill"],
        ),
    )
    result = {
        "experiment": "EXP-037",
        "candidate_family": "lowrank_source_combination_policies",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "base": "EXP-019 team all_prior_s1000 OOF",
            "source_effect": "source-season centered lowrank s300 rank6",
            "current_fold_labels_used_for_fit_or_policy": False,
            "test_row_aggregation": False,
            "policies_predeclared": True,
        },
        "model": {
            "smoothing": SMOOTHING,
            "rank": RANK,
            "policies": list(POLICIES),
            "trend_delta_clip": 0.03,
            "trend_scale": 0.25,
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_fixed_policy": best,
            "best_min_skill": aggregate[best]["min_skill"],
            "gate_each_season_1000": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
            ),
            "gate_mean_1050": bool(aggregate[best]["mean_skill"] >= 1050.0),
            "adopt_for_full_fit": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
                and aggregate[best]["mean_skill"] >= 1050.0
            ),
        },
        "source_diagnostics": {
            str(season): model["diagnostics"]
            for season, model in source_models.items()
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "total_seconds": time.time() - started,
    }
    (ARTIFACT_DIR / "validation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"best={best} mean={aggregate[best]['mean_skill']:.2f} "
        f"min={aggregate[best]['min_skill']:.2f} "
        f"adopt={result['selection']['adopt_for_full_fit']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
