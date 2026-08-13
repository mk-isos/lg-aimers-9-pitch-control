"""EXP-056: adapt exact Trackman fine-type propensity to official current mix.

Historical fine-pitch propensities are rescaled so their fastball/breaking/
offspeed group totals match the current-row official asof pitch-mix rates,
with fixed reliability shrinkage from asof_pitcher_pitchmix_n.  No evaluation
row peers or actual current pitch type are used.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics
from train_exp041_exact_game_trackman_sequence import (
    exact_aligned_rows,
    mapping_from_aligned,
)
from train_exp043_exact_pitchtype_control_eb import (
    FINE_TYPES,
    load_main,
    load_trackman,
    posterior_tables,
    propensity_table,
)


LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
ARTIFACT_DIR = Path("./artifacts/EXP-056/current_pitchmix_adaptation")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
CORRECTION_CLIP = 0.03
CORRECTION_WEIGHT = 0.10
SMOOTHING_GRID = (0.0, 30.0, 100.0, 300.0)
CANDIDATES = tuple(f"mixadapt_k{int(value):03d}" for value in SMOOTHING_GRID)
GROUP_MEMBERS = {
    "fastball": ("fastball", "sinker", "cutter"),
    "breaking": ("slider", "curveball"),
    "offspeed": ("changeup", "splitter"),
}


def recent_base(season: int) -> np.ndarray:
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    return 0.5 * recency + 0.5 * aggressive


def rate_matrix(
    rows: pd.DataFrame,
    mapped: pd.Series,
    pitcher_rate: pd.Series,
    type_rate: pd.Series,
    context_rate: pd.Series,
    league: float,
) -> np.ndarray:
    matrix = np.empty((len(rows), len(FINE_TYPES)), dtype=np.float32)
    overall = pitcher_rate.reindex(mapped).fillna(league).to_numpy(float)
    for position, pitch_type in enumerate(FINE_TYPES):
        context_index = pd.MultiIndex.from_arrays(
            [
                mapped,
                np.full(len(rows), pitch_type, dtype=object),
                rows["count_index"],
                rows["batter_hand"],
            ],
            names=[
                "pitcher_trackman_id",
                "fine_pitch_type",
                "count_index",
                "batter_hand",
            ],
        )
        type_index = pd.MultiIndex.from_arrays(
            [mapped, np.full(len(rows), pitch_type, dtype=object)],
            names=["pitcher_trackman_id", "fine_pitch_type"],
        )
        rate = context_rate.reindex(context_index).to_numpy(float)
        fallback = type_rate.reindex(type_index).to_numpy(float)
        rate = np.where(np.isfinite(rate), rate, fallback)
        matrix[:, position] = np.where(np.isfinite(rate), rate, overall)
    return matrix


def adapted_weights(
    rows: pd.DataFrame,
    historical: np.ndarray,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    official = rows[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].to_numpy(float)
    official_sum = np.nansum(official, axis=1, keepdims=True)
    official_valid = np.isfinite(official).all(axis=1) & (
        np.abs(official_sum[:, 0] - 1.0) <= 2e-5
    )
    official = np.divide(
        official,
        official_sum,
        out=np.zeros_like(official),
        where=official_sum > 0,
    )
    group_positions = [
        [FINE_TYPES.index(value) for value in GROUP_MEMBERS[group]]
        for group in ("fastball", "breaking", "offspeed")
    ]
    historical_group = np.column_stack(
        [historical[:, positions].sum(axis=1) for positions in group_positions]
    )
    historical_group_sum = historical_group.sum(axis=1, keepdims=True)
    historical_group = np.divide(
        historical_group,
        historical_group_sum,
        out=np.full_like(historical_group, 1.0 / 3.0),
        where=historical_group_sum > 0,
    )
    sample_n = rows["asof_pitcher_pitchmix_n"].to_numpy(float)
    reliability = np.divide(
        sample_n,
        sample_n + smoothing,
        out=np.zeros(len(rows)),
        where=np.isfinite(sample_n) & ((sample_n + smoothing) > 0),
    )
    reliability[~official_valid] = 0.0
    target_group = (
        reliability[:, None] * official
        + (1.0 - reliability[:, None]) * historical_group
    )
    adapted = np.zeros_like(historical)
    for group_position, positions in enumerate(group_positions):
        within = historical[:, positions]
        denominator = within.sum(axis=1, keepdims=True)
        fallback = np.full_like(within, 1.0 / len(positions))
        conditional = np.divide(
            within,
            denominator,
            out=fallback,
            where=denominator > 0,
        )
        adapted[:, positions] = conditional * target_group[:, [group_position]]
    total = adapted.sum(axis=1, keepdims=True)
    adapted = np.divide(adapted, total, out=historical.copy(), where=total > 0)
    return adapted, reliability


def corrections(
    main: pd.DataFrame,
    aligned: pd.DataFrame,
    trackman: pd.DataFrame,
    season: int,
) -> tuple[dict[float, np.ndarray], dict[str, object]]:
    cutoff = season - 1
    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    mapping, mapping_audit = mapping_from_aligned(aligned, cutoff)
    mapped = rows["pitcher_id"].map(mapping.mapping)
    propensity = propensity_table(trackman, cutoff)
    query = pd.MultiIndex.from_arrays(
        [mapped, rows["count_index"], rows["batter_hand"]],
        names=["pitcher_trackman_id", "count_index", "batter_hand_code"],
    )
    historical = propensity.reindex(query).to_numpy(float)
    pitcher_rate, _, type_rate, context_rate, league = posterior_tables(
        aligned, cutoff
    )
    rates = rate_matrix(
        rows, mapped, pitcher_rate, type_rate, context_rate, league
    )
    official_success = rows["asof_pitcher_success_rate"].to_numpy(float)
    historical_valid = np.isfinite(historical).all(axis=1)
    output: dict[float, np.ndarray] = {}
    reliability_audit: dict[str, object] = {}
    for smoothing in SMOOTHING_GRID:
        weights, reliability = adapted_weights(rows, historical, smoothing)
        expected = np.sum(np.nan_to_num(weights) * rates, axis=1)
        valid = (
            mapped.notna().to_numpy()
            & historical_valid
            & np.isfinite(expected)
            & np.isfinite(official_success)
        )
        correction = np.zeros(len(rows))
        correction[valid] = np.clip(
            expected[valid] - official_success[valid],
            -CORRECTION_CLIP,
            CORRECTION_CLIP,
        )
        output[smoothing] = correction
        reliability_audit[str(int(smoothing))] = {
            "mean": float(reliability.mean()),
            "nonzero_fraction": float((reliability > 0).mean()),
        }
    return output, {
        **mapping_audit,
        "row_mapping_propensity_coverage": float(
            (mapped.notna().to_numpy() & historical_valid).mean()
        ),
        "propensity_contexts": len(propensity),
        "reliability": reliability_audit,
    }


def main() -> None:
    started = time.time()
    main_frame = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = exact_aligned_rows()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        base = recent_base(season)
        values, audits[str(season)] = corrections(
            main_frame, aligned, trackman, season
        )
        predictions = {"base": base}
        for smoothing, correction in values.items():
            name = f"mixadapt_k{int(smoothing):03d}"
            predictions[name] = np.clip(
                base + CORRECTION_WEIGHT * correction, 0.0, 1.0
            )
        fold: dict[str, object] = {
            "validation_season": season,
            "history_cutoff": season - 1,
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(target, prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{season}.npy", prediction)
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", target.astype(np.int8))
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
        "experiment": "EXP-056",
        "candidate_family": "current_official_pitchmix_adapted_exact_trackman_control",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "mapping_trackman_cutoff": "validation season-1",
            "current_mix": "official current-row asof features only",
            "current_fold_labels_used_for_fit_or_selection": False,
            "actual_current_pitch_type_used": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "model": {
            "fine_pitch_types": list(FINE_TYPES),
            "group_members": {key: list(value) for key, value in GROUP_MEMBERS.items()},
            "pitchmix_smoothing_grid": list(SMOOTHING_GRID),
            "correction_weight": CORRECTION_WEIGHT,
            "correction_clip": CORRECTION_CLIP,
        },
        "exact_alignment": alignment_audit,
        "fold_feature_audit": audits,
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
