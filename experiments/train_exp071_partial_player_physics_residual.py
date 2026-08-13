"""EXP-071: partial-aligned pitcher-physics residual integration.

Unlike EXP-070, the supervised target is the strict EXP-051 OOF residual.
Only prior-season high-confidence aligned TrackMan rows are used for fitting.
The fitted residual model scores historical TrackMan pitches, and those scores
are integrated by mapped pitcher, count and batter hand for the current row.
No current pitch type or current physical measurement is required.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics
from train_exp041_exact_game_trackman_sequence import mapping_from_aligned
from train_exp043_exact_pitchtype_control_eb import load_main
from train_exp066_partial_sequence_alignment_control import (
    base_components,
    partial_aligned_rows,
)
from train_exp070_partial_player_physics_integration import (
    CONTEXT_SMOOTHING,
    add_normalized_physics,
    encoded,
    load_trackman,
    new_model,
)


LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-071/partial_player_physics_residual")
EVALUATED_SEASONS = (2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "playerphys_resid_w025",
    "playerphys_resid_w050",
    "playerphys_resid_w100",
)


def exp051_base(season: int) -> np.ndarray:
    recent, exact_correction = base_components(season)
    return np.clip(recent + 0.10 * exact_correction, 0.0, 1.0)


def attach_oof_residual(
    aligned: pd.DataFrame,
    cutoff: int,
    base_cache: dict[int, np.ndarray],
) -> pd.DataFrame:
    source = aligned.loc[
        aligned["season"].le(cutoff) & aligned["season"].isin(base_cache)
    ].copy()
    base = np.empty(len(source), dtype=float)
    for season in sorted(source["season"].unique()):
        season = int(season)
        mask = source["season"].eq(season).to_numpy()
        positions = source.loc[mask, "official_season_row_index"].to_numpy(int)
        candidate = base_cache[season]
        if positions.max(initial=-1) >= len(candidate):
            raise ValueError(f"official season-row index overflow in {season}")
        base[mask] = candidate[positions]
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        observed = source.loc[mask, "control_success"].to_numpy(float)
        if not np.array_equal(observed, target[positions]):
            raise ValueError(f"aligned target mismatch in {season}")
    source["oof_base"] = base
    source["oof_residual"] = source["control_success"].to_numpy(float) - base
    source["centered_residual"] = source["oof_residual"] - source.groupby(
        "season", sort=False
    )["oof_residual"].transform("mean")
    return source


def expected_residual(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    aligned: pd.DataFrame,
    season: int,
    base_cache: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    cutoff = season - 1
    source = attach_oof_residual(aligned, cutoff, base_cache)
    history = trackman.loc[trackman["season"].le(cutoff)].copy()
    source["pitcher_hand_code"] = source["pitcher_hand"].astype(np.int8)
    source["batter_hand_code"] = source["batter_hand"].astype(np.int8)
    source["velo_loss"] = source["rel_speed"] - source["zone_speed"]
    source = add_normalized_physics(source, history)
    history = add_normalized_physics(history, history)
    pitcher_categories = pd.Index(
        np.sort(source["pitcher_trackman_id"].dropna().unique())
    )
    source_x = encoded(source, pitcher_categories)
    history_x = encoded(history, pitcher_categories)
    season_counts = source["season"].value_counts()
    sample_weight = np.array(
        [1.0 / season_counts[value] for value in source["season"]], dtype=float
    )
    sample_weight *= len(sample_weight) / sample_weight.sum()
    model = new_model()
    model.fit(
        source_x,
        source["centered_residual"].to_numpy(float),
        sample_weight=sample_weight,
        categorical_feature=["pitcher_code"],
    )
    history["predicted_residual"] = np.clip(
        model.predict(history_x), -CORRECTION_CLIP, CORRECTION_CLIP
    )
    overall = history.groupby("pitcher_trackman_id")["predicted_residual"].agg(
        ["mean", "count"]
    )
    keys = ["pitcher_trackman_id", "count_index", "batter_hand_code"]
    context = history.groupby(keys)["predicted_residual"].agg(["sum", "count"])
    prior = overall["mean"].reindex(
        context.index.get_level_values("pitcher_trackman_id")
    ).to_numpy(float)
    context_rate = pd.Series(
        (context["sum"].to_numpy(float) + CONTEXT_SMOOTHING * prior)
        / (context["count"].to_numpy(float) + CONTEXT_SMOOTHING),
        index=context.index,
    )
    mapping, mapping_audit = mapping_from_aligned(aligned, cutoff)
    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    mapped = rows["pitcher_id"].map(mapping.mapping)
    query = pd.MultiIndex.from_arrays(
        [mapped, rows["count_index"], rows["batter_hand"]], names=keys
    )
    correction = context_rate.reindex(query).to_numpy(float)
    fallback = overall["mean"].reindex(mapped).to_numpy(float)
    correction = np.where(np.isfinite(correction), correction, fallback)
    valid = mapped.notna().to_numpy() & np.isfinite(correction)
    output = np.zeros(len(rows), dtype=float)
    output[valid] = np.clip(
        correction[valid], -CORRECTION_CLIP, CORRECTION_CLIP
    )
    importance = dict(
        zip(source_x.columns, model.booster_.feature_importance("gain"), strict=True)
    )
    top = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:20]
    centered_means = source.groupby("season")["centered_residual"].mean().abs()
    return output, {
        **mapping_audit,
        "aligned_fit_rows": len(source),
        "trackman_scored_rows": len(history),
        "mapped_pitcher_categories": len(pitcher_categories),
        "feature_count": source_x.shape[1],
        "row_mapping_coverage": float(valid.mean()),
        "context_groups": len(context_rate),
        "source_residual_center_max_abs": float(centered_means.max()),
        "source_season_equal_weight": True,
        "top_gain_features": {name: float(value) for name, value in top},
    }


def main() -> None:
    started = time.time()
    main = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = partial_aligned_rows()
    # Stored OOF components begin in 2021; every validation fold remains strict.
    base_cache = {season: exp051_base(season) for season in range(2021, 2025)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    audits: dict[str, object] = {}
    weights = {
        "playerphys_resid_w025": 0.25,
        "playerphys_resid_w050": 0.50,
        "playerphys_resid_w100": 1.00,
    }
    for season in EVALUATED_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        base = base_cache[season]
        correction, audits[str(season)] = expected_residual(
            main, trackman, aligned, season, base_cache
        )
        predictions = {"base": base}
        predictions.update(
            {
                name: np.clip(base + weight * correction, 0.0, 1.0)
                for name, weight in weights.items()
            }
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
        }
    best = max(
        CANDIDATES,
        key=lambda name: (aggregate[name]["min_skill"], aggregate[name]["mean_skill"]),
    )
    result = {
        "experiment": "EXP-071",
        "candidate_family": "partial_aligned_pitcher_physics_oof_residual",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "source_seasons_strictly_prior": True,
            "source_oof_base": "EXP-051 trackman_direct_recent_w010",
            "source_residual_season_centered": True,
            "actual_current_pitch_type_or_physics_used": False,
            "current_fold_labels_used_for_fit_or_selection": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "alignment_audit": alignment_audit,
        "fold_feature_audit": audits,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_fixed_candidate": best,
            "best_min_skill": aggregate[best]["min_skill"],
            "best_mean_skill": aggregate[best]["mean_skill"],
            "gate_each_season_1000": bool(aggregate[best]["min_skill"] >= 1000.0),
            "gate_mean_1100": bool(aggregate[best]["mean_skill"] >= 1100.0),
            "adopt": bool(
                aggregate[best]["min_skill"] >= 1000.0
                and aggregate[best]["mean_skill"] >= 1100.0
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
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
