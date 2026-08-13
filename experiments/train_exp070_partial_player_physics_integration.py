"""EXP-070: partial-aligned pitcher-specific normalized physics integration.

A shallow LightGBM fits past official control labels from high-confidence
partial alignment using pitcher identity, fine pitch type, count/hands and
source-season-relative TrackMan physics.  It then scores only historical
TrackMan pitches and aggregates those scores by pitcher/count/batter hand.
The unknown current pitch type and current physical measurement are never used.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from train_exp017_rolling_residual import calculate_metrics
from train_exp033_trackman_sequence_trend import FINE_TYPES, canonical_pitch_type
from train_exp041_exact_game_trackman_sequence import mapping_from_aligned
from train_exp043_exact_pitchtype_control_eb import load_main
from train_exp066_partial_sequence_alignment_control import (
    base_components,
    partial_aligned_rows,
)


DATA_DIR = Path("./data")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-070/partial_player_physics_integration")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
PHYSICAL = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
    "velo_loss",
)
CONTEXT_SMOOTHING = 100.0
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "playerphys_w005",
    "playerphys_w010",
    "playerphys_w015",
)


def load_trackman() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_trackman_id",
        "balls_before",
        "strikes_before",
        "pitcher_hand",
        "batter_hand",
        "tagged_pitch_type",
        "rel_speed",
        "spin_rate",
        "induced_vert_break",
        "horz_break",
        "extension",
        "rel_height",
        "rel_side",
        "zone_speed",
    ]
    frame = pd.read_csv(
        DATA_DIR / "trackman_history.csv",
        encoding="utf-8-sig",
        usecols=columns,
    )
    frame["count_index"] = (
        4 * frame["balls_before"] + frame["strikes_before"]
    ).astype(np.int8)
    frame["pitcher_hand_code"] = frame["pitcher_hand"].map(
        {"Left": 1, "Right": 2}
    )
    frame["batter_hand_code"] = frame["batter_hand"].map(
        {"Left": 1, "Right": 2}
    )
    frame["fine_pitch_type"] = canonical_pitch_type(frame["tagged_pitch_type"])
    frame["velo_loss"] = frame["rel_speed"] - frame["zone_speed"]
    return frame


def add_normalized_physics(
    source: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    output = source.copy()
    keys = ["season", "fine_pitch_type", "pitcher_hand_code"]
    stats = reference.groupby(keys, observed=True)[list(PHYSICAL)].agg(
        ["median", "std"]
    )
    for metric in PHYSICAL:
        center = stats[(metric, "median")]
        spread = stats[(metric, "std")].replace(0.0, np.nan)
        index = pd.MultiIndex.from_frame(output[keys])
        output[f"z_{metric}"] = np.clip(
            (output[metric].to_numpy(float) - center.reindex(index).to_numpy(float))
            / spread.reindex(index).to_numpy(float),
            -5.0,
            5.0,
        )
    return output


def encoded(
    frame: pd.DataFrame,
    pitcher_categories: pd.Index,
) -> pd.DataFrame:
    pitcher_map = pd.Series(
        np.arange(len(pitcher_categories), dtype=np.int32),
        index=pitcher_categories,
    )
    output = frame[[f"z_{metric}" for metric in PHYSICAL]].copy()
    output["pitcher_code"] = (
        frame["pitcher_trackman_id"].map(pitcher_map).fillna(-1).astype(np.int32)
    )
    output["count_index"] = frame["count_index"].astype(np.int8)
    output["pitcher_hand_code"] = frame["pitcher_hand_code"].astype(np.int8)
    output["batter_hand_code"] = frame["batter_hand_code"].astype(np.int8)
    fine = pd.get_dummies(
        pd.Categorical(frame["fine_pitch_type"], categories=FINE_TYPES),
        prefix="fine",
        dtype=np.int8,
    )
    return pd.concat([output.reset_index(drop=True), fine.reset_index(drop=True)], axis=1)


def new_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=250,
        learning_rate=0.015,
        num_leaves=15,
        min_child_samples=3000,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=15.0,
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def expected_correction(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    aligned: pd.DataFrame,
    season: int,
) -> tuple[np.ndarray, dict[str, object]]:
    cutoff = season - 1
    source = aligned.loc[aligned["season"].le(cutoff)].copy()
    history = trackman.loc[trackman["season"].le(cutoff)].copy()
    # Aligned hands use official codes; convert to TrackMan-style codes explicitly.
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
    source_y = source["control_success"].to_numpy(float)
    season_counts = source["season"].value_counts()
    sample_weight = np.array(
        [1.0 / season_counts[value] for value in source["season"]]
    )
    sample_weight *= len(sample_weight) / sample_weight.sum()
    model = new_model()
    model.fit(
        source_x,
        source_y,
        sample_weight=sample_weight,
        categorical_feature=["pitcher_code"],
    )
    history["predicted_control"] = model.predict(history_x)
    overall = history.groupby("pitcher_trackman_id")["predicted_control"].agg(
        ["mean", "count"]
    )
    keys = ["pitcher_trackman_id", "count_index", "batter_hand_code"]
    context = history.groupby(keys)["predicted_control"].agg(["sum", "count"])
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
    expected = context_rate.reindex(query).to_numpy(float)
    fallback = overall["mean"].reindex(mapped).to_numpy(float)
    expected = np.where(np.isfinite(expected), expected, fallback)
    official = rows["asof_pitcher_success_rate"].to_numpy(float)
    valid = mapped.notna().to_numpy() & np.isfinite(expected) & np.isfinite(official)
    correction = np.zeros(len(rows), dtype=float)
    correction[valid] = np.clip(
        expected[valid] - official[valid], -CORRECTION_CLIP, CORRECTION_CLIP
    )
    importance = dict(
        zip(source_x.columns, model.booster_.feature_importance("gain"), strict=True)
    )
    top = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:20]
    return correction, {
        **mapping_audit,
        "aligned_fit_rows": len(source),
        "trackman_scored_rows": len(history),
        "mapped_pitcher_categories": len(pitcher_categories),
        "feature_count": source_x.shape[1],
        "row_mapping_coverage": float(valid.mean()),
        "context_groups": len(context_rate),
        "season_equal_weight": True,
        "top_gain_features": {name: float(value) for name, value in top},
    }


def main() -> None:
    started = time.time()
    main = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = partial_aligned_rows()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        recent, exact_correction = base_components(season)
        base = np.clip(recent + 0.10 * exact_correction, 0, 1)
        correction, audits[str(season)] = expected_correction(
            main, trackman, aligned, season
        )
        predictions = {
            "base": base,
            "playerphys_w005": np.clip(base + 0.05 * correction, 0, 1),
            "playerphys_w010": np.clip(base + 0.10 * correction, 0, 1),
            "playerphys_w015": np.clip(base + 0.15 * correction, 0, 1),
        }
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
        "experiment": "EXP-070",
        "candidate_family": "partial_aligned_pitcher_specific_normalized_physics",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "alignment_model_trackman_cutoff": "validation season-1",
            "source_season_equal_weight": True,
            "current_fold_labels_used_for_fit_or_selection": False,
            "actual_current_pitch_type_or_physics_used": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "alignment_audit": alignment_audit,
        "model": {
            "physical_columns": list(PHYSICAL),
            "normalization": "source season x fine pitch type x pitcher hand",
            "categorical": ["pitcher_trackman_id"],
            "correction_clip": CORRECTION_CLIP,
            "lightgbm": new_model().get_params(),
        },
        "fold_feature_audit": audits,
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
