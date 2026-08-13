"""EXP-054: integrate a global physical-to-control model over past repertoire.

Exact-aligned past pitches train a shallow global model from pitch physics,
fine type, count and hands to official control_success.  For each outer fold,
the model scores only Trackman pitches through the previous season; those
scores are smoothed into pitcher x count x batter-hand expectations and mapped
to each validation row.  No current pitch measurement/type or test-row
aggregation is used.
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
from train_exp041_exact_game_trackman_sequence import (
    exact_aligned_rows,
    mapping_from_aligned,
)
from train_exp043_exact_pitchtype_control_eb import load_main


DATA_DIR = Path("./data")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
DIRECT_ROOT = Path("./artifacts/EXP-050/exact_dual_propensity_control")
ARTIFACT_DIR = Path("./artifacts/EXP-054/physical_control_integration")
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
)
CONTEXT_SMOOTHING = 100.0
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "integrated_w005",
    "integrated_w010",
    "integrated_w015",
    "direct010_integrated010",
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
        *PHYSICAL,
    ]
    frame = pd.read_csv(
        DATA_DIR / "trackman_history.csv", encoding="utf-8-sig", usecols=columns
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


def encoded_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[
        :,
        [
            *PHYSICAL,
            "velo_loss",
            "count_index",
            "pitcher_hand_code",
            "batter_hand_code",
        ],
    ].copy()
    fine = pd.get_dummies(
        pd.Categorical(frame["fine_pitch_type"], categories=FINE_TYPES),
        prefix="fine",
        dtype=np.int8,
    )
    return pd.concat([output.reset_index(drop=True), fine.reset_index(drop=True)], axis=1)


def aligned_features(aligned: pd.DataFrame) -> pd.DataFrame:
    work = aligned.copy()
    work["pitcher_hand_code"] = work["pitcher_hand"].astype(float)
    work["batter_hand_code"] = work["batter_hand"].astype(float)
    work["velo_loss"] = work["rel_speed"] - work["zone_speed"]
    return encoded_features(work)


def new_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=200,
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


def recent_base(season: int) -> np.ndarray:
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    return 0.5 * recency + 0.5 * aggressive


def expectation_correction(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    aligned: pd.DataFrame,
    validation_season: int,
) -> tuple[np.ndarray, dict[str, object]]:
    cutoff = validation_season - 1
    source = aligned.loc[aligned["season"].le(cutoff)].reset_index(drop=True)
    source_x = aligned_features(source)
    source_y = source["control_success"].to_numpy(float)
    season_counts = source["season"].value_counts()
    weight = np.array([1.0 / season_counts[value] for value in source["season"]])
    weight *= len(weight) / weight.sum()
    model = new_model()
    model.fit(source_x, source_y, sample_weight=weight)

    history = trackman.loc[trackman["season"].le(cutoff)].copy()
    history["predicted_control"] = model.predict(encoded_features(history))
    overall = history.groupby("pitcher_trackman_id")["predicted_control"].agg(
        ["mean", "count"]
    )
    keys = ["pitcher_trackman_id", "count_index", "batter_hand_code"]
    context = history.groupby(keys)["predicted_control"].agg(["sum", "count"])
    pitcher_index = context.index.get_level_values("pitcher_trackman_id")
    prior = overall["mean"].reindex(pitcher_index).to_numpy(float)
    context_rate = pd.Series(
        (context["sum"].to_numpy(float) + CONTEXT_SMOOTHING * prior)
        / (context["count"].to_numpy(float) + CONTEXT_SMOOTHING),
        index=context.index,
    )
    mapping, mapping_audit = mapping_from_aligned(aligned, cutoff)
    rows = main.loc[main["season"].eq(validation_season)].reset_index(drop=True)
    mapped = rows["pitcher_id"].map(mapping.mapping)
    query = pd.MultiIndex.from_arrays(
        [mapped, rows["count_index"], rows["batter_hand"]], names=keys
    )
    expected = context_rate.reindex(query).to_numpy(float)
    fallback = overall["mean"].reindex(mapped).to_numpy(float)
    expected = np.where(np.isfinite(expected), expected, fallback)
    official = rows["asof_pitcher_success_rate"].to_numpy(float)
    valid = mapped.notna().to_numpy() & np.isfinite(expected) & np.isfinite(official)
    correction = np.zeros(len(rows))
    correction[valid] = np.clip(
        expected[valid] - official[valid], -CORRECTION_CLIP, CORRECTION_CLIP
    )
    audit = {
        **mapping_audit,
        "aligned_fit_rows": len(source),
        "trackman_scored_rows": len(history),
        "feature_count": source_x.shape[1],
        "row_mapping_coverage": float(valid.mean()),
        "context_groups": len(context_rate),
        "season_equal_weight": True,
    }
    return correction, audit


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
        correction, audits[str(season)] = expectation_correction(
            main_frame, trackman, aligned, season
        )
        direct = (
            np.load(DIRECT_ROOT / f"predictions_pitcher_prop_w025_{season}.npy")
            - np.load(DIRECT_ROOT / f"predictions_base_{season}.npy")
        ) / 0.25
        predictions = {
            "base": base,
            "integrated_w005": np.clip(base + 0.05 * correction, 0.0, 1.0),
            "integrated_w010": np.clip(base + 0.10 * correction, 0.0, 1.0),
            "integrated_w015": np.clip(base + 0.15 * correction, 0.0, 1.0),
            "direct010_integrated010": np.clip(
                base + 0.10 * direct + 0.10 * correction, 0.0, 1.0
            ),
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
        "experiment": "EXP-054",
        "candidate_family": "global_physical_to_control_repertoire_integration",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "alignment_model_trackman_cutoff": "validation season-1",
            "current_fold_labels_used_for_fit_or_selection": False,
            "actual_current_pitch_type_or_physics_used": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "model": {
            "physical_columns": list(PHYSICAL),
            "fine_pitch_types": list(FINE_TYPES),
            "context_smoothing": CONTEXT_SMOOTHING,
            "correction_clip": CORRECTION_CLIP,
            "lightgbm": new_model().get_params(),
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
