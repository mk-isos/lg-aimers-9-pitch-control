"""EXP-062: exact TrackMan tagged/automatic joint pitch-type control.

The unknown current pitch type remains integrated out.  Historical TrackMan
joint propensities for (tagged fine type, automatic fine type) weight a
past-only hierarchical control table fit on exact-aligned official labels.
This tests whether pitch-classification disagreement adds information beyond
the existing tagged-only control correction.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
TAG_ROOT = Path("./artifacts/EXP-043/exact_pitchtype_control_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-062/tag_auto_joint_control")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
PITCHER_SMOOTHING = 500.0
TAG_SMOOTHING = 200.0
JOINT_SMOOTHING = 100.0
CONTEXT_SMOOTHING = 100.0
PROPENSITY_SMOOTHING = 20.0
CORRECTION_CLIP = 0.03
JOINT_TYPES = tuple(
    f"{tag}|{auto}" for tag in FINE_TYPES for auto in FINE_TYPES
)
CANDIDATES = (
    "joint_direct_w010",
    "tag_joint_blend_w010",
    "tag010_joint_delta_w025",
)


def load_trackman() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_trackman_id",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "tagged_pitch_type",
        "auto_pitch_type",
    ]
    frame = pd.read_csv(
        DATA_DIR / "trackman_history.csv",
        encoding="utf-8-sig",
        usecols=columns,
    )
    frame["count_index"] = (
        4 * frame["balls_before"] + frame["strikes_before"]
    ).astype(np.int8)
    frame["batter_hand_code"] = frame["batter_hand"].map(
        {"Left": 1, "Right": 2}
    )
    frame["fine_pitch_type"] = canonical_pitch_type(frame["tagged_pitch_type"])
    frame["auto_fine_type"] = canonical_pitch_type(frame["auto_pitch_type"])
    frame["joint_type"] = (
        frame["fine_pitch_type"].astype(str)
        + "|"
        + frame["auto_fine_type"].astype(str)
    )
    return frame


def add_joint_to_aligned(aligned: pd.DataFrame) -> pd.DataFrame:
    output = aligned.copy()
    output["joint_type"] = (
        output["fine_pitch_type"].astype(str)
        + "|"
        + output["auto_fine_type"].astype(str)
    )
    return output


def posterior_tables(
    aligned: pd.DataFrame,
    cutoff: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, float]:
    history = aligned.loc[aligned["season"].le(cutoff)]
    y = history["control_success"].astype(float)
    league = float(y.mean())
    pitcher_stats = history.groupby("pitcher_trackman_id")["control_success"].agg(
        ["sum", "count"]
    )
    pitcher_rate = (
        pitcher_stats["sum"] + PITCHER_SMOOTHING * league
    ) / (pitcher_stats["count"] + PITCHER_SMOOTHING)

    tag_keys = ["pitcher_trackman_id", "fine_pitch_type"]
    tag_stats = history.groupby(tag_keys)["control_success"].agg(["sum", "count"])
    tag_prior = pitcher_rate.reindex(
        tag_stats.index.get_level_values("pitcher_trackman_id")
    ).fillna(league).to_numpy(float)
    tag_rate = pd.Series(
        (tag_stats["sum"].to_numpy(float) + TAG_SMOOTHING * tag_prior)
        / (tag_stats["count"].to_numpy(float) + TAG_SMOOTHING),
        index=tag_stats.index,
    )

    joint_keys = ["pitcher_trackman_id", "joint_type"]
    joint_stats = history.groupby(joint_keys)["control_success"].agg(
        ["sum", "count"]
    )
    joint_tag_index = pd.MultiIndex.from_arrays(
        [
            joint_stats.index.get_level_values("pitcher_trackman_id"),
            joint_stats.index.get_level_values("joint_type").str.split("|").str[0],
        ],
        names=tag_keys,
    )
    joint_prior = tag_rate.reindex(joint_tag_index).to_numpy(float)
    joint_rate = pd.Series(
        (joint_stats["sum"].to_numpy(float) + JOINT_SMOOTHING * joint_prior)
        / (joint_stats["count"].to_numpy(float) + JOINT_SMOOTHING),
        index=joint_stats.index,
    )

    context_keys = [
        "pitcher_trackman_id",
        "joint_type",
        "count_index",
        "batter_hand",
    ]
    context_stats = history.groupby(context_keys)["control_success"].agg(
        ["sum", "count"]
    )
    context_joint_index = pd.MultiIndex.from_arrays(
        [
            context_stats.index.get_level_values("pitcher_trackman_id"),
            context_stats.index.get_level_values("joint_type"),
        ],
        names=joint_keys,
    )
    context_prior = joint_rate.reindex(context_joint_index).to_numpy(float)
    context_rate = pd.Series(
        (context_stats["sum"].to_numpy(float) + CONTEXT_SMOOTHING * context_prior)
        / (context_stats["count"].to_numpy(float) + CONTEXT_SMOOTHING),
        index=context_stats.index,
    )
    return pitcher_rate, tag_rate, joint_rate, context_rate, league


def propensity(trackman: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    history = trackman.loc[trackman["season"].le(cutoff)]
    pitcher_counts = pd.crosstab(
        history["pitcher_trackman_id"], history["joint_type"]
    ).reindex(columns=JOINT_TYPES, fill_value=0)
    pitcher_mix = pitcher_counts.div(
        pitcher_counts.sum(axis=1).replace(0, np.nan), axis=0
    )
    context_counts = pd.crosstab(
        [
            history["pitcher_trackman_id"],
            history["count_index"],
            history["batter_hand_code"],
        ],
        history["joint_type"],
    ).reindex(columns=JOINT_TYPES, fill_value=0)
    prior = pitcher_mix.reindex(
        context_counts.index.get_level_values("pitcher_trackman_id")
    ).to_numpy(float)
    counts = context_counts.to_numpy(float)
    probability = (
        counts + PROPENSITY_SMOOTHING * np.nan_to_num(prior)
    ) / (counts.sum(axis=1, keepdims=True) + PROPENSITY_SMOOTHING)
    return pd.DataFrame(probability, index=context_counts.index, columns=JOINT_TYPES)


def build_joint_correction(
    main: pd.DataFrame,
    aligned: pd.DataFrame,
    trackman: pd.DataFrame,
    season: int,
) -> tuple[np.ndarray, dict[str, object]]:
    cutoff = season - 1
    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    mapping, audit = mapping_from_aligned(aligned, cutoff)
    mapped = rows["pitcher_id"].map(mapping.mapping)
    pitcher_rate, tag_rate, joint_rate, context_rate, league = posterior_tables(
        aligned, cutoff
    )
    prop_table = propensity(trackman, cutoff)
    query = pd.MultiIndex.from_arrays(
        [mapped, rows["count_index"], rows["batter_hand"]],
        names=["pitcher_trackman_id", "count_index", "batter_hand_code"],
    )
    weights = prop_table.reindex(query).to_numpy(float)
    valid_weights = np.isfinite(weights).all(axis=1)
    expected = np.zeros(len(rows), dtype=float)
    rate_matrix = np.empty((len(rows), len(JOINT_TYPES)), dtype=np.float32)
    overall = pitcher_rate.reindex(mapped).fillna(league).to_numpy(float)
    for position, joint_type in enumerate(JOINT_TYPES):
        tag_type = joint_type.split("|", 1)[0]
        context_index = pd.MultiIndex.from_arrays(
            [
                mapped,
                np.full(len(rows), joint_type, dtype=object),
                rows["count_index"],
                rows["batter_hand"],
            ],
            names=[
                "pitcher_trackman_id",
                "joint_type",
                "count_index",
                "batter_hand",
            ],
        )
        joint_index = pd.MultiIndex.from_arrays(
            [mapped, np.full(len(rows), joint_type, dtype=object)],
            names=["pitcher_trackman_id", "joint_type"],
        )
        tag_index = pd.MultiIndex.from_arrays(
            [mapped, np.full(len(rows), tag_type, dtype=object)],
            names=["pitcher_trackman_id", "fine_pitch_type"],
        )
        rate = context_rate.reindex(context_index).to_numpy(float)
        rate = np.where(
            np.isfinite(rate), rate, joint_rate.reindex(joint_index).to_numpy(float)
        )
        rate = np.where(
            np.isfinite(rate), rate, tag_rate.reindex(tag_index).to_numpy(float)
        )
        rate = np.where(np.isfinite(rate), rate, overall)
        rate_matrix[:, position] = rate
        expected += np.nan_to_num(weights[:, position]) * rate
    valid = mapped.notna().to_numpy() & valid_weights & np.isfinite(expected)
    official = rows["asof_pitcher_success_rate"].to_numpy(float)
    correction = np.zeros(len(rows), dtype=float)
    correction[valid] = np.clip(
        expected[valid] - official[valid], -CORRECTION_CLIP, CORRECTION_CLIP
    )
    disagreement = np.array(
        [joint.split("|")[0] != joint.split("|")[1] for joint in JOINT_TYPES]
    )
    disagreement_probability = np.nansum(weights[:, disagreement], axis=1)
    return correction, {
        **audit,
        "row_mapping_coverage": float(valid.mean()),
        "joint_propensity_contexts": len(prop_table),
        "observed_joint_types": int(
            trackman.loc[trackman["season"].le(cutoff), "joint_type"].nunique()
        ),
        "mean_disagreement_probability": float(
            np.nanmean(disagreement_probability[valid_weights])
        ),
    }


def recent_components(season: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    strict = np.load(
        LOWRANK_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
    ).astype(float)
    tagged = np.load(TAG_ROOT / f"predictions_fine_direct_w025_{season}.npy").astype(float)
    return 0.5 * recency + 0.5 * aggressive, strict, (tagged - strict) / 0.25


def main() -> None:
    started = time.time()
    main = load_main()
    trackman = load_trackman()
    aligned_raw, alignment_audit = exact_aligned_rows()
    aligned = add_joint_to_aligned(aligned_raw)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        recent, _, tag_correction = recent_components(season)
        joint_correction, audits[str(season)] = build_joint_correction(
            main, aligned, trackman, season
        )
        base = np.clip(recent + 0.10 * tag_correction, 0, 1)
        predictions = {
            "base": base,
            "joint_direct_w010": np.clip(recent + 0.10 * joint_correction, 0, 1),
            "tag_joint_blend_w010": np.clip(
                recent + 0.05 * tag_correction + 0.05 * joint_correction, 0, 1
            ),
            "tag010_joint_delta_w025": np.clip(
                base + 0.25 * (joint_correction - tag_correction), 0, 1
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
        "experiment": "EXP-062",
        "candidate_family": "exact_tagged_auto_joint_pitchtype_control",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "history_mapping_and_tables_cutoff": "validation season-1",
            "current_fold_labels_used_for_fit_or_selection": False,
            "actual_current_pitch_type_used": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "model": {
            "fine_pitch_types": list(FINE_TYPES),
            "joint_type_count": len(JOINT_TYPES),
            "smoothing": {
                "pitcher": PITCHER_SMOOTHING,
                "tag": TAG_SMOOTHING,
                "joint": JOINT_SMOOTHING,
                "context": CONTEXT_SMOOTHING,
                "propensity": PROPENSITY_SMOOTHING,
            },
            "correction_clip": CORRECTION_CLIP,
        },
        "exact_alignment": alignment_audit,
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
