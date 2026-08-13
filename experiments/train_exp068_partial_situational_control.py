"""EXP-068: partial-aligned situational pitch-type control EB.

Extends the existing pitcher x fine-type x count x batter-hand hierarchy with
past-only outs and inning-phase contexts.  The current pitch type remains
unknown and is integrated over a matching historical propensity hierarchy.
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
from train_exp041_exact_game_trackman_sequence import mapping_from_aligned
from train_exp043_exact_pitchtype_control_eb import load_main
from train_exp066_partial_sequence_alignment_control import (
    base_components,
    partial_aligned_rows,
)


DATA_DIR = Path("./data")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-068/partial_situational_control")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
PITCHER_SMOOTHING = 500.0
TYPE_SMOOTHING = 200.0
BASE_CONTEXT_SMOOTHING = 100.0
EXTENDED_CONTEXT_SMOOTHING = 150.0
PROPENSITY_SMOOTHING = 30.0
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "outs_direct_w010",
    "phase_direct_w010",
    "outs_phase_direct_w010",
    "exact_situational_blend_w010",
)


def inning_phase(values: pd.Series) -> pd.Series:
    return (
        pd.cut(values, bins=[0, 3, 6, 99], labels=False)
        .fillna(-1)
        .astype(np.int8)
    )


def load_trackman() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_trackman_id",
        "balls_before",
        "strikes_before",
        "outs_before",
        "inning",
        "batter_hand",
        "tagged_pitch_type",
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
    frame["inning_phase"] = inning_phase(frame["inning"])
    frame["fine_pitch_type"] = canonical_pitch_type(frame["tagged_pitch_type"])
    return frame


def hierarchy(
    aligned: pd.DataFrame,
    cutoff: int,
    extra: tuple[str, ...],
) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    history = aligned.loc[aligned["season"].le(cutoff)].copy()
    history["inning_phase"] = inning_phase(history["inning"])
    league = float(history["control_success"].mean())
    pitcher_stats = history.groupby("pitcher_trackman_id")["control_success"].agg(
        ["sum", "count"]
    )
    pitcher_rate = (
        pitcher_stats["sum"] + PITCHER_SMOOTHING * league
    ) / (pitcher_stats["count"] + PITCHER_SMOOTHING)
    type_keys = ["pitcher_trackman_id", "fine_pitch_type"]
    type_stats = history.groupby(type_keys)["control_success"].agg(["sum", "count"])
    type_prior = pitcher_rate.reindex(
        type_stats.index.get_level_values("pitcher_trackman_id")
    ).fillna(league).to_numpy(float)
    type_rate = pd.Series(
        (type_stats["sum"].to_numpy(float) + TYPE_SMOOTHING * type_prior)
        / (type_stats["count"].to_numpy(float) + TYPE_SMOOTHING),
        index=type_stats.index,
    )
    base_keys = [
        "pitcher_trackman_id",
        "fine_pitch_type",
        "count_index",
        "batter_hand",
    ]
    base_stats = history.groupby(base_keys)["control_success"].agg(["sum", "count"])
    base_type_index = pd.MultiIndex.from_arrays(
        [
            base_stats.index.get_level_values("pitcher_trackman_id"),
            base_stats.index.get_level_values("fine_pitch_type"),
        ],
        names=type_keys,
    )
    base_prior = type_rate.reindex(base_type_index).to_numpy(float)
    base_rate = pd.Series(
        (base_stats["sum"].to_numpy(float) + BASE_CONTEXT_SMOOTHING * base_prior)
        / (base_stats["count"].to_numpy(float) + BASE_CONTEXT_SMOOTHING),
        index=base_stats.index,
    )
    extended_keys = [*base_keys, *extra]
    extended_stats = history.groupby(extended_keys)["control_success"].agg(
        ["sum", "count"]
    )
    extended_base_index = pd.MultiIndex.from_arrays(
        [
            extended_stats.index.get_level_values(column)
            for column in base_keys
        ],
        names=base_keys,
    )
    extended_prior = base_rate.reindex(extended_base_index).to_numpy(float)
    extended_rate = pd.Series(
        (
            extended_stats["sum"].to_numpy(float)
            + EXTENDED_CONTEXT_SMOOTHING * extended_prior
        )
        / (extended_stats["count"].to_numpy(float) + EXTENDED_CONTEXT_SMOOTHING),
        index=extended_stats.index,
    )
    return pitcher_rate, type_rate, extended_rate, league


def propensity(
    trackman: pd.DataFrame,
    cutoff: int,
    extra: tuple[str, ...],
) -> pd.DataFrame:
    history = trackman.loc[trackman["season"].le(cutoff)]
    pitcher_counts = pd.crosstab(
        history["pitcher_trackman_id"], history["fine_pitch_type"]
    ).reindex(columns=FINE_TYPES, fill_value=0)
    pitcher_mix = pitcher_counts.div(
        pitcher_counts.sum(axis=1).replace(0, np.nan), axis=0
    )
    keys = [
        history["pitcher_trackman_id"],
        history["count_index"],
        history["batter_hand_code"],
        *[history[column] for column in extra],
    ]
    context_counts = pd.crosstab(keys, history["fine_pitch_type"]).reindex(
        columns=FINE_TYPES, fill_value=0
    )
    prior = pitcher_mix.reindex(
        context_counts.index.get_level_values("pitcher_trackman_id")
    ).to_numpy(float)
    counts = context_counts.to_numpy(float)
    probability = (
        counts + PROPENSITY_SMOOTHING * np.nan_to_num(prior)
    ) / (counts.sum(axis=1, keepdims=True) + PROPENSITY_SMOOTHING)
    return pd.DataFrame(probability, index=context_counts.index, columns=FINE_TYPES)


def correction(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    aligned: pd.DataFrame,
    season: int,
    extra: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    cutoff = season - 1
    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    rows["inning_phase"] = inning_phase(rows["inning"])
    mapping, audit = mapping_from_aligned(aligned, cutoff)
    mapped = rows["pitcher_id"].map(mapping.mapping)
    pitcher_rate, type_rate, context_rate, league = hierarchy(
        aligned, cutoff, extra
    )
    prop_table = propensity(trackman, cutoff, extra)
    prop_names = [
        "pitcher_trackman_id",
        "count_index",
        "batter_hand_code",
        *extra,
    ]
    prop_arrays = [mapped, rows["count_index"], rows["batter_hand"]]
    prop_arrays.extend(rows[column] for column in extra)
    query = pd.MultiIndex.from_arrays(prop_arrays, names=prop_names)
    weights = prop_table.reindex(query).to_numpy(float)
    valid_weights = np.isfinite(weights).all(axis=1)
    expected = np.zeros(len(rows), dtype=float)
    overall = pitcher_rate.reindex(mapped).fillna(league).to_numpy(float)
    context_names = [
        "pitcher_trackman_id",
        "fine_pitch_type",
        "count_index",
        "batter_hand",
        *extra,
    ]
    for position, pitch_type in enumerate(FINE_TYPES):
        arrays = [
            mapped,
            np.full(len(rows), pitch_type, dtype=object),
            rows["count_index"],
            rows["batter_hand"],
        ]
        arrays.extend(rows[column] for column in extra)
        context_index = pd.MultiIndex.from_arrays(arrays, names=context_names)
        type_index = pd.MultiIndex.from_arrays(
            [mapped, np.full(len(rows), pitch_type, dtype=object)],
            names=["pitcher_trackman_id", "fine_pitch_type"],
        )
        rate = context_rate.reindex(context_index).to_numpy(float)
        rate = np.where(
            np.isfinite(rate), rate, type_rate.reindex(type_index).to_numpy(float)
        )
        rate = np.where(np.isfinite(rate), rate, overall)
        expected += np.nan_to_num(weights[:, position]) * rate
    official = rows["asof_pitcher_success_rate"].to_numpy(float)
    valid = mapped.notna().to_numpy() & valid_weights & np.isfinite(expected)
    output = np.zeros(len(rows), dtype=float)
    output[valid] = np.clip(
        expected[valid] - official[valid], -CORRECTION_CLIP, CORRECTION_CLIP
    )
    return output, {
        **audit,
        "extra_context": list(extra),
        "row_mapping_coverage": float(valid.mean()),
        "propensity_contexts": len(prop_table),
        "control_contexts": len(context_rate),
    }


def main() -> None:
    started = time.time()
    main = load_main()
    # load_main lacks outs_before; preserve exact row order and append it.
    extras = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=["outs_before"],
    )
    main["outs_before"] = extras["outs_before"].to_numpy()
    trackman = load_trackman()
    aligned, alignment_audit = partial_aligned_rows()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    audits: dict[str, object] = {}
    specs = {
        "outs": ("outs_before",),
        "phase": ("inning_phase",),
        "outs_phase": ("outs_before", "inning_phase"),
    }
    for season in EVALUATED_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        recent, exact_correction = base_components(season)
        values = {}
        audits[str(season)] = {}
        for name, extra in specs.items():
            values[name], audits[str(season)][name] = correction(
                main, trackman, aligned, season, extra
            )
        base = np.clip(recent + 0.10 * exact_correction, 0, 1)
        situational_blend = (
            values["outs"] + values["phase"] + values["outs_phase"]
        ) / 3.0
        predictions = {
            "base": base,
            "outs_direct_w010": np.clip(recent + 0.10 * values["outs"], 0, 1),
            "phase_direct_w010": np.clip(recent + 0.10 * values["phase"], 0, 1),
            "outs_phase_direct_w010": np.clip(
                recent + 0.10 * values["outs_phase"], 0, 1
            ),
            "exact_situational_blend_w010": np.clip(
                recent + 0.05 * exact_correction + 0.05 * situational_blend,
                0,
                1,
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
        "experiment": "EXP-068",
        "candidate_family": "partial_aligned_situational_pitchtype_control",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "alignment_and_tables_cutoff": "validation season-1",
            "current_fold_labels_used_for_fit_or_selection": False,
            "actual_current_pitch_type_used": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "alignment_audit": alignment_audit,
        "fold_feature_audit": audits,
        "model": {
            "contexts": {name: list(value) for name, value in specs.items()},
            "smoothing": {
                "pitcher": PITCHER_SMOOTHING,
                "type": TYPE_SMOOTHING,
                "base_context": BASE_CONTEXT_SMOOTHING,
                "extended_context": EXTENDED_CONTEXT_SMOOTHING,
                "propensity": PROPENSITY_SMOOTHING,
            },
            "correction_clip": CORRECTION_CLIP,
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
