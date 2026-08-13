"""EXP-059: season-normalized TrackMan arsenal geometry residual.

Raw TrackMan units drift between seasons.  This experiment first summarizes
each pitcher/pitch-type/source-season, then normalizes physical means and
dispersions against the same season, fine pitch type and pitcher hand.  The
row feature is a past-only, equal-source-season expected arsenal profile under
the current row's historical pitch-type propensity.  Current pitch type and
current TrackMan measurements are never used.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_exp017_rolling_residual import calculate_metrics
from train_exp033_trackman_sequence_trend import FINE_TYPES, canonical_pitch_type
from train_exp041_exact_game_trackman_sequence import (
    exact_aligned_rows,
    mapping_from_aligned,
)
from train_exp043_exact_pitchtype_control_eb import load_main, propensity_table


DATA_DIR = Path("./data")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
TRACKMAN_CONTROL_ROOT = Path("./artifacts/EXP-043/exact_pitchtype_control_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-059/normalized_arsenal_geometry")
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
ALPHAS = (1000.0, 10000.0)
WEIGHTS = (0.025, 0.05)
CORRECTION_CLIP = 0.03


def load_trackman() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_trackman_id",
        "pitcher_hand",
        "balls_before",
        "strikes_before",
        "batter_hand",
        "tagged_pitch_type",
        "auto_pitch_type",
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
    frame["batter_hand_code"] = frame["batter_hand"].map(
        {"Left": 1, "Right": 2}
    )
    frame["fine_pitch_type"] = canonical_pitch_type(
        frame["tagged_pitch_type"]
    )
    frame["auto_fine_pitch_type"] = canonical_pitch_type(
        frame["auto_pitch_type"]
    )
    frame["tag_auto_agree"] = frame["fine_pitch_type"].eq(
        frame["auto_fine_pitch_type"]
    ).astype(np.int8)
    frame["velo_loss"] = frame["rel_speed"] - frame["zone_speed"]
    return frame


def source_profiles(trackman: pd.DataFrame) -> dict[int, pd.DataFrame]:
    group_keys = [
        "season",
        "pitcher_trackman_id",
        "pitcher_hand",
        "fine_pitch_type",
    ]
    grouped = trackman.groupby(group_keys, observed=True, sort=True)
    summary = grouped[list(PHYSICAL)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["pitch_n"] = grouped.size()
    summary["tag_auto_agree"] = grouped["tag_auto_agree"].mean()
    summary = summary.reset_index()
    value_columns = [
        column
        for column in summary
        if column not in {*group_keys, "pitch_n"}
    ]
    peers = ["season", "pitcher_hand", "fine_pitch_type"]
    for column in value_columns:
        peer = summary.groupby(peers, observed=True)[column]
        center = peer.transform("median")
        spread = peer.transform("std").replace(0.0, np.nan)
        summary[f"z_{column}"] = np.clip(
            (summary[column] - center) / spread, -5.0, 5.0
        )
    keep = [
        "pitcher_trackman_id",
        "pitcher_hand",
        "fine_pitch_type",
        "pitch_n",
        *[f"z_{column}" for column in value_columns],
    ]
    return {
        int(season): rows[keep].set_index(
            ["pitcher_trackman_id", "fine_pitch_type", "pitcher_hand"]
        )
        for season, rows in summary.groupby("season", sort=True)
    }


def expected_source(
    rows: pd.DataFrame,
    mapped: pd.Series,
    propensity: pd.DataFrame,
    profile: pd.DataFrame,
    source_season: int,
) -> pd.DataFrame:
    query = pd.MultiIndex.from_arrays(
        [mapped, rows["count_index"], rows["batter_hand"]],
        names=["pitcher_trackman_id", "count_index", "batter_hand_code"],
    )
    weights = propensity.reindex(query).to_numpy(dtype=float)
    valid_propensity = np.isfinite(weights).all(axis=1)
    z_columns = [column for column in profile if column.startswith("z_")]
    pitcher_hand = rows["pitcher_hand"].map({1: "Left", 2: "Right"})
    output: dict[str, np.ndarray] = {}
    pitch_n = np.zeros((len(rows), len(FINE_TYPES)), dtype=float)
    for position, pitch_type in enumerate(FINE_TYPES):
        index = pd.MultiIndex.from_arrays(
            [
                mapped,
                np.full(len(rows), pitch_type, dtype=object),
                pitcher_hand,
            ],
            names=["pitcher_trackman_id", "fine_pitch_type", "pitcher_hand"],
        )
        pitch_n[:, position] = profile["pitch_n"].reindex(index).fillna(0).to_numpy()
    reliability = pitch_n / (pitch_n + 50.0)
    effective_weights = np.nan_to_num(weights) * reliability
    normalizer = effective_weights.sum(axis=1)
    for column in z_columns:
        matrix = np.empty((len(rows), len(FINE_TYPES)), dtype=np.float32)
        for position, pitch_type in enumerate(FINE_TYPES):
            index = pd.MultiIndex.from_arrays(
                [
                    mapped,
                    np.full(len(rows), pitch_type, dtype=object),
                    pitcher_hand,
                ],
                names=[
                    "pitcher_trackman_id",
                    "fine_pitch_type",
                    "pitcher_hand",
                ],
            )
            matrix[:, position] = profile[column].reindex(index).to_numpy(
                dtype=np.float32
            )
        present = np.isfinite(matrix)
        effective = np.where(present, effective_weights, 0.0)
        denominator = effective.sum(axis=1)
        mean = np.divide(
            np.nansum(effective * matrix, axis=1),
            denominator,
            out=np.full(len(rows), np.nan),
            where=denominator > 0,
        )
        variance = np.divide(
            np.nansum(effective * np.square(matrix - mean[:, None]), axis=1),
            denominator,
            out=np.full(len(rows), np.nan),
            where=denominator > 0,
        )
        output[f"src_{source_season}_{column}_expected"] = mean
        output[f"src_{source_season}_{column}_arsenal_var"] = variance
    output[f"src_{source_season}_log_n"] = np.log1p(pitch_n.sum(axis=1))
    output[f"src_{source_season}_type_coverage"] = (pitch_n > 0).sum(axis=1)
    output[f"src_{source_season}_reliability"] = np.divide(
        normalizer,
        np.nan_to_num(weights).sum(axis=1),
        out=np.zeros(len(rows)),
        where=np.nan_to_num(weights).sum(axis=1) > 0,
    )
    output[f"src_{source_season}_valid"] = (
        valid_propensity & (normalizer > 0)
    ).astype(np.int8)
    return pd.DataFrame(output)


def build_features(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    aligned: pd.DataFrame,
    profiles: dict[int, pd.DataFrame],
    season: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cutoff = season - 1
    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    mapping, map_audit = mapping_from_aligned(aligned, cutoff)
    mapped = rows["pitcher_id"].map(mapping.mapping)
    propensity = propensity_table(trackman, cutoff)
    source_frames = []
    source_seasons = [value for value in sorted(profiles) if value <= cutoff]
    for source_season in source_seasons:
        source_frames.append(
            expected_source(
                rows,
                mapped,
                propensity,
                profiles[source_season],
                source_season,
            )
        )
    raw = pd.concat(source_frames, axis=1)
    output: dict[str, np.ndarray] = {}
    suffixes = sorted(
        {
            column.split("_", 2)[2]
            for column in raw
            if column.startswith("src_")
            and not column.endswith("_valid")
        }
    )
    source_x = np.asarray(source_seasons, dtype=float)
    centered_x = source_x - source_x.mean()
    denominator = float(np.square(centered_x).sum())
    for suffix in suffixes:
        values = np.column_stack(
            [raw[f"src_{value}_{suffix}"].to_numpy(float) for value in source_seasons]
        )
        output[f"tm_equal_{suffix}"] = np.nanmean(values, axis=1)
        output[f"tm_last_{suffix}"] = values[:, -1]
        output[f"tm_last_minus_equal_{suffix}"] = (
            values[:, -1] - np.nanmean(values, axis=1)
        )
        if denominator > 0:
            output[f"tm_trend_{suffix}"] = np.nansum(
                (values - np.nanmean(values, axis=1)[:, None])
                * centered_x[None, :],
                axis=1,
            ) / denominator
    features = pd.DataFrame(output)
    features["trackman_mapped"] = mapped.notna().astype(np.int8)
    features["official_success_rate"] = rows[
        "asof_pitcher_success_rate"
    ].to_numpy(float)
    features["official_log_n"] = np.log1p(rows["asof_pitcher_n"].to_numpy(float))
    features["count_index"] = rows["count_index"].to_numpy(float)
    features["batter_hand"] = rows["batter_hand"].to_numpy(float)
    audit = {
        **map_audit,
        "source_seasons": source_seasons,
        "row_mapping_coverage": float(mapped.notna().mean()),
        "feature_count": int(features.shape[1] - 1),
    }
    return features, audit


def recent_base(season: int) -> np.ndarray:
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
        TRACKMAN_CONTROL_ROOT / f"predictions_fine_direct_w025_{season}.npy"
    ).astype(float)
    direct = (trackman - strict) / 0.25
    return np.clip(0.5 * recency + 0.5 * aggressive + 0.10 * direct, 0.0, 1.0)


def main() -> None:
    started = time.time()
    main_frame = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = exact_aligned_rows()
    profiles = source_profiles(trackman)
    features: dict[int, pd.DataFrame] = {}
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        features[season], audits[str(season)] = build_features(
            main_frame, trackman, aligned, profiles, season
        )
        targets[season] = np.load(
            LOWRANK_ROOT / f"targets_{season}.npy"
        ).astype(float)
        base[season] = recent_base(season)
        expected = main_frame.loc[
            main_frame["season"].eq(season), "control_success"
        ].to_numpy(float)
        if not np.array_equal(targets[season], expected):
            raise ValueError(f"target/order mismatch {season}")
        print(
            f"features {season}: cols={features[season].shape[1]} "
            f"coverage={audits[str(season)]['row_mapping_coverage']:.3f}",
            flush=True,
        )

    common = sorted(
        set.intersection(
            *[set(features[season].columns) for season in EVALUATED_SEASONS]
        )
        - {"trackman_mapped"}
    )
    residual = {season: targets[season] - base[season] for season in EVALUATED_SEASONS}
    for season in residual:
        residual[season] -= residual[season].mean()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    candidate_names = [
        f"normgeo_a{int(alpha)}_w{int(weight * 1000):03d}"
        for alpha in ALPHAS
        for weight in WEIGHTS
    ]
    for validation_season in EVALUATED_SEASONS:
        sources = [season for season in EVALUATED_SEASONS if season < validation_season]
        corrections = {
            int(alpha): np.zeros(len(targets[validation_season])) for alpha in ALPHAS
        }
        if sources:
            train_x = pd.concat(
                [features[season][common] for season in sources], ignore_index=True
            )
            train_y = np.concatenate([residual[season] for season in sources])
            source = np.concatenate(
                [np.full(len(features[season]), season) for season in sources]
            )
            eligible = pd.concat(
                [features[season]["trackman_mapped"] for season in sources],
                ignore_index=True,
            ).eq(1).to_numpy()
            valid = features[validation_season]["trackman_mapped"].eq(1).to_numpy()
            counts = pd.Series(source[eligible]).value_counts()
            sample_weight = np.array(
                [1.0 / counts[value] for value in source[eligible]]
            )
            sample_weight *= len(sample_weight) / sample_weight.sum()
            for alpha in ALPHAS:
                model = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                    Ridge(alpha=alpha),
                )
                model.fit(
                    train_x.loc[eligible],
                    train_y[eligible],
                    ridge__sample_weight=sample_weight,
                )
                corrections[int(alpha)][valid] = np.clip(
                    model.predict(features[validation_season].loc[valid, common]),
                    -CORRECTION_CLIP,
                    CORRECTION_CLIP,
                )
        predictions = {"base": base[validation_season]}
        for alpha in ALPHAS:
            for weight in WEIGHTS:
                name = f"normgeo_a{int(alpha)}_w{int(weight * 1000):03d}"
                predictions[name] = np.clip(
                    base[validation_season] + weight * corrections[int(alpha)],
                    0.0,
                    1.0,
                )
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_seasons": sources,
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(targets[validation_season], prediction)
            np.save(
                ARTIFACT_DIR / f"predictions_{name}_{validation_season}.npy",
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
                f"{name}={fold[name]['skill_score_unclipped']:.2f}"
                for name in candidate_names
            ),
            flush=True,
        )

    aggregate: dict[str, object] = {}
    for name in ("base", *candidate_names):
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
        candidate_names,
        key=lambda name: (
            aggregate[name]["min_skill"],
            aggregate[name]["mean_skill"],
        ),
    )
    result = {
        "experiment": "EXP-059",
        "candidate_family": "season_normalized_trackman_arsenal_geometry",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "history_mapping_and_physical_cutoff": "validation season-1",
            "source_season_equal_weight": True,
            "current_fold_labels_used_for_fit_or_selection": False,
            "actual_current_pitch_type_or_measurement_used": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "model": {
            "physical_columns": list(PHYSICAL),
            "peer_normalization": "source season x fine pitch type x pitcher hand",
            "profile_statistics": ["expected z mean", "arsenal variance", "last-equal", "trend"],
            "ridge_alphas": list(ALPHAS),
            "weights": list(WEIGHTS),
            "correction_clip": CORRECTION_CLIP,
            "feature_count": len(common),
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
            "sklearn": sklearn.__version__,
        },
        "total_seconds": time.time() - started,
    }
    (ARTIFACT_DIR / "validation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"best={best} mean={aggregate[best]['mean_skill']:.2f} "
        f"min={aggregate[best]['min_skill']:.2f} "
        f"adopt={result['selection']['adopt']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
