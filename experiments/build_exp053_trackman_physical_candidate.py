"""Fit, serialize and package EXP-053 exact Trackman physical correction."""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_exp021_final_candidates import build_zip, smoke_test
from exp021_submission_inference import map_trackman_physical_ridge
from train_exp017_rolling_residual import calculate_metrics
from train_exp041_exact_game_trackman_sequence import (
    exact_aligned_rows,
    mapping_from_aligned,
)
from train_exp045_exact_trackman_physical_residual import (
    grouped_physical,
    load_main,
    load_trackman,
    expected_summary,
    build_features,
)
from train_exp043_exact_pitchtype_control_eb import propensity_table


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "submissions" / "EXP-051-TMDIRECT"
DESTINATION = ROOT / "submissions" / "EXP-053-TMPHYS"
ZIP_PATH = ROOT / "submit_exp053_tmphys.zip"
ARTIFACT_DIR = ROOT / "artifacts" / "EXP-053" / "trackman_physical_candidate"
TEMPLATE = ROOT / "experiments" / "exp021_submission_inference.py"
LOWRANK_ROOT = ROOT / "artifacts" / "EXP-020" / "low_rank_pitcher_context_eb"
AGGRESSIVE_ROOT = ROOT / "artifacts" / "EXP-020" / "pitcher_count_eb_atop_team"
RECENCY_ROOT = ROOT / "artifacts" / "EXP-037" / "lowrank_source_policies"
DIRECT_ROOT = ROOT / "artifacts" / "EXP-050" / "exact_dual_propensity_control"
PHYSICAL_ROOT = ROOT / "artifacts" / "EXP-045" / "exact_trackman_physical_residual"
REPORT_SEASONS = (2022, 2023, 2024)
FIT_SEASONS = (2021, 2022, 2023, 2024)
DIRECT_WEIGHT = 0.10
PHYSICAL_WEIGHT = 0.15
RIDGE_ALPHA = 1000.0
DYNAMIC_FEATURES = (
    "official_success_rate",
    "official_log_n",
    "count_index",
    "batter_hand",
    "balls_before",
    "strikes_before",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def recent_base(season: int) -> np.ndarray:
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    return 0.5 * recency + 0.5 * aggressive


def validation() -> dict[str, object]:
    briers: dict[str, float] = {}
    skills: dict[str, float] = {}
    for season in REPORT_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        base = recent_base(season)
        direct = (
            np.load(DIRECT_ROOT / f"predictions_pitcher_prop_w025_{season}.npy")
            - np.load(DIRECT_ROOT / f"predictions_base_{season}.npy")
        ) / 0.25
        physical = (
            np.load(PHYSICAL_ROOT / f"predictions_ridge_a1000_w025_{season}.npy")
            - np.load(PHYSICAL_ROOT / f"predictions_base_{season}.npy")
        ) / 0.25
        prediction = np.clip(
            base + DIRECT_WEIGHT * direct + PHYSICAL_WEIGHT * physical,
            0.0,
            1.0,
        )
        metrics = calculate_metrics(target, prediction)
        briers[str(season)] = float(metrics["brier_score"])
        skills[str(season)] = float(metrics["skill_score_unclipped"])
    values = list(skills.values())
    return {
        "season_briers": briers,
        "season_skills": skills,
        "mean_skill": float(np.mean(values)),
        "min_skill": float(np.min(values)),
        "latest_2024_skill": skills["2024"],
    }


def fit_pipeline(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    aligned: pd.DataFrame,
) -> tuple[object, list[str], dict[str, object]]:
    feature_frames: list[pd.DataFrame] = []
    residuals: list[np.ndarray] = []
    seasons: list[np.ndarray] = []
    coverage: dict[str, float] = {}
    for season in FIT_SEASONS:
        features, audit = build_features(main, trackman, aligned, season)
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        residual = target - recent_base(season)
        residual -= residual.mean()
        feature_frames.append(features)
        residuals.append(residual)
        seasons.append(np.full(len(features), season, dtype=np.int16))
        coverage[str(season)] = float(audit["row_mapping_coverage"])
    feature_names = [
        column for column in feature_frames[0] if column != "trackman_mapped"
    ]
    frame = pd.concat(feature_frames, ignore_index=True)
    target = np.concatenate(residuals)
    season_values = np.concatenate(seasons)
    eligible = frame["trackman_mapped"].eq(1).to_numpy()
    counts = pd.Series(season_values[eligible]).value_counts()
    weights = np.array([1.0 / counts[value] for value in season_values[eligible]])
    weights *= len(weights) / weights.sum()
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )
    pipeline.fit(
        frame.loc[eligible, feature_names],
        target[eligible],
        ridge__sample_weight=weights,
    )
    audit = {
        "fit_seasons": list(FIT_SEASONS),
        "fit_rows": int(eligible.sum()),
        "feature_count": len(feature_names),
        "season_equal_weight": True,
        "mapping_coverage": coverage,
    }
    return pipeline, feature_names, audit


def physical_lookup(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    aligned: pd.DataFrame,
) -> tuple[list[int], list[dict[str, int]], list[str], np.ndarray, dict[str, object]]:
    mapping, mapping_audit = mapping_from_aligned(aligned, 2024)
    official_ids = sorted(mapping.mapping)
    contexts_frame = (
        main[["count_index", "batter_hand"]]
        .drop_duplicates()
        .sort_values(["count_index", "batter_hand"])
        .reset_index(drop=True)
    )
    contexts = [
        {
            "position": int(position),
            "count_index": int(row.count_index),
            "batter_hand": int(row.batter_hand),
        }
        for position, row in enumerate(contexts_frame.itertuples(index=False))
    ]
    synthetic = pd.DataFrame(
        [
            {
                "pitcher_id": official_id,
                "count_index": context["count_index"],
                "batter_hand": context["batter_hand"],
            }
            for official_id in official_ids
            for context in contexts
        ]
    )
    mapped = synthetic["pitcher_id"].map(mapping.mapping)
    all_history = trackman.loc[trackman["season"].le(2024)]
    last_history = trackman.loc[trackman["season"].eq(2024)]
    prior_history = trackman.loc[trackman["season"].lt(2024)]
    all_type, all_overall = grouped_physical(all_history)
    last_type, _ = grouped_physical(last_history)
    prior_type, _ = grouped_physical(prior_history)
    propensity = propensity_table(trackman, 2024)
    all_expected = expected_summary(
        synthetic, mapped, propensity, all_type, "tm_all"
    )
    last_expected = expected_summary(
        synthetic, mapped, propensity, last_type, "tm_last"
    )
    prior_expected = expected_summary(
        synthetic, mapped, propensity, prior_type, "tm_prior"
    )
    state_frame = pd.concat(
        [all_expected, last_expected, prior_expected], axis=1
    )
    for column in all_expected:
        if column.endswith("mix_entropy"):
            continue
        suffix = column.removeprefix("tm_all_")
        state_frame[f"tm_delta_{suffix}"] = (
            last_expected[f"tm_last_{suffix}"]
            - prior_expected[f"tm_prior_{suffix}"]
        )
    overall = all_overall.reindex(mapped).reset_index(drop=True)
    overall.columns = [f"tm_{column}" for column in overall.columns]
    state_frame = pd.concat([state_frame, overall], axis=1)
    state_names = list(state_frame.columns)
    values = state_frame.to_numpy(float).reshape(
        len(official_ids), len(contexts), len(state_names)
    )
    audit = {
        **mapping_audit,
        "official_pitchers": len(official_ids),
        "contexts": len(contexts),
        "state_feature_count": len(state_names),
        "lookup_rows": len(synthetic),
        "propensity_contexts": len(propensity),
    }
    return official_ids, contexts, state_names, values, audit


def export_pipeline(
    pipeline: object,
    feature_names: list[str],
    official_ids: list[int],
    contexts: list[dict[str, int]],
    state_names: list[str],
    state_values: np.ndarray,
) -> tuple[dict[str, object], dict[str, float]]:
    imputer = pipeline.named_steps["simpleimputer"]
    scaler = pipeline.named_steps["standardscaler"]
    ridge = pipeline.named_steps["ridge"]
    statistics = np.asarray(imputer.statistics_, dtype=float)
    if not np.isfinite(statistics).all():
        raise ValueError("all-missing physical model feature encountered")
    indicator = np.asarray(imputer.indicator_.features_, dtype=int)
    state = {
        "format": "median_standardized_ridge_v1",
        "through_season": 2024,
        "feature_names": feature_names,
        "state_feature_names": state_names,
        "dynamic_feature_names": list(DYNAMIC_FEATURES),
        "official_pitcher_ids": official_ids,
        "contexts": contexts,
        "state_values": state_values.tolist(),
        "imputer_statistics": statistics.tolist(),
        "indicator_features": indicator.tolist(),
        "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        "ridge_coefficient": np.asarray(ridge.coef_, dtype=float).tolist(),
        "ridge_intercept": float(ridge.intercept_),
        "ridge_alpha": RIDGE_ALPHA,
        "correction_clip": 0.03,
    }
    # Exact parity on a deliberately mixed finite/missing matrix.
    probe = np.tile(statistics, (5, 1))
    probe[0, 0] = np.nan
    probe[1, min(5, len(feature_names) - 1)] = np.nan
    expected = pipeline.predict(pd.DataFrame(probe, columns=feature_names))
    missing = np.isnan(probe)
    transformed = np.where(missing, statistics, probe)
    if indicator.size:
        transformed = np.column_stack(
            [transformed, missing[:, indicator].astype(float)]
        )
    actual = (
        (transformed - np.asarray(scaler.mean_)) / np.asarray(scaler.scale_)
    ) @ np.asarray(ridge.coef_) + float(ridge.intercept_)
    parity = float(np.max(np.abs(expected - actual)))
    if parity > 1e-12:
        raise ValueError(f"serialized Ridge parity failure: {parity}")
    return state, {"probe_max_abs": parity}


def main() -> None:
    started = time.time()
    metrics = validation()
    main_frame = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = exact_aligned_rows()
    pipeline, feature_names, fit_audit = fit_pipeline(
        main_frame, trackman, aligned
    )
    official_ids, contexts, state_names, values, lookup_audit = physical_lookup(
        main_frame, trackman, aligned
    )
    state, parity = export_pipeline(
        pipeline,
        feature_names,
        official_ids,
        contexts,
        state_names,
        values,
    )
    # Validate exported inference on generated 2025 lookup rows through the
    # production function itself; dynamic values are representative only.
    probe_rows = pd.DataFrame(
        {
            "pitcher_id": official_ids[:5],
            "count_index": [0, 1, 2, 4, 5],
            "batter_hand": [1, 2, 1, 2, 1],
            "asof_pitcher_success_rate": [0.5] * 5,
            "asof_pitcher_n": [100.0] * 5,
            "balls_before": [0, 0, 0, 1, 1],
            "strikes_before": [0, 1, 2, 0, 1],
        }
    )
    serialized_probe = map_trackman_physical_ridge(probe_rows, state)
    if not np.isfinite(serialized_probe).all():
        raise ValueError("serialized physical production probe is non-finite")

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    shutil.copytree(SOURCE_DIR, DESTINATION)
    shutil.copyfile(TEMPLATE, DESTINATION / "script.py")
    write_json(DESTINATION / "model" / "trackman_physical_ridge.json", state)
    metadata_path = DESTINATION / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "experiment": "EXP-053",
            "candidate": "trackman_physical_recent_w015",
            "component_formula": (
                "recentaggr + 0.10 exact control + 0.15 physical Ridge"
            ),
            "validation_aggregate_2022_2024": metrics,
            "trackman_physical_fit_audit": fit_audit,
            "trackman_physical_lookup_audit": lookup_audit,
            "selection_status": (
                "new complementary physical signal; weight comparison not nested"
            ),
            "probability_calibration": "identity",
        }
    )
    write_json(metadata_path, metadata)
    zip_result = build_zip(DESTINATION, ZIP_PATH)
    smoke = smoke_test(ZIP_PATH)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "EXP-053",
        "candidate": metadata["candidate"],
        "validation": metrics,
        "exact_alignment": alignment_audit,
        "fit_audit": fit_audit,
        "lookup_audit": lookup_audit,
        "serialization_parity": parity,
        "production_probe": serialized_probe.tolist(),
        "zip": zip_result,
        "smoke": smoke,
        "qa": {
            "current_fold_labels_used_to_fit_validation_models": False,
            "full_fit_source_oof_seasons": list(FIT_SEASONS),
            "test_row_aggregation": False,
            "actual_current_pitch_measurement_used": False,
            "native_json_no_pickle": True,
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        },
        "total_seconds": time.time() - started,
    }
    write_json(ARTIFACT_DIR / "validation_metrics.json", report)
    print(
        f"saved={ZIP_PATH} mean={metrics['mean_skill']:.3f} "
        f"min={metrics['min_skill']:.3f} smoke=passed",
        flush=True,
    )


if __name__ == "__main__":
    main()
