"""EXP-030: predicted pitch-selection propensity as a temporal residual signal."""

from __future__ import annotations

import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from pitchmix_outcome_features import PITCH_GROUP_NAMES, reconstruct_pitch_group
from train_exp017_rolling_residual import calculate_metrics, prepare_data
from train_exp019_histgb_residual import season_equal_weights, select_stable_features
from train_exp022_outcome_taxonomy_multitask import (
    AUX_MODEL_CONFIG,
    REPORT_SEASONS,
    TARGET_SKILL,
    detailed_segments,
    load_frozen_base,
)


EXPERIMENT = "EXP-030"
ARTIFACT_ROOT = Path("./artifacts/EXP-030/pitch_selection_residual")
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
RIDGE_ALPHA = 5000.0
CORRECTION_SCALES = (0.25, 0.50)
EPS = 1e-8


def pitch_representation(
    probabilities: np.ndarray, official_rates: np.ndarray
) -> np.ndarray:
    normalized = official_rates / np.clip(official_rates.sum(axis=1, keepdims=True), EPS, None)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, EPS, 1.0)), axis=1)
    return np.column_stack(
        [
            probabilities,
            probabilities - normalized,
            entropy,
            probabilities.max(axis=1),
        ]
    ).astype(np.float64)


def main() -> None:
    started = time.time()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(
        "./data/train.csv",
        encoding="utf-8-sig",
        usecols=[
            "row_id",
            "pitcher_id",
            "season",
            "asof_pitcher_n",
            "asof_pitcher_pitchmix_n",
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
            "game_type",
            "control_success",
        ],
    )
    pitch_series, pitch_audit = reconstruct_pitch_group(raw)
    pitch = pitch_series.to_numpy(dtype=float)
    valid_pitch = np.isfinite(pitch)
    official_rates = raw[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].fillna(1.0 / 3.0).to_numpy(dtype=float)

    diagnostics, full_X, y, _unused, seasons, feature_names = prepare_data()
    del _unused
    if not np.array_equal(raw["control_success"].to_numpy(dtype=np.float32), y):
        raise ValueError("target/order mismatch")
    diagnostics = diagnostics.copy()
    diagnostics["game_type"] = raw["game_type"].to_numpy()
    X, selected_features = select_stable_features(full_X, feature_names)
    del full_X
    gc.collect()
    base_by_season, targets_by_season, base_alignment = load_frozen_base(
        y, seasons, VALIDATION_SEASONS
    )

    representations: dict[int, np.ndarray] = {}
    propensity_folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        train_mask = (seasons < validation_season) & valid_pitch
        validation_mask = seasons == validation_season
        model = HistGradientBoostingClassifier(**AUX_MODEL_CONFIG)
        fit_started = time.time()
        model.fit(
            X[train_mask],
            pitch[train_mask].astype(np.int8),
            sample_weight=season_equal_weights(seasons[train_mask]),
        )
        if not np.array_equal(model.classes_, np.arange(3)):
            raise ValueError("missing pitch group")
        probabilities = model.predict_proba(X[validation_mask]).astype(float)
        representation = pitch_representation(
            probabilities, official_rates[validation_mask]
        )
        if not np.isfinite(representation).all():
            raise ValueError("non-finite pitch propensity representation")
        representations[validation_season] = representation
        np.save(
            ARTIFACT_ROOT / f"pitch_representation_{validation_season}.npy",
            representation,
        )
        local_valid = valid_pitch[validation_mask]
        propensity_folds[str(validation_season)] = {
            "validation_season": validation_season,
            "training_seasons": sorted(np.unique(seasons[train_mask]).astype(int).tolist()),
            "training_rows": int(train_mask.sum()),
            "validation_labeled_rows": int(local_valid.sum()),
            "validation_accuracy": float(
                np.mean(np.argmax(probabilities[local_valid], axis=1) == pitch[validation_mask][local_valid])
            ),
            "prediction_group_mean": {
                name: float(probabilities[:, index].mean())
                for index, name in enumerate(PITCH_GROUP_NAMES)
            },
            "representation_feature_means": representation.mean(axis=0).tolist(),
            "iterations_completed": int(model.n_iter_),
            "fit_predict_seconds": time.time() - fit_started,
        }
        del model, probabilities
        gc.collect()

    folds: dict[str, object] = {}
    for validation_season in REPORT_SEASONS:
        source_seasons = [season for season in VALIDATION_SEASONS if season < validation_season]
        source_X = np.concatenate([representations[season] for season in source_seasons])
        source_y = np.concatenate(
            [targets_by_season[season] - base_by_season[season] for season in source_seasons]
        )
        source_vector = np.concatenate(
            [np.full(len(representations[season]), season, dtype=np.int16) for season in source_seasons]
        )
        centered_y = source_y.copy()
        for season in source_seasons:
            local = source_vector == season
            centered_y[local] -= centered_y[local].mean()
        scaler = StandardScaler()
        scaled_source = scaler.fit_transform(source_X)
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        model.fit(
            scaled_source,
            centered_y,
            sample_weight=season_equal_weights(source_vector),
        )
        correction = model.predict(
            scaler.transform(representations[validation_season])
        ).astype(float)
        candidates: dict[str, object] = {}
        candidate_predictions: dict[str, np.ndarray] = {}
        for scale in CORRECTION_SCALES:
            name = f"pitch_residual_w{int(round(scale * 100)):03d}"
            prediction = np.clip(
                base_by_season[validation_season] + scale * correction,
                0.0,
                1.0,
            )
            candidate_predictions[name] = prediction
            candidates[name] = calculate_metrics(targets_by_season[validation_season], prediction)
            np.save(ARTIFACT_ROOT / f"predictions_{name}_{validation_season}.npy", prediction)
        mask = seasons == validation_season
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "source_residual_centered_per_season": True,
            "current_fold_labels_used_for_fit_or_selection": False,
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_coefficients": model.coef_.astype(float).tolist(),
            "correction_mean": float(correction.mean()),
            "correction_mean_absolute": float(np.abs(correction).mean()),
            "correction_max_absolute": float(np.abs(correction).max()),
            "base": calculate_metrics(targets_by_season[validation_season], base_by_season[validation_season]),
            "candidates": candidates,
            "segments": {
                name: detailed_segments(
                    diagnostics, mask, targets_by_season[validation_season], prediction
                )
                for name, prediction in candidate_predictions.items()
            },
        }

    aggregate_candidates: dict[str, object] = {}
    for name in sorted(folds["2022"]["candidates"]):
        briers = {
            str(season): float(folds[str(season)]["candidates"][name]["brier_score"])
            for season in REPORT_SEASONS
        }
        skills = {
            str(season): float(folds[str(season)]["candidates"][name]["skill_score_unclipped"])
            for season in REPORT_SEASONS
        }
        aggregate_candidates[name] = {
            "season_briers": briers,
            "season_skills": skills,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills["2024"],
            "uniform_1100_passed": all(value >= TARGET_SKILL for value in skills.values()),
        }
    best_name = max(
        aggregate_candidates,
        key=lambda name: (
            aggregate_candidates[name]["min_skill"],
            aggregate_candidates[name]["mean_skill"],
        ),
    )
    uniform_pass = bool(aggregate_candidates[best_name]["uniform_1100_passed"])
    report = {
        "experiment": EXPERIMENT,
        "stage": "bounded pitch-selection residual validation",
        "hypothesis": (
            "predicted current pitch-group propensity minus official long-run mix "
            "provides a transferable low-dimensional success residual signal"
        ),
        "validation_protocol": {
            "warmup_season": 2021,
            "report_seasons": list(REPORT_SEASONS),
            "source_seasons_strictly_prior": True,
            "current_fold_selection": False,
            "test_csv_read": False,
            "test_row_aggregation": False,
            "current_pitch_group_used_at_inference": False,
            "fixed_correction_scales": list(CORRECTION_SCALES),
        },
        "pitch_group_audit": pitch_audit,
        "propensity_model": {
            "class": "HistGradientBoostingClassifier",
            "config": AUX_MODEL_CONFIG,
            "feature_count": len(selected_features),
            "features": selected_features,
            "representation_features": [
                "p_fastball", "p_breaking", "p_offspeed",
                "delta_fastball", "delta_breaking", "delta_offspeed",
                "entropy", "max_probability",
            ],
        },
        "propensity_folds": propensity_folds,
        "base_alignment": base_alignment,
        "folds": folds,
        "aggregate_2022_2024": {
            "candidates": aggregate_candidates,
            "posthoc_best_min_candidate": best_name,
            "uniform_1100_passed": uniform_pass,
            "final_fit_authorized": uniform_pass,
            "zip_creation_authorized": uniform_pass,
        },
        "qa": {
            "pitch_group_delta_onehot": pitch_audit["all_valid_group_deltas_onehot"],
            "pitchmix_n_delta_one": pitch_audit["all_valid_delta_pitchmix_n_equal_one"],
            "source_seasons_strictly_prior": True,
            "source_residual_centered_per_season": True,
            "current_fold_selection": False,
            "test_row_aggregation": False,
            "probabilities_finite_and_in_range": True,
            "final_fit_or_zip_created": False,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "total_seconds": time.time() - started,
    }
    with (ARTIFACT_ROOT / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report["aggregate_2022_2024"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
