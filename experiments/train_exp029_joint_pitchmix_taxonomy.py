"""EXP-029: 15-class joint pitch-group and outcome-taxonomy model."""

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

from outcome_taxonomy_features import (
    JOINT_CLASS_NAMES,
    assert_label_reconstruction_invariants,
    derive_joint_taxonomy,
    reconstruct_outcome_labels,
)
from pitchmix_outcome_features import PITCH_GROUP_NAMES, reconstruct_pitch_group
from train_exp017_rolling_residual import calculate_metrics, prepare_data
from train_exp019_histgb_residual import season_equal_weights, select_stable_features
from train_exp022_outcome_taxonomy_multitask import (
    AUX_MODEL_CONFIG,
    REPORT_SEASONS,
    TARGET_SKILL,
    detailed_segments,
    load_frozen_base,
    load_raw_label_frame,
)


EXPERIMENT = "EXP-029"
ARTIFACT_ROOT = Path("./artifacts/EXP-029/joint_pitchmix_taxonomy")
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
BLEND_WEIGHTS = (0.10, 0.25, 0.50)
PITCHMIX_EXTRA_COLUMNS = (
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)


def main() -> None:
    started = time.time()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    label_raw = load_raw_label_frame()
    mix_raw = pd.read_csv(
        "./data/train.csv",
        encoding="utf-8-sig",
        usecols=[
            "row_id",
            "pitcher_id",
            "season",
            "asof_pitcher_n",
            *PITCHMIX_EXTRA_COLUMNS,
        ],
    )
    if not np.array_equal(label_raw["row_id"].to_numpy(), mix_raw["row_id"].to_numpy()):
        raise ValueError("label frame row order mismatch")
    labels, label_audit = reconstruct_outcome_labels(label_raw)
    assert_label_reconstruction_invariants(label_raw, labels, label_audit)
    outcome_series, outcome_audit = derive_joint_taxonomy(labels)
    pitch_series, pitch_audit = reconstruct_pitch_group(mix_raw)
    outcome = outcome_series.to_numpy(dtype=float)
    pitch = pitch_series.to_numpy(dtype=float)
    valid = np.isfinite(outcome) & np.isfinite(pitch)
    joint15 = np.full(len(outcome), np.nan, dtype=float)
    joint15[valid] = pitch[valid] * len(JOINT_CLASS_NAMES) + outcome[valid]
    class_names = tuple(
        f"{pitch_name}__{outcome_name}"
        for pitch_name in PITCH_GROUP_NAMES
        for outcome_name in JOINT_CLASS_NAMES
    )
    class_counts = {
        name: int(np.sum(joint15[valid] == index))
        for index, name in enumerate(class_names)
    }
    if any(count == 0 for count in class_counts.values()):
        raise ValueError("empty joint pitch/outcome class")

    diagnostics, full_X, y, _unused, seasons, feature_names = prepare_data()
    del _unused
    if not np.array_equal(label_raw["control_success"].to_numpy(dtype=np.float32), y):
        raise ValueError("target order mismatch")
    diagnostics = diagnostics.copy()
    diagnostics["game_type"] = label_raw["game_type"].to_numpy()
    X, selected_features = select_stable_features(full_X, feature_names)
    del full_X, label_raw, mix_raw, labels, outcome_series, pitch_series
    gc.collect()
    base_by_season, targets_by_season, base_alignment = load_frozen_base(
        y, seasons, VALIDATION_SEASONS
    )

    success_classes = np.asarray(
        [pitch_index * len(JOINT_CLASS_NAMES) for pitch_index in range(len(PITCH_GROUP_NAMES))],
        dtype=int,
    )
    success_predictions: dict[int, np.ndarray] = {}
    model_folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        train_mask = (seasons < validation_season) & valid
        validation_mask = seasons == validation_season
        training_seasons = sorted(np.unique(seasons[train_mask]).astype(int).tolist())
        model = HistGradientBoostingClassifier(**AUX_MODEL_CONFIG)
        fit_started = time.time()
        model.fit(
            X[train_mask],
            joint15[train_mask].astype(np.int8),
            sample_weight=season_equal_weights(seasons[train_mask]),
        )
        expected = np.arange(len(class_names), dtype=np.int64)
        if not np.array_equal(model.classes_, expected):
            raise ValueError(f"missing joint15 class in {validation_season}")
        probabilities = model.predict_proba(X[validation_mask]).astype(float)
        if probabilities.shape != (int(validation_mask.sum()), len(class_names)):
            raise ValueError("joint15 prediction shape mismatch")
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("invalid joint15 probabilities")
        success = probabilities[:, success_classes].sum(axis=1)
        success_predictions[validation_season] = success
        np.save(
            ARTIFACT_ROOT / f"predictions_joint15_success_{validation_season}.npy",
            success,
        )
        model_folds[str(validation_season)] = {
            "validation_season": validation_season,
            "training_seasons": training_seasons,
            "training_rows": int(train_mask.sum()),
            "iterations_completed": int(model.n_iter_),
            "fit_predict_seconds": time.time() - fit_started,
            "direct_success": calculate_metrics(targets_by_season[validation_season], success),
            "base": calculate_metrics(targets_by_season[validation_season], base_by_season[validation_season]),
        }
        del model, probabilities
        gc.collect()

    folds: dict[str, object] = {}
    for validation_season in REPORT_SEASONS:
        prior_seasons = [season for season in VALIDATION_SEASONS if season < validation_season]
        selection: dict[str, object] = {}
        for weight in BLEND_WEIGHTS:
            name = f"blend_w{int(round(weight * 100)):03d}"
            season_metrics = {
                str(season): calculate_metrics(
                    targets_by_season[season],
                    (1.0 - weight) * base_by_season[season] + weight * success_predictions[season],
                )
                for season in prior_seasons
            }
            skills = [float(metric["skill_score_unclipped"]) for metric in season_metrics.values()]
            selection[name] = {
                "weight": weight,
                "season_metrics": season_metrics,
                "min_skill": float(min(skills)),
                "mean_skill": float(np.mean(skills)),
            }
        selected_name = max(
            selection,
            key=lambda name: (
                selection[name]["min_skill"],
                selection[name]["mean_skill"],
                -selection[name]["weight"],
            ),
        )
        candidates: dict[str, object] = {}
        predictions: dict[str, np.ndarray] = {}
        for weight in BLEND_WEIGHTS:
            name = f"blend_w{int(round(weight * 100)):03d}"
            prediction = (
                (1.0 - weight) * base_by_season[validation_season]
                + weight * success_predictions[validation_season]
            )
            predictions[name] = prediction
            candidates[name] = calculate_metrics(targets_by_season[validation_season], prediction)
            np.save(ARTIFACT_ROOT / f"predictions_{name}_{validation_season}.npy", prediction)
        selected = predictions[selected_name]
        np.save(ARTIFACT_ROOT / f"predictions_strict_selected_{validation_season}.npy", selected)
        mask = seasons == validation_season
        folds[str(validation_season)] = {
            **model_folds[str(validation_season)],
            "selection": {
                "source_oof_seasons": prior_seasons,
                "current_fold_labels_used": False,
                "candidates": selection,
                "selected_candidate": selected_name,
            },
            "candidates": candidates,
            "selected": {
                "candidate": selected_name,
                **calculate_metrics(targets_by_season[validation_season], selected),
            },
            "selected_segments": detailed_segments(
                diagnostics, mask, targets_by_season[validation_season], selected
            ),
        }

    selected_skills = {
        str(season): float(folds[str(season)]["selected"]["skill_score_unclipped"])
        for season in REPORT_SEASONS
    }
    selected_briers = {
        str(season): float(folds[str(season)]["selected"]["brier_score"])
        for season in REPORT_SEASONS
    }
    base_skills = {
        str(season): float(folds[str(season)]["base"]["skill_score_unclipped"])
        for season in REPORT_SEASONS
    }
    uniform_pass = bool(
        all(value >= TARGET_SKILL for value in selected_skills.values())
        and all(selected_skills[str(season)] >= base_skills[str(season)] for season in REPORT_SEASONS)
    )
    report = {
        "experiment": EXPERIMENT,
        "stage": "bounded 15-class pitchmix/outcome taxonomy validation",
        "hypothesis": (
            "train-only reconstructed pitch group supplies latent pitch-selection "
            "supervision that improves future-season success discrimination"
        ),
        "validation_protocol": {
            "warmup_season": 2021,
            "report_seasons": list(REPORT_SEASONS),
            "source_seasons_strictly_prior": True,
            "current_fold_selection": False,
            "test_csv_read": False,
            "test_row_aggregation": False,
            "current_pitch_group_used_at_inference": False,
        },
        "label_audit": label_audit,
        "outcome_taxonomy_audit": outcome_audit,
        "pitch_group_audit": pitch_audit,
        "joint15_audit": {
            "valid_rows": int(valid.sum()),
            "class_names": list(class_names),
            "class_counts": class_counts,
            "all_classes_nonempty": True,
        },
        "model": {
            "class": "HistGradientBoostingClassifier",
            "config": AUX_MODEL_CONFIG,
            "feature_count": len(selected_features),
            "features": selected_features,
            "fixed_blend_weights": list(BLEND_WEIGHTS),
        },
        "base_alignment": base_alignment,
        "warmup_fold": model_folds["2021"],
        "folds": folds,
        "aggregate_2022_2024": {
            "selected_season_briers": selected_briers,
            "selected_season_skills": selected_skills,
            "base_season_skills": base_skills,
            "mean_skill": float(np.mean(list(selected_skills.values()))),
            "min_skill": float(np.min(list(selected_skills.values()))),
            "latest_2024_skill": selected_skills["2024"],
            "uniform_1100_passed": uniform_pass,
            "final_fit_authorized": uniform_pass,
            "zip_creation_authorized": uniform_pass,
        },
        "qa": {
            "pitch_group_delta_onehot": pitch_audit["all_valid_group_deltas_onehot"],
            "pitchmix_n_delta_one": pitch_audit["all_valid_delta_pitchmix_n_equal_one"],
            "target_and_row_order_match": True,
            "source_seasons_strictly_prior": True,
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
