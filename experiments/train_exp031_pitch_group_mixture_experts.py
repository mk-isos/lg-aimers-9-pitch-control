"""EXP-031: pitch-group conditional success mixture of experts."""

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


EXPERIMENT = "EXP-031"
ARTIFACT_ROOT = Path("./artifacts/EXP-031/pitch_group_mixture_experts")
PROPENSITY_ROOT = Path("./artifacts/EXP-030/pitch_selection_residual")
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
BLEND_WEIGHTS = (0.10, 0.25, 0.50)


def main() -> None:
    started = time.time()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(
        "./data/train.csv",
        encoding="utf-8-sig",
        usecols=[
            "row_id", "pitcher_id", "season", "asof_pitcher_n",
            "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
            "game_type", "control_success",
        ],
    )
    pitch_series, pitch_audit = reconstruct_pitch_group(raw)
    pitch = pitch_series.to_numpy(dtype=float)
    valid_pitch = np.isfinite(pitch)

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

    mixture_predictions: dict[int, np.ndarray] = {}
    expert_folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        validation_mask = seasons == validation_season
        representation = np.load(
            PROPENSITY_ROOT / f"pitch_representation_{validation_season}.npy"
        ).astype(float)
        propensities = representation[:, :3]
        if propensities.shape != (int(validation_mask.sum()), 3):
            raise ValueError("propensity shape mismatch")
        if not np.allclose(propensities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("propensities do not sum to one")
        expert_matrix = np.empty((int(validation_mask.sum()), 3), dtype=float)
        expert_diagnostics: dict[str, object] = {}
        for group_index, group_name in enumerate(PITCH_GROUP_NAMES):
            train_mask = (
                (seasons < validation_season)
                & valid_pitch
                & (pitch == group_index)
            )
            training_seasons = sorted(np.unique(seasons[train_mask]).astype(int).tolist())
            if not training_seasons or max(training_seasons) >= validation_season:
                raise AssertionError("expert source season is not strictly prior")
            model = HistGradientBoostingClassifier(**AUX_MODEL_CONFIG)
            fit_started = time.time()
            model.fit(
                X[train_mask],
                y[train_mask].astype(np.int8),
                sample_weight=season_equal_weights(seasons[train_mask]),
            )
            prediction = model.predict_proba(X[validation_mask])[:, 1].astype(float)
            if not np.isfinite(prediction).all():
                raise ValueError("non-finite expert prediction")
            expert_matrix[:, group_index] = prediction
            expert_diagnostics[group_name] = {
                "training_seasons": training_seasons,
                "training_rows": int(train_mask.sum()),
                "training_success_rate": float(y[train_mask].mean()),
                "prediction_mean": float(prediction.mean()),
                "iterations_completed": int(model.n_iter_),
                "fit_predict_seconds": time.time() - fit_started,
            }
            del model
            gc.collect()
        mixture = np.sum(propensities * expert_matrix, axis=1)
        if not np.isfinite(mixture).all() or not (
            ((mixture >= 0.0) & (mixture <= 1.0)).all()
        ):
            raise ValueError("invalid mixture prediction")
        mixture_predictions[validation_season] = mixture
        np.save(ARTIFACT_ROOT / f"predictions_direct_mixture_{validation_season}.npy", mixture)
        expert_folds[str(validation_season)] = {
            "validation_season": validation_season,
            "current_fold_labels_used_for_training": False,
            "propensity_source": "EXP-030 prior-only 3-class HGB",
            "experts": expert_diagnostics,
            "propensity_means": {
                name: float(propensities[:, index].mean())
                for index, name in enumerate(PITCH_GROUP_NAMES)
            },
            "direct_mixture": calculate_metrics(targets_by_season[validation_season], mixture),
            "base": calculate_metrics(targets_by_season[validation_season], base_by_season[validation_season]),
        }

    folds: dict[str, object] = {}
    for validation_season in REPORT_SEASONS:
        prior_seasons = [season for season in VALIDATION_SEASONS if season < validation_season]
        selection: dict[str, object] = {}
        for weight in BLEND_WEIGHTS:
            name = f"mixture_w{int(round(weight * 100)):03d}"
            metrics_by_season = {
                str(season): calculate_metrics(
                    targets_by_season[season],
                    (1.0 - weight) * base_by_season[season]
                    + weight * mixture_predictions[season],
                )
                for season in prior_seasons
            }
            skills = [float(metric["skill_score_unclipped"]) for metric in metrics_by_season.values()]
            selection[name] = {
                "weight": weight,
                "season_metrics": metrics_by_season,
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
        candidate_predictions: dict[str, np.ndarray] = {}
        for weight in BLEND_WEIGHTS:
            name = f"mixture_w{int(round(weight * 100)):03d}"
            prediction = (
                (1.0 - weight) * base_by_season[validation_season]
                + weight * mixture_predictions[validation_season]
            )
            candidate_predictions[name] = prediction
            candidates[name] = calculate_metrics(targets_by_season[validation_season], prediction)
            np.save(ARTIFACT_ROOT / f"predictions_{name}_{validation_season}.npy", prediction)
        selected = candidate_predictions[selected_name]
        np.save(ARTIFACT_ROOT / f"predictions_strict_selected_{validation_season}.npy", selected)
        validation_mask = seasons == validation_season
        folds[str(validation_season)] = {
            **expert_folds[str(validation_season)],
            "selection": {
                "source_oof_seasons": prior_seasons,
                "current_fold_labels_used": False,
                "candidate_summaries": selection,
                "selected_candidate": selected_name,
            },
            "candidates": candidates,
            "selected": {
                "candidate": selected_name,
                **calculate_metrics(targets_by_season[validation_season], selected),
            },
            "selected_segments": detailed_segments(
                diagnostics, validation_mask, targets_by_season[validation_season], selected
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
        "stage": "bounded pitch-group conditional mixture validation",
        "hypothesis": (
            "separate success functions by latent pitch group, then integrate "
            "with prior-only row-local pitch propensities"
        ),
        "validation_protocol": {
            "warmup_season": 2021,
            "report_seasons": list(REPORT_SEASONS),
            "source_seasons_strictly_prior": True,
            "current_fold_selection": False,
            "test_csv_read": False,
            "test_row_aggregation": False,
            "current_pitch_group_used_at_inference": False,
            "fixed_blend_weights": list(BLEND_WEIGHTS),
        },
        "pitch_group_audit": pitch_audit,
        "models": {
            "propensity": "EXP-030 prior-only 3-class HGB prediction",
            "success_experts": "three HistGradientBoostingClassifier models",
            "config": AUX_MODEL_CONFIG,
            "feature_count": len(selected_features),
            "features": selected_features,
        },
        "base_alignment": base_alignment,
        "warmup_fold": expert_folds["2021"],
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
            "propensities_sum_to_one": True,
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
