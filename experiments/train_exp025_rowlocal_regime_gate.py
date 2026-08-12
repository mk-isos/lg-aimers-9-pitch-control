"""EXP-025 row-local source-season similarity gate for joint experts."""

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

from train_exp017_rolling_residual import calculate_metrics, prepare_data
from train_exp019_histgb_residual import (
    season_equal_weights,
    select_stable_features,
)
from train_exp022_outcome_taxonomy_multitask import (
    REPORT_SEASONS,
    TARGET_SKILL,
    detailed_segments,
    load_frozen_base,
    load_raw_label_frame,
)


EXPERIMENT = "EXP-025"
ARTIFACT_ROOT = Path("./artifacts/EXP-025/rowlocal_regime_gate")
EXPERT_ROOT = Path("./artifacts/EXP-024/source_bagged_joint_taxonomy")
OOF_SEASONS = [2021, 2022, 2023, 2024]
SOURCE_SEASONS = [2019, 2020, 2021, 2022, 2023]
BLEND_WEIGHTS = (0.10, 0.25, 0.50)
GATE_CONFIG: dict[str, object] = {
    "learning_rate": 0.035,
    "max_iter": 120,
    "max_leaf_nodes": 7,
    "max_depth": 3,
    "min_samples_leaf": 5000,
    "l2_regularization": 30.0,
    "max_bins": 63,
    "max_features": 0.70,
    "early_stopping": False,
    "random_state": 42,
}


def build_gated_predictions(
    X: np.ndarray,
    seasons: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    predictions: dict[int, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    for validation_season in OOF_SEASONS:
        sources = [season for season in SOURCE_SEASONS if season < validation_season]
        train_mask = np.isin(seasons, sources)
        validation_mask = seasons == validation_season
        source_to_class = {season: index for index, season in enumerate(sources)}
        gate_target = np.array(
            [source_to_class.get(int(season), -1) for season in seasons],
            dtype=np.int16,
        )
        model = HistGradientBoostingClassifier(**GATE_CONFIG)
        started = time.time()
        model.fit(
            X[train_mask],
            gate_target[train_mask],
            sample_weight=season_equal_weights(seasons[train_mask]),
        )
        if not np.array_equal(model.classes_, np.arange(len(sources))):
            raise ValueError(f"gate class mismatch for {validation_season}")
        weights = model.predict_proba(X[validation_mask]).astype(float)
        experts = np.vstack(
            [
                np.load(
                    EXPERT_ROOT
                    / f"predictions_source{source}_to_{validation_season}.npy"
                ).astype(float)
                for source in sources
            ]
        ).T
        if weights.shape != experts.shape:
            raise ValueError("gate/expert shape mismatch")
        gated = np.sum(weights * experts, axis=1)
        if not np.isfinite(gated).all() or not ((gated >= 0.0) & (gated <= 1.0)).all():
            raise ValueError("invalid gated probability")
        predictions[validation_season] = gated
        np.save(
            ARTIFACT_ROOT / f"predictions_gated_success_{validation_season}.npy",
            gated,
        )
        entropy = -np.sum(weights * np.log(np.clip(weights, 1e-15, 1.0)), axis=1)
        diagnostics[str(validation_season)] = {
            "source_seasons": sources,
            "training_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "fit_predict_seconds": time.time() - started,
            "iterations_completed": int(model.n_iter_),
            "mean_source_weights": {
                str(source): float(weights[:, index].mean())
                for index, source in enumerate(sources)
            },
            "mean_gate_entropy": float(entropy.mean()),
            "mean_max_source_weight": float(weights.max(axis=1).mean()),
            "prediction_mean": float(gated.mean()),
        }
        print(
            f"gate {validation_season}: sources={sources} "
            f"mean={gated.mean():.4f} max_weight={weights.max(axis=1).mean():.3f}"
        )
        del model, weights, experts
        gc.collect()
    return predictions, diagnostics


def choose_prior_weight(
    validation_season: int,
    gated_by_season: dict[int, np.ndarray],
    base_by_season: dict[int, np.ndarray],
    targets_by_season: dict[int, np.ndarray],
) -> tuple[float, dict[str, object]]:
    source_oof = [season for season in OOF_SEASONS if season < validation_season]
    candidates: dict[str, object] = {}
    for weight in BLEND_WEIGHTS:
        skills: list[float] = []
        season_metrics: dict[str, object] = {}
        for season in source_oof:
            prediction = (
                (1.0 - weight) * base_by_season[season]
                + weight * gated_by_season[season]
            )
            metrics = calculate_metrics(targets_by_season[season], prediction)
            season_metrics[str(season)] = metrics
            skills.append(float(metrics["skill_score_unclipped"]))
        key = f"w{int(weight * 100):03d}"
        candidates[key] = {
            "weight": weight,
            "season_metrics": season_metrics,
            "min_skill": float(np.min(skills)),
            "mean_skill": float(np.mean(skills)),
        }
    selected = max(
        BLEND_WEIGHTS,
        key=lambda weight: (
            float(candidates[f"w{int(weight * 100):03d}"]["min_skill"]),
            float(candidates[f"w{int(weight * 100):03d}"]["mean_skill"]),
            -weight,
        ),
    )
    return selected, {
        "source_oof_seasons": source_oof,
        "current_fold_labels_used": False,
        "candidates": candidates,
        "selected_weight": selected,
    }


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw = load_raw_label_frame()
    diagnostics, full_X, y, _unused_base, seasons, feature_names = prepare_data()
    del _unused_base
    if not np.array_equal(raw["control_success"].to_numpy(dtype=np.float32), y):
        raise ValueError("raw/prepared target order mismatch")
    diagnostics = diagnostics.copy()
    diagnostics["game_type"] = raw["game_type"].to_numpy()
    X, selected_features = select_stable_features(full_X, feature_names)
    del raw, full_X
    gc.collect()
    base_by_season, targets_by_season, base_alignment = load_frozen_base(
        y,
        seasons,
        OOF_SEASONS,
    )
    gated_by_season, gate_diagnostics = build_gated_predictions(X, seasons)
    del X
    gc.collect()

    folds: dict[str, object] = {}
    for validation_season in REPORT_SEASONS:
        selected_weight, selection = choose_prior_weight(
            validation_season,
            gated_by_season,
            base_by_season,
            targets_by_season,
        )
        candidates: dict[str, object] = {}
        candidate_predictions: dict[float, np.ndarray] = {}
        for weight in BLEND_WEIGHTS:
            prediction = (
                (1.0 - weight) * base_by_season[validation_season]
                + weight * gated_by_season[validation_season]
            )
            candidate_predictions[weight] = prediction
            candidates[f"w{int(weight * 100):03d}"] = calculate_metrics(
                targets_by_season[validation_season], prediction
            )
            np.save(
                ARTIFACT_ROOT
                / f"predictions_blend_w{int(weight * 100):03d}_{validation_season}.npy",
                prediction,
            )
        selected = candidate_predictions[selected_weight]
        np.save(
            ARTIFACT_ROOT / f"predictions_strict_selected_{validation_season}.npy",
            selected,
        )
        mask = seasons == validation_season
        folds[str(validation_season)] = {
            "validation_season": validation_season,
            "base": calculate_metrics(
                targets_by_season[validation_season],
                base_by_season[validation_season],
            ),
            "gated_direct": calculate_metrics(
                targets_by_season[validation_season],
                gated_by_season[validation_season],
            ),
            "gate_diagnostics": gate_diagnostics[str(validation_season)],
            "selection": selection,
            "candidates": candidates,
            "selected": {
                "weight": selected_weight,
                **calculate_metrics(targets_by_season[validation_season], selected),
            },
            "selected_segments": detailed_segments(
                diagnostics,
                mask,
                targets_by_season[validation_season],
                selected,
            ),
        }
        print(
            f"selected {validation_season}: weight={selected_weight:.2f} "
            f"base={folds[str(validation_season)]['base']['skill_score_unclipped']:.2f} "
            f"skill={folds[str(validation_season)]['selected']['skill_score_unclipped']:.2f}"
        )

    selected_skills = {
        str(season): float(folds[str(season)]["selected"]["skill_score_unclipped"])
        for season in REPORT_SEASONS
    }
    base_skills = {
        str(season): float(folds[str(season)]["base"]["skill_score_unclipped"])
        for season in REPORT_SEASONS
    }
    each_1100 = all(value >= TARGET_SKILL for value in selected_skills.values())
    no_regression = all(
        selected_skills[str(season)] >= base_skills[str(season)]
        for season in REPORT_SEASONS
    )
    uniform_passed = bool(each_1100 and no_regression)
    aggregate = {
        "selected_season_skills": selected_skills,
        "base_season_skills": base_skills,
        "mean_skill": float(np.mean(list(selected_skills.values()))),
        "min_skill": float(np.min(list(selected_skills.values()))),
        "latest_2024_skill": selected_skills["2024"],
        "each_reported_season_skill_at_least_1100": each_1100,
        "no_reported_season_regresses_vs_exp021_strict": no_regression,
        "uniform_1100_passed": uniform_passed,
        "final_fit_authorized": uniform_passed,
        "zip_creation_authorized": uniform_passed,
    }
    result = {
        "experiment": EXPERIMENT,
        "stage": "rowlocal_source_season_similarity_gate",
        "validation_protocol": {
            "source_experts": "EXP-024 source-season joint multiclass",
            "oof_seasons": OOF_SEASONS,
            "reported_seasons": REPORT_SEASONS,
            "immutable_base": "EXP-021 fixed lowrank_s300_r6 temporal OOF",
            "gate_target": "past source season identity",
            "current_fold_labels_used_for_gate": False,
            "current_fold_labels_used_for_blend_selection": False,
            "validation_or_test_row_aggregation": False,
            "gate_is_row_local": True,
            "calibration": "identity",
        },
        "gate_model": {
            "class": "HistGradientBoostingClassifier",
            "config": GATE_CONFIG,
            "feature_count": len(selected_features),
            "features": selected_features,
            "fixed_blend_weights": BLEND_WEIGHTS,
        },
        "base_alignment": base_alignment,
        "gate_folds": gate_diagnostics,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "qa": {
            "gate_source_seasons_strictly_prior": True,
            "expert_source_seasons_strictly_prior": True,
            "current_fold_selection_false": True,
            "test_row_aggregation_false": True,
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
    output = ARTIFACT_ROOT / "validation_metrics.json"
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"saved={output} uniform_1100={uniform_passed}")


if __name__ == "__main__":
    main()
