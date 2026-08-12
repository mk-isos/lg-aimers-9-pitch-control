"""EXP-022 temporally safe auxiliary-outcome residual experiment.

Four fixed classifiers learn reverse, middle, ball, and strike outcomes that
are reconstructed from train-only cumulative pitcher-state transitions.  The
resulting prior-season OOF probabilities feed a strongly regularized Ridge
residual on top of immutable EXP-021 strict predictions.  Same-fold fits are
diagnostic only; deployable fold selection uses labels from earlier OOF
seasons and never aggregates validation or test rows.
"""

from __future__ import annotations

import argparse
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from outcome_taxonomy_features import (
    LABEL_COLUMNS,
    REQUIRED_COLUMNS,
    assert_label_reconstruction_invariants,
    reconstruct_outcome_labels,
)
from train_exp017_rolling_residual import (
    calculate_metrics,
    prepare_data,
    segment_metrics,
)
from train_exp019_histgb_residual import (
    season_equal_weights,
    select_stable_features,
)


EXPERIMENT = "EXP-022"
DEFAULT_ARTIFACT_ROOT = Path(
    "./artifacts/EXP-022/outcome_taxonomy_multitask"
)
BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
REPORT_SEASONS = [2022, 2023, 2024]
AUXILIARY_NAMES = ("reverse", "middle", "ball", "strike")
RIDGE_ALPHA = 5000.0
CORRECTION_SCALES = (0.25, 0.50)
TARGET_SKILL = 1100.0

AUX_MODEL_CONFIG: dict[str, object] = {
    "learning_rate": 0.025,
    "max_iter": 160,
    "max_leaf_nodes": 15,
    "max_depth": 4,
    "min_samples_leaf": 3000,
    "l2_regularization": 30.0,
    "max_bins": 127,
    "max_features": 0.70,
    "early_stopping": False,
    "random_state": 42,
}


def safe_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    """Return finite metrics even for an all-one/all-zero diagnostic segment."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(predictions, dtype=float)
    actual = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    baseline = actual * (1.0 - actual)
    if baseline > 0.0:
        return calculate_metrics(y, p)
    return {
        "rows": int(len(y)),
        "actual_rate": actual,
        "prediction_mean": float(p.mean()),
        "mean_gap": float(p.mean() - actual),
        "prediction_min": float(p.min()),
        "prediction_max": float(p.max()),
        "brier_score": brier,
        "baseline_brier": baseline,
        "skill_score": None,
        "skill_score_unclipped": None,
        "diagnostic_calibration_slope": None,
        "diagnostic_calibration_intercept": None,
    }


def load_raw_label_frame() -> pd.DataFrame:
    columns = list(dict.fromkeys([*REQUIRED_COLUMNS, "game_type"]))
    return pd.read_csv(
        "./data/train.csv",
        encoding="utf-8-sig",
        usecols=columns,
    )


def load_frozen_base(
    y: np.ndarray,
    seasons: np.ndarray,
    validation_seasons: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, object]]:
    base_by_season: dict[int, np.ndarray] = {}
    targets_by_season: dict[int, np.ndarray] = {}
    alignment: dict[str, object] = {}
    for season in validation_seasons:
        mask = seasons == season
        base = np.load(
            BASE_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
        ).astype(float)
        targets = np.load(BASE_ROOT / f"targets_{season}.npy").astype(float)
        expected = y[mask].astype(float)
        if len(base) != int(mask.sum()):
            raise ValueError(f"base length mismatch for {season}")
        if not np.array_equal(targets, expected):
            raise ValueError(f"frozen target order mismatch for {season}")
        if not np.isfinite(base).all() or not ((base >= 0.0) & (base <= 1.0)).all():
            raise ValueError(f"invalid frozen base probability for {season}")
        base_by_season[season] = base
        targets_by_season[season] = targets
        alignment[str(season)] = {
            "rows": len(base),
            "targets_exact": True,
            "base_finite_and_in_range": True,
        }
    return base_by_season, targets_by_season, alignment


def auxiliary_representation(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError(f"expected four auxiliary probabilities, got {probabilities.shape}")
    reverse, middle, ball, strike = probabilities.T
    return np.column_stack(
        [
            reverse,
            middle,
            ball,
            strike,
            strike - ball,
            reverse + middle,
        ]
    ).astype(np.float64)


def build_auxiliary_oof(
    X: np.ndarray,
    seasons: np.ndarray,
    labels: pd.DataFrame,
    validation_seasons: list[int],
    artifact_root: Path,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    probabilities_by_season: dict[int, np.ndarray] = {}
    fold_diagnostics: dict[str, object] = {}
    pair_valid = labels["pair_valid"].to_numpy(dtype=bool)
    for validation_season in validation_seasons:
        validation_mask = seasons == validation_season
        fold_probabilities = np.empty((int(validation_mask.sum()), 4), dtype=float)
        target_summaries: dict[str, object] = {}
        for target_index, target_name in enumerate(AUXILIARY_NAMES):
            label_column = f"aux_{target_name}"
            label_values = labels[label_column].to_numpy(dtype=float)
            train_mask = (
                (seasons < validation_season)
                & pair_valid
                & np.isfinite(label_values)
            )
            if not train_mask.any():
                raise ValueError(
                    f"no prior auxiliary rows for {target_name} {validation_season}"
                )
            training_seasons = sorted(
                np.unique(seasons[train_mask]).astype(int).tolist()
            )
            if max(training_seasons) >= validation_season:
                raise AssertionError("auxiliary model used current/future season")
            model = HistGradientBoostingClassifier(**AUX_MODEL_CONFIG)
            fit_started = time.time()
            model.fit(
                X[train_mask],
                label_values[train_mask].astype(np.int8),
                sample_weight=season_equal_weights(seasons[train_mask]),
            )
            fit_seconds = time.time() - fit_started
            predictions = model.predict_proba(X[validation_mask])[:, 1].astype(float)
            if not np.isfinite(predictions).all():
                raise ValueError("non-finite auxiliary prediction")
            if not ((predictions >= 0.0) & (predictions <= 1.0)).all():
                raise ValueError("auxiliary prediction outside [0, 1]")
            fold_probabilities[:, target_index] = predictions
            validation_label_mask = validation_mask & pair_valid
            local_valid = pair_valid[validation_mask]
            evaluation = safe_metrics(
                label_values[validation_label_mask],
                predictions[local_valid],
            )
            target_summaries[target_name] = {
                "training_seasons": training_seasons,
                "training_rows": int(train_mask.sum()),
                "training_positive_rate": float(label_values[train_mask].mean()),
                "validation_rows": int(validation_mask.sum()),
                "validation_labeled_rows": int(validation_label_mask.sum()),
                "prediction_mean": float(predictions.mean()),
                "prediction_min": float(predictions.min()),
                "prediction_max": float(predictions.max()),
                "fit_seconds": fit_seconds,
                "iterations_completed": int(model.n_iter_),
                "labeled_validation_metrics": evaluation,
            }
            del model
            gc.collect()
        representation = auxiliary_representation(fold_probabilities)
        probabilities_by_season[validation_season] = representation
        np.save(
            artifact_root / f"auxiliary_representation_{validation_season}.npy",
            representation,
        )
        fold_diagnostics[str(validation_season)] = {
            "validation_season": validation_season,
            "representation_rows": len(representation),
            "representation_columns": [
                "p_reverse",
                "p_middle",
                "p_ball",
                "p_strike",
                "p_strike_minus_ball",
                "p_reverse_plus_middle",
            ],
            "targets": target_summaries,
        }
        print(
            f"aux {validation_season}: rows={len(representation)} "
            + " ".join(
                f"{name}={fold_probabilities[:, index].mean():.4f}"
                for index, name in enumerate(AUXILIARY_NAMES)
            )
        )
    return probabilities_by_season, fold_diagnostics


def concatenate_source(
    features_by_season: dict[int, np.ndarray],
    base_by_season: dict[int, np.ndarray],
    targets_by_season: dict[int, np.ndarray],
    source_seasons: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.concatenate([features_by_season[s] for s in source_seasons])
    base = np.concatenate([base_by_season[s] for s in source_seasons])
    targets = np.concatenate([targets_by_season[s] for s in source_seasons])
    season_vector = np.concatenate(
        [np.full(len(features_by_season[s]), s, dtype=np.int16) for s in source_seasons]
    )
    return features, base, targets, season_vector


def centered_residual(
    targets: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    residual = targets.astype(float) - base.astype(float)
    means: dict[str, float] = {}
    for season in np.unique(seasons):
        mask = seasons == season
        mean = float(residual[mask].mean())
        residual[mask] -= mean
        means[str(int(season))] = mean
    return residual, means


def fit_temporal_ridge(
    features_by_season: dict[int, np.ndarray],
    base_by_season: dict[int, np.ndarray],
    targets_by_season: dict[int, np.ndarray],
    source_seasons: list[int],
) -> tuple[StandardScaler, Ridge, dict[str, object]]:
    features, base, targets, source_vector = concatenate_source(
        features_by_season,
        base_by_season,
        targets_by_season,
        source_seasons,
    )
    residual, residual_means = centered_residual(targets, base, source_vector)
    weights = season_equal_weights(source_vector)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features, sample_weight=weights)
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
    ridge.fit(scaled, residual, sample_weight=weights)
    diagnostics = {
        "source_seasons": source_seasons,
        "source_rows": len(features),
        "source_residual_means_removed": residual_means,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coefficients": ridge.coef_.astype(float).tolist(),
        "coefficient_l2": float(np.linalg.norm(ridge.coef_)),
    }
    return scaler, ridge, diagnostics


def prior_candidate_selection(
    scaler: StandardScaler,
    ridge: Ridge,
    features_by_season: dict[int, np.ndarray],
    base_by_season: dict[int, np.ndarray],
    targets_by_season: dict[int, np.ndarray],
    source_seasons: list[int],
) -> tuple[float, dict[str, object]]:
    candidates: dict[str, object] = {}
    for scale in CORRECTION_SCALES:
        season_metrics: dict[str, object] = {}
        skills: list[float] = []
        for season in source_seasons:
            correction = ridge.predict(
                scaler.transform(features_by_season[season])
            ).astype(float)
            predictions = np.clip(
                base_by_season[season] + scale * correction,
                0.0,
                1.0,
            )
            metrics = calculate_metrics(targets_by_season[season], predictions)
            season_metrics[str(season)] = metrics
            skills.append(float(metrics["skill_score_unclipped"]))
        key = f"w{int(scale * 100):03d}"
        candidates[key] = {
            "scale": scale,
            "season_metrics": season_metrics,
            "min_skill": float(np.min(skills)),
            "mean_skill": float(np.mean(skills)),
        }
    selected_scale = max(
        CORRECTION_SCALES,
        key=lambda scale: (
            float(candidates[f"w{int(scale * 100):03d}"]["min_skill"]),
            float(candidates[f"w{int(scale * 100):03d}"]["mean_skill"]),
            -scale,
        ),
    )
    return selected_scale, {
        "selection_labels": "prior OOF seasons only",
        "source_seasons": source_seasons,
        "candidates": candidates,
        "selected_scale": selected_scale,
        "tie_break": "max min Skill, then mean Skill, then smaller scale",
    }


def samefold_diagnostics(
    features: np.ndarray,
    base: np.ndarray,
    targets: np.ndarray,
) -> dict[str, object]:
    residual = targets.astype(float) - base.astype(float)
    residual -= residual.mean()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
    ridge.fit(scaled, residual)
    fitted = np.clip(base + ridge.predict(scaled), 0.0, 1.0)

    crossfit_correction = np.zeros(len(targets), dtype=float)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, validation_index in splitter.split(features):
        fold_scaler = StandardScaler()
        train_scaled = fold_scaler.fit_transform(features[train_index])
        fold_residual = targets[train_index] - base[train_index]
        fold_residual = fold_residual - fold_residual.mean()
        fold_ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        fold_ridge.fit(train_scaled, fold_residual)
        crossfit_correction[validation_index] = fold_ridge.predict(
            fold_scaler.transform(features[validation_index])
        )
    crossfit = np.clip(base + crossfit_correction, 0.0, 1.0)
    return {
        "deployable": False,
        "current_fold_labels_used": True,
        "samefold_ridge": calculate_metrics(targets, fitted),
        "fivefold_crossfit_ridge": calculate_metrics(targets, crossfit),
    }


def detailed_segments(
    diagnostics: pd.DataFrame,
    mask: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {
        "sample_and_player_status": segment_metrics(
            diagnostics,
            mask,
            targets,
            predictions,
        )
    }
    local = diagnostics.loc[mask]
    game_type: dict[str, object] = {}
    for value in sorted(local["game_type"].dropna().unique().tolist()):
        segment = local["game_type"].to_numpy() == value
        game_type[str(value)] = safe_metrics(targets[segment], predictions[segment])
    months: dict[str, object] = {}
    for month in sorted(local["game_month"].dropna().unique().tolist()):
        segment = local["game_month"].to_numpy() == month
        months[str(int(month))] = safe_metrics(targets[segment], predictions[segment])
    result["game_type"] = game_type
    result["month"] = months
    return result


def run_temporal_residual(
    features_by_season: dict[int, np.ndarray],
    base_by_season: dict[int, np.ndarray],
    targets_by_season: dict[int, np.ndarray],
    seasons: np.ndarray,
    diagnostics: pd.DataFrame,
    validation_seasons: list[int],
    artifact_root: Path,
) -> tuple[dict[str, object], dict[int, np.ndarray]]:
    folds: dict[str, object] = {}
    selected_predictions: dict[int, np.ndarray] = {}
    for validation_season in validation_seasons:
        if validation_season == min(validation_seasons):
            continue
        source_seasons = [s for s in validation_seasons if s < validation_season]
        scaler, ridge, ridge_diagnostics = fit_temporal_ridge(
            features_by_season,
            base_by_season,
            targets_by_season,
            source_seasons,
        )
        selected_scale, selection = prior_candidate_selection(
            scaler,
            ridge,
            features_by_season,
            base_by_season,
            targets_by_season,
            source_seasons,
        )
        correction = ridge.predict(
            scaler.transform(features_by_season[validation_season])
        ).astype(float)
        candidate_metrics: dict[str, object] = {}
        candidate_predictions: dict[float, np.ndarray] = {}
        for scale in CORRECTION_SCALES:
            predictions = np.clip(
                base_by_season[validation_season] + scale * correction,
                0.0,
                1.0,
            )
            candidate_predictions[scale] = predictions
            candidate_metrics[f"w{int(scale * 100):03d}"] = calculate_metrics(
                targets_by_season[validation_season], predictions
            )
            np.save(
                artifact_root
                / f"predictions_temporal_w{int(scale * 100):03d}_{validation_season}.npy",
                predictions,
            )
        selected = candidate_predictions[selected_scale]
        selected_predictions[validation_season] = selected
        np.save(
            artifact_root / f"predictions_strict_selected_{validation_season}.npy",
            selected,
        )
        validation_mask = seasons == validation_season
        fold = {
            "validation_season": validation_season,
            "base": calculate_metrics(
                targets_by_season[validation_season],
                base_by_season[validation_season],
            ),
            "ridge": ridge_diagnostics,
            "selection": selection,
            "candidates": candidate_metrics,
            "selected": {
                "scale": selected_scale,
                **calculate_metrics(targets_by_season[validation_season], selected),
            },
            "selected_segments": detailed_segments(
                diagnostics,
                validation_mask,
                targets_by_season[validation_season],
                selected,
            ),
            "diagnostic_representation_ceiling": samefold_diagnostics(
                features_by_season[validation_season],
                base_by_season[validation_season],
                targets_by_season[validation_season],
            ),
        }
        folds[str(validation_season)] = fold
        print(
            f"temporal {validation_season}: selected={selected_scale:.2f} "
            f"base={fold['base']['skill_score_unclipped']:.2f} "
            f"selected_skill={fold['selected']['skill_score_unclipped']:.2f} "
            f"samefold={fold['diagnostic_representation_ceiling']['samefold_ridge']['skill_score_unclipped']:.2f}"
        )
    return folds, selected_predictions


def aggregate_and_gate(folds: dict[str, object]) -> dict[str, object]:
    available_report = [season for season in REPORT_SEASONS if str(season) in folds]
    selected_skills = {
        str(season): float(folds[str(season)]["selected"]["skill_score_unclipped"])
        for season in available_report
    }
    base_skills = {
        str(season): float(folds[str(season)]["base"]["skill_score_unclipped"])
        for season in available_report
    }
    thresholds = {
        str(season): float(
            folds[str(season)]["selected"]["baseline_brier"]
            * (1.0 - TARGET_SKILL / 100000.0)
        )
        for season in available_report
    }
    complete = available_report == REPORT_SEASONS
    each_1100 = complete and all(score >= TARGET_SKILL for score in selected_skills.values())
    no_regression = complete and all(
        selected_skills[str(season)] >= base_skills[str(season)]
        for season in REPORT_SEASONS
    )
    samefold_2023_2024 = {
        str(season): float(
            folds[str(season)]["diagnostic_representation_ceiling"]["samefold_ridge"][
                "skill_score_unclipped"
            ]
        )
        for season in (2023, 2024)
        if str(season) in folds
    }
    linear_family_ceiling_passed = len(samefold_2023_2024) == 2 and all(
        score >= TARGET_SKILL for score in samefold_2023_2024.values()
    )
    uniform_passed = bool(each_1100 and no_regression)
    skills = list(selected_skills.values())
    return {
        "reported_seasons_complete": complete,
        "selected_season_skills": selected_skills,
        "base_season_skills": base_skills,
        "skill_1100_brier_thresholds": thresholds,
        "mean_skill": float(np.mean(skills)) if skills else None,
        "min_skill": float(np.min(skills)) if skills else None,
        "latest_2024_skill": selected_skills.get("2024"),
        "each_reported_season_skill_at_least_1100": each_1100,
        "no_reported_season_regresses_vs_exp021_strict": no_regression,
        "samefold_linear_family_2023_2024_skills": samefold_2023_2024,
        "linear_family_samefold_ceiling_passed": linear_family_ceiling_passed,
        "uniform_1100_passed": uniform_passed,
        "final_fit_authorized": uniform_passed,
        "zip_creation_authorized": uniform_passed,
        "stop_linear_family": not linear_family_ceiling_passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument(
        "--validation-seasons",
        type=int,
        nargs="+",
        default=VALIDATION_SEASONS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_seasons = sorted(args.validation_seasons)
    if validation_seasons[0] != 2021:
        raise ValueError("2021 warm-up OOF is required")
    artifact_root: Path = args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    raw = load_raw_label_frame()
    labels, label_audit = reconstruct_outcome_labels(raw)
    assert_label_reconstruction_invariants(raw, labels, label_audit)
    diagnostics, full_X, y, _unused_base, seasons, feature_names = prepare_data()
    del _unused_base
    raw_targets = raw["control_success"].to_numpy(dtype=np.float32)
    if len(raw) != len(y) or not np.array_equal(raw_targets, y):
        raise ValueError("raw train and prepared feature target order mismatch")
    diagnostics = diagnostics.copy()
    diagnostics["game_type"] = raw["game_type"].to_numpy()
    X, selected_features = select_stable_features(full_X, feature_names)
    del full_X, raw
    gc.collect()
    base_by_season, targets_by_season, base_alignment = load_frozen_base(
        y,
        seasons,
        validation_seasons,
    )
    print(
        f"rows={len(y)} stable_features={len(selected_features)} "
        f"valid_aux_pairs={label_audit['valid_pair_rows']}"
    )

    features_by_season, auxiliary_folds = build_auxiliary_oof(
        X,
        seasons,
        labels,
        validation_seasons,
        artifact_root,
    )
    del X, labels
    gc.collect()
    folds, _selected_predictions = run_temporal_residual(
        features_by_season,
        base_by_season,
        targets_by_season,
        seasons,
        diagnostics,
        validation_seasons,
        artifact_root,
    )
    aggregate = aggregate_and_gate(folds)
    result = {
        "experiment": EXPERIMENT,
        "stage": "outcome_taxonomy_temporal_multitask",
        "validation_protocol": {
            "warmup_season": 2021,
            "reported_seasons": REPORT_SEASONS,
            "evaluated_seasons": validation_seasons,
            "immutable_base": "EXP-021 fixed lowrank_s300_r6 temporal OOF",
            "auxiliary_label_source": (
                "train-only unique same-pitcher same-season n-to-n+1 cumulative deltas"
            ),
            "current_fold_labels_used_for_auxiliary_training": False,
            "current_fold_labels_used_for_temporal_ridge": False,
            "current_fold_labels_used_for_candidate_selection": False,
            "samefold_diagnostics_deployable": False,
            "test_csv_read": False,
            "test_row_aggregation": False,
            "calibration": "identity; no affine, isotonic, sigmoid, or fixed offset",
        },
        "auxiliary_label_audit": label_audit,
        "auxiliary_model": {
            "class": "HistGradientBoostingClassifier",
            "targets": AUXILIARY_NAMES,
            "config": AUX_MODEL_CONFIG,
            "feature_count": len(selected_features),
            "features": selected_features,
            "forbidden_features_absent": True,
        },
        "ridge_model": {
            "class": "Ridge",
            "alpha": RIDGE_ALPHA,
            "fit_intercept": False,
            "fixed_correction_scales": CORRECTION_SCALES,
            "representation_columns": [
                "p_reverse",
                "p_middle",
                "p_ball",
                "p_strike",
                "p_strike_minus_ball",
                "p_reverse_plus_middle",
            ],
        },
        "base_alignment": base_alignment,
        "auxiliary_folds": auxiliary_folds,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "qa": {
            "success_mismatch_zero": label_audit["success_mismatch_count"] == 0,
            "row_order_independent_label_lookup": True,
            "frozen_base_target_order_exact": True,
            "source_seasons_strictly_prior": True,
            "probabilities_finite_and_in_range": True,
            "current_fold_selection_false": True,
            "test_row_aggregation_false": True,
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
    output_path = artifact_root / "validation_metrics.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        f"saved={output_path} uniform_1100={aggregate['uniform_1100_passed']} "
        f"final_fit_authorized={aggregate['final_fit_authorized']}"
    )


if __name__ == "__main__":
    main()
