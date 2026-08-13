"""EXP-039: prior-OOF nonlinear stack of four deployable experts.

The four components are already serializable from EXP-032: strict rank-6,
R-specific rank-4, aggressive R/F pitcher-count EB, and recency2 rank-6.
A strongly regularized LightGBM learns season-centered residuals from earlier
OOF folds using component predictions/disagreements plus current-row official
numeric context.  No validation label or test-row aggregate enters fitting.
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


DATA_PATH = Path("./data/train.csv")
ARTIFACT_DIR = Path("./artifacts/EXP-039/temporal_expert_stack")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
CORRECTION_WEIGHTS = (0.25, 0.50, 1.0)
CORRECTION_CLIP = 0.03

DROP_COLUMNS = {
    "row_id",
    "season",
    "control_success",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "top_bottom",
    "game_type",
    "base_state",
}


def load_rows() -> dict[int, pd.DataFrame]:
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    frame["is_regular"] = frame["game_type"].astype(str).eq("R").astype(np.int8)
    frame["is_top"] = frame["top_bottom"].astype(str).eq("T").astype(np.int8)
    frame["count_index"] = (
        frame["balls_before"] * 4 + frame["strikes_before"]
    ).astype(np.int8)
    frame["same_hand"] = frame["pitcher_hand"].eq(frame["batter_hand"]).astype(
        np.int8
    )
    frame["score_pressure"] = (
        frame["score_diff_pitcher_team"].abs().le(1).astype(np.int8)
        * frame["li"].astype(float)
    )
    numeric = [
        column
        for column in frame.columns
        if column not in DROP_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    keep = ["season", "control_success", *numeric]
    frame = frame[keep]
    return {
        season: frame.loc[frame["season"].eq(season)].reset_index(drop=True)
        for season in EVALUATED_SEASONS
    }


def load_components(
    season: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
    components = {
        "strict": np.load(
            LOWRANK_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
        ).astype(float),
        "r_specific": np.load(
            LOWRANK_ROOT
            / f"predictions_lowrank_s300_r4_Rspecific_{season}.npy"
        ).astype(float),
        "aggressive": np.load(
            AGGRESSIVE_ROOT
            / f"predictions_r_gated_team_pc_all_{season}.npy"
        ).astype(float),
        "recency": np.load(
            RECENCY_ROOT / f"predictions_recency2_{season}.npy"
        ).astype(float),
    }
    if len({len(target), *(len(value) for value in components.values())}) != 1:
        raise ValueError(f"component length mismatch {season}")
    return target, components


def build_features(
    official: pd.DataFrame, components: dict[str, np.ndarray]
) -> pd.DataFrame:
    feature = official.drop(columns=["season", "control_success"]).copy()
    for name, prediction in components.items():
        feature[f"expert_{name}"] = prediction
    matrix = np.column_stack(list(components.values()))
    feature["expert_mean"] = matrix.mean(axis=1)
    feature["expert_std"] = matrix.std(axis=1)
    feature["expert_min"] = matrix.min(axis=1)
    feature["expert_max"] = matrix.max(axis=1)
    feature["expert_range"] = matrix.max(axis=1) - matrix.min(axis=1)
    names = list(components)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            feature[f"expert_diff_{left}_{right}"] = (
                components[left] - components[right]
            )
    return feature.astype(np.float32)


def main() -> None:
    started = time.time()
    rows = load_rows()
    targets: dict[int, np.ndarray] = {}
    components: dict[int, dict[str, np.ndarray]] = {}
    features: dict[int, pd.DataFrame] = {}
    residual: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        targets[season], components[season] = load_components(season)
        csv_target = rows[season]["control_success"].to_numpy(dtype=float)
        if not np.array_equal(csv_target, targets[season]):
            raise ValueError(f"target/order mismatch {season}")
        features[season] = build_features(rows[season], components[season])
        residual[season] = targets[season] - components[season]["strict"]
        residual[season] -= residual[season].mean()
    feature_names = list(features[2021].columns)
    if any(list(features[s].columns) != feature_names for s in EVALUATED_SEASONS):
        raise ValueError("stack feature schema drift")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        sources = [season for season in EVALUATED_SEASONS if season < validation_season]
        correction = np.zeros(len(targets[validation_season]), dtype=float)
        fit_seconds = 0.0
        feature_importance: dict[str, int] = {}
        if sources:
            train_x = pd.concat([features[s] for s in sources], ignore_index=True)
            train_y = np.concatenate([residual[s] for s in sources])
            source_labels = np.concatenate(
                [np.full(len(residual[s]), s, dtype=np.int16) for s in sources]
            )
            counts = pd.Series(source_labels).value_counts()
            weights = np.array(
                [1.0 / counts[int(value)] for value in source_labels], dtype=float
            )
            weights *= len(weights) / weights.sum()
            model = LGBMRegressor(
                objective="regression_l2",
                n_estimators=200,
                learning_rate=0.015,
                num_leaves=7,
                min_child_samples=5000,
                max_bin=127,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=16.0,
                random_state=42,
                n_jobs=-1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            fit_started = time.time()
            model.fit(train_x, train_y, sample_weight=weights)
            fit_seconds = time.time() - fit_started
            correction = np.clip(
                model.predict(features[validation_season]),
                -CORRECTION_CLIP,
                CORRECTION_CLIP,
            )
            feature_importance = {
                name: int(value)
                for name, value in sorted(
                    zip(feature_names, model.booster_.feature_importance(), strict=True),
                    key=lambda pair: (-pair[1], pair[0]),
                )
                if value > 0
            }
        predictions = {"base_strict": components[validation_season]["strict"]}
        for weight in CORRECTION_WEIGHTS:
            name = f"stack_w{int(weight * 100):03d}"
            predictions[name] = np.clip(
                components[validation_season]["strict"] + weight * correction,
                0.0,
                1.0,
            )
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_oof_seasons": sources,
            "fit_seconds": fit_seconds,
            "feature_importance": feature_importance,
        }
        for name, prediction in predictions.items():
            if not np.isfinite(prediction).all() or not (
                (prediction >= 0.0).all() and (prediction <= 1.0).all()
            ):
                raise ValueError(f"invalid prediction {validation_season} {name}")
            fold[name] = calculate_metrics(targets[validation_season], prediction)
            np.save(
                ARTIFACT_DIR / f"predictions_{name}_{validation_season}.npy",
                prediction,
            )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets[validation_season].astype(np.int8),
        )
        np.save(ARTIFACT_DIR / f"correction_{validation_season}.npy", correction)
        folds[str(validation_season)] = fold
        print(
            f"fold {validation_season}: "
            + " ".join(
                f"w{int(weight*100):03d}="
                f"{fold[f'stack_w{int(weight*100):03d}']['skill_score_unclipped']:.2f}"
                for weight in CORRECTION_WEIGHTS
            ),
            flush=True,
        )

    candidate_names = tuple(
        f"stack_w{int(weight * 100):03d}" for weight in CORRECTION_WEIGHTS
    )
    aggregate: dict[str, object] = {}
    for candidate in ("base_strict", *candidate_names):
        skills = {
            str(season): float(
                folds[str(season)][candidate]["skill_score_unclipped"]
            )
            for season in REPORT_SEASONS
        }
        briers = {
            str(season): float(folds[str(season)][candidate]["brier_score"])
            for season in REPORT_SEASONS
        }
        aggregate[candidate] = {
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
            aggregate[name]["latest_2024_skill"],
            aggregate[name]["mean_skill"],
        ),
    )
    result = {
        "experiment": "EXP-039",
        "candidate_family": "prior_oof_deployable_expert_stack",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "source_training": "earlier OOF seasons only; season-centered and season-equal",
            "current_fold_labels_used_for_training_or_selection": False,
            "test_row_aggregation": False,
            "candidate_weights_predeclared": True,
        },
        "model": {
            "components": ["strict", "r_specific", "aggressive", "recency"],
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "iterations": 200,
            "learning_rate": 0.015,
            "num_leaves": 7,
            "min_child_samples": 5000,
            "correction_clip": CORRECTION_CLIP,
            "correction_weights": list(CORRECTION_WEIGHTS),
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_fixed_candidate": best,
            "best_min_skill": aggregate[best]["min_skill"],
            "gate_each_season_1000": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
            ),
            "gate_mean_1050": bool(aggregate[best]["mean_skill"] >= 1050.0),
            "adopt_for_full_fit": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
                and aggregate[best]["mean_skill"] >= 1050.0
            ),
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
        f"min={aggregate[best]['min_skill']:.2f} "
        f"adopt={result['selection']['adopt_for_full_fit']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
