"""EXP-063: temporal residual experts for predictions near 0.5.

EXP-060 shows that reweighting existing candidates cannot reach Skill 1000 in
2023/2024.  Error decomposition localizes most remaining loss to rows where
the EXP-051 probability is close to 0.5.  This experiment trains a shallow,
ID-free residual model only on the analogous uncertain rows from prior OOF
seasons and leaves confident rows bitwise unchanged.
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

from temporal_multirate_features import attach_training_multirate_features
from temporal_residual_features import (
    add_static_features,
    attach_training_temporal_features,
)
from train_exp017_rolling_residual import calculate_metrics


DATA_DIR = Path("./data")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
TRACKMAN_ROOT = Path("./artifacts/EXP-043/exact_pitchtype_control_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-063/uncertain_region_residual")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
FEATURES = (
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_hand",
    "batter_hand",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "count_index",
    "count_advantage",
    "same_hand",
    "runner_in_scoring_position",
    "bases_loaded",
    "late_inning",
    "close_game",
    "log_li",
    "score_pressure",
    "pitcher_recent_success_delta_1_5",
    "pitcher_recent_success_delta_3_5",
    "pitcher_recent_middle_delta_1_5",
    "temporal_pitcher_log_season_n",
    "temporal_pitcher_season_global_30",
    "temporal_pitcher_season_player_30",
    "temporal_pitcher_season_minus_prior_rate",
    "temporal_batter_log_season_n",
    "temporal_batter_season_global_30",
    "multirate_pitcher_control_log_season_n",
    "multirate_pitcher_control_success_season_global_30",
    "multirate_pitcher_control_reverse_season_global_30",
    "multirate_pitcher_control_middle_season_global_30",
    "multirate_pitcher_control_ball_season_global_30",
    "multirate_pitcher_control_strike_season_global_30",
    "multirate_pitcher_pitchmix_fastball_season_global_30",
    "multirate_pitcher_pitchmix_breaking_season_global_30",
    "multirate_pitcher_pitchmix_offspeed_season_global_30",
)
CANDIDATES = (
    "close020_w025",
    "close040_w025",
    "close060_w025",
    "close060_w050",
    "close060_last_w025",
)
CORRECTION_CLIP = 0.03


def exp051_base(season: int) -> np.ndarray:
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
        TRACKMAN_ROOT / f"predictions_fine_direct_w025_{season}.npy"
    ).astype(float)
    return np.clip(
        0.5 * recency + 0.5 * aggressive + 0.10 * (trackman - strict) / 0.25,
        0.0,
        1.0,
    )


def new_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=200,
        learning_rate=0.015,
        num_leaves=7,
        min_child_samples=5000,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=20.0,
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def main() -> None:
    started = time.time()
    raw = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    frame, _ = attach_training_temporal_features(add_static_features(raw))
    frame, _, multirate_audit = attach_training_multirate_features(frame)
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"missing features: {missing}")
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    x: dict[int, pd.DataFrame] = {}
    for season in EVALUATED_SEASONS:
        mask = frame["season"].eq(season).to_numpy()
        targets[season] = np.load(
            LOWRANK_ROOT / f"targets_{season}.npy"
        ).astype(float)
        base[season] = exp051_base(season)
        x[season] = frame.loc[mask, list(FEATURES)].reset_index(drop=True)
        if not np.array_equal(
            targets[season], frame.loc[mask, "control_success"].to_numpy(float)
        ):
            raise ValueError(f"target/order mismatch {season}")
    residual = {season: targets[season] - base[season] for season in EVALUATED_SEASONS}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        sources = [season for season in EVALUATED_SEASONS if season < validation_season]
        corrections: dict[float, np.ndarray] = {
            threshold: np.zeros(len(targets[validation_season]))
            for threshold in (0.02, 0.04, 0.06)
        }
        last_correction = np.zeros(len(targets[validation_season]))
        fit_audit: dict[str, object] = {}
        for threshold in corrections:
            if not sources:
                continue
            eligible_parts = [np.abs(base[s] - 0.5) < threshold for s in sources]
            train_x = pd.concat(
                [x[s].loc[eligible_parts[i]] for i, s in enumerate(sources)],
                ignore_index=True,
            )
            centered_parts = []
            source_values = []
            for i, season in enumerate(sources):
                value = residual[season][eligible_parts[i]].copy()
                value -= value.mean()
                centered_parts.append(value)
                source_values.append(np.full(len(value), season))
            train_y = np.concatenate(centered_parts)
            source = np.concatenate(source_values)
            counts = pd.Series(source).value_counts()
            sample_weight = np.array([1.0 / counts[value] for value in source])
            sample_weight *= len(sample_weight) / sample_weight.sum()
            model = new_model()
            model.fit(train_x, train_y, sample_weight=sample_weight)
            valid = np.abs(base[validation_season] - 0.5) < threshold
            corrections[threshold][valid] = np.clip(
                model.predict(x[validation_season].loc[valid]),
                -CORRECTION_CLIP,
                CORRECTION_CLIP,
            )
            fit_audit[f"close_{threshold:.2f}"] = {
                "source_rows": {str(value): int(counts[value]) for value in sources},
                "validation_rows": int(valid.sum()),
            }
            if threshold == 0.06:
                last_source = sources[-1]
                last_eligible = np.abs(base[last_source] - 0.5) < threshold
                last_y = residual[last_source][last_eligible].copy()
                last_y -= last_y.mean()
                last_model = new_model()
                last_model.fit(x[last_source].loc[last_eligible], last_y)
                last_correction[valid] = np.clip(
                    last_model.predict(x[validation_season].loc[valid]),
                    -CORRECTION_CLIP,
                    CORRECTION_CLIP,
                )
        predictions = {
            "base": base[validation_season],
            "close020_w025": np.clip(
                base[validation_season] + 0.25 * corrections[0.02], 0, 1
            ),
            "close040_w025": np.clip(
                base[validation_season] + 0.25 * corrections[0.04], 0, 1
            ),
            "close060_w025": np.clip(
                base[validation_season] + 0.25 * corrections[0.06], 0, 1
            ),
            "close060_w050": np.clip(
                base[validation_season] + 0.50 * corrections[0.06], 0, 1
            ),
            "close060_last_w025": np.clip(
                base[validation_season] + 0.25 * last_correction, 0, 1
            ),
        }
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_seasons": sources,
            "fit_audit": fit_audit,
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(targets[validation_season], prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{validation_season}.npy", prediction)
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets[validation_season].astype(np.int8),
        )
        folds[str(validation_season)] = fold
        print(
            f"fold {validation_season}: "
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
        "experiment": "EXP-063",
        "candidate_family": "id_free_uncertain_region_temporal_residual",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "current_fold_labels_used_for_fit_or_selection": False,
            "source_season_residual_centering": True,
            "source_season_equal_weight": True,
            "test_row_aggregation": False,
            "raw_player_or_team_ids": False,
            "candidate_grid_predeclared": False,
            "last_source_variant_posthoc_bounded": True,
        },
        "model": {
            "base": "EXP-051 trackman_direct_recent_w010",
            "feature_count": len(FEATURES),
            "features": list(FEATURES),
            "uncertainty_thresholds": [0.02, 0.04, 0.06],
            "correction_clip": CORRECTION_CLIP,
            "lightgbm": new_model().get_params(),
        },
        "multirate_audit": multirate_audit,
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
            "lightgbm": lgb.__version__,
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
