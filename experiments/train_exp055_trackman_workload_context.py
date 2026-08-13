"""EXP-055: historical Trackman workload/context residual.

Past Trackman game logs provide pitcher workload distributions such as the
expected pitcher-game pitch number and pitch-of-PA at the current official
inning/count.  Only a frozen exact pitcher map and seasons before each outer
fold are used; evaluation rows remain independent.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_exp017_rolling_residual import calculate_metrics
from train_exp041_exact_game_trackman_sequence import (
    exact_aligned_rows,
    mapping_from_aligned,
)
from train_exp043_exact_pitchtype_control_eb import load_main


DATA_DIR = Path("./data")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
DIRECT_ROOT = Path("./artifacts/EXP-050/exact_dual_propensity_control")
ARTIFACT_DIR = Path("./artifacts/EXP-055/trackman_workload_context")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
CONTEXT_SMOOTHING = 100.0
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "workload_ridge_w025",
    "workload_ridge_w050",
    "workload_lgb_w025",
    "workload_blend_w025",
)


def load_trackman() -> pd.DataFrame:
    columns = [
        "season",
        "trackman_game_id",
        "pitch_no",
        "pitch_of_pa",
        "inning",
        "balls_before",
        "strikes_before",
        "pitcher_trackman_id",
        "batter_hand",
    ]
    frame = pd.read_csv(
        DATA_DIR / "trackman_history.csv", encoding="utf-8-sig", usecols=columns
    )
    frame = frame.sort_values(
        ["season", "trackman_game_id", "pitch_no"]
    ).reset_index(drop=True)
    frame["pitcher_game_pitch_no"] = (
        frame.groupby(
            ["season", "trackman_game_id", "pitcher_trackman_id"], sort=False
        ).cumcount()
        + 1
    )
    frame["count_index"] = (
        4 * frame["balls_before"] + frame["strikes_before"]
    ).astype(np.int8)
    frame["batter_hand_code"] = frame["batter_hand"].map(
        {"Left": 1, "Right": 2}
    )
    return frame


def smoothed_context(
    history: pd.DataFrame,
    value: str,
    context_columns: list[str],
    prior: pd.Series,
) -> pd.Series:
    keys = ["pitcher_trackman_id", *context_columns]
    stats = history.groupby(keys)[value].agg(["sum", "count"])
    pitcher = stats.index.get_level_values("pitcher_trackman_id")
    prior_values = prior.reindex(pitcher).to_numpy(float)
    return pd.Series(
        (stats["sum"].to_numpy(float) + CONTEXT_SMOOTHING * prior_values)
        / (stats["count"].to_numpy(float) + CONTEXT_SMOOTHING),
        index=stats.index,
    )


def summary_tables(history: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
    overall = history.groupby("pitcher_trackman_id").agg(
        game_pitch_mean=("pitcher_game_pitch_no", "mean"),
        game_pitch_std=("pitcher_game_pitch_no", "std"),
        pa_pitch_mean=("pitch_of_pa", "mean"),
        pa_pitch_std=("pitch_of_pa", "std"),
        history_n=("pitch_of_pa", "size"),
    )
    game_max = (
        history.groupby(
            ["season", "trackman_game_id", "pitcher_trackman_id"]
        )["pitcher_game_pitch_no"]
        .max()
        .groupby("pitcher_trackman_id")
        .agg(["mean", "std", "max"])
    )
    game_max.columns = ["game_workload_mean", "game_workload_std", "game_workload_max"]
    overall = overall.join(game_max)
    game_pitch = smoothed_context(
        history,
        "pitcher_game_pitch_no",
        ["inning", "count_index", "batter_hand_code"],
        overall["game_pitch_mean"],
    )
    pa_pitch = smoothed_context(
        history,
        "pitch_of_pa",
        ["count_index", "batter_hand_code"],
        overall["pa_pitch_mean"],
    )
    return {"overall": overall, "game_pitch": game_pitch, "pa_pitch": pa_pitch}


def attach_summary(
    rows: pd.DataFrame,
    mapped: pd.Series,
    tables: dict[str, pd.Series | pd.DataFrame],
    prefix: str,
) -> pd.DataFrame:
    overall = tables["overall"].reindex(mapped).reset_index(drop=True)
    overall.columns = [f"{prefix}_{column}" for column in overall.columns]
    game_index = pd.MultiIndex.from_arrays(
        [
            mapped,
            rows["inning"],
            rows["count_index"],
            rows["batter_hand"],
        ],
        names=[
            "pitcher_trackman_id",
            "inning",
            "count_index",
            "batter_hand_code",
        ],
    )
    pa_index = pd.MultiIndex.from_arrays(
        [mapped, rows["count_index"], rows["batter_hand"]],
        names=["pitcher_trackman_id", "count_index", "batter_hand_code"],
    )
    overall[f"{prefix}_expected_game_pitch_no"] = tables["game_pitch"].reindex(
        game_index
    ).to_numpy(float)
    overall[f"{prefix}_expected_pitch_of_pa"] = tables["pa_pitch"].reindex(
        pa_index
    ).to_numpy(float)
    return overall


def build_features(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    aligned: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cutoff = season - 1
    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    mapping, mapping_audit = mapping_from_aligned(aligned, cutoff)
    mapped = rows["pitcher_id"].map(mapping.mapping)
    all_history = trackman.loc[trackman["season"].le(cutoff)]
    last_history = trackman.loc[trackman["season"].eq(cutoff)]
    prior_history = trackman.loc[trackman["season"].lt(cutoff)]
    all_features = attach_summary(
        rows, mapped, summary_tables(all_history), "work_all"
    )
    last_features = attach_summary(
        rows, mapped, summary_tables(last_history), "work_last"
    )
    prior_features = attach_summary(
        rows, mapped, summary_tables(prior_history), "work_prior"
    )
    features = pd.concat([all_features, last_features, prior_features], axis=1)
    for column in all_features:
        suffix = column.removeprefix("work_all_")
        features[f"work_delta_{suffix}"] = (
            last_features[f"work_last_{suffix}"]
            - prior_features[f"work_prior_{suffix}"]
        )
    features["mapped"] = mapped.notna().astype(np.int8)
    features["inning"] = rows["inning"].to_numpy(float)
    features["count_index"] = rows["count_index"].to_numpy(float)
    features["balls_before"] = rows["balls_before"].to_numpy(float)
    features["strikes_before"] = rows["strikes_before"].to_numpy(float)
    features["batter_hand"] = rows["batter_hand"].to_numpy(float)
    features["official_success"] = rows[
        "asof_pitcher_success_rate"
    ].to_numpy(float)
    features["official_log_n"] = np.log1p(rows["asof_pitcher_n"].to_numpy(float))
    return features, {
        **mapping_audit,
        "row_mapping_coverage": float(mapped.notna().mean()),
        "trackman_history_rows": len(all_history),
        "feature_count": features.shape[1] - 1,
    }


def recent_direct_base(season: int) -> np.ndarray:
    recency = np.load(
        RECENCY_ROOT / f"predictions_recency2_{season}.npy"
    ).astype(float)
    aggressive = np.load(
        AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
    ).astype(float)
    base = 0.5 * recency + 0.5 * aggressive
    direct = (
        np.load(DIRECT_ROOT / f"predictions_pitcher_prop_w025_{season}.npy")
        - np.load(DIRECT_ROOT / f"predictions_base_{season}.npy")
    ) / 0.25
    return np.clip(base + 0.10 * direct, 0.0, 1.0)


def new_lgb() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=160,
        learning_rate=0.015,
        num_leaves=7,
        min_child_samples=3000,
        max_bin=127,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=15.0,
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def main() -> None:
    started = time.time()
    main_frame = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = exact_aligned_rows()
    features: dict[int, pd.DataFrame] = {}
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        features[season], audits[str(season)] = build_features(
            main_frame, trackman, aligned, season
        )
        targets[season] = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        base[season] = recent_direct_base(season)
        print(
            f"features {season}: coverage={audits[str(season)]['row_mapping_coverage']:.3f}",
            flush=True,
        )
    names = [column for column in features[2021] if column != "mapped"]
    residual = {season: targets[season] - base[season] for season in EVALUATED_SEASONS}
    for season in residual:
        residual[season] -= residual[season].mean()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        sources = [season for season in EVALUATED_SEASONS if season < validation_season]
        ridge_correction = np.zeros(len(targets[validation_season]))
        lgb_correction = np.zeros(len(targets[validation_season]))
        if sources:
            train_x = pd.concat([features[s] for s in sources], ignore_index=True)
            train_y = np.concatenate([residual[s] for s in sources])
            source = np.concatenate([np.full(len(features[s]), s) for s in sources])
            eligible = train_x["mapped"].eq(1).to_numpy()
            valid = features[validation_season]["mapped"].eq(1).to_numpy()
            counts = pd.Series(source[eligible]).value_counts()
            weight = np.array([1.0 / counts[value] for value in source[eligible]])
            weight *= len(weight) / weight.sum()
            ridge = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                Ridge(alpha=1000.0),
            )
            ridge.fit(
                train_x.loc[eligible, names],
                train_y[eligible],
                ridge__sample_weight=weight,
            )
            ridge_correction[valid] = ridge.predict(
                features[validation_season].loc[valid, names]
            )
            model = new_lgb()
            model.fit(
                train_x.loc[eligible, names], train_y[eligible], sample_weight=weight
            )
            lgb_correction[valid] = model.predict(
                features[validation_season].loc[valid, names]
            )
        ridge_correction = np.clip(ridge_correction, -CORRECTION_CLIP, CORRECTION_CLIP)
        lgb_correction = np.clip(lgb_correction, -CORRECTION_CLIP, CORRECTION_CLIP)
        predictions = {
            "base": base[validation_season],
            "workload_ridge_w025": np.clip(base[validation_season] + 0.25 * ridge_correction, 0.0, 1.0),
            "workload_ridge_w050": np.clip(base[validation_season] + 0.50 * ridge_correction, 0.0, 1.0),
            "workload_lgb_w025": np.clip(base[validation_season] + 0.25 * lgb_correction, 0.0, 1.0),
            "workload_blend_w025": np.clip(base[validation_season] + 0.125 * ridge_correction + 0.125 * lgb_correction, 0.0, 1.0),
        }
        fold: dict[str, object] = {"validation_season": validation_season, "source_oof_seasons": sources}
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(targets[validation_season], prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{validation_season}.npy", prediction)
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets[validation_season].astype(np.int8))
        folds[str(validation_season)] = fold
        print(
            f"fold {validation_season}: "
            + " ".join(f"{name}={fold[name]['skill_score_unclipped']:.2f}" for name in CANDIDATES),
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
    best = max(CANDIDATES, key=lambda name: (aggregate[name]["min_skill"], aggregate[name]["mean_skill"]))
    result = {
        "experiment": "EXP-055",
        "candidate_family": "historical_trackman_workload_context_residual",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "trackman_mapping_cutoff": "validation season-1",
            "source_residuals": "earlier OOF seasons, centered and season-equal",
            "current_fold_labels_used_for_fit_or_selection": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "model": {
            "context_smoothing": CONTEXT_SMOOTHING,
            "correction_clip": CORRECTION_CLIP,
            "feature_names": names,
            "ridge_alpha": 1000.0,
            "lightgbm": new_lgb().get_params(),
        },
        "exact_alignment": alignment_audit,
        "fold_feature_audit": audits,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_fixed_candidate": best,
            "best_min_skill": aggregate[best]["min_skill"],
            "gate_each_season_1000": bool(min(aggregate[best]["season_skills"].values()) >= 1000.0),
            "gate_mean_1050": bool(aggregate[best]["mean_skill"] >= 1050.0),
            "adopt": bool(min(aggregate[best]["season_skills"].values()) >= 1000.0),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__,
        },
        "total_seconds": time.time() - started,
    }
    (ARTIFACT_DIR / "validation_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"best={best} mean={aggregate[best]['mean_skill']:.2f} "
        f"min={aggregate[best]['min_skill']:.2f} adopt={result['selection']['adopt']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
