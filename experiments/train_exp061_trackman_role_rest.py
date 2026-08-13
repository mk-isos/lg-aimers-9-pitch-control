"""EXP-061: past TrackMan pitcher role, rest and roster-transition residual.

The TrackMan workload experiment used expected within-game pitch number but
did not explicitly model starter/reliever role, appearance rest, season
availability or team transitions.  This experiment summarizes those signals
through validation season-1 and maps them to each current row independently.
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

from trackman_features import (
    TEAM_ID_TO_TRACKMAN,
    TRACKMAN_TEAM_ALIASES,
)
from train_exp017_rolling_residual import calculate_metrics
from train_exp041_exact_game_trackman_sequence import (
    exact_aligned_rows,
    mapping_from_aligned,
)


DATA_DIR = Path("./data")
LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
AGGRESSIVE_ROOT = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
RECENCY_ROOT = Path("./artifacts/EXP-037/lowrank_source_policies")
TRACKMAN_CONTROL_ROOT = Path("./artifacts/EXP-043/exact_pitchtype_control_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-061/trackman_role_rest")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "role_ridge_a1000_w025",
    "role_ridge_a10000_w025",
    "role_lgb_w025",
    "role_lgb_last_w025",
)


def load_main() -> pd.DataFrame:
    columns = [
        "season",
        "game_month",
        "inning",
        "game_type",
        "balls_before",
        "strikes_before",
        "pitcher_id",
        "pitcher_team_id",
        "pitcher_hand",
        "batter_hand",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "control_success",
    ]
    frame = pd.read_csv(
        DATA_DIR / "train.csv", encoding="utf-8-sig", usecols=columns
    )
    frame["count_index"] = (
        4 * frame["balls_before"] + frame["strikes_before"]
    ).astype(np.int8)
    return frame


def load_trackman_games() -> pd.DataFrame:
    columns = [
        "season",
        "game_date",
        "game_month",
        "trackman_game_id",
        "pitch_no",
        "inning",
        "pitcher_trackman_id",
        "pitcher_team",
    ]
    frame = pd.read_csv(
        DATA_DIR / "trackman_history.csv",
        encoding="utf-8-sig",
        usecols=columns,
    )
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["pitcher_team"] = frame["pitcher_team"].replace(
        TRACKMAN_TEAM_ALIASES
    )
    games = (
        frame.groupby(
            [
                "season",
                "trackman_game_id",
                "pitcher_trackman_id",
            ],
            observed=True,
            sort=False,
        )
        .agg(
            game_date=("game_date", "first"),
            game_month=("game_month", "first"),
            pitcher_team=("pitcher_team", lambda value: value.mode().iloc[0]),
            first_inning=("inning", "min"),
            last_inning=("inning", "max"),
            pitches=("pitch_no", "size"),
        )
        .reset_index()
    )
    games["inning_span"] = games["last_inning"] - games["first_inning"]
    games["starter"] = games["first_inning"].le(2).astype(np.int8)
    games["late_relief"] = games["first_inning"].ge(7).astype(np.int8)
    games["multi_inning"] = games["inning_span"].ge(1).astype(np.int8)
    games["work_50"] = games["pitches"].ge(50).astype(np.int8)
    games["work_80"] = games["pitches"].ge(80).astype(np.int8)
    games = games.sort_values(
        ["pitcher_trackman_id", "season", "game_date", "trackman_game_id"]
    ).reset_index(drop=True)
    games["rest_days"] = games.groupby(
        ["pitcher_trackman_id", "season"], observed=True
    )["game_date"].diff().dt.days
    games["short_rest"] = games["rest_days"].le(3).astype(np.int8)
    games["long_rest"] = games["rest_days"].ge(7).astype(np.int8)
    return games


def summarize(games: pd.DataFrame) -> pd.DataFrame:
    grouped = games.groupby("pitcher_trackman_id", observed=True, sort=True)
    summary = grouped.agg(
        games=("trackman_game_id", "size"),
        seasons=("season", "nunique"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        first_inning_mean=("first_inning", "mean"),
        first_inning_std=("first_inning", "std"),
        last_inning_mean=("last_inning", "mean"),
        inning_span_mean=("inning_span", "mean"),
        starter_rate=("starter", "mean"),
        late_relief_rate=("late_relief", "mean"),
        multi_inning_rate=("multi_inning", "mean"),
        pitches_mean=("pitches", "mean"),
        pitches_std=("pitches", "std"),
        pitches_max=("pitches", "max"),
        work50_rate=("work_50", "mean"),
        work80_rate=("work_80", "mean"),
        rest_mean=("rest_days", "mean"),
        rest_std=("rest_days", "std"),
        rest_median=("rest_days", "median"),
        short_rest_rate=("short_rest", "mean"),
        long_rest_rate=("long_rest", "mean"),
        appearance_month_mean=("game_month", "mean"),
        appearance_month_std=("game_month", "std"),
    )
    for quantile in (0.1, 0.25, 0.75, 0.9):
        suffix = int(quantile * 100)
        summary[f"pitches_q{suffix}"] = grouped["pitches"].quantile(quantile)
        summary[f"rest_q{suffix}"] = grouped["rest_days"].quantile(quantile)
    summary["last_team"] = grouped["pitcher_team"].agg(
        lambda value: value.iloc[-1]
    )
    summary["log_games"] = np.log1p(summary["games"])
    return summary


def attach(
    rows: pd.DataFrame,
    mapped: pd.Series,
    summary: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    values = summary.reindex(mapped).reset_index(drop=True)
    last_team = values.pop("last_team")
    values.columns = [f"{prefix}_{column}" for column in values]
    current_team = rows["pitcher_team_id"].map(TEAM_ID_TO_TRACKMAN)
    values[f"{prefix}_same_team"] = (
        last_team.notna() & last_team.eq(current_team.reset_index(drop=True))
    ).astype(np.int8)
    values[f"{prefix}_team_known"] = last_team.notna().astype(np.int8)
    values[f"{prefix}_current_inning_minus_first"] = (
        rows["inning"].reset_index(drop=True)
        - values[f"{prefix}_first_inning_mean"]
    )
    values[f"{prefix}_current_month_minus_appearance"] = (
        rows["game_month"].reset_index(drop=True)
        - values[f"{prefix}_appearance_month_mean"]
    )
    values[f"{prefix}_starter_context"] = (
        rows["inning"].reset_index(drop=True).le(3).astype(float)
        * values[f"{prefix}_starter_rate"]
    )
    values[f"{prefix}_relief_context"] = (
        rows["inning"].reset_index(drop=True).ge(5).astype(float)
        * (1.0 - values[f"{prefix}_starter_rate"])
    )
    return values


def build_features(
    main: pd.DataFrame,
    games: pd.DataFrame,
    aligned: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cutoff = season - 1
    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    mapping, audit = mapping_from_aligned(aligned, cutoff)
    mapped = rows["pitcher_id"].map(mapping.mapping)
    all_games = games.loc[games["season"].le(cutoff)]
    last_games = games.loc[games["season"].eq(cutoff)]
    prior_games = games.loc[games["season"].lt(cutoff)]
    all_values = attach(rows, mapped, summarize(all_games), "role_all")
    last_values = attach(rows, mapped, summarize(last_games), "role_last")
    prior_values = attach(rows, mapped, summarize(prior_games), "role_prior")
    features = pd.concat([all_values, last_values, prior_values], axis=1)
    for column in all_values:
        suffix = column.removeprefix("role_all_")
        if f"role_last_{suffix}" in last_values and f"role_prior_{suffix}" in prior_values:
            features[f"role_delta_{suffix}"] = (
                last_values[f"role_last_{suffix}"]
                - prior_values[f"role_prior_{suffix}"]
            )
    features["mapped"] = mapped.notna().astype(np.int8)
    features["current_inning"] = rows["inning"].to_numpy(float)
    features["current_month"] = rows["game_month"].to_numpy(float)
    features["current_regular"] = rows["game_type"].astype(str).eq("R").astype(np.int8)
    features["current_count_index"] = rows["count_index"].to_numpy(float)
    features["current_pitcher_hand"] = rows["pitcher_hand"].to_numpy(float)
    features["current_batter_hand"] = rows["batter_hand"].to_numpy(float)
    features["official_success"] = rows["asof_pitcher_success_rate"].to_numpy(float)
    features["official_log_n"] = np.log1p(rows["asof_pitcher_n"].to_numpy(float))
    return features, {
        **audit,
        "history_games": len(all_games),
        "history_pitchers": int(all_games["pitcher_trackman_id"].nunique()),
        "row_mapping_coverage": float(mapped.notna().mean()),
        "feature_count": int(features.shape[1] - 1),
    }


def recent_direct_base(season: int) -> np.ndarray:
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
    return np.clip(
        0.5 * recency + 0.5 * aggressive + 0.10 * (trackman - strict) / 0.25,
        0.0,
        1.0,
    )


def new_lgb() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=160,
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
    main = load_main()
    games = load_trackman_games()
    aligned, alignment_audit = exact_aligned_rows()
    features: dict[int, pd.DataFrame] = {}
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        features[season], audits[str(season)] = build_features(
            main, games, aligned, season
        )
        targets[season] = np.load(
            LOWRANK_ROOT / f"targets_{season}.npy"
        ).astype(float)
        base[season] = recent_direct_base(season)
        expected = main.loc[main["season"].eq(season), "control_success"].to_numpy(float)
        if not np.array_equal(targets[season], expected):
            raise ValueError(f"target/order mismatch {season}")
        print(
            f"features {season}: coverage={audits[str(season)]['row_mapping_coverage']:.3f}",
            flush=True,
        )
    names = sorted(
        set.intersection(*[set(features[s].columns) for s in EVALUATED_SEASONS])
        - {"mapped"}
    )
    residual = {season: targets[season] - base[season] for season in EVALUATED_SEASONS}
    for season in residual:
        residual[season] -= residual[season].mean()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        sources = [season for season in EVALUATED_SEASONS if season < validation_season]
        corrections = {
            "ridge_a1000": np.zeros(len(targets[validation_season])),
            "ridge_a10000": np.zeros(len(targets[validation_season])),
            "lgb": np.zeros(len(targets[validation_season])),
            "lgb_last": np.zeros(len(targets[validation_season])),
        }
        if sources:
            train_x = pd.concat([features[s][names] for s in sources], ignore_index=True)
            train_y = np.concatenate([residual[s] for s in sources])
            source = np.concatenate([np.full(len(features[s]), s) for s in sources])
            eligible = pd.concat(
                [features[s]["mapped"] for s in sources], ignore_index=True
            ).eq(1).to_numpy()
            valid = features[validation_season]["mapped"].eq(1).to_numpy()
            counts = pd.Series(source[eligible]).value_counts()
            sample_weight = np.array([1.0 / counts[value] for value in source[eligible]])
            sample_weight *= len(sample_weight) / sample_weight.sum()
            for alpha in (1000.0, 10000.0):
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
                corrections[f"ridge_a{int(alpha)}"][valid] = np.clip(
                    model.predict(features[validation_season].loc[valid, names]),
                    -CORRECTION_CLIP,
                    CORRECTION_CLIP,
                )
            model_lgb = new_lgb()
            model_lgb.fit(
                train_x.loc[eligible], train_y[eligible], sample_weight=sample_weight
            )
            corrections["lgb"][valid] = np.clip(
                model_lgb.predict(features[validation_season].loc[valid, names]),
                -CORRECTION_CLIP,
                CORRECTION_CLIP,
            )
            last_source = sources[-1]
            last_x = features[last_source][names]
            last_y = residual[last_source]
            last_eligible = features[last_source]["mapped"].eq(1).to_numpy()
            model_last = new_lgb()
            model_last.fit(
                last_x.loc[last_eligible],
                last_y[last_eligible],
            )
            corrections["lgb_last"][valid] = np.clip(
                model_last.predict(
                    features[validation_season].loc[valid, names]
                ),
                -CORRECTION_CLIP,
                CORRECTION_CLIP,
            )
        predictions = {
            "base": base[validation_season],
            "role_ridge_a1000_w025": np.clip(
                base[validation_season] + 0.25 * corrections["ridge_a1000"], 0, 1
            ),
            "role_ridge_a10000_w025": np.clip(
                base[validation_season] + 0.25 * corrections["ridge_a10000"], 0, 1
            ),
            "role_lgb_w025": np.clip(
                base[validation_season] + 0.25 * corrections["lgb"], 0, 1
            ),
            "role_lgb_last_w025": np.clip(
                base[validation_season]
                + 0.25 * corrections["lgb_last"],
                0,
                1,
            ),
        }
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_seasons": sources,
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
        "experiment": "EXP-061",
        "candidate_family": "trackman_pitcher_role_rest_roster_transition",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "history_and_mapping_cutoff": "validation season-1",
            "current_fold_labels_used_for_fit_or_selection": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": False,
            "last_source_variant_posthoc_bounded": True,
        },
        "model": {
            "role_signals": [
                "starter/reliever",
                "game workload",
                "appearance rest",
                "season availability",
                "team transition",
                "current inning/month compatibility",
            ],
            "feature_count": len(names),
            "correction_clip": CORRECTION_CLIP,
            "lightgbm": new_lgb().get_params(),
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
