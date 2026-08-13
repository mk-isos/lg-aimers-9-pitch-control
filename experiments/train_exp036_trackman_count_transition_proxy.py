"""EXP-036: Trackman count-transition control proxies.

For two consecutive pitches in the same plate appearance, an increment in
``balls_before`` or ``strikes_before`` reveals a subset of the previous pitch
result without using location, current-pitch measurements, or future-season
data.  Prior-season Trackman logs are converted into pitcher, count/hand, and
fine-pitch-type strike proxies.  A small Ridge and LightGBM learn residuals on
earlier OOF seasons only atop the fixed EXP-021 strict base.
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
from sklearn.linear_model import Ridge

from trackman_features import build_pitcher_mapping
from train_exp017_rolling_residual import calculate_metrics
from train_exp033_trackman_sequence_trend import (
    FINE_TYPES,
    add_trackman_derivatives,
)


DATA_DIR = Path("./data")
ARTIFACT_DIR = Path("./artifacts/EXP-036/trackman_count_transition_proxy")
BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
MAX_MAPPING_COST = 0.10
OVERALL_SMOOTHING = 300.0
CONTEXT_SMOOTHING = 100.0
TYPE_SMOOTHING = 100.0
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "proxy_ridge_w025",
    "proxy_lgb_w025",
    "proxy_blend_w025",
    "proxy_blend_w050",
)


def reconstruct_count_transitions(trackman: pd.DataFrame) -> pd.DataFrame:
    rows = trackman.sort_values(
        ["trackman_game_id", "pitch_no"], kind="stable"
    ).reset_index(drop=True)
    same_game = rows["trackman_game_id"].eq(rows["trackman_game_id"].shift(-1))
    same_pa = (
        same_game
        & rows["inning"].eq(rows["inning"].shift(-1))
        & rows["top_bottom"].eq(rows["top_bottom"].shift(-1))
        & rows["pitcher_trackman_id"].eq(
            rows["pitcher_trackman_id"].shift(-1)
        )
        & rows["batter_trackman_id"].eq(
            rows["batter_trackman_id"].shift(-1)
        )
        & rows["pitch_of_pa"].add(1).eq(rows["pitch_of_pa"].shift(-1))
    )
    ball_delta = rows["balls_before"].shift(-1) - rows["balls_before"]
    strike_delta = rows["strikes_before"].shift(-1) - rows["strikes_before"]
    proxy_ball = same_pa & ball_delta.eq(1) & strike_delta.eq(0)
    proxy_strike = same_pa & ball_delta.eq(0) & strike_delta.eq(1)
    proxy_same = same_pa & ball_delta.eq(0) & strike_delta.eq(0)
    rows["proxy_known"] = (proxy_ball | proxy_strike).astype(np.int8)
    rows["proxy_strike"] = proxy_strike.astype(np.int8)
    rows["proxy_ball"] = proxy_ball.astype(np.int8)
    rows["proxy_same_count"] = proxy_same.astype(np.int8)
    rows["count_index"] = (
        rows["balls_before"] * 4 + rows["strikes_before"]
    ).astype(np.int8)
    return rows


def build_proxy_tables(
    history: pd.DataFrame, prefix: str
) -> dict[str, pd.DataFrame | float]:
    known = history.loc[history["proxy_known"].eq(1)].copy()
    league_rate = float(known["proxy_strike"].mean()) if len(known) else 0.5

    overall_group = known.groupby("pitcher_trackman_id", sort=False)
    overall = overall_group.size().rename(f"{prefix}_overall_n").to_frame()
    overall[f"{prefix}_overall_strikes"] = overall_group[
        "proxy_strike"
    ].sum()
    overall[f"{prefix}_overall_rate"] = (
        overall[f"{prefix}_overall_strikes"]
        + OVERALL_SMOOTHING * league_rate
    ) / (overall[f"{prefix}_overall_n"] + OVERALL_SMOOTHING)
    all_count = history.groupby("pitcher_trackman_id").size()
    same_count = history.groupby("pitcher_trackman_id")[
        "proxy_same_count"
    ].sum()
    overall[f"{prefix}_known_fraction"] = (
        overall[f"{prefix}_overall_n"] / all_count.reindex(overall.index)
    )
    overall[f"{prefix}_same_count_fraction"] = (
        same_count.reindex(overall.index) / all_count.reindex(overall.index)
    )
    overall[f"{prefix}_overall_log_n"] = np.log1p(
        overall[f"{prefix}_overall_n"]
    )

    context_keys = ["pitcher_trackman_id", "count_index", "batter_hand"]
    context_group = known.groupby(context_keys, sort=False)
    context = context_group.size().rename(f"{prefix}_context_n").to_frame()
    context[f"{prefix}_context_strikes"] = context_group[
        "proxy_strike"
    ].sum()
    context_pitcher = context.index.get_level_values("pitcher_trackman_id")
    prior = overall[f"{prefix}_overall_rate"].reindex(context_pitcher).to_numpy()
    context[f"{prefix}_context_rate"] = (
        context[f"{prefix}_context_strikes"].to_numpy(dtype=float)
        + CONTEXT_SMOOTHING * prior
    ) / (context[f"{prefix}_context_n"].to_numpy(dtype=float) + CONTEXT_SMOOTHING)
    context[f"{prefix}_context_log_n"] = np.log1p(
        context[f"{prefix}_context_n"]
    )

    type_keys = ["pitcher_trackman_id", "tagged_fine_type"]
    type_group = known.groupby(type_keys, sort=False)
    type_rate = type_group.size().rename("n").to_frame()
    type_rate["strikes"] = type_group["proxy_strike"].sum()
    type_pitcher = type_rate.index.get_level_values("pitcher_trackman_id")
    type_prior = overall[f"{prefix}_overall_rate"].reindex(type_pitcher).to_numpy()
    type_rate["rate"] = (
        type_rate["strikes"].to_numpy(dtype=float)
        + TYPE_SMOOTHING * type_prior
    ) / (type_rate["n"].to_numpy(dtype=float) + TYPE_SMOOTHING)
    type_pivot = type_rate["rate"].unstack("tagged_fine_type").reindex(
        columns=FINE_TYPES
    )

    propensity = pd.crosstab(
        [
            history["pitcher_trackman_id"],
            history["count_index"],
            history["batter_hand"],
        ],
        history["tagged_fine_type"],
    ).reindex(columns=FINE_TYPES, fill_value=0)
    propensity = propensity.div(
        propensity.sum(axis=1).replace(0, np.nan), axis=0
    )
    prop_pitcher = propensity.index.get_level_values("pitcher_trackman_id")
    aligned_type = type_pivot.reindex(prop_pitcher).to_numpy(dtype=float)
    overall_fallback = overall[f"{prefix}_overall_rate"].reindex(
        prop_pitcher
    ).to_numpy(dtype=float)
    aligned_type = np.where(
        np.isfinite(aligned_type), aligned_type, overall_fallback[:, None]
    )
    propensity_values = propensity.to_numpy(dtype=float)
    mixed = np.nansum(propensity_values * aligned_type, axis=1)
    propensity[f"{prefix}_mix_rate"] = mixed
    entropy = -np.nansum(
        propensity_values * np.log(np.clip(propensity_values, 1e-12, 1.0)),
        axis=1,
    )
    propensity[f"{prefix}_type_entropy"] = entropy
    mixed_table = propensity[
        [f"{prefix}_mix_rate", f"{prefix}_type_entropy"]
    ]
    return {
        "overall": overall,
        "context": context,
        "mixed": mixed_table,
        "league_rate": league_rate,
    }


def query_tables(
    tables: dict[str, pd.DataFrame | float],
    mapped_pitcher: pd.Series,
    count_index: pd.Series,
    batter_hand: pd.Series,
) -> pd.DataFrame:
    overall = tables["overall"]
    context = tables["context"]
    mixed = tables["mixed"]
    assert isinstance(overall, pd.DataFrame)
    assert isinstance(context, pd.DataFrame)
    assert isinstance(mixed, pd.DataFrame)
    context_index = pd.MultiIndex.from_arrays(
        [
            mapped_pitcher.to_numpy(),
            count_index.to_numpy(dtype=np.int8),
            batter_hand.to_numpy(),
        ],
        names=["pitcher_trackman_id", "count_index", "batter_hand"],
    )
    return pd.concat(
        [
            overall.reindex(mapped_pitcher.to_numpy()).reset_index(drop=True),
            context.reindex(context_index).reset_index(drop=True),
            mixed.reindex(context_index).reset_index(drop=True),
        ],
        axis=1,
    )


def build_feature_rows(
    main: pd.DataFrame, trackman: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    mapping = build_pitcher_mapping(
        main,
        trackman,
        cutoff_season=season - 1,
        max_cost=MAX_MAPPING_COST,
    )
    history = trackman.loc[trackman["season"].lt(season)]
    last = trackman.loc[trackman["season"].eq(season - 1)]
    history_tables = build_proxy_tables(history, "tm_proxy_hist")
    last_tables = build_proxy_tables(last, "tm_proxy_last")

    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    mapped_pitcher = rows["pitcher_id"].map(mapping.mapping)
    count_index = (rows["balls_before"] * 4 + rows["strikes_before"]).astype(
        np.int8
    )
    batter_hand = rows["batter_hand"].map({1: "Left", 2: "Right"})
    features = pd.concat(
        [
            query_tables(
                history_tables, mapped_pitcher, count_index, batter_hand
            ),
            query_tables(last_tables, mapped_pitcher, count_index, batter_hand),
        ],
        axis=1,
    )
    features["trackman_mapping_cost"] = rows["pitcher_id"].map(mapping.costs)
    features["trackman_mapped"] = features["trackman_mapping_cost"].notna().astype(
        np.int8
    )
    features["current_count_index"] = count_index.astype(float)
    features["current_pitcher_hand"] = rows["pitcher_hand"].astype(float)
    features["current_batter_hand"] = rows["batter_hand"].astype(float)
    features["official_ball_rate"] = rows["asof_pitcher_ball_rate"].astype(float)
    features["official_strike_rate"] = rows[
        "asof_pitcher_strike_rate"
    ].astype(float)
    features["official_called_share"] = (
        features["official_strike_rate"]
        / (
            features["official_strike_rate"]
            + features["official_ball_rate"]
        ).replace(0.0, np.nan)
    )
    for level in ("overall_rate", "context_rate", "mix_rate"):
        hist_name = f"tm_proxy_hist_{level}"
        last_name = f"tm_proxy_last_{level}"
        if hist_name in features and last_name in features:
            features[f"tm_proxy_delta_{level}"] = (
                features[last_name] - features[hist_name]
            )
            features[f"tm_proxy_official_gap_{level}"] = (
                features[last_name] - features["official_called_share"]
            )
    known_total = int(trackman["proxy_known"].sum())
    same_total = int(trackman["proxy_same_count"].sum())
    audit = {
        "mapping_cutoff_season": season - 1,
        "trackman_feature_seasons": sorted(
            history["season"].astype(int).unique().tolist()
        ),
        "mapped_pitchers": len(mapping.mapping),
        "row_mapping_coverage": float(features["trackman_mapped"].mean()),
        "history_known_transition_rows": int(history["proxy_known"].sum()),
        "last_known_transition_rows": int(last["proxy_known"].sum()),
        "all_trackman_known_transition_rows": known_total,
        "all_trackman_same_count_rows": same_total,
        "all_trackman_known_transition_rate": float(
            known_total / len(trackman)
        ),
        "feature_count": int(features.shape[1]),
        "history_proxy_league_rate": float(history_tables["league_rate"]),
        "last_proxy_league_rate": float(last_tables["league_rate"]),
    }
    return features, audit


def load_main() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_id",
        "pitcher_team_id",
        "pitcher_hand",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "control_success",
    ]
    return pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig", usecols=columns)


def fit_ridge(
    x: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[Ridge, np.ndarray, np.ndarray, np.ndarray]:
    values = x.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    median = np.nanmedian(values, axis=0)
    values = np.where(np.isfinite(values), values, median)
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average(np.square(values - mean), axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-8))
    standardized = (values - mean) / scale
    model = Ridge(alpha=5000.0, fit_intercept=False)
    model.fit(standardized, y, sample_weight=weights)
    return model, median, mean, scale


def ridge_predict(
    model: Ridge,
    frame: pd.DataFrame,
    median: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = frame.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, median)
    return model.predict((values - mean) / scale)


def main() -> None:
    started = time.time()
    main = load_main()
    raw_trackman = pd.read_csv(
        DATA_DIR / "trackman_history.csv", encoding="utf-8-sig"
    )
    trackman = reconstruct_count_transitions(
        add_trackman_derivatives(raw_trackman)
    )
    targets: dict[int, np.ndarray] = {}
    base: dict[int, np.ndarray] = {}
    feature_rows: dict[int, pd.DataFrame] = {}
    mapping_audit: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        mask = main["season"].eq(season)
        targets[season] = np.load(BASE_ROOT / f"targets_{season}.npy").astype(float)
        base[season] = np.load(
            BASE_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
        ).astype(float)
        if not np.array_equal(
            main.loc[mask, "control_success"].to_numpy(dtype=float), targets[season]
        ):
            raise ValueError(f"target/order mismatch {season}")
        feature_rows[season], mapping_audit[str(season)] = build_feature_rows(
            main, trackman, season
        )
        print(
            f"features {season}: coverage={mapping_audit[str(season)]['row_mapping_coverage']:.3f} "
            f"known={mapping_audit[str(season)]['history_known_transition_rows']}",
            flush=True,
        )
    feature_names = list(feature_rows[2021].columns)
    if any(list(feature_rows[s].columns) != feature_names for s in EVALUATED_SEASONS):
        raise ValueError("feature schema drift")
    residual = {s: targets[s] - base[s] for s in EVALUATED_SEASONS}
    for season in EVALUATED_SEASONS:
        residual[season] -= residual[season].mean()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        sources = [s for s in EVALUATED_SEASONS if s < validation_season]
        ridge_correction = np.zeros(len(targets[validation_season]), dtype=float)
        lgb_correction = np.zeros(len(targets[validation_season]), dtype=float)
        fit_seconds = 0.0
        if sources:
            x = pd.concat([feature_rows[s] for s in sources], ignore_index=True)
            y = np.concatenate([residual[s] for s in sources])
            labels = np.concatenate(
                [np.full(len(residual[s]), s, dtype=np.int16) for s in sources]
            )
            eligible = x["trackman_mapped"].eq(1).to_numpy()
            source_counts = pd.Series(labels[eligible]).value_counts()
            weights = np.array(
                [1.0 / source_counts[int(value)] for value in labels[eligible]]
            )
            weights *= len(weights) / weights.sum()
            fit_started = time.time()
            ridge, median, mean, scale = fit_ridge(
                x.loc[eligible, feature_names], y[eligible], weights
            )
            lgb_model = LGBMRegressor(
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
                reg_lambda=12.0,
                random_state=42,
                n_jobs=-1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            lgb_model.fit(
                x.loc[eligible, feature_names].replace([np.inf, -np.inf], np.nan),
                y[eligible],
                sample_weight=weights,
            )
            fit_seconds = time.time() - fit_started
            validation_eligible = feature_rows[validation_season][
                "trackman_mapped"
            ].eq(1).to_numpy()
            ridge_correction[validation_eligible] = ridge_predict(
                ridge,
                feature_rows[validation_season].loc[
                    validation_eligible, feature_names
                ],
                median,
                mean,
                scale,
            )
            lgb_correction[validation_eligible] = lgb_model.predict(
                feature_rows[validation_season]
                .loc[validation_eligible, feature_names]
                .replace([np.inf, -np.inf], np.nan)
            )
            ridge_correction = np.clip(
                ridge_correction, -CORRECTION_CLIP, CORRECTION_CLIP
            )
            lgb_correction = np.clip(
                lgb_correction, -CORRECTION_CLIP, CORRECTION_CLIP
            )
        blend_correction = 0.5 * ridge_correction + 0.5 * lgb_correction
        predictions = {
            "base": base[validation_season],
            "proxy_ridge_w025": np.clip(
                base[validation_season] + 0.25 * ridge_correction, 0.0, 1.0
            ),
            "proxy_lgb_w025": np.clip(
                base[validation_season] + 0.25 * lgb_correction, 0.0, 1.0
            ),
            "proxy_blend_w025": np.clip(
                base[validation_season] + 0.25 * blend_correction, 0.0, 1.0
            ),
            "proxy_blend_w050": np.clip(
                base[validation_season] + 0.50 * blend_correction, 0.0, 1.0
            ),
        }
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_oof_seasons": sources,
            "fit_seconds": fit_seconds,
        }
        for name, values in predictions.items():
            if not np.isfinite(values).all() or not (
                (values >= 0.0).all() and (values <= 1.0).all()
            ):
                raise ValueError(f"invalid predictions {name} {validation_season}")
            fold[name] = calculate_metrics(targets[validation_season], values)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{validation_season}.npy", values)
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets[validation_season].astype(np.int8))
        np.save(ARTIFACT_DIR / f"ridge_correction_{validation_season}.npy", ridge_correction)
        np.save(ARTIFACT_DIR / f"lgb_correction_{validation_season}.npy", lgb_correction)
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
    for candidate in ("base", *CANDIDATES):
        skills = {
            str(s): float(folds[str(s)][candidate]["skill_score_unclipped"])
            for s in REPORT_SEASONS
        }
        briers = {
            str(s): float(folds[str(s)][candidate]["brier_score"])
            for s in REPORT_SEASONS
        }
        aggregate[candidate] = {
            "season_briers": briers,
            "season_skills": skills,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills["2024"],
        }
    best = max(
        CANDIDATES,
        key=lambda name: (
            aggregate[name]["min_skill"],
            aggregate[name]["latest_2024_skill"],
            aggregate[name]["mean_skill"],
        ),
    )
    result = {
        "experiment": "EXP-036",
        "candidate_family": "trackman_count_transition_proxy",
        "transition_reconstruction": {
            "known_rows": int(trackman["proxy_known"].sum()),
            "known_rate": float(trackman["proxy_known"].mean()),
            "proxy_strike_rows": int(trackman["proxy_strike"].sum()),
            "proxy_ball_rows": int(trackman["proxy_ball"].sum()),
            "same_count_rows": int(trackman["proxy_same_count"].sum()),
            "terminal_or_unresolved_rows": int(
                len(trackman)
                - trackman["proxy_known"].sum()
                - trackman["proxy_same_count"].sum()
            ),
        },
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "base": "fixed EXP-021 lowrank_s300_r6 OOF",
            "source_training": "earlier OOF seasons only; season-centered and season-equal",
            "trackman_and_mapping_cutoff": "validation season-1",
            "current_fold_labels_used_for_training_or_selection": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "model": {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "ridge_alpha": 5000.0,
            "lightgbm_iterations": 160,
            "lightgbm_num_leaves": 7,
            "lightgbm_min_child_samples": 3000,
            "max_mapping_cost": MAX_MAPPING_COST,
            "overall_smoothing": OVERALL_SMOOTHING,
            "context_smoothing": CONTEXT_SMOOTHING,
            "type_smoothing": TYPE_SMOOTHING,
            "correction_clip": CORRECTION_CLIP,
        },
        "mapping_audit": mapping_audit,
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
