"""EXP-035: temporal Trackman batter and pitcher-batter matchup profiles.

The official and Trackman batter identifiers are anonymously separated.  A
season-cutoff Hungarian mapping uses only prior team, hand, season volume,
count-state exposure, and opposing-pitcher-hand exposure.  High-confidence
cost <= 0.02 mappings are used to attach prior Trackman batter and matchup
profiles.  No current pitch measurement or validation/test-row aggregate is
used.
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
from scipy.optimize import linear_sum_assignment

from trackman_features import (
    CORE_TEAMS,
    TEAM_ID_TO_TRACKMAN,
    TRACKMAN_TEAM_ALIASES,
    build_pitcher_mapping,
)
from train_exp017_rolling_residual import calculate_metrics
from train_exp033_trackman_sequence_trend import (
    FINE_TYPES,
    PROFILE_COLUMNS,
    add_trackman_derivatives,
)


DATA_DIR = Path("./data")
ARTIFACT_DIR = Path("./artifacts/EXP-035/trackman_batter_matchup")
BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
MAX_MAPPING_COST = 0.02
CORRECTION_CLIP = 0.03
CANDIDATES = (
    "batter_matchup_w025",
    "batter_matchup_reliable_w025",
    "batter_matchup_reliable_w050",
)
COUNT_INDICES = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)


def row_rates(
    rows: pd.DataFrame,
    keys: list[str],
    category: str,
    prefix: str,
) -> pd.DataFrame:
    rates = pd.crosstab(
        [rows[column] for column in keys], rows[category]
    ).reindex(columns=FINE_TYPES, fill_value=0)
    rates = rates.div(rates.sum(axis=1).replace(0, np.nan), axis=0)
    rates.columns = [f"{prefix}_{value}_rate" for value in rates.columns]
    return rates


def build_batter_mapping(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    cutoff_season: int,
) -> tuple[dict[int, int], dict[int, float], dict[str, int]]:
    seasons = list(range(2019, cutoff_season + 1))
    official = main.loc[
        main["season"].le(cutoff_season)
        & main["batter_team_id"].isin(TEAM_ID_TO_TRACKMAN)
    ].copy()
    official["team"] = official["batter_team_id"].map(TEAM_ID_TO_TRACKMAN)
    historical = trackman.loc[trackman["season"].le(cutoff_season)].copy()
    historical["team"] = historical["batter_team"].replace(
        TRACKMAN_TEAM_ALIASES
    )
    historical = historical.loc[historical["team"].isin(CORE_TEAMS)]
    official_ids = np.sort(official["batter_id"].unique())
    trackman_ids = np.sort(historical["batter_trackman_id"].unique())

    def season_counts(
        rows: pd.DataFrame, id_column: str, ids: np.ndarray
    ) -> np.ndarray:
        return (
            rows.groupby([id_column, "season"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=ids, columns=seasons, fill_value=0)
            .to_numpy(dtype=float)
        )

    official_counts = season_counts(official, "batter_id", official_ids)
    trackman_counts = season_counts(
        historical, "batter_trackman_id", trackman_ids
    )
    season_scale = np.array(
        [
            official["season"].eq(season).sum()
            / max(1, historical["season"].eq(season).sum())
            for season in seasons
        ]
    )
    trackman_counts = trackman_counts * season_scale

    def season_modes(
        rows: pd.DataFrame,
        id_column: str,
        value_column: str,
        ids: np.ndarray,
    ) -> np.ndarray:
        return (
            rows.groupby([id_column, "season"])[value_column]
            .agg(lambda values: values.mode().iloc[0])
            .unstack()
            .reindex(index=ids, columns=seasons)
            .fillna("")
            .to_numpy()
        )

    official_teams = season_modes(
        official, "batter_id", "team", official_ids
    )
    trackman_teams = season_modes(
        historical, "batter_trackman_id", "team", trackman_ids
    )
    official_hands = (
        official.groupby("batter_id")["batter_hand"]
        .agg(lambda values: values.mode().iloc[0])
        .reindex(official_ids)
        .map({1: "Left", 2: "Right"})
        .to_numpy()
    )
    trackman_hands = (
        historical.groupby("batter_trackman_id")["batter_hand"]
        .agg(lambda values: values.mode().iloc[0])
        .reindex(trackman_ids)
        .to_numpy()
    )

    official = official.assign(
        count_index=(official["balls_before"] * 4 + official["strikes_before"])
    )
    historical = historical.assign(
        count_index=(historical["balls_before"] * 4 + historical["strikes_before"])
    )

    def exposure(
        rows: pd.DataFrame,
        id_column: str,
        ids: np.ndarray,
        category: str,
        columns: list[object],
    ) -> np.ndarray:
        values = pd.crosstab(rows[id_column], rows[category]).reindex(
            index=ids, columns=columns, fill_value=0
        )
        matrix = values.to_numpy(dtype=float)
        return matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)

    official_count = exposure(
        official, "batter_id", official_ids, "count_index", list(COUNT_INDICES)
    )
    trackman_count = exposure(
        historical,
        "batter_trackman_id",
        trackman_ids,
        "count_index",
        list(COUNT_INDICES),
    )
    official_pitcher_hand = exposure(
        official, "batter_id", official_ids, "pitcher_hand", [1, 2]
    )
    trackman_pitcher_hand = exposure(
        historical,
        "batter_trackman_id",
        trackman_ids,
        "pitcher_hand",
        ["Left", "Right"],
    )

    cost_matrix = np.empty((len(official_ids), len(trackman_ids)), dtype=float)
    for index in range(len(official_ids)):
        active = official_counts[index] > 0
        count_mask = (trackman_counts > 0) | active
        volume_error = np.sum(
            np.square(
                (np.log1p(trackman_counts) - np.log1p(official_counts[index]))
                / 2.0
            )
            * count_mask,
            axis=1,
        ) / np.maximum(count_mask.sum(axis=1), 1)
        team_mismatch = np.sum(
            (trackman_teams[:, active] != official_teams[index, active])
            & (trackman_teams[:, active] != ""),
            axis=1,
        )
        missing_team = np.sum(trackman_teams[:, active] == "", axis=1)
        hand_mismatch = (trackman_hands != official_hands[index]).astype(float)
        count_error = np.mean(
            np.square(trackman_count - official_count[index]), axis=1
        )
        pitcher_hand_error = np.mean(
            np.square(trackman_pitcher_hand - official_pitcher_hand[index]),
            axis=1,
        )
        cost_matrix[index] = (
            volume_error
            + 2.0 * team_mismatch
            + 0.5 * missing_team
            + 20.0 * hand_mismatch
            + 2.0 * count_error
            + 2.0 * pitcher_hand_error
        )

    row_indices, column_indices = linear_sum_assignment(cost_matrix)
    mapping: dict[int, int] = {}
    costs: dict[int, float] = {}
    for row_index, column_index in zip(row_indices, column_indices):
        cost = float(cost_matrix[row_index, column_index])
        if cost <= MAX_MAPPING_COST:
            official_id = int(official_ids[row_index])
            mapping[official_id] = int(trackman_ids[column_index])
            costs[official_id] = cost
    return mapping, costs, {
        "candidate_main_ids": len(official_ids),
        "candidate_trackman_ids": len(trackman_ids),
    }


def physical_summary(
    rows: pd.DataFrame, keys: list[str], prefix: str, include_std: bool
) -> pd.DataFrame:
    grouped = rows.groupby(keys, sort=False)
    output = grouped.size().rename(f"{prefix}_n").to_frame()
    output[f"{prefix}_log_n"] = np.log1p(output[f"{prefix}_n"])
    for column in PROFILE_COLUMNS:
        output[f"{prefix}_{column}_mean"] = grouped[column].mean()
        if include_std:
            output[f"{prefix}_{column}_std"] = grouped[column].std()
    output = output.join(
        row_rates(rows, keys, "tagged_fine_type", f"{prefix}_tag"),
        how="left",
    )
    return output


def build_feature_rows(
    main: pd.DataFrame, trackman: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    batter_mapping, batter_costs, batter_counts = build_batter_mapping(
        main, trackman, season - 1
    )
    pitcher_mapping = build_pitcher_mapping(
        main,
        trackman,
        cutoff_season=season - 1,
        max_cost=MAX_MAPPING_COST,
    )
    history = trackman.loc[trackman["season"].lt(season)].copy()
    history["count_index"] = (
        history["balls_before"] * 4 + history["strikes_before"]
    ).astype(np.int8)
    batter_all = physical_summary(
        history, ["batter_trackman_id"], "tm_batter_hist", include_std=True
    )
    batter_hand = physical_summary(
        history,
        ["batter_trackman_id", "pitcher_hand"],
        "tm_batter_phand",
        include_std=False,
    )
    batter_count = physical_summary(
        history,
        ["batter_trackman_id", "count_index"],
        "tm_batter_count",
        include_std=False,
    )
    matchup = physical_summary(
        history,
        ["pitcher_trackman_id", "batter_trackman_id"],
        "tm_matchup",
        include_std=True,
    )

    rows = main.loc[main["season"].eq(season)].reset_index(drop=True)
    batter_id = rows["batter_id"].map(batter_mapping)
    pitcher_id = rows["pitcher_id"].map(pitcher_mapping.mapping)
    pitcher_hand = rows["pitcher_hand"].map({1: "Left", 2: "Right"})
    count_index = (rows["balls_before"] * 4 + rows["strikes_before"]).astype(
        np.int8
    )

    pieces: list[pd.DataFrame] = []
    pieces.append(batter_all.reindex(batter_id.to_numpy()).reset_index(drop=True))
    pieces.append(
        batter_hand.reindex(
            pd.MultiIndex.from_arrays(
                [batter_id.to_numpy(), pitcher_hand.to_numpy()],
                names=["batter_trackman_id", "pitcher_hand"],
            )
        ).reset_index(drop=True)
    )
    pieces.append(
        batter_count.reindex(
            pd.MultiIndex.from_arrays(
                [batter_id.to_numpy(), count_index.to_numpy()],
                names=["batter_trackman_id", "count_index"],
            )
        ).reset_index(drop=True)
    )
    pieces.append(
        matchup.reindex(
            pd.MultiIndex.from_arrays(
                [pitcher_id.to_numpy(), batter_id.to_numpy()],
                names=["pitcher_trackman_id", "batter_trackman_id"],
            )
        ).reset_index(drop=True)
    )
    features = pd.concat(pieces, axis=1)
    features["batter_mapping_cost"] = rows["batter_id"].map(batter_costs)
    features["pitcher_mapping_cost"] = rows["pitcher_id"].map(
        pitcher_mapping.costs
    )
    features["batter_mapped"] = features["batter_mapping_cost"].notna().astype(
        np.int8
    )
    features["pitcher_mapped"] = features["pitcher_mapping_cost"].notna().astype(
        np.int8
    )
    features["matchup_available"] = features["tm_matchup_n"].notna().astype(
        np.int8
    )
    features["current_count_index"] = count_index.astype(float)
    features["current_inning"] = rows["inning"].astype(float)
    features["current_game_month"] = rows["game_month"].astype(float)
    features["current_pitcher_hand"] = rows["pitcher_hand"].astype(float)
    features["current_batter_hand"] = rows["batter_hand"].astype(float)
    audit = {
        "mapping_cutoff_season": season - 1,
        "trackman_feature_seasons": sorted(
            history["season"].astype(int).unique().tolist()
        ),
        "batter_mapped_ids": len(batter_mapping),
        "pitcher_mapped_ids": len(pitcher_mapping.mapping),
        **batter_counts,
        "batter_row_coverage": float(features["batter_mapped"].mean()),
        "pitcher_row_coverage": float(features["pitcher_mapped"].mean()),
        "both_row_coverage": float(
            (features["batter_mapped"].eq(1) & features["pitcher_mapped"].eq(1)).mean()
        ),
        "matchup_row_coverage": float(features["matchup_available"].mean()),
        "feature_count": int(features.shape[1]),
    }
    return features, audit


def load_main() -> pd.DataFrame:
    columns = [
        "season",
        "game_month",
        "inning",
        "balls_before",
        "strikes_before",
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "batter_team_id",
        "pitcher_hand",
        "batter_hand",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
        "control_success",
    ]
    return pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig", usecols=columns)


def main() -> None:
    started = time.time()
    main = load_main()
    trackman = add_trackman_derivatives(
        pd.read_csv(DATA_DIR / "trackman_history.csv", encoding="utf-8-sig")
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
            f"features {season}: batter={mapping_audit[str(season)]['batter_row_coverage']:.3f} "
            f"both={mapping_audit[str(season)]['both_row_coverage']:.3f} "
            f"matchup={mapping_audit[str(season)]['matchup_row_coverage']:.3f}",
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
        correction = np.zeros(len(targets[validation_season]), dtype=float)
        fit_seconds = 0.0
        if sources:
            x = pd.concat([feature_rows[s] for s in sources], ignore_index=True)
            y = np.concatenate([residual[s] for s in sources])
            labels = np.concatenate(
                [np.full(len(residual[s]), s, dtype=np.int16) for s in sources]
            )
            eligible = x["batter_mapped"].eq(1).to_numpy()
            source_counts = pd.Series(labels[eligible]).value_counts()
            weights = np.array(
                [1.0 / source_counts[int(value)] for value in labels[eligible]]
            )
            weights *= len(weights) / weights.sum()
            model = LGBMRegressor(
                objective="regression_l2",
                n_estimators=200,
                learning_rate=0.015,
                num_leaves=7,
                min_child_samples=2000,
                max_bin=127,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=12.0,
                random_state=42,
                n_jobs=-1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            fit_started = time.time()
            model.fit(
                x.loc[eligible, feature_names].replace([np.inf, -np.inf], np.nan),
                y[eligible],
                sample_weight=weights,
            )
            fit_seconds = time.time() - fit_started
            validation_eligible = feature_rows[validation_season][
                "batter_mapped"
            ].eq(1).to_numpy()
            correction[validation_eligible] = model.predict(
                feature_rows[validation_season]
                .loc[validation_eligible, feature_names]
                .replace([np.inf, -np.inf], np.nan)
            )
            correction = np.clip(correction, -CORRECTION_CLIP, CORRECTION_CLIP)

        batter_cost = feature_rows[validation_season]["batter_mapping_cost"].to_numpy(
            dtype=float
        )
        reliability = np.clip(
            1.0
            - np.nan_to_num(batter_cost, nan=MAX_MAPPING_COST) / MAX_MAPPING_COST,
            0.0,
            1.0,
        )
        predictions = {
            "base": base[validation_season],
            "batter_matchup_w025": np.clip(
                base[validation_season] + 0.25 * correction, 0.0, 1.0
            ),
            "batter_matchup_reliable_w025": np.clip(
                base[validation_season] + 0.25 * reliability * correction,
                0.0,
                1.0,
            ),
            "batter_matchup_reliable_w050": np.clip(
                base[validation_season] + 0.50 * reliability * correction,
                0.0,
                1.0,
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
        np.save(ARTIFACT_DIR / f"correction_{validation_season}.npy", correction)
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
        "experiment": "EXP-035",
        "candidate_family": "trackman_batter_and_matchup_profiles",
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
            "type": "LightGBM regression residual",
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "max_mapping_cost": MAX_MAPPING_COST,
            "iterations": 200,
            "num_leaves": 7,
            "min_child_samples": 2000,
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
