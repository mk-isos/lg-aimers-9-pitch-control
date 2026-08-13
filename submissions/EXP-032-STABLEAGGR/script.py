"""EXP-021 final candidate inference (copied to the ZIP root as script.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
MODEL_DIR = Path("./model")
TEST_PATH = Path("./data/test.csv")
SAMPLE_PATH = Path("./data/sample_submission.csv")
OUTPUT_PATH = Path("./output/submission.csv")
SHRINKAGE_STRENGTHS = (10.0, 30.0, 100.0, 300.0)


def add_static_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["count_index"] = (
        out["balls_before"] * 4 + out["strikes_before"]
    ).astype("int8")
    out["count_out_index"] = (
        out["count_index"] * 3 + out["outs_before"]
    ).astype("int8")
    out["is_full_count"] = (
        (out["balls_before"] == 3) & (out["strikes_before"] == 2)
    ).astype("int8")
    out["has_two_strikes"] = (out["strikes_before"] == 2).astype("int8")
    out["has_three_balls"] = (out["balls_before"] == 3).astype("int8")
    out["count_advantage"] = (
        out["strikes_before"] - out["balls_before"]
    ).astype("int8")
    out["runner_in_scoring_position"] = (
        (out["runner_on_2b"] == 1) | (out["runner_on_3b"] == 1)
    ).astype("int8")
    out["bases_loaded"] = (
        (out["runner_on_1b"] == 1)
        & (out["runner_on_2b"] == 1)
        & (out["runner_on_3b"] == 1)
    ).astype("int8")
    out["same_hand"] = (
        out["pitcher_hand"] == out["batter_hand"]
    ).astype("int8")
    out["late_inning"] = (out["inning"] >= 7).astype("int8")
    out["close_game"] = (
        out["score_diff_pitcher_team"].abs() <= 1
    ).astype("int8")
    out["log_li"] = np.log1p(out["li"].clip(lower=0)).astype("float32")
    out["score_pressure"] = (
        out["score_diff_pitcher_team"].abs() * out["log_li"]
    ).astype("float32")
    out["win_expectancy_gap"] = (
        out["home_win_expectancy"] - out["away_win_expectancy"]
    ).astype("float32")
    out["pitcher_batter_success_gap"] = (
        out["asof_pitcher_success_rate"]
        - out["asof_batter_success_rate"]
    ).astype("float32")
    out["pitcher_recent_success_delta_1_5"] = (
        out["asof_pitcher_prev1_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    ).astype("float32")
    out["pitcher_recent_success_delta_3_5"] = (
        out["asof_pitcher_prev3_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    ).astype("float32")
    out["pitcher_recent_middle_delta_1_5"] = (
        out["asof_pitcher_prev1_game_middle_rate"]
        - out["asof_pitcher_prev5_game_middle_rate"]
    ).astype("float32")
    out["log_pitcher_n"] = np.log1p(
        out["asof_pitcher_n"].clip(lower=0)
    ).astype("float32")
    out["log_batter_n"] = np.log1p(
        out["asof_batter_n"].clip(lower=0)
    ).astype("float32")
    out["log_pitchmix_n"] = np.log1p(
        out["asof_pitcher_pitchmix_n"].clip(lower=0)
    ).astype("float32")
    return out


def _records_index(records: list[dict[str, object]], id_column: str) -> pd.DataFrame:
    return pd.DataFrame.from_records(records).set_index(id_column)


def attach_entity_features(
    rows: pd.DataFrame,
    entity: str,
    state: pd.DataFrame,
    league_prior: float,
) -> pd.DataFrame:
    out = rows.copy()
    id_column = f"{entity}_id"
    count_column = f"asof_{entity}_n"
    rate_column = f"asof_{entity}_success_rate"
    ids = out[id_column]
    prior_n = ids.map(state["prior_n"]).fillna(0.0).to_numpy(dtype=float)
    prior_successes = (
        ids.map(state["prior_successes"]).fillna(0.0).to_numpy(dtype=float)
    )
    prior_exists = ids.isin(state.index).to_numpy(dtype=np.int8)
    career_n = out[count_column].to_numpy(dtype=float)
    career_successes = np.rint(
        career_n * out[rate_column].fillna(0.0).to_numpy(dtype=float)
    )
    season_n_raw = career_n - prior_n
    season_successes_raw = career_successes - prior_successes
    if np.any(season_n_raw < -1e-6):
        raise ValueError(f"{entity}: career count is below stored history")
    if np.any(season_successes_raw < -0.01) or np.any(
        season_successes_raw - season_n_raw > 0.01
    ):
        raise ValueError(f"{entity}: reconstructed successes are invalid")
    season_n = np.maximum(season_n_raw, 0.0)
    season_successes = np.clip(season_successes_raw, 0.0, season_n)
    prior_rate = np.divide(
        prior_successes,
        prior_n,
        out=np.full(len(out), league_prior, dtype=float),
        where=prior_n > 0,
    )
    season_rate = np.divide(
        season_successes,
        season_n,
        out=np.full(len(out), league_prior, dtype=float),
        where=season_n > 0,
    )
    player_prior = (prior_successes + 200.0 * league_prior) / (
        prior_n + 200.0
    )
    prefix = f"temporal_{entity}"
    out[f"{prefix}_prior_exists"] = prior_exists
    out[f"{prefix}_prior_n"] = prior_n.astype("float32")
    out[f"{prefix}_log_prior_n"] = np.log1p(prior_n).astype("float32")
    out[f"{prefix}_prior_rate"] = prior_rate.astype("float32")
    out[f"{prefix}_prior_rate_shrunk_200"] = player_prior.astype("float32")
    out[f"{prefix}_season_n"] = season_n.astype("float32")
    out[f"{prefix}_log_season_n"] = np.log1p(season_n).astype("float32")
    out[f"{prefix}_season_rate"] = season_rate.astype("float32")
    out[f"{prefix}_season_minus_prior_rate"] = (
        season_rate - prior_rate
    ).astype("float32")
    for strength in SHRINKAGE_STRENGTHS:
        suffix = int(strength)
        out[f"{prefix}_season_global_{suffix}"] = (
            (season_successes + strength * league_prior)
            / (season_n + strength)
        ).astype("float32")
        out[f"{prefix}_season_player_{suffix}"] = (
            (season_successes + strength * player_prior)
            / (season_n + strength)
        ).astype("float32")
        out[f"{prefix}_reliability_{suffix}"] = (
            season_n / (season_n + strength)
        ).astype("float32")
    return out


def attach_temporal_features(
    frame: pd.DataFrame,
    history: dict[str, object],
) -> pd.DataFrame:
    through_season = int(history["through_season"])
    if (frame["season"] <= through_season).any():
        raise ValueError("inference season must follow temporal history")
    league_rate = float(history["league_rate"])
    out = frame.copy()
    out["temporal_prior_league_rate"] = np.float32(league_rate)
    for entity in ("pitcher", "batter"):
        state = _records_index(history[entity], f"{entity}_id")
        out = attach_entity_features(out, entity, state, league_rate)
    out["temporal_base_global_30"] = (
        0.7 * out["temporal_pitcher_season_global_30"]
        + 0.3 * out["temporal_batter_season_global_30"]
    ).astype("float32")
    out["temporal_base_player_30"] = (
        0.7 * out["temporal_pitcher_season_player_30"]
        + 0.3 * out["temporal_batter_season_player_30"]
    ).astype("float32")
    return out


MULTIRATE_GROUPS = (
    (
        "pitcher_control",
        "pitcher_id",
        "asof_pitcher_n",
        (
            ("success", "asof_pitcher_success_rate"),
            ("reverse", "asof_pitcher_reverse_rate"),
            ("middle", "asof_pitcher_middle_rate"),
            ("ball", "asof_pitcher_ball_rate"),
            ("strike", "asof_pitcher_strike_rate"),
        ),
    ),
    (
        "batter_control",
        "batter_id",
        "asof_batter_n",
        (
            ("success", "asof_batter_success_rate"),
            ("middle", "asof_batter_middle_rate"),
        ),
    ),
    (
        "pitcher_pitchmix",
        "pitcher_id",
        "asof_pitcher_pitchmix_n",
        (
            ("fastball", "asof_pitcher_fastball_rate"),
            ("breaking", "asof_pitcher_breaking_rate"),
            ("offspeed", "asof_pitcher_offspeed_rate"),
        ),
    ),
)


def attach_multirate_features(
    frame: pd.DataFrame,
    stored: dict[str, object],
) -> pd.DataFrame:
    if (frame["season"] <= int(stored["through_season"])).any():
        raise ValueError("inference season must follow multirate history")
    out = frame.copy()
    global_rates = stored["global_rates"]
    tables = stored["tables"]
    for group_name, id_column, n_column, rates in MULTIRATE_GROUPS:
        state = _records_index(tables[group_name], id_column)
        ids = out[id_column]
        prior_n = ids.map(state["prior_n"]).fillna(0.0).to_numpy(dtype=float)
        career_n = out[n_column].fillna(0.0).to_numpy(dtype=float)
        season_n_raw = career_n - prior_n
        if np.any(season_n_raw < -1e-6):
            raise ValueError(f"{group_name}: career count is below history")
        season_n = np.maximum(season_n_raw, 0.0)
        prefix = f"multirate_{group_name}"
        out[f"{prefix}_season_n"] = season_n.astype("float32")
        out[f"{prefix}_log_season_n"] = np.log1p(season_n).astype("float32")
        out[f"{prefix}_reliability_30"] = (
            season_n / (season_n + 30.0)
        ).astype("float32")
        for metric, rate_column in rates:
            key = f"{group_name}_{metric}"
            global_rate = float(global_rates[key])
            prior_count = (
                ids.map(state[f"prior_{metric}_count"])
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            rate = out[rate_column].fillna(global_rate).to_numpy(dtype=float)
            career_count = np.rint(career_n * rate)
            season_count = np.clip(
                career_count - prior_count, 0.0, season_n
            )
            prior_rate = np.divide(
                prior_count,
                prior_n,
                out=np.full(len(out), global_rate, dtype=float),
                where=prior_n > 0,
            )
            player_prior = (prior_count + 200.0 * global_rate) / (
                prior_n + 200.0
            )
            season_rate = np.divide(
                season_count,
                season_n,
                out=np.full(len(out), global_rate, dtype=float),
                where=season_n > 0,
            )
            metric_prefix = f"{prefix}_{metric}"
            out[f"{metric_prefix}_prior_rate"] = prior_rate.astype("float32")
            out[f"{metric_prefix}_prior_shrunk_200"] = player_prior.astype(
                "float32"
            )
            out[f"{metric_prefix}_season_rate"] = season_rate.astype("float32")
            out[f"{metric_prefix}_season_global_30"] = (
                (season_count + 30.0 * global_rate) / (season_n + 30.0)
            ).astype("float32")
            out[f"{metric_prefix}_season_player_30"] = (
                (season_count + 30.0 * player_prior) / (season_n + 30.0)
            ).astype("float32")
            out[f"{metric_prefix}_season_minus_prior"] = (
                season_rate - prior_rate
            ).astype("float32")
    return out


def _effect_lookup(
    records: list[dict[str, object]],
    columns: list[str],
) -> dict[tuple[int, ...], float]:
    return {
        tuple(int(record[column]) for column in columns): float(record["effect"])
        for record in records
    }


def map_effect_records(
    frame: pd.DataFrame,
    records: list[dict[str, object]],
    columns: list[str],
) -> np.ndarray:
    lookup = _effect_lookup(records, columns)
    keys = frame[columns].astype(int)
    return np.fromiter(
        (
            lookup.get(tuple(row), 0.0)
            for row in keys.itertuples(index=False, name=None)
        ),
        dtype=float,
        count=len(frame),
    )


def map_exp018_group(
    frame: pd.DataFrame,
    stored: dict[str, object],
) -> np.ndarray:
    base_columns = ["count_index", "pitcher_hand", "batter_hand"]
    base = map_effect_records(frame, stored["base"], base_columns)
    reverse_rate = frame["asof_pitcher_reverse_rate"].to_numpy(dtype=float)
    keyed = frame.copy()
    keyed["reverse_rate_bin"] = np.where(
        np.isfinite(reverse_rate), np.floor(reverse_rate / 0.05), -1
    ).astype(int)
    reverse = map_effect_records(
        keyed, stored["reverse"], base_columns + ["reverse_rate_bin"]
    )
    return 0.7 * base + 0.3 * reverse


def map_team_effects(
    frame: pd.DataFrame,
    stored: dict[str, object],
) -> np.ndarray:
    family_values = []
    for family, columns in (
        ("pitcher_team", ["pitcher_team_id", "pitcher_hand", "batter_hand"]),
        ("batter_team", ["batter_team_id", "pitcher_hand", "batter_hand"]),
    ):
        per_source = [
            map_effect_records(frame, source["records"], columns)
            for source in stored[family]
        ]
        family_values.append(np.mean(np.vstack(per_source), axis=0))
    return 0.5 * family_values[0] + 0.5 * family_values[1]


def map_pitcher_count_effects(
    frame: pd.DataFrame,
    stored: dict[str, object],
) -> np.ndarray:
    columns = ["pitcher_id", "count_index", "batter_hand"]
    values = [
        map_effect_records(frame, source["records"], columns)
        for source in stored["sources"]
    ]
    return np.mean(np.vstack(values), axis=0)


def map_lowrank_effects(
    frame: pd.DataFrame,
    stored: dict[str, object],
) -> np.ndarray:
    count_index = frame["count_index"].to_numpy(dtype=int)
    batter_hand = frame["batter_hand"].to_numpy(dtype=int)
    context_lookup = {
        (int(item["count_index"]), int(item["batter_hand"])): int(
            item["position"]
        )
        for item in stored["contexts"]
    }
    context = np.fromiter(
        (
            context_lookup[(int(count), int(hand))]
            for count, hand in zip(count_index, batter_hand, strict=True)
        ),
        dtype=np.int16,
        count=len(frame),
    )
    pitcher = frame["pitcher_id"].to_numpy(dtype=int)
    corrections = []
    for source in stored["sources"]:
        source_ids = np.asarray(source["pitcher_ids"], dtype=int)
        source_matrix = np.asarray(source["values"], dtype=float)
        indexer = pd.Index(source_ids).get_indexer(pitcher)
        seen = indexer >= 0
        values = np.zeros(len(frame), dtype=float)
        values[seen] = source_matrix[indexer[seen], context[seen]]
        corrections.append(values)
    correction_matrix = np.vstack(corrections)
    source_weights = stored.get("source_weights")
    if source_weights is None:
        return np.mean(correction_matrix, axis=0)
    weights = np.asarray(source_weights, dtype=float)
    if len(weights) != len(corrections) or not np.isfinite(weights).all() or (
        weights < 0.0
    ).any() or weights.sum() <= 0.0:
        raise ValueError("invalid serialized low-rank source weights")
    return np.average(correction_matrix, axis=0, weights=weights)


def encoded_matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    if len(names) != len(set(names)):
        raise ValueError("serialized feature schema contains duplicates")
    encoded = pd.get_dummies(
        frame,
        columns=["top_bottom", "base_state"],
        dummy_na=True,
        dtype=np.int8,
    )
    allowed_missing_prefixes = ("top_bottom_", "base_state_")
    missing = [
        name
        for name in names
        if name not in encoded.columns
        and not name.startswith(allowed_missing_prefixes)
    ]
    if missing:
        raise ValueError(f"required model features are missing: {missing}")
    encoded = encoded.reindex(columns=names, fill_value=0)
    return encoded.to_numpy(dtype=np.float32)


def predict_histgradientboosting(
    matrix: np.ndarray,
    state: dict[str, object],
) -> np.ndarray:
    """Run exported numerical HGB trees without a versioned sklearn pickle."""
    if state.get("format") != "numeric_hgb_v1":
        raise ValueError("unsupported HGB model format")
    if matrix.ndim != 2 or matrix.shape[1] != int(state["n_features"]):
        raise ValueError("HGB feature schema/model width mismatch")

    predictions = np.full(
        matrix.shape[0], float(state["baseline"]), dtype=np.float64
    )
    for stored_tree in state["trees"]:
        tree = dict(stored_tree)
        values = np.asarray(tree["value"], dtype=np.float64)
        feature_idx = np.asarray(tree["feature_idx"], dtype=np.intp)
        thresholds = np.asarray(tree["num_threshold"], dtype=np.float64)
        missing_left = np.asarray(
            tree["missing_go_to_left"], dtype=bool
        )
        left_children = np.asarray(tree["left"], dtype=np.intp)
        right_children = np.asarray(tree["right"], dtype=np.intp)
        is_leaf = np.asarray(tree["is_leaf"], dtype=bool)

        node_index = np.zeros(matrix.shape[0], dtype=np.intp)
        active = np.ones(matrix.shape[0], dtype=bool)
        while active.any():
            rows = np.flatnonzero(active)
            nodes = node_index[rows]
            leaf_rows = rows[is_leaf[nodes]]
            if leaf_rows.size:
                predictions[leaf_rows] += values[node_index[leaf_rows]]
                active[leaf_rows] = False

            split_rows = rows[~is_leaf[nodes]]
            if split_rows.size:
                split_nodes = node_index[split_rows]
                split_values = matrix[
                    split_rows, feature_idx[split_nodes]
                ]
                go_left = np.where(
                    np.isnan(split_values),
                    missing_left[split_nodes],
                    split_values <= thresholds[split_nodes],
                )
                node_index[split_rows] = np.where(
                    go_left,
                    left_children[split_nodes],
                    right_children[split_nodes],
                )
    return predictions


def validate_inputs(test: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError("sample submission columns are invalid")
    if len(test) != len(sample):
        raise ValueError("test/sample row counts differ")
    if test[ID_COL].isna().any() or sample[ID_COL].isna().any():
        raise ValueError("row_id contains missing values")
    if test[ID_COL].duplicated().any() or sample[ID_COL].duplicated().any():
        raise ValueError("row_id contains duplicates")
    if set(test[ID_COL]) != set(sample[ID_COL]):
        raise ValueError("test/sample row_id sets differ")


def main() -> None:
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    validate_inputs(test, sample)
    metadata = json.loads(
        (MODEL_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    history = json.loads(
        (MODEL_DIR / "history_state.json").read_text(encoding="utf-8")
    )
    multirate = json.loads(
        (MODEL_DIR / "multirate_state.json").read_text(encoding="utf-8")
    )
    schemas = json.loads(
        (MODEL_DIR / "feature_schemas.json").read_text(encoding="utf-8")
    )

    frame = add_static_features(test.drop(columns=[ID_COL]))
    frame = attach_temporal_features(frame, history)
    frame = attach_multirate_features(frame, multirate)
    temporal_base = frame["temporal_base_global_30"].to_numpy(dtype=float)
    group_state = json.loads(
        (MODEL_DIR / "group_effects.json").read_text(encoding="utf-8")
    )
    group_base = np.clip(
        temporal_base + map_exp018_group(frame, group_state), 0.0, 1.0
    )

    lgb_model = lgb.Booster(
        model_file=str(MODEL_DIR / "rfull_lightgbm.txt")
    )
    hgb_state = json.loads(
        (MODEL_DIR / "histgradientboosting.json").read_text(
            encoding="utf-8"
        )
    )
    if int(lgb_model.num_feature()) != len(schemas["lightgbm"]):
        raise ValueError("LightGBM feature schema/model width mismatch")
    if int(hgb_state["n_features"]) != len(
        schemas["histgradientboosting"]
    ):
        raise ValueError("HGB feature schema/model width mismatch")
    lgb_correction = lgb_model.predict(
        encoded_matrix(frame, schemas["lightgbm"])
    ).astype(float)
    hgb_correction = predict_histgradientboosting(
        encoded_matrix(frame, schemas["histgradientboosting"]),
        hgb_state,
    )
    is_regular = frame["game_type"].astype(str).eq("R").to_numpy()
    lightgbm_branch = group_base.copy()
    histgradient_branch = group_base.copy()
    lightgbm_branch[is_regular] = np.clip(
        group_base[is_regular] + 0.75 * lgb_correction[is_regular],
        0.0,
        1.0,
    )
    histgradient_branch[is_regular] = np.clip(
        group_base[is_regular] + hgb_correction[is_regular],
        0.0,
        1.0,
    )
    backbone = 0.5 * lightgbm_branch + 0.5 * histgradient_branch

    team_state = json.loads(
        (MODEL_DIR / "team_effects.json").read_text(encoding="utf-8")
    )
    team_base = np.clip(backbone + map_team_effects(frame, team_state), 0.0, 1.0)
    candidate = str(metadata["candidate"])
    lowrank_candidates = {
        "strict_lowrank_s300_r6",
        "dualrank_consensus_50",
        "strict_aggressive_consensus_50",
    }
    if candidate in lowrank_candidates:
        lowrank_state = json.loads(
            (MODEL_DIR / "lowrank_effects.json").read_text(encoding="utf-8")
        )
        strict_predictions = np.clip(
            team_base + map_lowrank_effects(frame, lowrank_state), 0.0, 1.0
        )

    aggressive_candidates = {
        "r_gated_team_pc_all",
        "strict_aggressive_consensus_50",
        "recency_aggressive_consensus_50",
    }
    if candidate in aggressive_candidates:
        pitcher_count_state = json.loads(
            (MODEL_DIR / "pitcher_count_effects.json").read_text(
                encoding="utf-8"
            )
        )
        gated = np.where(is_regular, team_base, backbone)
        aggressive_predictions = np.clip(
            gated + map_pitcher_count_effects(frame, pitcher_count_state),
            0.0,
            1.0,
        )

    rank_consensus_candidates = {
        "dualrank_consensus_50",
    }
    if candidate in rank_consensus_candidates:
        r_specific_state = json.loads(
            (MODEL_DIR / "lowrank_rspecific_effects.json").read_text(
                encoding="utf-8"
            )
        )
        r_specific_correction = map_lowrank_effects(frame, r_specific_state)
        r_specific_correction[~is_regular] = 0.0
        r_specific_predictions = np.clip(
            team_base + r_specific_correction, 0.0, 1.0
        )

    if candidate == "recency_aggressive_consensus_50":
        recency_state = json.loads(
            (MODEL_DIR / "lowrank_recency_effects.json").read_text(
                encoding="utf-8"
            )
        )
        recency_predictions = np.clip(
            team_base + map_lowrank_effects(frame, recency_state), 0.0, 1.0
        )

    if candidate == "strict_lowrank_s300_r6":
        predictions = strict_predictions
    elif candidate == "r_gated_team_pc_all":
        predictions = aggressive_predictions
    elif candidate == "dualrank_consensus_50":
        predictions = 0.5 * strict_predictions + 0.5 * r_specific_predictions
    elif candidate == "strict_aggressive_consensus_50":
        predictions = 0.5 * strict_predictions + 0.5 * aggressive_predictions
    elif candidate == "recency_aggressive_consensus_50":
        predictions = 0.5 * recency_predictions + 0.5 * aggressive_predictions
    else:
        raise ValueError(f"unknown candidate: {candidate}")

    prediction_map = dict(zip(test[ID_COL], predictions, strict=True))
    sample[TARGET_COL] = sample[ID_COL].map(prediction_map)
    if sample[TARGET_COL].isna().any():
        raise ValueError("submission contains missing predictions")
    if not sample[TARGET_COL].between(0.0, 1.0).all():
        raise ValueError("submission probability is outside [0, 1]")
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | candidate={candidate} | rows={len(sample)} | "
        f"mean={predictions.mean():.6f} | min={predictions.min():.6f} | "
        f"max={predictions.max():.6f}"
    )


if __name__ == "__main__":
    main()
