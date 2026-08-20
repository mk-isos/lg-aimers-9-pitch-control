"""EXP-072 dynamic-AR candidate on the frozen EXP-051 inference base."""

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


def map_exact_pitchtype_control(
    frame: pd.DataFrame, stored: dict[str, object]
) -> np.ndarray:
    """Apply frozen exact-aligned fine-pitch control EB per current row."""
    mapping = {
        int(key): int(value) for key, value in stored["pitcher_mapping"].items()
    }
    pitcher_rate = {
        int(key): float(value) for key, value in stored["pitcher_rate"].items()
    }
    type_rate = {
        (int(row[0]), str(row[1])): float(row[2])
        for row in stored["type_rate"]
    }
    context_rate = {
        (int(row[0]), str(row[1]), int(row[2]), int(row[3])): float(row[4])
        for row in stored["context_rate"]
    }
    propensity = {
        (int(row[0]), int(row[1]), int(row[2])): np.asarray(row[3], dtype=float)
        for row in stored["propensity"]
    }
    pitch_types = [str(value) for value in stored["fine_pitch_types"]]
    league = float(stored["league_rate"])
    correction = np.zeros(len(frame), dtype=float)
    for position, row in enumerate(frame.itertuples(index=False)):
        trackman_id = mapping.get(int(getattr(row, "pitcher_id")))
        official = float(getattr(row, "asof_pitcher_success_rate"))
        if trackman_id is None or not np.isfinite(official):
            continue
        count_index = int(getattr(row, "count_index"))
        batter_hand = int(getattr(row, "batter_hand"))
        weights = propensity.get((trackman_id, count_index, batter_hand))
        if weights is None or not np.isfinite(weights).all():
            continue
        overall = pitcher_rate.get(trackman_id, league)
        expected = 0.0
        for type_position, pitch_type in enumerate(pitch_types):
            rate = context_rate.get(
                (trackman_id, pitch_type, count_index, batter_hand),
                type_rate.get((trackman_id, pitch_type), overall),
            )
            expected += float(weights[type_position]) * rate
        correction[position] = np.clip(expected - official, -0.03, 0.03)
    return correction


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



def map_dynamic_pitcher_ar(
    frame: pd.DataFrame, stored: dict[str, object]
) -> np.ndarray:
    """Apply frozen EXP-072 AR-k30 correction to each current row alone."""
    required = {"pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"dynamic AR input is missing columns: {missing}")
    if int(stored["version"]) != 1:
        raise ValueError("unsupported dynamic pitcher state version")
    prior = {
        int(row["pitcher_id"]): (
            float(row["prior_n"]), float(row["prior_successes"])
        )
        for row in stored["prior_career_states"]
    }
    latest = {
        int(row["pitcher_id"]): (
            int(row["last_season"]), float(row["latent_logit"])
        )
        for row in stored["latest_latent_states"]
    }
    prediction_season = int(stored["prediction_season"])
    league = float(stored["league_prior"])
    rho = float(stored["rho"])
    strength = float(stored["current_season_prior_strength"])
    if (
        int(stored["league_prior_season"]) != prediction_season - 1
        or max(int(value) for value in stored["source_seasons"])
        >= prediction_season
        or not (0.0 <= rho <= 1.0)
    ):
        raise ValueError("invalid frozen dynamic AR configuration")
    clipped_league = float(np.clip(league, 1e-6, 1.0 - 1e-6))
    league_logit = float(np.log(clipped_league / (1.0 - clipped_league)))
    correction = np.empty(len(frame), dtype=float)
    for position, row in enumerate(frame.itertuples(index=False)):
        pitcher_id = int(getattr(row, "pitcher_id"))
        career_n = float(getattr(row, "asof_pitcher_n"))
        career_rate = float(getattr(row, "asof_pitcher_success_rate"))
        if not np.isfinite(career_rate):
            if career_n != 0.0:
                raise ValueError(
                    "missing pitcher success rate at positive career count"
                )
            career_rate = 0.0
        career_successes = float(np.rint(career_n * career_rate))
        prior_n, prior_successes = prior.get(pitcher_id, (0.0, 0.0))
        season_n = career_n - prior_n
        season_successes = career_successes - prior_successes
        if season_n < -1e-6:
            raise ValueError("career count is below frozen 2024 state")
        if season_successes < -0.01 or season_successes - season_n > 0.01:
            raise ValueError("reconstructed 2025 successes are invalid")
        season_n = max(season_n, 0.0)
        season_successes = float(np.clip(season_successes, 0.0, season_n))
        last_season, last_latent = latest.get(
            pitcher_id, (prediction_season, 0.0)
        )
        gap = max(prediction_season - last_season, 0)
        ar_latent = last_latent * (rho ** gap)
        ar_probability = 1.0 / (
            1.0
            + np.exp(
                -np.clip(league_logit + ar_latent, -30.0, 30.0)
            )
        )
        dynamic = (season_successes + strength * ar_probability) / (
            season_n + strength
        )
        global_posterior = (season_successes + strength * league) / (
            season_n + strength
        )
        correction[position] = dynamic - global_posterior
    return correction


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
        "trackman_recent_consensus_50",
        "trackman_recent_consensus_25",
        "public_simplex_act_25_60_15",
        "trackman_direct_recent_w010",
        "ar_k30_w050",
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
        "recency_aggressive_consensus_70",
        "trackman_recent_consensus_50",
        "trackman_recent_consensus_25",
        "public_simplex_act_25_60_15",
        "trackman_direct_recent_w010",
        "ar_k30_w050",
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

    if candidate in {
        "recency_aggressive_consensus_50",
        "recency_aggressive_consensus_70",
        "trackman_recent_consensus_50",
        "trackman_recent_consensus_25",
        "public_simplex_act_25_60_15",
        "trackman_direct_recent_w010",
        "ar_k30_w050",
    }:
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
    elif candidate == "recency_aggressive_consensus_70":
        predictions = 0.7 * recency_predictions + 0.3 * aggressive_predictions
    elif candidate in {
        "trackman_recent_consensus_50",
        "trackman_recent_consensus_25",
    }:
        exact_state = json.loads(
            (MODEL_DIR / "exact_pitchtype_control.json").read_text(
                encoding="utf-8"
            )
        )
        exact_correction = map_exact_pitchtype_control(frame, exact_state)
        exact_prediction = np.clip(
            strict_predictions + 0.25 * exact_correction, 0.0, 1.0
        )
        recent_prediction = 0.5 * recency_predictions + 0.5 * aggressive_predictions
        trackman_weight = (
            0.5 if candidate == "trackman_recent_consensus_50" else 0.25
        )
        predictions = (
            trackman_weight * exact_prediction
            + (1.0 - trackman_weight) * recent_prediction
        )
    elif candidate == "public_simplex_act_25_60_15":
        exact_state = json.loads(
            (MODEL_DIR / "exact_pitchtype_control.json").read_text(
                encoding="utf-8"
            )
        )
        exact_correction = map_exact_pitchtype_control(frame, exact_state)
        exact_prediction = np.clip(
            strict_predictions + 0.25 * exact_correction, 0.0, 1.0
        )
        predictions = (
            0.25 * aggressive_predictions
            + 0.60 * recency_predictions
            + 0.15 * exact_prediction
        )
    elif candidate == "trackman_direct_recent_w010":
        exact_state = json.loads(
            (MODEL_DIR / "exact_pitchtype_control.json").read_text(
                encoding="utf-8"
            )
        )
        exact_correction = map_exact_pitchtype_control(frame, exact_state)
        recent_prediction = 0.5 * recency_predictions + 0.5 * aggressive_predictions
        predictions = np.clip(
            recent_prediction + 0.10 * exact_correction, 0.0, 1.0
        )
    elif candidate == "ar_k30_w050":
        exact_state = json.loads(
            (MODEL_DIR / "exact_pitchtype_control.json").read_text(
                encoding="utf-8"
            )
        )
        dynamic_state = json.loads(
            (MODEL_DIR / "dynamic_pitcher_state.json").read_text(
                encoding="utf-8"
            )
        )
        exact_correction = map_exact_pitchtype_control(frame, exact_state)
        recent_prediction = 0.5 * recency_predictions + 0.5 * aggressive_predictions
        exp051_prediction = np.clip(
            recent_prediction + 0.10 * exact_correction, 0.0, 1.0
        )
        dynamic_correction = map_dynamic_pitcher_ar(frame, dynamic_state)
        additive_weight = float(dynamic_state["additive_delta_weight"])
        if additive_weight != 0.50:
            raise ValueError("unexpected EXP-072 additive weight")
        predictions = np.clip(
            exp051_prediction + additive_weight * dynamic_correction,
            0.0,
            1.0,
        )
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
    print(f"Saved: {OUTPUT_PATH} | candidate={candidate}")


if __name__ == "__main__":
    main()
