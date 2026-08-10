"""EXP-018 최종 모델을 2025 추론용 네이티브 형식으로 학습한다."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from temporal_residual_features import (
    TARGET,
    add_static_features,
    attach_training_temporal_features,
)
from train_exp018_constrained_multiscale import (
    GROUP_SMOOTHING,
    GROUP_WINDOW,
    MIN_CHILD_SAMPLES,
    NUM_LEAVES,
    RECENT_ITERATIONS,
    RECENT_WEIGHT,
    REVERSE_GROUP_SMOOTHING,
    REVERSE_GROUP_WEIGHT,
    build_group_keys,
    centered_residual,
)


DATA_DIR = Path("./data")
MODEL_DIR = Path("./submissions/EXP-018/model")
VALIDATION_METRICS = Path(
    "./artifacts/EXP-018/constrained_multiscale/validation_metrics.json"
)
ID = "row_id"
ONE_HOT_COLUMNS = ["top_bottom", "game_type", "base_state"]
DROP_MODEL_COLUMNS = [ID, TARGET, "pitcher_id", "batter_id"]
BASE_COLUMN = "temporal_base_global_30"


def records_from_state(frame: pd.DataFrame, id_column: str) -> list[dict[str, object]]:
    records = frame.reset_index().to_dict(orient="records")
    return [
        {
            id_column: int(record[id_column]),
            "prior_n": float(record["prior_n"]),
            "prior_successes": float(record["prior_successes"]),
        }
        for record in records
    ]


def group_effect_records(
    group_keys: pd.DataFrame,
    residual: np.ndarray,
    seasons: np.ndarray,
    columns: list[str],
    smoothing: float,
) -> list[dict[str, object]]:
    train_mask = seasons >= int(seasons.max()) - GROUP_WINDOW + 1
    grouped = group_keys.loc[train_mask, columns].copy()
    grouped["residual"] = residual[train_mask]
    statistics = grouped.groupby(columns, sort=False)["residual"].agg(
        ["sum", "count"]
    )
    effects = (statistics["sum"] / (statistics["count"] + smoothing)).rename(
        "effect"
    )
    records = effects.reset_index().to_dict(orient="records")
    output: list[dict[str, object]] = []
    for record in records:
        clean = {
            column: int(record[column])
            for column in columns
        }
        clean["effect"] = float(record["effect"])
        output.append(clean)
    return output


def main() -> None:
    print("Environment")
    print(f" python={platform.python_version()}")
    print(f" pandas={pd.__version__}")
    print(f" numpy={np.__version__}")
    print(f" lightgbm={lgb.__version__}")
    test_columns = pd.read_csv(
        DATA_DIR / "test.csv", encoding="utf-8-sig", nrows=0
    ).columns
    base_features = [column for column in test_columns if column != ID]
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=base_features + [TARGET],
    )
    train = add_static_features(train)
    train, history_state = attach_training_temporal_features(train, target=TARGET)
    model_features = [
        column for column in train.columns if column not in DROP_MODEL_COLUMNS
    ]
    encoded = pd.get_dummies(
        train[model_features],
        columns=ONE_HOT_COLUMNS,
        dummy_na=True,
        dtype=np.int8,
    )
    for column in encoded.select_dtypes(include=["float64"]).columns:
        encoded[column] = encoded[column].astype("float32")
    for column in encoded.select_dtypes(include=["int64"]).columns:
        encoded[column] = encoded[column].astype("int32")
    feature_names = list(encoded.columns)
    X = encoded.to_numpy(dtype=np.float32)
    y = train[TARGET].to_numpy(dtype=np.float32)
    seasons = train["season"].to_numpy(dtype=np.int16)
    base = train[BASE_COLUMN].to_numpy(dtype=np.float32)
    residual = centered_residual(y, base, seasons)

    final_season = int(seasons.max())
    recent_mask = seasons == final_season
    model = LGBMRegressor(
        objective="regression_l2",
        metric="l2",
        n_estimators=RECENT_ITERATIONS,
        learning_rate=0.015,
        num_leaves=NUM_LEAVES,
        max_depth=-1,
        min_child_samples=MIN_CHILD_SAMPLES,
        max_bin=255,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=4.0,
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    started_at = time.time()
    model.fit(X[recent_mask], residual[recent_mask])
    fit_seconds = time.time() - started_at

    group_keys = build_group_keys(X, feature_names)
    base_group_columns = ["count_index", "pitcher_hand", "batter_hand"]
    reverse_group_columns = base_group_columns + ["reverse_rate_bin"]
    group_effects = {
        "base": group_effect_records(
            group_keys,
            residual,
            seasons,
            base_group_columns,
            GROUP_SMOOTHING,
        ),
        "reverse": group_effect_records(
            group_keys,
            residual,
            seasons,
            reverse_group_columns,
            REVERSE_GROUP_SMOOTHING,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "recent_residual_lightgbm.txt"
    model.booster_.save_model(model_path)
    with (MODEL_DIR / "encoded_features.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(feature_names, file, ensure_ascii=False)
    with (MODEL_DIR / "history_state.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "through_season": int(history_state.through_season),
                "league_rate": float(history_state.league_rate),
                "pitcher": records_from_state(
                    history_state.pitcher, "pitcher_id"
                ),
                "batter": records_from_state(
                    history_state.batter, "batter_id"
                ),
            },
            file,
            ensure_ascii=False,
        )
    with (MODEL_DIR / "group_effects.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(group_effects, file, ensure_ascii=False)

    validation = json.loads(VALIDATION_METRICS.read_text(encoding="utf-8"))
    metadata = {
        "experiment": "EXP-018",
        "training_seasons": sorted(np.unique(seasons).astype(int).tolist()),
        "training_rows": int(len(train)),
        "history_through_season": int(history_state.through_season),
        "recent_residual_training_season": final_season,
        "recent_residual_training_rows": int(recent_mask.sum()),
        "recent_weight": RECENT_WEIGHT,
        "group_window": GROUP_WINDOW,
        "group_smoothing": GROUP_SMOOTHING,
        "reverse_group_weight": REVERSE_GROUP_WEIGHT,
        "reverse_group_smoothing": REVERSE_GROUP_SMOOTHING,
        "model_features": len(feature_names),
        "lightgbm_fit_seconds": fit_seconds,
        "probability_calibration": "identity",
        "validation_aggregate_2022_2024": validation[
            "aggregate_2022_2024"
        ],
        "format": {"lightgbm": "native text", "pickle": False},
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    with (MODEL_DIR / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"fit_seconds={fit_seconds:.1f}")
    for path in sorted(MODEL_DIR.iterdir()):
        print(f"{path.name}={os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
