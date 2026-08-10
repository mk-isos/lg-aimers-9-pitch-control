"""Train, package, and smoke-test the two fixed EXP-021 final candidates."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from exp021_submission_inference import predict_histgradientboosting
from temporal_multirate_features import attach_training_multirate_features
from temporal_residual_features import (
    TARGET,
    add_static_features,
    attach_training_temporal_features,
)
from train_exp019_histgb_residual import (
    ENCODED_PREFIXES,
    STABLE_FEATURES,
    season_equal_weights,
)
from train_exp020_low_rank_pitcher_context_eb import (
    CONTEXTS,
    CONTEXT_TO_POSITION,
    fit_source_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEMPLATE = ROOT / "experiments" / "exp021_submission_inference.py"
TEAM_ROOT = ROOT / "artifacts" / "EXP-019" / "team_eb_ensemble"
LOWRANK_METRICS = (
    ROOT
    / "artifacts"
    / "EXP-020"
    / "low_rank_pitcher_context_eb"
    / "validation_metrics.json"
)
PC_METRICS = (
    ROOT
    / "artifacts"
    / "EXP-020"
    / "pitcher_count_eb_atop_team"
    / "validation_metrics.json"
)
PYTHON = ROOT / ".venv" / "bin" / "python"
SOURCE_SEASONS = (2021, 2022, 2023, 2024)
ID = "row_id"
BASE_COLUMN = "temporal_base_global_30"
EXP018_GROUP_WINDOW = 3
EXP018_GROUP_SMOOTHING = 100.0
EXP018_REVERSE_SMOOTHING = 300.0
TEAM_SMOOTHING = 1000.0
PC_SMOOTHING = 600.0

VARIANTS = {
    "strict": {
        "directory": ROOT / "submissions" / "EXP-021-STRICT",
        "zip": ROOT / "submit_exp021_strict.zip",
        "candidate": "strict_lowrank_s300_r6",
    },
    "aggressive": {
        "directory": ROOT / "submissions" / "EXP-021-AGGRESSIVE",
        "zip": ROOT / "submit_exp021_aggr.zip",
        "candidate": "r_gated_team_pc_all",
    },
}


def state_records(frame: pd.DataFrame, id_column: str) -> list[dict[str, object]]:
    output = []
    for record in frame.reset_index().to_dict(orient="records"):
        output.append(
            {
                id_column: int(record[id_column]),
                "prior_n": float(record["prior_n"]),
                "prior_successes": float(record["prior_successes"]),
            }
        )
    return output


def export_history_state(state: object) -> dict[str, object]:
    return {
        "through_season": int(state.through_season),
        "league_rate": float(state.league_rate),
        "pitcher": state_records(state.pitcher, "pitcher_id"),
        "batter": state_records(state.batter, "batter_id"),
    }


def export_multirate_state(state: object) -> dict[str, object]:
    tables: dict[str, list[dict[str, object]]] = {}
    id_columns = {
        "pitcher_control": "pitcher_id",
        "batter_control": "batter_id",
        "pitcher_pitchmix": "pitcher_id",
    }
    for group_name, table in state.tables.items():
        id_column = id_columns[group_name]
        records = []
        for raw in table.reset_index().to_dict(orient="records"):
            clean: dict[str, object] = {id_column: int(raw[id_column])}
            for key, value in raw.items():
                if key != id_column:
                    clean[key] = float(value)
            records.append(clean)
        tables[group_name] = records
    return {
        "through_season": int(state.through_season),
        "global_rates": {
            key: float(value) for key, value in state.global_rates.items()
        },
        "tables": tables,
    }


def group_keys(frame: pd.DataFrame) -> pd.DataFrame:
    reverse = frame["asof_pitcher_reverse_rate"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "count_index": frame["count_index"].to_numpy(dtype=np.int8),
            "pitcher_hand": frame["pitcher_hand"].to_numpy(dtype=np.int8),
            "batter_hand": frame["batter_hand"].to_numpy(dtype=np.int8),
            "reverse_rate_bin": np.where(
                np.isfinite(reverse), np.floor(reverse / 0.05), -1
            ).astype(np.int16),
        }
    )


def centered_by_season(
    values: np.ndarray,
    seasons: np.ndarray,
) -> np.ndarray:
    result = values.astype(np.float32, copy=True)
    for season in np.unique(seasons):
        mask = seasons == season
        result[mask] -= result[mask].mean()
    return result


def estimate_effect_series(
    rows: pd.DataFrame,
    residual: np.ndarray,
    columns: list[str],
    smoothing: float,
) -> pd.Series:
    grouped = rows.loc[:, columns].copy()
    grouped["residual"] = residual
    stats = grouped.groupby(columns, sort=True)["residual"].agg(
        ["sum", "count"]
    )
    return stats["sum"] / (stats["count"].astype(float) + smoothing)


def effect_records(series: pd.Series, columns: list[str]) -> list[dict[str, object]]:
    records = []
    for raw in series.rename("effect").reset_index().to_dict(orient="records"):
        clean = {column: int(raw[column]) for column in columns}
        clean["effect"] = float(raw["effect"])
        records.append(clean)
    return records


def map_series(
    series: pd.Series,
    rows: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(rows.loc[:, columns])
    return series.reindex(keys).fillna(0.0).to_numpy(dtype=float)


def build_temporal_group_oof(
    frame: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    seasons: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[dict[str, object]]]]:
    keys = group_keys(frame)
    initial_residual = centered_by_season(y - base, seasons)
    predictions = np.empty(len(frame), dtype=np.float64)
    base_columns = ["count_index", "pitcher_hand", "batter_hand"]
    reverse_columns = base_columns + ["reverse_rate_bin"]
    for validation_season in sorted(np.unique(seasons).astype(int).tolist()):
        validation_mask = seasons == validation_season
        train_mask = (
            (seasons < validation_season)
            & (seasons >= validation_season - EXP018_GROUP_WINDOW)
        )
        if train_mask.any():
            base_effect = estimate_effect_series(
                keys.loc[train_mask],
                initial_residual[train_mask],
                base_columns,
                EXP018_GROUP_SMOOTHING,
            )
            reverse_effect = estimate_effect_series(
                keys.loc[train_mask],
                initial_residual[train_mask],
                reverse_columns,
                EXP018_REVERSE_SMOOTHING,
            )
            correction = (
                0.7
                * map_series(
                    base_effect, keys.loc[validation_mask], base_columns
                )
                + 0.3
                * map_series(
                    reverse_effect,
                    keys.loc[validation_mask],
                    reverse_columns,
                )
            )
        else:
            correction = np.zeros(int(validation_mask.sum()), dtype=float)
        predictions[validation_mask] = np.clip(
            base[validation_mask] + correction, 0.0, 1.0
        )

    final_mask = seasons >= int(seasons.max()) - EXP018_GROUP_WINDOW + 1
    final_base = estimate_effect_series(
        keys.loc[final_mask],
        initial_residual[final_mask],
        base_columns,
        EXP018_GROUP_SMOOTHING,
    )
    final_reverse = estimate_effect_series(
        keys.loc[final_mask],
        initial_residual[final_mask],
        reverse_columns,
        EXP018_REVERSE_SMOOTHING,
    )
    exported = {
        "base": effect_records(final_base, base_columns),
        "reverse": effect_records(final_reverse, reverse_columns),
    }
    return predictions, exported


def build_lgb_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    drop = {
        ID,
        TARGET,
        "season",
        "game_type",
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "batter_team_id",
        "top_bottom",
        "base_state",
    }
    numeric_names = [
        column
        for column in frame.columns
        if column not in drop and pd.api.types.is_numeric_dtype(frame[column])
    ]
    numeric = frame[numeric_names].astype(np.float32)
    categorical = pd.get_dummies(
        frame[["top_bottom", "base_state"]],
        dummy_na=True,
        dtype=np.int8,
    )
    names = numeric_names + categorical.columns.tolist()
    matrix = np.column_stack(
        [
            numeric.to_numpy(dtype=np.float32),
            categorical.to_numpy(dtype=np.float32),
        ]
    )
    return np.ascontiguousarray(matrix), names


def build_hgb_matrix(
    frame: pd.DataFrame,
    temporal_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    excluded = {
        ID,
        TARGET,
        "pitcher_id",
        "batter_id",
        "top_bottom",
        "game_type",
        "base_state",
    }
    numeric_names = [
        column
        for column in temporal_columns
        if column not in excluded
        and column in frame.columns
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical = pd.get_dummies(
        frame[["top_bottom", "game_type", "base_state"]],
        dummy_na=True,
        dtype=np.int8,
    )
    full_names = numeric_names + categorical.columns.tolist()
    selected_names = [
        name
        for name in full_names
        if name in STABLE_FEATURES
        or any(name.startswith(prefix) for prefix in ENCODED_PREFIXES)
    ]
    missing = sorted(STABLE_FEATURES.difference(selected_names))
    if missing:
        raise ValueError(f"missing HGB stable features: {missing}")
    numeric_selected = [name for name in selected_names if name in frame.columns]
    dummy_selected = [name for name in selected_names if name in categorical]
    pieces = []
    piece_names = []
    for name in selected_names:
        if name in frame.columns:
            pieces.append(frame[name].to_numpy(dtype=np.float32)[:, None])
            piece_names.append(name)
        elif name in categorical:
            pieces.append(categorical[name].to_numpy(dtype=np.float32)[:, None])
            piece_names.append(name)
    if piece_names != selected_names:
        raise AssertionError("HGB feature order drift")
    del numeric_selected, dummy_selected
    return np.ascontiguousarray(np.column_stack(pieces)), selected_names


def load_source_oof(
    frame: pd.DataFrame,
    y: np.ndarray,
    seasons: np.ndarray,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, pd.DataFrame],
]:
    fixed: dict[int, np.ndarray] = {}
    team: dict[int, np.ndarray] = {}
    rows: dict[int, pd.DataFrame] = {}
    for season in SOURCE_SEASONS:
        mask = seasons == season
        rows[season] = frame.loc[mask].reset_index(drop=True)
        target = y[mask]
        saved_target = np.load(TEAM_ROOT / f"targets_{season}.npy").astype(float)
        fixed[season] = np.load(
            TEAM_ROOT / f"base_ensemble_predictions_{season}.npy"
        ).astype(float)
        team[season] = np.load(
            TEAM_ROOT / f"predictions_all_prior_s1000_{season}.npy"
        ).astype(float)
        if not (
            len(target) == len(saved_target) == len(fixed[season]) == len(team[season])
            and np.array_equal(target, saved_target)
        ):
            raise ValueError(f"source OOF alignment failed: {season}")
    return fixed, team, rows


def build_team_states(
    rows: dict[int, pd.DataFrame],
    y: np.ndarray,
    seasons: np.ndarray,
    fixed: dict[int, np.ndarray],
) -> dict[str, object]:
    result: dict[str, object] = {
        "source_seasons": list(SOURCE_SEASONS),
        "pitcher_team": [],
        "batter_team": [],
    }
    families = {
        "pitcher_team": ["pitcher_team_id", "pitcher_hand", "batter_hand"],
        "batter_team": ["batter_team_id", "pitcher_hand", "batter_hand"],
    }
    for season in SOURCE_SEASONS:
        target = y[seasons == season]
        raw = target - fixed[season]
        residual = raw - raw.mean()
        for family, columns in families.items():
            effect = estimate_effect_series(
                rows[season], residual, columns, TEAM_SMOOTHING
            )
            result[family].append(
                {
                    "season": season,
                    "records": effect_records(effect, columns),
                }
            )
    return result


def build_pc_state(
    rows: dict[int, pd.DataFrame],
    y: np.ndarray,
    seasons: np.ndarray,
    team: dict[int, np.ndarray],
) -> dict[str, object]:
    columns = ["pitcher_id", "count_index", "batter_hand"]
    sources = []
    for season in SOURCE_SEASONS:
        target = y[seasons == season]
        raw = target - team[season]
        residual = raw - raw.mean()
        effect = estimate_effect_series(
            rows[season], residual, columns, PC_SMOOTHING
        )
        sources.append(
            {"season": season, "records": effect_records(effect, columns)}
        )
    return {"smoothing": PC_SMOOTHING, "sources": sources}


def build_lowrank_state(
    rows: dict[int, pd.DataFrame],
    y: np.ndarray,
    seasons: np.ndarray,
    team: dict[int, np.ndarray],
) -> dict[str, object]:
    sources = []
    for season in SOURCE_SEASONS:
        source_rows = rows[season].copy()
        source_rows["context_position"] = [
            CONTEXT_TO_POSITION[(int(count), int(hand))]
            for count, hand in zip(
                source_rows["count_index"],
                source_rows["batter_hand"],
                strict=True,
            )
        ]
        model = fit_source_matrix(
            season,
            source_rows,
            y[seasons == season],
            team[season],
            smoothing_grid=(300.0,),
            rank_grid=(6,),
        )
        reconstruction = model["matrices"][300.0]["reconstructions"][6]
        sources.append(
            {
                "season": season,
                "pitcher_ids": [int(value) for value in model["pitcher_ids"]],
                "values": reconstruction.astype(float).tolist(),
            }
        )
    contexts = [
        {
            "position": position,
            "count_index": count,
            "batter_hand": hand,
        }
        for position, (count, hand) in enumerate(CONTEXTS)
    ]
    return {
        "smoothing": 300.0,
        "rank": 6,
        "contexts": contexts,
        "sources": sources,
    }


def write_json(path: Path, value: object, indent: int | None = None) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def export_histgradientboosting(model: object) -> dict[str, object]:
    """Export numerical HGB trees without sklearn/NumPy pickle metadata."""
    if int(model.n_trees_per_iteration_) != 1:
        raise ValueError("only single-output HGB models are supported")
    trees = []
    for iteration in model._predictors:
        predictor = iteration[0]
        nodes = predictor.nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("categorical HGB nodes are not supported")
        trees.append(
            {
                "value": nodes["value"].astype(float).tolist(),
                "feature_idx": nodes["feature_idx"].astype(int).tolist(),
                "num_threshold": nodes["num_threshold"].astype(float).tolist(),
                "missing_go_to_left": nodes["missing_go_to_left"]
                .astype(int)
                .tolist(),
                "left": nodes["left"].astype(int).tolist(),
                "right": nodes["right"].astype(int).tolist(),
                "is_leaf": nodes["is_leaf"].astype(int).tolist(),
            }
        )
    return {
        "format": "numeric_hgb_v1",
        "n_features": int(model.n_features_in_),
        "baseline": float(model._baseline_prediction.ravel()[0]),
        "trees": trees,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_zip(source: Path, output: Path) -> dict[str, object]:
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        archive.write(source / "script.py", "script.py")
        archive.write(source / "requirements.txt", "requirements.txt")
        for path in sorted((source / "model").iterdir()):
            if path.is_file():
                archive.write(path, f"model/{path.name}")
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure: {bad}")
        names = archive.namelist()
        if names[:2] != ["script.py", "requirements.txt"]:
            raise ValueError("ZIP root order is invalid")
        if not all(
            name in {"script.py", "requirements.txt"}
            or name.startswith("model/")
            for name in names
        ):
            raise ValueError("ZIP contains an invalid path")
        if any(
            Path(name).suffix.lower() in {".csv", ".npy", ".npz", ".pkl", ".pickle"}
            for name in names
        ):
            raise ValueError("ZIP contains data, prediction, or pickle files")
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "crc": "passed",
        "files": names,
    }


def smoke_test(output: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="exp021-smoke-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(output) as archive:
            archive.extractall(stage)
        (stage / "data").symlink_to(DATA_DIR, target_is_directory=True)
        started = time.time()
        result = subprocess.run(
            [str(PYTHON), "script.py"],
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
        )
        runtime = time.time() - started
        prediction = pd.read_csv(stage / "output" / "submission.csv")
        test = pd.read_csv(DATA_DIR / "test.csv", usecols=[ID])
        sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
        if len(prediction) != len(test):
            raise ValueError("smoke row count mismatch")
        if not prediction[ID].equals(sample[ID]):
            raise ValueError("smoke row order mismatch")
        if prediction[ID].duplicated().any():
            raise ValueError("smoke duplicate row_id")
        values = prediction[TARGET].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not (
            (values >= 0.0).all() and (values <= 1.0).all()
        ):
            raise ValueError("smoke probabilities are invalid")
        return {
            "runtime_seconds": runtime,
            "stdout": result.stdout.strip(),
            "rows": len(prediction),
            "row_id_order": "passed",
            "row_id_unique": "passed",
            "missing": int(np.isnan(values).sum()),
            "range": "passed",
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }


def main() -> None:
    started = time.time()
    print(
        f"python={platform.python_version()} pandas={pd.__version__} "
        f"numpy={np.__version__} lightgbm={lgb.__version__} "
        f"sklearn={sklearn.__version__}",
        flush=True,
    )
    train_header = pd.read_csv(DATA_DIR / "train.csv", nrows=0).columns
    base_columns = [
        column for column in train_header if column not in {ID, TARGET}
    ]
    frame = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=base_columns + [TARGET],
    )
    frame = add_static_features(frame)
    frame, history_state = attach_training_temporal_features(
        frame, target=TARGET
    )
    temporal_columns = list(frame.columns)
    frame, multirate_state, multirate_diagnostics = (
        attach_training_multirate_features(frame, target=TARGET)
    )
    y = frame[TARGET].to_numpy(dtype=np.float32)
    seasons = frame["season"].to_numpy(dtype=np.int16)
    base = frame[BASE_COLUMN].to_numpy(dtype=np.float32)
    is_regular = frame["game_type"].astype(str).eq("R").to_numpy()
    group_predictions, final_group_state = build_temporal_group_oof(
        frame, y, base, seasons
    )
    residual_target = (y.astype(float) - group_predictions).astype(np.float32)
    for season in np.unique(seasons):
        mask = (seasons == season) & is_regular
        residual_target[mask] -= residual_target[mask].mean()
    weights = season_equal_weights(seasons[is_regular])

    shared = Path(tempfile.mkdtemp(prefix="exp021-final-models-"))
    lgb_matrix, lgb_names = build_lgb_matrix(frame)
    lgb_model = LGBMRegressor(
        objective="regression_l2",
        metric="l2",
        n_estimators=300,
        learning_rate=0.015,
        num_leaves=63,
        min_child_samples=1000,
        max_bin=255,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=8.0,
        random_state=42,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    fit_started = time.time()
    lgb_model.fit(
        lgb_matrix[is_regular],
        residual_target[is_regular],
        sample_weight=weights,
    )
    lgb_fit_seconds = time.time() - fit_started
    lgb_model.booster_.save_model(shared / "rfull_lightgbm.txt")
    del lgb_matrix, lgb_model
    print(f"LGB fit {lgb_fit_seconds:.1f}s", flush=True)

    hgb_matrix, hgb_names = build_hgb_matrix(frame, temporal_columns)
    hgb_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.025,
        max_iter=160,
        max_leaf_nodes=15,
        max_depth=4,
        min_samples_leaf=3000,
        l2_regularization=30.0,
        max_features=0.70,
        max_bins=127,
        early_stopping=False,
        random_state=42,
    )
    fit_started = time.time()
    hgb_model.fit(
        hgb_matrix[is_regular],
        residual_target[is_regular],
        sample_weight=weights,
    )
    hgb_fit_seconds = time.time() - fit_started
    hgb_state = export_histgradientboosting(hgb_model)
    parity_indices = np.linspace(
        0,
        len(hgb_matrix) - 1,
        num=min(4096, len(hgb_matrix)),
        dtype=np.int64,
    )
    hgb_native_parity = hgb_model.predict(hgb_matrix[parity_indices])
    hgb_json_parity = predict_histgradientboosting(
        hgb_matrix[parity_indices], hgb_state
    )
    hgb_export_parity_max_abs = float(
        np.max(np.abs(hgb_native_parity - hgb_json_parity))
    )
    if hgb_export_parity_max_abs > 1e-12:
        raise AssertionError(
            "HGB JSON export parity failed: "
            f"max_abs={hgb_export_parity_max_abs}"
        )
    del hgb_matrix, hgb_model
    print(f"HGB fit {hgb_fit_seconds:.1f}s", flush=True)

    fixed_oof, team_oof, source_rows = load_source_oof(
        frame, y, seasons
    )
    team_state = build_team_states(
        source_rows, y, seasons, fixed_oof
    )
    pc_state = build_pc_state(source_rows, y, seasons, team_oof)
    lowrank_state = build_lowrank_state(
        source_rows, y, seasons, team_oof
    )
    history_json = export_history_state(history_state)
    multirate_json = export_multirate_state(multirate_state)
    schemas = {
        "lightgbm": lgb_names,
        "histgradientboosting": hgb_names,
    }
    shared_json = {
        "history_state.json": history_json,
        "multirate_state.json": multirate_json,
        "feature_schemas.json": schemas,
        "histgradientboosting.json": hgb_state,
        "group_effects.json": final_group_state,
        "team_effects.json": team_state,
        "pitcher_count_effects.json": pc_state,
        "lowrank_effects.json": lowrank_state,
    }
    lowrank_metrics = json.loads(LOWRANK_METRICS.read_text(encoding="utf-8"))[
        "aggregate_2022_2024"
    ]["lowrank_s300_r6"]
    pc_metrics = json.loads(PC_METRICS.read_text(encoding="utf-8"))[
        "aggregate_2022_2024"
    ]["r_gated_team_pc_all"]

    results: dict[str, object] = {}
    # The evaluation image already provides NumPy, pandas, scikit-learn, and
    # joblib. Pinning the newer local training versions here can make package
    # installation fail on the server's Python 3.11 runtime. Only install the
    # inference dependency that is absent from the base image.
    requirements = "lightgbm==4.6.0\n"
    for label, details in VARIANTS.items():
        source = details["directory"]
        model_dir = source / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TEMPLATE, source / "script.py")
        (source / "requirements.txt").write_text(
            requirements, encoding="utf-8"
        )
        shutil.copyfile(
            shared / "rfull_lightgbm.txt",
            model_dir / "rfull_lightgbm.txt",
        )
        (model_dir / "histgradientboosting.joblib").unlink(
            missing_ok=True
        )
        for filename, value in shared_json.items():
            write_json(model_dir / filename, value)
        candidate = str(details["candidate"])
        metrics = lowrank_metrics if label == "strict" else pc_metrics
        metadata = {
            "experiment": "EXP-021",
            "candidate": candidate,
            "training_rows": int(len(frame)),
            "training_seasons": sorted(
                np.unique(seasons).astype(int).tolist()
            ),
            "history_through_season": int(history_state.through_season),
            "full_fit_backbone": {
                "base": "EXP018 temporal base + last3 group",
                "R_lightgbm": "rfull_l63_m1000_i300 weight 0.75",
                "R_histgradientboosting": "hist_l15_d4_m3000_i160 weight 1.0",
                "ensemble": "fixed 50:50",
                "F": "EXP018 temporal base + last3 group",
            },
            "source_effect_seasons": list(SOURCE_SEASONS),
            "source_combination": "equal average; missing mapping contributes zero",
            "probability_calibration": "identity",
            "validation_aggregate_2022_2024": metrics,
            "model_fit_seconds": {
                "lightgbm": lgb_fit_seconds,
                "histgradientboosting": hgb_fit_seconds,
            },
            "feature_counts": {
                "lightgbm": len(lgb_names),
                "histgradientboosting": len(hgb_names),
            },
            "multirate_reconstruction_diagnostics": multirate_diagnostics,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "lightgbm": lgb.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
        write_json(model_dir / "metadata.json", metadata, indent=2)
        zip_result = build_zip(source, details["zip"])
        smoke = smoke_test(details["zip"])
        results[label] = {
            "candidate": candidate,
            "validation": metrics,
            "zip": zip_result,
            "smoke": smoke,
        }
        print(f"{label} smoke passed", flush=True)

    report = {
        "experiment": "EXP-021",
        "stage": "final_candidate_packages",
        "results": results,
        "qa": {
            "hgb_json_export_parity_rows": int(len(parity_indices)),
            "hgb_json_export_max_abs": hgb_export_parity_max_abs,
            "hgb_json_export_tolerance": 1e-12,
            "hgb_json_export_passed": True,
        },
        "total_seconds": time.time() - started,
    }
    artifact_dir = ROOT / "artifacts" / "EXP-021" / "final_packages"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "validation_metrics.json", report, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
