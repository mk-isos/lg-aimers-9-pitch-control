"""Build the prospective EXP-072 2025 dynamic-AR submission package.

The package keeps the frozen EXP-051 predictor intact and adds one row-local
correction selected prospectively by EXP-072: ``ar_k30_w050``.  Every dynamic
state is fitted once from official 2019-2024 training rows and serialized.  At
inference, a row consumes only its own pitcher id, career count and career
success rate plus the frozen state; no query-row aggregate or retraining is
allowed.

This builder intentionally never opens test.csv or sample_submission.csv.
Its smoke test uses source-derived state and constructed synthetic rows only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from build_exp021_final_candidates import build_zip
from train_exp072_dynamic_pitcher_state import (
    EPS,
    STATE_SMOOTHING,
    TRANSITION_RIDGE,
    PriorCareerState,
    dynamic_deltas,
    end_state,
    fit_transition,
    latest_latent_before,
    season_latent_states,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
SOURCE_DIR = ROOT / "submissions" / "EXP-051-TMDIRECT"
DESTINATION = ROOT / "submissions" / "EXP-072-DYNAMIC-AR"
READY_DIR = ROOT / "ready_to_submit" / "2026-08-20-post58"
ZIP_PATH = READY_DIR / "submit_exp072_dynamic_ar_k30_w050.zip"
REPORT_PATH = READY_DIR / "submit_exp072_dynamic_ar_k30_w050.report.json"
VALIDATION_REPORT = (
    ROOT / "artifacts" / "EXP-072" / "dynamic_pitcher_state"
    / "validation_metrics.json"
)

EXPERIMENT = "EXP-072"
CANDIDATE = "ar_k30_w050"
PREDICTION_SEASON = 2025
SOURCE_SEASONS = tuple(range(2019, 2025))
PRIOR_STRENGTH = 30.0
ADDITIVE_WEIGHT = 0.50
STATE_FILENAME = "dynamic_pitcher_state.json"
STATE_VERSION = 1
SOURCE_COLUMNS = (
    "season",
    "pitcher_id",
    "batter_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_batter_n",
    "control_success",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_training_frame(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read only the official source train fields required by EXP-072."""

    if path.name != "train.csv":
        raise ValueError("EXP-072 state fitting accepts train.csv only")
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=list(SOURCE_COLUMNS))
    missing = sorted(set(SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"official source train is missing columns: {missing}")
    required = [name for name in SOURCE_COLUMNS if name != "asof_pitcher_success_rate"]
    if frame.loc[:, required].isna().any().any():
        raise ValueError("official source train has a missing required value")
    missing_rate = frame["asof_pitcher_success_rate"].isna()
    if not frame.loc[missing_rate, "asof_pitcher_n"].eq(0).all():
        raise ValueError("pitcher success rate is missing at positive career count")
    frame["asof_pitcher_success_rate"] = frame[
        "asof_pitcher_success_rate"
    ].fillna(0.0)
    seasons = tuple(sorted(frame["season"].astype(int).unique().tolist()))
    if seasons != SOURCE_SEASONS:
        raise ValueError(
            f"expected official source seasons {SOURCE_SEASONS}, found {seasons}"
        )
    if not frame["season"].is_monotonic_increasing:
        raise ValueError("official source train must be season-monotone")
    return frame


def full_prior_career_state(frame: pd.DataFrame) -> PriorCareerState:
    """Return each pitcher's career state immediately after the last source row."""

    latest_n = pd.Series(dtype=float)
    latest_successes = pd.Series(dtype=float)
    for season in sorted(frame["season"].astype(int).unique()):
        state = end_state(frame.loc[frame["season"].eq(season)])
        latest_n = pd.concat([latest_n, state.n])
        latest_n = latest_n.loc[~latest_n.index.duplicated(keep="last")]
        latest_successes = pd.concat([latest_successes, state.successes])
        latest_successes = latest_successes.loc[
            ~latest_successes.index.duplicated(keep="last")
        ]
    latest_n = latest_n.sort_index()
    latest_successes = latest_successes.reindex(latest_n.index)
    if latest_successes.isna().any():
        raise RuntimeError("full career success state does not match count state")
    return PriorCareerState(n=latest_n, successes=latest_successes)


def _finite_float(value: object, label: str) -> float:
    output = float(value)
    if not np.isfinite(output):
        raise ValueError(f"non-finite {label}")
    return output


def build_dynamic_state(
    frame: pd.DataFrame,
    *,
    prediction_season: int = PREDICTION_SEASON,
    expected_source_seasons: tuple[int, ...] = SOURCE_SEASONS,
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit and serialize an EXP-072 state at one strict season cutoff."""

    observed_seasons = tuple(sorted(frame["season"].astype(int).unique().tolist()))
    if observed_seasons != expected_source_seasons:
        raise ValueError(
            f"expected dynamic source seasons {expected_source_seasons}, "
            f"found {observed_seasons}"
        )
    if max(observed_seasons) >= prediction_season:
        raise ValueError("dynamic source must strictly precede prediction season")

    states, league_rates = season_latent_states(frame)
    rho, transition_audit = fit_transition(states, prediction_season)
    latest = latest_latent_before(states, prediction_season).sort_index()
    career = full_prior_career_state(frame)

    state_records: list[dict[str, object]] = []
    for row in states.reset_index().sort_values(["season", "pitcher_id"]).itertuples(
        index=False
    ):
        state_records.append(
            {
                "season": int(row.season),
                "pitcher_id": int(row.pitcher_id),
                "successes": _finite_float(row.sum, "season successes"),
                "count": _finite_float(row.count, "season count"),
                "league_rate": _finite_float(row.league_rate, "season league rate"),
                "posterior_rate": _finite_float(
                    row.posterior_rate, "season posterior rate"
                ),
                "latent_logit": _finite_float(row.latent_logit, "season latent"),
                "reliability": _finite_float(row.reliability, "season reliability"),
            }
        )

    prior_records = [
        {
            "pitcher_id": int(pitcher_id),
            "prior_n": _finite_float(career.n.loc[pitcher_id], "prior career n"),
            "prior_successes": _finite_float(
                career.successes.loc[pitcher_id], "prior career successes"
            ),
        }
        for pitcher_id in career.n.index
    ]
    latest_records = [
        {
            "pitcher_id": int(pitcher_id),
            "last_season": int(latest.loc[pitcher_id, "season"]),
            "latent_logit": _finite_float(
                latest.loc[pitcher_id, "latent_logit"], "latest latent"
            ),
            "season_count": _finite_float(
                latest.loc[pitcher_id, "count"], "latest season count"
            ),
        }
        for pitcher_id in latest.index
    ]
    league_prior_season = prediction_season - 1
    league_prior = _finite_float(
        league_rates[league_prior_season], f"{league_prior_season} league rate"
    )
    state: dict[str, object] = {
        "version": STATE_VERSION,
        "experiment": EXPERIMENT,
        "candidate": CANDIDATE,
        "prediction_season": prediction_season,
        "source_seasons": list(observed_seasons),
        "source_rows": int(len(frame)),
        "state_smoothing": float(STATE_SMOOTHING),
        "transition_ridge": float(TRANSITION_RIDGE),
        "transition": "bounded zero-intercept AR(1), rho in [0,1]",
        "rho": _finite_float(rho, "rho"),
        "league_prior_season": league_prior_season,
        "league_prior": league_prior,
        "current_season_prior_strength": PRIOR_STRENGTH,
        "additive_delta_weight": ADDITIVE_WEIGHT,
        "missing_or_new_pitcher_effect": 0.0,
        "gap_transition": "rho ** (prediction_season - last_observed_season)",
        "season_latent_states": state_records,
        "latest_latent_states": latest_records,
        "prior_career_states": prior_records,
        "inference_inputs": [
            "pitcher_id",
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
        ],
        "row_local": True,
        "inference_retraining": False,
        "query_row_aggregation": False,
    }
    if prediction_season == PREDICTION_SEASON:
        # Retain the explicit competition-season name requested by the build
        # protocol while inference itself consumes the generic frozen prior.
        state["league_2024"] = league_prior
    audit = {
        **transition_audit,
        "source_rows": int(len(frame)),
        "source_seasons": list(observed_seasons),
        "season_latent_state_records": len(state_records),
        "latest_latent_state_records": len(latest_records),
        "prior_career_state_records": len(prior_records),
        "prediction_season": prediction_season,
        "league_prior_season": league_prior_season,
        "league_prior": league_prior,
    }
    return state, audit


def dynamic_ar_correction(
    frame: pd.DataFrame, stored: dict[str, object]
) -> np.ndarray:
    """Reference implementation of the package's independent row correction."""

    required = {"pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"dynamic AR input is missing columns: {missing}")
    if int(stored["version"]) != STATE_VERSION:
        raise ValueError("unsupported dynamic pitcher state version")
    prior = {
        int(row["pitcher_id"]): (float(row["prior_n"]), float(row["prior_successes"]))
        for row in stored["prior_career_states"]
    }
    latest = {
        int(row["pitcher_id"]): (int(row["last_season"]), float(row["latent_logit"]))
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
    league_logit = float(np.log(np.clip(league, EPS, 1.0 - EPS) / np.clip(1.0 - league, EPS, 1.0)))
    correction = np.empty(len(frame), dtype=float)
    for position, row in enumerate(frame.itertuples(index=False)):
        pitcher_id = int(getattr(row, "pitcher_id"))
        career_n = float(getattr(row, "asof_pitcher_n"))
        career_rate = float(getattr(row, "asof_pitcher_success_rate"))
        if not np.isfinite(career_rate):
            if career_n != 0.0:
                raise ValueError("missing pitcher success rate at positive career count")
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
        last_season, last_latent = latest.get(pitcher_id, (prediction_season, 0.0))
        gap = max(prediction_season - last_season, 0)
        ar_latent = last_latent * (rho**gap)
        ar_probability = 1.0 / (1.0 + np.exp(-np.clip(league_logit + ar_latent, -30.0, 30.0)))
        dynamic = (season_successes + strength * ar_probability) / (season_n + strength)
        global_posterior = (season_successes + strength * league) / (season_n + strength)
        correction[position] = dynamic - global_posterior
    return correction


PACKAGE_DYNAMIC_HELPER = r'''
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
'''.strip()


def render_submission_script(base_source: str) -> str:
    """Add the frozen EXP-072 branch without changing EXP-051 computations."""

    output = base_source.replace(
        '"""EXP-021 final candidate inference (copied to the ZIP root as script.py)."""',
        '"""EXP-072 dynamic-AR candidate on the frozen EXP-051 inference base."""',
        1,
    )
    helper_marker = "\ndef validate_inputs(test: pd.DataFrame, sample: pd.DataFrame) -> None:\n"
    if output.count(helper_marker) != 1:
        raise ValueError("EXP-051 template validate_inputs marker changed")
    output = output.replace(
        helper_marker,
        f"\n\n{PACKAGE_DYNAMIC_HELPER}\n\n\ndef validate_inputs(test: pd.DataFrame, sample: pd.DataFrame) -> None:\n",
        1,
    )

    set_marker = '        "trackman_direct_recent_w010",\n    }'
    if output.count(set_marker) != 3:
        raise ValueError("EXP-051 template candidate sets changed")
    output = output.replace(
        set_marker,
        '        "trackman_direct_recent_w010",\n'
        '        "ar_k30_w050",\n'
        "    }",
    )

    branch_marker = '''        predictions = np.clip(
            recent_prediction + 0.10 * exact_correction, 0.0, 1.0
        )
    else:
'''
    if output.count(branch_marker) != 1:
        raise ValueError("EXP-051 template candidate branch changed")
    dynamic_branch = '''        predictions = np.clip(
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
'''
    output = output.replace(branch_marker, dynamic_branch, 1)

    aggregate_log = '''    print(
        f"Saved: {OUTPUT_PATH} | candidate={candidate} | rows={len(sample)} | "
        f"mean={predictions.mean():.6f} | min={predictions.min():.6f} | "
        f"max={predictions.max():.6f}"
    )
'''
    if output.count(aggregate_log) != 1:
        raise ValueError("EXP-051 template aggregate log changed")
    output = output.replace(
        aggregate_log,
        '    print(f"Saved: {OUTPUT_PATH} | candidate={candidate}")\n',
        1,
    )
    forbidden = (
        "predictions.mean()",
        "predictions.min()",
        "predictions.max()",
        "rows={len(sample)}",
    )
    if any(value in output for value in forbidden):
        raise RuntimeError("rendered script retains a query-prediction aggregate log")
    return output


def load_rendered_module(script_path: Path) -> ModuleType:
    name = f"exp072_package_smoke_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load rendered EXP-072 script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_source_rows(stored: dict[str, object]) -> pd.DataFrame:
    """Construct legal 2025-like rows solely from frozen source state."""

    prior_records = list(stored["prior_career_states"])
    if len(prior_records) < 2:
        raise ValueError("synthetic smoke requires at least two source pitchers")
    records: list[dict[str, object]] = []
    for position, record in enumerate(prior_records[:4]):
        addition_n = float(position * 7)
        addition_successes = float(position * 3)
        career_n = float(record["prior_n"]) + addition_n
        career_successes = float(record["prior_successes"]) + addition_successes
        records.append(
            {
                "pitcher_id": int(record["pitcher_id"]),
                "asof_pitcher_n": career_n,
                "asof_pitcher_success_rate": (
                    career_successes / career_n if career_n > 0.0 else np.nan
                ),
            }
        )
    unknown_id = max(int(row["pitcher_id"]) for row in prior_records) + 10_000_003
    records.append(
        {
            "pitcher_id": unknown_id,
            "asof_pitcher_n": 0.0,
            "asof_pitcher_success_rate": np.nan,
        }
    )
    return pd.DataFrame.from_records(records)


def synthetic_invariance_smoke(
    script_path: Path, stored: dict[str, object]
) -> dict[str, object]:
    """Audit singleton/permutation/split/duplicate behavior without test data."""

    module = load_rendered_module(script_path)
    rows = synthetic_source_rows(stored)
    normal = module.map_dynamic_pitcher_ar(rows, stored)
    reference = dynamic_ar_correction(rows, stored)
    singleton = np.concatenate(
        [
            module.map_dynamic_pitcher_ar(rows.iloc[[position]], stored)
            for position in range(len(rows))
        ]
    )
    reversed_prediction = module.map_dynamic_pitcher_ar(
        rows.iloc[::-1].reset_index(drop=True), stored
    )[::-1]
    midpoint = max(1, len(rows) // 2)
    split = np.concatenate(
        [
            module.map_dynamic_pitcher_ar(rows.iloc[:midpoint], stored),
            module.map_dynamic_pitcher_ar(rows.iloc[midpoint:], stored),
        ]
    )
    duplicated_rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    duplicate = module.map_dynamic_pitcher_ar(duplicated_rows, stored)
    differences = {
        "reference_implementation": float(np.max(np.abs(normal - reference))),
        "singleton": float(np.max(np.abs(normal - singleton))),
        "reversed_permutation": float(
            np.max(np.abs(normal - reversed_prediction))
        ),
        "split_batch": float(np.max(np.abs(normal - split))),
        "duplicate_original_rows": float(
            np.max(np.abs(normal - duplicate[: len(rows)]))
        ),
        "duplicated_row_itself": float(abs(normal[0] - duplicate[-1])),
    }
    synthetic_base = np.linspace(0.35, 0.65, len(rows), dtype=float)
    candidate = np.clip(synthetic_base + ADDITIVE_WEIGHT * normal, 0.0, 1.0)
    tolerance = 1e-12
    passed = bool(
        max(differences.values()) <= tolerance
        and np.isfinite(candidate).all()
        and ((0.0 <= candidate) & (candidate <= 1.0)).all()
    )
    if not passed:
        raise RuntimeError(f"EXP-072 source-only synthetic smoke failed: {differences}")
    return {
        "source": "constructed rows from frozen official-train state only",
        "rows": int(len(rows)),
        "tolerance": tolerance,
        "maximum_absolute_differences": differences,
        "probability_range": "passed",
        "singleton_reverse_split_duplicate_invariance": "passed",
        "query_row_aggregation": False,
        "test_or_sample_file_opened": False,
    }


def source_only_package_fixture(
    path: Path = DATA_PATH, rows: int = 4
) -> pd.DataFrame:
    """Create unseen-id season-2025 inference rows from source schema only."""

    if path.name != "train.csv":
        raise ValueError("full package smoke accepts train.csv as schema source only")
    source = pd.read_csv(path, encoding="utf-8-sig", nrows=rows)
    if len(source) != rows:
        raise ValueError("not enough source rows for full package smoke")
    source = source.drop(columns=["control_success"], errors="ignore").copy()
    if "row_id" not in source:
        source.insert(0, "row_id", np.arange(rows, dtype=np.int64))
    positions = np.arange(rows, dtype=np.int64)
    source["row_id"] = 910_000_000 + positions
    source["season"] = PREDICTION_SEASON
    source["pitcher_id"] = 1_800_000_000 + positions
    source["batter_id"] = 1_700_000_000 + positions
    source["pitcher_team_id"] = 1_600_000_000 + positions
    source["batter_team_id"] = 1_500_000_000 + positions
    for column in (
        "asof_pitcher_n",
        "asof_batter_n",
        "asof_pitcher_pitchmix_n",
    ):
        source[column] = 0
    for column in source.columns:
        if column.startswith("asof_") and column.endswith("_rate"):
            source[column] = np.nan
    return source


def _run_staged_package(stage: Path, rows: pd.DataFrame) -> tuple[pd.Series, str]:
    data_dir = stage / "data"
    output_dir = stage / "output"
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    rows.to_csv(data_dir / "test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "row_id": rows["row_id"].to_numpy(),
            "control_success": np.zeros(len(rows), dtype=float),
        }
    ).to_csv(data_dir / "sample_submission.csv", index=False, encoding="utf-8-sig")
    result = subprocess.run(
        [sys.executable, "script.py"],
        cwd=stage,
        check=True,
        capture_output=True,
        text=True,
    )
    output = pd.read_csv(output_dir / "submission.csv", encoding="utf-8-sig")
    if not output["row_id"].equals(rows["row_id"].reset_index(drop=True)):
        raise RuntimeError("synthetic package output row-id order changed")
    prediction = output.set_index("row_id")["control_success"].astype(float)
    if not np.isfinite(prediction.to_numpy()).all():
        raise RuntimeError("synthetic package emitted a non-finite prediction")
    if "mean=" in result.stdout or "min=" in result.stdout or "max=" in result.stdout:
        raise RuntimeError("package logged a synthetic query prediction aggregate")
    return prediction, result.stdout.strip()


def full_package_synthetic_smoke(
    destination: Path, path: Path = DATA_PATH
) -> dict[str, object]:
    """Run the complete package with source-schema synthetic query variants."""

    fixture = source_only_package_fixture(path)
    with tempfile.TemporaryDirectory(prefix="exp072-source-smoke-") as temporary:
        stage = Path(temporary) / "package"
        shutil.copytree(destination, stage)
        normal, stdout = _run_staged_package(stage, fixture)

        singleton_parts = []
        for position in range(len(fixture)):
            values, _ = _run_staged_package(stage, fixture.iloc[[position]].copy())
            singleton_parts.append(values)
        singleton = pd.concat(singleton_parts).reindex(normal.index)

        reverse, _ = _run_staged_package(
            stage, fixture.iloc[::-1].reset_index(drop=True)
        )
        reverse = reverse.reindex(normal.index)

        midpoint = max(1, len(fixture) // 2)
        split_left, _ = _run_staged_package(stage, fixture.iloc[:midpoint].copy())
        split_right, _ = _run_staged_package(stage, fixture.iloc[midpoint:].copy())
        split = pd.concat([split_left, split_right]).reindex(normal.index)

        duplicate_rows = pd.concat([fixture, fixture.iloc[[0]]], ignore_index=True)
        duplicate_id = int(fixture["row_id"].max()) + 1
        duplicate_rows.loc[len(duplicate_rows) - 1, "row_id"] = duplicate_id
        duplicate, _ = _run_staged_package(stage, duplicate_rows)

    normal_values = normal.to_numpy(dtype=float)
    differences = {
        "singleton": float(
            np.max(np.abs(normal_values - singleton.to_numpy(dtype=float)))
        ),
        "reversed_permutation": float(
            np.max(np.abs(normal_values - reverse.to_numpy(dtype=float)))
        ),
        "split_batch": float(
            np.max(np.abs(normal_values - split.to_numpy(dtype=float)))
        ),
        "duplicate_original_rows": float(
            np.max(
                np.abs(
                    normal_values
                    - duplicate.reindex(normal.index).to_numpy(dtype=float)
                )
            )
        ),
        "duplicated_row_itself": float(
            abs(float(normal.iloc[0]) - float(duplicate.loc[duplicate_id]))
        ),
    }
    tolerance = 1e-12
    if max(differences.values()) > tolerance:
        raise RuntimeError(f"full package row invariance failed: {differences}")
    return {
        "source": "official train schema mutated to season 2025 and unseen raw ids",
        "rows": int(len(fixture)),
        "tolerance": tolerance,
        "maximum_absolute_differences": differences,
        "singleton_reverse_split_duplicate_invariance": "passed",
        "exp051_history_checks": "passed",
        "aggregate_prediction_log": False,
        "stdout": stdout,
        "actual_test_or_sample_file_opened": False,
    }


def cutoff_2023_parity_audit(
    frame: pd.DataFrame, script_path: Path
) -> dict[str, object]:
    """Match the original EXP-072 2024 correction at a strict 2023 cutoff."""

    cutoff = frame.loc[frame["season"] <= 2023].copy()
    validation = frame.loc[frame["season"] == 2024].reset_index(drop=True)
    states, league_rates = season_latent_states(cutoff)
    career = full_prior_career_state(cutoff)
    original, _ = dynamic_deltas(
        validation,
        2024,
        states,
        league_rates,
        career,
    )
    stored, _ = build_dynamic_state(
        cutoff,
        prediction_season=2024,
        expected_source_seasons=tuple(range(2019, 2024)),
    )
    expected = original["ar_k30"]
    reference = dynamic_ar_correction(validation, stored)
    module = load_rendered_module(script_path)
    packaged = module.map_dynamic_pitcher_ar(validation, stored)
    differences = {
        "serializer_reference_vs_original": float(
            np.max(np.abs(reference - expected))
        ),
        "package_helper_vs_original": float(np.max(np.abs(packaged - expected))),
        "weighted_candidate_delta_vs_original": float(
            np.max(
                np.abs(
                    ADDITIVE_WEIGHT * packaged - ADDITIVE_WEIGHT * expected
                )
            )
        ),
    }
    tolerance = 1e-12
    if max(differences.values()) > tolerance:
        raise RuntimeError(f"EXP-072 cutoff parity failed: {differences}")
    return {
        "source_cutoff": 2023,
        "prediction_season": 2024,
        "candidate": CANDIDATE,
        "rows": int(len(validation)),
        "tolerance": tolerance,
        "maximum_absolute_differences": differences,
        "original_train_exp072_dynamic_deltas_parity": "passed",
        "target_used_for_correction": False,
        "test_or_sample_file_opened": False,
    }


def validate_prospective_selection(report: dict[str, object]) -> dict[str, object]:
    prospective = dict(report["prospective_2025_selection"])
    if prospective.get("candidate") != CANDIDATE:
        raise ValueError("EXP-072 prospective 2025 selection is not ar_k30_w050")
    if bool(prospective.get("uses_2025_labels")):
        raise ValueError("EXP-072 prospective selection unexpectedly uses 2025 labels")
    selection = dict(report["selection"])
    if bool(selection.get("adopt")) or bool(selection.get("build_submission_zip")):
        raise ValueError("EXP-072 gate status unexpectedly changed")
    return {
        "prospective_2025_selection": prospective,
        "original_gate": selection,
        "candidate_validation_2022_2024": report["aggregate_2022_2024"][CANDIDATE],
    }


def main() -> None:
    started = time.time()
    frame = load_source_training_frame()
    dynamic_state, state_audit = build_dynamic_state(frame)
    validation_report = json.loads(VALIDATION_REPORT.read_text(encoding="utf-8"))
    selection_audit = validate_prospective_selection(validation_report)

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    shutil.copytree(
        SOURCE_DIR,
        DESTINATION,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    base_script_path = SOURCE_DIR / "script.py"
    rendered = render_submission_script(base_script_path.read_text(encoding="utf-8"))
    script_path = DESTINATION / "script.py"
    script_path.write_text(rendered, encoding="utf-8")

    state_path = DESTINATION / "model" / STATE_FILENAME
    write_json(state_path, dynamic_state)
    metadata_path = DESTINATION / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "experiment": EXPERIMENT,
            "candidate": CANDIDATE,
            "parent_candidate": "EXP-051 trackman_direct_recent_w010",
            "component_formula": (
                "clip(EXP-051 + 0.50 * (AR_k30_current_posterior - "
                "league_k30_current_posterior), 0, 1)"
            ),
            "probability_calibration": "identity",
            "dynamic_pitcher_state": {
                "prediction_season": PREDICTION_SEASON,
                "source_seasons": list(SOURCE_SEASONS),
                "state_file": STATE_FILENAME,
                "state_sha256": sha256_file(state_path),
                "rho": dynamic_state["rho"],
                "league_2024": dynamic_state["league_2024"],
                "current_season_prior_strength": PRIOR_STRENGTH,
                "additive_delta_weight": ADDITIVE_WEIGHT,
            },
            "prospective_selection": selection_audit[
                "prospective_2025_selection"
            ],
            "selection_status": (
                "prospective EXP-072 2025 selection; original family gate failed; "
                "submission-day exploratory candidate"
            ),
            "original_gate": selection_audit["original_gate"],
            "submission_candidate_gate_passed": False,
            "exploratory_candidate": True,
            "deployment_audit": {
                "frozen_source_state_only": True,
                "current_row_official_asof_only": True,
                "other_query_rows_required": False,
                "query_row_aggregation": False,
                "inference_time_retraining": False,
                "test_distribution_or_order_used": False,
                "2025_labels_used": False,
            },
        }
    )
    write_json(metadata_path, metadata)

    dynamic_smoke = synthetic_invariance_smoke(script_path, dynamic_state)
    package_smoke = full_package_synthetic_smoke(DESTINATION)
    parity = cutoff_2023_parity_audit(frame, script_path)
    READY_DIR.mkdir(parents=True, exist_ok=True)
    zip_result = build_zip(DESTINATION, ZIP_PATH)
    report = {
        "experiment": EXPERIMENT,
        "candidate": CANDIDATE,
        "status": "built_exploratory_gate_failed",
        "destination": str(DESTINATION),
        "selection": selection_audit,
        "state": {
            **state_audit,
            "path": str(state_path),
            "sha256": sha256_file(state_path),
            "bytes": state_path.stat().st_size,
        },
        "synthetic_smoke": {
            "dynamic_helper": dynamic_smoke,
            "full_package": package_smoke,
        },
        "cutoff_2023_original_parity": parity,
        "zip": zip_result,
        "structure": {
            "root_files": ["script.py", "requirements.txt"],
            "model_prefix_only": True,
            "contains_dynamic_state": STATE_FILENAME in {
                Path(name).name for name in zip_result["files"]
            },
            "contains_data_prediction_or_pickle": False,
        },
        "qa": {
            "source_train_2019_2024_only": True,
            "test_csv_opened": False,
            "sample_submission_opened": False,
            "test_prediction_aggregate_logged": False,
            "row_local_inference": True,
            "inference_time_retraining": False,
            "crc": zip_result["crc"],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(REPORT_PATH, report)
    print(
        f"saved={ZIP_PATH} sha256={zip_result['sha256']} crc={zip_result['crc']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
