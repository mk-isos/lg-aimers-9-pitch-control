"""EXP-020: pitcher-count empirical-Bayes correction atop team OOF.

The immutable primary base is the saved ``all_prior_s1000`` team OOF
prediction.  For each validation season, every earlier evaluated OOF season
produces an independent pitcher_id x count_index x batter_hand effect from its
season-centered residual.  Effects use fixed smoothing 600 and are averaged
equally across source seasons; missing source-season keys contribute zero.

Three predeclared compositions are reported:

* team base plus the correction on all rows;
* team base plus the correction only on current-row regular-season rows;
* a regime-gated base (fixed ensemble for F, team base for R) plus the same
  all-row correction.

No current-fold labels, validation/test-row aggregation, or post-result
parameter tuning is used.  Candidate ranking is explicitly post-hoc.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_residual_features import (
    TARGET,
    attach_training_temporal_features,
)
from train_exp017_rolling_residual import calculate_metrics, segment_metrics


DATA_DIR = Path("./data")
TEAM_ROOT = Path("./artifacts/EXP-019/team_eb_ensemble")
ARTIFACT_DIR = Path("./artifacts/EXP-020/pitcher_count_eb_atop_team")
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
SMOOTHING = 600.0
GROUP_COLUMNS = ("pitcher_id", "count_index", "batter_hand")
CANDIDATES = (
    "team_pc_all",
    "team_pc_r_only",
    "r_gated_team_pc_all",
)


def load_frame() -> pd.DataFrame:
    columns = [
        "season",
        "game_type",
        "pitcher_id",
        "batter_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        TARGET,
    ]
    frame = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=columns,
    )
    frame, _ = attach_training_temporal_features(frame, target=TARGET)
    frame["count_index"] = (
        frame["balls_before"].to_numpy(dtype=np.int8) * 4
        + frame["strikes_before"].to_numpy(dtype=np.int8)
    ).astype(np.int8)
    return frame


def load_oof(
    frame: pd.DataFrame,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    fixed: dict[int, np.ndarray] = {}
    team: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    indices: dict[int, np.ndarray] = {}
    seasons = frame["season"].to_numpy(dtype=np.int16)
    raw_targets = frame[TARGET].to_numpy(dtype=np.int8)
    for season in VALIDATION_SEASONS:
        season_indices = np.flatnonzero(seasons == season)
        current_targets = raw_targets[season_indices]
        fixed_predictions = np.load(
            TEAM_ROOT / f"base_ensemble_predictions_{season}.npy"
        ).astype(float)
        team_predictions = np.load(
            TEAM_ROOT / f"predictions_all_prior_s1000_{season}.npy"
        ).astype(float)
        saved_targets = np.load(
            TEAM_ROOT / f"targets_{season}.npy"
        ).astype(np.int8)
        if not (
            len(season_indices)
            == len(fixed_predictions)
            == len(team_predictions)
            == len(saved_targets)
            and np.array_equal(current_targets, saved_targets)
        ):
            raise ValueError(f"OOF alignment mismatch for {season}")
        if not (
            np.isfinite(fixed_predictions).all()
            and np.isfinite(team_predictions).all()
        ):
            raise ValueError(f"non-finite OOF base for {season}")
        fixed[season] = np.clip(fixed_predictions, 0.0, 1.0)
        team[season] = np.clip(team_predictions, 0.0, 1.0)
        targets[season] = current_targets.astype(float)
        indices[season] = season_indices
    return fixed, team, targets, indices


def estimate_effect(
    source_rows: pd.DataFrame,
    centered_residual: np.ndarray,
) -> pd.Series:
    grouped = source_rows.loc[:, list(GROUP_COLUMNS)].copy()
    grouped["residual"] = centered_residual
    statistics = grouped.groupby(list(GROUP_COLUMNS), sort=True)[
        "residual"
    ].agg(["sum", "count"])
    return statistics["sum"] / (
        statistics["count"].astype(float) + SMOOTHING
    )


def map_effect(
    effects: pd.Series,
    validation_rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    keys = pd.MultiIndex.from_frame(
        validation_rows.loc[:, list(GROUP_COLUMNS)]
    )
    mapped = effects.reindex(keys)
    return (
        mapped.fillna(0.0).to_numpy(dtype=float),
        mapped.notna().to_numpy(dtype=bool),
    )


def regime_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    game_types: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        game_type: calculate_metrics(
            targets[game_types == game_type],
            predictions[game_types == game_type],
        )
        for game_type in sorted(np.unique(game_types))
    }


def coverage_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    matched_any: np.ndarray,
    matched_all: np.ndarray,
) -> dict[str, dict[str, float]]:
    masks = {
        "matched_any_source": matched_any,
        "matched_every_source": matched_all,
        "never_matched": ~matched_any,
    }
    return {
        name: calculate_metrics(targets[mask], predictions[mask])
        for name, mask in masks.items()
        if mask.any()
    }


def aggregate_folds(folds: dict[str, object]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for candidate in CANDIDATES:
        skills = {
            season: float(
                folds[str(season)]["candidates"][candidate]["metrics"][
                    "skill_score_unclipped"
                ]
            )
            for season in REPORT_SEASONS
        }
        briers = {
            season: float(
                folds[str(season)]["candidates"][candidate]["metrics"][
                    "brier_score"
                ]
            )
            for season in REPORT_SEASONS
        }
        aggregate[candidate] = {
            "season_skills": {
                str(season): value for season, value in skills.items()
            },
            "season_briers": {
                str(season): value for season, value in briers.items()
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
        }
    return aggregate


def main() -> None:
    started = time.time()
    frame = load_frame()
    fixed, team, targets_by_season, indices_by_season = load_oof(frame)
    seasons = frame["season"].to_numpy(dtype=np.int16)
    diagnostics = frame[
        [
            "season",
            "temporal_pitcher_season_n",
            "temporal_pitcher_prior_exists",
            "temporal_batter_prior_exists",
        ]
    ].copy()
    effect_cache: dict[int, pd.Series] = {}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        validation_indices = indices_by_season[validation_season]
        validation_rows = frame.iloc[validation_indices]
        validation_mask = seasons == validation_season
        targets = targets_by_season[validation_season]
        fixed_base = fixed[validation_season]
        team_base = team[validation_season]
        game_types = validation_rows["game_type"].astype(str).to_numpy()
        is_r = game_types == "R"
        source_seasons = [
            season
            for season in VALIDATION_SEASONS
            if season < validation_season
        ]
        mapped_effects: list[np.ndarray] = []
        matched_sources: list[np.ndarray] = []
        source_details: dict[str, object] = {}
        for source_season in source_seasons:
            source_indices = indices_by_season[source_season]
            source_rows = frame.iloc[source_indices]
            raw_residual = (
                targets_by_season[source_season] - team[source_season]
            )
            residual_mean = float(raw_residual.mean())
            centered = raw_residual - residual_mean
            if source_season not in effect_cache:
                effect_cache[source_season] = estimate_effect(
                    source_rows,
                    centered,
                )
            effect, matched = map_effect(
                effect_cache[source_season],
                validation_rows,
            )
            mapped_effects.append(effect)
            matched_sources.append(matched)
            source_details[str(source_season)] = {
                "raw_residual_mean_subtracted": residual_mean,
                "groups": int(len(effect_cache[source_season])),
                "matched_rows": int(matched.sum()),
                "match_rate": float(matched.mean()),
                "mapped_effect_mean": float(effect.mean()),
                "mapped_effect_std": float(effect.std()),
            }

        if source_seasons:
            correction = np.mean(mapped_effects, axis=0)
            source_match_count = np.sum(matched_sources, axis=0).astype(
                np.int8
            )
            matched_any = source_match_count > 0
            matched_all = source_match_count == len(source_seasons)
        else:
            correction = np.zeros(len(targets), dtype=float)
            source_match_count = np.zeros(len(targets), dtype=np.int8)
            matched_any = np.zeros(len(targets), dtype=bool)
            matched_all = np.zeros(len(targets), dtype=bool)

        r_correction = np.where(is_r, correction, 0.0)
        gated_base = np.where(is_r, team_base, fixed_base)
        candidate_predictions = {
            "team_pc_all": np.clip(
                team_base + correction,
                0.0,
                1.0,
            ),
            "team_pc_r_only": np.clip(
                team_base + r_correction,
                0.0,
                1.0,
            ),
            "r_gated_team_pc_all": np.clip(
                gated_base + correction,
                0.0,
                1.0,
            ),
        }
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "current_fold_labels_used_for_effect": False,
            "bases": {
                "fixed_50_50": calculate_metrics(targets, fixed_base),
                "team_all_prior_s1000": calculate_metrics(
                    targets,
                    team_base,
                ),
                "r_gated_fixed_F_team_R": calculate_metrics(
                    targets,
                    gated_base,
                ),
            },
            "source_details": source_details,
            "correction": {
                "smoothing": SMOOTHING,
                "source_season_weighting": "equal; missing key contributes zero",
                "mean": float(correction.mean()),
                "std": float(correction.std()),
                "min": float(correction.min()),
                "max": float(correction.max()),
                "nonzero_rows": int(np.count_nonzero(correction)),
                "matched_any_rows": int(matched_any.sum()),
                "matched_any_rate": float(matched_any.mean()),
                "matched_every_source_rows": int(matched_all.sum()),
                "matched_every_source_rate": float(matched_all.mean()),
                "mean_matched_source_count": float(
                    source_match_count.mean()
                ),
            },
            "candidates": {},
        }
        for candidate, predictions in candidate_predictions.items():
            fold["candidates"][candidate] = {
                "metrics": calculate_metrics(targets, predictions),
                "regimes": regime_metrics(
                    targets,
                    predictions,
                    game_types,
                ),
                "segments": segment_metrics(
                    diagnostics,
                    validation_mask,
                    targets,
                    predictions,
                ),
                "coverage_segments": coverage_metrics(
                    targets,
                    predictions,
                    matched_any,
                    matched_all,
                ),
            }
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate}_{validation_season}.npy",
                predictions,
            )
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets)
        np.save(
            ARTIFACT_DIR / f"correction_{validation_season}.npy",
            correction,
        )
        np.save(
            ARTIFACT_DIR / f"matched_source_count_{validation_season}.npy",
            source_match_count,
        )
        folds[str(validation_season)] = fold
        print(
            f"pitcher_count_eb {validation_season} coverage="
            f"{fold['correction']['matched_any_rate']:.3f}: "
            + " ".join(
                f"{candidate}="
                f"{fold['candidates'][candidate]['metrics']['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    aggregate = aggregate_folds(folds)
    posthoc_best = max(
        aggregate,
        key=lambda candidate: (
            float(aggregate[candidate]["min_skill"]),
            float(aggregate[candidate]["latest_2024_skill"]),
            float(aggregate[candidate]["mean_skill"]),
        ),
    )
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": "pitcher_count_eb_atop_team",
        "validation_protocol": {
            "outer_folds": list(VALIDATION_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "primary_base": "saved team all_prior_s1000 OOF",
            "effect_target": (
                "source-season centered y minus team all_prior_s1000 OOF"
            ),
            "effect_keys": list(GROUP_COLUMNS),
            "source_season_weighting": "equal",
            "missing_source_key": "zero contribution",
            "current_fold_labels_used_for_effect": False,
            "test_row_aggregation": False,
            "candidate_selection": "post-hoc diagnostic ranking only",
        },
        "predeclared": {
            "smoothing": SMOOTHING,
            "candidates": {
                "team_pc_all": "team base; correction applied all rows",
                "team_pc_r_only": "team base; correction applied R rows only",
                "r_gated_team_pc_all": (
                    "fixed base for F and team base for R; correction applied all rows"
                ),
            },
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "post-hoc diagnostic; not a nested selection",
            "posthoc_best_candidate": posthoc_best,
            "posthoc_best_min_skill": aggregate[posthoc_best]["min_skill"],
            "posthoc_best_latest_2024_skill": aggregate[posthoc_best][
                "latest_2024_skill"
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "total_seconds": time.time() - started,
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f"selection={result['selection']}", flush=True)
    print(f"saved={ARTIFACT_DIR / 'validation_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
