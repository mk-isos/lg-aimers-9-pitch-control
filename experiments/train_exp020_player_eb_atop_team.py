"""EXP-020: temporal player empirical-Bayes correction atop team EB.

The immutable base is the saved ``all_prior_s1000`` team-EB OOF prediction
from EXP-019.  For every validation season, pitcher and batter effects are
estimated only from earlier evaluated OOF seasons.  Residuals are centered
inside each source season before an ID map is fitted, so the maps cannot
transfer a source-season global calibration offset.

The candidate grid is deliberately small and declared before evaluation:

* all earlier OOF seasons, smoothing 1000, pitcher/batter/50:50;
* immediately previous OOF season, stronger smoothing 2000, same families;
* all earlier OOF seasons, smoothing 1000, 50:50, but only effects whose sign
  agrees in at least two matched source seasons.
* two bounded pitcher-batter pair diagnostics: pair-only with smoothing 2000,
  and the all-prior player 50:50 correction plus 25% of that pair effect.

Every validation row uses only its own official pitcher/batter ID to map a
fixed earlier-season effect.  There is no validation/test-row aggregation.
Candidate comparison on the reported folds is diagnostic and non-nested.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics


DATA_DIR = Path("./data")
SOURCE_DIR = Path("./artifacts/EXP-019/team_eb_ensemble")
ARTIFACT_DIR = Path("./artifacts/EXP-020/player_eb_atop_team")
SOURCE_CANDIDATE = "all_prior_s1000"
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)


@dataclass(frozen=True)
class Candidate:
    name: str
    prior_window: int | None
    smoothing: float
    pitcher_weight: float
    stable_min_sources: int = 0
    player_weight: float = 1.0
    pair_smoothing: float = 0.0
    pair_weight: float = 0.0


CANDIDATES = (
    Candidate("all_prior_s1000_pitcher", None, 1000.0, 1.0),
    Candidate("all_prior_s1000_batter", None, 1000.0, 0.0),
    Candidate("all_prior_s1000_5050", None, 1000.0, 0.5),
    Candidate("prior1_s2000_pitcher", 1, 2000.0, 1.0),
    Candidate("prior1_s2000_batter", 1, 2000.0, 0.0),
    Candidate("prior1_s2000_5050", 1, 2000.0, 0.5),
    Candidate(
        "stable2_all_prior_s1000_5050",
        None,
        1000.0,
        0.5,
        stable_min_sources=2,
    ),
    Candidate(
        "all_prior_pair_s2000",
        None,
        1000.0,
        0.5,
        player_weight=0.0,
        pair_smoothing=2000.0,
        pair_weight=1.0,
    ),
    Candidate(
        "all_prior_s1000_5050_plus_pair_s2000_w025",
        None,
        1000.0,
        0.5,
        player_weight=1.0,
        pair_smoothing=2000.0,
        pair_weight=0.25,
    ),
)


def estimate_effect(
    source_rows: pd.DataFrame,
    residual: np.ndarray,
    id_column: str,
    smoothing: float,
) -> pd.Series:
    if len(source_rows) != len(residual):
        raise ValueError("source rows and residual length differ")
    values = pd.DataFrame(
        {
            id_column: source_rows[id_column].to_numpy(),
            "residual": residual,
        }
    )
    statistics = values.groupby(id_column, sort=True)["residual"].agg(
        ["sum", "count"]
    )
    return statistics["sum"] / (
        statistics["count"].astype(float) + smoothing
    )


def estimate_pair_effect(
    source_rows: pd.DataFrame,
    residual: np.ndarray,
    smoothing: float,
) -> pd.Series:
    if len(source_rows) != len(residual):
        raise ValueError("source rows and residual length differ")
    columns = ["pitcher_id", "batter_id"]
    values = source_rows.loc[:, columns].copy()
    values["residual"] = residual
    statistics = values.groupby(columns, sort=True)["residual"].agg(
        ["sum", "count"]
    )
    return statistics["sum"] / (
        statistics["count"].astype(float) + smoothing
    )


def map_source_effects(
    effects: list[pd.Series],
    ids: pd.Series,
    stable_min_sources: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not effects:
        zeros = np.zeros(len(ids), dtype=float)
        false = np.zeros(len(ids), dtype=bool)
        counts = np.zeros(len(ids), dtype=np.int8)
        return zeros, false, false, counts

    mapped_rows: list[np.ndarray] = []
    matched_rows: list[np.ndarray] = []
    for effect in effects:
        mapped = ids.map(effect)
        matched_rows.append(mapped.notna().to_numpy())
        mapped_rows.append(mapped.fillna(0.0).to_numpy(dtype=float))
    mapped_matrix = np.vstack(mapped_rows)
    matched_matrix = np.vstack(matched_rows)
    matched_count = matched_matrix.sum(axis=0).astype(np.int8)
    source_seen = matched_count > 0
    averaged = mapped_matrix.mean(axis=0)

    if stable_min_sources <= 0:
        active = source_seen
        return averaged, source_seen, active, matched_count

    positive_count = ((mapped_matrix > 0.0) & matched_matrix).sum(axis=0)
    negative_count = ((mapped_matrix < 0.0) & matched_matrix).sum(axis=0)
    sign_consistent = (matched_count >= stable_min_sources) & (
        (positive_count == matched_count) | (negative_count == matched_count)
    )
    averaged = np.where(sign_consistent, averaged, 0.0)
    return averaged, source_seen, sign_consistent, matched_count


def map_pair_source_effects(
    effects: list[pd.Series],
    rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not effects:
        return (
            np.zeros(len(rows), dtype=float),
            np.zeros(len(rows), dtype=bool),
            np.zeros(len(rows), dtype=np.int8),
        )
    keys = pd.MultiIndex.from_frame(rows[["pitcher_id", "batter_id"]])
    mapped_rows: list[np.ndarray] = []
    matched_rows: list[np.ndarray] = []
    for effect in effects:
        mapped = effect.reindex(keys)
        matched = mapped.notna().to_numpy()
        matched_rows.append(matched)
        mapped_rows.append(mapped.fillna(0.0).to_numpy(dtype=float))
    mapped_matrix = np.vstack(mapped_rows)
    matched_matrix = np.vstack(matched_rows)
    matched_count = matched_matrix.sum(axis=0).astype(np.int8)
    return (
        mapped_matrix.mean(axis=0),
        matched_count > 0,
        matched_count,
    )


def segment_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    pitcher_seen: np.ndarray,
    batter_seen: np.ndarray,
    correction: np.ndarray,
    pair_seen: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    correction_active = np.abs(correction) > 0.0
    masks = {
        "both_player_ids_seen": pitcher_seen & batter_seen,
        "either_player_id_unseen": ~(pitcher_seen & batter_seen),
        "correction_active": correction_active,
        "correction_inactive": ~correction_active,
    }
    if pair_seen is not None:
        masks["pitcher_batter_pair_seen"] = pair_seen
        masks["pitcher_batter_pair_unseen"] = ~pair_seen
    return {
        name: calculate_metrics(targets[mask], predictions[mask])
        for name, mask in masks.items()
        if mask.any()
    }


def aggregate_metrics(
    folds: dict[str, object],
    prediction_name: str,
) -> dict[str, object]:
    skills = {
        season: float(
            folds[str(season)][prediction_name]["skill_score_unclipped"]
        )
        for season in REPORT_SEASONS
    }
    briers = {
        season: float(folds[str(season)][prediction_name]["brier_score"])
        for season in REPORT_SEASONS
    }
    return {
        "season_skills": {
            str(season): value for season, value in skills.items()
        },
        "season_briers": {
            str(season): value for season, value in briers.items()
        },
        "mean_skill": float(np.mean(list(skills.values()))),
        "min_skill": float(np.min(list(skills.values()))),
        "latest_2024_skill": skills[2024],
        "uniform_1100_passed": bool(
            all(value >= 1100.0 for value in skills.values())
        ),
    }


def main() -> None:
    started_at = time.time()
    frame = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=[
            "season",
            "pitcher_id",
            "batter_id",
            "control_success",
        ],
    )
    rows_by_season: dict[int, pd.DataFrame] = {}
    base_by_season: dict[int, np.ndarray] = {}
    targets_by_season: dict[int, np.ndarray] = {}
    residual_by_season: dict[int, np.ndarray] = {}
    for season in VALIDATION_SEASONS:
        rows = frame.loc[frame["season"] == season].reset_index(drop=True)
        base = np.load(
            SOURCE_DIR / f"predictions_{SOURCE_CANDIDATE}_{season}.npy"
        ).astype(float)
        targets = np.load(SOURCE_DIR / f"targets_{season}.npy").astype(
            np.int8
        )
        current_targets = rows["control_success"].to_numpy(dtype=np.int8)
        if not (
            len(rows) == len(base) == len(targets)
            and np.array_equal(current_targets, targets)
        ):
            raise ValueError(f"OOF alignment mismatch for {season}")
        if not np.isfinite(base).all() or not (
            (base >= 0.0).all() and (base <= 1.0).all()
        ):
            raise ValueError(f"invalid base predictions for {season}")
        residual = targets.astype(float) - base
        residual -= residual.mean()
        rows_by_season[season] = rows
        base_by_season[season] = base
        targets_by_season[season] = targets
        residual_by_season[season] = residual

    effect_cache: dict[tuple[int, str, float], pd.Series] = {}

    def cached_effect(
        season: int,
        entity: str,
        smoothing: float,
    ) -> pd.Series:
        key = (season, entity, smoothing)
        if key not in effect_cache:
            if entity == "pair":
                effect_cache[key] = estimate_pair_effect(
                    rows_by_season[season],
                    residual_by_season[season],
                    smoothing,
                )
            else:
                effect_cache[key] = estimate_effect(
                    rows_by_season[season],
                    residual_by_season[season],
                    f"{entity}_id",
                    smoothing,
                )
        return effect_cache[key]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        rows = rows_by_season[validation_season]
        targets = targets_by_season[validation_season]
        base = base_by_season[validation_season]
        all_prior_seasons = [
            season
            for season in VALIDATION_SEASONS
            if season < validation_season
        ]
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "base_team_all_prior": calculate_metrics(targets, base),
            "candidates": {},
        }
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets)

        for candidate in CANDIDATES:
            source_seasons = all_prior_seasons
            if candidate.prior_window is not None:
                source_seasons = source_seasons[-candidate.prior_window :]
            pitcher_effects = [
                cached_effect(season, "pitcher", candidate.smoothing)
                for season in source_seasons
            ]
            batter_effects = [
                cached_effect(season, "batter", candidate.smoothing)
                for season in source_seasons
            ]
            (
                pitcher_correction,
                pitcher_seen,
                pitcher_active,
                pitcher_match_count,
            ) = map_source_effects(
                pitcher_effects,
                rows["pitcher_id"],
                candidate.stable_min_sources,
            )
            (
                batter_correction,
                batter_seen,
                batter_active,
                batter_match_count,
            ) = map_source_effects(
                batter_effects,
                rows["batter_id"],
                candidate.stable_min_sources,
            )
            player_correction = (
                candidate.pitcher_weight * pitcher_correction
                + (1.0 - candidate.pitcher_weight) * batter_correction
            )
            if candidate.pair_weight > 0.0:
                pair_effects = [
                    cached_effect(
                        season,
                        "pair",
                        candidate.pair_smoothing,
                    )
                    for season in source_seasons
                ]
                (
                    pair_correction,
                    pair_seen,
                    pair_match_count,
                ) = map_pair_source_effects(pair_effects, rows)
            else:
                pair_correction = np.zeros(len(rows), dtype=float)
                pair_seen = np.zeros(len(rows), dtype=bool)
                pair_match_count = np.zeros(len(rows), dtype=np.int8)
            correction = (
                candidate.player_weight * player_correction
                + candidate.pair_weight * pair_correction
            )
            predictions = np.clip(base + correction, 0.0, 1.0)
            if not np.isfinite(predictions).all() or not (
                (predictions >= 0.0).all()
                and (predictions <= 1.0).all()
            ):
                raise ValueError(
                    f"invalid {candidate.name} predictions for "
                    f"{validation_season}"
                )
            active = (
                (candidate.player_weight > 0.0)
                & (candidate.pitcher_weight > 0.0)
                & pitcher_active
            ) | (
                (candidate.player_weight > 0.0)
                & (candidate.pitcher_weight < 1.0)
                & batter_active
            ) | (
                (candidate.pair_weight > 0.0) & pair_seen
            )
            metrics = calculate_metrics(targets, predictions)
            fold["candidates"][candidate.name] = {
                "source_oof_seasons": source_seasons,
                "current_fold_labels_used_for_effect": False,
                "metrics": metrics,
                "coverage": {
                    "pitcher_seen_rows": int(pitcher_seen.sum()),
                    "batter_seen_rows": int(batter_seen.sum()),
                    "both_seen_rows": int((pitcher_seen & batter_seen).sum()),
                    "pitcher_active_rows": int(pitcher_active.sum()),
                    "batter_active_rows": int(batter_active.sum()),
                    "combined_active_rows": int(active.sum()),
                    "pair_seen_rows": int(pair_seen.sum()),
                    "pitcher_match_count_mean": float(
                        pitcher_match_count.mean()
                    ),
                    "batter_match_count_mean": float(
                        batter_match_count.mean()
                    ),
                    "pair_match_count_mean": float(
                        pair_match_count.mean()
                    ),
                },
                "correction": {
                    "mean": float(correction.mean()),
                    "mean_absolute": float(np.abs(correction).mean()),
                    "min": float(correction.min()),
                    "max": float(correction.max()),
                },
                "segments": segment_metrics(
                    targets,
                    predictions,
                    pitcher_seen,
                    batter_seen,
                    correction,
                    pair_seen=(
                        pair_seen if candidate.pair_weight > 0.0 else None
                    ),
                ),
                "base_segments_same_masks": segment_metrics(
                    targets,
                    base,
                    pitcher_seen,
                    batter_seen,
                    correction,
                    pair_seen=(
                        pair_seen if candidate.pair_weight > 0.0 else None
                    ),
                ),
            }
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate.name}_{validation_season}.npy",
                predictions,
            )
            print(
                f"player_eb {validation_season} {candidate.name}: "
                f"skill={metrics['skill_score_unclipped']:.2f} "
                f"active={int(active.sum())}"
            )
        folds[str(validation_season)] = fold

    # Flatten candidate metrics so the common aggregate helper can address
    # them without recomputing any score.
    aggregate_folds: dict[str, object] = {}
    for season in VALIDATION_SEASONS:
        source_fold = folds[str(season)]
        aggregate_folds[str(season)] = {
            "base_team_all_prior": source_fold["base_team_all_prior"],
            **{
                candidate.name: source_fold["candidates"][candidate.name][
                    "metrics"
                ]
                for candidate in CANDIDATES
            },
        }
    aggregate = {
        "base_team_all_prior": aggregate_metrics(
            aggregate_folds, "base_team_all_prior"
        ),
        **{
            candidate.name: aggregate_metrics(
                aggregate_folds, candidate.name
            )
            for candidate in CANDIDATES
        },
    }
    base_season_skills = aggregate["base_team_all_prior"][
        "season_skills"
    ]
    for candidate in CANDIDATES:
        candidate_aggregate = aggregate[candidate.name]
        skill_change = {
            str(season): float(
                candidate_aggregate["season_skills"][str(season)]
                - base_season_skills[str(season)]
            )
            for season in REPORT_SEASONS
        }
        candidate_aggregate["season_skill_change_vs_base"] = skill_change
        candidate_aggregate["mean_skill_change_vs_base"] = float(
            candidate_aggregate["mean_skill"]
            - aggregate["base_team_all_prior"]["mean_skill"]
        )
        candidate_aggregate["min_skill_change_vs_base"] = float(
            candidate_aggregate["min_skill"]
            - aggregate["base_team_all_prior"]["min_skill"]
        )
        candidate_aggregate["improved_every_reported_season"] = bool(
            all(value > 0.0 for value in skill_change.values())
        )
    candidate_names = [candidate.name for candidate in CANDIDATES]
    best_mean = max(
        candidate_names,
        key=lambda name: float(aggregate[name]["mean_skill"]),
    )
    best_min = max(
        candidate_names,
        key=lambda name: float(aggregate[name]["min_skill"]),
    )
    result = {
        "experiment": "EXP-020",
        "candidate": "temporal_player_eb_atop_team_all_prior",
        "validation_protocol": {
            "evaluated_oof_seasons": list(VALIDATION_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "immutable_base": (
                "EXP-019 team_eb_ensemble predictions_all_prior_s1000"
            ),
            "effect_training": (
                "one season-centered residual map per earlier evaluated "
                "OOF season; missing IDs contribute zero before equal "
                "source-season averaging"
            ),
            "current_fold_labels_used_for_effects": False,
            "validation_or_test_row_aggregation": False,
            "candidate_grid_predeclared": True,
            "candidate_comparison_nested": False,
            "stable_variant": (
                "at least two matched source seasons and unanimous nonzero "
                "effect sign, checked separately for pitcher and batter"
            ),
            "pair_variants": (
                "all-prior source-season-centered pitcher_id x batter_id "
                "effect, smoothing 2000, unseen pair zero; evaluated only "
                "as pair-only and player50:50 plus pair weight 0.25"
            ),
        },
        "candidate_configs": [asdict(candidate) for candidate in CANDIDATES],
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "diagnostic only; candidate comparison is non-nested",
            "posthoc_best_mean_candidate": best_mean,
            "posthoc_best_min_candidate": best_min,
            "any_candidate_improved_every_reported_season": bool(
                any(
                    aggregate[candidate.name][
                        "improved_every_reported_season"
                    ]
                    for candidate in CANDIDATES
                )
            ),
            "adopt_without_nested_confirmation": False,
        },
        "qa": {
            "source_alignment_checked": True,
            "source_and_output_probability_ranges_checked": True,
            "saved_prediction_arrays": True,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "total_seconds": float(time.time() - started_at),
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
