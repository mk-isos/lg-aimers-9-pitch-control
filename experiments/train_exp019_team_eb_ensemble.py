"""EXP-019: temporal team empirical-Bayes correction on a fixed OOF ensemble.

The immutable prediction base is a 50/50 average of the saved rolling OOF
predictions from:

* r_full LightGBM ``rfull_l63_m1000_i300 / branch_w075``
* HistGradientBoosting ``hist_l15_d4_m3000_i160 / branch_w100``

For each validation season, team effects use only earlier *evaluated OOF*
seasons.  The OOF residual is centered inside its source season so a team map
cannot act as a transferred global calibration offset.  Effects are estimated
separately inside each earlier season for
``pitcher_team_id x pitcher_hand x batter_hand`` and
``batter_team_id x pitcher_hand x batter_hand``.  Each seasonal effect is
shrunk toward zero.  Two candidates are declared explicitly: all earlier OOF
seasons with smoothing 1000, and the immediately prior OOF season with
smoothing 500.  Mapped effects are averaged equally across the candidate's
source seasons; missing keys contribute zero.  The two team families are then
averaged 50/50.  Candidate comparison is diagnostic and not nested.

No label from the current validation fold and no aggregation over validation
or test rows is used.  Team and hand columns are official values from the
current row, while all effect maps are fixed from earlier train OOF rows.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_residual_features import (
    TARGET,
    attach_training_temporal_features,
)
from train_exp017_rolling_residual import calculate_metrics, segment_metrics


DATA_DIR = Path("./data")
ARTIFACT_DIR = Path("./artifacts/EXP-019/team_eb_ensemble")
LGB_ROOT = Path(
    "./artifacts/EXP-019/r_full_residual/rfull_l63_m1000_i300"
)
HGB_ROOT = Path(
    "./artifacts/EXP-019/histgb_residual/hist_l15_d4_m3000_i160"
)
LGB_VARIANT = "branch_w075"
HGB_VARIANT = "branch_w100"
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
OOF_ENSEMBLE_WEIGHTS = {"lightgbm": 0.50, "histgradientboosting": 0.50}
TEAM_FAMILY_WEIGHTS = {"pitcher_team": 0.50, "batter_team": 0.50}
PITCHER_TEAM_COLUMNS = (
    "pitcher_team_id",
    "pitcher_hand",
    "batter_hand",
)
BATTER_TEAM_COLUMNS = (
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
)


@dataclass(frozen=True)
class TeamCandidate:
    name: str
    prior_window: int | None
    smoothing: float


CANDIDATES = (
    TeamCandidate(
        name="all_prior_s1000",
        prior_window=None,
        smoothing=1000.0,
    ),
    TeamCandidate(
        name="prior1_s500",
        prior_window=1,
        smoothing=500.0,
    ),
)


def load_temporal_frame() -> pd.DataFrame:
    columns = [
        "season",
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "batter_team_id",
        "pitcher_hand",
        "batter_hand",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        TARGET,
    ]
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=columns,
    )
    train, _ = attach_training_temporal_features(train, target=TARGET)
    return train


def prediction_path(
    root: Path,
    variant: str,
    season: int,
) -> Path:
    return root / f"predictions_{variant}_{season}.npy"


def target_path(root: Path, season: int) -> Path:
    return root / f"targets_{season}.npy"


def load_fixed_oof(
    frame: pd.DataFrame,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    ensemble: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    frame_indices: dict[int, np.ndarray] = {}
    frame_seasons = frame["season"].to_numpy(dtype=np.int16)
    frame_targets = frame[TARGET].to_numpy(dtype=np.int8)
    for season in VALIDATION_SEASONS:
        indices = np.flatnonzero(frame_seasons == season)
        lgb_prediction = np.load(
            prediction_path(LGB_ROOT, LGB_VARIANT, season)
        ).astype(float)
        hgb_prediction = np.load(
            prediction_path(HGB_ROOT, HGB_VARIANT, season)
        ).astype(float)
        lgb_target = np.load(target_path(LGB_ROOT, season)).astype(np.int8)
        hgb_target = np.load(target_path(HGB_ROOT, season)).astype(np.int8)
        current_target = frame_targets[indices]
        if not (
            len(indices) == len(lgb_prediction) == len(hgb_prediction)
            and np.array_equal(current_target, lgb_target)
            and np.array_equal(current_target, hgb_target)
        ):
            raise ValueError(f"OOF alignment mismatch for season {season}")
        combined = (
            OOF_ENSEMBLE_WEIGHTS["lightgbm"] * lgb_prediction
            + OOF_ENSEMBLE_WEIGHTS["histgradientboosting"]
            * hgb_prediction
        )
        if not np.isfinite(combined).all():
            raise ValueError(f"non-finite OOF prediction for season {season}")
        ensemble[season] = np.clip(combined, 0.0, 1.0)
        targets[season] = current_target.astype(float)
        frame_indices[season] = indices
    return ensemble, targets, frame_indices


def estimate_season_effect(
    source_rows: pd.DataFrame,
    residual: np.ndarray,
    columns: tuple[str, ...],
    smoothing: float,
) -> pd.Series:
    if len(source_rows) != len(residual):
        raise ValueError("source rows and residual length differ")
    grouped = source_rows.loc[:, list(columns)].copy()
    grouped["residual"] = residual
    statistics = grouped.groupby(list(columns), sort=True)["residual"].agg(
        ["sum", "count"]
    )
    return statistics["sum"] / (
        statistics["count"].astype(float) + smoothing
    )


def map_effect(
    effects: pd.Series,
    validation_rows: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    keys = pd.MultiIndex.from_frame(validation_rows.loc[:, list(columns)])
    mapped = effects.reindex(keys)
    matched = mapped.notna().to_numpy()
    return mapped.fillna(0.0).to_numpy(dtype=float), matched


def calculate_team_segment_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    pitcher_matched: np.ndarray,
    batter_matched: np.ndarray,
) -> dict[str, dict[str, float]]:
    masks = {
        "pitcher_team_group_seen": pitcher_matched,
        "pitcher_team_group_unseen": ~pitcher_matched,
        "batter_team_group_seen": batter_matched,
        "batter_team_group_unseen": ~batter_matched,
        "both_team_groups_seen": pitcher_matched & batter_matched,
        "either_team_group_unseen": ~(pitcher_matched & batter_matched),
    }
    return {
        name: calculate_metrics(targets[mask], predictions[mask])
        for name, mask in masks.items()
        if mask.any()
    }


def main() -> None:
    started = time.time()
    frame = load_temporal_frame()
    ensemble, targets_by_season, indices_by_season = load_fixed_oof(frame)
    diagnostics = frame[
        [
            "season",
            "temporal_pitcher_season_n",
            "temporal_pitcher_prior_exists",
            "temporal_batter_prior_exists",
        ]
    ].copy()
    seasons = frame["season"].to_numpy(dtype=np.int16)
    effect_cache: dict[str, dict[str, dict[int, pd.Series]]] = {
        candidate.name: {"pitcher_team": {}, "batter_team": {}}
        for candidate in CANDIDATES
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        all_prior_oof_seasons = [
            season
            for season in VALIDATION_SEASONS
            if season < validation_season
        ]
        indices = indices_by_season[validation_season]
        validation_rows = frame.iloc[indices]
        targets = targets_by_season[validation_season]
        base_predictions = ensemble[validation_season]
        validation_mask = seasons == validation_season
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "current_fold_labels_used_for_effects": False,
            "base_50_50_ensemble": calculate_metrics(
                targets,
                base_predictions,
            ),
            "candidates": {},
        }
        if validation_season == 2022:
            fold["nested_caveat"] = (
                "Only 2021 OOF is available for the 2022 team effect; this "
                "single-source estimate cannot demonstrate cross-season "
                "persistence. Candidate-level model selection also remains "
                "diagnostic rather than fully nested."
            )
        np.save(
            ARTIFACT_DIR / f"base_ensemble_predictions_{validation_season}.npy",
            base_predictions,
        )
        np.save(
            ARTIFACT_DIR / f"targets_{validation_season}.npy",
            targets.astype(np.int8),
        )
        for candidate in CANDIDATES:
            prior_oof_seasons = all_prior_oof_seasons
            if candidate.prior_window is not None:
                prior_oof_seasons = prior_oof_seasons[
                    -candidate.prior_window :
                ]
            pitcher_effects: list[np.ndarray] = []
            batter_effects: list[np.ndarray] = []
            pitcher_matches: list[np.ndarray] = []
            batter_matches: list[np.ndarray] = []
            source_details: dict[str, object] = {}
            candidate_cache = effect_cache[candidate.name]
            for source_season in prior_oof_seasons:
                if source_season not in candidate_cache["pitcher_team"]:
                    source_indices = indices_by_season[source_season]
                    source_rows = frame.iloc[source_indices]
                    raw_source_residual = (
                        targets_by_season[source_season]
                        - ensemble[source_season]
                    )
                    source_residual_mean = float(raw_source_residual.mean())
                    source_residual = (
                        raw_source_residual - source_residual_mean
                    )
                    candidate_cache["pitcher_team"][source_season] = (
                        estimate_season_effect(
                            source_rows,
                            source_residual,
                            PITCHER_TEAM_COLUMNS,
                            candidate.smoothing,
                        )
                    )
                    candidate_cache["batter_team"][source_season] = (
                        estimate_season_effect(
                            source_rows,
                            source_residual,
                            BATTER_TEAM_COLUMNS,
                            candidate.smoothing,
                        )
                    )
                pitcher_effect, pitcher_matched = map_effect(
                    candidate_cache["pitcher_team"][source_season],
                    validation_rows,
                    PITCHER_TEAM_COLUMNS,
                )
                batter_effect, batter_matched = map_effect(
                    candidate_cache["batter_team"][source_season],
                    validation_rows,
                    BATTER_TEAM_COLUMNS,
                )
                pitcher_effects.append(pitcher_effect)
                batter_effects.append(batter_effect)
                pitcher_matches.append(pitcher_matched)
                batter_matches.append(batter_matched)
                raw_residual = (
                    targets_by_season[source_season]
                    - ensemble[source_season]
                )
                source_details[str(source_season)] = {
                    "raw_residual_mean_subtracted": float(
                        raw_residual.mean()
                    ),
                    "pitcher_team_groups": int(
                        len(candidate_cache["pitcher_team"][source_season])
                    ),
                    "batter_team_groups": int(
                        len(candidate_cache["batter_team"][source_season])
                    ),
                    "pitcher_match_rows": int(pitcher_matched.sum()),
                    "batter_match_rows": int(batter_matched.sum()),
                    "pitcher_effect_mean": float(pitcher_effect.mean()),
                    "batter_effect_mean": float(batter_effect.mean()),
                }

            if prior_oof_seasons:
                pitcher_average = np.mean(pitcher_effects, axis=0)
                batter_average = np.mean(batter_effects, axis=0)
                pitcher_matched_any = np.logical_or.reduce(pitcher_matches)
                batter_matched_any = np.logical_or.reduce(batter_matches)
            else:
                pitcher_average = np.zeros(len(targets), dtype=float)
                batter_average = np.zeros(len(targets), dtype=float)
                pitcher_matched_any = np.zeros(len(targets), dtype=bool)
                batter_matched_any = np.zeros(len(targets), dtype=bool)
            correction = (
                TEAM_FAMILY_WEIGHTS["pitcher_team"] * pitcher_average
                + TEAM_FAMILY_WEIGHTS["batter_team"] * batter_average
            )
            predictions = np.clip(
                base_predictions + correction,
                0.0,
                1.0,
            )
            candidate_fold = {
                "prior_oof_seasons": prior_oof_seasons,
                "source_details": source_details,
                "correction": {
                    "mean": float(correction.mean()),
                    "std": float(correction.std()),
                    "min": float(correction.min()),
                    "max": float(correction.max()),
                    "nonzero_rows": int(np.count_nonzero(correction)),
                },
                "team_eb": calculate_metrics(targets, predictions),
                "segments_team_eb": segment_metrics(
                    diagnostics,
                    validation_mask,
                    targets,
                    predictions,
                ),
                "team_coverage_segments": calculate_team_segment_metrics(
                    targets,
                    predictions,
                    pitcher_matched_any,
                    batter_matched_any,
                ),
            }
            fold["candidates"][candidate.name] = candidate_fold
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate.name}_{validation_season}.npy",
                predictions,
            )
            if candidate.name == "all_prior_s1000":
                np.save(
                    ARTIFACT_DIR / f"predictions_{validation_season}.npy",
                    predictions,
                )
            base_skill = fold["base_50_50_ensemble"][
                "skill_score_unclipped"
            ]
            team_skill = candidate_fold["team_eb"][
                "skill_score_unclipped"
            ]
            print(
                f"team_eb {candidate.name} {validation_season}: "
                f"base={base_skill:.2f} team={team_skill:.2f} "
                f"delta={team_skill - base_skill:+.2f} "
                f"sources={prior_oof_seasons}"
            )
        folds[str(validation_season)] = fold

    base_skills = {
        season: folds[str(season)]["base_50_50_ensemble"][
            "skill_score_unclipped"
        ]
        for season in REPORT_SEASONS
    }
    aggregate: dict[str, object] = {}
    for candidate in CANDIDATES:
        team_skills = {
            season: folds[str(season)]["candidates"][candidate.name][
                "team_eb"
            ]["skill_score_unclipped"]
            for season in REPORT_SEASONS
        }
        aggregate[candidate.name] = {
            "base_mean_skill": float(np.mean(list(base_skills.values()))),
            "base_min_skill": float(np.min(list(base_skills.values()))),
            "team_eb_mean_skill": float(
                np.mean(list(team_skills.values()))
            ),
            "team_eb_min_skill": float(np.min(list(team_skills.values()))),
            "mean_skill_change": float(
                np.mean(list(team_skills.values()))
                - np.mean(list(base_skills.values()))
            ),
            "min_skill_change": float(
                np.min(list(team_skills.values()))
                - np.min(list(base_skills.values()))
            ),
            "improved_every_reported_season": bool(
                all(
                    team_skills[season] > base_skills[season]
                    for season in REPORT_SEASONS
                )
            ),
        }
    result = {
        "experiment": "EXP-019",
        "candidate": "fixed_lgb_hgb_50_50_plus_centered_team_eb",
        "validation_protocol": {
            "evaluated_oof_seasons": list(VALIDATION_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "base_predictions": {
                "lightgbm": {
                    "path": str(LGB_ROOT),
                    "variant": LGB_VARIANT,
                    "weight": OOF_ENSEMBLE_WEIGHTS["lightgbm"],
                },
                "histgradientboosting": {
                    "path": str(HGB_ROOT),
                    "variant": HGB_VARIANT,
                    "weight": OOF_ENSEMBLE_WEIGHTS[
                        "histgradientboosting"
                    ],
                },
            },
            "effect_training": (
                "source-season mean-centered residual; one independent "
                "smoothed map per eligible earlier evaluated OOF season; "
                "equal average including zero for missing keys"
            ),
            "current_fold_labels_used_for_effects": False,
            "test_row_aggregation": False,
            "candidate_comparison_nested": False,
            "nested_caveat_2022": (
                "2022 has only the 2021 OOF season available for team EB; "
                "candidate-level base-model selection is not fully nested"
            ),
        },
        "team_effect": {
            "pitcher_columns": list(PITCHER_TEAM_COLUMNS),
            "batter_columns": list(BATTER_TEAM_COLUMNS),
            "residual_centering": "subtract each source OOF season mean",
            "family_weights": TEAM_FAMILY_WEIGHTS,
            "current_row_keys_only": True,
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "candidate comparison is non-nested",
            "parameters_predeclared_in_this_comparison": True,
            "nested_temporal_confirmation_required": True,
            "adoption_rule": (
                "require improvement across 2022, 2023, and 2024 without "
                "materially reducing the robust minimum"
            ),
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
    print(f"saved={ARTIFACT_DIR / 'validation_metrics.json'}")


if __name__ == "__main__":
    main()
