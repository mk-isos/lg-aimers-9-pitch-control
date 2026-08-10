"""EXP-020: temporal context EB corrections on the team all-prior OOF base.

For each evaluated validation season, every empirical-Bayes map is fitted on
residuals from earlier evaluated OOF seasons only.  Residuals are centered
within their source season, maps are fitted independently by source season,
and target-season corrections are the equal-weight mean of all eligible source
maps, including zero for a missing key.  No current-fold label or test row is
used for aggregation.

The candidate family and smoothing strengths are fixed before evaluation.
Candidate comparisons are nevertheless post-hoc at the experiment-family
level, because the team all-prior base and this family were developed after
inspecting earlier OOF experiments.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics


DATA_PATH = Path("./data/train.csv")
BASE_ROOT = Path("./artifacts/EXP-019/team_eb_ensemble")
BASE_METRICS_PATH = BASE_ROOT / "validation_metrics.json"
ARTIFACT_DIR = Path("./artifacts/EXP-020/context_eb_team")

EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
SOURCE_EFFECT_CLIP = 0.02


@dataclass(frozen=True)
class Family:
    name: str
    columns: tuple[str, ...]
    smoothing: float


FAMILIES = (
    Family(
        "count_hands",
        ("count_index", "pitcher_hand", "batter_hand"),
        1000.0,
    ),
    Family(
        "game_type_count_hands",
        ("game_type", "count_index", "pitcher_hand", "batter_hand"),
        1000.0,
    ),
    Family(
        "base_state_count_hands",
        ("base_state", "count_index", "pitcher_hand", "batter_hand"),
        1000.0,
    ),
    Family(
        "outs_count_hands",
        ("outs_before", "count_index", "pitcher_hand", "batter_hand"),
        1000.0,
    ),
    Family(
        "inning_bucket_count_hands",
        (
            "inning_bucket",
            "count_index",
            "pitcher_hand",
            "batter_hand",
        ),
        1000.0,
    ),
    Family(
        "pitcher_team_count_hands",
        (
            "pitcher_team_id",
            "count_index",
            "pitcher_hand",
            "batter_hand",
        ),
        3000.0,
    ),
    Family(
        "batter_team_count_hands",
        (
            "batter_team_id",
            "count_index",
            "pitcher_hand",
            "batter_hand",
        ),
        3000.0,
    ),
)
FAMILY_BY_NAME = {family.name: family for family in FAMILIES}

# One small, fixed equal-weight composite.  Team families are intentionally
# excluded because the base already contains pitcher-team and batter-team EB.
COMPOSITES = {
    "situational_equal3": (
        "game_type_count_hands",
        "base_state_count_hands",
        "inning_bucket_count_hands",
    )
}
CANDIDATE_ORDER = (
    "base_identity",
    *(family.name for family in FAMILIES),
    *COMPOSITES,
)


def prepare_context_rows() -> dict[int, pd.DataFrame]:
    columns = [
        "season",
        "balls_before",
        "strikes_before",
        "game_type",
        "base_state",
        "outs_before",
        "inning",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team_id",
        "batter_team_id",
        "control_success",
    ]
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=columns)
    frame = frame.loc[frame["season"].isin(EVALUATED_SEASONS)].copy()
    frame["count_index"] = (
        frame["balls_before"].to_numpy(dtype=np.int8) * 4
        + frame["strikes_before"].to_numpy(dtype=np.int8)
    ).astype(np.int8)
    innings = frame["inning"].to_numpy(dtype=np.int16)
    frame["inning_bucket"] = np.where(
        innings <= 3,
        0,
        np.where(innings <= 6, 1, 2),
    ).astype(np.int8)
    return {
        season: frame.loc[frame["season"].eq(season)].reset_index(drop=True)
        for season in EVALUATED_SEASONS
    }


def load_base_oof(
    rows: dict[int, pd.DataFrame],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    targets: dict[int, np.ndarray] = {}
    predictions: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        targets[season] = np.load(BASE_ROOT / f"targets_{season}.npy").astype(
            np.float64
        )
        predictions[season] = np.load(
            BASE_ROOT / f"predictions_all_prior_s1000_{season}.npy"
        ).astype(np.float64)
        csv_targets = rows[season]["control_success"].to_numpy(dtype=np.float64)
        if not np.array_equal(targets[season], csv_targets):
            raise ValueError(f"target/order mismatch for season {season}")
        if len(predictions[season]) != len(targets[season]):
            raise ValueError(f"prediction length mismatch for season {season}")
        if not np.all(np.isfinite(predictions[season])):
            raise ValueError(f"non-finite base predictions for season {season}")
    return targets, predictions


def fit_source_effect(
    keys: pd.DataFrame,
    residual: np.ndarray,
    family: Family,
) -> tuple[pd.Series, dict[str, object]]:
    work = keys.loc[:, list(family.columns)].copy()
    work["residual"] = residual
    stats = work.groupby(list(family.columns), sort=False, dropna=False)[
        "residual"
    ].agg(["sum", "count"])
    effect = stats["sum"] / (stats["count"] + family.smoothing)
    effect = effect.clip(-SOURCE_EFFECT_CLIP, SOURCE_EFFECT_CLIP)
    return effect, {
        "source_rows": int(len(work)),
        "groups": int(len(effect)),
        "max_abs_effect": float(np.max(np.abs(effect.to_numpy(dtype=float)))),
        "mean_abs_effect": float(np.mean(np.abs(effect.to_numpy(dtype=float)))),
        "largest_group_rows": int(stats["count"].max()),
    }


def apply_source_effect(
    target_keys: pd.DataFrame,
    family: Family,
    effect: pd.Series,
) -> tuple[np.ndarray, float]:
    if len(family.columns) == 1:
        index: pd.Index | pd.MultiIndex = pd.Index(
            target_keys[family.columns[0]],
            name=family.columns[0],
        )
    else:
        index = pd.MultiIndex.from_frame(
            target_keys.loc[:, list(family.columns)]
        )
    matched = effect.reindex(index)
    coverage = float(matched.notna().mean())
    return matched.fillna(0.0).to_numpy(dtype=np.float64), coverage


def build_family_corrections(
    rows: dict[int, pd.DataFrame],
    targets: dict[int, np.ndarray],
    base_predictions: dict[int, np.ndarray],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object]]:
    centered_residuals: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        residual = targets[season] - base_predictions[season]
        centered_residuals[season] = residual - float(np.mean(residual))

    corrections: dict[int, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, object] = {}
    for validation_season in EVALUATED_SEASONS:
        source_seasons = [
            season for season in EVALUATED_SEASONS if season < validation_season
        ]
        corrections[validation_season] = {}
        validation_diagnostics: dict[str, object] = {
            "source_oof_seasons": source_seasons,
            "families": {},
        }
        for family in FAMILIES:
            source_values: list[np.ndarray] = []
            source_details: dict[str, object] = {}
            for source_season in source_seasons:
                effect, fit_details = fit_source_effect(
                    rows[source_season],
                    centered_residuals[source_season],
                    family,
                )
                values, coverage = apply_source_effect(
                    rows[validation_season],
                    family,
                    effect,
                )
                source_values.append(values)
                source_details[str(source_season)] = {
                    **fit_details,
                    "target_row_key_coverage": coverage,
                    "applied_mean": float(np.mean(values)),
                    "applied_std": float(np.std(values)),
                }
            if source_values:
                correction = np.mean(np.vstack(source_values), axis=0)
            else:
                correction = np.zeros(len(rows[validation_season]), dtype=float)
            correction = np.clip(
                correction,
                -SOURCE_EFFECT_CLIP,
                SOURCE_EFFECT_CLIP,
            )
            corrections[validation_season][family.name] = correction
            validation_diagnostics["families"][family.name] = {
                "smoothing": family.smoothing,
                "columns": list(family.columns),
                "source_details": source_details,
                "correction_mean": float(np.mean(correction)),
                "correction_std": float(np.std(correction)),
                "correction_min": float(np.min(correction)),
                "correction_max": float(np.max(correction)),
            }
        diagnostics[str(validation_season)] = validation_diagnostics
    return corrections, diagnostics


def candidate_predictions(
    season: int,
    base: np.ndarray,
    corrections: dict[int, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    outputs = {"base_identity": base.copy()}
    for family in FAMILIES:
        outputs[family.name] = np.clip(
            base + corrections[season][family.name],
            0.0,
            1.0,
        )
    for name, members in COMPOSITES.items():
        composite = np.mean(
            np.vstack([corrections[season][member] for member in members]),
            axis=0,
        )
        outputs[name] = np.clip(base + composite, 0.0, 1.0)
    if tuple(outputs) != CANDIDATE_ORDER:
        raise AssertionError("candidate order drift")
    return outputs


def summarize_candidates(folds: dict[str, object]) -> dict[str, object]:
    summaries: dict[str, object] = {}
    base_skills = {
        season: float(
            folds[str(season)]["candidates"]["base_identity"][
                "skill_score_unclipped"
            ]
        )
        for season in REPORT_SEASONS
    }
    base_briers = {
        season: float(
            folds[str(season)]["candidates"]["base_identity"]["brier_score"]
        )
        for season in REPORT_SEASONS
    }
    for candidate in CANDIDATE_ORDER:
        skills = {
            season: float(
                folds[str(season)]["candidates"][candidate][
                    "skill_score_unclipped"
                ]
            )
            for season in REPORT_SEASONS
        }
        briers = {
            season: float(
                folds[str(season)]["candidates"][candidate]["brier_score"]
            )
            for season in REPORT_SEASONS
        }
        summaries[candidate] = {
            "season_skills": {
                str(season): value for season, value in skills.items()
            },
            "season_briers": {
                str(season): value for season, value in briers.items()
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
            "season_skill_change_vs_base": {
                str(season): float(skills[season] - base_skills[season])
                for season in REPORT_SEASONS
            },
            "season_brier_change_vs_base": {
                str(season): float(briers[season] - base_briers[season])
                for season in REPORT_SEASONS
            },
            "mean_skill_change_vs_base": float(
                np.mean(list(skills.values()))
                - np.mean(list(base_skills.values()))
            ),
            "min_skill_change_vs_base": float(
                np.min(list(skills.values()))
                - np.min(list(base_skills.values()))
            ),
        }
    return summaries


def select_from_previous_folds(
    validation_season: int,
    folds: dict[str, object],
) -> tuple[str, list[int], dict[str, object]]:
    history = [
        season for season in EVALUATED_SEASONS if season < validation_season
    ]
    if not history:
        return "base_identity", [], {}

    selection_metrics: dict[str, object] = {}
    for candidate in CANDIDATE_ORDER:
        skills = [
            float(
                folds[str(season)]["candidates"][candidate][
                    "skill_score_unclipped"
                ]
            )
            for season in history
        ]
        selection_metrics[candidate] = {
            "history_skills": {
                str(season): value for season, value in zip(history, skills)
            },
            "history_min_skill": float(np.min(skills)),
            "history_mean_skill": float(np.mean(skills)),
        }

    # CANDIDATE_ORDER is the deterministic tie break; identity wins exact ties.
    selected = max(
        CANDIDATE_ORDER,
        key=lambda candidate: (
            selection_metrics[candidate]["history_min_skill"],
            selection_metrics[candidate]["history_mean_skill"],
            -CANDIDATE_ORDER.index(candidate),
        ),
    )
    return selected, history, selection_metrics


def main() -> None:
    started = time.time()
    rows = prepare_context_rows()
    targets, base_predictions = load_base_oof(rows)
    corrections, correction_diagnostics = build_family_corrections(
        rows,
        targets,
        base_predictions,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    prediction_cache: dict[int, dict[str, np.ndarray]] = {}
    for season in EVALUATED_SEASONS:
        predictions = candidate_predictions(
            season,
            base_predictions[season],
            corrections,
        )
        prediction_cache[season] = predictions
        candidate_metrics = {
            candidate: calculate_metrics(targets[season], values)
            for candidate, values in predictions.items()
        }
        folds[str(season)] = {
            "validation_season": season,
            "source_oof_seasons": correction_diagnostics[str(season)][
                "source_oof_seasons"
            ],
            "candidates": candidate_metrics,
        }
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", targets[season])
        for candidate, values in predictions.items():
            np.save(
                ARTIFACT_DIR / f"predictions_{candidate}_{season}.npy",
                values,
            )
        print(
            f"context_eb {season}: "
            + " ".join(
                f"{candidate}={candidate_metrics[candidate]['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATE_ORDER
            ),
            flush=True,
        )

    strict_folds: dict[str, object] = {}
    strict_predictions: dict[int, np.ndarray] = {}
    for season in EVALUATED_SEASONS:
        selected, history, selection_metrics = select_from_previous_folds(
            season,
            folds,
        )
        values = prediction_cache[season][selected]
        strict_predictions[season] = values
        strict_folds[str(season)] = {
            "validation_season": season,
            "selection_history_seasons": history,
            "selected_candidate": selected,
            "selection_metrics": selection_metrics,
            "metrics": calculate_metrics(targets[season], values),
        }
        np.save(
            ARTIFACT_DIR / f"predictions_strict_previous_{season}.npy",
            values,
        )

    summaries = summarize_candidates(folds)
    robust_best = max(
        CANDIDATE_ORDER,
        key=lambda candidate: (
            summaries[candidate]["min_skill"],
            summaries[candidate]["latest_2024_skill"],
            summaries[candidate]["mean_skill"],
            -CANDIDATE_ORDER.index(candidate),
        ),
    )
    best_context_candidate = max(
        CANDIDATE_ORDER[1:],
        key=lambda candidate: (
            summaries[candidate]["min_skill"],
            summaries[candidate]["latest_2024_skill"],
            summaries[candidate]["mean_skill"],
            -CANDIDATE_ORDER.index(candidate),
        ),
    )
    strict_skills = {
        season: float(
            strict_folds[str(season)]["metrics"]["skill_score_unclipped"]
        )
        for season in REPORT_SEASONS
    }
    strict_briers = {
        season: float(strict_folds[str(season)]["metrics"]["brier_score"])
        for season in REPORT_SEASONS
    }
    strict_summary = {
        "season_skills": {
            str(season): value for season, value in strict_skills.items()
        },
        "season_briers": {
            str(season): value for season, value in strict_briers.items()
        },
        "mean_skill": float(np.mean(list(strict_skills.values()))),
        "min_skill": float(np.min(list(strict_skills.values()))),
        "latest_2024_skill": strict_skills[2024],
        "selection_path": {
            str(season): strict_folds[str(season)]["selected_candidate"]
            for season in REPORT_SEASONS
        },
    }
    next_selected, next_history, next_selection_metrics = (
        select_from_previous_folds(2025, folds)
    )

    base_metrics = json.loads(BASE_METRICS_PATH.read_text(encoding="utf-8"))
    base_reference = base_metrics["aggregate_2022_2024"]["all_prior_s1000"]
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": "temporal_context_EB_on_team_allprior",
        "validation_protocol": {
            "evaluated_oof_seasons": list(EVALUATED_SEASONS),
            "reported_seasons": list(REPORT_SEASONS),
            "base": "EXP-019 team all_prior_s1000 temporal-safe OOF",
            "effect_training": (
                "one independent source-season EB map from centered OOF "
                "residual; equal average across all earlier source seasons, "
                "including zero for missing keys"
            ),
            "current_fold_labels_used_for_effects": False,
            "strict_selection_uses_current_fold": False,
            "test_row_aggregation": False,
            "candidate_comparison_post_hoc": True,
            "nested_caveat": (
                "candidate family and weights are fixed inside this run, but "
                "the team base and context family were proposed after earlier "
                "OOF inspection"
            ),
        },
        "effect_configuration": {
            "source_residual_centering": "subtract source OOF season mean",
            "source_effect_clip": SOURCE_EFFECT_CLIP,
            "families": {
                family.name: {
                    "columns": list(family.columns),
                    "smoothing": family.smoothing,
                }
                for family in FAMILIES
            },
            "composites": {
                name: {"members": list(members), "weights": "equal"}
                for name, members in COMPOSITES.items()
            },
            "inning_bucket": {
                "early": "inning <= 3",
                "middle": "4 <= inning <= 6",
                "late": "inning >= 7",
            },
        },
        "correction_diagnostics": correction_diagnostics,
        "folds": folds,
        "aggregate_2022_2024": summaries,
        "strict_previous_fold_selection": {
            "objective": "maximize worst earlier-fold Skill, then earlier-fold mean Skill",
            "tie_break": "CANDIDATE_ORDER; base_identity first",
            "folds": strict_folds,
            "aggregate_2022_2024": strict_summary,
            "next_2025_selection": {
                "selection_history_seasons": next_history,
                "selected_candidate": next_selected,
                "selection_metrics": next_selection_metrics,
            },
        },
        "base_reference": {
            "source": str(BASE_METRICS_PATH),
            "variant": "all_prior_s1000",
            "mean_skill": float(base_reference["team_eb_mean_skill"]),
            "min_skill": float(base_reference["team_eb_min_skill"]),
        },
        "selection": {
            "robust_best_candidate_including_identity": robust_best,
            "robust_best_min_skill": float(
                summaries[robust_best]["min_skill"]
            ),
            "best_nonidentity_context_candidate": best_context_candidate,
            "best_nonidentity_min_skill": float(
                summaries[best_context_candidate]["min_skill"]
            ),
            "base_identity_min_skill": float(
                summaries["base_identity"]["min_skill"]
            ),
            "any_context_beats_base_min": bool(
                summaries[best_context_candidate]["min_skill"]
                > summaries["base_identity"]["min_skill"]
            ),
            "strict_path_beats_base_min": bool(
                strict_summary["min_skill"]
                > summaries["base_identity"]["min_skill"]
            ),
            "stop_rule_triggered": bool(
                summaries[best_context_candidate]["min_skill"]
                <= summaries["base_identity"]["min_skill"]
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "total_seconds": time.time() - started,
    }
    metrics_path = ARTIFACT_DIR / "validation_metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"strict={strict_summary}", flush=True)
    print(f"saved={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
