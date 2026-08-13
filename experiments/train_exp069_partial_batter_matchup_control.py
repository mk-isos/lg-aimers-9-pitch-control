"""EXP-069: partial-aligned pitcher/batter propensity and matchup control."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from train_exp017_rolling_residual import calculate_metrics
from train_exp050_exact_dual_propensity_control import (
    build_corrections,
    load_main,
    load_trackman,
)
from train_exp066_partial_sequence_alignment_control import (
    base_components,
    partial_aligned_rows,
)


LOWRANK_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ARTIFACT_DIR = Path("./artifacts/EXP-069/partial_batter_matchup_control")
EVALUATED_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
CANDIDATES = (
    "partial_pitcher_w010",
    "partial_dual25_w010",
    "partial_matchup_w010",
    "exact_partial_matchup_blend_w010",
)


def main() -> None:
    started = time.time()
    main = load_main()
    trackman = load_trackman()
    aligned, alignment_audit = partial_aligned_rows()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    audits: dict[str, object] = {}
    for season in EVALUATED_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        recent, exact_correction = base_components(season)
        corrections, audits[str(season)] = build_corrections(
            main, trackman, aligned, season
        )
        base = np.clip(recent + 0.10 * exact_correction, 0, 1)
        partial_blend = 0.5 * corrections["dual25"] + 0.5 * corrections["matchup"]
        predictions = {
            "base": base,
            "partial_pitcher_w010": np.clip(
                recent + 0.10 * corrections["pitcher_prop"], 0, 1
            ),
            "partial_dual25_w010": np.clip(
                recent + 0.10 * corrections["dual25"], 0, 1
            ),
            "partial_matchup_w010": np.clip(
                recent + 0.10 * corrections["matchup"], 0, 1
            ),
            "exact_partial_matchup_blend_w010": np.clip(
                recent + 0.05 * exact_correction + 0.05 * partial_blend,
                0,
                1,
            ),
        }
        fold: dict[str, object] = {
            "validation_season": season,
            "history_cutoff": season - 1,
        }
        for name, prediction in predictions.items():
            fold[name] = calculate_metrics(target, prediction)
            np.save(ARTIFACT_DIR / f"predictions_{name}_{season}.npy", prediction)
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", target.astype(np.int8))
        folds[str(season)] = fold
        print(
            f"fold {season}: "
            + " ".join(
                f"{name}={fold[name]['skill_score_unclipped']:.2f}"
                for name in CANDIDATES
            ),
            flush=True,
        )
    aggregate: dict[str, object] = {}
    for name in ("base", *CANDIDATES):
        skills = {
            str(season): float(folds[str(season)][name]["skill_score_unclipped"])
            for season in REPORT_SEASONS
        }
        briers = {
            str(season): float(folds[str(season)][name]["brier_score"])
            for season in REPORT_SEASONS
        }
        aggregate[name] = {
            "season_briers": briers,
            "season_skills": skills,
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills["2024"],
        }
    best = max(
        CANDIDATES,
        key=lambda name: (aggregate[name]["min_skill"], aggregate[name]["mean_skill"]),
    )
    result = {
        "experiment": "EXP-069",
        "candidate_family": "partial_aligned_pitcher_batter_matchup_control",
        "validation_protocol": {
            "outer_folds": list(EVALUATED_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "alignment_mapping_and_tables_cutoff": "validation season-1",
            "same_state_matching_blocks_only": True,
            "current_fold_labels_used_for_fit_or_selection": False,
            "actual_current_pitch_type_used": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
        },
        "alignment_audit": alignment_audit,
        "fold_feature_audit": audits,
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "best_fixed_candidate": best,
            "best_min_skill": aggregate[best]["min_skill"],
            "best_mean_skill": aggregate[best]["mean_skill"],
            "gate_each_season_1000": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
            ),
            "gate_mean_1100": bool(aggregate[best]["mean_skill"] >= 1100.0),
            "adopt": bool(
                min(aggregate[best]["season_skills"].values()) >= 1000.0
                and aggregate[best]["mean_skill"] >= 1100.0
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "total_seconds": time.time() - started,
    }
    (ARTIFACT_DIR / "validation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"best={best} mean={aggregate[best]['mean_skill']:.2f} "
        f"min={aggregate[best]['min_skill']:.2f} adopt={result['selection']['adopt']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
