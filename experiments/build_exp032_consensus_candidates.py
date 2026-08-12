"""Build three bounded EXP-032 consensus submission candidates.

The candidates reuse the already verified EXP-021 frozen backbone and combine
three temporally evaluated prediction branches only:

* strict: all-row pitcher-context low-rank SVD, smoothing 300, rank 6;
* R-specific: the same residual family fitted/applied on regular-season rows,
  smoothing 300, rank 4;
* aggressive: R/F gated team base plus pitcher-count EB.

No model is fitted from test rows.  The additional R-specific state is fitted
from the four frozen source OOF seasons (2021--2024), independently by season,
then averaged at inference with an absent source contributing zero.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from build_exp021_final_candidates import build_zip, smoke_test
from train_exp017_rolling_residual import calculate_metrics
from train_exp020_low_rank_pitcher_context_eb import (
    CONTEXTS,
    fit_source_matrix,
    load_oof,
    load_rows,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "experiments" / "exp021_submission_inference.py"
SOURCE_DIR = ROOT / "submissions" / "EXP-021-STRICT"
LOWRANK_ROOT = ROOT / "artifacts" / "EXP-020" / "low_rank_pitcher_context_eb"
AGGRESSIVE_ROOT = ROOT / "artifacts" / "EXP-020" / "pitcher_count_eb_atop_team"
ARTIFACT_DIR = ROOT / "artifacts" / "EXP-032" / "consensus_candidates"
SOURCE_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)

VARIANTS = {
    "dualrank": {
        "candidate": "dualrank_consensus_50",
        "directory": ROOT / "submissions" / "EXP-032-DUALRANK",
        "zip": ROOT / "submit_exp032_dualrank.zip",
        "weights": {"strict": 0.5, "r_specific": 0.5, "aggressive": 0.0},
    },
    "stableaggr": {
        "candidate": "strict_aggressive_consensus_50",
        "directory": ROOT / "submissions" / "EXP-032-STABLEAGGR",
        "zip": ROOT / "submit_exp032_stableaggr.zip",
        "weights": {"strict": 0.5, "r_specific": 0.0, "aggressive": 0.5},
    },
    "threeway": {
        "candidate": "threeway_consensus_equal",
        "directory": ROOT / "submissions" / "EXP-032-THREEWAY",
        "zip": ROOT / "submit_exp032_threeway.zip",
        "weights": {
            "strict": 1.0 / 3.0,
            "r_specific": 1.0 / 3.0,
            "aggressive": 1.0 / 3.0,
        },
    },
}


def write_json(path: Path, value: object, indent: int | None = None) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def build_r_specific_state() -> dict[str, object]:
    rows = load_rows()
    targets, base, _ = load_oof(rows)
    sources: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {}
    for season in SOURCE_SEASONS:
        source_rows = rows[season]
        is_regular = source_rows["game_type"].astype(str).eq("R").to_numpy()
        model = fit_source_matrix(
            season,
            source_rows.loc[is_regular].reset_index(drop=True),
            targets[season][is_regular],
            base[season][is_regular],
            smoothing_grid=(300.0,),
            rank_grid=(4,),
        )
        reconstruction = model["matrices"][300.0]["reconstructions"][4]
        sources.append(
            {
                "season": season,
                "pitcher_ids": [int(value) for value in model["pitcher_ids"]],
                "values": reconstruction.astype(float).tolist(),
            }
        )
        diagnostics[str(season)] = model["diagnostics"]
    contexts = [
        {
            "position": position,
            "count_index": count_index,
            "batter_hand": batter_hand,
        }
        for position, (count_index, batter_hand) in enumerate(CONTEXTS)
    ]
    return {
        "smoothing": 300.0,
        "rank": 4,
        "source_training_rows": "R only",
        "application_rows": "R only; inference explicitly zeros F rows",
        "source_seasons": list(SOURCE_SEASONS),
        "source_combination": "equal average; absent pitcher contributes zero",
        "contexts": contexts,
        "sources": sources,
        "diagnostics": diagnostics,
    }


def load_component_predictions(season: int) -> dict[str, np.ndarray]:
    return {
        "strict": np.load(
            LOWRANK_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
        ).astype(float),
        "r_specific": np.load(
            LOWRANK_ROOT
            / f"predictions_lowrank_s300_r4_Rspecific_{season}.npy"
        ).astype(float),
        "aggressive": np.load(
            AGGRESSIVE_ROOT
            / f"predictions_r_gated_team_pc_all_{season}.npy"
        ).astype(float),
    }


def calculate_variant_metrics(weights: dict[str, float]) -> dict[str, object]:
    season_briers: dict[str, float] = {}
    season_skills: dict[str, float] = {}
    for season in REPORT_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        components = load_component_predictions(season)
        lengths = {len(target), *(len(value) for value in components.values())}
        if len(lengths) != 1:
            raise ValueError(f"OOF length mismatch for {season}")
        prediction = np.zeros(len(target), dtype=float)
        for name, weight in weights.items():
            prediction += float(weight) * components[name]
        prediction = np.clip(prediction, 0.0, 1.0)
        metrics = calculate_metrics(target, prediction)
        season_briers[str(season)] = float(metrics["brier_score"])
        season_skills[str(season)] = float(
            metrics["skill_score_unclipped"]
        )
    skills = list(season_skills.values())
    return {
        "season_briers": season_briers,
        "season_skills": season_skills,
        "mean_skill": float(np.mean(skills)),
        "min_skill": float(np.min(skills)),
        "latest_2024_skill": season_skills["2024"],
    }


def copy_shared_package(destination: Path) -> None:
    model_dir = destination / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, destination / "script.py")
    shutil.copyfile(SOURCE_DIR / "requirements.txt", destination / "requirements.txt")
    for source in sorted((SOURCE_DIR / "model").iterdir()):
        if source.is_file() and source.name != "metadata.json":
            shutil.copyfile(source, model_dir / source.name)


def main() -> None:
    started = time.time()
    if not SOURCE_DIR.exists():
        raise FileNotFoundError("verified EXP-021 strict package source is missing")
    r_specific_state = build_r_specific_state()
    strict_metadata = json.loads(
        (SOURCE_DIR / "model" / "metadata.json").read_text(encoding="utf-8")
    )

    results: dict[str, object] = {}
    for label, details in VARIANTS.items():
        destination = details["directory"]
        copy_shared_package(destination)
        write_json(
            destination / "model" / "lowrank_rspecific_effects.json",
            r_specific_state,
        )
        weights = details["weights"]
        metrics = calculate_variant_metrics(weights)
        metadata = dict(strict_metadata)
        metadata.update(
            {
                "experiment": "EXP-032",
                "candidate": details["candidate"],
                "component_weights": weights,
                "validation_aggregate_2022_2024": metrics,
                "selection_status": (
                    "bounded post-hoc consensus diagnostic; weights were not "
                    "selected by a fully nested outer protocol"
                ),
                "parent_public_scores": {
                    "strict": 1043.6074197937,
                    "aggressive": 1043.1871309639,
                },
                "probability_calibration": "identity",
            }
        )
        write_json(destination / "model" / "metadata.json", metadata, indent=2)
        zip_result = build_zip(destination, details["zip"])
        smoke = smoke_test(details["zip"])
        results[label] = {
            "candidate": details["candidate"],
            "component_weights": weights,
            "validation": metrics,
            "zip": zip_result,
            "smoke": smoke,
        }
        print(
            f"{label}: mean={metrics['mean_skill']:.3f} "
            f"min={metrics['min_skill']:.3f} smoke=passed",
            flush=True,
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "EXP-032",
        "stage": "three_consensus_submission_candidates",
        "protocol": {
            "reported_folds": list(REPORT_SEASONS),
            "components": ["strict", "r_specific", "aggressive"],
            "current_fold_labels_used_to_fit_component_models": False,
            "test_row_aggregation": False,
            "candidate_weight_selection_nested": False,
            "fixed_calibration": "identity",
        },
        "results": results,
        "qa": {
            "source_seasons": list(SOURCE_SEASONS),
            "r_specific_smoothing": 300.0,
            "r_specific_rank": 4,
            "r_specific_context_count": len(CONTEXTS),
            "python": platform.python_version(),
        },
        "total_seconds": time.time() - started,
    }
    write_json(ARTIFACT_DIR / "validation_metrics.json", report, indent=2)
    print(f"saved {ARTIFACT_DIR / 'validation_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
