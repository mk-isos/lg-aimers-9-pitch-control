"""Package the bounded EXP-044 exact-Trackman/recent consensus candidate.

The exact game-sequence alignment is fit through 2024 only.  It exports a
frozen official-pitcher to Trackman-pitcher map, fine-pitch-type hierarchical
control rates, and current-row pitch-type propensities.  Inference never
aggregates evaluation rows.
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
from train_exp043_exact_pitchtype_control_eb import (
    CONTEXT_SMOOTHING,
    CORRECTION_CLIP,
    FINE_TYPES,
    PITCHER_SMOOTHING,
    PROPENSITY_SMOOTHING,
    TYPE_SMOOTHING,
    load_main,
    load_trackman,
    posterior_tables,
    propensity_table,
)
from train_exp041_exact_game_trackman_sequence import (
    exact_aligned_rows,
    mapping_from_aligned,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "submissions" / "EXP-032-RECENTAGGR"
DESTINATION = ROOT / "submissions" / "EXP-044-TRACKREC"
ZIP_PATH = ROOT / "submit_exp044_trackrec.zip"
ARTIFACT_DIR = ROOT / "artifacts" / "EXP-044" / "exact_trackman_consensus"
TEMPLATE = ROOT / "experiments" / "exp021_submission_inference.py"
LOWRANK_ROOT = ROOT / "artifacts" / "EXP-020" / "low_rank_pitcher_context_eb"
AGGRESSIVE_ROOT = ROOT / "artifacts" / "EXP-020" / "pitcher_count_eb_atop_team"
RECENCY_ROOT = ROOT / "artifacts" / "EXP-037" / "lowrank_source_policies"
TRACKMAN_ROOT = ROOT / "artifacts" / "EXP-043" / "exact_pitchtype_control_eb"
REPORT_SEASONS = (2022, 2023, 2024)
TRACKMAN_WEIGHT = 0.50
RECENT_WEIGHT = 0.50
DIRECT_CORRECTION_WEIGHT = 0.25


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def export_exact_state() -> tuple[dict[str, object], dict[str, object]]:
    # Loading main here is an explicit schema/order sanity check.  The exported
    # state itself is derived only from the exact-aligned 2019--2024 history.
    main = load_main()
    aligned, alignment_audit = exact_aligned_rows()
    if len(main) <= len(aligned):
        raise ValueError("exact aligned subset unexpectedly covers all training rows")
    trackman = load_trackman()
    mapping, mapping_audit = mapping_from_aligned(aligned, 2024)
    pitcher_rate, _, type_rate, context_rate, league = posterior_tables(
        aligned, 2024
    )
    propensity = propensity_table(trackman, 2024)

    type_records = [
        [int(pitcher), str(pitch_type), float(value)]
        for (pitcher, pitch_type), value in type_rate.items()
    ]
    context_records = [
        [
            int(pitcher),
            str(pitch_type),
            int(count_index),
            int(batter_hand),
            float(value),
        ]
        for (pitcher, pitch_type, count_index, batter_hand), value
        in context_rate.items()
    ]
    propensity_records = [
        [
            int(pitcher),
            int(count_index),
            int(batter_hand),
            [float(value) for value in row],
        ]
        for (pitcher, count_index, batter_hand), row
        in zip(propensity.index, propensity.to_numpy(dtype=float), strict=True)
    ]
    state = {
        "through_season": 2024,
        "pitcher_mapping": {
            str(key): int(value) for key, value in mapping.mapping.items()
        },
        "pitcher_rate": {
            str(key): float(value) for key, value in pitcher_rate.items()
        },
        "type_rate": type_records,
        "context_rate": context_records,
        "propensity": propensity_records,
        "fine_pitch_types": list(FINE_TYPES),
        "league_rate": float(league),
        "smoothing": {
            "pitcher": PITCHER_SMOOTHING,
            "type": TYPE_SMOOTHING,
            "context": CONTEXT_SMOOTHING,
            "propensity": PROPENSITY_SMOOTHING,
        },
        "correction_clip": CORRECTION_CLIP,
        "mapping_source": "exact full-game pre-pitch sequence alignment",
    }
    audit = {
        "alignment": alignment_audit,
        "mapping": mapping_audit,
        "mapping_entries": len(mapping.mapping),
        "pitcher_rates": len(pitcher_rate),
        "type_rates": len(type_rate),
        "context_rates": len(context_rate),
        "propensity_contexts": len(propensity),
    }
    return state, audit


def validation_metrics() -> dict[str, object]:
    briers: dict[str, float] = {}
    skills: dict[str, float] = {}
    component_distance: dict[str, float] = {}
    for season in REPORT_SEASONS:
        target = np.load(LOWRANK_ROOT / f"targets_{season}.npy").astype(float)
        recency = np.load(
            RECENCY_ROOT / f"predictions_recency2_{season}.npy"
        ).astype(float)
        aggressive = np.load(
            AGGRESSIVE_ROOT / f"predictions_r_gated_team_pc_all_{season}.npy"
        ).astype(float)
        exact = np.load(
            TRACKMAN_ROOT / f"predictions_fine_direct_w025_{season}.npy"
        ).astype(float)
        recent = 0.5 * recency + 0.5 * aggressive
        prediction = np.clip(
            TRACKMAN_WEIGHT * exact + RECENT_WEIGHT * recent, 0.0, 1.0
        )
        metrics = calculate_metrics(target, prediction)
        briers[str(season)] = float(metrics["brier_score"])
        skills[str(season)] = float(metrics["skill_score_unclipped"])
        component_distance[str(season)] = float(np.mean(np.square(exact - recent)))
    skill_values = list(skills.values())
    return {
        "season_briers": briers,
        "season_skills": skills,
        "mean_skill": float(np.mean(skill_values)),
        "min_skill": float(np.min(skill_values)),
        "latest_2024_skill": skills["2024"],
        "component_mse_distance": component_distance,
        "gate_each_season_1000": bool(min(skill_values) >= 1000.0),
        "gate_mean_1050": bool(np.mean(skill_values) >= 1050.0),
    }


def main() -> None:
    started = time.time()
    metrics = validation_metrics()
    state, state_audit = export_exact_state()
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    shutil.copytree(SOURCE_DIR, DESTINATION)
    shutil.copyfile(TEMPLATE, DESTINATION / "script.py")
    write_json(DESTINATION / "model" / "exact_pitchtype_control.json", state)

    metadata_path = DESTINATION / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "experiment": "EXP-044",
            "candidate": "trackman_recent_consensus_50",
            "component_weights": {
                "exact_pitchtype_direct": TRACKMAN_WEIGHT,
                "recentaggr": RECENT_WEIGHT,
                "inside_recentaggr": {"recency2": 0.5, "aggressive": 0.5},
                "inside_exact_pitchtype_direct_correction": DIRECT_CORRECTION_WEIGHT,
            },
            "validation_aggregate_2022_2024": metrics,
            "trackman_state_audit": state_audit,
            "public_reference": {
                "recentaggr": 1046.9889925352,
                "source": "user-provided leaderboard result",
            },
            "selection_status": (
                "bounded row-level Trackman signal consensus; component family "
                "was inspected on reported folds and is not fully nested"
            ),
            "probability_calibration": "identity",
        }
    )
    write_json(metadata_path, metadata)

    zip_result = build_zip(DESTINATION, ZIP_PATH)
    smoke = smoke_test(ZIP_PATH)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "EXP-044",
        "candidate": metadata["candidate"],
        "purpose": "exact aligned fine-pitch control plus recentaggr consensus",
        "validation": metrics,
        "trackman_state_audit": state_audit,
        "zip": zip_result,
        "smoke": smoke,
        "qa": {
            "current_fold_labels_used_to_fit_components": False,
            "candidate_comparison_fully_nested": False,
            "test_row_aggregation": False,
            "test_rows_used_for_mapping_or_propensity": False,
            "fixed_calibration": "identity",
            "python": platform.python_version(),
        },
        "total_seconds": time.time() - started,
    }
    write_json(ARTIFACT_DIR / "validation_metrics.json", report)
    print(
        f"saved={ZIP_PATH} mean={metrics['mean_skill']:.3f} "
        f"min={metrics['min_skill']:.3f} smoke=passed",
        flush=True,
    )


if __name__ == "__main__":
    main()
