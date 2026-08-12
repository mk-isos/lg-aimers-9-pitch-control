"""Independent saved-artifact QA for EXP-027 through EXP-030."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ROOTS = {
    "EXP-027": Path("./artifacts/EXP-027/logit_offset_context_eb"),
    "EXP-028": Path("./artifacts/EXP-028/prevalence_invariant_taxonomy"),
    "EXP-029": Path("./artifacts/EXP-029/joint_pitchmix_taxonomy"),
    "EXP-030": Path("./artifacts/EXP-030/pitch_selection_residual"),
}
REPORT_SEASONS = (2022, 2023, 2024)


def brier(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.square(targets.astype(float) - predictions.astype(float))))


def assert_finite_json(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            assert_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite_json(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def candidate_file(experiment: str, name: str, season: int) -> Path:
    if experiment == "EXP-029":
        return ROOTS[experiment] / f"predictions_{name}_{season}.npy"
    return ROOTS[experiment] / f"predictions_{name}_{season}.npy"


def main() -> None:
    checked = 0
    for experiment, root in ROOTS.items():
        with (root / "validation_metrics.json").open(encoding="utf-8") as file:
            metrics = json.load(file)
        assert metrics["experiment"] == experiment
        assert_finite_json(metrics)
        aggregate = metrics["aggregate_2022_2024"]
        assert aggregate["uniform_1100_passed"] is False
        assert aggregate["final_fit_authorized"] is False
        assert aggregate["zip_creation_authorized"] is False
        assert metrics["qa"]["final_fit_or_zip_created"] is False

        for path in sorted(root.glob("*.npy")):
            values = np.load(path)
            assert np.isfinite(values).all(), path
            if path.name.startswith("predictions_"):
                assert ((values >= 0.0) & (values <= 1.0)).all(), path
            checked += 1

        for season in REPORT_SEASONS:
            targets = np.load(BASE_ROOT / f"targets_{season}.npy").astype(float)
            if experiment == "EXP-029":
                fold = metrics["folds"][str(season)]
                name = fold["selected"]["candidate"]
                predictions = np.load(candidate_file(experiment, name, season))
                stored_brier = float(fold["selected"]["brier_score"])
            else:
                fold = metrics["folds"][str(season)]
                for name, stored in fold["candidates"].items():
                    predictions = np.load(candidate_file(experiment, name, season))
                    assert len(predictions) == len(targets)
                    assert abs(brier(targets, predictions) - stored["brier_score"]) < 1e-14
                continue
            assert len(predictions) == len(targets)
            assert abs(brier(targets, predictions) - stored_brier) < 1e-14
    print(f"EXP027_030_QA_OK experiments={len(ROOTS)} arrays={checked} uniform_1100=all_false")


if __name__ == "__main__":
    main()
