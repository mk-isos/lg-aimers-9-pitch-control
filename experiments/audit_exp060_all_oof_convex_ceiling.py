"""EXP-060: certified convex ceiling over every saved OOF prediction.

This is a non-deployable diagnostic.  It uses each validation season's labels
only to determine whether any convex reweighting of already-produced row-level
signals could attain the current 1000-per-season gate.  No model or submission
candidate is selected from this audit.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


ARTIFACTS = Path("./artifacts")
TARGET_ROOT = ARTIFACTS / "EXP-020" / "low_rank_pitcher_context_eb"
OUTPUT = ARTIFACTS / "EXP-060" / "all_oof_convex_ceiling"
SEASONS = (2022, 2023, 2024)
MAX_ITERATIONS = 80
GAP_TOLERANCE = 1e-11


def unique_prediction_files(season: int, target: np.ndarray) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in sorted(ARTIFACTS.rglob(f"predictions_*_{season}.npy")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in unique:
            continue
        value = np.load(path, mmap_mode="r")
        if (
            value.shape != target.shape
            or not np.isfinite(value).all()
            or float(value.min()) < 0.0
            or float(value.max()) > 1.0
        ):
            continue
        unique[digest] = path
    return list(unique.values())


def frank_wolfe(
    matrix: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, dict[int, float], float, int]:
    target32 = target.astype(np.float32)
    individual = np.mean(np.square(matrix - target32[None, :]), axis=1)
    start = int(np.argmin(individual))
    prediction = matrix[start].astype(np.float64)
    weights: dict[int, float] = {start: 1.0}
    gap = float("inf")
    for iteration in range(1, MAX_ITERATIONS + 1):
        error = prediction - target
        gradient = (matrix @ error.astype(np.float32)).astype(np.float64)
        vertex = int(np.argmin(gradient))
        direction = matrix[vertex].astype(np.float64) - prediction
        denominator = float(direction @ direction)
        current_dot = float(error @ prediction)
        gap = float(2.0 * (current_dot - gradient[vertex]) / len(target))
        if gap <= GAP_TOLERANCE or denominator == 0.0:
            break
        gamma = float(np.clip(-(error @ direction) / denominator, 0.0, 1.0))
        prediction += gamma * direction
        weights = {key: value * (1.0 - gamma) for key, value in weights.items()}
        weights[vertex] = weights.get(vertex, 0.0) + gamma
        weights = {key: value for key, value in weights.items() if value > 1e-10}
    return prediction, weights, gap, iteration


def main() -> None:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for season in SEASONS:
        target = np.load(TARGET_ROOT / f"targets_{season}.npy").astype(float)
        paths = unique_prediction_files(season, target)
        matrix = np.vstack(
            [np.load(path).astype(np.float32, copy=False) for path in paths]
        )
        prediction, weights, gap, iterations = frank_wolfe(matrix, target)
        brier = float(np.mean(np.square(target - prediction)))
        baseline = float(target.mean() * (1.0 - target.mean()))
        certified_lower_brier = float(max(0.0, brier - max(gap, 0.0)))
        threshold = float(baseline * (1.0 - 1000.0 / 100000.0))
        fold = {
            "rows": len(target),
            "discovered_prediction_files": len(
                list(ARTIFACTS.rglob(f"predictions_*_{season}.npy"))
            ),
            "unique_valid_candidates": len(paths),
            "iterations": iterations,
            "frank_wolfe_gap_brier": gap,
            "oracle_brier": brier,
            "oracle_skill": float(100000.0 * (1.0 - brier / baseline)),
            "certified_lower_brier": certified_lower_brier,
            "certified_upper_skill": float(
                100000.0 * (1.0 - certified_lower_brier / baseline)
            ),
            "skill_1000_brier_threshold": threshold,
            "certified_margin_to_1000_brier": float(
                certified_lower_brier - threshold
            ),
            "can_convex_hull_reach_1000": bool(certified_lower_brier <= threshold),
            "active_weights": [
                {"path": str(paths[index]), "weight": float(weight)}
                for index, weight in sorted(
                    weights.items(), key=lambda item: item[1], reverse=True
                )
            ],
        }
        folds[str(season)] = fold
        np.save(OUTPUT / f"predictions_oracle_{season}.npy", prediction)
        np.save(OUTPUT / f"targets_{season}.npy", target.astype(np.int8))
        print(
            f"{season}: candidates={len(paths)} iter={iterations} "
            f"skill={fold['oracle_skill']:.3f} "
            f"upper={fold['certified_upper_skill']:.3f} "
            f"reach1000={fold['can_convex_hull_reach_1000']}",
            flush=True,
        )
        del matrix
    result = {
        "experiment": "EXP-060",
        "purpose": "all saved OOF nonnegative convex-hull ceiling audit",
        "protocol": {
            "same_fold_labels_used": True,
            "deployable": False,
            "candidate_selection_nested": False,
            "test_rows_used": False,
            "scope": "all unique finite [0,1] predictions_* OOF arrays",
            "max_iterations": MAX_ITERATIONS,
            "gap_tolerance": GAP_TOLERANCE,
        },
        "folds": folds,
        "conclusion": {
            "latest_2023_2024_certified_below_1000": bool(
                all(
                    not fold["can_convex_hull_reach_1000"]
                    for season, fold in folds.items()
                    if int(season) >= 2023
                )
            ),
            "reweighting_adopt": False,
        },
        "total_seconds": time.time() - started,
    }
    (OUTPUT / "validation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
