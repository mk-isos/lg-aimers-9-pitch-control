"""EXP-038: optimistic convex ceiling of every saved 2022--2024 prediction.

This is a non-deployable diagnostic.  It scans prediction candidates that have
saved arrays for all three reported seasons, verifies target/order/range, drops
exact duplicates, and solves each season's convex-hull Brier projection with
Frank-Wolfe line search.  The Frank-Wolfe dual gap supplies a certified lower
bound on the best attainable Brier inside this intentionally optimistic pool.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


ARTIFACT_ROOT = Path("./artifacts")
OUTPUT_DIR = Path("./artifacts/EXP-038/expanded_ensemble_ceiling")
SEASONS = (2022, 2023, 2024)
CANONICAL_TARGET_ROOT = Path(
    "./artifacts/EXP-020/low_rank_pitcher_context_eb"
)
MAX_ITERATIONS = 300
TOLERANCE = 1e-14


def brier_skill(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    brier = float(np.mean(np.square(target - prediction)))
    baseline = float(target.mean() * (1.0 - target.mean()))
    return {
        "brier_score": brier,
        "baseline_brier": baseline,
        "skill_score": float(100000.0 * (1.0 - brier / baseline)),
    }


def candidate_paths() -> list[tuple[str, dict[int, Path]]]:
    candidates: list[tuple[str, dict[int, Path]]] = []
    suffix = "_2022.npy"
    for path in sorted(ARTIFACT_ROOT.rglob(f"predictions_*{suffix}")):
        stem = path.name[len("predictions_") : -len(suffix)]
        paths = {
            season: path.parent / f"predictions_{stem}_{season}.npy"
            for season in SEASONS
        }
        if all(value.exists() for value in paths.values()):
            key = f"{path.parent.as_posix()}::{stem}"
            candidates.append((key, paths))
    return candidates


def load_pool(
    targets: dict[int, np.ndarray],
) -> tuple[list[str], dict[int, np.ndarray], dict[str, object]]:
    names: list[str] = []
    arrays: dict[int, list[np.ndarray]] = {season: [] for season in SEASONS}
    seen_hashes: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    rejected: dict[str, str] = {}
    for name, paths in candidate_paths():
        loaded: dict[int, np.ndarray] = {}
        valid = True
        for season, path in paths.items():
            prediction = np.load(path).astype(np.float64)
            if len(prediction) != len(targets[season]):
                rejected[name] = f"length mismatch {season}"
                valid = False
                break
            if not np.isfinite(prediction).all() or not (
                (prediction >= 0.0).all() and (prediction <= 1.0).all()
            ):
                rejected[name] = f"invalid probability {season}"
                valid = False
                break
            loaded[season] = prediction
        if not valid:
            continue
        digest = hashlib.sha256()
        for season in SEASONS:
            digest.update(loaded[season].astype(np.float32).tobytes())
        signature = digest.hexdigest()
        if signature in seen_hashes:
            duplicates[name] = seen_hashes[signature]
            continue
        seen_hashes[signature] = name
        names.append(name)
        for season in SEASONS:
            arrays[season].append(loaded[season].astype(np.float32))
    matrices = {
        season: np.vstack(arrays[season]).astype(np.float64)
        for season in SEASONS
    }
    return names, matrices, {
        "discovered_complete_candidates": len(candidate_paths()),
        "unique_candidates": len(names),
        "exact_duplicate_count": len(duplicates),
        "rejected": rejected,
        "duplicates": duplicates,
    }


def frank_wolfe(
    target: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    candidate_briers = np.mean(
        np.square(matrix - target[None, :]), axis=1
    )
    start = int(np.argmin(candidate_briers))
    weights = np.zeros(matrix.shape[0], dtype=np.float64)
    weights[start] = 1.0
    prediction = matrix[start].copy()
    gap = float("inf")
    iteration = 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        residual = prediction - target
        gradient = 2.0 * np.mean(matrix * residual[None, :], axis=1)
        vertex = int(np.argmin(gradient))
        gap = float(np.dot(weights, gradient) - gradient[vertex])
        if gap <= TOLERANCE:
            break
        direction = matrix[vertex] - prediction
        denominator = float(np.dot(direction, direction))
        if denominator <= 0.0:
            break
        step = float(
            np.clip(-np.dot(residual, direction) / denominator, 0.0, 1.0)
        )
        if step <= 0.0:
            break
        weights *= 1.0 - step
        weights[vertex] += step
        prediction += step * direction
    objective = float(np.mean(np.square(target - prediction)))
    certified_lower = max(0.0, objective - max(gap, 0.0))
    return weights, prediction, {
        "iterations": iteration,
        "frank_wolfe_gap": gap,
        "objective_brier": objective,
        "certified_lower_brier": certified_lower,
    }


def main() -> None:
    started = time.time()
    targets = {
        season: np.load(
            CANONICAL_TARGET_ROOT / f"targets_{season}.npy"
        ).astype(np.float64)
        for season in SEASONS
    }
    names, matrices, pool_audit = load_pool(targets)
    if not names:
        raise ValueError("no complete prediction candidates found")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seasons: dict[str, object] = {}
    for season in SEASONS:
        weights, prediction, diagnostics = frank_wolfe(
            targets[season], matrices[season]
        )
        metrics = brier_skill(targets[season], prediction)
        threshold_brier = float(
            metrics["baseline_brier"] * (1.0 - 1000.0 / 100000.0)
        )
        active = [
            {"candidate": names[index], "weight": float(value)}
            for index, value in enumerate(weights)
            if value > 1e-8
        ]
        seasons[str(season)] = {
            "metrics": metrics,
            **diagnostics,
            "skill_1000_threshold_brier": threshold_brier,
            "certified_lower_minus_threshold": float(
                diagnostics["certified_lower_brier"] - threshold_brier
            ),
            "skill_1000_attainable_in_convex_pool": bool(
                diagnostics["certified_lower_brier"] <= threshold_brier
            ),
            "active_weights": active,
        }
        np.save(OUTPUT_DIR / f"predictions_oracle_{season}.npy", prediction)
        np.save(OUTPUT_DIR / f"targets_{season}.npy", targets[season].astype(np.int8))
        print(
            f"{season}: skill={metrics['skill_score']:.3f} "
            f"brier={metrics['brier_score']:.12f} gap={diagnostics['frank_wolfe_gap']:.3e} "
            f"active={len(active)}",
            flush=True,
        )
    report = {
        "experiment": "EXP-038",
        "stage": "expanded_saved_prediction_convex_ceiling",
        "scope": {
            "non_deployable_same_fold_oracle": True,
            "includes_posthoc_and_diagnostic_candidates": True,
            "interpretation": (
                "optimistic upper bound for convex reweighting only; not a "
                "valid candidate selection protocol"
            ),
            "seasons": list(SEASONS),
        },
        "pool_audit": pool_audit,
        "candidate_names": names,
        "seasons": seasons,
        "selection": {
            "all_2022_2024_reach_skill_1000": bool(
                all(
                    seasons[str(season)][
                        "skill_1000_attainable_in_convex_pool"
                    ]
                    for season in SEASONS
                )
            ),
            "continue_convex_weight_search": False,
        },
        "total_seconds": time.time() - started,
    }
    (OUTPUT_DIR / "validation_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
