"""EXP-019: temporal group OOF를 offset으로 둔 multirate residual.

각 학습 행에도 그 시즌보다 과거인 데이터로만 만든 group 예측을 붙인다.
따라서 residual 모델은 count/hand 그룹 효과를 다시 학습하지 않고, 현재 시즌
다중 비율이 설명하는 추가 오차만 학습한다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import train_exp019_multirate_residual as multirate


ARTIFACT_ROOT = Path("./artifacts/EXP-019/multirate_group_residual")


def main() -> None:
    started = time.time()
    frame, diagnostics, y, base, seasons, reconstruction = (
        multirate.prepare_multirate_data()
    )
    preliminary_residual = multirate.centered_residual(y, base, seasons)
    group_all = np.empty(len(y), dtype=np.float64)
    group_predictions: dict[int, np.ndarray] = {}
    for season in sorted(np.unique(seasons).astype(int).tolist()):
        mask = seasons == season
        correction = multirate.multirate_group_correction(
            frame, preliminary_residual, seasons, season
        )
        prediction = np.clip(base[mask].astype(float) + correction, 0.0, 1.0)
        group_all[mask] = prediction
        if season in multirate.VALIDATION_SEASONS:
            group_predictions[season] = prediction

    residual_target = (y.astype(float) - group_all).astype(np.float32)
    for season in np.unique(seasons):
        mask = seasons == season
        residual_target[mask] -= residual_target[mask].mean()

    multirate.ARTIFACT_ROOT = ARTIFACT_ROOT
    summaries: dict[str, object] = {}
    for config in multirate.CONFIGS:
        result = multirate.run_config(
            config,
            frame,
            diagnostics,
            y,
            base,
            seasons,
            residual_target,
            group_predictions,
        )
        summaries[config.name] = result["aggregate_2022_2024"]

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "experiment": "EXP-019",
                "stage": "multirate_group_oof_residual_candidate_search",
                "selection_status": "not selected; nested selection required",
                "target": "season-centered y minus temporal group OOF prediction",
                "reconstruction_diagnostics": reconstruction,
                "summaries": summaries,
                "total_seconds": time.time() - started,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
