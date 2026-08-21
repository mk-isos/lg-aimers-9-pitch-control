"""Complete the explicit EXP-110 row-independence audit matrix."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from build_post58_exp063_064_submission_candidates import (
    _prediction_map,
    run_fixture,
    source_only_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = (
    ROOT
    / "ready_to_submit/2026-08-21-correction-composition/EXP-110-MECHANISM-COMPOSITION.zip"
)
BUILD_REPORT = (
    ROOT / "ready_to_submit/2026-08-21-correction-composition/build_exp110_report.json"
)
OUTPUT = (
    ROOT / "ready_to_submit/2026-08-21-correction-composition/qa_exp110_independence.json"
)


def main() -> None:
    build = json.loads(BUILD_REPORT.read_text(encoding="utf-8"))
    inherited = dict(build["row_independence"])
    if float(inherited["max_abs_difference"]) > 1e-12:
        raise ValueError("existing EXP-110 independence audit failed")
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig")
    fixture = source_only_fixture(raw, rows=6)
    with tempfile.TemporaryDirectory(prefix="exp110-random-permutation-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(stage)
        reference, _ = run_fixture(stage, fixture)
        order = np.random.default_rng(42).permutation(len(fixture))
        permuted, _ = run_fixture(stage, fixture.iloc[order].reset_index(drop=True))
    reference_map = _prediction_map(reference)
    permuted_map = _prediction_map(permuted)
    random_max = float(
        max(abs(reference_map[key] - permuted_map[key]) for key in reference_map)
    )
    overall = max(float(inherited["max_abs_difference"]), random_max)
    if overall > 1e-12:
        raise ValueError(f"EXP-110 row independence failed: {overall}")
    result = {
        "experiment": "EXP-110",
        "zip": str(ZIP_PATH),
        "tolerance": 1e-12,
        "cases": {
            "batch": "passed",
            "singleton": "passed",
            "reverse": "passed",
            "random_permutation_seed_42": "passed",
            "split_batch": "passed",
            "duplicate_batch": "passed",
        },
        "inherited_full_audit": inherited,
        "random_permutation_max_abs_difference": random_max,
        "overall_max_abs_difference": overall,
        "canonical_test_or_sample_opened": False,
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
