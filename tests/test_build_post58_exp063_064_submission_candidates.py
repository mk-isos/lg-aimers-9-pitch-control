from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
MODULE_PATH = (
    ROOT / "experiments" / "build_post58_exp063_064_submission_candidates.py"
)
SPEC = importlib.util.spec_from_file_location("post58_builder", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class Post58BuilderUnitTest(unittest.TestCase):
    def test_source_only_fixture_uses_unseen_2025_entities_and_no_target(self) -> None:
        raw = pd.DataFrame(
            {
                "row_id": ["source-a", "source-b"],
                "season": [2019, 2019],
                "pitcher_id": [1, 2],
                "batter_id": [3, 4],
                "pitcher_team_id": [5, 6],
                "batter_team_id": [7, 8],
                "asof_pitcher_n": [10, 20],
                "asof_batter_n": [30, 40],
                "asof_pitcher_pitchmix_n": [10, 20],
                "asof_pitcher_success_rate": [0.5, np.nan],
                "asof_batter_success_rate": [0.4, 0.6],
                "control_success": [0, 1],
            }
        )
        fixture = builder.source_only_fixture(raw, rows=2)
        self.assertNotIn(builder.TARGET_COL, fixture)
        self.assertTrue(fixture["season"].eq(2025).all())
        self.assertTrue(fixture["asof_pitcher_n"].eq(0).all())
        self.assertTrue(fixture["asof_batter_n"].eq(0).all())
        self.assertTrue(fixture["asof_pitcher_pitchmix_n"].eq(0).all())
        self.assertGreaterEqual(int(fixture["pitcher_id"].min()), 9_100_000)
        self.assertGreaterEqual(int(fixture["batter_id"].min()), 9_200_000)
        self.assertTrue(fixture["asof_pitcher_success_rate"].notna().all())

    def test_stable_runner_mapping_is_row_local_and_uses_exp051_pbin(self) -> None:
        rows = pd.DataFrame(
            {
                "balls_before": [0, 1, 0],
                "strikes_before": [0, 1, 0],
                "pitcher_hand": [0, 0, 0],
                "batter_hand": [1, 1, 1],
                "num_runners_on": [0, 2, 0],
            }
        )
        base = np.asarray([0.40, 0.50, 0.40])
        state = {
            "records": [
                {"count": 0, "runners": 0, "pbin": 2, "effect": 0.012},
                {"count": 5, "runners": 2, "pbin": 6, "effect": -0.009},
            ]
        }
        correction = builder.map_stable_runner_state(rows, base, state)
        np.testing.assert_allclose(correction, [0.012, -0.009, 0.012])
        reverse = builder.map_stable_runner_state(
            rows.iloc[::-1].reset_index(drop=True), base[::-1], state
        )
        np.testing.assert_allclose(reverse[::-1], correction)

    def test_inference_patch_preserves_base_dispatch_and_suppresses_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "package"
            destination.mkdir()
            shutil.copy2(builder.FROZEN_BASE / "script.py", destination / "script.py")
            builder.patch_inference_script(destination, "close060_last_w025")
            script = (destination / "script.py").read_text(encoding="utf-8")
            self.assertIn('candidate = str(metadata["base_candidate"])', script)
            self.assertIn("_post58_apply_candidate", script)
            self.assertNotIn("predictions.mean()", script)
            self.assertNotIn("predictions.min()", script)
            self.assertNotIn("predictions.max()", script)
            self.assertNotIn('f"rows={len(sample)}"', script)
            self.assertNotIn("| rows=", script)
            compile(script, str(destination / "script.py"), "exec")


class GeneratedPackageAuditTest(unittest.TestCase):
    def test_generated_manifest_records_historical_parity_and_invariance(self) -> None:
        manifest_path = builder.MANIFEST_PATH
        if not manifest_path.is_file():
            self.skipTest("builder has not generated the package manifest yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["canonical_data_test_csv_opened"])
        self.assertFalse(manifest["canonical_sample_submission_csv_opened"])
        self.assertEqual(len(manifest["candidates"]), 2)
        for candidate in manifest["candidates"]:
            parity = candidate["build"][
                "saved_2024_cutoff2023_reconstruction_parity"
            ]
            self.assertEqual(parity["status"], "passed")
            self.assertLessEqual(parity["max_abs_difference"], parity["tolerance"])
            smoke = candidate["source_only_smoke_and_row_invariance"]
            self.assertEqual(
                smoke["full_reverse_singleton_split_duplicate"], "passed"
            )
            self.assertTrue(smoke["script_prediction_summary_stats_suppressed"])
            self.assertLessEqual(smoke["max_abs_difference"], smoke["tolerance"])
            self.assertEqual(candidate["zip"]["crc"], "passed")

    def test_generated_metadata_marks_candidates_exploratory_gate_failed(self) -> None:
        for destination in (builder.EXP063_DESTINATION, builder.EXP064_DESTINATION):
            metadata_path = destination / "model" / "metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["exploratory"])
            self.assertFalse(metadata["validation_gate_passed"])
            self.assertFalse(metadata["canonical_test_or_sample_opened_during_build"])
            parity = metadata["saved_2024_cutoff2023_reconstruction_parity"]
            self.assertEqual(parity["status"], "passed")


if __name__ == "__main__":
    unittest.main()
