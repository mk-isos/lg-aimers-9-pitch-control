from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from build_post58_exp070_071_submission_candidates import (
    INFERENCE_LOOKUP_HELPER,
    LOOKUP_FILENAME,
    PARITY_TOLERANCE,
    SPECS,
    apply_frozen_lookup,
    build_zip,
    freeze_scored_history,
    guard_canonical_competition_inputs,
    parity_against_saved,
    patched_inference_script,
)


def state(kind: str) -> dict[str, object]:
    return {
        "history_cutoff_season": 2024,
        "correction_clip": 0.03,
        "lookup_kind": kind,
        "pitcher_mapping": {"10": 100},
        "pitcher_fallback": [
            {"pitcher_trackman_id": 100, "value": 0.48, "count": 20}
        ],
        "context_lookup": [
            {
                "pitcher_trackman_id": 100,
                "count_index": 2,
                "batter_hand_code": 1,
                "value": 0.55 if kind == "expected_control" else 0.02,
                "count": 5,
            }
        ],
    }


class FrozenLookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = pd.DataFrame(
            {
                "pitcher_id": [10, 10, 999],
                "count_index": [2, 7, 2],
                "batter_hand": [1, 2, 1],
                "asof_pitcher_success_rate": [0.50, 0.50, 0.50],
            }
        )

    def test_exp070_context_fallback_clip_and_unmapped_zero(self) -> None:
        observed = apply_frozen_lookup(self.rows, state("expected_control"))
        np.testing.assert_allclose(observed, [0.03, -0.02, 0.0], atol=1e-15)

    def test_exp071_context_fallback_and_unmapped_zero(self) -> None:
        residual = state("predicted_residual")
        residual["pitcher_fallback"][0]["value"] = -0.012
        observed = apply_frozen_lookup(self.rows, residual)
        np.testing.assert_allclose(observed, [0.02, -0.012, 0.0], atol=1e-15)

    def test_generated_helper_matches_builder_and_is_row_independent(self) -> None:
        namespace = {"np": np, "pd": pd}
        exec(INFERENCE_LOOKUP_HELPER, namespace)
        deployed = namespace["map_post58_playerphysics_lookup"]
        lookup = state("expected_control")
        expected = apply_frozen_lookup(self.rows, lookup)
        np.testing.assert_array_equal(deployed(self.rows, lookup), expected)
        reverse = deployed(self.rows.iloc[::-1].reset_index(drop=True), lookup)[::-1]
        np.testing.assert_array_equal(reverse, expected)
        singleton = np.concatenate(
            [deployed(self.rows.iloc[[index]], lookup) for index in range(3)]
        )
        np.testing.assert_array_equal(singleton, expected)
        duplicate = deployed(self.rows.iloc[[0, 1, 0]].reset_index(drop=True), lookup)
        self.assertEqual(duplicate[0], duplicate[2])


class FrozenAggregationTest(unittest.TestCase):
    def test_history_is_smoothed_by_pitcher_context_and_serialized(self) -> None:
        spec = SPECS[0]
        history = pd.DataFrame(
            {
                "season": [2023, 2023, 2024],
                "pitcher_trackman_id": [100, 100, 100],
                "count_index": [2, 2, 7],
                "batter_hand_code": [1, 1, 2],
                "predicted_control": [0.4, 0.6, 0.8],
            }
        )
        frozen = freeze_scored_history(
            history,
            {10: 100},
            spec=spec,
            cutoff=2024,
            value_column="predicted_control",
            fit_audit={"synthetic": True},
        )
        self.assertEqual(frozen["pitcher_mapping"], {"10": 100})
        fallback = frozen["pitcher_fallback"][0]
        self.assertAlmostEqual(fallback["value"], 0.6)
        context = frozen["context_lookup"][0]
        expected = (1.0 + 100.0 * 0.6) / 102.0
        self.assertAlmostEqual(context["value"], expected)
        self.assertFalse(
            frozen["deployment"]["post58_lightgbm_booster_required"]
        )

    def test_history_after_cutoff_is_rejected(self) -> None:
        history = pd.DataFrame(
            {
                "season": [2025],
                "pitcher_trackman_id": [100],
                "count_index": [2],
                "batter_hand_code": [1],
                "predicted_control": [0.5],
            }
        )
        with self.assertRaisesRegex(ValueError, "exceeds cutoff"):
            freeze_scored_history(
                history,
                {10: 100},
                spec=SPECS[0],
                cutoff=2024,
                value_column="predicted_control",
                fit_audit={},
            )


class ScriptAndZipTest(unittest.TestCase):
    def test_both_scripts_are_patched_to_frozen_row_local_lookup(self) -> None:
        source = (ROOT / "submissions" / "EXP-051-TMDIRECT" / "script.py").read_text(
            encoding="utf-8"
        )
        for spec in SPECS:
            script = patched_inference_script(source, spec)
            self.assertIn(spec.candidate, script)
            self.assertIn("map_post58_playerphysics_lookup", script)
            self.assertIn("SUBMISSION_TEST_PATH", script)
            self.assertIn("SUBMISSION_SAMPLE_PATH", script)
            self.assertIn(LOOKUP_FILENAME, script)
            self.assertNotIn("predictions.mean()", script)
            self.assertNotIn("test.groupby(", script)
            compile(script, f"{spec.experiment}-script.py", "exec")

    def test_zip_crc_root_order_and_forbidden_artifact_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            model = package / "model"
            model.mkdir(parents=True)
            (package / "script.py").write_text("print('ok')\n", encoding="utf-8")
            (package / "requirements.txt").write_text(
                "lightgbm==4.6.0\n", encoding="utf-8"
            )
            (model / LOOKUP_FILENAME).write_text("{}\n", encoding="utf-8")
            output = root / "candidate.zip"
            audit = build_zip(package, output)
            self.assertEqual(audit["crc"], "passed")
            self.assertEqual(
                audit["root_order"], ["script.py", "requirements.txt"]
            )
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())

    def test_input_guard_blocks_configured_forbidden_path_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            forbidden = Path(directory) / "forbidden.csv"
            with patch(
                "build_post58_exp070_071_submission_candidates."
                "CANONICAL_TEST_PATH",
                forbidden.resolve(),
            ):
                with guard_canonical_competition_inputs() as audit:
                    with self.assertRaisesRegex(RuntimeError, "forbidden"):
                        pd.read_csv(forbidden)
                self.assertEqual(audit["blocked_attempts"], 1)


class ParityTest(unittest.TestCase):
    def test_synthetic_saved_prediction_and_correction_parity(self) -> None:
        spec = SPECS[0]
        rows = pd.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "pitcher_id": [10, 10, 999],
                "count_index": [2, 7, 2],
                "batter_hand": [1, 2, 1],
                "asof_pitcher_success_rate": [0.50, 0.50, 0.50],
            }
        )
        lookup = state("expected_control")
        base = np.array([0.45, 0.55, 0.60])
        correction = apply_frozen_lookup(rows, lookup)
        saved = np.clip(base + spec.correction_weight * correction, 0.0, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.npy"
            np.save(path, saved)
            synthetic_spec = type(spec)(
                experiment=spec.experiment,
                candidate=spec.candidate,
                destination_name=spec.destination_name,
                zip_name=spec.zip_name,
                lookup_kind=spec.lookup_kind,
                correction_weight=spec.correction_weight,
                target_description=spec.target_description,
                validation_artifact=spec.validation_artifact,
                saved_prediction=path,
            )
            audit = parity_against_saved(synthetic_spec, lookup, rows, base)
        self.assertTrue(audit["passed"])
        self.assertLessEqual(
            audit["prediction_max_abs_difference"], PARITY_TOLERANCE
        )
        self.assertLessEqual(
            audit["correction_max_abs_difference"], PARITY_TOLERANCE
        )


if __name__ == "__main__":
    unittest.main()
