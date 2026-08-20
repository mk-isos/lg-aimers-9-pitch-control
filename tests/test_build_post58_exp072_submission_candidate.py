from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from build_post58_exp072_submission_candidate import (
    ADDITIVE_WEIGHT,
    CANDIDATE,
    PACKAGE_DYNAMIC_HELPER,
    PREDICTION_SEASON,
    SOURCE_SEASONS,
    build_dynamic_state,
    dynamic_ar_correction,
    full_prior_career_state,
    load_rendered_module,
    render_submission_script,
    synthetic_invariance_smoke,
)
from train_exp072_dynamic_pitcher_state import (
    dynamic_deltas,
    prior_career_states,
    season_latent_states,
)


def synthetic_training_frame() -> pd.DataFrame:
    outcomes = {
        10: [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
        20: [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
        # Pitcher 30 disappears after 2021; the full state must retain it.
        30: [1, 1, 0, 1, 0, 1, 1, 0, 0],
    }
    appearances = {
        2019: (10, 20, 30),
        2020: (10, 20, 30),
        2021: (10, 20, 30),
        2022: (10, 20),
        2023: (10, 20),
        2024: (10, 20),
    }
    positions = {pitcher_id: 0 for pitcher_id in outcomes}
    career_n = {pitcher_id: 0 for pitcher_id in outcomes}
    career_successes = {pitcher_id: 0 for pitcher_id in outcomes}
    records: list[dict[str, object]] = []
    for season, pitchers in appearances.items():
        for local_pitch in range(3):
            for pitcher_id in pitchers:
                n = career_n[pitcher_id]
                successes = career_successes[pitcher_id]
                outcome = outcomes[pitcher_id][positions[pitcher_id]]
                records.append(
                    {
                        "season": season,
                        "pitcher_id": pitcher_id,
                        "batter_id": 1000 + (pitcher_id + local_pitch) % 7,
                        "asof_pitcher_n": n,
                        "asof_pitcher_success_rate": (
                            successes / n if n > 0 else 0.0
                        ),
                        "asof_batter_n": season - 2019 + local_pitch,
                        "control_success": outcome,
                    }
                )
                positions[pitcher_id] += 1
                career_n[pitcher_id] += 1
                career_successes[pitcher_id] += outcome
    return pd.DataFrame.from_records(records)


class FullStateTest(unittest.TestCase):
    def test_full_state_freezes_requested_2025_values(self) -> None:
        frame = synthetic_training_frame()
        state, audit = build_dynamic_state(frame)
        self.assertEqual(state["candidate"], CANDIDATE)
        self.assertEqual(state["prediction_season"], PREDICTION_SEASON)
        self.assertEqual(state["source_seasons"], list(SOURCE_SEASONS))
        self.assertEqual(state["league_prior_season"], 2024)
        self.assertEqual(state["league_prior"], state["league_2024"])
        self.assertEqual(state["current_season_prior_strength"], 30.0)
        self.assertEqual(state["additive_delta_weight"], ADDITIVE_WEIGHT)
        self.assertGreaterEqual(state["rho"], 0.0)
        self.assertLessEqual(state["rho"], 1.0)
        self.assertEqual(audit["prediction_season"], 2025)
        prior = {
            int(row["pitcher_id"]): row for row in state["prior_career_states"]
        }
        self.assertIn(30, prior)
        self.assertEqual(prior[30]["prior_n"], 9.0)
        self.assertEqual(prior[30]["prior_successes"], 5.0)
        # JSON production serialization must reject neither NaN nor Infinity.
        json.dumps(state, allow_nan=False)

    def test_full_prior_career_state_retains_absent_pitcher(self) -> None:
        frame = synthetic_training_frame()
        state = full_prior_career_state(frame)
        self.assertEqual(float(state.n.loc[30]), 9.0)
        self.assertEqual(float(state.successes.loc[30]), 5.0)
        self.assertEqual(float(state.n.loc[10]), 18.0)


class OriginalParityTest(unittest.TestCase):
    def test_cutoff_2023_serializer_and_package_match_original_2024_delta(self) -> None:
        full = synthetic_training_frame()
        cutoff = full.loc[full["season"] <= 2023].copy()
        validation = full.loc[full["season"] == 2024].reset_index(drop=True)
        states, league_rates = season_latent_states(cutoff)
        career = prior_career_states(full)[2024]
        original, _ = dynamic_deltas(
            validation,
            2024,
            states,
            league_rates,
            career,
        )
        serialized, _ = build_dynamic_state(
            cutoff,
            prediction_season=2024,
            expected_source_seasons=tuple(range(2019, 2024)),
        )
        reference = original["ar_k30"]
        np.testing.assert_allclose(
            dynamic_ar_correction(validation, serialized),
            reference,
            rtol=0.0,
            atol=1e-15,
        )

        base_source = (
            ROOT / "submissions" / "EXP-051-TMDIRECT" / "script.py"
        ).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            script_path = Path(directory) / "script.py"
            script_path.write_text(
                render_submission_script(base_source), encoding="utf-8"
            )
            module = load_rendered_module(script_path)
            np.testing.assert_allclose(
                module.map_dynamic_pitcher_ar(validation, serialized),
                reference,
                rtol=0.0,
                atol=1e-15,
            )
        expected_candidate = np.clip(
            np.full(len(reference), 0.5) + 0.50 * reference, 0.0, 1.0
        )
        self.assertTrue(np.isfinite(expected_candidate).all())


class RenderAndInvarianceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_source = (
            ROOT / "submissions" / "EXP-051-TMDIRECT" / "script.py"
        ).read_text(encoding="utf-8")

    def test_rendered_script_has_frozen_branch_and_no_aggregate_log(self) -> None:
        rendered = render_submission_script(self.base_source)
        compile(rendered, "EXP-072-DYNAMIC-AR/script.py", "exec")
        self.assertIn('elif candidate == "ar_k30_w050":', rendered)
        self.assertIn(PACKAGE_DYNAMIC_HELPER, rendered)
        self.assertNotIn("predictions.mean()", rendered)
        self.assertNotIn("predictions.min()", rendered)
        self.assertNotIn("predictions.max()", rendered)
        self.assertNotIn("rows={len(sample)}", rendered)
        self.assertIn("exp051_prediction + additive_weight * dynamic_correction", rendered)

    def test_dynamic_helper_is_singleton_permutation_split_duplicate_invariant(self) -> None:
        state, _ = build_dynamic_state(synthetic_training_frame())
        with tempfile.TemporaryDirectory() as directory:
            script_path = Path(directory) / "script.py"
            script_path.write_text(
                render_submission_script(self.base_source), encoding="utf-8"
            )
            smoke = synthetic_invariance_smoke(script_path, state)
        self.assertEqual(
            smoke["singleton_reverse_split_duplicate_invariance"], "passed"
        )
        self.assertFalse(smoke["query_row_aggregation"])
        self.assertFalse(smoke["test_or_sample_file_opened"])
        self.assertEqual(
            max(smoke["maximum_absolute_differences"].values()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
