import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.regression_routing_ablations import (
    FIELDS,
    detailed_metrics,
    masked_actor,
    mixture_weight,
    repeat_upper,
    weighted_rows,
)
from scripts.run_regression_ablations import validate_config
from scripts.simbank_loop_ablation import ARMS
from src.benchmark_models import seeds


ROOT = Path(__file__).resolve().parents[1]


class RegressionAblationTests(unittest.TestCase):
    def test_config_generates_three_seeds_from_master(self):
        config = json.loads((ROOT / "configs/regression_ablation_suite.json").read_text())
        validate_config(config, check_sources=False)
        generated = seeds(config["master_seed"], config["replicates"])
        self.assertEqual(len(generated), 3)
        self.assertEqual(len(set(generated)), 3)
        self.assertNotIn("seeds", config)

    def test_mixture_is_exact_convex_combination(self):
        reference = np.array([[1.0, 4.0], [3.0, 8.0]])
        candidate = np.array([[5.0, 0.0], [7.0, 4.0]])
        np.testing.assert_allclose(weighted_rows(reference, candidate, 0.25),
                                   np.array([[2.0, 3.0], [4.0, 7.0]]))
        np.testing.assert_allclose(weighted_rows(reference, candidate, 0.0), reference)
        np.testing.assert_allclose(weighted_rows(reference, candidate, 1.0), candidate)

    def test_repeat_upper_and_safety_weight_are_conservative(self):
        reference = np.zeros((4, len(FIELDS)))
        candidate = reference.copy()
        candidate[:, 0] = [0.0, 0.004, 0.008, 0.012]
        upper = repeat_upper(candidate, reference)
        self.assertGreater(upper, candidate[:, 0].mean())
        weight = mixture_weight(upper, 0.005, 0.75)
        self.assertGreaterEqual(weight, 0.0)
        self.assertLessEqual(weight, 1.0)
        self.assertLessEqual(weight * upper, 0.75 * 0.005 + 1e-12)

    def test_static_mask_removes_every_residual_state_term(self):
        context = lambda _ep: (0.1, 0.0, 0.0, 0.7, 0.2, 0.0, 0.0,
                               3.0, 1.0, 2, 0.5, 3, 8, 1, 0.0)
        episode = SimpleNamespace(
            state=SimpleNamespace(case=0), size=10, slots=3, seen=2, closed=1,
            accrued_wait=np.zeros(10), env=SimpleNamespace(law=SimpleNamespace(mean_service=1.0)))
        weights = np.ones(11)
        actor = masked_actor(context, weights, 3, True, [1] + [0] * 10)
        self.assertEqual(actor(episode), 1)
        actor = masked_actor(context, -weights, 3, True, [1] + [0] * 10)
        self.assertEqual(actor(episode), 0)

    def test_reward_factorial_contains_exact_two_by_two_design(self):
        self.assertEqual(ARMS, {"profit": (0, 0), "cancel": (0, 1),
                                "refuse": (1, 0), "both": (1, 1)})

    def test_routing_evaluator_passes_slot_episode_to_actor(self):
        seen = []

        class FakeEpisode:
            state = object()
            ep = SimpleNamespace(env=SimpleNamespace(case_rows=lambda: [{
                "attempts": 1, "elapsed": 1.0, "work": 1.0, "outcome": "resolved",
                "queue_wait": 0.0, "pre_service_wait": 0.0, "specialist_attempts": 0}],
                records=[]), result=lambda: np.zeros(7))

            def step(self, action):
                seen.append(action)
                self.state = None

        detailed_metrics(lambda _workload: FakeEpisode(), [(np.array([0.0]), np.array([0]))],
                         {"policy": lambda episode: int(episode is FakeEpisode.ep)}, 1)
        self.assertEqual(seen, [1])


if __name__ == "__main__":
    unittest.main()
