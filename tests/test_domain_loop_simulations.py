from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.simfactory_tradeoff_replay import select_mixtures
from scripts.specialist_slot_search import compare_objective
from scripts.specialist_slots_pilot import SlotEpisode, finish, opportunity_values, upper_tail_mean
from src.bpisim_data import extract_bpi_case
from src.domain_loop_data import Attempt, Case, Event, chronological_split, extract_case, selected_domains
from src.domain_loop_simulations import generate
from src.domain_loop_simulator import OUTCOMES, LoopEnvironment, LoopLaw, calibrate, run_policy
from src.domain_loop_training import read_replay


class DomainLoopTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("configs/domain_loop_simulations.json").read_text())
        self.insurance = self.config["domains"]["insurance"]
        self.manufacturing = self.config["domains"]["manufacturing"]
        self.bpisim = self.config["domains"]["bpisim"]
        self.law = LoopLaw([Case("a", 0., 15., (Attempt("repeat", 7., 2., 4.), Attempt("resolved", 8., 2.))),
                            Case("b", 0., 9., (Attempt("negative", 9., 2.),))], 1, 1)

    def test_insurance_groups_adjacent_requests_and_keeps_response_marker(self):
        names = ["Document request", "Document request", "Damage documents", "Document request",
                 "Notification of payment", "Document request"]
        audit = Counter()
        case = extract_case("a", [Event(name, time, time) for time, name in enumerate(names)],
                            "insurance", self.insurance, audit)
        self.assertEqual([item.outcome for item in case.attempts], ["repeat", "resolved"])
        self.assertEqual(case.attempts[0].milestone, 2.)
        self.assertEqual(audit["markers_after_exit"], 1)
        self.assertEqual(audit["markers_merged_into_episode"], 1)

    def test_slot_rollout_fork_has_independent_calendar_and_exact_quota(self):
        env = LoopEnvironment(self.law, self.insurance, self.config, effect=.25, scenario=True)
        episode = SlotEpisode(env, ([0., 1., 2., 3.], 41), 2)
        episode.step(0)
        branch = episode.fork()
        np.testing.assert_array_equal(finish(branch, lambda _: 1), finish(episode.fork(), lambda _: 1))
        self.assertEqual((episode.seen, episode.slots), (2, 2))
        self.assertEqual(finish(episode, lambda _: 0)[-1], 2)

    def test_budget_price_uses_next_opportunity(self):
        price = opportunity_values(np.array([0., 2.]), 2, 1)
        np.testing.assert_allclose(price[:, 1], [0., 1., 1.5])

    def test_repeat_improvement_requires_queue_preservation(self):
        cfg = json.loads(Path("configs/bpisim_repeat_search.json").read_text())
        baseline = np.tile([.5, 100., 20., 4., .1, 40., .125], (4, 1))
        candidate = baseline.copy()
        candidate[:, 0] -= .02
        draws = np.array([[0, 1, 2, 3], [3, 0, 2, 0]])
        candidate[:, 5] = 41.
        self.assertTrue(compare_objective(candidate, baseline, baseline, cfg, draws, "repeat")["passed"])
        candidate[:, 5] = 43.
        result = compare_objective(candidate, baseline, baseline, cfg, draws, "repeat")
        self.assertTrue(result["objective_pass"])
        self.assertFalse(result["passed"])

    def test_tail_wait_is_mean_of_eight_longest_and_requires_repeat_preservation(self):
        self.assertEqual(upper_tail_mean(np.arange(80), .1), 75.5)
        cfg = json.loads(Path("configs/simfactory_tail_search.json").read_text())
        baseline = np.tile([.5, 100., 20., 4., .1, 40., .125, 100.], (4, 1))
        candidate = baseline.copy()
        candidate[:, 7] = 93.
        draws = np.array([[0, 1, 2, 3], [3, 0, 2, 0]])
        self.assertTrue(compare_objective(candidate, baseline, baseline, cfg, draws, "tail_queue")["passed"])
        candidate[:, 0] += .01
        self.assertFalse(compare_objective(candidate, baseline, baseline, cfg, draws, "tail_queue")["passed"])

    def test_simfactory_blends_use_same_validation_repeat_cap(self):
        cfg = json.loads(Path("configs/simfactory_tail_search.json").read_text())
        reference = dict(extra_attempts=.32, elapsed=100., work=20., specialist_work=4., unresolved=.10,
                         queue=40., slots=.125, tail_queue=100.)
        validation = {"FIFO": reference | {"slots": 0., "specialist_work": 0.}, "ML_tail_0.04": reference,
                      "RL_round_1": reference | {"extra_attempts": .33, "tail_queue": 80.}}
        frozen = {"seed": 17, "best_control": "ML_tail_0.04", "settings": cfg, "validation": validation}
        choice = select_mixtures(frozen, validation)
        self.assertEqual(choice["ml_mix"], {"ML_tail_0.04": 1.})
        self.assertAlmostEqual(choice["hybrid_mix"]["RL_round_1"], .5)
        self.assertAlmostEqual(choice["validation_hybrid"]["extra_attempts"], .325)
        self.assertAlmostEqual(choice["validation_hybrid"]["tail_queue"], 90.)

    def test_manufacturing_requires_inspection_between_repairs_and_uses_final_packing(self):
        events = [Event("Grinding Rework", 0, 2), Event("Packing", 3, 4), Event("Milling", 5, 6),
                  Event("Rework Milling", 7, 8), Event("Final Inspection Q.C.", 9, 10),
                  Event("Turning Rework", 11, 14), Event("Packing", 15, 16)]
        case = extract_case("a", events, "manufacturing", self.manufacturing, Counter())
        self.assertEqual(case.attempts, (Attempt("repeat", 11., 3., 10.), Attempt("resolved", 5., 3.)))

    def test_manufacturing_ignores_inspection_overlapping_repair(self):
        events = [Event("Grinding Rework", 0, 5), Event("Final Inspection Q.C.", 4, 6),
                  Event("Rework Milling", 7, 8), Event("Packing", 9, 10)]
        case = extract_case("a", events, "manufacturing", self.manufacturing, Counter())
        self.assertEqual(case.attempts, (Attempt("resolved", 10., 6.),))

    def test_unobserved_exit_is_not_a_resolution(self):
        event = Event("Document request", 2, 2)
        case = extract_case("a", [event], "insurance", self.insurance, Counter())
        self.assertEqual(case.attempts, (Attempt("unobserved", 0., 0.),))

    def test_chronological_split_purges_boundary_overlap(self):
        cases = [Case("a", 1, 2, ()), Case("b", 2, 5, ()), Case("c", 3, 6, ()), Case("d", 4, 5, ())]
        fit, test, boundary, excluded = chronological_split(cases, .5)
        self.assertEqual(([x.key for x in fit], [x.key for x in test], boundary, excluded), (["a"], ["c", "d"], 3, 1))

    def test_bpi_episode_excludes_preexisting_work_and_unions_overlapping_service(self):
        rows = [(0, "W_Validate application", "start"), (60, "A_Incomplete", "complete"),
                (90, "W_Validate application", "suspend"), (120, "W_Call incomplete files", "start"),
                (180, "A_Validating", "complete"), (180, "W_Validate application", "resume"),
                (240, "W_Call incomplete files", "complete"), (300, "W_Validate application", "complete"),
                (420, "A_Incomplete", "complete"), (480, "W_Call incomplete files", "start"),
                (540, "W_Call incomplete files", "complete"), (600, "A_Pending", "complete")]
        events = [{"concept:name": name, "time:timestamp": datetime.fromtimestamp(time, timezone.utc).isoformat(),
                   "lifecycle:transition": life} for time, name, life in rows]
        case = extract_bpi_case("a", events, 600, self.bpisim, Counter())
        self.assertEqual(case.attempts, (Attempt("repeat", 6., 3., 2., 1.), Attempt("resolved", 3., 1., None, 1.)))
        events[-1]["concept:name"] = "A_Denied"
        self.assertEqual(extract_bpi_case("a", events, 600, self.bpisim, Counter()).attempts[-1].outcome, "negative")
        self.assertEqual(extract_bpi_case("a", events[:-1], 600, self.bpisim, Counter()).attempts[-1].outcome, "unobserved")

    def test_bpi_future_work_does_not_reserve_reviewer_capacity(self):
        cases = [Case("a", 0, 14, (Attempt("resolved", 14., 2., 4., 10.),)),
                 Case("b", 1, 5, (Attempt("resolved", 4., 3., 1., 1.),))]
        law = LoopLaw(cases, 1, 1)
        env = LoopEnvironment(law, self.bpisim | {"standard_servers": 1}, self.config)
        env.reset([0., 1.], 42)
        env.draws[0, 0], env.draws[1, 0] = (.5, 0., 0.), (.5, .9, 0.)
        state, reward, done, completed = env.step("standard")
        self.assertEqual((state.case, state.standard_jobs, reward, done, completed), (1, 0, 0., False, ()))
        _, _, done, (early, late) = env.step("standard")
        self.assertTrue(done)
        self.assertEqual((early.case, early.service_ready, early.service_start, early.end), (1, 2., 2., 5.))
        self.assertEqual((late.case, late.milestone_time, late.service_ready, late.service_start), (0, 4., 10., 10.))
        self.assertEqual([row["queue_wait"] for row in env.case_rows()], [0., 0.])

    def test_bpi_routing_at_readiness_keeps_sampled_delays_and_reveals_current_queue(self):
        law = LoopLaw([Case("a", 0., 15., (Attempt("resolved", 15., 8., 4., 5.),))], 1, 1)
        spec = self.bpisim | {"standard_servers": 1}
        arrivals = [0., 1., 2.]
        original = LoopEnvironment(law, spec, self.config)
        deferred = LoopEnvironment(law, spec, self.config, route_when_ready=True)
        self.assertEqual(original.reset(arrivals, 42).time, 0.)
        self.assertEqual(deferred.reset(arrivals, 42).time, 5.)
        marker, *_ = original.step("standard")
        ready, *_ = deferred.step("standard")
        self.assertEqual((marker.case, marker.time, marker.standard_jobs), (1, 1., 0))
        self.assertEqual((ready.case, ready.time, ready.standard_jobs), (1, 6., 1))
        for env in (original, deferred):
            while env.observation is not None:
                env.step("standard")
        self.assertEqual(original.case_rows(), deferred.case_rows())
        self.assertEqual(original.event_log(), deferred.event_log())

    def test_bpi_is_selectable_without_changing_both_domain_selection(self):
        self.assertEqual([name for _, name, _ in selected_domains(self.config, "both")], ["insurance", "manufacturing"])
        self.assertEqual([name for _, name, _ in selected_domains(self.config, "all")], ["insurance", "manufacturing", "bpisim"])

    def test_simulator_names_keep_the_same_domain_and_seed_index(self):
        for display, domain in (("SimClaims", "insurance"), ("SimFactory", "manufacturing"), ("BPISim", "bpisim")):
            named = list(selected_domains(self.config, display))
            self.assertEqual(named, list(selected_domains(self.config, domain)))
            self.assertEqual(named[0][2]["name"], display)

    def test_fast_sampling_matches_reference_on_cutoffs_and_zero_mass_outcomes(self):
        for attempt in (1, 2, 3):
            baseline = self.law.probabilities[min(attempt, 2)]
            for override, shift in ((None, 0.), (None, -.75), (0., .75), (.2, .75)):
                probabilities = baseline
                if override is not None or shift:
                    p = baseline[0] if override is None else override
                    odds = np.exp(shift)
                    p = p * odds / (1 - p + p * odds)
                    exits = baseline[1:]
                    probabilities = np.r_[p, (1 - p) * exits / exits.sum()]
                np.testing.assert_allclose(self.law.probability(attempt, override, shift), probabilities, rtol=1e-15)
                # The sampler uses math.exp. Test exact cutoff ties against its weights.
                cutoffs = np.cumsum(self.law.probability(attempt, override, shift))
                for draw in (0., .25, .8, np.nextafter(1., 0.), *[x for x in cutoffs if x < 1.]):
                    index = min(np.searchsorted(cutoffs, draw, side="right"), len(OUTCOMES) - 1)
                    outcome, timing = self.law.sample(attempt, draw, 0., override, shift)
                    self.assertEqual(outcome, OUTCOMES[index])
                    self.assertEqual(timing, self.law.timing_pool(min(attempt, 2), outcome)[0])

    def test_calibration_rejects_missing_exits_and_handles_absent_second_stage(self):
        with self.assertRaisesRegex(ValueError, "observed exit"):
            LoopLaw([], 1, 1)
        cases = [Case("a", 0., 5., (Attempt("resolved", 5., 2.),))]
        law = LoopLaw(cases, 1, 0)
        self.assertEqual(law.sample(2, .9, .5)[0], "resolved")

    def test_online_step_hides_unfinished_outcome_and_conserves_wait(self):
        env = LoopEnvironment(self.law, self.insurance, self.config, effect=1.)
        state = env.reset([0., 0.], 42)
        env.draws[:, :, :] = (0., 0., 0.)
        self.assertEqual(state.case, 0)
        state, reward, done, completed = env.step("specialist")
        self.assertEqual((state.case, state.specialist_jobs, reward, done, completed), (1, 1, 0., False, ()))
        _, reward, done, completed = env.step("specialist")
        first, second = completed
        self.assertTrue(done)
        self.assertEqual(first.outcome, "resolved")
        self.assertEqual(second.outcome, "resolved")
        self.assertEqual(first.service_end, second.service_start)
        self.assertAlmostEqual(reward, env.reward(first) + env.reward(second))
        self.assertAlmostEqual(env.case_rows()[1]["queue_wait"], first.service_end)
        self.assertTrue(all(row.service_end <= row.milestone_time <= row.end for row in completed))

    def test_no_effect_preserves_branch_outcomes_with_common_randomness(self):
        for spec in (self.insurance, self.manufacturing):
            env = LoopEnvironment(self.law, spec, self.config, scenario=True)
            arrivals = np.arange(100) * .25
            standard = run_policy(env, arrivals, 345, lambda _: "standard")
            specialist = run_policy(env, arrivals, 345, lambda _: "specialist")
            self.assertEqual([(row["attempts"], row["outcome"]) for row in standard],
                             [(row["attempts"], row["outcome"]) for row in specialist])
            self.assertGreater(sum(row["work"] for row in specialist), sum(row["work"] for row in standard))

    def test_increased_scenario_difficulty_raises_repeat_probability(self):
        easier = self.law.probability(1, .2, -1.)
        harder = self.law.probability(1, .2, 1.)
        self.assertAlmostEqual(easier.sum(), 1.)
        self.assertAlmostEqual(harder.sum(), 1.)
        self.assertLess(easier[0], harder[0])

    def test_manufacturing_sparse_repeat_timing_keeps_reinspection(self):
        cases = [Case("a", 0, 12, (Attempt("repeat", 5., 2., 4.), Attempt("resolved", 7., 1.))),
                 Case("b", 1, 8, (Attempt("resolved", 7., 1.),))]
        law = calibrate(cases, self.manufacturing, self.config)
        env = LoopEnvironment(law, self.manufacturing, self.config, scenario=True)
        run_policy(env, np.arange(100), 74, lambda _: "standard")
        repeats = [row for row in env.records if row.outcome == "repeat"]
        self.assertTrue(repeats)
        self.assertTrue(all(row.service_end <= row.milestone_time <= row.end for row in repeats))

    def test_cap_and_nonrepeat_negative_outcomes_are_distinct(self):
        env = LoopEnvironment(self.law, self.insurance, self.config | {"maximum_attempts": 1})
        env.reset([0.], 42)
        env.draws[0, 0] = (0., 0., 0.)
        self.assertEqual(env.step("standard")[3][0].outcome, "cap")
        env = LoopEnvironment(self.law, self.insurance, self.config, effect=1.)
        env.reset([0.], 42)
        env.draws[0, 0] = (.999, 0., 0.)
        self.assertEqual(env.step("specialist")[3][0].outcome, "negative")

    def test_reproducible_logs_are_trainable_and_timestamped(self):
        with TemporaryDirectory() as temp:
            first, second = (Path(temp) / name for name in ("first", "second"))
            for target in (first, second):
                target.mkdir()
                generate(self.law, self.insurance, self.config, np.random.SeedSequence(3), target, .6, .25, 4, 12, .5)
            for name in ("transitions.jsonl", "events.jsonl"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            replay = read_replay(first / "transitions.jsonl")
            self.assertEqual(Counter(row["done"] for row, _ in replay)[True], 4)
            self.assertEqual({row["action"] for row, _ in replay}, {"standard", "specialist"})
            self.assertTrue(all(row["propensity"] == .5 for row, _ in replay))


if __name__ == "__main__":
    unittest.main()
