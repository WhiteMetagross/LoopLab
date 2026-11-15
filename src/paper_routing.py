"""Paper comparison for the three specialist routing environments."""

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error

from scripts.audit_routing_rewards import feasible, ml_plan
from scripts.specialist_slots_pilot import SlotEpisode, baseline, ml_actor, upper_tail_mean
from src.benchmark_models import FAMILIES, estimator, predict
from src.domain_loop_data import acquire, chronological_split, load_cases, selected_domains
from src.domain_loop_simulations import generate, metrics, save
from src.domain_loop_simulator import LoopEnvironment, Observation, calibrate
from src.domain_loop_training import MASK, ml_features, read_replay, representation, scenarios
from src.simulator_benchmark import fit_policies


FIELDS = ("extra_attempts", "elapsed", "work", "specialist_work", "unresolved", "queue", "slots",
          "repeat_rate", "p90_elapsed", "negative", "unobserved", "cap", "tail_queue")
OUTCOMES = ("repeat", "resolved", "unresolved")


def prediction_rows(replay, bins, history):
    inputs, targets = [], []
    for row, event in replay:
        stage, difficulty, action = ml_features(Observation(**row["state"]), row["action"], bins)
        inputs.append((stage, difficulty, action) if history else (difficulty, action))
        outcome = event["outcome"]
        work = event["service_end"] - event["service_start"]
        nonqueue_elapsed = event["end"] - event["ready"] - (event["service_start"] - event["service_ready"])
        targets.append((OUTCOMES.index(outcome) if outcome in OUTCOMES else 2, work, nonqueue_elapsed))
    return np.asarray(inputs), np.asarray(targets)


def fit_forecasts(fit, calibration, test, settings, seed, threads, families=FAMILIES):
    bins = settings["difficulty_bins"]
    grids, checkpoints, reports = {}, {}, {}
    grid = np.array([(stage, difficulty, action) for stage in (1, 2) for difficulty in range(bins) for action in range(2)])
    for history in (True, False):
        name = "history" if history else "without_attempt"
        x, y = prediction_rows(fit, bins, history)
        xcal, ycal = prediction_rows(calibration, bins, history)
        xt, yt = prediction_rows(test, bins, history)
        classes, labels = np.unique(y[:, 0].astype(int), return_inverse=True)
        reports[name] = {}
        for family in families:
            def fit_model(target, data):
                return estimator(family, target, seed, settings["trees"], threads).fit(x, data)

            classifier = fit_model("outcome", labels)
            work, elapsed = fit_model("effort", y[:, 1]), fit_model("elapsed", y[:, 2])

            def probabilities(features, temperature):
                raw = np.zeros((len(features), len(OUTCOMES)))
                raw[:, classes] = predict(classifier, features, True)
                calibrated = np.power(np.clip(raw, 1e-9, 1), 1 / temperature)
                return calibrated / calibrated.sum(axis=1, keepdims=True)

            temperature = min(settings["temperatures"], key=lambda t: -np.log(
                probabilities(xcal, t)[np.arange(len(ycal)), ycal[:, 0].astype(int)]).mean())
            test_probs = probabilities(xt, temperature)
            target = yt[:, 0] == 0
            reports[name][family] = {"repeat_brier": float(brier_score_loss(target, test_probs[:, 0])),
                                     "repeat_ap": float(average_precision_score(target, test_probs[:, 0])) if target.any() else None,
                                     "multiclass_logloss": float(-np.log(test_probs[np.arange(len(yt)), yt[:, 0].astype(int)]).mean()),
                                     "work_mae": float(mean_absolute_error(yt[:, 1], predict(work, xt).clip(0))),
                                     "elapsed_mae": float(mean_absolute_error(yt[:, 2], predict(elapsed, xt).clip(0))),
                                     "temperature": temperature}
            checkpoints[name, family] = (classifier, work, elapsed, classes, temperature)
            if history:
                p = probabilities(grid, temperature)
                grids[family] = np.c_[p[:, 0], predict(elapsed, grid).clip(0), predict(work, grid).clip(0), p[:, 2]].reshape(2, bins, 2, 4)
        if history:
            truth = yt[:, 0] == 0
            pooled = np.bincount(y[:, 0].astype(int), minlength=3) / len(y)
            lookup = {(a, b): y[(x[:, 0] == a) & (x[:, 2] == b)] for a in (1, 2) for b in (0, 1)}
            buckets = [lookup[a, b] for a, _, b in xt]
            expected = np.array([(sum(values[:, 0] == 0) + 4 * pooled[0]) / (len(values) + 4) for values in buckets])
            reports[name]["stage_action_lookup"] = {"repeat_brier": float(brier_score_loss(truth, expected)),
                "repeat_ap": float(average_precision_score(truth, expected)) if truth.any() else None,
                "work_mae": float(mean_absolute_error(yt[:, 1], [values[:, 1].mean() if len(values) else y[:, 1].mean() for values in buckets])),
                "elapsed_mae": float(mean_absolute_error(yt[:, 2], [values[:, 2].mean() if len(values) else y[:, 2].mean() for values in buckets]))}
    return grids, reports, checkpoints


def event_cost(row, weights):
    work = row.service_end - row.service_start
    return (weights[0] * (row.outcome == "repeat") + weights[1] * (row.end - row.ready) + weights[2] * work
            + weights[3] * work * (row.action == "specialist") + weights[4] * (row.outcome not in ("repeat", "resolved")))


class Routing:
    def __init__(self, env, workload, quota, grid, weights, first_only=False):
        self.ep = SlotEpisode(env, workload, quota or 0, first_only=first_only)
        # Residual controllers use accrued wait as an observable queue state.
        # Keep the standard result schema by tracking it without enabling tail output.
        self.ep.accrued_wait = np.zeros(self.ep.size)
        self.unlimited, self.grid, self.weights = quota is None, grid, weights
        self.quota = quota
        self.risk_grid = grid[..., [0, 2]] if grid is not None else None
        self._advance_forced()

    @property
    def state(self):
        return self.ep.state

    def _advance_forced(self):
        while self.state is not None and not self.unlimited:
            action = self.ep.forced()
            if action is None:
                return
            self.ep.step(action)

    def observe(self):
        grid = self.grid
        base = representation(self.state, self.risk_grid, self.ep.env)
        stage = min(2, self.state.attempt) - 1
        difficulty = min(grid.shape[1] - 1, int(self.state.difficulty_signal * grid.shape[1]))
        extra = grid[stage, difficulty, :, [1, 3]].T.copy()
        extra[:, 0] /= self.ep.env.law.time_scale
        progress = (self.ep.slots / max(1, self.quota or 0), 1 - self.ep.seen / self.ep.size,
                    (self.ep.seen - self.ep.closed) / self.ep.size)
        return np.column_stack((base, extra, np.tile(progress, (2, 1)))), MASK

    def step(self, action):
        n = len(self.ep.env.records)
        if self.unlimited:
            self.ep.slots += action
        self.ep.step(action)
        self._advance_forced()
        return -sum(event_cost(row, self.weights) for row in self.ep.env.records[n:]), self.state is None

    def audit(self, reward):
        values = self.ep.result()
        expected = -np.dot(values[[0, 1, 2, 3, 4]], self.weights)
        np.testing.assert_allclose(reward, expected, atol=1e-8)
        assert self.unlimited or values[6] == self.quota


def policy_metrics(factory, workloads, policies, n, no_effect=False):
    result = {}
    expected = None
    for name, (policy, pool) in policies.items():
        values, signature = [], []
        for workload in workloads:
            env = factory(workload, pool)
            while env.state is not None:
                env.step(policy(env))
            cases = env.ep.env.case_rows()
            if no_effect:
                signature.append(tuple((row["attempts"], row["outcome"]) for row in cases))
            raw = env.ep.result()
            summary = metrics(cases)
            values.append([*(raw / n), summary["repeat_rate"], summary["p90_elapsed_minutes"],
                           summary["negative_exit_rate"], summary["unobserved_exit_rate"], summary["cap_rate"],
                           upper_tail_mean([row["queue_wait"] for row in cases], .1)])
            if pool == "quota" and raw[6] != env.quota:
                raise ValueError("Quota policy did not use the exact number of slots")
        if no_effect:
            if expected is None:
                expected = signature
            elif signature != expected:
                raise ValueError("Routing changed a zero-effect case outcome")
        result[name] = np.asarray(values)
    return result


def run(out, sim, settings, domain, seed, namespace, threads, torch, progress):
    index, _, spec = next(iter(selected_domains(sim, domain)))
    path = acquire(spec, sim["maximum_download_bytes"])
    cases, provenance = load_cases(path, domain, spec)
    fit, historical, boundary, dropped = chronological_split(cases, sim["fit_fraction"])
    law = calibrate(fit, spec, sim)
    save(out / "data.json", {"source": str(path), "sha256": spec["sha256"], "fit_cases": len(fit),
                             "heldout_cases": len(historical), "boundary": boundary, "overlap_purged": dropped,
                             "calibration": law.summary(), "extraction": dict(provenance)})
    fit_seed, cal_seed, pred_seed, scale_seed, off_seed, on_seed, val_seed, final_seed, search_seed = np.random.SeedSequence(
        [seed, namespace, index]).spawn(9)
    families = tuple(settings.get("forecast_families", FAMILIES))
    if not families or set(families) - set(FAMILIES):
        raise ValueError(f"Unknown forecast families: {families}")

    def workloads(stream, number, loads):
        streams = stream.spawn(len(loads))
        return [row for i, (child, load) in enumerate(zip(streams, loads, strict=True)) for row in
                scenarios(child, law, spec, settings | {"load": load}, number // len(loads) + (i < number % len(loads)))]

    def logs(name, stream, number):
        target = out / name
        target.mkdir()
        replay = []
        loads = settings["training_loads"]
        for i, (child, load) in enumerate(zip(stream.spawn(len(loads)), loads, strict=True)):
            folder = target / str(i)
            folder.mkdir()
            generate(law, spec, sim, child, folder, load, settings["effect"], number // len(loads) + (i < number % len(loads)),
                     settings["cases_per_workload"], .5)
            replay.extend(read_replay(folder / "transitions.jsonl"))
        return replay

    progress("forecasts")
    ml_fit = logs("ml_fit", fit_seed, settings["ml_fit_workloads"])
    grids, prediction, models = fit_forecasts(ml_fit, logs("ml_calibration", cal_seed, settings["ml_calibration_workloads"]),
                                             logs("prediction_test", pred_seed, settings["ml_test_workloads"]), settings, seed, threads,
                                             families=families)
    joblib.dump((grids, models), out / "predictors.joblib")
    save(out / "prediction.json", prediction)
    raw_env = LoopEnvironment(law, spec, sim, settings["effect"], scenario=True)
    scale_workloads = workloads(scale_seed, settings["scale_workloads"], settings["training_loads"])
    n = settings["cases_per_workload"]
    scale = np.mean([Routing(raw_env, workload, 0, None, None).ep.result() / n for workload in scale_workloads], axis=0)
    weights = np.array(settings["reward_weights"]) / np.array([1., scale[1], scale[2], scale[2], 1.])
    pooled = spec | {"standard_servers": spec["standard_servers"] + spec["specialist_servers"]}

    def make_factory(grid, effect):
        def factory(workload, pool="quota"):
            env = LoopEnvironment(law, pooled if pool == "pooled" else spec, sim, effect, scenario=True)
            quota = settings["slots"] if pool == "quota" else None if pool == "unlimited" else 0
            return Routing(env, workload, quota, grid, weights)

        return factory

    policies = {"FIFO": (lambda _: 0, "standard"), "Pooled_FIFO": (lambda _: 0, "pooled"),
                "Always_specialist_stress": (lambda _: 1, "unlimited"),
                "Repeat_escalation": (lambda env: int(env.state.attempt > 1 and env.state.specialist_wait_estimate == 0), "quota"),
                "Earliest_expected_finish": (lambda env: int(env.state.standard_wait_estimate - env.state.specialist_wait_estimate
                                                              > (raw_env.multiplier - 1) * law.mean_service), "quota")}
    ml_names = {}
    for family, grid in grids.items():
        for horizon in (1, 2):
            choose = ml_plan(raw_env, grid, weights, horizon == 2)
            name = f"{family}_ML_{horizon}"
            ml_names[name] = family
            policies[name] = (lambda env, choose=choose: int(choose(env.state) == "specialist"), "quota")
        ctx = baseline(raw_env, grid, ml_fit, settings, "queue_aware", "repeat")
        name = f"{family}_ML_slot"
        ml_names[name] = family
        policies[name] = (lambda env, policy=ml_actor(ctx): policy(env.ep), "quota")
        if settings.get("elapsed_ml_policy", False) and domain == "bpisim":
            elapsed = baseline(raw_env, grid, ml_fit, settings, "queue_aware", "elapsed",
                               settings.get("elapsed_wait_weight"))
            name = f"{family}_ML_elapsed"
            ml_names[name] = family
            policies[name] = (lambda env, policy=ml_actor(elapsed): policy(env.ep), "quota")
    val_workloads = workloads(val_seed, settings["validation_workloads"], [settings["primary_load"]])
    validation = policy_metrics(make_factory(grids[families[0]], settings["effect"]), val_workloads, policies, n)
    eligible = [name for name in ml_names if feasible(validation[name], validation["FIFO"], settings["limits"])]
    strongest = min(eligible, default=None, key=lambda name: (validation[name][:, 0].mean(), validation[name][:, 5].mean()))
    fixed_by_domain = settings.get("fixed_ml_policy_by_domain", {})
    matched = fixed_by_domain.get(domain, settings.get("fixed_ml_policy")) or strongest or f"{families[0]}_ML_slot"
    if matched not in policies or matched not in ml_names:
        raise ValueError(f"Fixed ML policy is unavailable: {matched}")
    grid = grids[ml_names[matched]]
    progress("rl_training")
    factory = make_factory(grid, settings["effect"])

    def q_key(env):
        s = env.state
        forecasts = grid[min(s.attempt, 2) - 1, min(grid.shape[1] - 1, int(s.difficulty_signal * grid.shape[1]))]
        return (min(s.attempt, 2), min(3, int(s.difficulty_signal * 4)), min(2, s.standard_jobs), min(2, s.specialist_jobs),
                min(2, env.ep.slots // 3), min(2, (env.ep.size - env.ep.seen) * 3 // env.ep.size), int(forecasts[0, 0] - forecasts[1, 0] > .05))

    rl = fit_policies(factory, workloads(off_seed, settings["offline_workloads"], settings["training_loads"]),
                      workloads(on_seed, settings["online_workloads"], settings["training_loads"]), seed, settings,
                      out, policies[matched][0], q_key, [*range(7), 11, 12, 13], threads, torch)
    rl = {name: (policy, "quota") for name, policy in rl.items()}
    validation.update(policy_metrics(factory, val_workloads, rl, n))

    episodic = None
    search_cfg = settings.get("episodic_search", {})
    if search_cfg.get("enabled", False):
        from scripts.specialist_slot_search import residual_actor

        objective = search_cfg["objective_by_domain"][domain]
        metric = {"elapsed": 1, "queue": 5, "tail_queue": 12}[objective]
        context = baseline(raw_env, grid, ml_fit, settings, "queue_aware", objective,
                           search_cfg.get("anchor_wait_weight"))
        training_workloads = workloads(search_seed, search_cfg["workloads"], settings["training_loads"])
        blocks = np.array_split(np.asarray(training_workloads, dtype=object), search_cfg["blocks"])
        rng = np.random.default_rng(np.random.SeedSequence([seed, namespace, index, 91]))
        center = np.zeros(len(search_cfg["initial_spread"]), dtype=float)
        spread = np.asarray(search_cfg["initial_spread"], dtype=float)
        candidates, training = {}, []
        reference_rows = validation[matched]
        block_references = [policy_metrics(factory, list(block), {matched: policies[matched]}, n)[matched]
                            for block in blocks]
        repeat_penalty = search_cfg.get("repeat_penalty_by_domain", {}).get(domain,
                         search_cfg["repeat_penalty"])
        for step in range(search_cfg["rounds"]):
            block_index = step % len(blocks)
            block = list(blocks[block_index])
            population_weights = np.vstack((center, rng.normal(center, spread,
                                           size=(search_cfg["population"] - 1, len(center)))))
            actors = {str(i): (lambda env, actor=residual_actor(context, row, settings["slots"]): actor(env.ep))
                      for i, row in enumerate(population_weights)}
            scored = policy_metrics(factory, block, {name: (actor, "quota") for name, actor in actors.items()}, n)
            costs = []
            for values in scored.values():
                repeat_excess = max(0., values[:, 0].mean() - block_references[block_index][:, 0].mean()
                                    - search_cfg["repeat_tolerance"])
                costs.append(values[:, metric].mean() + repeat_penalty * repeat_excess)
            elite = np.argsort(costs)[:search_cfg["elite"]]
            best = int(elite[0])
            name = f"round_{step + 1}"
            candidates[name] = population_weights[best]
            training.append({"round": step + 1, "cost": float(costs[best]),
                             "weights": population_weights[best].tolist()})
            center = .3 * center + .7 * population_weights[elite].mean(axis=0)
            spread = np.maximum(search_cfg["minimum_spread"],
                                .4 * spread + .6 * population_weights[elite].std(axis=0))
        candidate_policies = {name: (lambda env, actor=residual_actor(context, row, settings["slots"]): actor(env.ep))
                              for name, row in candidates.items()}
        candidate_validation = policy_metrics(factory, val_workloads,
            {name: (actor, "quota") for name, actor in candidate_policies.items()}, n)

        def repeat_upper(values):
            delta = values[:, 0] - reference_rows[:, 0]
            return float(delta.mean() + search_cfg.get("repeat_confidence_z", 0.)
                         * delta.std(ddof=1) / np.sqrt(len(delta)))

        minimum_gain = search_cfg.get("minimum_validation_gain_fraction", 0.)
        blend_cfg = search_cfg.get("safe_blend_by_domain", {}).get(domain)
        blend_weight, blended_validation = 1., None
        if blend_cfg:
            allowed_blends = []
            for name, values in candidate_validation.items():
                upper = repeat_upper(values)
                weight = 1. if upper <= 0 else min(1., blend_cfg["safety_factor"]
                                                    * search_cfg["repeat_tolerance"] / upper)
                blended = reference_rows + weight * (values - reference_rows)
                if (weight >= blend_cfg["minimum_weight"]
                    and feasible(blended, validation["FIFO"], settings["limits"], reference_rows)
                    and blended[:, metric].mean() <= (1 - minimum_gain) * reference_rows[:, metric].mean()):
                    allowed_blends.append((name, weight, blended))
            if allowed_blends:
                selected, blend_weight, blended_validation = min(
                    allowed_blends, key=lambda row: row[2][:, metric].mean())
            else:
                selected = None
            allowed = [row[0] for row in allowed_blends]
        else:
            allowed = [name for name, values in candidate_validation.items()
                       if feasible(values, validation["FIFO"], settings["limits"], reference_rows)
                       and repeat_upper(values) <= search_cfg["repeat_tolerance"]
                       and values[:, metric].mean() <= (1 - minimum_gain) * reference_rows[:, metric].mean()]
            selected = min(allowed, default=None, key=lambda name: candidate_validation[name][:, metric].mean())
        fallback = selected is None and search_cfg.get("fallback_to_ml", False)
        if selected is None and not fallback:
            selected = min(candidate_validation, key=lambda name: candidate_validation[name][:, metric].mean())
        if fallback:
            rl["EpisodicSearch"] = policies[matched]
            validation["EpisodicSearch"] = reference_rows
        else:
            rl["EpisodicSearch"] = (candidate_policies[selected], "quota")
            validation["EpisodicSearch"] = (blended_validation if blended_validation is not None
                                             else candidate_validation[selected])
        episodic = {"selected_round": selected, "validation_feasible": selected in allowed if selected else False,
                    "fallback_to_ml": fallback, "objective": objective,
                    "repeat_upper": None if fallback else repeat_upper(candidate_validation[selected]),
                    "safe_blend_weight": 0. if fallback else blend_weight,
                    "repeat_penalty": repeat_penalty, "training": training}
    eligible_rl = [name for name in rl if name.startswith("CQL_") and name != "CQL_history"
                   and feasible(validation[name], validation["FIFO"], settings["limits"], validation[matched])]
    primary_cql = min(eligible_rl, default=None, key=lambda name: validation[name][:, 0].mean())

    def summarize(rows):
        return {key: dict(zip(FIELDS, values.mean(axis=0).tolist(), strict=True)) for key, values in rows.items()}

    frozen = {"matched_ml": matched, "strongest_feasible_ml": strongest, "selected_cql": primary_cql,
              "forecast_family": ml_names[matched], "forecast_families": list(families),
              "reward_weights": weights.tolist(), "validation": summarize(validation), "episodic_search": episodic}
    save(out / "freeze.json", frozen)
    cells = [(load, effect) for load in settings["test_loads"] for effect in settings["test_effects"]]
    reports = {}
    for child, (load, effect) in zip(final_seed.spawn(len(cells)), cells, strict=True):
        primary = load == settings["primary_load"] and effect == settings["effect"]
        testing = workloads(child, settings["primary_workloads"] if primary else settings["sensitivity_workloads"], [load])
        key = f"load_{load:g}_effect_{effect:g}"
        progress(f"heldout_{key}")
        rows = policy_metrics(make_factory(grid, effect), testing, policies | rl, n, no_effect=effect == 0)
        np.savez_compressed(out / f"{key}.npz", **rows)
        reports[key] = summarize(rows)
    save(out / "results.json", frozen | {"metrics": reports, "fields": FIELDS, "zero_effect_case_invariance_verified": 0 in settings["test_effects"]})
