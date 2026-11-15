"""Shared discrete learners for paper simulator tasks."""

from collections import Counter

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src import paper_rl
from src.domain_loop_training import attach_returns
from src.domain_loop_simulations import save


def trajectory(env, actor, normalize=lambda value: value, profit_reference=False):
    records = []
    while env.state is not None:
        x, mask = env.observe()
        action, log_probability = actor(normalize(x), mask)
        if not mask[action]:
            raise ValueError("Policy selected a masked action")
        reward, done = env.step(action)
        nxt, next_mask = (np.zeros_like(x), np.zeros_like(mask)) if done else env.observe()
        records.append([x, mask, action, reward, nxt, next_mask, done, 0., log_probability])
        if profit_reference:
            records[-1].append(env.profit_reward)
    if records:
        env.audit(sum(row[3] for row in records))
    return attach_returns(records)


def fit_policies(factory, offline, online, seed, settings, out, fallback, q_key, history, threads, torch):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))

    def random_action(_x, mask):
        choices = np.flatnonzero(mask)
        return int(rng.choice(choices)), -np.log(len(choices))

    profit_reference = settings.get("profit_cql_reference", False)
    rows = [row for workload in offline for row in trajectory(factory(workload), random_action, profit_reference=profit_reference)]
    raw = tuple(map(np.asarray, zip(*rows, strict=True)))
    names = ("x", "mask", "action", "reward", "next_x", "next_mask", "done", "return", "log_behavior_probability")
    if profit_reference:
        names = (*names, "profit_reward")
    np.savez_compressed(out / "offline_replay.npz", **dict(zip(names, raw, strict=True)))
    counts = {"offline_decisions": len(rows), "offline_actions": dict(Counter(int(row[2]) for row in rows))}
    if len(counts["offline_actions"]) != raw[0].shape[1]:
        raise ValueError("Randomized replay lacks action support")
    paper_rl.THREADS = threads
    policies, models = {}, {}

    def fit_view(keep):
        scaler = StandardScaler().fit(raw[0][raw[1]][:, keep])

        def normalize(x):
            return np.clip((x[..., keep] - scaler.mean_) / scaler.scale_, -10., 10.).astype(np.float32)

        return scaler, normalize, (normalize(raw[0]), *raw[1:4], normalize(raw[4]), *raw[5:8])

    models["scaler"], normalize, replay = fit_view(np.arange(raw[0].shape[-1]))
    fitted, models["FittedQ"] = paper_rl.fitted(replay, seed, settings["fitted_q_rounds"], settings["trees"])

    def apply(actor, normalizer):
        def choose(env):
            x, mask = env.observe()
            return actor(normalizer(x), mask)

        return choose

    policies["FittedQ"] = apply(fitted, normalize)
    for alpha in (0., *settings["cql_alphas"]):
        name = "NeuralQ_zero_penalty" if alpha == 0 else f"CQL_{alpha:g}"
        actor, checkpoint = paper_rl.neural(replay, seed, settings["cql_updates"], alpha)
        policies[name] = apply(actor, normalize)
        torch.save(checkpoint, out / f"{name}.pt")
    if profit_reference:
        profit_replay = (*replay[:3], raw[-1], *replay[4:])
        actor, checkpoint = paper_rl.neural(profit_replay, seed, settings["cql_updates"], settings["cql_alphas"][0])
        policies["CQL_profit_reference"] = apply(actor, normalize)
        torch.save(checkpoint, out / "CQL_profit_reference.pt")
        ensemble_size = settings.get("profit_fitted_ensemble_size", 0)
        if ensemble_size:
            ensemble = []
            for index, child in enumerate(np.random.SeedSequence([seed, 31]).spawn(ensemble_size)):
                model_seed = int(child.generate_state(1)[0] % (2**31 - 1))
                _, model = paper_rl.fitted(profit_replay, model_seed, settings["fitted_q_rounds"], settings["trees"])
                ensemble.append(model)

            def profit_actor(x, mask):
                values = np.mean([model.predict(x, thread_count=1) for model in ensemble], axis=0)
                return int(np.where(mask, values, -np.inf).argmax())

            models["FittedQ_profit_ensemble"] = ensemble
            policies["FittedQ_profit_ensemble"] = apply(profit_actor, normalize)
            counts["profit_fitted_ensemble_size"] = ensemble_size
    if settings["history_ablation"]:
        models["history_scaler"], raw_normalize, history_replay = fit_view(np.asarray(history))
        alpha = settings["cql_alphas"][0]
        actor, checkpoint = paper_rl.neural(history_replay, seed, settings["cql_updates"], alpha)
        policies["CQL_history"] = apply(actor, raw_normalize)
        torch.save(checkpoint, out / "CQL_history.pt")

    current = None

    def reset(workload):
        nonlocal current
        current = factory(workload)
        return current if current.state is not None else None

    def step(action):
        reward, done = current.step(action)
        return None if done else current, reward, done

    tabular_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0])
    policies["TabularQ"], models["TabularQ"], counts["tabular_updates"] = paper_rl.tabular(
        online, reset, step, q_key, fallback, raw[0].shape[1], tabular_seed, settings)

    def online_episodes(actor):
        for workload in online:
            records = trajectory(factory(workload), actor, normalize)
            if records:
                yield [[normalize(row[0]), *row[1:]] for row in records]

    ppo_seed = int(np.random.SeedSequence([seed, 2]).generate_state(1)[0])
    actor, checkpoint, counts["ppo"] = paper_rl.ppo(online_episodes, raw[0].shape[-1], ppo_seed, settings["ppo_horizon"])
    policies["PPO"] = apply(actor, normalize)
    torch.save(checkpoint, out / "PPO.pt")
    joblib.dump(models, out / "policy_models.joblib")
    save(out / "training_counts.json", counts)
    return policies
