"""Learners for discrete simulator actions."""
from collections import defaultdict

import numpy as np
from catboost import CatBoostRegressor

DEVICE = "cpu"
THREADS = 1


def tabular(workloads, reset, step, key, fallback, actions, seed, cfg):
    rng = np.random.default_rng(seed)
    table = defaultdict(lambda: np.zeros(actions))
    updates = 0

    def choose(state, values):
        preferred = fallback(state)
        if values is None or values[preferred] == values.max():
            return preferred
        return int(values.argmax())

    exploration = np.linspace(cfg["epsilon_start"], cfg["epsilon_end"], len(workloads))
    for workload, epsilon in zip(workloads, exploration, strict=True):
        state = reset(workload)
        while state is not None:
            cell = key(state)
            if rng.random() < epsilon:
                action = int(rng.integers(actions))
            else:
                action = choose(state, table.get(cell))
            q = table[cell]
            state, reward, done = step(action)
            future = None if done else table.get(key(state))
            target = reward + (0. if future is None else future.max())
            q[action] += cfg["q_learning_rate"] * (target - q[action])
            updates += 1
    learned = dict(table)

    def policy(state):
        return choose(state, learned.get(key(state)))

    return policy, learned, updates


def fitted(replay, seed, rounds, iterations, monte_carlo=False):
    x, mask, a, r, nxt, nm, done, returns = replay
    rows = np.arange(len(a))
    target = returns if monte_carlo else r
    for iteration in range(1 if monte_carlo else rounds):
        model = CatBoostRegressor(iterations=iterations, depth=4, learning_rate=.05,
                                  random_seed=seed, verbose=False, thread_count=THREADS,
                                  allow_writing_files=False).fit(x[rows, a], target)
        if not monte_carlo and iteration + 1 < rounds:
            value = model.predict(nxt.reshape(-1, nxt.shape[-1]), thread_count=THREADS).reshape(nm.shape)
            target = r + np.where(done, 0., np.where(nm, value, -1e9).max(axis=1))
    return lambda x, mask: int(np.where(mask, model.predict(x, thread_count=1), -np.inf).argmax()), model


def neural(replay, seed, updates, alpha):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, mask, a, r, nxt, nm, done, _ = replay
    net = network(x.shape[-1]).to(DEVICE)
    target = network(x.shape[-1]).to(DEVICE)
    target.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=.0003)
    for step in range(updates):
        i = rng.integers(len(a), size=min(128, len(a)))
        q = logits(net, x[i], mask[i])
        chosen = q[torch.arange(len(i), device=DEVICE), torch.as_tensor(a[i], device=DEVICE)]
        with torch.no_grad():
            future = logits(target, nxt[i], nm[i]).max(dim=1).values
            y = tensor(r[i], DEVICE) + torch.where(torch.as_tensor(done[i], device=DEVICE), 0., future)
        loss = nn.functional.smooth_l1_loss(chosen, y)
        loss += alpha * (q.logsumexp(dim=1) - chosen).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            target.load_state_dict(net.state_dict())
    net.cpu()
    return greedy(net), net.state_dict()


def tensor(x, device="cpu"):
    import torch
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def network(width):
    from torch import nn
    return nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, 1))


def logits(net, x, mask):
    import torch
    device = next(net.parameters()).device
    return net(tensor(x, device)).squeeze(-1).masked_fill(~torch.as_tensor(mask, device=device), -1e9)


def greedy(net):
    import torch

    def act(x, mask):
        with torch.no_grad():
            return int(logits(net, x, mask).argmax())
    return act


def ppo(episodes, width, seed, horizon):
    import torch
    from torch.distributions import Categorical

    torch.manual_seed(seed)
    actor, critic = network(width).to(DEVICE), network(width).to(DEVICE)
    sampler = network(width)
    sampler.load_state_dict(actor.state_dict())
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=.0003)
    buffer, decisions, updates = [], 0, 0

    def policy(x, mask):
        with torch.no_grad():
            distribution = Categorical(logits=logits(sampler, x, mask))
            a = distribution.sample()
            return int(a), float(distribution.log_prob(a))

    def update():
        nonlocal updates
        x, mask, a, old, returns = map(np.asarray, zip(*buffer))
        pooled = (x * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
        with torch.no_grad():
            advantages = tensor(returns, DEVICE) - critic(tensor(pooled, DEVICE)).flatten()
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
        for _ in range(4):
            dist = Categorical(logits=logits(actor, x, mask))
            ratio = (dist.log_prob(torch.as_tensor(a, device=DEVICE)) - tensor(old, DEVICE)).exp()
            surrogate = torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages)
            value = critic(tensor(pooled, DEVICE)).flatten()
            loss = -surrogate.mean() + .5 * (value - tensor(returns, DEVICE)).square().mean() - .01 * dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], .5)
            optimizer.step()
            updates += 1
        buffer.clear()
        sampler.load_state_dict(actor.state_dict())

    for transitions in episodes(policy):
        decisions += len(transitions)
        buffer.extend((x, mask, a, old, ret) for x, mask, a, _, _, _, _, ret, old in transitions)
        if len(buffer) >= horizon:
            update()
    if buffer:
        update()
    return greedy(sampler), sampler.state_dict(), {"decisions": decisions, "optimizer_updates": updates}
