"""Check native customer sampling against an independent response stream."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from src.loop_feasibility import write_json
from src.negotiation_pilot import native_acceptance
from src.negotiation_support import NegotiationEnvironment, run_policy


class IndependentNegotiationEnvironment(NegotiationEnvironment):
    """Keep native equations; give each offer an independent customer stream."""

    def reset(self, scenario):
        self.scenario = int(scenario)
        return super().reset(scenario)

    def step(self, action):
        state = self.observe()
        key = f"customer-response:{self.config['seed']}:{self.scenario}:{state['offer_count']}"
        response_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], "big")
        self.generator.extra_flow_conditioner.random_obj = random.Random(response_seed)
        return super().step(action)


def main():
    config = json.loads(Path("configs/negotiation_pilot.json").read_text())
    report = {}
    for label, cls in [("native_streams", NegotiationEnvironment),
                       ("independent_customer_stream", IndependentNegotiationEnvironment)]:
        env, rows = cls(config), []
        rng = np.random.default_rng(config["seed"])
        for i in range(500):
            _, transitions = run_policy(env, 900011 + i, lambda state: int(rng.integers(len(config["rates"]))))
            for r in transitions:
                if r["immediate_refusal"] or r["accepted_now"]:
                    p = 1 - native_acceptance(r["state"], config["rates"][r["action"]])
                    rows.append((p, r["immediate_refusal"]))
        p, y = np.asarray(rows).T
        report[label] = {"response_opportunities": len(y), "predicted_refusal": float(p.mean()),
                         "observed_refusal": float(y.mean()), "Brier": float(np.mean((p - y) ** 2)),
                         "expected_Bernoulli_Brier": float(np.mean(p * (1 - p)))}
    destination = Path("research/negotiation_study/randomness_probe.json")
    write_json(destination, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
