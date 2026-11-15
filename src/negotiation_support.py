"""Adapter to the existing offer refusal loop. Native source is unchanged."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import numpy as np

from src.simbank_support import generator_class, load_released


FEATURES = ["amount", "est_quality", "unc_quality", "elapsed_time", "cum_cost", "previous_rate",
            "offer_count", "refusal_count", "last_activity", "minimum_rate", "offer_time"]


def parameters(config, max_offers=None):
    params = deepcopy(load_released(config["parameter_file"]))
    params["policies_info"]["max_nor"] = max_offers if max_offers is not None else config["max_offers"]
    info = params["intervention_info"]
    assert info["name"] == ["set_ir_3_levels"]
    info.update(RCT=False, actions=[list(config["rates"])], action_width=[len(config["rates"])],
                action_depth=[params["policies_info"]["max_nor"]], len=[params["policies_info"]["max_nor"]],
                action_combinations=[[rate] for rate in config["rates"]], action_width_combinations=len(config["rates"]),
                action_depth_combinations=params["policies_info"]["max_nor"])
    return params


def minimum_rate(amount, quality, cost):
    discount = .03 + (10 - quality) / 200
    return math.ceil((((cost + 10 + 100) / amount + 1) ** .1 * (1 + discount) - 1) * 100) / 100


def finite(value, default=0.):
    return float(value) if np.isfinite(value) else default


def prefix_features(prefix, executioner):
    last, counts = prefix[-1], Counter(event["activity"] for event in prefix)
    begin = prefix[0]["timestamp"]
    # Offer start is the end of observed work, including completed parallel strands.
    known_end = max((event["timestamp"] - begin).total_seconds() / 86400 + executioner.times_dic[event["activity"]] / 86400 for event in prefix)
    state = {key: float(last[key]) for key in ["amount", "est_quality", "unc_quality", "elapsed_time", "cum_cost"]}
    state.update(previous_rate=finite(last["interest_rate"]), offer_count=counts["calculate_offer"],
                 refusal_count=counts["receive_refusal"], last_activity=last["activity"],
                 minimum_rate=minimum_rate(last["amount"], last["est_quality"], last["cum_cost"]),
                 offer_time=float(known_end))
    return state


def bank_rate(state, rates):
    if state["refusal_count"] > 0:
        value = state["previous_rate"] - .01
    else:
        value = .07 if state["amount"] > 60000 else (.08 if state["amount"] > 30000 else .09)
        if state["elapsed_time"] < 6:
            value += .01
        value = max(value, state["minimum_rate"])
        value = min([.07, .08, .09], key=lambda rate: abs(rate - value))
    return int(np.argmin(np.abs(np.array(rates) - value)))


class NegotiationEnvironment:
    def __init__(self, config, max_offers=None):
        self.config = config
        self.params = parameters(config, max_offers)
        self.generator = generator_class()(deepcopy(self.params), seed=config["seed"])
        self.branches = []

    def reset(self, scenario):
        self.generator.random_seed = int(scenario)
        self.generator.simulation_start = self.params["simulation_start"]
        self.branches = self.generator.start_simulation_inference(seed_to_add=0)
        return self.observe()

    def observe(self):
        if not self.generator.int_points_available:
            return None
        assert self.generator.current_int_index == 0
        assert len(self.branches) == len(self.config["rates"])
        states = [prefix_features(branch[:-1], self.generator.activity_executioner) for branch in self.branches]
        assert all(state == states[0] for state in states)
        assert states[0]["offer_count"] < self.params["policies_info"]["max_nor"]
        return states[0]

    def step(self, action):
        if not 0 <= action < len(self.config["rates"]):
            raise ValueError("Invalid rate action.")
        self.branches = self.generator.continue_simulation_inference(int(action))
        return self.observe()

    def trace(self):
        return self.branches[0][:-1] if self.generator.int_points_available else self.generator.end_simulation_inference()

    def result(self):
        assert not self.generator.int_points_available
        trace = self.trace()
        counts = Counter(event["activity"] for event in trace)
        assert counts["calculate_offer"] <= self.params["policies_info"]["max_nor"]
        assert counts["receive_refusal"] <= counts["calculate_offer"]
        last = trace[-1]
        return {"profit": float(last["outcome"]), "accepted": int(last["activity"] == "receive_acceptance"),
                "cancelled": int(last["activity"] == "cancel_application"), "offers": counts["calculate_offer"],
                "refusals": counts["receive_refusal"], "revised_offers": max(0, counts["calculate_offer"] - 1),
                "negotiation_effort": sum(self.generator.activity_executioner.times_dic[event["activity"]] / 60 for event in trace if event["activity"] in ["calculate_offer", "receive_refusal", "receive_acceptance", "cancel_application"]),
                "elapsed_days": float(last["elapsed_time"]), "native_cost": float(last["cum_cost"])}


def run_policy(env, scenario, policy):
    state, transitions = env.reset(scenario), []
    while state is not None:
        before = len(env.trace())
        action = int(policy(state))
        next_state = env.step(action)
        after = env.trace()
        new = after[before:]
        refusal = int(any(event["activity"] == "receive_refusal" for event in new))
        accepted = int(any(event["activity"] == "receive_acceptance" for event in new))
        transitions.append({"state": state, "action": action, "next_state": next_state,
                            "immediate_refusal": refusal, "accepted_now": accepted})
        state = next_state
    result = env.result()
    for transition in transitions:
        prior = transition["state"]["refusal_count"]
        transition.update(terminal_profit=result["profit"], remaining_refusals=result["refusals"] - prior,
                          remaining_effort=(result["offers"] - transition["state"]["offer_count"]) * 1440 +
                          (result["refusals"] - prior) * 288 + (288 if result["accepted"] else 144))
    return result, transitions


def refusal_successor(state, rate):
    next_state = dict(state)
    next_state.update(cum_cost=state["cum_cost"] + 400 + 10, previous_rate=rate,
                      offer_count=state["offer_count"] + 1, refusal_count=state["refusal_count"] + 1,
                      last_activity="receive_refusal", elapsed_time=state["offer_time"] + 1.,
                      offer_time=state["offer_time"] + 1.2)
    next_state["minimum_rate"] = minimum_rate(next_state["amount"], next_state["est_quality"], next_state["cum_cost"])
    return next_state


def cancel_profit(state, refusal=False):
    return -(state["cum_cost"] + 400 + (10 if refusal else 0) + 30 + 100)
