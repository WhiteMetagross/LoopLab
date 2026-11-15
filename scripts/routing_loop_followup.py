"""Decision depth, decision timing, and burst replay on paired workloads."""

from pathlib import Path

import numpy as np

from scripts import routing_loop_ablations as replay
from scripts import simfactory_tradeoff_replay as blends
from scripts import specialist_slot_search as search
from scripts.specialist_slots_pilot import baseline, ml_actor
from src.domain_loop_data import acquire, chronological_split, load_cases, sha256
from src.domain_loop_simulations import save
from src.domain_loop_simulator import LoopEnvironment, calibrate
from src.domain_loop_training import read_replay, scenarios


FIELDS = (*replay.FIELDS, "repeat_slots")


def cluster_arrivals(arrivals, size, factor):
    gaps = np.diff(arrivals)
    between = np.arange(1, len(arrivals)) % size == 0
    if not 0 < factor < 1 or not between.any() or gaps[between].sum() <= 0:
        raise ValueError("Burst profile needs positive between-burst gaps and a compression factor in (0, 1)")
    updated = gaps.copy()
    updated[~between] *= factor
    updated[between] *= (gaps.sum() - updated[~between].sum()) / gaps[between].sum()
    result = np.r_[arrivals[0], arrivals[0] + updated.cumsum()]
    np.testing.assert_allclose(result[-1], arrivals[-1])
    return result


def fitted_policies(folder, index, namespace, domain, settings, make, law, fit_source=None):
    path = folder.with_suffix(".json")
    save(path, settings)
    search.run(folder, path, index, namespace, domain, prepare_only=True, fit_source=fit_source, law=law)
    if not (folder / "freeze.json").is_file():
        raise ValueError(f"No feasible ML policy in {folder.name}")
    frozen = blends.read(folder / "freeze.json")
    data = read_replay((fit_source or folder) / "ml_fit" / "transitions.jsonl", settings.get("route_when_ready", False))
    grid = np.load(folder / "forecast_grid.npy")
    objective = "elapsed" if domain == "bpisim" else "repeat"
    name = frozen["best_ml"]
    ml = ml_actor(baseline(make(), grid, data, settings, "queue_aware", objective, float(name.split("_")[-1])))
    selected = frozen["selected_rl"]
    hybrid = ml
    if selected is not None:
        ctx = baseline(make(), grid, data, settings, "queue_aware", objective, settings.get("anchor_wait_weight", .04))
        hybrid = search.residual_actor(ctx, frozen["weights"][selected], settings["slots"])
    return {"ML": ml, "hybrid": hybrid, "FIFO": lambda _: 0}, {"ml": name, "hybrid": selected}


def run(folder, index, seed, domain, config, profile, progress):
    sim = blends.read(Path(config["domain_config"]))
    spec = sim["domains"][domain]
    raw = acquire(spec, sim["maximum_download_bytes"])
    cases, _ = load_cases(raw, domain, spec)
    fit, historical, boundary, excluded = chronological_split(cases, sim["fit_fraction"])
    law = calibrate(fit, spec, sim)
    source_hashes = {}
    if domain == "manufacturing":
        source = Path(config["factory_run"]) / f"seed_{index}"
        frozen = blends.read(source / "freeze.json")
        if frozen["seed"] != seed:
            raise ValueError("SimFactory source seed mismatch")
        settings = frozen["settings"]
        source_hashes = {str(path): sha256(path.read_bytes()) for path in (source / "freeze.json", source / "blend_freeze.json",
                         source / "forecast_grid.npy", source / "ml_fit" / "transitions.jsonl")}
    else:
        base = "bpisim_turnaround_search" if domain == "bpisim" else "specialist_slot_search"
        settings = blends.read(Path("configs") / f"{base}.json") | profile["routing"]

    def make(ready=False):
        return LoopEnvironment(law, spec, sim, settings["effect"], scenario=True, route_when_ready=ready)

    workloads = scenarios(np.random.SeedSequence([seed, profile["namespace"], 937]), law, spec, settings, profile["workloads"])
    arrays = {"arrivals": np.asarray([w[0] for w in workloads]), "noise": np.asarray([w[1] for w in workloads], dtype=np.int64)}
    rows, choice, contrasts = {}, {}, {}
    if domain == "manufacturing":
        clustered = [(cluster_arrivals(arr, config["burst_size"], config["burst_gap_factor"]), noise) for arr, noise in workloads]
        arrays["clustered_arrivals"] = np.asarray([w[0] for w in clustered])
        picks = blends.read(source / "blend_freeze.json")
        choice = {"hybrid": picks["hybrid_mix"], "control": picks["ml_mix"], "original": {picks["reference"]: 1.}}
        controls = blends.make_controls(frozen, np.load(source / "forecast_grid.npy"),
                                        read_replay(source / "ml_fit" / "transitions.jsonl"), make)
        names = sorted({name for weights in choice.values() for name in weights} | {"FIFO"})
        for arm, panel in (("original_arrivals", workloads), ("clustered", clustered)):
            progress(arm)
            scored = replay.assess(make, panel, {name: controls[name] for name in names}, settings, include_repeat_slots=True)
            rows[f"{arm}_FIFO"] = scored["FIFO"]
            for method, weights in choice.items():
                rows[f"{arm}_{method}"] = blends.weighted_rows(weights, scored)
        first, second = "original_arrivals", "clustered"
        comparisons = ("control", "original")
    else:
        arms = ("marker", "ready") if domain == "bpisim" else ("repeat_allowed", "first_only")
        for arm in arms:
            progress(f"fit_{arm}")
            cfg = settings | {"route_when_ready": arm == "ready", "first_only": arm == "first_only"}
            factory = lambda: make(ready=cfg["route_when_ready"])
            fit_source = folder / "repeat_allowed" if arm == "first_only" else None
            policies, choice[arm] = fitted_policies(folder / arm, index, profile["namespace"], domain, cfg, factory, law, fit_source)
            if domain == "insurance":
                progress(f"fit_{arm}_matched_threshold")
                static, selected = fitted_policies(folder / f"{arm}_threshold", index, profile["namespace"], domain,
                                                   cfg | {"ablation": "static_ml"}, factory, law, folder / "repeat_allowed")
                policies["supervised_ML"], policies["ML"] = policies["ML"], static["hybrid"]
                choice[arm]["threshold_round"] = selected["hybrid"]
            scored = replay.assess(factory, workloads, policies, cfg, include_repeat_slots=True)
            rows.update({f"{arm}_{name}": values for name, values in scored.items()})
        first, second = arms
        comparisons = ("ML",)
    for ref in comparisons:
        contrasts[f"{first}_hybrid_minus_{ref}"] = {f"{first}_hybrid": 1, f"{first}_{ref}": -1}
        contrasts[f"{second}_hybrid_minus_{ref}"] = {f"{second}_hybrid": 1, f"{second}_{ref}": -1}
        contrasts[f"interaction_vs_{ref}"] = {f"{second}_hybrid": 1, f"{second}_{ref}": -1,
                                               f"{first}_hybrid": -1, f"{first}_{ref}": 1}
        contrasts[f"{second}_minus_{first}_{ref}"] = {f"{second}_{ref}": 1, f"{first}_{ref}": -1}
    contrasts[f"{second}_minus_{first}_hybrid"] = {f"{second}_hybrid": 1, f"{first}_hybrid": -1}
    np.savez_compressed(folder / "workloads.npz", **arrays)
    np.savez_compressed(folder / "paired.npz", **rows)
    save(folder / "design.json", {"fields": FIELDS, "contrasts": contrasts, "choice": choice, "settings": settings,
                                  "source_sha256": source_hashes, "data": spec, "split": {"fit_cases": len(fit),
                                  "holdout_cases": len(historical), "boundary": boundary, "excluded": excluded}})
