"""Build Table III, Figure 3 and Table IV from the audited ablation run."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.summarize_focused_algorithm_study import hierarchical_interval


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def ci_text(value, interval, digits=2):
    threshold = 0.5 * 10 ** (-digits)
    value = 0.0 if abs(value) < threshold else value
    interval = [0.0 if abs(item) < threshold else item for item in interval]
    return rf"${value:.{digits}f}\;[{interval[0]:.{digits}f},\,{interval[1]:.{digits}f}]$"


def saved_effect(comparison, field, scale=1.0, direction=-1.0):
    value = direction * scale * comparison["mean"][field]
    raw = direction * scale * np.asarray(comparison["paired_ci95"][field], dtype=float)
    return float(value), [float(raw.min()), float(raw.max())]


def direct_bank_comparison(root, results):
    fields = results["domains"]["simbank"]["fields"]
    panels = []
    for index in range(3):
        with np.load(root / "simbank" / f"seed_{index}" / "paired.npz") as arrays:
            panels.append(arrays["CQL_both"] - arrays["Regression_DP_500"])
    rng = np.random.default_rng(np.random.SeedSequence([17, 1403]))
    interval = hierarchical_interval(panels, rng, repetitions=10000, case_level=True)
    mean = np.asarray([panel.mean(axis=0) for panel in panels]).mean(axis=0)
    return {
        field: {
            "difference": float(mean[fields.index(field)]),
            "ci95": [float(interval[0, fields.index(field)]),
                     float(interval[2, fields.index(field)])],
        }
        for field in ("profit", "refusals", "repeat_refusals", "cancelled")
    }


def focused_table(results, bank_direct):
    domains = results["domains"]
    claims = domains["insurance"]
    bpi = domains["bpisim"]
    factory = domains["manufacturing"]
    bank = domains["simbank"]

    claims_gain = saved_effect(claims["comparisons"]["full_minus_supervised_ML"], "queue")
    claims_repeat = saved_effect(
        claims["comparisons"]["full_minus_supervised_ML"], "extra_attempts", 100, 1)
    bpi_gain = saved_effect(bpi["comparisons"]["marker_forced_minus_ML"], "elapsed")
    bpi_repeat = saved_effect(
        bpi["comparisons"]["marker_forced_minus_ML"], "extra_attempts", 100)
    factory_gain = saved_effect(factory["comparisons"]["safe_075_minus_ML"], "tail_queue")
    factory_repeat = saved_effect(
        factory["comparisons"]["safe_075_minus_ML"], "extra_attempts", 100, 1)
    bank_refusal = {
        "value": -100 * bank_direct["refusals"]["difference"],
        "ci": sorted(-100 * np.asarray(bank_direct["refusals"]["ci95"])),
    }
    bank_profit = bank_direct["profit"]

    claims_control = claims["policies"]["full_ML"]["mean"]["queue"]
    claims_candidate = claims["policies"]["full"]["mean"]["queue"]
    bpi_control = bpi["policies"]["marker_ML"]["mean"]["elapsed"]
    bpi_candidate = bpi["policies"]["marker_forced"]["mean"]["elapsed"]
    factory_control = factory["policies"]["ML"]["mean"]["tail_queue"]
    factory_candidate = factory["policies"]["safe_075"]["mean"]["tail_queue"]
    bank_control = 100 * bank["policies"]["Regression_DP_500"]["mean"]["refusals"]
    bank_candidate = 100 * bank["policies"]["CQL_both"]["mean"]["refusals"]

    rows = [
        (rf"SimClaims & Regression & Regression with episodic policy search & "
         rf"${claims_control:.2f}\to{claims_candidate:.2f}$ queue min & "
         rf"{ci_text(*claims_gain)} min $(11.25\%)$ & "
         rf"{ci_text(*claims_repeat, digits=3)} more attempts/100 \\"),
        (rf"BPISim & Regression & Episodic policy search & "
         rf"${bpi_control:.2f}\to{bpi_candidate:.2f}$ elapsed min & "
         rf"{ci_text(*bpi_gain)} min $({100 * bpi_gain[0] / bpi_control:.2f}\%)$ & "
         rf"{ci_text(*bpi_repeat, digits=3)} fewer attempts/100 \\"),
        (rf"SimFactory & Regression & Regression with safety constrained search & "
         rf"${factory_control:.2f}\to{factory_candidate:.2f}$ tail min & "
         rf"{ci_text(*factory_gain)} min $(7.30\%)$ & "
         rf"{ci_text(*factory_repeat, digits=3)} more repairs/100 \\"),
        (rf"SimBank & Regression based dynamic planner & Regression informed CQL & "
         rf"${bank_control:.2f}\to{bank_candidate:.2f}$ refusals/100 & "
         rf"{ci_text(bank_refusal['value'], bank_refusal['ci'])} fewer/100 $(22.95\%)$ & "
         rf"{ci_text(bank_profit['difference'], bank_profit['ci95'])} profit \\"),
    ]
    return "\n".join([
        r"\begin{table*}[!t]",
        r"\caption{Focused held out comparisons using Regression forecasts.}",
        r"\label{tab:focused}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\begin{tabularx}{\textwidth}{@{}L{.58in}L{.90in}L{1.20in}L{1.15in}L{1.42in}Y@{}}",
        r"\toprule",
        r"Simulator & Control & Controller & Outcome $\downarrow$ & Gain $\uparrow$ [95\% CI] & Other outcome [95\% CI] \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\par\smallskip",
        r"\begin{minipage}{\textwidth}\footnotesize",
        (r"Lower primary outcomes are better. Gain is control minus controller. "
         r"Means and paired 95\% intervals use three random seeds. The SimBank other outcome is native profit."),
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ])


def ablation_table(results):
    domains = results["domains"]
    claims = domains["insurance"]["comparisons"]
    bpi = domains["bpisim"]
    factory = domains["manufacturing"]["comparisons"]
    bank = domains["simbank"]["comparisons"]

    def effect(comparison, field, scale=1, direction=-1, digits=2, unit=""):
        value, interval = saved_effect(comparison, field, scale, direction)
        return f"{ci_text(value, interval, digits)}{unit}"

    marker = bpi["policies"]["marker_ML"]["mean"]
    reviewer_share = 100 * (marker["queue"] + marker["work"]) / marker["elapsed"]
    rows = [
        (r"\textit{SimClaims:} Does search add to equal static tuning? & Full vs. equal static & "
         f"{effect(claims['full_minus_equal_static'], 'queue')} queue min saved. "
         r"No confirmed residual gain. \\"),
        (r"\textit{SimClaims:} Is congestion state useful? & Full vs. no congestion & "
         f"{effect(claims['full_minus_no_congestion'], 'queue')} queue min saved. "
         r"Congestion is essential. \\"),
        (r"\textit{SimClaims:} Do forecast residuals help? & Full vs. no forecast & "
         f"{effect(claims['full_minus_no_forecast'], 'queue')} queue min saved. "
         r"Forecast residuals reduce queue performance. \\"),
        (r"\textit{SimClaims:} Do later repeat decisions help? & Full vs. first only & "
         f"{effect(claims['full_minus_first_only'], 'queue')} queue min saved. "
         r"First only routing is better. \\"),
        r"\midrule",
        (r"\textit{BPISim:} Does validation select search? & Selected vs. Regression & "
         f"{effect(bpi['comparisons']['marker_selected_minus_ML'], 'elapsed')} elapsed min saved. "
         r"Regression is selected for all seeds. \\"),
        (r"\textit{BPISim:} Can episodic search improve elapsed time? & Episodic search vs. Regression & "
         f"{effect(bpi['comparisons']['marker_forced_minus_ML'], 'elapsed')} elapsed min saved. "
         r"The gain is not confirmed. \\"),
        (r"\textit{BPISim:} How much elapsed time can review affect? & Review queue and work share & "
         rf"${reviewer_share:.2f}\%$ of elapsed time. Most elapsed time is outside the decision. \\"),
        r"\midrule",
        (r"\textit{SimFactory:} Does accrued waiting state help? & Safe vs. no accrued wait & "
         f"{effect(factory['safe_075_minus_no_accrued_wait'], 'tail_queue')} tail min saved. "
         r"Accrued wait is useful. \\"),
        (r"\textit{SimFactory:} Does the tail reward beat mean queue reward? & Tail vs. mean queue objective & "
         f"{effect(factory['safe_075_minus_mean_queue_objective'], 'tail_queue')} tail min saved. "
         r"No confirmed advantage. \\"),
        (r"\textit{SimFactory:} Does the gain survive burst arrivals? & Safe vs. Regression under bursts & "
         f"{effect(factory['burst_safe_075_minus_ML'], 'tail_queue')} tail min saved $(3.17\%)$. "
         r"The gain remains below the 5\% target. \\"),
        r"\midrule",
        (r"\textit{SimBank:} Do process penalties reduce refusals? & Both penalties vs. profit only & "
         f"{effect(bank['both_minus_profit'], 'refusals', 100)} refusals avoided/100. "
         f"{effect(bank['both_minus_profit'], 'profit', direction=1)} profit change. " + r"\\"),
        (r"\textit{SimBank:} Do forecast features help CQL? & Full vs. history only CQL & "
         f"{effect(bank['full_CQL_minus_history'], 'profit', direction=1)} profit change. "
         f"{effect(bank['full_CQL_minus_history'], 'refusals', 100)} refusals avoided/100. " + r"\\"),
        (r"\textit{SimBank:} Does ensembling improve fitted Q? & Ensemble vs. single model & "
         f"{effect(bank['ensemble_minus_single'], 'profit', direction=1)} profit change. "
         r"The gain is not confirmed. \\"),
        (r"\textit{SimBank:} Does biased logging change the policy? & Biased vs. random logs & "
         f"{effect(bank['logging_CQL_biased_minus_random'], 'profit', direction=1)} profit change. "
         r"Biased logs reduce profit. \\"),
    ]
    return "\n".join([
        r"\begin{table*}[!t]",
        r"\caption{Component, timing, reward and logging ablations on held out streams.}",
        r"\label{tab:ablations}",
        r"\centering\fontsize{6.5}{7.1}\selectfont",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{tabularx}{\textwidth}{@{}L{2.35in}L{1.55in}Y@{}}",
        r"\toprule",
        r"Domain and question & Comparison & Held out result [95\% CI] \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\par\smallskip",
        r"\begin{minipage}{\textwidth}\fontsize{6.5}{7.1}\selectfont",
        (r"Positive saved or avoided values favor the first policy. Profit change is first minus second. "
         r"Intervals are paired across three random seeds. The review share is descriptive."),
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ])


def focused_figure(results, bank_direct, output_paths):
    domains = results["domains"]
    seeds = [str(seed) for seed in results["seeds"]]

    def values(domain, policy, field):
        rows = domains[domain]["policies"][policy]["by_seed"]
        return np.asarray([rows[seed][field] for seed in seeds], dtype=float)

    def route_panel(domain, control, candidate, primary_field, primary_title, secondary_title):
        primary_control = values(domain, control, primary_field)
        primary_candidate = values(domain, candidate, primary_field)
        repeat_control = values(domain, control, "extra_attempts")
        repeat_candidate = values(domain, candidate, "extra_attempts")
        comparison_key = {
            "insurance": "full_minus_supervised_ML",
            "bpisim": "marker_forced_minus_ML",
            "manufacturing": "safe_075_minus_ML",
        }[domain]
        comparison = domains[domain]["comparisons"][comparison_key]
        gain, gain_ci = saved_effect(comparison, primary_field)
        repeat_gain, repeat_ci = saved_effect(comparison, "extra_attempts", 100)
        denominator = primary_control.mean()
        return {
            "primary_title": primary_title,
            "primary": 100 * (primary_control - primary_candidate) / denominator,
            "primary_ci": [100 * gain_ci[0] / denominator, 100 * gain_ci[1] / denominator],
            "secondary_title": secondary_title,
            "secondary": 100 * (repeat_control - repeat_candidate),
            "secondary_ci": repeat_ci,
            "safety_limit": -0.5,
        }

    panels = [
        {"simulator": "SimClaims", **route_panel(
            "insurance", "full_ML", "full", "queue", "Queue saved (%)", "Attempts avoided /100")},
        {"simulator": "BPISim", **route_panel(
            "bpisim", "marker_ML", "marker_forced", "elapsed", "Elapsed time saved (%)",
            "Attempts avoided /100")},
        {"simulator": "SimFactory", **route_panel(
            "manufacturing", "ML", "safe_075", "tail_queue", "Tail wait saved (%)",
            "Repairs avoided /100")},
    ]
    bank_control_refusals = values("simbank", "Regression_DP_500", "refusals")
    bank_candidate_refusals = values("simbank", "CQL_both", "refusals")
    bank_control_profit = values("simbank", "Regression_DP_500", "profit")
    bank_candidate_profit = values("simbank", "CQL_both", "profit")
    panels.append({
        "simulator": "SimBank",
        "primary_title": "Refusals avoided /100",
        "primary": 100 * (bank_control_refusals - bank_candidate_refusals),
        "primary_ci": sorted(-100 * np.asarray(bank_direct["refusals"]["ci95"])),
        "secondary_title": "Profit gain (%)",
        "secondary": 100 * (bank_candidate_profit - bank_control_profit) / bank_control_profit.mean(),
        "secondary_ci": [100 * item / bank_control_profit.mean() for item in bank_direct["profit"]["ci95"]],
        "safety_limit": -0.5,
    })

    colors = ["#1963B5", "#008391", "#8425A5", "#D74714"]
    grid = "#D9D9D9"
    ink = "#161616"
    limit_color = "#B62F67"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": ink,
        "axes.labelcolor": ink, "axes.edgecolor": ink, "xtick.color": ink, "ytick.color": ink,
    })
    fig, axes = plt.subplots(2, 4, figsize=(14.32, 6.0))
    x = np.arange(4)
    for column, panel in enumerate(panels):
        fig.text(0.14 + column * 0.24, 0.865, panel["simulator"] + ":",
                 ha="center", va="center", fontsize=12, weight="bold")
        for row, key in enumerate(("primary", "secondary")):
            axis = axes[row, column]
            seed_values = np.asarray(panel[key], dtype=float)
            mean = float(seed_values.mean())
            displayed = np.append(seed_values, mean)
            bars = axis.bar(x, displayed, width=0.66, color=colors, edgecolor=ink,
                            linewidth=0.65, zorder=3)
            interval = np.asarray(panel[f"{key}_ci"], dtype=float)
            error = np.asarray([[max(0.0, mean - interval[0])],
                                [max(0.0, interval[1] - mean)]])
            axis.errorbar([3], [mean], yerr=error, fmt="none", ecolor=ink,
                          elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=5)
            axis.axhline(0, color=ink, linewidth=0.85, zorder=2)
            if key == "secondary":
                axis.axhline(panel["safety_limit"], color=limit_color, linewidth=1.0,
                             linestyle=(0, (4, 2)), zorder=2)
            bounds = [*displayed, *interval, 0.0]
            if key == "secondary":
                bounds.append(panel["safety_limit"])
            low, high = min(bounds), max(bounds)
            span = max(high - low, 0.15)
            axis.set_ylim(low - 0.24 * span, high + 0.35 * span)
            axis.set_xlim(-0.65, 3.65)
            axis.set_xticks(x, ["1", "2", "3", "Mean"])
            axis.set_title(panel[f"{key}_title"] + " ↑", fontsize=10, weight="normal", pad=7)
            axis.grid(axis="y", color=grid, linewidth=0.7, zorder=0)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_linewidth(0.85)
            axis.tick_params(labelsize=8.5, width=0.8, length=3, pad=3)
            offset = 0.07 * span
            label_y = interval[1] + offset if mean >= 0 else interval[0] - offset
            axis.text(3, label_y, f"{mean:+.2f}", ha="center",
                      va="bottom" if mean >= 0 else "top", fontsize=8.7, weight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=ink, linewidth=0.65,
                             label=label)
               for color, label in zip(colors, ("Seed 1", "Seed 2", "Seed 3", "Mean"), strict=True)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.58, 0.995),
               ncol=4, frameon=False, fontsize=10, handlelength=0.8,
               handletextpad=0.35, columnspacing=1.3)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.10,
                        wspace=0.34, hspace=0.78)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return json.loads(json.dumps(
        panels,
        default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item(),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/reproduction/full/ablations"))
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    parser.add_argument("--report", type=Path, default=Path("report/paper_figures"))
    args = parser.parse_args()

    results = read(args.run / "results.json")
    audit = read(args.run / "audit.json")
    if audit["status"] != "pass" or results["failures"]:
        raise ValueError("The source run has not passed its integrity audit")
    if results["seeds"] != [1154929522, 2035845825, 1193624495]:
        raise ValueError("Unexpected seed set")

    bank_direct = direct_bank_comparison(args.run, results)
    focused_path = args.paper / "tables" / "focused_results.tex"
    ablation_path = args.paper / "tables" / "ablations.tex"
    focused_path.write_text(focused_table(results, bank_direct), encoding="utf-8")
    ablation_path.write_text(ablation_table(results), encoding="utf-8")

    figure_paths = [args.paper / "img" / "fig03_focused_results.png",
                    args.report / "fig03_focused_results.png"]
    panels = focused_figure(results, bank_direct, figure_paths)
    manifest = {
        "source_run": str(args.run),
        "source_results_sha256": digest(args.run / "results.json"),
        "source_audit_sha256": digest(args.run / "audit.json"),
        "seeds": results["seeds"],
        "bank_direct_comparison": bank_direct,
        "figure_panels": panels,
        "outputs": {str(path): digest(path) for path in [focused_path, ablation_path, *figure_paths]},
    }
    manifest_path = args.report / "regression_paper_artifacts.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tables": [str(focused_path), str(ablation_path)],
                      "figures": [str(path) for path in figure_paths],
                      "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
