"""Build the common benchmark table and gain heatmap from one audited run."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


FAMILIES = ("Regression", "RandomForest", "CatBoost", "XGBoost")
FAMILY_LABELS = ("Regression", "Random Forest", "CatBoost", "XGBoost")
RL = (("TabularQ", "Q learning"), ("FittedQ", "Fitted Q"),
      ("CQL_0.1", r"CQL $\alpha{=}0.1$"),
      ("CQL_0.5", r"CQL $\alpha{=}0.5$"), ("PPO", "PPO"))
RL_FIGURE_LABELS = ("Q learning", "Fitted Q", "CQL 0.1", "CQL 0.5", "PPO")
CELL = "load_1.2_effect_0.25"
INK = "#161616"
LOSS_RED = "#B2182B"
GAIN_GREEN = "#1A9850"
ML_COLORS = ("#1963B5", "#008391", "#8425A5", "#D74714")
RL_COLORS = (*ML_COLORS, "#B62F67")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.labelweight": "bold", "axes.labelsize": 9.5,
    "axes.unicode_minus": True, "pdf.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
    "legend.frameon": False,
})


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values, scale=1, digits=1):
    values = np.asarray(values, dtype=float) * scale
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("The paper table requires three finite seed values")
    return rf"${values.mean():.{digits}f}\pm{values.std(ddof=1):.{digits}f}$"


def metric_values(aggregate, domain, policy, metric):
    block = aggregate[domain]["negotiation" if domain == "simbank" else CELL]
    return block["policies"][policy][metric]["by_seed"]


def write_table(aggregate, target: Path):
    route_metrics = {"insurance": "queue", "bpisim": "queue", "manufacturing": "tail_queue"}
    rows = [r"\multicolumn{6}{@{}l}{\textit{Deterministic reference}} \\",
            "FIFO / bank rule & " + " & ".join([
                stats(metric_values(aggregate, domain, "FIFO", metric))
                for domain, metric in route_metrics.items()
            ] + [stats(metric_values(aggregate, "simbank", "Bank", "profit"), digits=0),
                 stats(metric_values(aggregate, "simbank", "Bank", "refusals"), 100)]) + r" \\",
            r"\midrule\multicolumn{6}{@{}l}{\textit{ML only}} \\"]
    for family, label in zip(FAMILIES, FAMILY_LABELS, strict=True):
        values = [stats(metric_values(aggregate, domain, f"{family}_ML_slot", metric))
                  for domain, metric in route_metrics.items()]
        values += [stats(metric_values(aggregate, "simbank", f"{family}_DP_500", "profit"), digits=0),
                   stats(metric_values(aggregate, "simbank", f"{family}_DP_500", "refusals"), 100)]
        rows.append(label + " & " + " & ".join(values) + r" \\")
    rows.append(r"\midrule\multicolumn{6}{@{}l}{\textit{RL with best validation selected forecaster}} \\")
    for policy, label in RL:
        values = [stats(metric_values(aggregate, domain, policy, metric))
                  for domain, metric in route_metrics.items()]
        values += [stats(metric_values(aggregate, "simbank", policy, "profit"), digits=0),
                   stats(metric_values(aggregate, "simbank", policy, "refusals"), 100)]
        rows.append(label + " & " + " & ".join(values) + r" \\")
    text = "\n".join([
        r"\begin{table*}[!tbp]",
        r"\caption{Common benchmark results across model and controller families.}",
        r"\label{tab:screening}", r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.14}",
        r"\begin{tabularx}{\textwidth}{@{}Yrrrrr@{}}", r"\toprule",
        r"Method & Claims wait $\downarrow$ & BPI wait $\downarrow$ & Factory tail $\downarrow$ & Bank profit $\uparrow$ & Bank refusals $\downarrow$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabularx}", r"\par\smallskip",
        r"\begin{minipage}{\textwidth}\footnotesize",
        ("Means $\\pm$ SD across three random seeds. Routing columns report minutes. "
         "SimBank ML rows use refusal aware planners. RL rows use the best forecast family selected on validation, "
         "not every ML and RL pairing. Routing uses 256 workloads per seed. SimBank uses 30,000 cases per seed."),
        r"\end{minipage}", r"\end{table*}", "",
    ])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def heatmap_data(aggregate):
    labels = ["Basic rule", "Regression", "Random Forest", "CatBoost", "XGBoost",
              "Q learning", "Fitted Q", "CQL (0.1)", "CQL (0.5)", "PPO"]
    domains = (("insurance", "queue", "FIFO"), ("bpisim", "queue", "FIFO"),
               ("manufacturing", "tail_queue", "FIFO"), ("simbank", "profit", "Bank"))
    matrix = np.empty((len(labels), len(domains)))
    for column, (domain, metric, reference) in enumerate(domains):
        block = aggregate[domain]["negotiation" if domain == "simbank" else CELL]["policies"]
        suffix = "DP_500" if domain == "simbank" else "ML_slot"
        policies = [reference, *[f"{family}_{suffix}" for family in FAMILIES], *[name for name, _ in RL]]
        for row, policy in enumerate(policies):
            candidate = block[policy][metric]["mean"]
            baseline = block[reference][metric]["mean"]
            sign = 1 if metric == "profit" else -1
            matrix[row, column] = sign * 100 * (candidate - baseline) / baseline
    return matrix


def write_heatmap(aggregate, target: Path):
    matrix = heatmap_data(aggregate)
    limit = max(10.0, np.ceil(np.abs(matrix).max() / 10) * 10)
    palette = LinearSegmentedColormap.from_list("process_gain", [LOSS_RED, "#FFFFFF", GAIN_GREEN])
    norm = Normalize(-limit, limit)
    fig = plt.figure(figsize=(7.16, 2.72))
    for start, end, left, width, title, labels, colors in [
        (1, 5, .17, .325, "(a) ML only:", list(FAMILY_LABELS), ML_COLORS),
        (5, 10, .56, .405, "(b) RL:", RL_FIGURE_LABELS, RL_COLORS),
    ]:
        ax = fig.add_axes([left, .36, width, .43])
        values = matrix[start:end].T
        image = ax.imshow(values, cmap=palette, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=10, weight="bold", pad=10)
        ax.set_xticks(np.arange(end - start), labels=labels, rotation=30, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=5, labelsize=8.5)
        ax.set_yticks(np.arange(4), labels=["SimClaims", "BPISim", "SimFactory", "SimBank"] if start == 1 else [])
        ax.tick_params(axis="y", length=0, pad=8, labelsize=9)
        for label, color in zip(ax.get_xticklabels(), colors, strict=True):
            label.set_color(color)
            label.set_fontweight("semibold")
        ax.set_xticks(np.arange(-.5, end - start, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(4):
            for column in range(end - start):
                value = values[row, column]
                rgba = palette(norm(value))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                ax.text(column, row, f"{value:+.1f}", ha="center", va="center", fontsize=9,
                        weight="semibold", color="white" if luminance < .50 else INK)
    cax = fig.add_axes([.39, .945, .36, .022])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.set_ticks([-limit, -limit / 2, 0, limit / 2, limit])
    bar.ax.tick_params(labelsize=8, length=2, pad=2)
    bar.outline.set_linewidth(.65)
    fig.text(.36, .956, "Loss", ha="right", va="center", fontsize=8.5, color=LOSS_RED)
    fig.text(.78, .956, "Gain", ha="left", va="center", fontsize=8.5, color=GAIN_GREEN)
    fig.text(.17, .956, "Gain (%)", ha="left", va="center", fontsize=9)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(target, dpi=600, facecolor="white")
    plt.close(fig)
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    parser.add_argument("--report", type=Path, default=Path("report/paper_figures"))
    args = parser.parse_args()
    aggregate = read(args.run / "aggregate.json")
    table = args.paper / "tables/algorithm_screening.tex"
    figure = args.paper / "img/fig02_policy_gains.png"
    write_table(aggregate, table)
    matrix = write_heatmap(aggregate, figure)
    args.report.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(args.run / "aggregate.json"), "gain_percent": matrix.tolist(),
                "outputs": {str(path): sha256(path.read_bytes()).hexdigest() for path in (table, figure)}}
    (args.report / "common_artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
