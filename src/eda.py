"""
Exploratory Data Analysis — runs entirely offline on downloaded data.
Uses plain matplotlib only (no seaborn/scipy dependency) so it can't be
blocked by a broken or slow scipy install.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_liar_all, load_mbfc

_COLOR = "#2b6cb0"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.axisbelow"] = True


def _bar_from_counts(ax, labels, counts, title, rotation=0):
    ax.bar(labels, counts, color=_COLOR)
    ax.set_title(title)
    if rotation:
        ax.tick_params(axis="x", rotation=rotation)


def plot_label_distribution(liar):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order6 = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    counts6 = liar["label_6way"].value_counts().reindex(order6).fillna(0)
    _bar_from_counts(axes[0], order6, counts6.values, "LIAR — original 6-way labels", rotation=30)

    order3 = ["false", "mixed", "true"]
    counts3 = liar["label_3way"].value_counts().reindex(order3).fillna(0)
    _bar_from_counts(axes[1], order3, counts3.values, "LIAR — collapsed to 3-way")

    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "eda_label_distribution.png", dpi=150)
    plt.close(fig)


def plot_claim_length(liar):
    liar = liar.copy()
    liar["n_words"] = liar["statement"].str.split().apply(len)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(liar["n_words"], bins=30, color=_COLOR, edgecolor="white")
    ax.set_title("Claim length distribution")
    ax.set_xlabel("Words per statement")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "eda_claim_length.png", dpi=150)
    plt.close(fig)
    return liar["n_words"].describe()


def plot_mbfc_distribution(mbfc):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order_fact = ["low", "mixed", "high"]
    counts_fact = mbfc["fact"].value_counts().reindex(order_fact).fillna(0)
    _bar_from_counts(axes[0], order_fact, counts_fact.values, "MBFC — factuality rating distribution")

    bias_order = ["left", "center", "right"]
    present = [b for b in bias_order if b in mbfc["bias"].unique()]
    counts_bias = mbfc["bias"].value_counts().reindex(present).fillna(0)
    _bar_from_counts(axes[1], present, counts_bias.values, "MBFC — political bias distribution")

    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "eda_mbfc_distribution.png", dpi=150)
    plt.close(fig)


def run_eda():
    liar = load_liar_all()
    mbfc = load_mbfc()
    plot_label_distribution(liar)
    length_stats = plot_claim_length(liar)
    plot_mbfc_distribution(mbfc)
    summary = (
        f"EDA SUMMARY\n===========\nLIAR claims total: {len(liar)}\n"
        f"  3-way label counts:\n{liar['label_3way'].value_counts().to_string()}\n\n"
        f"  Claim length (words): mean={length_stats['mean']:.1f}, median={length_stats['50%']:.0f}\n"
        f"  Distinct speakers: {liar['speaker'].nunique()}\n\n"
        f"MBFC rated domains: {len(mbfc)}\n{mbfc['fact'].value_counts().to_string()}\n"
    )
    (config.RESULTS_DIR / "eda_summary.txt").write_text(summary)
    print(summary)
    print(f"[saved] plots and eda_summary.txt")


if __name__ == "__main__":
    run_eda()
