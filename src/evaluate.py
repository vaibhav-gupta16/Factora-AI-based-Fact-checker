"""
Evaluation: metrics comparing weighted vs baseline against ground truth.
Uses plain matplotlib + sklearn's built-in ConfusionMatrixDisplay only
(no seaborn/scipy dependency) so it can't be blocked by a broken scipy install.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

LABELS = ["false", "mixed", "true"]


def _metrics_block(y_true, y_pred, name: str) -> str:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    return (f"--- {name} ---\nAccuracy: {acc:.3f}\nMacro Precision: {prec:.3f}\n"
            f"Macro Recall: {rec:.3f}\nMacro F1: {f1:.3f}\n\n{report}\n")


def _plot_confusion(y_true, y_pred, title: str, filename: str):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(config.PLOTS_DIR / filename, dpi=150); plt.close(fig)


def evaluate() -> None:
    path = config.RESULTS_DIR / "pipeline_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.pipeline first.")
    # dtype=str is required here: pandas silently infers boolean dtype for any
    # column containing only the strings "true"/"false" (e.g. a small sample
    # with no "mixed" claims), which breaks comparison against prediction
    # columns that also contain "unverifiable". Forcing str prevents this.
    df = pd.read_csv(path, dtype={
        "true_label_3way": str, "pred_label_weighted": str, "pred_label_baseline": str,
    })
    for col in ["true_label_3way", "pred_label_weighted", "pred_label_baseline"]:
        df[col] = df[col].str.lower()
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No successful pipeline runs. Check API keys and quota.")

    report_lines = [
        "FACTORA — EVALUATION REPORT",
        f"Claims evaluated: {len(ok)} / {len(df)} attempted "
        f"({len(df) - len(ok)} failed — see status column)\n",
    ]
    n_unverif_w = (ok["pred_label_weighted"] == "unverifiable").sum()
    n_unverif_b = (ok["pred_label_baseline"] == "unverifiable").sum()
    if n_unverif_w or n_unverif_b:
        report_lines.append(
            f"Note: {n_unverif_w} claim(s) 'unverifiable' in weighted run, "
            f"{n_unverif_b} in baseline run — no evidence found, no guess forced.\n"
        )

    report_lines.append(_metrics_block(ok["true_label_3way"], ok["pred_label_weighted"],
                                        "Credibility-WEIGHTED evidence selection"))
    report_lines.append(_metrics_block(ok["true_label_3way"], ok["pred_label_baseline"],
                                        "BASELINE (equal-weight, no credibility signal)"))

    acc_w = accuracy_score(ok["true_label_3way"], ok["pred_label_weighted"])
    acc_b = accuracy_score(ok["true_label_3way"], ok["pred_label_baseline"])
    delta = acc_w - acc_b
    report_lines.append(
        f"--- Headline comparison ---\nWeighted accuracy: {acc_w:.3f}\n"
        f"Baseline accuracy: {acc_b:.3f}\nDifference: {delta:+.3f}\n"
    )

    report_text = "\n".join(report_lines)
    (config.RESULTS_DIR / "evaluation_report.txt").write_text(report_text)
    print(report_text)

    _plot_confusion(ok["true_label_3way"], ok["pred_label_weighted"],
                     "Confusion Matrix — Credibility-Weighted", "confusion_weighted.png")
    _plot_confusion(ok["true_label_3way"], ok["pred_label_baseline"],
                     "Confusion Matrix — Baseline", "confusion_baseline.png")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Weighted", "Baseline"], [acc_w, acc_b], color=["#2b6cb0", "#a0aec0"])
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.set_title("Credibility weighting vs baseline")
    for i, v in enumerate([acc_w, acc_b]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "accuracy_comparison.png", dpi=150); plt.close(fig)

    ok.to_csv(config.RESULTS_DIR / "tableau_export.csv", index=False)
    print(f"[saved] plots, report, and tableau_export.csv")


if __name__ == "__main__":
    evaluate()
