"""Génère les figures PDF vectorielles à partir des JSON de results/."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

from . import config

RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = config.FIGURES_DIR


def fig_recall_at_k():
    path = RESULTS_DIR / "retrieval_results.json"
    if not path.exists():
        print("retrieval_results.json absent, figure ignorée")
        return
    with open(path) as f:
        retrieval = json.load(f)

    ks = config.RECALL_KS
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, run in retrieval["runs"].items():
        ys = [run.get(f"recall@{k}", float("nan")) for k in ks]
        ax.plot(ks, ys, marker="o", label=name)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title(f"Recall@k par méthode de retrieval ({retrieval['n_test_questions']} questions test)")
    ax.legend()
    ax.grid(alpha=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "recall_at_k_comparison.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_category_distribution():
    from . import data

    test_df = data.load_questions_test()
    counts = test_df["category"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(counts.index, counts.values, color="#4C72B0")
    ax.set_xlabel("Nombre de questions (split test, n=222)")
    ax.set_title("Distribution des catégories juridiques — BSARD test")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.5, i, str(v), va="center")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "bsard_category_distribution.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


if __name__ == "__main__":
    fig_recall_at_k()
    fig_category_distribution()
