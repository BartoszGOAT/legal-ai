"""Génère les tableaux .tex (booktabs) à partir des JSON de results/."""
from __future__ import annotations

import json

from . import config

RESULTS_DIR = config.RESULTS_DIR
TABLES_DIR = config.TABLES_DIR


def table_retrieval():
    path = RESULTS_DIR / "retrieval_results.json"
    if not path.exists():
        print("retrieval_results.json absent, tableau ignoré")
        return
    with open(path) as f:
        retrieval = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparaison des méthodes de retrieval sur les " + str(retrieval["n_test_questions"]) + r" questions test de BSARD.}",
        r"\label{tab:retrieval-comparison}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Méthode & Recall@1 & Recall@3 & Recall@5 & Recall@10 & Recall@20 & MRR@10 & nDCG@10 \\",
        r"\midrule",
    ]
    name_map = {
        "bm25": "BM25",
        "mpnet": "mpnet (baseline)",
        "e5_large": "multilingual-e5-large",
        "hybrid_bm25_e5": "Hybride (RRF)",
    }
    for key, run in retrieval["runs"].items():
        name = name_map.get(key, key)
        lines.append(
            f"{name} & {run.get('recall@1', float('nan')):.3f} & {run.get('recall@3', float('nan')):.3f} "
            f"& {run.get('recall@5', float('nan')):.3f} & {run.get('recall@10', float('nan')):.3f} "
            f"& {run.get('recall@20', float('nan')):.3f} & {run['mrr@10']:.3f} & {run['ndcg@10']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "retrieval_comparison.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tableau écrit: {out}")


if __name__ == "__main__":
    table_retrieval()
