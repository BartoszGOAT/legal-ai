"""Arène d'évaluation humaine (brief §4.5), interface Gradio locale.

Réplique le protocole Derby LLM: question + contexte + 2 réponses anonymisées
et présentées dans un ordre aléatoire, 4 boutons (A meilleure / B meilleure /
Match nul / Aucune). Comparaisons prioritaires : C2 (RAG) vs C3 (fine-tune
seul) -- le duel central de la question de recherche -- puis C4 vs C2.

Nécessite generation_results.json (produit par le kernel de génération). Ne
nécessite AUCUN GPU: tourne en local sur les réponses déjà générées.

Usage: python -m src.arena_app --pair C2_rag:C3_finetune --annotator bartosz
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from . import config

N_ARENA_QUESTIONS = 60
PRIORITY_PAIRS = [
    ("C2_rag", "C3_finetune"),  # duel central de la question de recherche
    ("C4_finetune_rag", "C2_rag"),
]


def load_generation_results() -> dict:
    path = config.RESULTS_DIR / "generation_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent. L'arène nécessite les réponses générées par le kernel de génération."
        )
    with open(path) as f:
        return json.load(f)


def build_arena_items(gen: dict, config_a: str, config_b: str, seed: int = config.SEED) -> list[dict]:
    questions = gen["questions"]
    n_total = len(questions)
    rng = random.Random(seed)
    sample_idx = sorted(rng.sample(range(n_total), min(N_ARENA_QUESTIONS, n_total)))

    answers_a = gen["configs"][config_a]["answers"]
    answers_b = gen["configs"][config_b]["answers"]

    items = []
    for i in sample_idx:
        # Randomise l'ordre d'affichage (anonymisation gauche/droite)
        swap = rng.random() < 0.5
        left, right = (answers_a[i], answers_b[i]) if not swap else (answers_b[i], answers_a[i])
        left_cfg, right_cfg = (config_a, config_b) if not swap else (config_b, config_a)
        items.append(
            {
                "question_index": i,
                "question": questions[i],
                "left_answer": left,
                "right_answer": right,
                "left_config": left_cfg,  # non affiché à l'annotateur, utilisé pour désanonymiser après export
                "right_config": right_cfg,
            }
        )
    return items


def run_app(config_a: str, config_b: str, annotator: str):
    import gradio as gr

    gen = load_generation_results()
    items = build_arena_items(gen, config_a, config_b)
    state = {"idx": 0, "votes": []}

    out_dir = config.RESULTS_DIR / "arena_votes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"arena_{config_a}_vs_{config_b}_{annotator}.csv"

    def render(idx):
        if idx >= len(items):
            return (
                "Terminé — merci ! Toutes les comparaisons ont été votées.",
                "",
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        item = items[idx]
        progress = f"Question {idx + 1}/{len(items)}"
        q_text = f"**{progress}**\n\n**Question :** {item['question']}"
        return q_text, item["left_answer"], item["right_answer"], gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

    def vote(choice):
        idx = state["idx"]
        if idx >= len(items):
            return render(idx)
        item = items[idx]
        state["votes"].append(
            {
                "annotator": annotator,
                "question_index": item["question_index"],
                "config_a": config_a,
                "config_b": config_b,
                "left_config": item["left_config"],
                "right_config": item["right_config"],
                "choice": choice,  # "left", "right", "tie", "neither"
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Sauvegarde incrémentale (résistant à une fermeture accidentelle du navigateur)
        write_header = not out_path.exists()
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(state["votes"][-1].keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(state["votes"][-1])
        state["idx"] += 1
        return render(state["idx"])

    with gr.Blocks(title="Arène TER — évaluation humaine") as demo:
        gr.Markdown(f"## Arène : {config_a} vs {config_b} — annotateur : {annotator}")
        question_md = gr.Markdown()
        with gr.Row():
            left_box = gr.Textbox(label="Réponse A", lines=10, interactive=False)
            right_box = gr.Textbox(label="Réponse B", lines=10, interactive=False)
        with gr.Row():
            btn_a = gr.Button("A est meilleure")
            btn_b = gr.Button("B est meilleure")
            btn_tie = gr.Button("Match nul")
            btn_neither = gr.Button("Aucune (les deux mauvaises)")

        demo.load(lambda: render(state["idx"]), outputs=[question_md, left_box, right_box, btn_a, btn_b, btn_tie, btn_neither])
        btn_a.click(lambda: vote("left"), outputs=[question_md, left_box, right_box, btn_a, btn_b, btn_tie, btn_neither])
        btn_b.click(lambda: vote("right"), outputs=[question_md, left_box, right_box, btn_a, btn_b, btn_tie, btn_neither])
        btn_tie.click(lambda: vote("tie"), outputs=[question_md, left_box, right_box, btn_a, btn_b, btn_tie, btn_neither])
        btn_neither.click(lambda: vote("neither"), outputs=[question_md, left_box, right_box, btn_a, btn_b, btn_tie, btn_neither])

    demo.launch()


def compute_inter_annotator_agreement(config_a: str, config_b: str, annotators: list[str]) -> dict:
    """Cohen's Kappa par paire d'annotateurs sur les votes de l'arène."""
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    out_dir = config.RESULTS_DIR / "arena_votes"
    dfs = {}
    for ann in annotators:
        path = out_dir / f"arena_{config_a}_vs_{config_b}_{ann}.csv"
        if path.exists():
            dfs[ann] = pd.read_csv(path).set_index("question_index")["choice"]

    results = {}
    from itertools import combinations

    for a1, a2 in combinations(dfs.keys(), 2):
        common = dfs[a1].index.intersection(dfs[a2].index)
        if len(common) == 0:
            continue
        kappa = cohen_kappa_score(dfs[a1].loc[common], dfs[a2].loc[common])
        agreement = float((dfs[a1].loc[common] == dfs[a2].loc[common]).mean())
        results[f"{a1}_vs_{a2}"] = {"cohen_kappa": float(kappa), "exact_agreement": agreement, "n_common": len(common)}
    return results


def compute_bradley_terry_ranking(pairs: list[tuple[str, str]] = PRIORITY_PAIRS) -> dict:
    """Classement des configurations à partir de TOUS les votes d'arène
    disponibles (toutes paires, tous annotateurs), par modèle de Bradley-Terry
    -- la méthode utilisée par Chatbot Arena/LMSYS (dont Derby LLM et nous nous
    inspirons tous deux pour le protocole d'arène), réimplémentée ici via une
    estimation par maximum de vraisemblance (pas copiée).

    P(i bat j) = exp(s_i) / (exp(s_i) + exp(s_j)). Donne un score de force
    relative sur une échelle unique, plus interprétable qu'un taux de victoire
    par paire isolée quand plusieurs configurations sont en jeu. Un match nul
    compte pour 0.5 victoire de chaque côté ; "aucune" (neither) est exclu
    (ne renseigne pas sur la force relative des deux réponses).
    """
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    out_dir = config.RESULTS_DIR / "arena_votes"
    configs_involved = sorted({c for pair in pairs for c in pair})
    idx = {c: i for i, c in enumerate(configs_involved)}
    n = len(configs_involved)
    wins = np.zeros((n, n))  # wins[i, j] = nb de victoires (ponderees) de i sur j

    n_votes_used = 0
    for config_a, config_b in pairs:
        for path in out_dir.glob(f"arena_{config_a}_vs_{config_b}_*.csv"):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                if row["choice"] == "neither":
                    continue
                winner_cfg = None
                if row["choice"] == "tie":
                    wins[idx[row["left_config"]], idx[row["right_config"]]] += 0.5
                    wins[idx[row["right_config"]], idx[row["left_config"]]] += 0.5
                elif row["choice"] == "left":
                    winner_cfg = row["left_config"]
                elif row["choice"] == "right":
                    winner_cfg = row["right_config"]
                if winner_cfg is not None:
                    loser_cfg = row["right_config"] if winner_cfg == row["left_config"] else row["left_config"]
                    wins[idx[winner_cfg], idx[loser_cfg]] += 1.0
                n_votes_used += 1

    if n_votes_used == 0:
        return {"error": "Aucun vote d'arène trouvé dans results/arena_votes/"}

    def neg_log_likelihood(strengths):
        s = np.concatenate([[0.0], strengths])  # ancre la premiere config a 0 (identifiabilite)
        ll = 0.0
        for i in range(n):
            for j in range(n):
                if wins[i, j] > 0:
                    ll += wins[i, j] * (s[i] - np.logaddexp(s[i], s[j]))
        return -ll

    x0 = np.zeros(n - 1)
    result = minimize(neg_log_likelihood, x0, method="BFGS")
    strengths = np.concatenate([[0.0], result.x])
    # Normalise pour lisibilite (comme un score Elo, echelle arbitraire)
    strengths_display = 1000 + 400 * (strengths - strengths.mean())

    ranking = sorted(
        [{"config": c, "strength_bt": float(strengths[idx[c]]), "score_elo_like": float(strengths_display[idx[c]])} for c in configs_involved],
        key=lambda r: -r["strength_bt"],
    )
    return {
        "ranking": ranking,
        "n_votes_used": n_votes_used,
        "converged": bool(result.success),
        "note": "config non presente dans une comparaison d'arene = absente du classement (ex: C1 si jamais oppose)",
    }


def compute_full_agreement_distribution(config_a: str, config_b: str, annotators: list[str]) -> dict:
    """Distribution d'accord entre TOUS les annotateurs sur chaque question
    (unanime / majoritaire / aucun accord), pas seulement par paire.
    S'inspire de la Figure 7 de Derby LLM (Bouvard et al., APIA@PFIA 2024) --
    répliquée ici parce qu'avec exactement 3 annotateurs comme eux, le Kappa
    par paire seul masque si le désaccord est généralisé ou concentré sur
    quelques questions limites."""
    import pandas as pd

    out_dir = config.RESULTS_DIR / "arena_votes"
    dfs = {}
    for ann in annotators:
        path = out_dir / f"arena_{config_a}_vs_{config_b}_{ann}.csv"
        if path.exists():
            dfs[ann] = pd.read_csv(path).set_index("question_index")["choice"]

    if len(dfs) < 3:
        return {"error": f"besoin de {len(annotators)} annotateurs, {len(dfs)} disponibles"}

    common_idx = None
    for s in dfs.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)

    unanimous = majority = none = 0
    per_question = {}
    for q in common_idx:
        votes = [dfs[ann].loc[q] for ann in dfs]
        counts = pd.Series(votes).value_counts()
        top_count = counts.iloc[0]
        if top_count == len(votes):
            unanimous += 1
            level = "unanime"
        elif top_count >= 2:
            majority += 1
            level = "majoritaire"
        else:
            none += 1
            level = "aucun"
        per_question[int(q)] = {"votes": votes, "agreement_level": level}

    n = len(common_idx)
    return {
        "n_questions": n,
        "pct_unanimous": 100 * unanimous / n if n else None,
        "pct_majority_only": 100 * majority / n if n else None,
        "pct_no_agreement": 100 * none / n if n else None,
        "per_question": per_question,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="C2_rag:C3_finetune", help="config_a:config_b")
    parser.add_argument("--annotator", required=True)
    args = parser.parse_args()
    ca, cb = args.pair.split(":")
    run_app(ca, cb, args.annotator)
