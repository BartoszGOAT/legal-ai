"""Kaggle kernel (GPU T4): chatbot interactif Gradio, questions juridiques
libres (droit belge francophone). Version de base : C1 (zero-shot) et
C2 (RAG) uniquement, entièrement autonome (aucune dépendance à un kernel_source
ou à un artefact RunPod). C3/C4 (fine-tuning) seront ajoutés une fois
l'adaptateur LoRA récupéré depuis RunPod.

À lancer en session INTERACTIVE Kaggle (pas "Save & Run All" en mode batch:
le lien Gradio doit rester actif tant que la session tourne). Le lien public
temporaire (*.gradio.live, valable ~72h) s'affiche dans les logs de la
cellule une fois le modèle chargé.

L'index de retrieval (embeddings e5-large sur les 22 633 articles BSARD)
est recalculé au démarrage -- ~2-5 min sur T4 -- au lieu d'être chargé
depuis un kernel_source, pour ne dépendre que de HuggingFace (aucun accès
à un kernel tiers requis).
"""
import subprocess
import os
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes", "accelerate", "sentence-transformers", "gradio"],
    check=True,
)

import numpy as np
import pandas as pd
import torch
import urllib.request
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import gradio as gr

# Chemins portables: par defaut Kaggle, surchargeables via variables
# d'environnement pour tourner ailleurs (RunPod, etc.) sans modifier le script.
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")

BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
TOP_K = 5
MAX_NEW_TOKENS = 450

SYSTEM_PROMPT_FR = (
    "Tu es un assistant juridique spécialisé en droit belge francophone. "
    "Réponds à la question posée en français, de manière claire et rédigée. "
    "Si un contexte juridique t'est fourni, appuie-toi dessus. "
    "Termine impérativement ta réponse par une citation précise de l'article "
    "sur lequel tu t'appuies, au format : \"Article <numéro>\". "
    "Si tu ne connais pas la réponse ou si le contexte ne permet pas de répondre "
    "avec certitude, réponds explicitement \"Je ne sais pas\" plutôt que d'inventer "
    "une réponse ou une référence."
)

CONFIGS = {
    "C1_zero_shot": {"use_rag": False},
    "C2_rag": {"use_rag": True},
}

# --- Corpus + index de retrieval (calculé au demarrage) ---
DATA_DIR = Path(f"{WORK_DIR}/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
HF_BASE = "https://huggingface.co/datasets/maastrichtlawtech/bsard/resolve/main"
articles_path = DATA_DIR / "articles.csv"
if not articles_path.exists():
    urllib.request.urlretrieve(f"{HF_BASE}/articles.csv", articles_path)
articles = pd.read_csv(articles_path)
article_lookup = articles.set_index("id").to_dict(orient="index")
doc_ids = articles["id"].tolist()
doc_texts = articles["article"].fillna("").tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

retriever = SentenceTransformer(EMBEDDING_MODEL, device=device)
print(f"calcul des embeddings sur {len(doc_texts)} articles...")
doc_embeddings = retriever.encode(
    [f"passage: {t}" for t in doc_texts],
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)
print("index de retrieval pret")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model.eval()
print("modele charge")


def retrieve_context(question: str, top_k: int = TOP_K) -> list[dict]:
    q_emb = retriever.encode(f"query: {question}", normalize_embeddings=True, convert_to_numpy=True)
    scores = doc_embeddings @ q_emb
    top_idx = np.argsort(-scores)[:top_k]
    return [article_lookup[doc_ids[i]] for i in top_idx if doc_ids[i] in article_lookup]


def build_messages(question: str, context_articles: list[dict] | None) -> list[dict]:
    user_parts = []
    if context_articles:
        ctx = "\n".join(f"- {a['reference']} : {a['article']}" for a in context_articles)
        user_parts.append(f"Contexte juridique :\n{ctx}\n")
    user_parts.append(f"Question : {question}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT_FR},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


@torch.no_grad()
def generate_answer(question: str, config_name: str) -> tuple[str, str]:
    cfg = CONFIGS[config_name]
    context_articles = retrieve_context(question) if cfg["use_rag"] else None
    messages = build_messages(question, context_articles)
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    output_ids = model.generate(input_ids, **gen_kwargs)

    generated = output_ids[0][input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if context_articles:
        sources = "\n\n**Articles récupérés (RAG) :**\n" + "\n".join(
            f"- {a['reference']}" for a in context_articles
        )
    else:
        sources = ""
    return answer, sources


def respond(question: str, config_name: str, history: list):
    history = history or []
    if not question.strip():
        return history, ""
    answer, sources = generate_answer(question, config_name)
    history = history + [(question, answer + sources)]
    return history, ""


with gr.Blocks(title="Chatbot juridique TER — C1/C2") as demo:
    gr.Markdown("## Chatbot juridique (droit belge francophone) — choisis la configuration")
    config_dropdown = gr.Dropdown(
        choices=list(CONFIGS.keys()), value="C2_rag", label="Configuration",
        info="C1 zero-shot | C2 RAG (fine-tuning C3/C4 a venir)",
    )
    chatbot = gr.Chatbot(height=500)
    question_box = gr.Textbox(label="Ta question", placeholder="Ex: Quels sont mes droits en cas de licenciement ?")
    with gr.Row():
        submit_btn = gr.Button("Envoyer", variant="primary")
        clear_btn = gr.Button("Effacer la conversation")

    submit_btn.click(respond, inputs=[question_box, config_dropdown, chatbot], outputs=[chatbot, question_box])
    question_box.submit(respond, inputs=[question_box, config_dropdown, chatbot], outputs=[chatbot, question_box])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, question_box])

demo.launch(share=True)
