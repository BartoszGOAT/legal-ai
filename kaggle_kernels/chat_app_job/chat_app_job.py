"""Kaggle kernel (GPU T4): chatbot interactif Gradio, questions juridiques
libres (droit belge francophone), au choix entre les 4 configurations C1-C4
définies dans src/config.py.

À lancer en session INTERACTIVE Kaggle (pas "Save & Run All" en mode batch:
le lien Gradio doit rester actif tant que la session tourne). Le lien public
temporaire (*.gradio.live, valable ~72h) s'affiche dans les logs de la
cellule une fois le modèle chargé.

Dépend des sorties de deux kernels attachés via kernel_sources (mêmes
sources que generation_job.py):
  - ter-bsard-retrieval-eval  -> index e5-large (index_e5_large.npz)
  - ter-bsard-qlora-finetune  -> adaptateur LoRA (adapter/)

Un seul modèle de base est chargé en mémoire (4-bit). Le passage C1/C2
<-> C3/C4 se fait en activant/désactivant l'adaptateur LoRA à la volée
(peft `disable_adapter()`), pas en rechargeant un second modèle -- évite
de doubler l'empreinte VRAM sur un T4 (16 Go).
"""
import glob
import subprocess
import os
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "peft", "bitsandbytes", "accelerate", "sentence-transformers", "gradio"],
    check=True,
)

import numpy as np
import pandas as pd
import torch
import urllib.request
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import gradio as gr

# Chemins portables: par defaut Kaggle, surchargeables via variables
# d'environnement pour tourner ailleurs (RunPod, etc.) sans modifier le script.
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
INPUT_DIR = os.environ.get("INPUT_DIR", "/kaggle/input")

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
    "C1_zero_shot": {"use_rag": False, "use_finetuned": False},
    "C2_rag": {"use_rag": True, "use_finetuned": False},
    "C3_finetune": {"use_rag": False, "use_finetuned": True},
    "C4_finetune_rag": {"use_rag": True, "use_finetuned": True},
}


def find_file(pattern):
    matches = glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {INPUT_DIR}")
    return matches[0]


# --- Corpus (pour afficher le texte des articles récupérés) ---
DATA_DIR = Path(f"{WORK_DIR}/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
HF_BASE = "https://huggingface.co/datasets/maastrichtlawtech/bsard/resolve/main"
articles_path = DATA_DIR / "articles.csv"
if not articles_path.exists():
    urllib.request.urlretrieve(f"{HF_BASE}/articles.csv", articles_path)
articles = pd.read_csv(articles_path)
article_lookup = articles.set_index("id").to_dict(orient="index")

# --- Index de retrieval (dense e5-large, précalculé par retrieval_job) ---
index_path = find_file("index_e5_large.npz")
idx = np.load(index_path)
doc_ids = idx["doc_ids"].tolist()
doc_embeddings = idx["embeddings"]
print(f"index de retrieval charge: {len(doc_ids)} articles")

# --- Adaptateur LoRA (produit par finetune_job) ---
adapter_dir = str(Path(find_file("adapter_config.json")).parent)
print(f"adaptateur LoRA: {adapter_dir}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

retriever = SentenceTransformer(EMBEDDING_MODEL, device=device)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()
print("modele + adaptateur charges")


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

    if cfg["use_finetuned"]:
        output_ids = model.generate(input_ids, **gen_kwargs)
    else:
        with model.disable_adapter():
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


with gr.Blocks(title="Chatbot juridique TER — C1-C4") as demo:
    gr.Markdown("## Chatbot juridique (droit belge francophone) — choisis la configuration")
    config_dropdown = gr.Dropdown(
        choices=list(CONFIGS.keys()), value="C2_rag", label="Configuration",
        info="C1 zero-shot | C2 RAG | C3 fine-tune | C4 fine-tune+RAG",
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
