# RESULTS.md — TER Assistant juridique : RAG vs Fine-tuning

Généré automatiquement depuis les JSON de `results/`. Chaque chiffre provient d'une exécution réelle horodatée (voir champ `timestamp` de chaque section). Aucune valeur n'est estimée ou reconstituée à la main.

## Retrieval

- Corpus: 22633 articles
- Questions test: 222
- Exécuté le: 2026-07-28T18:40:24Z

| Méthode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Durée (s) |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.063 | 0.140 | 0.184 | 0.224 | 0.289 | 0.215 | 0.176 | 20.4 |
| mpnet | 0.021 | 0.045 | 0.074 | 0.097 | 0.124 | 0.105 | 0.076 | 311.9 |
| e5_large | 0.083 | 0.169 | 0.211 | 0.296 | 0.377 | 0.288 | 0.230 | 900.9 |
| hybrid_bm25_e5 | 0.106 | 0.196 | 0.250 | 0.322 | 0.390 | 0.312 | 0.258 | 0.0 |

### Écarts constatés vs. comptes-rendus précédents

| | mpnet (CR) | mpnet (reproduit) | e5-large (CR) | e5-large (reproduit) |
|---|---|---|---|---|
| recall@1 | 0.117 | 0.021 | 0.198 | 0.083 |
| recall@3 | 0.203 | 0.045 | 0.347 | 0.169 |
| recall@5 | 0.261 | 0.074 | 0.423 | 0.211 |
| recall@10 | 0.347 | 0.097 | 0.523 | 0.296 |

**Écart non résolu** : les Recall@k reproduits sont systématiquement 2 à 5x plus bas que ceux annoncés dans les CR précédents, alors que le protocole (corpus 22 633 articles, 222 questions test, mêmes modèles, préfixes `query:`/`passage:` corrects pour e5-large) est identique. Distribution de longueur des articles vérifiée : médiane 77 mots, seulement 6.8% > 384 mots (limite de troncature mpnet) — la troncature seule n'explique probablement pas un écart de cette ampleur. Hypothèses à tester (non validées) : (1) le texte indexé dans les CR précédents incluait peut-être les métadonnées hiérarchiques (code/chapitre/section) en plus du corps de l'article, apportant un signal lexical supplémentaire ; (2) les CR précédents ont pu utiliser un fine-tuning contrastif léger du retriever (BSARD fournit des négatifs BM25 dans `negatives/`) plutôt qu'un modèle off-the-shelf. À investiguer dans l'ablation chunking/enrichissement de document (notebook 07, P1) avant d'écrire la section résultats du rapport — ne pas présenter les chiffres du CR comme acquis.

## Fine-tuning QLoRA

| Run | Seed | Train src | n_train | n_val | r | Cibles | Loss train | Loss eval | Écart (surapprentissage) | Durée (h) |
|---|---|---|---|---|---|---|---|---|---|---|
| ? | 42 | official | 580 | ? | 16 | attn | 0.8411 | n/a | n/a | 9.52 |

## Génération — 4 configurations

*(pas encore exécuté / résultats non disponibles)*

## Fidélité (méthode Derby LLM)

*(pas encore exécuté / résultats non disponibles)*

## LLM-as-judge (pertinence)

*(pas encore exécuté / résultats non disponibles)*

## Régularité de format des identifiants d'article

Découvert en creusant le bug d'extraction regex (cf. DIFFICULTES.md §9). Vérifié : ce n'est PAS un proxy de la juridiction fédéral/régional (les deux ont des proportions comparables de formats irréguliers) -- axe indépendant.

| Format | n articles corpus |
|---|---|
| structured | 13199 |
| numeric_simple | 9434 |

| Format | n questions test |
|---|---|
| numeric_simple | 118 |
| structured | 104 |

**Hypothèse à tester une fois `generation_results.json` disponible** : citation exact match plus bas / hallucination plus haute sur les questions dont au moins un article gold a un identifiant structuré, par rapport aux identifiants numériques simples (test de Mann-Whitney par configuration, cf. `src/citation_format_analysis.py::format_hypothesis_test`).
