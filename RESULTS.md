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
| seed123_n580_r32_attn | 123 | official | 580 | 100 | 32 | attn | 0.8439 | 0.6894 | +0.1076 | 0.25 |
| seed2026_n580_r32_attn | 2026 | official | 580 | 100 | 32 | attn | 0.8496 | 0.6889 | +0.1409 | 0.25 |
| seed42_n190_r32_attn | 42 | official | 190 | 100 | 32 | attn | 1.1224 | 1.0981 | +0.1020 | 0.09 |
| seed42_n380_r32_attn | 42 | official | 380 | 100 | 32 | attn | 0.9656 | 0.8604 | +0.1371 | 0.17 |
| seed42_n580_r32_attn | 42 | official | 580 | 100 | 32 | attn | 0.8344 | 0.6793 | +0.1423 | 0.26 |
| seed42_n580_r32_attn_mlp | 42 | official | 580 | 100 | 32 | attn_mlp | 0.5269 | 0.4158 | +0.2136 | 0.33 |
| seed42_n580_r8_attn | 42 | official | 580 | 100 | 8 | attn | 0.8504 | 0.7101 | +0.1352 | 0.26 |

**Variance inter-seeds (config principale, n=3 seeds)** : loss finale = 0.8426 ± 0.0063

## Génération — 4 configurations (222 questions test)

- Exécuté le: 2026-07-30T00:07:29Z
- top_k RAG: 5

| Config | ROUGE-L | BERTScore F1 | Précision citation | Rappel citation | Exact match | Taux hallucination | Durée (s) |
|---|---|---|---|---|---|---|---|
| C1_zero_shot | 0.1064 | 0.7781 | 0.016 | 0.011 | 0.005 | 0.126 | 174.5 |
| C2_rag | 0.1317 | 0.7937 | 0.180 | 0.120 | 0.050 | 0.207 | 432.8 |
| C3_finetune | 0.1444 | 0.7973 | 0.007 | 0.004 | 0.000 | 0.027 | 235.9 |
| C4_finetune_rag | 0.1014 | 0.7643 | 0.030 | 0.041 | 0.005 | 0.090 | 472.5 |

### IC bootstrap 95% (ROUGE-L, 1000 rééchantillonnages)

| Config | Moyenne | IC 95% bas | IC 95% haut |
|---|---|---|---|
| C1_zero_shot | 0.1064 | 0.1005 | 0.1125 |
| C2_rag | 0.1317 | 0.1223 | 0.1427 |
| C3_finetune | 0.1444 | 0.1345 | 0.1567 |
| C4_finetune_rag | 0.1014 | 0.0857 | 0.1191 |

### Tests de significativité appariés (Holm-Bonferroni)

| Comparaison | Différence observée | p-value | Significatif (corrigé) |
|---|---|---|---|
| C1_zero_shot_vs_C2_rag | -0.0253 | 0.0000 | oui |
| C1_zero_shot_vs_C3_finetune | -0.0379 | 0.0000 | oui |
| C1_zero_shot_vs_C4_finetune_rag | +0.0050 | 0.5640 | non |
| C2_rag_vs_C3_finetune | -0.0127 | 0.0480 | non |
| C2_rag_vs_C4_finetune_rag | +0.0303 | 0.0020 | oui |
| C3_finetune_vs_C4_finetune_rag | +0.0430 | 0.0000 | oui |

### Analyse par catégorie juridique (demande A. Habrard)

| Catégorie | n | C1_zero_shot | C2_rag | C3_finetune | C4_finetune_rag |
|---|---|---|---|---|---|
| Argent | 36 | 0.112 | 0.156 | 0.143 | 0.079 |
| Etrangers | 13 | 0.087 | 0.106 | 0.097 | 0.075 |
| Famille | 67 | 0.114 | 0.144 | 0.181 | 0.112 |
| Justice | 30 | 0.078 | 0.092 | 0.100 | 0.084 |
| Logement | 66 | 0.112 | 0.129 | 0.138 | 0.117 |
| Protection sociale | 4 | 0.084 | 0.109 | 0.123 | 0.097 |
| Travail | 6 | 0.122 | 0.143 | 0.156 | 0.101 |

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
