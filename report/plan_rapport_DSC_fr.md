# Plan du rapport — M1 DSC (Bartosz Konior, encadrant F. Jacquenet)

Conventions : chaque section liste les figures/tableaux à insérer (`figures/`, `tables/`)
et les références `.bib` à citer (`report/references.bib`). Contenu rédigé en français, LaTeX/Overleaf.

## 1. Introduction
- Contexte : essor des LLM open-source, besoin d'adaptation au domaine juridique
- Questions de recherche (cf. proposition de sujet) :
  1. Un petit modèle spécialisé peut-il rivaliser avec un modèle généraliste sur une tâche précise ?
  2. Dans quels cas le fine-tuning apporte-t-il un gain réel par rapport au RAG seul ?
  3. Le modèle sait-il citer sa source et reconnaître ses limites (abstention) ?
- Annonce du plan

## 2. Pourquoi le domaine juridique ? (section imposée par F. Jacquenet)
- Caractéristiques qui en font un bon banc d'essai : besoin de précision factuelle, traçabilité
  de la source (citation d'article) obligatoire, coût élevé d'une hallucination, structure
  hiérarchique du texte de loi propice au chunking par article
- Comparaison avec d'autres domaines candidats (médical : accès aux données restreint ;
  code source : moins exigeant sur la citation de source exacte)
- Contrainte pratique : disponibilité de corpus libres (Légifrance/DILA vs BSARD) — lien avec
  §4 (bascule LLeQA → BSARD)
- Réfs : @guha2023legalbench, @hendrycks2021cuad, @chalkidis2022lexglue

## 3. État de l'art (section prioritaire selon F. Jacquenet — nombreuses réfs scientifiques)
### 3.1 Fine-tuning efficient de LLM
- Adapters, prefix-tuning, LoRA (@hu2021lora), QLoRA (@dettmers2023qlora)
- Compromis mémoire/performance de la quantification 4-bit NF4
### 3.2 Retrieval-Augmented Generation
- RAG original (@lewis2020rag), état de l'art récent (@gao2023ragsurvey)
- Retrieval dense vs BM25 vs hybride, reranking cross-encoder
### 3.3 Comparaisons RAG vs fine-tuning existantes
- **Positionnement direct par rapport à Derby LLM (@bouvard2024derbyllm)** : mêmes métriques
  d'arène et de fidélité, mais domaine juridique (vs. domaine étudié dans Derby LLM) et
  protocole étendu (222 questions test complètes, IC bootstrap, sous-groupes de lois,
  ablations retrieval/LoRA — cf. §6)
- Autres comparaisons RAG/fine-tuning (Ovadia et al., Balaguer et al. — *à vérifier avant
  citation finale*, cf. TODO dans references.bib)
### 3.4 Évaluation des systèmes de QA / IA générative
- Métriques de surface : ROUGE, BERTScore (@zhang2020bertscore)
- LLM-as-judge : RAGAS (@es2023ragas), Chatbot Arena (@chiang2024chatbotarena)
- Hallucination : surveys (*à compléter, vérifiés avant citation*)
### 3.5 Datasets et benchmarks juridiques
- BSARD (@louis2022bsard), LegalBench (@guha2023legalbench), CUAD (@hendrycks2021cuad),
  LexGLUE (@chalkidis2022lexglue)
- LLeQA : dataset initialement visé, accès jamais accordé (cf. DIFFICULTES.md) — justifie
  la bascule vers BSARD
### 3.6 Modèles de base
- Mistral 7B (@jiang2023mistral7b), Llama 2 (@touvron2023llama2), quantification

## 4. Données et méthodologie
- BSARD : structure réelle (22 633 articles, splits 886/222/113 165), champ `category`/`subcategory`
  natif (7 catégories) — Fig. `figures/bsard_category_distribution.pdf`
- Limite méthodologique assumée : pas de réponse rédigée gold, référence construite par
  concaténation des articles cités (cf. DIFFICULTES.md §5) — à discuter avant les résultats
- Justification du droit belge francophone comme substitut acceptable au droit français :
  objet d'étude = comparaison de méthodes d'adaptation, pas le contenu juridique lui-même ;
  le français est conservé ; blocage LLeQA documenté

## 5. Techniques classiques d'évaluation des outils d'IA générative (imposé par F. Jacquenet — AVANT les expérimentations propres)
- Panorama : métriques de similarité lexicale (BLEU, ROUGE), métriques d'embedding
  (BERTScore), évaluation humaine (arène par paires, Elo), LLM-as-judge et ses biais connus
  (auto-préférence, biais de position) — réfs @es2023ragas, @chiang2024chatbotarena,
  @zhang2020bertscore
- Limites connues de chaque famille de métriques, avant présentation des choix retenus (§6)

## 6. Expérimentations
### 6.1 Protocole (4 configurations C1-C4)
- Tableau `tables/config_description.tex`
### 6.2 Retrieval
- Recall@k / MRR / nDCG, mpnet vs e5-large vs BM25 vs hybride — Fig. `figures/recall_at_k_comparison.pdf`
### 6.3 Génération — résultats principaux
- ROUGE-L, BERTScore, IC bootstrap 95% — Fig. `figures/config_comparison_ci.pdf`
- Tests de significativité appariés, correction Holm-Bonferroni — `tables/pairwise_tests.tex`
### 6.4 Fiabilité : citation et hallucination (cœur du sujet)
- Exactitude de citation (precision/recall/exact match), taux d'hallucination d'article
- Tableau comparatif 4 configs — `tables/citation_hallucination.tex`
### 6.5 Analyse par sous-groupe de lois (réponse directe à la remarque d'A. Habrard)
- Heatmap configuration × catégorie juridique — Fig. `figures/heatmap_category.pdf`
### 6.6 Capacité d'abstention
- 50 questions hors-domaine, taux d'abstention correcte/fausse — `tables/abstention.tex`
### 6.7 Coût pratique
- Temps entraînement/inférence, VRAM, taille adaptateurs — `tables/cost_comparison.tex`
### 6.8 Ablations (P1, si temps disponible)
- Retrieval (BM25/dense/hybride/reranker), k, chunking, rang LoRA, courbe d'apprentissage
### 6.9 Évaluation humaine (arène, P1)
- Accord inter-annotateurs (Kappa de Cohen), comparaison C2 vs C3

## 7. Analyse d'erreurs
- Taxonomie (9 catégories, cf. brief §4.7), matrice type d'erreur × configuration
- 5-8 exemples qualitatifs commentés

## 8. Discussion
- Réponses aux 3 questions de recherche à la lumière des résultats
- Comparaison explicite avec les conclusions de Derby LLM (@bouvard2024derbyllm)
- Limites (BSARD droit belge, pas de réponse gold rédigée, budget GPU réduit)

## 9. Conclusion et perspectives

## Annexe A — Difficultés techniques (DIFFICULTES.md, format recommandations pratiques)

## Annexe B — Détails d'implémentation (config QLoRA complète, prompts utilisés)
