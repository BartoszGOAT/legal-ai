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
- **Ovadia et al. (@ovadia2023finetuneorretrieve, EMNLP 2024)** : le RAG surpasse
  systématiquement le fine-tuning **non-supervisé** (poursuite du pré-entraînement sur du
  texte brut) pour l'injection de connaissances -- **exactement la même limite que celle
  identifiée par Derby LLM** sur leur propre fine-tuning. Nous utilisons du fine-tuning
  **supervisé** (paires question→réponse formatées avec citation), une tâche bien plus
  proche de ce pour quoi QLoRA est conçu -- si nos résultats divergent de ces deux
  références, cette distinction méthodologique (supervisé vs non-supervisé) en est
  l'explication la plus probable, pas une contradiction.
- **Balaguer et al. (@balaguer2024ragvsft)** : sur un cas d'usage agricole, le fine-tuning
  seul gagne +6 points de précision, et **combiner fine-tuning et RAG en ajoute encore +5**
  -- cohérent avec notre propre résultat où C4 (fine-tuning+RAG) dépasse les deux approches
  prises séparément, un point de convergence à souligner explicitement.
### 3.4 Évaluation des systèmes de QA / IA générative
- Métriques de surface : ROUGE (@lin2004rouge), BERTScore (@zhang2020bertscore)
- LLM-as-judge : RAGAS (@es2023ragas), Chatbot Arena (@chiang2024chatbotarena, méthode
  Bradley-Terry que nous réutilisons pour classer les 4 configurations à partir des votes
  d'arène)
- **Nuance méthodologique sur la fidélité** : RAGAS calcule la fidélité par extraction de
  claims via LLM puis vérification NLI claim-par-claim (coûteux, potentiellement fragile
  si l'extraction échoue) ; ARES utilise des juges spécialisés fine-tunés (DeBERTa). Notre
  métrique de fidélité (façon Derby LLM, recouvrement d'entités spaCy) est délibérément
  plus simple et déterministe -- à assumer explicitement comme une limite/simplification,
  pas présenter comme équivalent à RAGAS.
- Efficacité des LLM et PEFT (@wan2023efficientllmsurvey)
- Hallucination : survey général (@rawte2023hallucinationsurvey). **Point de comparaison
  concret trouvé** : une étude Stanford (citée dans la littérature sur l'hallucination
  juridique) mesure des taux d'hallucination de 43% (GPT-4), 33% (Westlaw AI) et 17%
  (Lexis+) sur des outils juridiques commerciaux -- **notre C3 (fine-tuning seul, 5,0%)
  est nettement en dessous de ces trois outils commerciaux**, un résultat frappant et
  citable pour la discussion, à condition de trouver et vérifier la référence exacte de
  cette étude avant de l'inclure dans le rapport final (pas encore fait).
### 3.5 Datasets et benchmarks juridiques
- BSARD (@louis2022bsard), LegalBench (@guha2023legalbench), CUAD (@hendrycks2021cuad),
  LexGLUE (@chalkidis2022lexglue)
- LLeQA (@louis2023lleqa) : dataset initialement visé (construit sur BSARD, +69% de
  questions, annotations enrichies), accès jamais accordé (cf. DIFFICULTES.md) — justifie
  la bascule vers BSARD, dont il prolonge directement le travail
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

**Démarche** : chaque sous-section ci-dessous est formulée explicitement comme
Question de recherche → Hypothèse (H0/H1) → Test statistique → Résultat →
Conclusion, plutôt que comme une simple analyse exploratoire. C'est la
démarche scientifique classique (voir aussi §5) appliquée systématiquement,
pas seulement pour la comparaison principale des 4 configurations.

### 6.1 Protocole (4 configurations C1-C4)
- Tableau `tables/config_description.tex`
### 6.2 Retrieval
- Recall@k / MRR / nDCG, mpnet vs e5-large vs BM25 vs hybride — Fig. `figures/recall_at_k_comparison.pdf`
### 6.3 Génération — résultats principaux
- **Q** : le fine-tuning apporte-t-il un gain réel par rapport au RAG seul (question de recherche n°2) ?
- **H0** : pas de différence significative entre C2 et C3 sur ROUGE-L/BERTScore/citation exacte.
- ROUGE-L, BERTScore, IC bootstrap 95% — Fig. `figures/config_comparison_ci.pdf`
- **Test** : bootstrap apparié + Wilcoxon signé, correction Holm-Bonferroni (comparaisons multiples) — `tables/pairwise_tests.tex`
### 6.4 Fiabilité : citation et hallucination (cœur du sujet)
- Exactitude de citation (precision/recall/exact match), taux d'hallucination d'article
- Tableau comparatif 4 configs — `tables/citation_hallucination.tex`
### 6.5 Analyse par sous-groupe de lois (réponse directe à la remarque d'A. Habrard)
- **Q** : le RAG et le fine-tuning se comportent-ils différemment selon le sous-domaine juridique ?
- **H0** : l'écart de performance C2 vs C3 est constant à travers les catégories (pas d'interaction config×catégorie).
- Heatmap configuration × catégorie juridique — Fig. `figures/heatmap_category.pdf`
- **Test** : modèle de régression avec terme d'interaction config×catégorie (cf. §6.10), significativité du terme d'interaction
### 6.6 Capacité d'abstention
- 50 questions hors-domaine, taux d'abstention correcte/fausse — `tables/abstention.tex`
### 6.7 Coût pratique
- Temps entraînement/inférence, VRAM, taille adaptateurs — `tables/cost_comparison.tex`
### 6.8 Ablations
- Retrieval (BM25/dense/hybride/reranker), k, chunking, rang/cibles LoRA, courbe d'apprentissage (+ extension données synthétiques), spécialiste Code Civil vs généraliste
- Chaque ablation formulée en H0 (ex. **k** : "H0 : au-delà de k=5, ajouter des fragments n'améliore pas la fiabilité — Derby LLM signale que 10 fragments saturent le prompt, à vérifier empiriquement")
### 6.9 Évaluation humaine (arène)
- Accord inter-annotateurs (Kappa de Cohen par paire + distribution d'accord unanime/majoritaire/aucun à 3 annotateurs, façon Derby LLM Fig. 7)
- **Classement par modèle Bradley-Terry** (méthode utilisée par Chatbot Arena/LMSYS, dont Derby LLM et nous nous inspirons tous deux) à partir des votes par paires — donne un score de force relative sur une échelle unique plutôt que des taux de victoire par paire isolés — Fig. `figures/bradley_terry_ranking.pdf`
- **Q** : le jugement automatique (LLM-judge, métriques de surface, fidélité) est-il un proxy valide du jugement humain ?
- **Test** : corrélation de Spearman entre score du juge LLM et issue du vote arène sur les questions communes

### 6.10 Prédiction supervisée de la fiabilité (approche apprentissage automatique)
- **Q** : peut-on prédire, à partir de caractéristiques observables avant génération (longueur de
  la question, nombre d'articles gold, catégorie, régularité de format d'identifiant, configuration,
  score de similarité du retrieval), le risque qu'une réponse soit peu fiable (citation incorrecte
  ou hallucination) ? Inspiré de la littérature sur la *quality estimation* sans référence en TAL
  (à la COMET-QE) et de la *sélection avec option de rejet* (*selective prediction*), réimplémenté
  pour notre tâche — pas copié.
- **H0** : un classifieur entraîné sur ces caractéristiques ne fait pas mieux qu'un modèle qui
  prédit toujours la classe majoritaire (AUC = 0.5).
- **Méthode** : régression logistique (interprétable, coefficients directement lisibles) et
  Random Forest (capture les interactions non-linéaires), validation croisée à 5 plis stratifiée
  vu le nombre limité d'observations (~888 lignes = 222 questions × 4 configs)
- **Sorties** : courbe ROC + AUC, matrice de confusion, importance des variables (quel facteur
  domine : la configuration ? la difficulté de la question ? le retrieval ?), courbe de calibration
  (les probabilités prédites sont-elles fiables ?) — Fig. `figures/quality_prediction_*.pdf`
- **Discussion** : si la configuration domine largement les autres variables, cela renforce la
  conclusion principale (le choix RAG/fine-tuning importe plus que les caractéristiques de la
  question) ; si une interaction config×difficulté ressort, cela nuance §6.5

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
