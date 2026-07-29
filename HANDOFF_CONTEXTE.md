# Contexte TER — Assistant juridique RAG vs Fine-tuning (à partager avec Chaabane / à coller dans une nouvelle session Claude)

## Le projet

TER de Master 1, Université Jean Monnet Saint-Étienne, soutenance **fin août 2026** (nous sommes le 29 juillet).
Binôme : **Bartosz Konior** (M1 DSC, encadrant **François Jacquenet**) / **Chaabane Ouammou** (M1 MLDM, encadrant **Amaury Habrard**). Deux rapports séparés (FR pour DSC, EN pour MLDM), code et expériences communs.

Sujet : comparer **fine-tuning QLoRA** et **RAG** pour un assistant de question-réponse juridique en français, sur **BSARD** (corpus de droit belge francophone — substitution documentée et assumée au droit français, LLeQA ayant été bloqué par une demande d'accès jamais accordée). Modèle : **Mistral-7B-Instruct-v0.3**.

**Contrainte imposée** : ressources gratuites uniquement, sans demande d'accès (d'où BSARD + miroir non-gated `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`).

**Référence imposée par Jacquenet** : Bouvard, Ciancone, Gourru, Schaeffer (2024), *Derby LLM*, APIA@PFIA 2024 (hal-04638460) — comparaison RAG/fine-tuning sur données d'entreprise en français. Le rapport doit se positionner explicitement contre cet article (mêmes métriques d'arène et de fidélité).

**Retour explicite d'Habrard** (juin/juillet) : une seule expérience ne suffit pas pour un binôme, il faut étendre à plusieurs échantillons/seeds, étudier des sous-groupes de lois, documenter les difficultés techniques en annexe.

**Constat assumé par les deux étudiants** : avec ~580-886 exemples d'entraînement réels, le modèle sera nécessairement faible. La note ne viendra pas de la qualité brute du modèle mais de la **rigueur, profondeur et compréhension démontrées** dans les analyses.

## Dépôt

`/Users/kimwilde/Desktop/ter-legal-rag-vs-finetuning/` — **⚠️ zéro commit git à ce jour, tout est en `untracked`**. Premier commit recommandé avant toute manipulation risquée.

Structure : `src/` (modules Python réutilisables, CPU), `kaggle_kernels/<job>/` (scripts GPU autonomes, un dossier par job, poussés indépendamment), `data/` (BSARD téléchargé), `results/` (JSON horodatés, une exécution = un fichier), `RESULTS.md` (généré automatiquement, ne jamais éditer à la main), `DIFFICULTES.md` (journal des problèmes techniques, format demandé par Habrard pour l'annexe).

## Ce qui est réellement terminé (vérifié via l'API Kaggle, pas supposé)

- **Retrieval** (`ter-bsard-retrieval-eval`, COMPLETE) : bm25/mpnet/e5-large/hybride sur 222 questions test, corpus 22 633 articles.
- **Ablation retrieval** (`ter-retr-ablation`, COMPLETE) : enrichissement métadonnées (Recall@1 0.083→0.119) et reranker cross-encoder (0.083→**0.151**) confirment l'hypothèse expliquant l'écart avec les CR précédents.
- **Fine-tuning principal** (`ter-bsard-qlora-finetune`, COMPLETE, terminé aujourd'hui) : seed 42, 580 exemples, r=16, 3 epochs → **loss finale 0.841, durée réelle 9h32** (vs 5-6h estimé — à budgéter large pour RunPod). Ce run est antérieur au split de validation (voir plus bas), donc pas de loss eval pour lui spécifiquement.

## Bugs trouvés et corrigés aujourd'hui (documentés DIFFICULTES.md §13-15)

1. **k-ablation avait perdu des résultats** : OOM à k=10 (batch fixe) sans sauvegarde incrémentale → k=1/k=3 déjà réussis perdus. Corrigé (batch adaptatif par k + retry OOM + sauvegarde après chaque k).
2. **~2,5% des identifiants d'articles (555/22633) ont un suffixe parasite** collé par le scraping (`"1714bis_REGION_DE_BRUXELLES-CAPITALE"`) — contaminait les cibles d'entraînement du fine-tuning et pénalisait injustement les métriques de citation sur **16/222 questions test**. Corrigé partout (`config.py::clean_article_ref_id` + tous les jobs concernés). Vérifié : ce défaut n'est PAS corrélé à la juridiction fédéral/régional.
3. **Aucun split de validation n'existait** pour le fine-tuning — impossible de détecter un surapprentissage sur si peu d'exemples. Ajouté : 100 questions réservées (seed fixe 999, indépendant du seed d'entraînement, identiques sur toute la courbe d'apprentissage), éval à chaque epoch, courbe train/eval sauvegardée.
4. **Juge LLM en conflit** : Qwen2.5-7B-Instruct était à la fois juge ET second modèle de base testé (biais d'auto-préférence). Remplacé par Phi-3.5-mini-instruct (MIT, absent des systèmes comparés).
5. Les 4 fichiers dupliqués `finetune_job_n190/n380/r8/r32` ont été supprimés — tout passe maintenant par des variables d'environnement sur `finetune_job/finetune_job.py` (`FT_SEED`, `FT_TRAIN_SIZE`, `FT_LORA_R`, `FT_TARGET_MODULES`, `FT_TRAIN_SOURCE`, `FT_N_SYNTHETIC_EXTRA`) pour éviter la dérive entre copies.

## Nouvelles métriques ajoutées en lisant Derby LLM (réimplémentées, pas copiées)

- **`src/metrics.py::fidelity_score`** : fidélité déterministe par recouvrement de "passages d'intérêt" (entités nommées spaCy + nombres/emails/URLs) — métrique qui n'existait pas du tout avant, nécessaire pour la comparaison imposée par Jacquenet.
- **`llm_judge_job.py`** : ne juge plus QUE la pertinence par LLM (10 échantillons à température>0, moyennés — un jugement greedy unique est bruyant), fidélité confiée à la métrique déterministe ci-dessus. C'est le choix méthodologique exact de Derby LLM (LLM pour le subjectif, règle pour ce qui peut être déterministe).
- **`arena_app.py::compute_full_agreement_distribution`** : distribution d'accord à 3 annotateurs (unanime/majoritaire/aucun), en plus du Kappa par paire déjà existant.
- **Point de positionnement pour le rapport** : Derby LLM n'a ni IC bootstrap ni tests de significativité (juste des moyennes brutes) — notre protocole va déjà plus loin qu'eux statistiquement.

## Analyses déjà codées et testées sur données réelles (CPU, pas besoin de GPU)

- `src/difficulty_analysis.py` : longueur question / nb articles gold / fréquence catégorie train, corrélation Spearman avec la qualité.
- `src/citation_format_analysis.py` : régularité de format d'identifiant (numeric_simple vs structured) — **104/222 questions test structurées, 118 simples** (vérifié : indépendant de fédéral/régional). Hypothèse : citation moins fiable sur IDs irréguliers (Mann-Whitney par config, une fois la génération faite).
- `src/error_analysis.py` : taxonomie à 9 catégories, génère un CSV de 200 lignes (≥50 réponses × 4 configs) pour annotation manuelle double + Cohen's Kappa.
- `src/arena_app.py` : Gradio, duels C2vC3 puis C4vC2, 60 questions, 3 annotateurs (le 3ᵉ est confirmé).
- `src/stats.py` : bootstrap 1000 + tests appariés Holm-Bonferroni.
- `src/fidelity_analysis.py` : nouveau, voir ci-dessus.

## Plan complet pour demain (RunPod) — organisé en sous-problèmes

**0. Préparatifs** : commit git initial · petit ajout env-var à `generation_job.py` (éviter de regénérer C1/C2 3×) · `FT_CODE_FILTER` pour le spécialiste Code Civil · script `adversarial_hallucination_job.py` (nouveau) · export per-question du retrieval (actuellement seules les métriques agrégées sont sauvées, pas les IDs récupérés).

**1. Retrieval** : déjà fait, juste un rerun léger pour exporter les IDs par question.

**2. Fine-tuning (10 runs, tous via `finetune_job.py` + env vars)** : 3 seeds (42/123/2026) config principale **r=32** (⚠️ corrigé 29/07 : la référence historique 580 ex. utilisait r=32, pas r=16 — trouvé en relisant `contexte/chaab1.pdf`, le run seed42 déjà fait sur Kaggle avait tourné à r=16 et redevient un point d'ablation) · rang LoRA r=8 et r=16 · cibles attn+MLP (teste le risque de surapprentissage vs le run principal via `overfit_gap`) · courbe d'apprentissage n=190/380/786 · données synthétiques (quantité à trancher demain) · **spécialiste Code Civil** (382 questions train / 86 test touchent ce code — opérationnalise directement le conseil d'Habrard "focus on subgroups", teste généraliste vs spécialiste sur le même sous-ensemble + coût en généralisation ailleurs).

**3. Génération** : run complet (seed42) + 2 runs allégés (seed123/2026, C3/C4 seulement) + run spécialiste Code Civil sur les 222 questions.

**4. Jobs indépendants** : oracle ceiling (décompose erreur retrieval vs génération) · Qwen2.5-7B second modèle de base · abstention (50 questions) · **stress-test adversarial hallucination** (nouveau : ~30 références d'articles plausibles mais inexistantes, teste si le modèle invente un contenu).

**5. Dépendants de la génération** : LLM-judge (Phi-3.5, pertinence) · fidélité (spaCy) · hypothèse citation/format (Mann-Whitney) · corrélation difficulté/qualité · corrélation juge↔humain.

**6. Humain (toi + Chaabane + 3ᵉ annotateur, confirmé)** : arène (60 questions × 2 duels) · analyse d'erreurs (200 lignes, double annotation).

**7. Statistique et analyses approfondies** : bootstrap + tests appariés · variance inter-seeds · **extrapolation de la courbe d'apprentissage** (loi de puissance : combien d'exemples faudrait-il pour atteindre X ?) · **puissance statistique minimale détectable par sous-groupe** (flag explicite des sous-groupes sous-dimensionnés type Protection sociale n=4) · **quasi-échec vs confusion totale du retrieval** (une fois les IDs par question exportés) · spécialiste vs généraliste (in-domain/out-of-domain).

**8. Restitution** : compléter `make_figures.py`/`make_tables.py` (aujourd'hui ils ne couvrent QUE le retrieval — il manque barres 4 configs+IC, heatmap catégorie, courbe d'apprentissage, matrice d'erreurs, arène, **visualisation embedding t-SNE/UMAP par catégorie**, quasi-échec retrieval).

**9. Stretch (fin de journée)** : sensibilité au prompt système, uniquement sur la config gagnante une fois connue.

## Décisions encore ouvertes

- Nombre de GPU RunPod loués en parallèle (change l'ordre d'exécution).
- Quantité de données synthétiques à tester pour l'augmentation (BSARD a 113 165 paraphrases non annotées, jamais utilisées).
- Le git du projet n'a toujours aucun commit — à faire avant de lancer les runs RunPod.
