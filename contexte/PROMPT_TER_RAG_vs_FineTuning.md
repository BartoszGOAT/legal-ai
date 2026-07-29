# PROMPT MAÎTRE — Réalisation complète du projet TER « Assistant juridique : RAG vs Fine-tuning »

> **Mode d'emploi** : copier-coller l'intégralité de ce document comme premier message dans une session Claude (idéalement Claude Code, ou Claude Cowork). Répondre ensuite « go » pour lancer la Phase 0.

---

## 1. QUI TU ES ET CE QU'ON ATTEND DE TOI

Tu es ingénieur ML senior, spécialiste des LLM, du RAG et du fine-tuning efficient (LoRA/QLoRA), et tu as l'habitude de produire du code de recherche reproductible pour des publications académiques.

Tu vas **construire intégralement, de zéro**, le code, les expériences, les résultats et les artefacts d'un TER de Master 1 rendu fin août 2026. Le projet a été décrit à l'encadrant dans des comptes-rendus hebdomadaires, mais le code n'existe pas dans un état exploitable : **tout est à (re)faire proprement**.

Deux règles absolues :

- **Tu ne fabriques jamais de résultat.** Chaque chiffre produit doit provenir d'une exécution réelle et être écrit dans un fichier JSON horodaté. Si un résultat obtenu diffère de celui annoncé dans les comptes-rendus, tu le signales explicitement dans une section « Écarts constatés » plutôt que de l'ajuster.
- **Tu ne cites jamais une référence bibliographique que tu n'as pas vérifiée.** Chaque entrée `.bib` doit correspondre à un article réellement existant (arXiv ID, DOI ou lien HAL vérifiable).

---

## 2. CONTEXTE DU PROJET (à lire intégralement avant de coder)

### 2.1 Cadre académique

- **Établissement** : Université Jean Monnet, Saint-Étienne.
- **Étudiants** : Bartosz Konior (M1 Data Science & Cybersécurité, encadrant **François Jacquenet**) et Chaabane Ouammou (M1 MLDM, encadrant **Amaury Habrard**).
- **Deux rapports et deux soutenances distincts** : rapport DSC en français (Jacquenet), rapport MLDM en anglais (Habrard). Le code et les expériences sont communs, les rédactions sont séparées.
- **Soutenance : fin août 2026.** Nous sommes le **27 juillet 2026**. Il reste environ **4 à 5 semaines**.
- Volume de travail attendu : équivalent **deux personnes à temps plein sur trois mois**.
- Rapport rédigé en **LaTeX** (Overleaf).

### 2.2 Sujet

Conception et évaluation d'un assistant de question-answering juridique en français à partir d'un LLM open-source, comparant deux stratégies d'adaptation au domaine :

- **Fine-tuning QLoRA** de Mistral-7B sur des paires question/article ;
- **RAG** (recherche dense dans un corpus d'articles de loi + génération conditionnée).

Questions de recherche formulées dans la proposition de sujet :

1. Un petit modèle open-source spécialisé peut-il rivaliser avec un modèle généraliste de grande taille sur une tâche précise ?
2. Dans quels cas le fine-tuning apporte-t-il un gain réel par rapport au RAG seul ?
3. Le modèle sait-il **citer correctement sa source** et **reconnaître ses limites** (abstention) ?

### 2.3 Consignes explicites des encadrants (à respecter à la lettre)

**François Jacquenet (DSC)** :
- Accent important sur l'**état de l'art**, avec de **nombreuses références bibliographiques** à des **articles scientifiques** (pas de vulgarisation) : fine-tuning, RAG, évaluation des systèmes de QA, datasets existants, comparaisons RAG/fine-tuning existantes.
- **Justifier le choix du domaine juridique** plutôt qu'un autre.
- Section expérimentations importante, qui doit **présenter d'abord les techniques classiques d'évaluation des outils d'IA générative** avant de montrer les expériences propres.
- Référence imposée : Bouvard, Ciancone, Gourru, Schaeffer (2024), *Derby LLM : Évaluation comparative des approches RAG et fine-tuning*, APIA@PFIA 2024, HAL-04638460. **Notre travail doit se positionner explicitement par rapport à cet article** (mêmes métriques d'arène et de fidélité, mais domaine juridique et protocole étendu).
- Rapport en LaTeX.

**Amaury Habrard (MLDM), retours sur le rapport mensuel de juin/juillet** :
- « The comparisons may appear limited since you make only one experiment, it could be interesting to **extend the experimental setup to make tests on different test data/sampling to evaluate the average behavior** of your model on more diverse tasks. »
- « Considering that you do not work alone, the global impression is that **the work appears a bit limited**, so be careful to expand a bit your work and follow the perspectives you mention. »
- « Regarding the technical difficulties, you may imagine in the final report to **document these difficulties in an appendix**, like by making recommendations on how to use efficiently the different libraries. »
- « It is a good practice to reduce the difficulty by changing the dataset, the task or the problem. **You can focus on particular subgroups of laws.** »

**Contrainte matérielle imposée** (en majuscules dans les consignes) :
> **UTILISER UNIQUEMENT DES RESSOURCES GRATUITES ET DISPONIBLES SANS DEMANDE D'ACCÈS.**

Cela signifie : pas d'API payante, pas de dataset sous demande d'accès (c'est ce qui a fait perdre six semaines avec LLeQA), pas de modèle « gated » nécessitant une validation manuelle. Google Colab niveau gratuit uniquement.

### 2.4 Historique du projet (ce qui a été annoncé à l'encadrant)

| Période | Contenu annoncé |
|---|---|
| S1 (27–30 avr.) | Lectures : LoRA (Hu et al. 2022), QLoRA (Dettmers et al. 2023), RAG (Lewis et al. 2020), Derby LLM (2024) |
| S2 (1–7 mai) | Exploration Légifrance / API DILA ; Mistral 7B retenu |
| S3 (8–15 mai) | Inscription portail PISTE, premier script d'appel API Légifrance, dépôt Git initialisé |
| S4 (21 mai) | Élargissement corpus, intégration génération |
| S5–S6 (22 mai–5 juin) | **Bascule LLeQA → BSARD** (demande d'accès HuggingFace jamais accordée depuis le 21 avril). Env. Colab configuré (conflits numpy / bitsandbytes / CUDA). Notebooks 01 (exploration), 02 (éval retrieval), 03 (QLoRA). |
| 6 juin–3 juil. | Amélioration du retrieval (mpnet → multilingual-e5-large), 2ᵉ itération de fine-tuning (380 → 580 exemples), comparaison de 3 configurations |
| 10 juil. | 4ᵉ configuration (fine-tuning + RAG), début d'analyse d'erreurs |
| 24 juil. | Reformulation du prompt de génération (réponse rédigée + citation d'article), métrique d'exactitude de citation, catégorisation manuelle des erreurs sur 40 questions/config |

**Difficultés déjà documentées** (à reprendre dans l'annexe demandée par Habrard) : incompatibilité CUDA 12.8 de Colab avec les versions figées de `bitsandbytes` ; `article_ids` de BSARD stockés en chaînes de caractères et non en listes, provoquant des bugs silencieux ; quota GPU gratuit épuisé en cours d'évaluation ; indexation du corpus avec E5-large > 45 min sur T4 (jusqu'à 2 h selon la configuration).

### 2.5 Résultats déjà communiqués aux encadrants — À REPRODUIRE

Ces chiffres figurent dans des rapports déjà envoyés. Le pipeline doit être configuré **à l'identique** pour que les valeurs reproduites soient cohérentes. Tout écart doit être documenté honnêtement, jamais masqué.

**Retrieval, 222 questions du split test BSARD :**

| Niveau | `all-mpnet-base-v2` (baseline) | `intfloat/multilingual-e5-large` |
|---|---|---|
| Recall@1 | 11,7 % | 19,8 % |
| Recall@3 | 20,3 % | 34,7 % |
| Recall@5 | 26,1 % | 42,3 % |
| Recall@10 | 34,7 % | 52,3 % |

**Génération, 50 questions test :**

| Configuration | ROUGE-L | BERTScore F1 |
|---|---|---|
| Mistral base, zero-shot | 0.0821 | 0.6112 |
| RAG (retrieval amélioré) | 0.1535 | 0.6589 |
| Fine-tuning seul, 2ᵉ itération | 0.1612 | 0.7043 |
| Fine-tuning + RAG | *(mesuré, non chiffré dans les CR)* | *(idem)* |

**Fine-tuning :** Mistral-7B-Instruct, QLoRA r=16, α=32, quantification 4-bit NF4, 0,575 % des paramètres entraînés, 3 epochs, ~5–6 h sur T4, adaptateurs ~100 Mo. Itération 1 : 380 exemples → ROUGE-L 0.1375 / BERTScore 0.6814. Itération 2 : 580 exemples → 0.1612 / 0.7043.

---

## 3. CE QUE TU DOIS PRODUIRE

### 3.1 Arborescence exigée

```
ter-legal-rag-vs-finetuning/
├── README.md                     # vue d'ensemble + badges "Open in Colab"
├── INSTALL.md                    # instructions pas-à-pas de A à Z (cf. §6)
├── RESULTS.md                    # tous les résultats, tenu à jour automatiquement
├── DIFFICULTES.md                # journal des problèmes techniques → annexe du rapport
├── requirements-colab.txt        # versions ÉPINGLÉES et testées
├── src/
│   ├── config.py                 # tous les chemins, seeds, hyperparamètres — SOURCE UNIQUE
│   ├── setup_colab.py            # bootstrap : install + montage Drive + vérif GPU
│   ├── data.py                   # chargement/nettoyage BSARD, construction des splits
│   ├── retrieval.py              # index dense, BM25, hybride, reranking
│   ├── generation.py             # prompts, inférence, parsing des citations
│   ├── finetune.py               # QLoRA
│   ├── metrics.py                # ROUGE-L, BERTScore, Recall@k, MRR, citation, abstention
│   ├── judge.py                  # LLM-as-judge (pertinence, fidélité)
│   └── stats.py                  # bootstrap, IC 95 %, tests appariés
├── notebooks/
│   ├── 00_setup_and_smoke_test.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_build_indexes.ipynb
│   ├── 03_retrieval_eval.ipynb
│   ├── 04_finetune_qlora.ipynb
│   ├── 05_generate_all_configs.ipynb
│   ├── 06_evaluate_and_compare.ipynb
│   ├── 07_ablations.ipynb
│   ├── 08_error_analysis.ipynb
│   └── 09_human_arena.ipynb
├── results/                      # JSON horodatés, une exécution = un fichier
├── figures/                      # PDF vectoriels pour LaTeX
├── tables/                       # .tex générés automatiquement
└── report/
    ├── references.bib            # ≥ 45 références VÉRIFIÉES
    ├── plan_rapport_DSC_fr.md
    └── plan_rapport_MLDM_en.md
```

### 3.2 Contraintes techniques non négociables

- **Modèle de base** : Mistral-7B-Instruct-v0.3. ⚠️ Le dépôt officiel `mistralai/...` sur HuggingFace demande une acceptation de licence, ce qui contrevient à la consigne « sans demande d'accès ». **Vérifie l'accessibilité réelle et prévois un miroir non restreint** (par exemple les versions pré-quantifiées `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`). Documente ce choix et la procédure exacte dans `INSTALL.md`.
- **Dataset** : BSARD (`maastrichtlawtech/bsard` sur HuggingFace) — corpus d'articles statutaires belges francophones + questions de citoyens annotées avec les articles pertinents. Libre d'accès. **Justifie explicitement dans le rapport** pourquoi le droit belge francophone est une substitution acceptable au droit français : l'objet d'étude est la comparaison de *méthodes d'adaptation*, pas le contenu juridique ; le français est conservé ; le blocage LLeQA est documenté.
- **Plateforme** : Google Colab gratuit, GPU T4 16 Go. Chaque notebook doit :
  - s'exécuter de bout en bout en **moins de 3 h 30** (limite de session), sinon être découpé ;
  - **sauvegarder ses sorties sur Google Drive** et **reprendre où il s'est arrêté** en cas de déconnexion (checkpointing systématique) ;
  - annoncer en première cellule sa durée estimée et sa consommation VRAM.
- **Reproductibilité** : `seed` fixée dans `config.py`, propagée à `random`, `numpy`, `torch`, `transformers`. Toute exécution écrit un JSON contenant : seed, versions des bibliothèques, hyperparamètres, horodatage, résultats.
- **Aucune dépendance à un service payant.** Pas d'API OpenAI, Anthropic, Cohere. Le LLM-as-judge tourne en local sur le GPU Colab.

---

## 4. PROTOCOLE EXPÉRIMENTAL À IMPLÉMENTER

### 4.1 Les quatre configurations (axe principal)

| Config | Description |
|---|---|
| **C1** | Mistral-7B-Instruct zero-shot, sans adaptation |
| **C2** | RAG : retrieval dense + Mistral de base |
| **C3** | Fine-tuning QLoRA seul, sans contexte récupéré |
| **C4** | Fine-tuning QLoRA + RAG |

Le **même prompt de génération** est appliqué aux quatre configurations pour que la comparaison reste équitable. Format imposé : réponse rédigée en langage naturel **suivie d'une citation précise de l'article** (ex. « Article 1382 du Code civil »), et possibilité explicite de répondre « Je ne sais pas ».

### 4.2 Métriques

**Retrieval** : Recall@{1,3,5,10,20}, MRR@10, nDCG@10.

**Génération, similarité de surface** : ROUGE-L (F1), BERTScore F1 (modèle multilingue — préciser lequel, les scores ne sont pas comparables entre modèles).

**Génération, fiabilité — c'est le cœur du sujet** :
- **Exactitude de citation** : extraction par expression régulière de l'identifiant d'article cité, comparaison à l'article de référence. Reporter *precision*, *recall* et *exact match*, pas seulement un pourcentage global.
- **Taux d'hallucination d'article** : proportion de réponses citant un numéro d'article **inexistant dans le corpus**. Métrique directe, vérifiable, et c'est un des arguments forts du rapport.
- **Fidélité** (inspirée de Derby LLM, éq. 1) : proportion des passages d'intérêt de la réponse présents dans le fragment pertinent. Utiliser spaCy (`fr_core_news_sm`, gratuit) pour l'extraction des passages.
- **Pertinence** (*answer relevance*) via LLM-as-judge.

**Capacité d'abstention** : construire un jeu de **50 questions hors-domaine ou sans réponse dans le corpus** (questions de droit d'un autre pays, questions non juridiques, questions dont l'article de référence a été retiré de l'index). Mesurer le taux d'abstention correcte et le taux de fausse abstention sur les questions valides.

**Coût pratique** : temps d'entraînement, temps d'inférence par question, VRAM pic, taille des artefacts. Tableau comparatif — c'est une dimension que Derby LLM valorise et qui différencie RAG et fine-tuning au-delà de la qualité.

### 4.3 Robustesse — réponse directe au retour d'Habrard

C'est la partie qui transforme « une seule expérience » en évaluation sérieuse. Elle est **prioritaire**.

1. **Évaluer sur les 222 questions test complètes**, pas sur 50. Le sous-échantillon de 50 n'est conservé que pour les analyses coûteuses (jugement LLM, annotation manuelle).
2. **Trois seeds** pour le fine-tuning et pour l'échantillonnage, résultats reportés en **moyenne ± écart-type**.
3. **Bootstrap sur 1 000 rééchantillonnages** du jeu de test → **intervalles de confiance à 95 %** sur chaque métrique. Sans IC, on ne peut pas dire si 0.1535 et 0.1612 diffèrent réellement.
4. **Tests de significativité appariés** (bootstrap apparié ou Wilcoxon signé) pour chaque paire de configurations, avec correction de Holm-Bonferroni pour comparaisons multiples.
5. **Analyse par sous-groupes de lois** (suggestion explicite d'Habrard) : BSARD annote les questions par catégorie/thématique juridique (famille, logement, travail, protection sociale, etc.). Produire un tableau **configuration × catégorie** et identifier les domaines où le fine-tuning rattrape ou dépasse le RAG. C'est très probablement le résultat le plus intéressant du rapport.
6. **Analyse par difficulté** : segmenter selon la longueur de la question, le nombre d'articles pertinents, la fréquence de la thématique dans le train.

### 4.4 Ablations (notebook 07)

- **Retrieval** : BM25 seul / dense `mpnet` / dense `e5-large` / hybride BM25+dense (RRF) / dense + reranker cross-encoder (`BAAI/bge-reranker-v2-m3`, gratuit).
- **Nombre de fragments injectés** : k ∈ {1, 3, 5, 10}. Derby LLM signale que 10 fragments saturent inutilement le prompt — le vérifier empiriquement et en tirer une recommandation.
- **Chunking** : article entier vs découpage en passages de 256/512 tokens.
- **LoRA** : rang r ∈ {8, 16, 32}, et cibles d'adaptation (attention seule vs attention + MLP).
- **Taille du jeu d'entraînement** : 190 / 380 / 580 exemples → **courbe d'apprentissage**. Elle répond directement à la perspective ouverte par Derby LLM (« étudier l'impact de la taille des jeux de données »).
- **Température de génération** : 0 (référence) vs 0.3.

Si le temps GPU manque, prioriser : reranker > k > taille du train > rang LoRA.

### 4.5 Évaluation humaine — arène (notebook 09)

Réplique du protocole Derby LLM, adapté :
- Interface **Gradio** (gratuit, tourne dans Colab) affichant question, contexte, et deux réponses **anonymisées et ordonnées aléatoirement**.
- Boutons : « A est meilleure », « B est meilleure », « Match nul », « Aucune ».
- **60 questions minimum × 3 annotateurs** (les deux étudiants + un tiers). Export CSV.
- Calcul de l'**accord inter-annotateurs** (Kappa de Cohen par paire, accord total/partiel), avec la mise en garde de Derby LLM sur la fiabilité du Kappa en classes déséquilibrées.
- Comparaisons prioritaires : C2 vs C3 (le duel central), puis C4 vs C2.

### 4.6 LLM-as-judge (notebook 06)

Juge local et gratuit (`Qwen2.5-7B-Instruct` ou Mistral lui-même en 4-bit), prompt inspiré d'Athina Evals / RAGAS, notant la **pertinence** et la **fidélité** sur une échelle continue avec justification.
**Obligation méthodologique** : mesurer la **corrélation entre le jugement LLM et les votes humains** sur les questions communes, et discuter le biais d'auto-préférence si le juge partage l'architecture du modèle évalué. Cette validation du juge est un point qui distingue un travail sérieux d'un travail superficiel.

### 4.7 Analyse d'erreurs (notebook 08)

Taxonomie à appliquer sur **≥ 50 réponses par configuration**, annotées par les deux étudiants avec mesure d'accord :

1. Hallucination d'article (numéro inexistant)
2. Article existant mais non pertinent
3. Confusion entre systèmes juridiques (droit français / belge / autre)
4. Réponse correcte mais sans citation
5. Absence d'abstention alors qu'elle était requise
6. Abstention excessive
7. Erreur de langue (réponse en anglais — problème observé dans Derby LLM)
8. Réponse incomplète / recopie brute de l'article sans reformulation
9. Dégénérescence / répétition (sur-apprentissage, cf. exemples Derby LLM)

Sortie : matrice type d'erreur × configuration, + **5 à 8 exemples qualitatifs commentés** (format des Figures 4-5-6 de Derby LLM) directement réutilisables dans le rapport.

---

## 5. MÉTHODE DE TRAVAIL IMPOSÉE

### Phase 0 — Audit et plan (avant tout code)

1. Vérifier l'accessibilité réelle, **aujourd'hui**, de : BSARD sur HuggingFace, Mistral-7B-Instruct-v0.3 (et miroir non restreint), `intfloat/multilingual-e5-large`, `sentence-transformers/all-mpnet-base-v2`, le reranker, le modèle juge.
2. Inspecter la structure exacte de BSARD : colonnes, splits, format des `article_ids` (⚠️ chaînes et non listes — bug déjà rencontré), présence et nom du champ de catégorie juridique, taille du corpus, taille des splits. **Confirmer que le split test contient bien 222 questions.**
3. Identifier précisément **comment est construite la « réponse de référence »** utilisée pour ROUGE-L et BERTScore. BSARD est un dataset de *retrieval* : il n'y a pas de réponse rédigée gold. Si la référence est la concaténation des articles pertinents, **le dire explicitement** — cela conditionne l'interprétation de toutes les métriques de surface et doit être discuté comme une limite dans le rapport.
4. Produire un **plan d'exécution chiffré** : liste des runs, durée GPU estimée de chacun, total, et priorisation P0/P1/P2 tenant compte des ~4 semaines restantes et du quota Colab gratuit.
5. **S'arrêter et me présenter ce plan.** Ne pas commencer la Phase 1 sans validation.

### Phases suivantes

- **Phase 1** : `src/` + notebook 00 (smoke test complet en < 15 min, qui valide toute la chaîne sur 10 questions).
- **Phase 2** : données, index, évaluation retrieval (notebooks 01–03) → premiers chiffres comparables à ceux du §2.5.
- **Phase 3** : fine-tuning QLoRA (notebook 04).
- **Phase 4** : génération des 4 configurations + évaluation complète (notebooks 05–06).
- **Phase 5** : ablations, analyse d'erreurs, arène (07–09).
- **Phase 6** : figures, tableaux LaTeX, `RESULTS.md`, `DIFFICULTES.md`, `references.bib`, plans de rapport.

Après chaque phase : **résumé de ce qui a été produit, chiffres obtenus, écarts avec l'attendu, et proposition pour la suite.**

---

## 6. INSTRUCTIONS D'INSTALLATION — EXIGENCE SPÉCIFIQUE

`INSTALL.md` doit permettre à quelqu'un qui n'a jamais ouvert le projet de tout faire tourner **sans jamais avoir à deviner**. Format imposé : étapes numérotées, une action par étape, commande exacte, résultat attendu, et que faire en cas d'échec.

Doit couvrir :

1. Création/clonage du dépôt GitHub, avec la commande exacte.
2. Ouverture de chaque notebook dans Colab (badge « Open in Colab » + méthode manuelle GitHub → Colab).
3. **Activation du GPU** : Exécution → Modifier le type d'exécution → T4 GPU. Vérification par `!nvidia-smi` avec la sortie attendue.
4. **Montage de Google Drive**, création de l'arborescence de travail, quantité d'espace nécessaire (prévoir ~15 Go).
5. **Compte HuggingFace et token** : création pas-à-pas, où le coller (Colab Secrets, pas en clair dans le notebook), et quels modèles nécessitent ou non ce token.
6. **Installation des dépendances** : `requirements-colab.txt` avec versions épinglées **testées ensemble**, ordre d'installation, redémarrage du runtime si nécessaire. Traiter explicitement le conflit `numpy` / `bitsandbytes` / CUDA déjà rencontré, et fournir une **procédure de secours** si Colab met à jour son image CUDA.
7. **Ordre d'exécution des notebooks** et dépendances entre eux (quel notebook produit quel fichier consommé par quel autre).
8. **Gestion du quota GPU gratuit** : durée de chaque notebook, où sont les checkpoints, comment reprendre après déconnexion, comment basculer sur Kaggle (30 h GPU/semaine gratuites) si le quota Colab est épuisé.
9. **Section dépannage** : au moins 8 erreurs concrètes avec message d'erreur exact et solution (OOM, `bitsandbytes` non compilé, Drive non monté, index absent, adaptateurs LoRA introuvables, session expirée, `article_ids` mal parsés, modèle gated).

Même exigence pour tout téléchargement : chaque modèle, dataset ou ressource externe doit avoir sa taille, sa durée de téléchargement, son emplacement de cache et sa commande exacte.

---

## 7. LIVRABLES POUR LE RAPPORT

- **Figures** en PDF vectoriel : courbe Recall@k par modèle d'embedding, barres des 4 configurations avec IC 95 %, heatmap configuration × catégorie juridique, courbe d'apprentissage (taille du train), matrice d'erreurs, résultats de l'arène, matrice d'accord inter-annotateurs.
- **Tableaux `.tex`** générés automatiquement depuis les JSON de résultats (`booktabs`), directement `\input{}`-ables dans Overleaf.
- **`references.bib`** : ≥ 45 références **vérifiées**, couvrant au minimum — fine-tuning efficient (LoRA, QLoRA, adapters, surveys PEFT) ; RAG (Lewis 2020, surveys Gao 2023, retrieval dense, rerankers, RAG avancé) ; comparaisons RAG vs fine-tuning (Ovadia et al., Balaguer et al. agriculture, Dodgson et al., **Bouvard et al. APIA 2024**) ; évaluation de QA et de LLM (RAGAS, BERTScore, ROUGE, LLM-as-judge, Chatbot Arena, hallucination surveys) ; **NLP juridique** (BSARD/Louis & Spanakis, LLeQA, LegalBench, CUAD, LexGLUE, modèles juridiques francophones) ; Mistral 7B, Llama 2, quantification.
- **Plans de rapport** structurés (FR pour DSC, EN pour MLDM), avec pour chaque section les figures/tableaux à insérer et les références à citer. Le plan DSC doit contenir une section « Pourquoi le domaine juridique ? » et une section « Techniques classiques d'évaluation des systèmes d'IA générative » **avant** les expériences, conformément aux consignes de Jacquenet.
- **`DIFFICULTES.md`** : journal des problèmes techniques rencontrés avec leur résolution, rédigé sous forme de **recommandations pratiques d'usage des bibliothèques** — c'est le format demandé par Habrard pour l'annexe.

---

## 8. PRIORISATION (4 semaines restantes)

- **P0 — indispensable** : pipeline complet reproductible, 4 configurations évaluées sur les 222 questions, métriques de citation et d'hallucination, IC bootstrap, analyse par sous-groupes de lois, `INSTALL.md`, figures et tableaux.
- **P1 — fortement souhaitable** : ablations retrieval et k, courbe d'apprentissage, analyse d'erreurs annotée, arène humaine.
- **P2 — si le temps le permet** : LLM-as-judge validé contre l'humain, ablations LoRA, jeu d'abstention étendu, second modèle de base (Qwen2.5-7B) pour tester la généralisation des conclusions.

---

## 9. POUR COMMENCER

Exécute la **Phase 0** et présente-moi :
1. Le résultat des vérifications d'accessibilité des ressources.
2. La structure réelle de BSARD et la façon dont la référence de génération sera construite.
3. Le plan d'exécution chiffré en heures GPU, priorisé P0/P1/P2.
4. Les risques identifiés et tes recommandations, y compris tout point où tu es en désaccord avec ce cahier des charges.

Puis attends ma validation avant d'écrire du code.
