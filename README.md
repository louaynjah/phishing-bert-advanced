<div align="center">

# 🎣 Phishing-BERT — Détection avancée d'e-mails de phishing

### Fine-tuning de BERT pour contrer les techniques d'hameçonnage de nouvelle génération

[![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-informational)](#méthodologie)
[![Model](https://img.shields.io/badge/Model-BERT--base-EE4C2C)](#modélisation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Accuracy 99.10 %** · **F1-score 98.86 %** · **Recall 99.01 %**
sur un ensemble de test où le phishing représente 39 % des exemples

</div>

---

## 📌 Le problème

Le courrier électronique reste la première porte d'entrée des cyberattaques. Les techniques de phishing évoluent vite — usurpation d'identité, typosquatting, *quishing* (QR code), deepfakes audio/vidéo, fraude à la chaîne d'approvisionnement — rendant obsolètes les modèles de détection entraînés sur des données anciennes.

Le modèle en place dans l'entreprise d'accueil, un BERT fine-tuné publié en 2023, affichait une accuracy annoncée de **99,68 %**. Une analyse critique a révélé que ce chiffre était trompeur :

| Limite identifiée | Détail |
|---|---|
| ⚖️ Déséquilibre sévère des classes | Seulement **0,86 %** du corpus (4 379 / 508 547 e-mails) était du phishing — un modèle prédisant toujours "légitime" aurait déjà dépassé 99 % d'accuracy |
| 🕰️ Données obsolètes | Corpus couvrant 1998–2015, techniques de phishing basiques uniquement |
| 🌍 Aucun support multilingue | Entraîné exclusivement sur de l'anglais (Enron, Nigerian Fraud, SpamAssassin) |

Ce projet reconstruit le pipeline de détection pour corriger ces trois limites.

---

## 🧭 Méthodologie

Le projet suit intégralement **CRISP-DM** (Cross-Industry Standard Process for Data Mining), structuré en cinq phases.

### 1️⃣ Compréhension du métier
Audit du modèle BERT existant : architecture, données d'entraînement, pipeline de déploiement ONNX. Diagnostic des trois limites ci-dessus, qui deviennent les objectifs du projet.

### 2️⃣ Compréhension des données
Analyse du framework **PhishFuzzer** (23 100 e-mails), qui génère des variantes d'e-mails réels via un LLM (*Entity-Seed Expansion*) : classification à 3 classes, métadonnées riches (URL, pièces jointes, motivation de l'attaquant). Deux limites persistent : couverture quasi exclusivement anglophone, absence des techniques les plus récentes (quishing, deepfake, HTML smuggling).

### 3️⃣ Préparation des données
Construction de deux jeux de données synthétiques complémentaires :
- **Génération par gabarits** (FR/EN) — scénarios courants, contournement de filtres antispam
- **Génération par LLM local** (Ollama / Llama 3.1) — menaces avancées : fraude supply chain, injections de prompt cachées (XPIA), ingénierie sociale OSINT

Fusion des trois sources en un corpus homogène de **44 655 e-mails**, bilingue, équilibré (~39 % phishing).

### 4️⃣ Modélisation
**Transfer learning** depuis le checkpoint BERT existant plutôt qu'un ré-entraînement générique — convergence plus rapide et comparaison équitable avec le modèle d'origine. Résolution d'un bug de compatibilité sur les couches `LayerNorm`, pondération de la fonction de perte pour le déséquilibre résiduel des classes, arrêt anticipé basé sur la perte de validation pour éviter le sur-apprentissage.

### 5️⃣ Évaluation
Sélection du meilleur modèle par comparaison systématique de plusieurs *checkpoints* sur un ensemble de test totalement indépendant.

---

## 🏆 Résultats

| Métrique | Score |
|---|---|
| **Accuracy** | 99.10 % |
| **Precision** | 98.71 % |
| **Recall** | 99.01 % |
| **F1-score** | 98.86 % |

<details>
<summary><b>📉 Matrice de confusion</b> (6 699 e-mails de test)</summary>

|  | Prédit : Légitime | Prédit : Phishing |
|---|---|---|
| **Réel : Légitime** (4 081) | 4 047 ✅ | 34 ⚠️ faux positifs |
| **Réel : Phishing** (2 618) | 26 🔴 faux négatifs | 2 592 ✅ |

- Taux de non-détection : **0,99 %** — l'erreur la plus critique en cybersécurité
- Taux de fausse alerte : **0,83 %**

</details>

<details>
<summary><b>📈 Tableau de bord d'entraînement</b></summary>

![Training dashboard](docs/training_dashboard.png)

Convergence rapide (effet du transfer learning), sélection du *checkpoint* de l'époque 4 par perte de validation minimale plutôt que par la meilleure accuracy brute — un choix plus prudent face au sur-apprentissage observé à partir de l'époque 3.

</details>

### Pourquoi ces chiffres sont plus fiables que ceux du modèle existant

L'accuracy de 99,10 % obtenue ici l'est sur un ensemble de test où le phishing représente **39,1 %** des exemples — une proportion **45 fois plus exigeante** que le 0,86 % du jeu de données d'origine. Combinée à un rappel de 99,01 %, cette performance constitue une preuve statistiquement robuste de la capacité réelle du modèle à distinguer le phishing des e-mails légitimes.

---

## 🗂️ Données

| Source | Volume | Rôle |
|---|---|---|
| PhishFuzzer | 23 100 e-mails | Base réelle enrichie en métadonnées, générée par *Entity-Seed Expansion* |
| Génération par gabarits | — | Complète la couverture francophone et les scénarios courants |
| Génération par LLM local | — | Couvre les menaces avancées (supply chain, OSINT, prompt injection) |

---


## 📄 Licence

Ce projet est distribué sous licence [MIT](LICENSE).

## 🙏 Remerciements

Projet réalisé dans le cadre d'un stage ingénieur chez **Novation City** (centre d'excellence en IA), sous l'encadrement de Mme Nehla Dabbebi.
