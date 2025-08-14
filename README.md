# BERT-From-Scratch-for-Sentiment-Analysis



**Objectif :**

```
Ce projet implémente un modèle BERT entraîné from scratch pour l’analyse de sentiment.
Il couvre tout le pipeline : prétraitement des données, entraînement, inférence, et visualisation des performances.
L’objectif est de comprendre en profondeur le fonctionnement de BERT et de montrer comment le construire, l’entraîner et l’évaluer sans se limiter à l’utilisation de modèles pré-entraînés.

```

---



🚀 **Fonctionnalités :**

```
- Prétraitement et nettoyage des données textuelles
- Construction et entraînement d’un modèle BERT from scratch
- Évaluation avec métriques de classification (accuracy, F1-score…)
- Inférence sur de nouveaux textes
- Visualisation des performances avec courbes et graphiques

```

---

🛠️ **Technologies utilisées :**

```
- Langage : Python
- NLP & Deep Learning : PyTorch, Transformers (ou implémentation custom selon le repo)
- Traitement des données : pandas, NumPy
- Visualisation : Matplotlib, Seaborn
- Scripts :
  - 🎯 `train.py` → Entraînement du modèle  
  - 🔍 `inference.py` → Prédictions sur nouveaux échantillons  
  - 📊 `plot.py` → Visualisation des résultats

```
---

## 📂 Structure du projet  

```bash
BERT-From-Scratch-for-Sentiment-Analysis/
│
├── data/ # Jeux de données
├── models/ # Modèles sauvegardés
├── scripts/ # Scripts utilitaires
│       - train.py → Script d’entraînement
│       - inference.py → Script de prédiction
│       - plot.py → Visualisation des métriques
└── README.md

```


---

## 🚀 Exécution du projet  



1️⃣ **Cloner le dépôt**

```bash
git clone https://github.com/JulWebana/BERT-From-Scratch-for-Sentiment-Analysis.git
cd BERT-From-Scratch-for-Sentiment-Analysis

```

---


2️⃣ Installer les dépendances

```
pip install -r requirements.txt

```
---

3️⃣ Entraîner le modèle

```
python train.py

```

---

4️⃣ Faire une prédiction

```
python inference.py --text "I love this product!"

```


---

📄 Licence
```
Ce projet est sous licence MIT – utilisation libre et modification autorisée.

```


---
