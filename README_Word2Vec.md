# 🎬 Word2Vec - Analyse Sémantique des Films

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Gensim](https://img.shields.io/badge/Gensim-4.3+-green.svg)](https://radimrehurek.com/gensim/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Description

Ce projet implémente un modèle **Word2Vec** entraîné sur le dataset Kaggle Movies pour apprendre des représentations vectorielles (embeddings) des mots à partir des synopsis de films. Le modèle capture les relations sémantiques entre les mots du domaine cinématographique.

## 🎯 Objectifs

- Prétraiter un corpus de descriptions de films
- Entraîner un modèle Word2Vec avec Gensim
- Explorer les similarités sémantiques entre mots
- Visualiser les embeddings en 2D avec t-SNE
- Évaluer les performances via des tests d'analogies

## 📊 Dataset

**Source** : [Kaggle Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Le dataset contient des métadonnées de films incluant :
- Titres et synopsis (descriptions)
- Genres, dates de sortie
- Informations de production

## 🛠️ Installation

### Prérequis

```bash
Python 3.8+
```

### Dépendances

```bash
pip install gensim pandas nltk matplotlib seaborn scikit-learn numpy
```

Ou via le fichier requirements :

```bash
pip install -r requirements.txt
```

### Téléchargement des ressources NLTK

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 📁 Structure du Projet

```
word2vec-movies/
│
├── word2vec_movies.ipynb    # Notebook principal
├── word2vec_movies.model    # Modèle entraîné (généré)
├── movies_metadata.csv      # Dataset (à télécharger)
├── README.md                # Ce fichier
└── requirements.txt         # Dépendances
```

## 🚀 Utilisation

### 1. Exécuter le notebook

```bash
jupyter notebook word2vec_movies.ipynb
```

### 2. Charger un modèle pré-entraîné

```python
from gensim.models import Word2Vec

# Charger le modèle
model = Word2Vec.load("word2vec_movies.model")

# Trouver les mots similaires
similar_words = model.wv.most_similar("love", topn=5)
print(similar_words)

# Test d'analogie
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(result)  # -> queen
```

## 📈 Résultats

### Performances Globales

| Métrique | Valeur |
|----------|--------|
| Score moyen de similarité | **0.581** |
| Médiane | 0.567 |
| Écart-type | 0.082 |

### Tests de Similarité

| Mot | Top 3 mots similaires |
|-----|----------------------|
| `love` | falls (0.72), meets (0.64), romantic (0.57) |
| `family` | father (0.67), mother (0.65), home (0.64) |
| `murder` | case (0.62), crime (0.60), detective (0.57) |
| `hero` | villain (0.41), army (0.40), world (0.39) |

### Tests d'Analogies

| Analogie | Résultat attendu | Résultat obtenu | Score |
|----------|------------------|-----------------|-------|
| king - man + woman | queen | ✅ queen | 0.550 |
| evil - good + hero | villain | ⚠️ villain (3ème) | 0.388 |

### Visualisations t-SNE

Le modèle produit des clusters cohérents :
- **Genres** : thriller/horror proches, documentary isolé
- **Famille** : father, mother, son, daughter regroupés
- **Personnages** : hero, villain dans des zones proches

## ⚙️ Hyperparamètres du Modèle

```python
Word2Vec(
    sentences=corpus,
    vector_size=100,    # Dimension des vecteurs
    window=5,           # Taille de la fenêtre contextuelle
    min_count=5,        # Fréquence minimale des mots
    workers=4,          # Parallélisation
    sg=0,               # CBOW (0) ou Skip-gram (1)
    epochs=10           # Nombre d'itérations
)
```

## 🔍 Analyse des Résultats

### Points Forts

✅ Excellente capture des relations sémantiques thématiques (famille, romance, crime)

✅ Analogie classique "king - man + woman = queen" réussie

✅ Clusters t-SNE visuellement interprétables

✅ Relations contextuelles hero/villain bien capturées

### Limites

⚠️ Vocabulaire spécialisé : "antagonist" absent, "protagonist" peu similaire à "hero"

⚠️ Relations abstraites moins bien capturées (day/night/dark)

⚠️ Biais du corpus orienté synopsis de films

## 📚 Références

- [Word2Vec Paper (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [t-SNE Visualization](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

## 👤 Auteur

**Lys** - Master 2 Big Data & Business Intelligence, Sorbonne Paris Nord

---

## 📄 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

*Projet réalisé dans le cadre du cours de Machine Learning - M2 BIDABI*
