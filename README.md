# EuroMillions ML — Démonstration des limites du Machine Learning

> **Avertissement important** : Ce projet est un **exercice pédagogique**. L'EuroMillions
> utilise un générateur aléatoire certifié. Il est **mathématiquement impossible** de
> prédire les tirages. Ce projet existe pour démontrer ce fait en pratique.

## Objectif

Entraîner plusieurs modèles de Machine Learning sur l'historique des tirages EuroMillions
et mesurer s'ils font mieux qu'un tirage aléatoire — spoiler : **ils ne le font pas**.

C'est un exercice d'apprentissage sur :
- La distinction entre processus stochastique pur et signal exploitable
- L'évaluation rigoureuse de modèles ML (split temporel, tests statistiques)
- Les biais cognitifs liés aux jeux de hasard (gambler's fallacy)

## Structure

```
euromillions-ml/
├── src/
│   ├── collect.py          # Collecte des données (FDJ open data + fallback CSV)
│   ├── features.py         # Fenêtres glissantes, encodage one-hot
│   ├── analyse.py          # EDA : fréquences, écarts, test chi²
│   ├── models/
│   │   ├── baseline.py     # Tirage aléatoire uniforme (référence)
│   │   ├── random_forest.py
│   │   ├── lstm.py         # PyTorch
│   │   └── frequency.py    # Heuristique fréquence hot/cold
│   └── evaluate.py         # Métriques + tests statistiques
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_modelisation.ipynb
├── run_pipeline.py         # Lance tout le pipeline
└── requirements.txt
```

## Installation

```bash
python -m venv venv
# Windows :
venv\Scripts\activate
# Linux/Mac :
source venv/bin/activate

pip install -r requirements.txt
```

## Utilisation

```bash
# Pipeline complet
python run_pipeline.py

# Si le téléchargement auto échoue, placez votre CSV dans data/raw/euromillions_manual.csv
# Format : date,n1,n2,n3,n4,n5,etoile1,etoile2
python run_pipeline.py --skip-collect
```

## Résultats attendus

| Modèle | Hits moyens / tirage | p-value vs baseline |
|---|---|---|
| Baseline aléatoire | ~0.833 | — |
| Random Forest | ~0.833 | > 0.05 |
| LSTM | ~0.833 | > 0.05 |
| Fréquence (hot) | ~0.833 | > 0.05 |
| Fréquence (cold) | ~0.833 | > 0.05 |

**Conclusion** : Aucun modèle ne bat significativement la baseline aléatoire.
Le p-value > 0.05 confirme l'absence de signal exploitable — exactement ce qu'on attend
d'un processus aléatoire certifié.

## Pourquoi le ML échoue ici ?

1. **Indépendance des tirages** : chaque tirage est statistiquement indépendant du précédent
2. **Distribution uniforme** : le test chi² confirme que tous les numéros sortent avec la même probabilité
3. **Pas de motif temporel** : le LSTM ne peut pas capter ce qui n'existe pas
4. **Overfitting sur du bruit** : le Random Forest apprend des corrélations spurieuses qui ne se généralisent pas

Ce projet illustre une leçon fondamentale : un bon modèle ML a besoin d'un **signal** dans les données.
Quand les données sont du bruit pur, même le meilleur modèle ne peut rien faire.

---

*Jouer doit rester un loisir. Aucun système ne peut améliorer vos chances à l'EuroMillions.*
