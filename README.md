# EuroMillions ML — Démonstration des limites du Machine Learning

## Objectif

Entraîner plusieurs modèles de Machine Learning sur l'historique des tirages EuroMillions
et mesurer s'ils font mieux qu'un tirage aléatoire — spoiler : **ils ne le font pas**.

Ce projet explore :
- La distinction entre processus stochastique pur et signal exploitable
- L'évaluation rigoureuse de modèles ML (split temporel, tests statistiques)
- Les biais cognitifs liés aux jeux de hasard (gambler's fallacy)

## Structure

```
euromillions-ml/
├── src/
│   ├── collect.py          # Collecte des données (mes-resultats-fdj.fr + fallback CSV)
│   ├── features.py         # Fenêtres glissantes (W=10), encodage one-hot 62 dims
│   ├── analyse.py          # EDA : fréquences, écarts, test chi²
│   ├── scheduler.py        # Logique de scheduling (veille/lendemain des tirages)
│   ├── predict_next.py     # Génération des prédictions pour le prochain tirage
│   ├── models/
│   │   ├── baseline.py     # Tirage aléatoire uniforme (référence)
│   │   ├── random_forest.py
│   │   ├── lstm.py         # PyTorch — 2 couches, hidden=128
│   │   └── frequency.py    # Heuristique fréquence hot/cold
│   └── evaluate.py         # Métriques + tests statistiques
├── app.py                  # Dashboard Streamlit (3 onglets)
├── run_pipeline.py         # Lance tout le pipeline
├── auto_update.py          # Runner quotidien (Planificateur Windows)
└── requirements.txt
```

## Installation

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```powershell
# Pipeline complet (collecte → analyse → entraînement → évaluation)
python run_pipeline.py

# Relancer uniquement l'entraînement (données déjà collectées)
python run_pipeline.py --skip-collect --skip-analyse

# Lancer le dashboard
streamlit run app.py
```

## Données

- **Source** : [mes-resultats-fdj.fr](https://www.mes-resultats-fdj.fr/api/telecharger/euromillions) — CSV complet fusionné
- **Périmètre** : tirages du **27/09/2016 au 11/08/2026** (format unifié à 12 étoiles)
- **Volume** : **1 031 tirages** après filtre (suppression des données pré-2016 pour uniformité du format)
- **Split** : 80% train (826 tirages) / 20% test (205 tirages) — **split temporel strict, sans shuffle**

## Analyse exploratoire — Résultats

### Test chi² d'uniformité (sur 1 031 tirages)

| Cible | chi² | p-value | Conclusion |
|---|---|---|---|
| Numéros (1-50) | 47.22 | **0.545** | ✅ Uniforme |
| Étoiles (1-12) | 14.72 | **0.196** | ✅ Uniforme |

p-value >> 0.05 dans les deux cas : on ne peut pas rejeter l'hypothèse d'uniformité.
Aucun numéro ne sort significativement plus souvent qu'un autre.

## Résultats ML — Évaluation réelle

Entraînement réalisé le **13/08/2026** sur 826 tirages, évaluation sur 205 tirages.

### Métrique : hits moyens par tirage (numéros + étoiles corrects, max = 7)

| Modèle | Architecture | Hits moyen | Écart-type | t-stat vs baseline | p-value | Significatif ? |
|---|---|---|---|---|---|---|
| 🎲 Baseline aléatoire | Tirage uniforme (référence) | 0.766 | 0.897 | — | — | — |
| 🌲 Random Forest | 200 arbres, fenêtre 10 tirages, max_depth=6 | **0.839** | 0.795 | 0.872 | 0.384 | ❌ Non |
| 🧠 LSTM | 2 couches, hidden=128, 30 epochs, BCEWithLogitsLoss | **0.839** | 0.820 | 0.860 | 0.390 | ❌ Non |
| 🔥 Fréquence hot | Numéros les plus sortis historiquement | 0.820 | 0.833 | 0.626 | 0.532 | ❌ Non |
| ❄️ Fréquence cold | Numéros les moins sortis historiquement | 0.790 | 0.832 | 0.285 | 0.776 | ❌ Non |

### Conclusion

**Aucun modèle ne bat significativement la baseline aléatoire.**

Toutes les p-values sont largement supérieures à 0.05 (seuil classique de significativité).
Les légères différences de hits observées (0.766 à 0.839) sont statistiquement indiscernables
du bruit — c'est exactement ce qu'on attend d'un processus aléatoire certifié.

Points notables :
- Le **LSTM** voit sa loss stagner à ~0.349 dès l'epoch 5 et n'apprend rien de plus :
  il converge vers les fréquences marginales de chaque numéro, sans détecter de séquence.
- Le **Random Forest** produit des corrélations spurieuses sur le train set qui ne se
  généralisent pas au test set.
- Les heuristiques **hot/cold** valident empiriquement que la notion de numéro
  « chaud » ou « froid » est un biais cognitif sans fondement statistique.

## Pourquoi le ML échoue ici ?

1. **Indépendance des tirages** : le tirage N+1 est mathématiquement indépendant
   de tous les tirages précédents — il n'existe aucun signal à mémoriser.
2. **Distribution uniforme** : confirmée par le test chi² (p=0.55 pour les numéros) —
   tous les numéros sont équiprobables à chaque tirage.
3. **Entropie maximale** : un processus purement aléatoire a une entropie maximale,
   ce qui rend toute compression de l'information (= tout modèle prédictif) impossible.
4. **Overfitting sur du bruit** : avec suffisamment de paramètres, un modèle trouve
   toujours des patterns dans les données d'entraînement — qui ne se généralisent jamais.

## Dashboard interactif

```powershell
streamlit run app.py
```

Trois onglets :
- **Analyse exploratoire** : histogrammes de fréquence, heatmap des écarts, test chi²
- **Modèles & Prédictions** : comparaison des performances, violinplot, analyse des échecs
- **Prédictions vs Réalité** : tableau de suivi des prédictions sur les tirages réels,
  mis à jour automatiquement avant et après chaque tirage (mardi / vendredi)

## Mise à jour automatique

Le planificateur Windows (`EuroMillions-ML-AutoUpdate`) s'exécute chaque jour à 9h :

| Jour | Action |
|---|---|
| Lundi / Jeudi | 🔮 Génère les prédictions pour le tirage du lendemain |
| Mercredi / Samedi | 📡 Récupère les vrais résultats et met à jour le tableau |

