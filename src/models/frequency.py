"""
Heuristique fréquence pondérée : sélectionne les numéros selon leur fréquence
historique (biais vers les plus/moins sortis).

Deux variantes :
  - "hot"  : favorise les numéros les plus fréquents (gambler's fallacy version 1)
  - "cold" : favorise les numéros les moins fréquents (gambler's fallacy version 2)
"""

import numpy as np
import pandas as pd
from numpy.random import default_rng

N_NUMBERS = 50
N_STARS = 12
SEED = 42


class FrequencyModel:
    def __init__(self, mode: str = "hot", seed: int = SEED):
        assert mode in ("hot", "cold"), "mode doit être 'hot' ou 'cold'"
        self.mode = mode
        self.rng = default_rng(seed)
        self.num_proba: np.ndarray | None = None
        self.star_proba: np.ndarray | None = None
        self.trained = False

    def fit(self, df: pd.DataFrame) -> None:
        """Calcule les fréquences historiques sur l'ensemble d'entraînement."""
        nums = pd.concat([df["n1"], df["n2"], df["n3"], df["n4"], df["n5"]])
        freq_n = nums.value_counts().reindex(range(1, N_NUMBERS + 1), fill_value=0).values.astype(float)

        stars = pd.concat([df["etoile1"], df["etoile2"]])
        freq_s = stars.value_counts().reindex(range(1, N_STARS + 1), fill_value=0).values.astype(float)

        if self.mode == "cold":
            freq_n = freq_n.max() - freq_n + 1
            freq_s = freq_s.max() - freq_s + 1

        self.num_proba = freq_n / freq_n.sum()
        self.star_proba = freq_s / freq_s.sum()
        self.trained = True
        print(f"[frequency/{self.mode}] Entraînement terminé.")

    def predict_grille(self, n: int = 1) -> np.ndarray:
        """Génère n grilles pondérées par fréquence. Shape : (n, 7)."""
        assert self.trained, "Appeler fit() d'abord."
        grilles = []
        for _ in range(n):
            nums = self.rng.choice(N_NUMBERS, size=5, replace=False, p=self.num_proba) + 1
            nums.sort()
            stars = self.rng.choice(N_STARS, size=2, replace=False, p=self.star_proba) + 1
            stars.sort()
            grilles.append(np.concatenate([nums, stars]))
        return np.array(grilles)
