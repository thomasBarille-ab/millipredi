"""
Baseline aléatoire : génère des grilles uniformes sans mémoire.
Sert de point de comparaison théorique pour tous les autres modèles.
"""

import numpy as np
from numpy.random import default_rng

N_NUMBERS = 50
N_STARS = 12
SEED = 42


class RandomBaseline:
    def __init__(self, seed: int = SEED):
        self.rng = default_rng(seed)

    def predict(self, n_grilles: int = 1) -> np.ndarray:
        """
        Retourne n_grilles tirages aléatoires uniformes.
        Shape : (n_grilles, 7) — [n1, n2, n3, n4, n5, e1, e2]
        """
        results = []
        for _ in range(n_grilles):
            nums = self.rng.choice(N_NUMBERS, size=5, replace=False) + 1
            nums.sort()
            stars = self.rng.choice(N_STARS, size=2, replace=False) + 1
            stars.sort()
            results.append(np.concatenate([nums, stars]))
        return np.array(results)

    def predict_onehot(self, n_grilles: int = 1) -> np.ndarray:
        """Retourne les grilles encodées en one-hot (62 dimensions)."""
        grilles = self.predict(n_grilles)
        out = np.zeros((n_grilles, N_NUMBERS + N_STARS), dtype=np.float32)
        for i, g in enumerate(grilles):
            for n in g[:5]:
                out[i, int(n) - 1] = 1.0
            for s in g[5:]:
                out[i, N_NUMBERS + int(s) - 1] = 1.0
        return out


def expected_hits_random() -> dict:
    """
    Calcul analytique du nombre moyen de numéros corrects d'une grille aléatoire
    par rapport au tirage réel, sous l'hypothèse d'indépendance uniforme.

    Pour les numéros : E[bons numéros] = 5 * (5/50) = 0.5
    Pour les étoiles : E[bonnes étoiles] = 2 * (2/12) ≈ 0.333
    """
    e_nums = 5 * (5 / N_NUMBERS)
    e_stars = 2 * (2 / N_STARS)
    return {
        "expected_nums": e_nums,
        "expected_stars": e_stars,
        "expected_total": e_nums + e_stars,
    }
