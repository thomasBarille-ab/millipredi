"""
Random Forest : prédit la probabilité de sortie de chaque numéro/étoile
à partir d'une fenêtre glissante des W tirages précédents.

Un classifier binaire indépendant est entraîné par position (62 au total).
Chaque classifier répond à : "ce numéro sortira-t-il au prochain tirage ?"
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "outputs" / "results"
SEED = 42


class RFModel:
    def __init__(self, n_estimators: int = 200, seed: int = SEED):
        base = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.model = MultiOutputClassifier(base, n_jobs=-1)
        self.trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """X : (N, window*62), y : (N, 62) binaire."""
        print(f"[rf] Entraînement sur {X.shape[0]} exemples...")
        self.model.fit(X, y.astype(int))
        self.trained = True
        print("[rf] Entraînement terminé.")

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """Retourne la matrice de probabilités (N, 62) — P(numéro sort)."""
        proba_list = self.model.predict_proba(X)
        # Chaque élément de proba_list est (N, 2) ; on prend la colonne classe=1
        out = np.stack([p[:, 1] if p.shape[1] == 2 else p[:, 0] for p in proba_list], axis=1)
        return out.astype(np.float32)

    def predict_grille(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit une grille (5 numéros + 2 étoiles) en prenant les top-k probabilités.
        Shape retournée : (N, 7)
        """
        proba = self.predict_proba_matrix(X)
        n = proba.shape[0]
        grilles = np.zeros((n, 7), dtype=int)

        for i in range(n):
            num_proba = proba[i, :50]
            star_proba = proba[i, 50:]
            nums = np.argsort(num_proba)[-5:][::-1] + 1
            nums.sort()
            stars = np.argsort(star_proba)[-2:][::-1] + 1
            stars.sort()
            grilles[i] = np.concatenate([nums, stars])

        return grilles

    def save(self, path: Path | None = None) -> None:
        if path is None:
            path = MODELS_DIR / "rf_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[rf] Modèle sauvegardé : {path}")

    def load(self, path: Path | None = None) -> None:
        if path is None:
            path = MODELS_DIR / "rf_model.pkl"
        self.model = joblib.load(path)
        self.trained = True
        print(f"[rf] Modèle chargé : {path}")
