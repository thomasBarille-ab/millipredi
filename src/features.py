"""
Construction des features pour les modèles ML.

Approche : fenêtre glissante de W tirages consécutifs pour prédire le tirage suivant.
Chaque tirage est encodé en vecteur one-hot (50 + 12 = 62 dimensions).
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

N_NUMBERS = 50
N_STARS = 12
WINDOW = 10  # nombre de tirages précédents utilisés comme contexte


def tirage_to_onehot(row: pd.Series) -> np.ndarray:
    """Encode un tirage en vecteur binaire de dimension 62."""
    vec = np.zeros(N_NUMBERS + N_STARS, dtype=np.float32)
    for col in ["n1", "n2", "n3", "n4", "n5"]:
        idx = int(row[col]) - 1
        if 0 <= idx < N_NUMBERS:
            vec[idx] = 1.0
    for col in ["etoile1", "etoile2"]:
        idx = N_NUMBERS + int(row[col]) - 1
        if 0 <= idx < N_NUMBERS + N_STARS:
            vec[idx] = 1.0
    return vec


def build_features(df: pd.DataFrame, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit X et y à partir du DataFrame de tirages.

    X : (N - window, window * 62)  — fenêtre aplatie pour RF
    y : (N - window, 62)           — tirage cible en one-hot

    Retourne aussi X_seq de shape (N - window, window, 62) pour le LSTM.
    """
    onehots = np.stack([tirage_to_onehot(row) for _, row in df.iterrows()])

    X_seq, y = [], []
    for i in range(window, len(onehots)):
        X_seq.append(onehots[i - window : i])
        y.append(onehots[i])

    X_seq = np.array(X_seq, dtype=np.float32)   # (N, window, 62)
    y = np.array(y, dtype=np.float32)            # (N, 62)
    X_flat = X_seq.reshape(len(X_seq), -1)       # (N, window * 62) pour RF

    return X_flat, X_seq, y


def train_test_split_temporal(
    X_flat: np.ndarray,
    X_seq: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
) -> dict:
    """Split temporel strict — pas de shuffle."""
    n = len(y)
    split = int(n * (1 - test_ratio))
    return {
        "X_train_flat": X_flat[:split],
        "X_test_flat":  X_flat[split:],
        "X_train_seq":  X_seq[:split],
        "X_test_seq":   X_seq[split:],
        "y_train":      y[:split],
        "y_test":       y[split:],
        "split_idx":    split,
    }


def save_processed(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "euromillions.parquet"
    df.to_parquet(path, index=False)
    print(f"[features] Sauvegardé : {path}")


def load_processed() -> pd.DataFrame:
    path = PROCESSED_DIR / "euromillions.parquet"
    return pd.read_parquet(path)
