"""
Modèle LSTM (PyTorch) : séquence de W tirages en entrée, prédiction one-hot du suivant.

Architecture :
  - Entrée : (batch, W, 62)
  - LSTM bi-couche → hidden dim 128
  - Couche fully-connected → 62 sorties
  - Sigmoid par dimension (classification multi-label indépendante)
  - Loss : BCEWithLogitsLoss
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "outputs" / "results"
SEED = 42


class LSTMNet(nn.Module):
    def __init__(self, input_dim: int = 62, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # dernier pas de temps


class LSTMModel:
    def __init__(self, input_dim: int = 62, hidden_dim: int = 128, num_layers: int = 2,
                 lr: float = 1e-3, seed: int = SEED):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = LSTMNet(input_dim, hidden_dim, num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.trained = False
        print(f"[lstm] Device : {self.device}")

    def fit(self, X_seq: np.ndarray, y: np.ndarray,
            epochs: int = 30, batch_size: int = 64) -> list[float]:
        """X_seq : (N, W, 62), y : (N, 62)."""
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        losses = []
        self.net.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.net(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * len(xb)
            avg = epoch_loss / len(dataset)
            losses.append(avg)
            if epoch % 5 == 0 or epoch == 1:
                print(f"[lstm] Epoch {epoch:3d}/{epochs} — loss : {avg:.4f}")

        self.trained = True
        return losses

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Retourne les probabilités sigmoid (N, 62)."""
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            logits = self.net(X_t)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba.astype(np.float32)

    def predict_grille(self, X_seq: np.ndarray) -> np.ndarray:
        """Top-5 numéros + top-2 étoiles selon la probabilité. Shape : (N, 7)."""
        proba = self.predict_proba(X_seq)
        n = proba.shape[0]
        grilles = np.zeros((n, 7), dtype=int)
        for i in range(n):
            nums = np.argsort(proba[i, :50])[-5:][::-1] + 1
            nums.sort()
            stars = np.argsort(proba[i, 50:])[-2:][::-1] + 1
            stars.sort()
            grilles[i] = np.concatenate([nums, stars])
        return grilles

    def save(self, path: Path | None = None) -> None:
        if path is None:
            path = MODELS_DIR / "lstm_model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        print(f"[lstm] Modèle sauvegardé : {path}")

    def load(self, path: Path | None = None) -> None:
        if path is None:
            path = MODELS_DIR / "lstm_model.pt"
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.trained = True
        print(f"[lstm] Modèle chargé : {path}")
