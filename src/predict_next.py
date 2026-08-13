"""
Génère les prédictions de chaque modèle pour le prochain tirage EuroMillions
et les enregistre dans outputs/results/predictions_log.csv.

Le log accumule les prédictions au fil du temps. Chaque entrée contient :
  - made_at       : horodatage de la prédiction
  - after_date    : date du dernier tirage connu (le prochain tirage sera après cette date)
  - model         : nom du modèle
  - n1..n5        : numéros prédits (triés)
  - etoile1/2     : étoiles prédites (triées)

Quand collect.py récupère de nouvelles données, le dashboard détecte
automatiquement si le tirage prévu a eu lieu et affiche le résultat réel.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from features import build_features, WINDOW
from models.baseline import RandomBaseline
from models.random_forest import RFModel
from models.lstm import LSTMModel
from models.frequency import FrequencyModel

RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "results"
DATA_PATH   = Path(__file__).parent.parent / "data" / "raw" / "euromillions.csv"
LOG_PATH    = RESULTS_DIR / "predictions_log.csv"

SEED = 42

LABELS = {
    "baseline":  "🎲 Baseline aléatoire",
    "rf":        "🌲 Random Forest",
    "lstm":      "🧠 LSTM",
    "freq_hot":  "🔥 Fréquence hot",
    "freq_cold": "❄️ Fréquence cold",
}


def _grille_to_str(nums: list[int], stars: list[int]) -> tuple[str, str]:
    return " · ".join(f"{n:02d}" for n in sorted(nums)), \
           " · ".join(f"{s:02d}" for s in sorted(stars))


def generate_predictions() -> pd.DataFrame:
    """Génère une prédiction de chaque modèle pour le prochain tirage."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    last_date = df["date"].max()
    print(f"[predict] Dernier tirage connu : {last_date.date()}")

    # Features — on prend la toute dernière fenêtre de W tirages
    X_flat, X_seq, _ = build_features(df, window=WINDOW)
    last_flat = X_flat[[-1]]   # (1, W*62)
    last_seq  = X_seq[[-1]]    # (1, W, 62)

    rows = []
    made_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Baseline ──────────────────────────────────────────────────────────
    baseline = RandomBaseline(seed=SEED)
    g = baseline.predict(1)[0]
    rows.append(_make_row(made_at, last_date, "baseline", g[:5], g[5:]))

    # ── Random Forest ─────────────────────────────────────────────────────
    rf_path = RESULTS_DIR / "rf_model.pkl"
    if rf_path.exists():
        rf = RFModel()
        rf.load(rf_path)
        g = rf.predict_grille(last_flat)[0]
        rows.append(_make_row(made_at, last_date, "rf", g[:5], g[5:]))
    else:
        print("[predict] Random Forest non trouvé — ignoré.")

    # ── LSTM ──────────────────────────────────────────────────────────────
    lstm_path = RESULTS_DIR / "lstm_model.pt"
    if lstm_path.exists():
        lstm = LSTMModel()
        lstm.load(lstm_path)
        g = lstm.predict_grille(last_seq)[0]
        rows.append(_make_row(made_at, last_date, "lstm", g[:5], g[5:]))
    else:
        print("[predict] LSTM non trouvé — ignoré.")

    # ── Fréquence hot / cold ──────────────────────────────────────────────
    for mode in ("hot", "cold"):
        freq = FrequencyModel(mode=mode, seed=SEED)
        freq.fit(df)
        g = freq.predict_grille(1)[0]
        rows.append(_make_row(made_at, last_date, f"freq_{mode}", g[:5], g[5:]))

    new_df = pd.DataFrame(rows)

    # Ajoute au log existant (sans doublons sur made_at + model)
    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["made_at", "model"], keep="last")
    else:
        combined = new_df

    combined.to_csv(LOG_PATH, index=False)
    print(f"[predict] {len(new_df)} prédictions sauvegardées dans {LOG_PATH}")
    return new_df


def _make_row(made_at: str, last_date: pd.Timestamp, model: str,
              nums: np.ndarray, stars: np.ndarray) -> dict:
    nums  = sorted(int(x) for x in nums)
    stars = sorted(int(x) for x in stars)
    return {
        "made_at":    made_at,
        "after_date": last_date.strftime("%Y-%m-%d"),
        "model":      model,
        "label":      LABELS.get(model, model),
        "n1": nums[0], "n2": nums[1], "n3": nums[2], "n4": nums[3], "n5": nums[4],
        "etoile1": stars[0], "etoile2": stars[1],
    }


def load_log_with_results() -> pd.DataFrame:
    """
    Charge le log de prédictions et enrichit chaque ligne avec le résultat
    réel si le tirage correspondant a eu lieu dans les données.
    """
    if not LOG_PATH.exists():
        return pd.DataFrame()

    log = pd.read_csv(LOG_PATH, parse_dates=["after_date"])
    df  = pd.read_csv(DATA_PATH, parse_dates=["date"])

    results = []
    for _, row in log.iterrows():
        # Cherche le premier tirage survenu APRÈS after_date
        future = df[df["date"] > row["after_date"]].sort_values("date")
        if future.empty:
            actual = {
                "draw_date": None,
                "actual_nums":  "En attente...",
                "actual_stars": "En attente...",
                "hits_nums": None, "hits_stars": None, "hits_total": None,
            }
        else:
            real = future.iloc[0]
            pred_nums  = {row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]}
            pred_stars = {row["etoile1"], row["etoile2"]}
            real_nums  = {int(real[c]) for c in ["n1","n2","n3","n4","n5"]}
            real_stars = {int(real[c]) for c in ["etoile1","etoile2"]}
            hn = len(pred_nums  & real_nums)
            hs = len(pred_stars & real_stars)
            actual = {
                "draw_date":    real["date"].strftime("%d/%m/%Y"),
                "actual_nums":  " · ".join(f"{n:02d}" for n in sorted(real_nums)),
                "actual_stars": " · ".join(f"{s:02d}" for s in sorted(real_stars)),
                "hits_nums":  hn,
                "hits_stars": hs,
                "hits_total": hn + hs,
            }
        results.append({**row.to_dict(), **actual})

    return pd.DataFrame(results)


if __name__ == "__main__":
    generate_predictions()
