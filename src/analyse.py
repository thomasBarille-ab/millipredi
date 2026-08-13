"""
Analyse exploratoire des tirages EuroMillions.

Produit :
  - outputs/figures/freq_numeros.png
  - outputs/figures/freq_etoiles.png
  - outputs/figures/ecarts_heatmap.png
  - outputs/figures/evolution_temps.png
  - outputs/results/chi2_test.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "outputs" / "figures"
RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "results"

N_NUMBERS = 50
N_STARS = 12


def _freq_numeros(df: pd.DataFrame) -> pd.Series:
    nums = pd.concat([df["n1"], df["n2"], df["n3"], df["n4"], df["n5"]])
    return nums.value_counts().sort_index()


def _freq_etoiles(df: pd.DataFrame) -> pd.Series:
    stars = pd.concat([df["etoile1"], df["etoile2"]])
    return stars.value_counts().sort_index()


def plot_frequencies(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_tirages = len(df)

    freq_n = _freq_numeros(df)
    expected_n = n_tirages * 5 / N_NUMBERS

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(freq_n.index, freq_n.values, color="steelblue", alpha=0.8, label="Observé")
    ax.axhline(expected_n, color="red", linestyle="--", linewidth=1.5,
               label=f"Attendu uniforme ({expected_n:.1f})")
    ax.set_title(f"Fréquence d'apparition des numéros (1-50) sur {n_tirages} tirages")
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Nombre d'apparitions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "freq_numeros.png", dpi=150)
    plt.close(fig)
    print("[analyse] Sauvegardé : freq_numeros.png")

    freq_e = _freq_etoiles(df)
    expected_e = n_tirages * 2 / N_STARS

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(freq_e.index, freq_e.values, color="goldenrod", alpha=0.8, label="Observé")
    ax.axhline(expected_e, color="red", linestyle="--", linewidth=1.5,
               label=f"Attendu uniforme ({expected_e:.1f})")
    ax.set_title(f"Fréquence d'apparition des étoiles (1-12) sur {n_tirages} tirages")
    ax.set_xlabel("Étoile")
    ax.set_ylabel("Nombre d'apparitions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "freq_etoiles.png", dpi=150)
    plt.close(fig)
    print("[analyse] Sauvegardé : freq_etoiles.png")


def plot_ecarts(df: pd.DataFrame) -> None:
    """Heatmap des écarts (nb de tirages depuis la dernière apparition de chaque numéro)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    last_seen = {n: -1 for n in range(1, N_NUMBERS + 1)}
    ecarts_history = {n: [] for n in range(1, N_NUMBERS + 1)}

    for i, row in df.iterrows():
        drawn = {int(row[c]) for c in ["n1","n2","n3","n4","n5"]}
        for n in range(1, N_NUMBERS + 1):
            ecart = i - last_seen[n] if last_seen[n] >= 0 else 0
            ecarts_history[n].append(ecart)
            if n in drawn:
                last_seen[n] = i

    # On prend les 100 derniers tirages pour la lisibilité
    tail = 100
    matrix = np.array([ecarts_history[n][-tail:] for n in range(1, N_NUMBERS + 1)])

    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(matrix, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Écart (tirages)"})
    ax.set_title(f"Heatmap des écarts — {tail} derniers tirages")
    ax.set_xlabel("Tirage (index)")
    ax.set_ylabel("Numéro")
    ax.set_yticks(np.arange(0, N_NUMBERS, 5) + 0.5)
    ax.set_yticklabels(range(1, N_NUMBERS + 1, 5))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ecarts_heatmap.png", dpi=150)
    plt.close(fig)
    print("[analyse] Sauvegardé : ecarts_heatmap.png")


def plot_evolution(df: pd.DataFrame, rolling: int = 50) -> None:
    """Fréquence cumulée de quelques numéros dans le temps."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sample_nums = [7, 17, 23, 42, 50]
    fig, ax = plt.subplots(figsize=(14, 5))

    for n in sample_nums:
        mask = (df[["n1","n2","n3","n4","n5"]] == n).any(axis=1).astype(int)
        cumfreq = mask.cumsum() / (np.arange(len(df)) + 1)
        ax.plot(df["date"], cumfreq, label=f"N°{n}")

    ax.axhline(5 / 50, color="black", linestyle="--", linewidth=1.5, label="Théorique (5/50)")
    ax.set_title("Fréquence cumulée de quelques numéros dans le temps")
    ax.set_xlabel("Date")
    ax.set_ylabel("Fréquence cumulée")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "evolution_temps.png", dpi=150)
    plt.close(fig)
    print("[analyse] Sauvegardé : evolution_temps.png")


def chi2_test(df: pd.DataFrame) -> pd.DataFrame:
    """Test chi² d'uniformité pour les numéros et les étoiles."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    freq_n = _freq_numeros(df).reindex(range(1, N_NUMBERS + 1), fill_value=0)
    chi2_n, p_n = stats.chisquare(freq_n.values)

    freq_e = _freq_etoiles(df).reindex(range(1, N_STARS + 1), fill_value=0)
    chi2_e, p_e = stats.chisquare(freq_e.values)

    results = pd.DataFrame([
        {"cible": "numéros (1-50)", "chi2": chi2_n, "p_value": p_n,
         "conclusion": "uniforme" if p_n > 0.05 else "non uniforme"},
        {"cible": "étoiles (1-12)", "chi2": chi2_e, "p_value": p_e,
         "conclusion": "uniforme" if p_e > 0.05 else "non uniforme"},
    ])

    out = RESULTS_DIR / "chi2_test.csv"
    results.to_csv(out, index=False)
    print("[analyse] Test chi² :")
    print(results.to_string(index=False))
    print(f"[analyse] Sauvegardé : {out}")
    return results


def run(df: pd.DataFrame) -> None:
    print(f"\n[analyse] {len(df)} tirages chargés.")
    plot_frequencies(df)
    plot_ecarts(df)
    plot_evolution(df)
    chi2_test(df)
    print("[analyse] Analyse terminée.")


if __name__ == "__main__":
    from collect import collect
    df = collect()
    run(df)
