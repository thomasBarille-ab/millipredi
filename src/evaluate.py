"""
Évaluation comparative de tous les modèles.

Métrique principale : nombre moyen de bons numéros prédits par tirage.
Test statistique : t-test de Welch entre chaque modèle et la baseline aléatoire.

Produit :
  - outputs/results/evaluation.csv
  - outputs/figures/comparaison_modeles.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "results"
FIGURES_DIR = Path(__file__).parent.parent / "outputs" / "figures"

N_NUMBERS = 50
N_STARS = 12


def _grille_from_row(row: pd.Series) -> tuple[set, set]:
    nums = {int(row[c]) for c in ["n1","n2","n3","n4","n5"]}
    stars = {int(row[c]) for c in ["etoile1","etoile2"]}
    return nums, stars


def hits_per_draw(predicted: np.ndarray, df_test: pd.DataFrame) -> np.ndarray:
    """
    Calcule le nombre de bons numéros (nums + étoiles) pour chaque tirage.
    predicted : (N, 7) — [n1..n5, e1, e2]
    Retourne : array (N,) de hits totaux (max 7).
    """
    assert len(predicted) == len(df_test), "Tailles incohérentes"
    scores = []
    for i, (_, row) in enumerate(df_test.iterrows()):
        true_nums, true_stars = _grille_from_row(row)
        pred_nums = set(predicted[i, :5].tolist())
        pred_stars = set(predicted[i, 5:].tolist())
        hit = len(true_nums & pred_nums) + len(true_stars & pred_stars)
        scores.append(hit)
    return np.array(scores)


def evaluate_model(name: str, predicted: np.ndarray, df_test: pd.DataFrame,
                   baseline_hits: np.ndarray) -> dict:
    scores = hits_per_draw(predicted, df_test)
    mean = scores.mean()
    std = scores.std()

    t_stat, p_val = stats.ttest_ind(scores, baseline_hits, equal_var=False)

    return {
        "modèle": name,
        "hits_moyen": round(mean, 4),
        "hits_std": round(std, 4),
        "t_stat_vs_baseline": round(t_stat, 4),
        "p_value_vs_baseline": round(p_val, 4),
        "significatif (p<0.05)": p_val < 0.05,
        "n_tirages_test": len(scores),
    }


def run_evaluation(models_predictions: dict[str, np.ndarray],
                   df_test: pd.DataFrame) -> pd.DataFrame:
    """
    models_predictions : {nom_modele: array (N, 7)}
    Inclure "baseline" dans le dict.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baseline_hits = hits_per_draw(models_predictions["baseline"], df_test)

    rows = []
    all_scores = {}
    for name, preds in models_predictions.items():
        result = evaluate_model(name, preds, df_test, baseline_hits)
        rows.append(result)
        all_scores[name] = hits_per_draw(preds, df_test)

    df_results = pd.DataFrame(rows)
    out = RESULTS_DIR / "evaluation.csv"
    df_results.to_csv(out, index=False)
    print("\n[evaluate] Résultats :")
    print(df_results.to_string(index=False))
    print(f"[evaluate] Sauvegardé : {out}")

    _save_history(models_predictions, df_test)
    _plot_comparison(all_scores, df_results)
    return df_results


def _save_history(models_predictions: dict[str, np.ndarray],
                  df_test: pd.DataFrame) -> None:
    """Sauvegarde le détail tirage par tirage dans history.csv."""
    LABELS = {
        "baseline":  "🎲 Baseline aléatoire",
        "rf":        "🌲 Random Forest",
        "lstm":      "🧠 LSTM",
        "freq_hot":  "🔥 Fréquence hot",
        "freq_cold": "❄️ Fréquence cold",
    }
    MODEL_ORDER = ["baseline", "rf", "lstm", "freq_hot", "freq_cold"]

    df_test = df_test.reset_index(drop=True)
    history_rows = []

    for i, (_, real) in enumerate(df_test.iterrows()):
        real_nums  = sorted(int(real[c]) for c in ["n1","n2","n3","n4","n5"])
        real_stars = sorted(int(real[c]) for c in ["etoile1","etoile2"])
        actual_nums_str  = " · ".join(f"{n:02d}" for n in real_nums)
        actual_stars_str = " · ".join(f"{s:02d}" for s in real_stars)
        date_str = pd.to_datetime(real["date"]).strftime("%d/%m/%Y") if "date" in real else str(i)

        for model in MODEL_ORDER:
            if model not in models_predictions:
                continue
            pred = models_predictions[model][i]
            pred_nums  = sorted(int(x) for x in pred[:5])
            pred_stars = sorted(int(x) for x in pred[5:])
            hn = len(set(pred_nums) & set(real_nums))
            hs = len(set(pred_stars) & set(real_stars))
            history_rows.append({
                "date":             date_str,
                "model":            model,
                "label":            LABELS.get(model, model),
                "pred_nums":        " · ".join(f"{n:02d}" for n in pred_nums),
                "pred_stars":       " · ".join(f"{s:02d}" for s in pred_stars),
                "actual_nums":      actual_nums_str,
                "actual_stars":     actual_stars_str,
                "hits_nums":        hn,
                "hits_stars":       hs,
                "hits_total":       hn + hs,
            })

    df_hist = pd.DataFrame(history_rows)
    out = RESULTS_DIR / "history.csv"
    df_hist.to_csv(out, index=False)
    print(f"[evaluate] Sauvegardé : {out} ({len(df_test)} tirages × {len(models_predictions)} modèles)")


def _plot_comparison(all_scores: dict[str, np.ndarray], df_results: pd.DataFrame) -> None:
    names = list(all_scores.keys())
    means = [all_scores[n].mean() for n in names]
    stds  = [all_scores[n].std() for n in names]

    colors = ["#2196F3" if n == "baseline" else "#FF5722" for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Barplot moyennes
    ax = axes[0]
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_title("Hits moyens par tirage (± 1 écart-type)")
    ax.set_ylabel("Nombre moyen de bons numéros")
    ax.set_ylim(0, max(means) * 1.4)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    # Distribution violinplot
    ax = axes[1]
    data = [all_scores[n] for n in names]
    vp = ax.violinplot(data, showmedians=True)
    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    ax.set_title("Distribution des hits par tirage")
    ax.set_ylabel("Hits")

    fig.suptitle("Comparaison des modèles — EuroMillions ne se prédit pas", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "comparaison_modeles.png", dpi=150)
    plt.close(fig)
    print("[evaluate] Sauvegardé : comparaison_modeles.png")
