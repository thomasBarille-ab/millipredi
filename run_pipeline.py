"""
Pipeline complet : collecte → analyse → entraînement → évaluation.
Usage : python run_pipeline.py [--skip-collect] [--skip-analyse]
"""

import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "src"))

from collect import collect
from features import build_features, train_test_split_temporal, save_processed
from analyse import run as run_analyse
from models.baseline import RandomBaseline, expected_hits_random
from models.random_forest import RFModel
from models.lstm import LSTMModel
from models.frequency import FrequencyModel
from evaluate import run_evaluation

SEED = 42
np.random.seed(SEED)


def main(skip_collect: bool = False, skip_analyse: bool = False) -> None:
    print("=" * 60)
    print("  Pipeline EuroMillions ML")
    print("=" * 60)

    # 1. Collecte
    print("\n[1/4] Collecte des données")
    df = collect(force=not skip_collect)
    save_processed(df)

    # 2. Analyse exploratoire
    if not skip_analyse:
        print("\n[2/4] Analyse exploratoire")
        run_analyse(df)
    else:
        print("\n[2/4] Analyse exploratoire — ignorée (--skip-analyse)")

    # 3. Construction des features & split
    print("\n[3/4] Entraînement des modèles")
    X_flat, X_seq, y = build_features(df, window=10)
    split = train_test_split_temporal(X_flat, X_seq, y, test_ratio=0.2)

    df_test = df.iloc[split["split_idx"] + 10:].reset_index(drop=True)
    n_test = len(split["y_test"])
    print(f"  Train : {len(split['y_train'])} exemples | Test : {n_test} exemples")

    # Baseline aléatoire
    baseline = RandomBaseline(seed=SEED)
    baseline_preds = baseline.predict(n_test)

    theory = expected_hits_random()
    print(f"  Baseline théorique — hits attendus : {theory['expected_total']:.4f} / tirage")

    # Random Forest
    rf = RFModel(seed=SEED)
    rf.fit(split["X_train_flat"], split["y_train"])
    rf_preds = rf.predict_grille(split["X_test_flat"])
    rf.save()

    # LSTM
    lstm = LSTMModel(seed=SEED)
    lstm.fit(split["X_train_seq"], split["y_train"], epochs=30, batch_size=64)
    lstm_preds = lstm.predict_grille(split["X_test_seq"])
    lstm.save()

    # Heuristique fréquence
    df_train = df.iloc[: split["split_idx"] + 10].reset_index(drop=True)
    freq_hot = FrequencyModel(mode="hot", seed=SEED)
    freq_hot.fit(df_train)
    freq_hot_preds = freq_hot.predict_grille(n_test)

    freq_cold = FrequencyModel(mode="cold", seed=SEED)
    freq_cold.fit(df_train)
    freq_cold_preds = freq_cold.predict_grille(n_test)

    # 4. Évaluation
    print("\n[4/4] Évaluation comparative")
    predictions = {
        "baseline":   baseline_preds,
        "rf":         rf_preds,
        "lstm":       lstm_preds,
        "freq_hot":   freq_hot_preds,
        "freq_cold":  freq_cold_preds,
    }
    run_evaluation(predictions, df_test)

    print("\n" + "=" * 60)
    print("  Pipeline terminé. Résultats dans outputs/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-analyse", action="store_true")
    args = parser.parse_args()
    main(skip_collect=args.skip_collect, skip_analyse=args.skip_analyse)
