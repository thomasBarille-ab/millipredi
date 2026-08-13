"""
Collecte des tirages EuroMillions.

Stratégie (par ordre de priorité) :
  1. mes-resultats-fdj.fr — CSV complet 2004-aujourd'hui (1971+ tirages)
  2. FDJ open data — ZIP par période (fallback)
  3. Fichier CSV local fourni manuellement → data/raw/euromillions_manual.csv

Format attendu en sortie : date, n1, n2, n3, n4, n5, etoile1, etoile2
"""

import sys
import io
import zipfile
from pathlib import Path

import requests
import pandas as pd

# Format actuel : 5 numéros (1-50) + 2 étoiles (1-12), en vigueur depuis le 24/09/2016
FORMAT_DATE = pd.Timestamp("2016-09-24")

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "euromillions.csv"

# Source principale : CSV complet fusionné (2004 → aujourd'hui)
MRF_URL = "https://www.mes-resultats-fdj.fr/api/telecharger/euromillions"

# Sources FDJ par période (fallback) — format ZIP contenant un CSV
FDJ_ZIP_URLS = [
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_202002.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201903.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201609.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201402.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201105.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_200402.zip",
]

MANUAL_CSV = RAW_DIR / "euromillions_manual.csv"

EXPECTED_COLS = ["date", "n1", "n2", "n3", "n4", "n5", "etoile1", "etoile2"]

COL_MAP = {
    "date_de_tirage": "date",
    "boule_1": "n1",
    "boule_2": "n2",
    "boule_3": "n3",
    "boule_4": "n4",
    "boule_5": "n5",
    "etoile_1": "etoile1",
    "etoile_2": "etoile2",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes vers le format cible et nettoie."""
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {k: v for k, v in COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing} (colonnes dispo : {list(df.columns)})")

    df = df[EXPECTED_COLS].copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    for col in EXPECTED_COLS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _download_mrf() -> pd.DataFrame | None:
    """Télécharge le CSV complet depuis mes-resultats-fdj.fr."""
    try:
        print(f"[collect] Téléchargement depuis mes-resultats-fdj.fr...")
        r = requests.get(MRF_URL, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8")), sep=";", low_memory=False)
        return _normalize(df)
    except Exception as e:
        print(f"[collect] mes-resultats-fdj.fr échoué : {e}")
        return None


def _download_fdj_zips() -> pd.DataFrame | None:
    """Télécharge et concatène les ZIP FDJ par période."""
    frames = []
    for url in FDJ_ZIP_URLS:
        try:
            print(f"[collect] ZIP FDJ : {url}")
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    continue
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f, sep=";", encoding="utf-8", low_memory=False)
                    frames.append(_normalize(df))
        except Exception as e:
            print(f"[collect] Échec {url} : {e}")

    if not frames:
        return None
    combined = pd.concat(frames).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return combined


def _load_manual() -> pd.DataFrame:
    """Charge un CSV fourni manuellement."""
    if not MANUAL_CSV.exists():
        raise FileNotFoundError(
            f"Aucune source disponible.\n"
            f"Placez un fichier CSV dans : {MANUAL_CSV}\n"
            f"Colonnes attendues : {EXPECTED_COLS}"
        )
    print(f"[collect] Chargement du fichier manuel : {MANUAL_CSV}")
    df = pd.read_csv(MANUAL_CSV)
    return _normalize(df)


def _validate(df: pd.DataFrame) -> None:
    assert set(EXPECTED_COLS).issubset(df.columns), "Colonnes manquantes"
    assert (df[["n1","n2","n3","n4","n5"]].min().min() >= 1), "Numéro < 1"
    assert (df[["n1","n2","n3","n4","n5"]].max().max() <= 50), "Numéro > 50"
    assert (df[["etoile1","etoile2"]].min().min() >= 1), "Étoile < 1"
    assert (df[["etoile1","etoile2"]].max().max() <= 12), "Étoile > 12"
    print(f"[collect] Validation OK — {len(df)} tirages du {df['date'].min().date()} au {df['date'].max().date()}")


def collect(force: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        print(f"[collect] Données existantes trouvées : {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH, parse_dates=["date"])
        _validate(df)
        return df

    # 1. mes-resultats-fdj.fr (CSV complet)
    df = _download_mrf()

    if df is None:
        # 2. FDJ ZIP par période
        df = _download_fdj_zips()

    if df is None:
        # 3. Fallback manuel
        df = _load_manual()

    df = df[df["date"] >= FORMAT_DATE].reset_index(drop=True)
    print(f"[collect] Filtre post-2016 appliqué — {len(df)} tirages conservés.")
    _validate(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[collect] Sauvegardé : {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    force = "--force" in sys.argv
    collect(force=force)
