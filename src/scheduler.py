"""
Logique de scheduling EuroMillions.

Tirages : mardi et vendredi ~21h CET
  - Veille du tirage (lundi / jeudi)  → générer les prédictions
  - Lendemain du tirage (mercredi / samedi) → récupérer les vrais résultats
"""

from datetime import date, timedelta
from pathlib import Path
import pandas as pd

# 0=lun 1=mar 2=mer 3=jeu 4=ven 5=sam 6=dim
DRAW_WEEKDAYS   = {1, 4}   # mardi, vendredi
PRED_WEEKDAYS   = {0, 3}   # lundi, jeudi  (veille)
FETCH_WEEKDAYS  = {2, 5}   # mercredi, samedi (lendemain)

LOG_PATH  = Path(__file__).parent.parent / "outputs" / "results" / "scheduler_log.csv"


def get_next_draw_date(from_date: date | None = None) -> date:
    d = from_date or date.today()
    for i in range(1, 8):
        candidate = d + timedelta(days=i)
        if candidate.weekday() in DRAW_WEEKDAYS:
            return candidate
    raise RuntimeError("Impossible de trouver le prochain tirage")


def get_prev_draw_date(from_date: date | None = None) -> date:
    d = from_date or date.today()
    for i in range(1, 8):
        candidate = d - timedelta(days=i)
        if candidate.weekday() in DRAW_WEEKDAYS:
            return candidate
    raise RuntimeError("Impossible de trouver le tirage précédent")


def should_generate(today: date | None = None) -> bool:
    """Vrai si aujourd'hui est la veille d'un tirage (lun ou jeu)."""
    return (today or date.today()).weekday() in PRED_WEEKDAYS


def should_fetch(today: date | None = None) -> bool:
    """Vrai si hier était un tirage (mer ou sam)."""
    return (today or date.today()).weekday() in FETCH_WEEKDAYS


def _log(action: str, detail: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = pd.DataFrame([{
        "date": date.today().isoformat(),
        "action": action,
        "detail": detail,
    }])
    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        pd.concat([existing, entry], ignore_index=True).to_csv(LOG_PATH, index=False)
    else:
        entry.to_csv(LOG_PATH, index=False)


def run_daily(today: date | None = None, force: bool = False) -> str:
    """
    Point d'entrée principal — à appeler chaque jour.
    Retourne un message décrivant l'action effectuée.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from collect import collect
    from predict_next import generate_predictions

    today = today or date.today()
    day_name = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"][today.weekday()]
    print(f"[scheduler] {today} ({day_name})")

    if should_generate(today) or force:
        # Veille du tirage → générer les prédictions
        next_draw = get_next_draw_date(today)
        print(f"[scheduler] Génération des prédictions pour le tirage du {next_draw}")
        generate_predictions()
        msg = f"Prédictions générées pour le tirage du {next_draw}"
        _log("predict", msg)
        return msg

    if should_fetch(today) or force:
        # Lendemain du tirage → mettre à jour les données
        prev_draw = get_prev_draw_date(today)
        print(f"[scheduler] Récupération des résultats du tirage du {prev_draw}")
        df = collect(force=True)
        latest = df["date"].max().date()
        if latest >= prev_draw:
            msg = f"Résultats du {prev_draw} récupérés (dernier tirage connu : {latest})"
        else:
            msg = f"Résultats du {prev_draw} pas encore disponibles (dernier : {latest})"
        _log("fetch", msg)
        return msg

    msg = f"Rien à faire aujourd'hui ({day_name}) — prochain tirage : {get_next_draw_date(today)}"
    print(f"[scheduler] {msg}")
    _log("idle", msg)
    return msg
