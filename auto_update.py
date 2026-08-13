"""
Runner quotidien — à lancer via le Planificateur de tâches Windows.

  python auto_update.py          → suit le calendrier automatiquement
  python auto_update.py --force  → force les deux actions (predict + fetch)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scheduler import run_daily

if __name__ == "__main__":
    force = "--force" in sys.argv
    result = run_daily(force=force)
    print(result)
