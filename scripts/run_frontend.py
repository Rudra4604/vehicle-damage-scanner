"""Compatibility wrapper for running the Vehicle Damage Detection application.

DEPRECATION NOTICE:
Running a separate frontend server on port 3000 is no longer necessary.
FastAPI now serves both the REST API and the static frontend on port 8000.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_api import main as run_unified_server


def main():
    print("=" * 65)
    print(" [COMPATIBILITY NOTICE]")
    print(" A separate frontend server on port 3000 is deprecated.")
    print(" The complete application (Frontend + REST API) is now served")
    print(" on a single unified server at: http://127.0.0.1:8000/")
    print(" Forwarding to 'python scripts/run_api.py'...")
    print("=" * 65)
    run_unified_server()


if __name__ == "__main__":
    main()
