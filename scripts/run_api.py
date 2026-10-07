"""Local startup script for the Vehicle Damage Detection unified application.
Hosts both the web application and the FastAPI REST API on 127.0.0.1:8000.
"""
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from src.config import settings


def main():
    """Runs the FastAPI server locally on configured HOST:PORT with auto-reload."""
    host = settings.HOST
    port = settings.PORT
    print("=" * 60)
    print("Starting Vehicle Damage Detection Unified Application")
    print(f"Web Application: http://{host}:{port}/")
    print(f"Documentation:   http://{host}:{port}/docs")
    print(f"Health Check:    http://{host}:{port}/api/health")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
