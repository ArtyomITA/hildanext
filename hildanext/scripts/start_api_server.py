"""Start the HildaNext FastAPI server on port 8080."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "backend" / "src"))

from hildanext.api import run_server  # noqa: E402

if __name__ == "__main__":
    config = str(root / "runs" / "configs" / "default.json")
    print(f"Starting API server — config={config}", flush=True)
    run_server(config, host="127.0.0.1", port=8080)
