"""Launcher script – avvia il backend server HildaNext."""
import sys
import os
import logging

# Force line-buffered output so logs appaiono subito nel terminale anche sotto conda run.
os.environ.setdefault("PYTHONUNBUFFERED", "1")

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "backend", "src"))
os.chdir(_here)

from hildanext.api import create_app, load_config  # noqa: E402
import uvicorn  # noqa: E402

# Silenzia il polling continuo di /run/status nei log di accesso uvicorn.
class _NoStatusPoll(logging.Filter):
    _SKIP = ('"/run/status', '"GET /run/status', '"/frontend/wsd', '"GET /frontend/wsd')
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._SKIP)

logging.getLogger("uvicorn.access").addFilter(_NoStatusPoll())

_CONFIG = "runs/configs/default.json"

print(f"[start_server] cwd={_here}", flush=True)
print(f"[start_server] config={_CONFIG}", flush=True)
print(f"[start_server] python={sys.executable}", flush=True)
try:
    import torch  # noqa: E402
    print(
        "[start_server] torch="
        f"{torch.__version__} cuda_build={torch.version.cuda} "
        f"cuda_available={bool(torch.cuda.is_available())} "
        f"gpu_count={int(torch.cuda.device_count())}",
        flush=True,
    )
except Exception as e:
    print(f"[start_server] torch_diag_error={e}", flush=True)
print("[start_server] model = LAZY (loaded only on first /generate call)", flush=True)

cfg = load_config(_CONFIG)
app = create_app(cfg, config_path=_CONFIG)

print("[start_server] starting uvicorn on http://127.0.0.1:8080  (no weights loaded yet)", flush=True)
uvicorn.run(
    app,
    host="127.0.0.1",
    port=8080,
    log_level="info",
    access_log=True,
    use_colors=True,
)
