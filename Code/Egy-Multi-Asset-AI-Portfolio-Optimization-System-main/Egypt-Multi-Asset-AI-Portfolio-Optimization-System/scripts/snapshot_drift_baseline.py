"""Write config/drift_baseline.json from the current completed return panel (last 504 rows)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.settings import PROJECT_ROOT
from src.data.loaders import load_market_panel
from src.governance.drift import fingerprint_recent_panel


def main() -> int:
    panel = load_market_panel()
    fp = fingerprint_recent_panel(panel.completed_returns, tail=504)
    out = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fp,
        "window": "last_504_rows_completed_returns",
    }
    dest = PROJECT_ROOT / "config" / "drift_baseline.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
