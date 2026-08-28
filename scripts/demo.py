"""Regenerate docs/demo.svg, the terminal capture in the README.

The session is fabricated rather than captured. A real one needs the whole
compose stack up with a model already trained, so the numbers here are the ones
the commands really print, in the shapes the API really returns, without
anybody having to stand the pipeline up to refresh a picture.

    python3 scripts/demo.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# The renderer prints timestamps in the reader's own zone, which would move the
# capture depending on who regenerated it.
os.environ["TZ"] = "UTC"
if hasattr(time, "tzset"):
    time.tzset()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from rich.console import Console  # noqa: E402
from rich.text import Text  # noqa: E402

from sentinel import render  # noqa: E402

HEALTH = {"status": "ok", "model_loaded": True}

SERVICES = [
    {"name": "API", "status": "online", "response_time_ms": 0.0, "details": None},
    {"name": "Spark", "status": "online", "response_time_ms": 41.2, "details": None},
    {"name": "Kafka", "status": "online", "response_time_ms": 1.8, "details": None},
    {"name": "Zookeeper", "status": "online", "response_time_ms": 1.1, "details": None},
]

MODEL = {
    "loaded": True,
    "model_type": "IsolationForest",
    "n_estimators": 200,
    "contamination": 0.01,
    "max_samples": "auto",
    "model_file_size_kb": 428.5,
    "model_file_modified": "2026-08-27T09:41:02+00:00",
}

# symbol, score, flagged, the three z-scores
FEED = [
    ("BTC-USDT", 0.1442, False, (-1.57, 0.36, 0.82)),
    ("ETH-USDT", 0.1305, False, (0.04, -1.16, 1.32)),
    ("BNB-USDT", 0.0553, False, (1.09, -0.36, 0.15)),
    ("BTC-USDT", -0.1286, True, (4.50, 3.80, 1.50)),
    ("ETH-USDT", 0.1626, False, (0.26, 0.68, 0.85)),
    ("BTC-USDT", -0.0124, True, (-7.11, -0.51, 5.86)),
]


def feed_items() -> list[dict]:
    return [
        {
            "id": number,
            "timestamp": f"2026-08-27T10:14:{40 + number * 3:02d}+00:00",
            "symbol": symbol,
            "z_score_price": price,
            "z_score_log_return": log_return,
            "z_score_volume": volume,
            "rolling_price_std": 0.0031,
            "rolling_volume_std": 12.4,
            "anomaly_score": score,
            "is_anomaly": flagged,
        }
        for number, (symbol, score, flagged, (price, log_return, volume))
        in enumerate(FEED, start=1)
    ]


def main() -> None:
    console = Console(record=True, width=92, force_terminal=True, highlight=False)

    render.wordmark(console)
    console.print(Text("$ sentinel status", style="white"))
    console.print()
    render.status_view(console, "http://localhost:8000", HEALTH, SERVICES, MODEL,
                       "2026-08-27T10:14:58+00:00")

    console.print()
    console.print(Text("$ sentinel feed --tail 6", style="white"))
    console.print()
    console.print(Text("following http://localhost:8000", style="bright_black"))
    console.print()
    for item in feed_items():
        console.print(render.feed_line(item))

    destination = Path(__file__).resolve().parent.parent / "docs" / "demo.svg"
    console.save_svg(str(destination), title="sentinel status")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
