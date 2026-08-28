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
]

# Deliberately a Binance run, so the capture shows the shape of the scorecard
# without putting a precision and a recall in the README that no run produced.
# The real ones come off `sentinel status` once the stack has been up on
# simulated data, and they belong in prose where they can be dated.
MODEL = {
    "loaded": True,
    "model_type": "IsolationForest",
    "n_estimators": 200,
    "contamination": 0.05,
    "max_samples": "auto",
    "model_file_size_kb": 428.5,
    "model_file_modified": "2026-08-27T09:41:02+00:00",
    "metrics": {"n_train": 1840, "labelled": False, "holdout": None},
}

# symbol, score, flagged, then the three the feed prints: the largest absolute
# log return, the relative price range, and the biggest trade over the average
FEED = [
    ("BTC-USDT", 0.1442, False, (0.0012, 0.0035, 1.82)),
    ("ETH-USDT", 0.1305, False, (0.0009, 0.0028, 2.14)),
    ("BNB-USDT", 0.0553, False, (0.0021, 0.0044, 1.63)),
    ("BTC-USDT", -0.1286, True, (0.1150, 0.1240, 4.21)),
    ("ETH-USDT", 0.1626, False, (0.0014, 0.0031, 1.95)),
    ("BTC-USDT", -0.0124, True, (0.0018, 0.0041, 9.47)),
]


def feed_items() -> list[dict]:
    return [
        {
            "id": number,
            "timestamp": f"2026-08-27T10:14:{40 + number * 3:02d}+00:00",
            "symbol": symbol,
            "abs_return_max": abs_return,
            "price_range_rel": price_range,
            "volume_max_ratio": volume_peak,
            "return_std": 0.0008,
            "volume_cv": 0.37,
            "anomaly_score": score,
            "is_anomaly": flagged,
        }
        for number, (symbol, score, flagged, (abs_return, price_range, volume_peak))
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
