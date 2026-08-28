"""Drive a running stack until it produces a prediction, or fail loudly.

CI used to build the images and stop there, which proves the Dockerfiles parse
and nothing else. Everything this project claims happens between the services:
the generator has to reach Kafka, Spark has to close a window and write it,
training has to find the files, the API has to load what training wrote, and
the scorer has to put a row back through it. A build cannot tell you any of
that went right.

Run it against a stack that is already up:

    python scripts/smoke_test.py
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

API = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT = float(os.environ.get("SMOKE_TIMEOUT_SECONDS", "600"))
POLL = float(os.environ.get("SMOKE_POLL_SECONDS", "5"))

# The contract, end to end: Spark writes these, training fits on them, the API
# serves them under these names. If the three ever drift apart this is where it
# shows, instead of at inference time as a confusing 422.
EXPECTED_FEATURES = [
    "abs_return_max",
    "return_std",
    "price_range_rel",
    "volume_max_ratio",
    "volume_cv",
]


def get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=15) as response:
        return json.loads(response.read())


def wait_for(what, check, deadline):
    """Poll until `check` returns something truthy, or run out of time."""
    last = None
    while time.time() < deadline:
        try:
            result = check()
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            last = f"{type(error).__name__}: {error}"
        else:
            if result:
                left = int(deadline - time.time())
                print(f"ok   {what} ({left}s to spare)", flush=True)
                return result
            last = "not yet"
        print(f"..   {what}: {last}", flush=True)
        time.sleep(POLL)

    print(f"FAIL {what}: gave up after {TIMEOUT:.0f}s, last was {last}", file=sys.stderr)
    sys.exit(1)


def main():
    deadline = time.time() + TIMEOUT
    print(f"smoke test against {API}, {TIMEOUT:.0f}s budget", flush=True)

    wait_for("the API answers", lambda: get("/health") is not None, deadline)
    wait_for("a model is loaded", lambda: get("/health")["model_loaded"], deadline)

    info = get("/model-info")
    if info.get("feature_names") != EXPECTED_FEATURES:
        print(f"FAIL the model was fitted on {info.get('feature_names')}, "
              f"expected {EXPECTED_FEATURES}", file=sys.stderr)
        sys.exit(1)
    print(f"ok   fitted on {info['feature_names']}", flush=True)

    metrics = info.get("metrics") or {}
    print(f"     trained {metrics.get('trained_at')} on {metrics.get('n_train')} rows "
          f"at contamination {metrics.get('contamination')}", flush=True)
    if metrics.get("holdout"):
        print(f"     holdout {json.dumps(metrics['holdout'])}", flush=True)

    predictions = wait_for(
        "the scorer put a row through the model",
        lambda: get("/latest-predictions?limit=5") or None,
        deadline,
    )

    first = predictions[0]
    missing = [name for name in EXPECTED_FEATURES if name not in first]
    if missing:
        print(f"FAIL a stored prediction is missing {missing}", file=sys.stderr)
        sys.exit(1)

    stats = get("/stats")
    print(f"ok   {stats['total_predictions']} scored, "
          f"{stats['total_anomalies']} flagged, "
          f"{stats['anomaly_rate']} % anomaly rate", flush=True)
    print(f"     per symbol: {json.dumps(stats.get('per_symbol', {}))}", flush=True)

    services = {s["name"]: s["status"] for s in get("/system-status")["services"]}
    print(f"ok   services: {services}", flush=True)
    if services.get("Kafka") != "online":
        print(f"FAIL Kafka reported {services.get('Kafka')}", file=sys.stderr)
        sys.exit(1)
    if services.get("Spark") != "online":
        # Not a failure, and not worth making one. The Spark UI is served off
        # the driver thread, so a probe that lands during a micro-batch times
        # out on a job that is running perfectly well. A prediction came out
        # the far end, which is the stronger evidence and it is already in.
        print("     (the Spark probe missed, which it does under load. Rows came "
              "through, so the job is running.)", flush=True)

    print("smoke test passed", flush=True)


if __name__ == "__main__":
    main()
