"""Command line entry point."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

from rich.text import Text

from . import __version__, render
from .client import DEFAULT_API, ApiError, Client

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ANOMALY = 2
EXIT_INTERRUPTED = 130

SYMBOLS = ("BTC-USDT", "ETH-USDT", "BNB-USDT")

# The same four shapes the manual test page offered, kept because they are the
# quickest way to find out where the decision boundary sits. The numbers are
# what a minute of the simulator actually produces: a quiet BTC minute moves
# about a tenth of a percent, and the injected spike is 5 to 15 %.
PRESETS = {
    "normal": {
        "abs_return_max": 0.0012, "return_std": 0.0008, "price_range_rel": 0.0035,
        "volume_max_ratio": 1.8, "volume_cv": 0.35,
    },
    "price-spike": {
        "abs_return_max": 0.0620, "return_std": 0.0110, "price_range_rel": 0.0680,
        "volume_max_ratio": 2.4, "volume_cv": 0.45,
    },
    "volume-spike": {
        "abs_return_max": 0.0015, "return_std": 0.0009, "price_range_rel": 0.0040,
        "volume_max_ratio": 9.5, "volume_cv": 2.10,
    },
    "flash-crash": {
        "abs_return_max": 0.1150, "return_std": 0.0210, "price_range_rel": 0.1240,
        "volume_max_ratio": 4.2, "volume_cv": 1.30,
    },
}

CSV_COLUMNS = (
    "id", "timestamp", "symbol", "abs_return_max", "return_std",
    "price_range_rel", "volume_max_ratio", "volume_cv",
    "anomaly_score", "is_anomaly",
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK

    console = render.build_console(plain=getattr(args, "plain", False))
    handler = {
        "status": cmd_status,
        "feed": cmd_feed,
        "stats": cmd_stats,
        "predict": cmd_predict,
        "export": cmd_export,
        "version": cmd_version,
    }[args.command]

    try:
        return handler(args, console)
    except KeyboardInterrupt:
        console.print("\ninterrupted", style="yellow")
        return EXIT_INTERRUPTED
    except ApiError as error:
        console.print(str(error), style="red")
        return EXIT_ERROR
    except ValueError as error:
        console.print(str(error), style="red")
        return EXIT_ERROR


# -- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--api", default=DEFAULT_API, metavar="URL",
                        help=f"where the API is (default: {DEFAULT_API})")
    common.add_argument("--plain", action="store_true", help="no colour, greppable")

    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Terminal client for the Sentinel anomaly detection pipeline.",
    )
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", parents=[common],
                            help="what is up, and what the model looks like")
    status.add_argument("-d", "--details", action="store_true",
                        help="also print the scaler the model was fitted with")
    status.add_argument("--json", action="store_true", help="raw payloads instead")

    feed = sub.add_parser("feed", parents=[common], help="follow predictions as they land")
    feed.add_argument("-s", "--symbol", metavar="PAIR", help="one pair only")
    feed.add_argument("--every", type=float, default=3.0, metavar="SECONDS",
                      help="how often to ask (default: 3)")
    feed.add_argument("--tail", type=int, default=10, metavar="N",
                      help="how much history to print before following (default: 10)")
    feed.add_argument("--anomalies", action="store_true", help="skip the normal ones")
    feed.add_argument("--once", action="store_true", help="print the tail and stop")
    feed.add_argument("--json", action="store_true", help="one JSON object per line")

    stats = sub.add_parser("stats", parents=[common], help="what has been scored, aggregated")
    stats.add_argument("-s", "--symbol", metavar="PAIR", help="one pair only")
    stats.add_argument("--depth", type=int, default=200, metavar="N",
                       help="how many predictions the trend is drawn from (default: 200)")
    stats.add_argument("--json", action="store_true", help="raw payload instead")

    predict = sub.add_parser("predict", parents=[common], help="score a feature vector by hand")
    predict.add_argument("-s", "--symbol", default=SYMBOLS[0], metavar="PAIR",
                         help=f"the pair to label it with (default: {SYMBOLS[0]})")
    predict.add_argument("-p", "--preset", choices=sorted(PRESETS),
                         help="start from a known shape, then override what you want")
    predict.add_argument("--max-return", type=float, dest="abs_return_max",
                         help="largest absolute log return in the window")
    predict.add_argument("--return-std", type=float, dest="return_std",
                         help="realised volatility over the window")
    predict.add_argument("--price-range", type=float, dest="price_range_rel",
                         help="(high - low) / mean price")
    predict.add_argument("--volume-peak", type=float, dest="volume_max_ratio",
                         help="largest trade over the window's mean volume")
    predict.add_argument("--volume-cv", type=float, dest="volume_cv",
                         help="volume deviation over mean volume")
    predict.add_argument("--stdin", action="store_true",
                         help="read one JSON vector per line instead, and score each")
    predict.add_argument("--fail-on-anomaly", action="store_true",
                         help="exit 2 when something is flagged")
    predict.add_argument("--json", action="store_true", help="raw result instead")

    export = sub.add_parser("export", parents=[common], help="the buffer as csv or json")
    export.add_argument("-f", "--format", choices=("csv", "json"), default="csv")
    export.add_argument("-o", "--output", type=Path, metavar="PATH",
                        help="write here instead of stdout")
    export.add_argument("-s", "--symbol", metavar="PAIR", help="one pair only")
    export.add_argument("--limit", type=int, default=500, metavar="N",
                        help="how many rows at most (default: 500)")
    export.add_argument("--anomalies", action="store_true", help="only the flagged ones")

    sub.add_parser("version", parents=[common], help="print the version")
    return parser


# -- commands --------------------------------------------------------------


def cmd_version(args, console) -> int:
    console.print(f"sentinel {__version__}")
    return EXIT_OK


def cmd_status(args, console) -> int:
    with Client(args.api) as client:
        # Every other command gives up when the API is down. This one reports
        # it, because an outage is the answer status was asked for.
        try:
            health = client.health()
        except ApiError as error:
            health, failure = None, str(error)
        else:
            failure = None

        services, model, checked_at = [], None, None
        if health is not None:
            payload = client.system_status()
            services = payload.get("services", [])
            checked_at = payload.get("timestamp")
            model = client.model_info()

        if args.json:
            console.print_json(data={"health": health, "system_status": services,
                                     "model": model})
            return EXIT_OK if health is not None else EXIT_ERROR

        render.wordmark(console)
        render.status_view(console, args.api, health, services, model, checked_at,
                           details=args.details)
        if failure:
            console.print(Text(failure, style="red"))
            return EXIT_ERROR
    return EXIT_OK


def cmd_feed(args, console) -> int:
    if not args.json:
        render.wordmark(console, clear=not args.once)
        console.print(Text(f"following {args.api}"
                           f"{'  ' + args.symbol if args.symbol else ''}",
                           style="bright_black"))
        console.print()

    seen, anomalies, last_id = 0, 0, None
    with Client(args.api) as client:
        try:
            while True:
                # `after` is a hint the API honours; the id filter here is what
                # actually guarantees no line is printed twice, since a restarted
                # API comes back with a buffer that starts over.
                first = last_id is None
                batch = client.latest_predictions(
                    limit=min(max(args.tail, 1), 500) if first else 500,
                    symbol=args.symbol,
                    after=last_id,
                )
                fresh = [item for item in batch
                         if last_id is None or (item.get("id") or 0) > last_id]
                if fresh:
                    last_id = max(item.get("id") or 0 for item in fresh)
                if first and args.tail <= 0:
                    fresh = []      # --tail 0 starts at the tip and replays nothing

                for item in fresh:
                    flagged = bool(item.get("is_anomaly"))
                    seen += 1
                    anomalies += flagged
                    if args.anomalies and not flagged:
                        continue
                    if args.json:
                        print(json.dumps(item), flush=True)
                    else:
                        console.print(render.feed_line(item))

                if args.once:
                    break
                time.sleep(max(args.every, 0.2))
        except KeyboardInterrupt:
            if not args.json:
                console.print()
                render.feed_tally(console, seen, anomalies)
            return EXIT_INTERRUPTED

    if args.once and not seen and not args.json:
        # Say which of the two empties this is. A filter that matched nothing is
        # not the same as a pipeline that has scored nothing.
        if args.symbol:
            console.print(Text(f"nothing scored for {args.symbol} yet", style="bright_black"))
        else:
            console.print(Text(
                "nothing scored yet. The scorer sends a window a minute once Spark has "
                "written one, so give it a minute, or push a vector in with "
                "`sentinel predict`.",
                style="bright_black",
            ))
    return EXIT_OK


def cmd_stats(args, console) -> int:
    with Client(args.api) as client:
        stats = client.stats()
        history = client.latest_predictions(limit=args.depth, symbol=args.symbol)

    if args.symbol:
        stats = _narrow(stats, args.symbol)

    if args.json:
        console.print_json(data=stats)
        return EXIT_OK

    render.wordmark(console)
    render.stats_view(console, stats, history)
    return EXIT_OK


def cmd_predict(args, console) -> int:
    if args.stdin:
        return _predict_stream(args, console)

    features = _features(args)
    with Client(args.api) as client:
        result = client.predict(features)

    if args.json:
        console.print_json(data=result)
    else:
        render.wordmark(console)
        render.prediction_view(console, features, result)

    if args.fail_on_anomaly and result.get("is_anomaly"):
        return EXIT_ANOMALY
    return EXIT_OK


def cmd_export(args, console) -> int:
    with Client(args.api) as client:
        rows = client.latest_predictions(limit=args.limit, symbol=args.symbol)

    if args.anomalies:
        rows = [row for row in rows if row.get("is_anomaly")]

    payload = _as_csv(rows) if args.format == "csv" else json.dumps(rows, indent=2)

    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        # soft_wrap so a long path stays on one line and survives being copied
        console.print(Text(f"{len(rows)} rows to {args.output}", style="bright_black"),
                      soft_wrap=True)
    else:
        # Straight to stdout rather than through the console, so redirecting it
        # gives a file with nothing wrapped and no escape codes in it.
        sys.stdout.write(payload if payload.endswith("\n") else payload + "\n")
    return EXIT_OK


# -- helpers ---------------------------------------------------------------


def _features(args, overrides: dict | None = None) -> dict:
    """A full vector out of a preset, the flags, and whatever came in on stdin.

    Order matters: the preset is a starting point, an explicit flag beats it,
    and a field from stdin beats both. Anything still missing is an error
    rather than a zero, because a silent zero is a feature value the model
    will happily score.
    """
    features = dict(PRESETS.get(args.preset, {}))
    for name, _ in render.FEATURES:
        value = getattr(args, name, None)
        if value is not None:
            features[name] = value
    features.update(overrides or {})

    missing = [name for name, _ in render.FEATURES if name not in features]
    if missing:
        raise ValueError(
            f"no value for {', '.join(missing)}. Pass --preset, or give every field."
        )
    return {"symbol": features.get("symbol", args.symbol),
            **{name: float(features[name]) for name, _ in render.FEATURES}}


def _predict_stream(args, console) -> int:
    flagged = False
    with Client(args.api) as client:
        for number, line in enumerate(sys.stdin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                incoming = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {number} is not JSON: {error}") from error
            if not isinstance(incoming, dict):
                raise ValueError(f"line {number} is not an object")

            features = _features(args, overrides=incoming)
            result = client.predict(features)
            flagged = flagged or bool(result.get("is_anomaly"))

            if args.json:
                print(json.dumps({**features, **result}), flush=True)
            else:
                console.print(render.feed_line({**features, **result}))

    if args.fail_on_anomaly and flagged:
        return EXIT_ANOMALY
    return EXIT_OK


def _narrow(stats: dict, symbol: str) -> dict:
    """Recompute the headline numbers for one pair.

    /stats answers for the whole buffer, and its per-symbol block already
    carries what a single pair needs, so filtering happens here rather than
    asking the API for an endpoint it does not have.
    """
    row = (stats.get("per_symbol") or {}).get(symbol)
    if row is None:
        raise ValueError(f"nothing scored for {symbol}")
    return {
        **stats,
        "total_predictions": row.get("count", 0),
        "total_anomalies": row.get("anomalies", 0),
        "anomaly_rate": row.get("anomaly_rate", 0.0),
        "avg_score": row.get("avg_score", 0.0),
        "per_symbol": {symbol: row},
        # The percentiles and the feature stats are computed across every pair,
        # so they would be a lie under a symbol heading.
        "score_percentiles": None,
        "feature_stats": None,
    }


def _as_csv(rows: list[dict]) -> str:
    """Fixed columns in a fixed order, so exports concatenate across runs."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return buffer.getvalue()


if __name__ == "__main__":
    sys.exit(main())
