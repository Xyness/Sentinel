"""Terminal rendering. Everything the client prints goes through here."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

from rich.box import ASCII, ROUNDED, SIMPLE
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __author__, __version__

WORDMARK = """█▀▀ █▀▀ █▄ █ ▀█▀ █ █▄ █ █▀▀ █
▄██ ██▄ █ ▀█  █  █ █ ▀█ ██▄ █▄▄"""

ACCENT = "bright_cyan"

STATUS_STYLE = {
    "online": "green",
    "degraded": "yellow",
    "offline": "red",
    "unchecked": "bright_black",
}

STATUS_MARK = {"online": "+", "degraded": "~", "offline": "!", "unchecked": "?"}

# The path a prediction takes, in order. Training is not on it: it produces the
# model file, which the block under the pipeline reports on instead.
PIPELINE = ("generator", "kafka", "spark", "scorer", "api")

# The five the model is fitted on, in the order the API sends them, with the
# labels that fit a narrow terminal. All dimensionless, all with a floor of 0.
FEATURES = (
    ("abs_return_max", "max return"),
    ("return_std", "volatility"),
    ("price_range_rel", "price range"),
    ("volume_max_ratio", "volume peak"),
    ("volume_cv", "volume cv"),
)


def _encodes(glyphs: str) -> bool:
    """Whether this console can print these characters at all.

    An older Windows console is still cp850 or cp1252, where the blocks and the
    box drawing come out as question marks or as nothing. UTF-8 is the default
    when there is no encoding to ask, which is what a pipe looks like.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        glyphs.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


FANCY = _encodes("█░▀▄╭▁▇")

BAR_FULL = "█" if FANCY else "#"
BAR_EMPTY = "░" if FANCY else "."
SPARK = "▁▂▃▄▅▆▇█" if FANCY else "._-=+*#%"
PANEL_BOX = ROUNDED if FANCY else ASCII
LINE_BOX = SIMPLE if FANCY else ASCII
ARROW = "->"


def build_console(plain: bool = False) -> Console:
    """One console for the process. NO_COLOR and pipes are honoured."""
    return Console(
        no_color=plain or bool(os.environ.get("NO_COLOR")),
        highlight=False,
    )


def wordmark(console: Console, clear: bool = False) -> None:
    """Draw the banner, on a real terminal only.

    Only the feed clears first, because it takes the screen over. The one-shot
    commands print above whatever was already there, since wiping somebody's
    scrollback to show them a service table is not a fair trade.
    """
    if not console.is_terminal:
        return

    if clear:
        console.clear()
    console.print()
    if FANCY:
        console.print(Text(WORDMARK, style=ACCENT))
    else:
        console.print(Text("Sentinel", style=f"bold {ACCENT}"))
    console.print(Text(f"  streaming anomaly detection  v{__version__}", style="bright_black"))

    byline = Text("  developed by ", style="bright_black")
    byline.append(__author__, style=ACCENT)
    console.print(byline)
    console.print()


# -- small pieces ----------------------------------------------------------


def rate_bar(rate: float, width: int = 16) -> Text:
    """An anomaly rate as a bar. The model is trained at 1% contamination, so
    a couple of percent is the shape it was asked for and anything past ten is
    either a market doing something or a detector that needs retraining."""
    style = "green" if rate <= 2 else "yellow" if rate <= 10 else "red"
    filled = min(width, round(width * rate / 100))
    bar = Text(BAR_FULL * filled, style=style)
    bar.append(BAR_EMPTY * (width - filled), style="bright_black")
    bar.append(f"  {rate:.2f} %", style=style)
    return bar


def sparkline(values: list[float], flags: list[bool] | None = None, width: int = 60) -> Text:
    """The last `width` scores as one row of blocks, anomalies in red.

    The ramp is scaled to the window rather than to a fixed range, because
    decision_function has no declared bounds. A window where every score is
    identical draws flat at the bottom instead of halfway up, since a detector
    that has not moved should not look like one that has.
    """
    if not values:
        return Text("no scores yet", style="bright_black")

    window = values[-width:]
    marks = (flags or [False] * len(values))[-width:]
    low, high = min(window), max(window)
    span = high - low

    line = Text()
    for value, flagged in zip(window, marks, strict=False):
        level = 0 if span == 0 else round((value - low) / span * (len(SPARK) - 1))
        line.append(SPARK[level], style="red" if flagged else ACCENT)
    return line


def verdict(is_anomaly: bool) -> Text:
    return Text("ANOMALY", style="bold red") if is_anomaly else Text("normal", style="green")


def _short(text: str | None, limit: int = 80) -> str:
    if not text:
        return ""
    first = text.splitlines()[0]
    return first if len(first) <= limit else first[: limit - 1] + "…"


def _local(stamp: str | None) -> str:
    """An ISO timestamp from the API in the reader's own clock."""
    if not stamp:
        return ""
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return stamp


def _panel(grid: Table) -> Panel:
    return Panel(grid, title=Text("SENTINEL", style=f"bold {ACCENT}"), title_align="left",
                 border_style="bright_black", box=PANEL_BOX, padding=(0, 1), expand=False)


def _grid(label_width: int = 10) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bright_black", width=label_width)
    grid.add_column()
    return grid


# -- status ----------------------------------------------------------------


def status_panel(base_url: str, health: dict | None, services: list[dict],
                 checked_at: str | None) -> Panel:
    grid = _grid()
    grid.add_row("api", Text(base_url, style=f"bold {ACCENT}"))

    if health is None:
        grid.add_row("model", Text("unknown, the API did not answer", style="red"))
    elif health.get("model_loaded"):
        grid.add_row("model", Text("loaded", style="green"))
    else:
        # Not an error on a cold start: training runs for a few minutes and the
        # API is deliberately up before it finishes.
        grid.add_row("model", Text("not loaded yet, training may still be running",
                                   style="yellow"))

    tally = Text()
    for state in ("online", "degraded", "offline"):
        count = sum(1 for s in services if s.get("status") == state)
        if count:
            tally.append(f"{count} {state}  ", style=STATUS_STYLE[state])
    grid.add_row("services", tally or Text("none reported", style="bright_black"))

    if checked_at:
        grid.add_row("checked", Text(_local(checked_at), style="bright_black"))
    return _panel(grid)


def service_table(services: list[dict]) -> Table:
    table = Table(box=LINE_BOX, header_style="bright_black", padding=(0, 1))
    table.add_column("service", style="white")
    table.add_column("status")
    table.add_column("latency", justify="right", style="bright_black")
    table.add_column("note", style="bright_black", overflow="fold")

    for service in services:
        state = service.get("status", "offline")
        style = STATUS_STYLE.get(state, "red")
        table.add_row(
            service.get("name", "?"),
            Text(f"{STATUS_MARK.get(state, '!')} {state}", style=style),
            f"{service.get('response_time_ms', 0):.0f} ms",
            _short(service.get("details")),
        )
    return table


def pipeline_line(services: list[dict]) -> Text:
    """The flow, with a mark on each stage the API can actually reach.

    The generator and the scorer answer on nothing, so they are drawn unchecked
    rather than green. A stage nobody looked at is not a stage that is up, and
    this line is the first thing read when something is wrong.
    """
    known = {s.get("name", "").lower(): s.get("status", "offline") for s in services}

    line = Text("  ")
    for index, stage in enumerate(PIPELINE):
        state = known.get(stage, "unchecked")
        style = STATUS_STYLE.get(state, "red")
        line.append(STATUS_MARK.get(state, "?"), style=style)
        line.append(f" {stage}", style="white" if state == "online" else "bright_black")
        if index < len(PIPELINE) - 1:
            line.append(f"  {ARROW}  ", style="bright_black")
    return line


def model_table(info: dict, details: bool = False) -> Table:
    grid = _grid(label_width=13)
    grid.add_row("type", Text(str(info.get("model_type", "unknown")), style="white"))
    grid.add_row("estimators", Text(str(info.get("n_estimators", "unknown")), style="white"))
    grid.add_row("contamination", Text(f"{info.get('contamination', 0):.3f}", style="white"))
    grid.add_row("max samples", Text(str(info.get("max_samples", "auto")), style="white"))

    size = info.get("model_file_size_kb")
    modified = _local(info.get("model_file_modified"))
    if size is not None:
        stamp = Text(f"{size:.1f} KB", style="white")
        if modified:
            stamp.append(f"   trained {modified}", style="bright_black")
        grid.add_row("file", stamp)

    # What the training run measured on rows it had not fitted on. The model
    # carries it, so the client can print it without going near the training
    # logs, and a number nobody can find is a number nobody trusts.
    metrics = info.get("metrics") or {}
    if metrics.get("n_train"):
        fitted = Text(f"{metrics['n_train']:,} rows", style="white")
        if metrics.get("label_rate") is not None:
            fitted.append(f"  {metrics['label_rate'] * 100:.1f} % labelled anomalous",
                          style="bright_black")
        grid.add_row("fitted on", fitted)

    holdout = metrics.get("holdout")
    if holdout:
        grid.add_row("holdout", Text(
            f"precision {holdout['precision']:.3f}   "
            f"recall {holdout['recall']:.3f}   "
            f"f1 {holdout['f1']:.3f}", style="white"))
        grid.add_row("", Text(f"over {holdout['support']} labelled anomalies",
                              style="bright_black"))
    elif metrics.get("labelled") is False:
        grid.add_row("holdout", Text("unlabelled data, nothing to score against",
                                     style="bright_black"))

    if details and info.get("scaler_means"):
        names = [label for _, label in FEATURES]
        for name, mean, deviation in zip(names, info["scaler_means"],
                                         info.get("scaler_stds") or [], strict=False):
            grid.add_row(name, Text(f"mean {mean:>12.6f}   std {deviation:>12.6f}",
                                    style="bright_black"))
    return grid


def status_view(console: Console, base_url: str, health: dict | None, services: list[dict],
                model: dict | None, checked_at: str | None, details: bool = False) -> None:
    console.print(status_panel(base_url, health, services, checked_at))
    console.print()

    if services:
        console.print(service_table(services))
        console.print()
        console.print(pipeline_line(services))
        console.print(Text("  the generator and the scorer expose nothing to ask, so they "
                           "are not checked", style="bright_black"))
        console.print()

    if model is None:
        return
    if model.get("loaded"):
        console.print(Text("  model", style="bright_black"))
        console.print(Padding(model_table(model, details=details), (0, 0, 0, 2)))
    else:
        console.print(Text("no model loaded, so nothing is being scored", style="yellow"))


# -- stats -----------------------------------------------------------------


def stats_panel(stats: dict) -> Panel:
    total = stats.get("total_predictions", 0)
    grid = _grid()
    grid.add_row("scored", Text(f"{total:,}", style=f"bold {ACCENT}"))

    if total:
        anomalies = Text(f"{stats.get('total_anomalies', 0):,}   ", style="white")
        anomalies.append_text(rate_bar(stats.get("anomaly_rate", 0.0)))
        grid.add_row("anomalies", anomalies)
        grid.add_row("avg score", Text(f"{stats.get('avg_score', 0.0):+.6f}", style="white"))
    else:
        grid.add_row("anomalies", Text("nothing scored yet", style="bright_black"))
    return _panel(grid)


def symbol_table(per_symbol: dict) -> Table:
    table = Table(box=LINE_BOX, header_style="bright_black", padding=(0, 1))
    table.add_column("symbol", style="white")
    table.add_column("scored", justify="right", style="bright_black")
    table.add_column("anomalies", justify="right")
    table.add_column("rate")
    table.add_column("avg score", justify="right", style="bright_black")

    for symbol, row in sorted(per_symbol.items(),
                              key=lambda item: item[1].get("anomaly_rate", 0), reverse=True):
        count = row.get("anomalies", 0)
        table.add_row(
            symbol,
            f"{row.get('count', 0):,}",
            Text(str(count), style="red" if count else "bright_black"),
            rate_bar(row.get("anomaly_rate", 0.0), width=12),
            f"{row.get('avg_score', 0.0):+.6f}",
        )
    return table


def percentile_line(percentiles: dict) -> Text:
    """Where the score sits across the buffer.

    The p25 to p99 spread says more about the boundary than the mean does: the
    model flags the low tail, so a p25 close to zero means it is barely
    separating anything.
    """
    line = Text("  ")
    for key in ("p25", "p50", "p75", "p95", "p99"):
        line.append(f"{key} ", style="bright_black")
        line.append(f"{percentiles.get(key, 0.0):+.4f}   ", style="white")
    return line


def feature_table(feature_stats: dict) -> Table:
    table = Table(box=LINE_BOX, header_style="bright_black", padding=(0, 1))
    table.add_column("feature", style="white")
    for column in ("mean", "std", "min", "max"):
        table.add_column(column, justify="right", style="bright_black")

    labels = dict(FEATURES)
    for name, row in feature_stats.items():
        table.add_row(
            labels.get(name, name),
            f"{row.get('mean', 0.0):+.6f}",
            f"{row.get('std', 0.0):.6f}",
            f"{row.get('min', 0.0):+.6f}",
            f"{row.get('max', 0.0):+.6f}",
        )
    return table


def stats_view(console: Console, stats: dict, history: list[dict]) -> None:
    console.print(stats_panel(stats))
    console.print()

    if not stats.get("total_predictions"):
        console.print(Text(
            "nothing scored yet: the scorer feeds this once Spark has written a window",
            style="bright_black",
        ))
        return

    if stats.get("per_symbol"):
        console.print(symbol_table(stats["per_symbol"]))
        console.print()

    if history:
        console.print(Text("  score trend", style="bright_black"))
        console.print(Text("  ") + sparkline(
            [item.get("anomaly_score", 0.0) for item in history],
            [bool(item.get("is_anomaly")) for item in history],
        ))
        console.print()

    if stats.get("score_percentiles"):
        console.print(Text("  score distribution", style="bright_black"))
        console.print(percentile_line(stats["score_percentiles"]))
        console.print()

    if stats.get("feature_stats"):
        console.print(feature_table(stats["feature_stats"]))


# -- feed ------------------------------------------------------------------


def feed_line(item: dict) -> Text:
    """One prediction, one line, columns wide enough to stay aligned so the
    output survives being piped through awk or grep."""
    stamp = item.get("timestamp", "")
    try:
        clock = datetime.fromisoformat(stamp).astimezone().strftime("%H:%M:%S")
    except (TypeError, ValueError):
        clock = datetime.now(UTC).astimezone().strftime("%H:%M:%S")

    flagged = bool(item.get("is_anomaly"))
    line = Text(f"{clock}  ", style="bright_black")
    line.append(f"{item.get('symbol', '?'):<10}", style="white")
    line.append(f"{item.get('anomaly_score', 0.0):+9.4f}  ",
                style="bold red" if flagged else "white")
    line.append(f"{'anomaly' if flagged else 'normal':<8}",
                style="bold red" if flagged else "green")
    # The three that carry most of a verdict. The other two are in `predict`
    # and in `stats`, where there is room for all five.
    line.append("  ret ", style="bright_black")
    line.append(
        f"{item.get('abs_return_max', 0.0):7.4f} "
        f"{item.get('price_range_rel', 0.0):7.4f} "
        f"{item.get('volume_max_ratio', 0.0):6.2f}",
        style="bright_black",
    )
    return line


def feed_tally(console: Console, seen: int, anomalies: int) -> None:
    if not seen:
        console.print(Text("nothing came through", style="bright_black"))
        return
    rate = anomalies / seen * 100
    console.print(Text(f"{seen} scored, {anomalies} flagged, {rate:.2f} %",
                       style="bright_black"))


# -- predict ---------------------------------------------------------------


def prediction_view(console: Console, features: dict, result: dict) -> None:
    grid = _grid(label_width=11)
    grid.add_row("symbol", Text(str(result.get("symbol", features.get("symbol", "?"))),
                                style=f"bold {ACCENT}"))
    grid.add_row("verdict", verdict(bool(result.get("is_anomaly"))))

    # No bar behind the score. decision_function has no declared range, so any
    # gauge would be inventing a scale; the boundary is the one number that is
    # actually defined, and it is the one worth printing.
    score = Text(f"{result.get('anomaly_score', 0.0):+.6f}", style="white")
    score.append("   below 0 is an anomaly", style="bright_black")
    grid.add_row("score", score)

    for field, label in FEATURES:
        grid.add_row(label, Text(f"{features.get(field, 0.0):+.6f}", style="bright_black"))

    console.print(_panel(grid))
