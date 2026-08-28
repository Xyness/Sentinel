"""Where predictions go once they have been scored.

This used to be a 500-entry deque in the process, which meant restarting the
API threw away everything it had ever flagged and `sentinel export` could only
ever hand you the last few minutes. SQLite on a volume costs one file and
survives the restart.

With no PREDICTIONS_DB set it opens in memory and behaves exactly like the
deque did, which is what the tests want and what running the API on its own
without a volume gets you.
"""

import logging
import os
import sqlite3
import threading
from datetime import UTC, datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# The contract with FeatureAssembler.java, preprocess.py and the API schema.
FEATURE_COLUMNS = (
    "abs_return_max",
    "return_std",
    "price_range_rel",
    "volume_max_ratio",
    "volume_cv",
)

DB_PATH = os.environ.get("PREDICTIONS_DB", ":memory:")

# Roughly a week at three windows a minute. Past that the old rows are only
# making the file bigger.
MAX_ROWS = int(os.environ.get("PREDICTIONS_MAX_ROWS", "50000"))

# Pruning on every insert would mean a delete per prediction for one row's worth
# of benefit.
PRUNE_EVERY = 200

_COLUMNS = ("id", "timestamp", "symbol", *FEATURE_COLUMNS, "anomaly_score", "is_anomaly")


class PredictionStore:

    def __init__(self, path: str = DB_PATH, max_rows: int = MAX_ROWS):
        self.path = path
        self.max_rows = max_rows
        self._lock = threading.Lock()
        self._since_prune = 0

        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # One connection shared across the threadpool FastAPI runs sync
        # endpoints in, with a lock around every use of it.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create()
        logger.info(f"Prediction store at {path}, keeping {max_rows} rows")

    def _create(self):
        feature_ddl = ",\n".join(f"  {name} REAL NOT NULL" for name in FEATURE_COLUMNS)
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS predictions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT NOT NULL,
                  symbol TEXT NOT NULL,
                {feature_ddl},
                  anomaly_score REAL NOT NULL,
                  is_anomaly INTEGER NOT NULL
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS predictions_symbol ON predictions(symbol)"
            )
            self._conn.commit()

    # -- writing ------------------------------------------------------------

    def append(self, symbol: str, features: dict, score: float, is_anomaly: bool) -> dict:
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            **{name: float(features[name]) for name in FEATURE_COLUMNS},
            "anomaly_score": float(score),
            "is_anomaly": bool(is_anomaly),
        }
        placeholders = ", ".join(f":{name}" for name in _COLUMNS[1:])
        names = ", ".join(_COLUMNS[1:])

        with self._lock:
            cursor = self._conn.execute(
                f"INSERT INTO predictions ({names}) VALUES ({placeholders})",
                {**row, "is_anomaly": int(is_anomaly)},
            )
            self._conn.commit()
            row["id"] = cursor.lastrowid
            self._since_prune += 1
            if self._since_prune >= PRUNE_EVERY:
                self._prune()
        return row

    def _prune(self):
        """Caller holds the lock."""
        self._since_prune = 0
        self._conn.execute(
            "DELETE FROM predictions WHERE id <= "
            "(SELECT MAX(id) FROM predictions) - ?",
            (self.max_rows,),
        )
        self._conn.commit()

    # -- reading ------------------------------------------------------------

    def latest(self, limit: int = 100, symbol: str | None = None,
               after: int | None = None) -> list[dict]:
        """The most recent `limit` rows, oldest first.

        Oldest first because `sentinel feed` prints them in the order they come
        back and a feed that runs backwards is not a feed.
        """
        clauses, params = [], []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if after is not None:
            clauses.append("id > ?")
            params.append(after)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM predictions {where} ORDER BY id DESC LIMIT ?", params
            ).fetchall()
        return [_as_dict(row) for row in reversed(rows)]

    def recent(self, limit: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_as_dict(row) for row in reversed(rows)]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def close(self):
        with self._lock:
            self._conn.close()


def _as_dict(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["is_anomaly"] = bool(item["is_anomaly"])
    return item
