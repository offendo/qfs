"""On-disk record of every polynomial whose height was computed.

A single SQLite file (Python's built-in sqlite3, no server). Tables are append-only with no
indexes, and each oracle batch is written in one transaction: about 700,000 rows per second on a
laptop, far more than the height workers can produce. Several runs can share one file at the same
time (WAL mode).

Table `heights`, one row per height computation:
    p, n, d     the setting
    coeffs      coefficients as bytes (uint8, monomial order of rlmath.polynomials.monomials),
                scaled so the first nonzero coefficient is 1
    cutoff      cutoff used for the height computation
    status      "ok" | "inf" | "past_cutoff" | "invalid" | "error"
    height      the height when status is "ok", else NULL
    created     unix time

Table `smoothness`: (p, n, d, coeffs, smooth) for the polynomials that were checked, which is
those scoring at least 2.

Within one run each polynomial appears once (the oracle caches results); separate runs sharing a
file can repeat polynomials. Reading it back:

    import sqlite3, numpy as np
    db = sqlite3.connect("runs/heights.sqlite")
    q = "SELECT DISTINCT coeffs, height FROM heights WHERE p = 5 AND height >= 8"
    for coeffs, h in db.execute(q):
        print(np.frombuffer(coeffs, dtype=np.uint8), h)

For many lookups by polynomial, add an index once (it slows writes, so drop it before training):

    CREATE INDEX heights_by_coeffs ON heights (p, n, d, coeffs);
"""

from pathlib import Path
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS heights (
    p INTEGER NOT NULL,
    n INTEGER NOT NULL,
    d INTEGER NOT NULL,
    coeffs BLOB NOT NULL,
    cutoff INTEGER NOT NULL,
    status TEXT NOT NULL,
    height INTEGER,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS smoothness (
    p INTEGER NOT NULL,
    n INTEGER NOT NULL,
    d INTEGER NOT NULL,
    coeffs BLOB NOT NULL,
    smooth INTEGER NOT NULL
);
"""


class HeightStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")  # durable at each checkpoint, much faster than FULL
        self.db.executescript(SCHEMA)

    def add_heights(self, p: int, n: int, d: int, cutoff: int, rows: list[tuple[bytes, str, int | None]]):
        """rows: (coeffs key, status, height)."""
        now = time.time()
        with self.db:
            self.db.executemany(
                "INSERT INTO heights (p, n, d, coeffs, cutoff, status, height, created) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(p, n, d, k, cutoff, status, height, now) for k, status, height in rows],
            )

    def set_smooth(self, p: int, n: int, d: int, rows: list[tuple[bytes, bool]]):
        with self.db:
            self.db.executemany(
                "INSERT INTO smoothness (p, n, d, coeffs, smooth) VALUES (?, ?, ?, ?, ?)",
                [(p, n, d, k, int(s)) for k, s in rows],
            )

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM heights").fetchone()[0]

    def close(self):
        self.db.close()
