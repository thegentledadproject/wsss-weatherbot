"""
db/ledger.py — SQLite state store
Tables:
  calibration_logs  : model mu vs actual settled temp (residuals)
  open_positions    : live bracket entries, keyed by token_id
  signal_log        : every edge scan result, win/loss after settle
  token_matrix      : today's bracket → token_id mapping (refreshed daily)
  scan_snapshots    : append-only inputs of each Job 2 scan (forecast, bias, obs, probs)
  book_snapshots    : append-only full order books seen at scan and at execution
"""

import json
import sqlite3
import datetime
import logging
import contextlib
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.ledger")


class Ledger:
    def __init__(self, db_path: str = "hermes.db"):
        self.db_path = db_path
        self._init_schema()
        self._migrate_schema()

    @contextlib.contextmanager
    def _conn(self):
        """
        Yields a connection, commits on clean exit, closes always.

        BUG FIX: sqlite3.Connection used as `with conn:` only wraps a
        transaction (commit on success, rollback on exception) — it does
        NOT close the connection. Every method in this file previously did
        `conn = sqlite3.connect(...); with conn: ...` and never closed,
        leaking one file descriptor per call. Confirmed: 200 calls leaked
        46+ FDs relying only on GC to reclaim them. For a bot running 24/7
        with Job 2 every 15 min and Job 5 every 5 min across weeks, this
        risks hitting the OS file-descriptor limit and crashing. This
        wrapper preserves the exact same call-site syntax everywhere
        (`with self._conn() as conn:`) while guaranteeing close() in a
        finally block, so no other method in this file needed to change.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS calibration_logs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    market_date     TEXT    NOT NULL DEFAULT '',
                    icao_code       TEXT    NOT NULL,
                    model_mu        REAL    NOT NULL,
                    actual_settled  REAL    NOT NULL,
                    residual        REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS open_positions (
                    token_id        TEXT    PRIMARY KEY,
                    bracket_label   TEXT    NOT NULL,
                    icao_code       TEXT    NOT NULL,
                    entry_price     REAL    NOT NULL,
                    size_usd        REAL    NOT NULL,
                    opened_at       TEXT    NOT NULL,
                    peak_price      REAL    NOT NULL DEFAULT 0.0,
                    market_date     TEXT    NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS signal_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    date            TEXT    NOT NULL,
                    bracket_label   TEXT    NOT NULL,
                    model_prob      REAL    NOT NULL,
                    market_price    REAL    NOT NULL,
                    edge            REAL    NOT NULL,
                    action          TEXT    NOT NULL,
                    settled_outcome TEXT,
                    gate_reason     TEXT    DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS token_matrix (
                    bracket_label   TEXT    PRIMARY KEY,
                    token_id        TEXT    NOT NULL,
                    no_token_id     TEXT    NOT NULL DEFAULT '',
                    market_date     TEXT    NOT NULL,
                    updated_at      TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS exit_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    market_date     TEXT    NOT NULL DEFAULT '',
                    token_id        TEXT    NOT NULL,
                    bracket_label   TEXT    NOT NULL,
                    direction       TEXT    NOT NULL,
                    reason          TEXT    NOT NULL,
                    entry_price     REAL    NOT NULL,
                    exit_price      REAL,
                    size_usd        REAL    NOT NULL,
                    realised_pnl    REAL    NOT NULL,
                    opened_at       TEXT    NOT NULL,
                    closed_at       TEXT    NOT NULL
                );

                -- Point-in-time record: never UPDATEd, so "what did the engine
                -- know at T" is the latest row with scan_at <= T.
                CREATE TABLE IF NOT EXISTS scan_snapshots (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_at          TEXT    NOT NULL,
                    market_date      TEXT    NOT NULL,
                    forecast_source  TEXT    NOT NULL,
                    mu               REAL    NOT NULL,
                    sigma            REAL    NOT NULL,
                    mu_gfs           REAL,
                    mu_ecmwf         REAL,
                    sigma_gfs        REAL,
                    sigma_ecmwf      REAL,
                    trailing_bias    REAL    NOT NULL,
                    historical_sigma REAL,
                    observed_high    REAL,
                    observed_at      TEXT    NOT NULL DEFAULT '',
                    model_probs      TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS book_snapshots (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id     INTEGER,
                    fetched_at  TEXT    NOT NULL,
                    token_id    TEXT    NOT NULL,
                    purpose     TEXT    NOT NULL,
                    bids        TEXT    NOT NULL,
                    asks        TEXT    NOT NULL
                );
            """)
            conn.commit()

    def _migrate_schema(self):
        """
        Idempotent migration for DBs created before market_date columns existed.
        SQLite has no `ADD COLUMN IF NOT EXISTS`, so check pragma first.
        Safe to run on every startup — no-ops if columns already present.

        Indexes referencing market_date are created HERE (after the ALTER
        TABLE calls), not in _init_schema's CREATE TABLE script — on a
        pre-existing DB, _init_schema's CREATE TABLE IF NOT EXISTS is a
        no-op that leaves the OLD schema in place, so creating a
        market_date index in that same script would fail with
        "no such column" before this migration ever runs.
        """
        with self._conn() as conn:
            for table, col in [("calibration_logs", "market_date"),
                                ("exit_log", "market_date"),
                                ("open_positions", "market_date"),
                                ("token_matrix", "no_token_id")]:
                cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
                if col not in cols:
                    logger.info(f"[LEDGER] Migrating: adding {col} to {table}")
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")

            # Indexes — safe now that market_date is guaranteed present on both tables.
            signal_cols = {row["name"] for row in conn.execute("PRAGMA table_info(signal_log)")}
            for col, definition in [("observed_high", "REAL"), ("observed_at", "TEXT NOT NULL DEFAULT ''"),
                                    ("forecast_source", "TEXT NOT NULL DEFAULT ''"), ("scan_id", "INTEGER")]:
                if col not in signal_cols:
                    conn.execute(f"ALTER TABLE signal_log ADD COLUMN {col} {definition}")
            for table in ("open_positions", "exit_log"):
                if "scan_id" not in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN scan_id INTEGER")
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_calib_icao_date
                    ON calibration_logs(icao_code, market_date);
                CREATE INDEX IF NOT EXISTS idx_signal_date_bracket
                    ON signal_log(date, bracket_label);
                CREATE INDEX IF NOT EXISTS idx_exit_market_date
                    ON exit_log(market_date);
                CREATE INDEX IF NOT EXISTS idx_scan_date_at
                    ON scan_snapshots(market_date, scan_at);
                CREATE INDEX IF NOT EXISTS idx_book_scan
                    ON book_snapshots(scan_id);
            """)

    # ── Calibration ───────────────────────────────────────────────────────────

    def log_outcome(self, icao: str, model_mu: float, actual_settled: float, market_date: str = ""):
        """
        market_date: the SGT calendar date (YYYY-MM-DD) this outcome belongs to.
        Stored separately from `timestamp` (write time, UTC) so idempotency
        checks and daily grouping are correct regardless of when Job 4 runs —
        important once settlement can fire in a round-the-clock schedule and
        cross the UTC/SGT midnight boundary. If not passed, falls back to
        the UTC date derived from timestamp (legacy behaviour — can mismatch
        the true SGT market date for writes between 00:00-07:59 SGT).
        """
        residual  = actual_settled - model_mu
        ts        = datetime.datetime.utcnow().isoformat()
        if not market_date:
            market_date = ts[:10]
            logger.warning(
                "[LEDGER] log_outcome called without market_date — "
                f"falling back to UTC write date {market_date}, which can "
                "mismatch the true SGT market date. Callers should pass it explicitly."
            )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO calibration_logs "
                "(timestamp, market_date, icao_code, model_mu, actual_settled, residual) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts, market_date, icao.upper(), model_mu, actual_settled, residual),
            )
        logger.info(
            f"[LEDGER] Outcome: date={market_date} model={model_mu:.2f} "
            f"actual={actual_settled:.2f} residual={residual:+.2f}°C"
        )

    def has_calibration_for_date(self, icao: str, date: str) -> bool:
        """
        Check whether a calibration_logs row already exists for this ICAO
        on this SGT market date. Prevents Job 4 from writing a duplicate
        row every 10-minute settlement cycle throughout the evening —
        without this guard, one real day of trading produces 15-20+ junk
        rows, each with a stale/partial actual_temp and a fallback model_mu
        if Job 2 never ran that day.

        Queries the explicit market_date column (not a timestamp LIKE match —
        that broke for any write between 00:00-07:59 SGT, which stores a UTC
        timestamp on the PREVIOUS calendar date and silently missed its own
        row, causing exactly the duplicate-write bug this guard exists to prevent).
        """
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT 1 FROM calibration_logs "
                    "WHERE icao_code = ? AND market_date = ? LIMIT 1",
                    (icao.upper(), date),
                ).fetchone()
            return row is not None
        except Exception as e:
            logger.error(f"[-] has_calibration_for_date failed: {e}")
            return False  # fail open — rare dup beats losing calibration entirely

    def fetch_trailing_bias(self, icao: str, n: int = 10) -> float:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT AVG(residual) FROM (
                    SELECT residual FROM calibration_logs
                    WHERE icao_code = ?
                    ORDER BY id DESC LIMIT ?
                )
            """, (icao.upper(), n)).fetchone()
        val = row[0] if row else None
        return float(val) if val is not None else 0.0

    def fetch_residual_std(self, icao: str, n: int = 10) -> Optional[float]:
        """
        Sample std dev (ddof=1) of the last n calibration residuals — the
        model's REAL day-to-day forecast error, as opposed to the ensemble
        spread core/model.py reports as its own sigma. The two can diverge
        sharply: ensemble spread reflects member disagreement on a single
        run, not the model's actual historical accuracy, and weather
        ensembles are well known to be under-dispersive (spread understates
        true error). Returns None (not 0.0) with fewer than 2 data points —
        std dev is undefined for 0-1 samples, and the caller must not treat
        "insufficient data" the same as "confirmed zero variance".
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT residual FROM calibration_logs
                WHERE icao_code = ?
                ORDER BY id DESC LIMIT ?
            """, (icao.upper(), n)).fetchall()
        residuals = [r[0] for r in rows]
        if len(residuals) < 2:
            return None
        mean = sum(residuals) / len(residuals)
        variance = sum((r - mean) ** 2 for r in residuals) / (len(residuals) - 1)
        return variance ** 0.5

    # ── Token matrix (refreshed daily by discovery job) ──────────────────────

    def upsert_token_matrix(
        self, bracket_label: str, token_id: str, no_token_id: str, market_date: str,
    ):
        ts = datetime.datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO token_matrix "
                "(bracket_label, token_id, no_token_id, market_date, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (bracket_label, token_id, no_token_id, market_date, ts),
            )

    def get_token_matrix(self, market_date: str) -> Dict[str, Dict[str, str]]:
        """Returns {bracket_label: {"yes": token_id, "no": no_token_id}}."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT bracket_label, token_id, no_token_id FROM token_matrix "
                "WHERE market_date = ?",
                (market_date,),
            ).fetchall()
        return {
            r["bracket_label"]: {"yes": r["token_id"], "no": r["no_token_id"]}
            for r in rows
        }

    # ── Positions ─────────────────────────────────────────────────────────────

    def is_position_open(self, token_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM open_positions WHERE token_id = ?", (token_id,)
            ).fetchone()
        return row is not None

    def record_position(
        self, token_id: str, label: str, icao: str,
        entry_price: float, size_usd: float, market_date: str = "",
        scan_id: Optional[int] = None,
    ):
        """
        market_date: SGT calendar date this position was opened under.
        Stored separately from `opened_at` (UTC write timestamp) so
        position_monitor.py's per-position time-exit logic can compare
        SGT dates directly instead of parsing a UTC timestamp — the same
        00:00-07:59 SGT mismatch window that affected calibration_logs and
        exit_log applies here too (a position opened at 06:00 SGT stores a
        UTC timestamp dated the PREVIOUS calendar day).
        """
        ts = datetime.datetime.utcnow().isoformat()
        if not market_date:
            market_date = ts[:10]
            logger.warning(
                "[LEDGER] record_position called without market_date — "
                f"falling back to UTC write date {market_date}."
            )
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO open_positions "
                "(token_id, bracket_label, icao_code, entry_price, size_usd, "
                " opened_at, peak_price, market_date, scan_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (token_id, label, icao.upper(), entry_price, size_usd, ts,
                 entry_price, market_date, scan_id),
            )
        logger.info(
            f"[LEDGER] Position open: {label} @ {entry_price:.4f} "
            f"${size_usd:.2f} (market_date={market_date})"
        )

    def close_position(self, token_id: str):
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM open_positions WHERE token_id = ?", (token_id,)
            )
        logger.info(f"[LEDGER] Position closed: {token_id}")

    def get_open_positions(self) -> List[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute("SELECT * FROM open_positions").fetchall()

    def update_peak_price(self, token_id: str, new_peak: float):
        """
        Update the peak_price for a position — called by trailing stop
        each cycle when the current mid exceeds the stored peak.
        BUY positions: peak_price = highest mid seen since entry.
        SELL positions: peak_price = lowest mid seen since entry.
        """
        try:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE open_positions SET peak_price = ? WHERE token_id = ?",
                    (new_peak, token_id),
                )
        except Exception as e:
            logger.error(f"[-] update_peak_price failed for {token_id}: {e}")

    def get_peak_price(self, token_id: str) -> float:
        """Return stored peak_price for a position, or 0.0 if not found."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT peak_price FROM open_positions WHERE token_id = ?",
                    (token_id,),
                ).fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception as e:
            logger.error(f"[-] get_peak_price failed for {token_id}: {e}")
            return 0.0

    def find_stuck_positions(self, ttl_hours: int = 28) -> List[sqlite3.Row]:
        """
        Returns open_positions rows that have been open longer than
        ttl_hours — for ALERTING only. Does NOT delete or otherwise touch
        them.

        Replaces the old expire_stale_positions(), which silently DELETEd
        these rows without ever placing a closing order. A position open
        this long means position_monitor.py's own per-position
        force_time_exit logic (which already retries a REAL close every
        Job 5 cycle for anything carried over from a prior market_date)
        has been failing to actually close it — thin book, repeated FOK
        rejections, or a bug. The wallet still holds the real shares.
        Confirmed in production: two positions vanished from the dashboard
        while still sitting open, unmanaged (no more trailing stop/stop
        loss), and visibly held in the Polymarket wallet UI the whole
        time — the old delete-only behaviour hid a live, unmanaged
        financial exposure instead of surfacing it.

        Job 5 keeps retrying a genuine close every cycle regardless of
        whether this method is ever called; this only surfaces "something
        has been stuck a long time" so an operator can investigate
        manually if those retries keep failing.
        """
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=ttl_hours)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT token_id, bracket_label, opened_at FROM open_positions "
                "WHERE opened_at < ?",
                (cutoff,),
            ).fetchall()
        return rows

    # ── Signal log ────────────────────────────────────────────────────────────

    def log_signal(
        self, date: str, bracket_label: str,
        model_prob: float, market_price: float,
        edge: float, action: str, gate_reason: str = "",
        observed_high: Optional[float] = None, observed_at: str = "", forecast_source: str = "",
        scan_id: Optional[int] = None,
    ):
        """
        gate_reason: previously a schema column that existed but was never
        populated by this method (always silently defaulted to ''). action
        already encodes the same information (e.g. SKIP_ILLIQUID, HOLD_EDGE),
        so this is mostly redundant, but wiring it through means the
        dashboard's gate_reason queries return real data instead of always
        empty strings.
        """
        ts = datetime.datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO signal_log "
                "(timestamp, date, bracket_label, model_prob, market_price, edge, action, gate_reason, "
                "observed_high, observed_at, forecast_source, scan_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, date, bracket_label, model_prob, market_price, edge, action, gate_reason,
                 observed_high, observed_at, forecast_source, scan_id),
            )

    def mark_signal_settled(self, date: str, bracket_label: str, outcome: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE signal_log SET settled_outcome = ? "
                "WHERE date = ? AND bracket_label = ? AND settled_outcome IS NULL",
                (outcome, date, bracket_label),
            )

    # ── Point-in-time snapshots ───────────────────────────────────────────────

    def log_scan(
        self, scan_at: str, market_date: str, forecast, trailing_bias: float,
        historical_sigma: Optional[float], observations: Optional[dict],
        model_probs: Dict[str, float],
    ) -> int:
        """Record every input Job 2 used; returns the scan_id that signals, books and trades cite."""
        with self._conn() as conn:
            return conn.execute(
                "INSERT INTO scan_snapshots (scan_at, market_date, forecast_source, mu, sigma, "
                "mu_gfs, mu_ecmwf, sigma_gfs, sigma_ecmwf, trailing_bias, historical_sigma, "
                "observed_high, observed_at, model_probs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (scan_at, market_date, forecast.source, forecast.mu, forecast.sigma,
                 forecast.mu_gfs, forecast.mu_ecmwf, forecast.sigma_gfs, forecast.sigma_ecmwf,
                 trailing_bias, historical_sigma,
                 observations["high_c"] if observations else None,
                 observations["observed_at"] if observations else "",
                 json.dumps(model_probs, sort_keys=True)),
            ).lastrowid

    def log_book(self, scan_id: Optional[int], token_id: str, purpose: str,
                 bids: List[Tuple[float, float]], asks: List[Tuple[float, float]],
                 fetched_at: Optional[str] = None):
        """Full-depth book as [[price, size], ...]; purpose is 'scan' or 'exec'."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO book_snapshots (scan_id, fetched_at, token_id, purpose, bids, asks) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (scan_id, fetched_at or datetime.datetime.utcnow().isoformat(), token_id,
                 purpose, json.dumps(bids), json.dumps(asks)),
            )

    def scan_as_of(self, market_date: str, at: str) -> Optional[dict]:
        """
        What the engine knew for market_date at UTC ISO time `at`: the latest
        scan at or before it, with the books that scan saw. None if no scan yet.
        """
        with self._conn() as conn:
            scan = conn.execute(
                "SELECT * FROM scan_snapshots WHERE market_date = ? AND scan_at <= ? "
                "ORDER BY scan_at DESC, id DESC LIMIT 1",
                (market_date, at),
            ).fetchone()
            if scan is None:
                return None
            books = conn.execute(
                "SELECT * FROM book_snapshots WHERE scan_id = ? AND fetched_at <= ? ORDER BY id",
                (scan["id"], at),
            ).fetchall()
        result = dict(scan)
        result["model_probs"] = json.loads(result["model_probs"])
        result["books"] = [dict(b, bids=json.loads(b["bids"]), asks=json.loads(b["asks"])) for b in books]
        return result

    # ── Exit log ──────────────────────────────────────────────────────────────

    def log_exit(
        self,
        token_id:     str,
        bracket_label: str,
        direction:    str,
        reason:       str,
        entry_price:  float,
        exit_price:   float,
        size_usd:     float,
        realised_pnl: float,
        opened_at:    str,
        market_date:  str = "",
        scan_id:      Optional[int] = None,
    ):
        """
        Record a completed position exit with P&L.
        Called by PositionMonitor after confirmed fill.

        market_date: SGT calendar date this position's market belongs to.
        Same rationale as log_outcome() — stored separately from the UTC
        write timestamp so daily_pnl() and get_exit_log(date=...) are
        correct for exits that happen between 00:00-07:59 SGT (whose UTC
        timestamp falls on the previous calendar date).
        """
        ts = datetime.datetime.utcnow().isoformat()
        if not market_date:
            market_date = ts[:10]
            logger.warning(
                "[LEDGER] log_exit called without market_date — "
                f"falling back to UTC write date {market_date}."
            )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO exit_log "
                "(timestamp, market_date, token_id, bracket_label, direction, reason, "
                " entry_price, exit_price, size_usd, realised_pnl, opened_at, closed_at, scan_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, market_date, token_id, bracket_label, direction, reason,
                 entry_price, exit_price, size_usd, realised_pnl, opened_at, ts, scan_id),
            )
        pnl_pct = (realised_pnl / size_usd * 100) if size_usd else 0.0
        logger.info(
            f"[LEDGER] Exit logged: {bracket_label} [{direction}] "
            f"reason={reason} P&L={realised_pnl:+.4f} ({pnl_pct:+.1f}%)"
        )

    def get_exit_log(self, date: str = None) -> List[sqlite3.Row]:
        """Return exit records. If date given (YYYY-MM-DD SGT market date), filter on it."""
        with self._conn() as conn:
            if date:
                return conn.execute(
                    "SELECT * FROM exit_log WHERE market_date = ? ORDER BY id DESC",
                    (date,),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM exit_log ORDER BY id DESC LIMIT 100"
            ).fetchall()

    def daily_pnl(self, date: str) -> float:
        """Sum of realised_pnl for all exits on a given SGT market date."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realised_pnl), 0.0) FROM exit_log WHERE market_date = ?",
                (date,),
            ).fetchone()
        return float(row[0]) if row else 0.0

