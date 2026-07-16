"""
dashboard_api.py — Hermes v4.5 Dashboard REST API
FastAPI backend. Reads hermes.db and serves JSON to the web frontend.

Run:  uvicorn dashboard_api:app --host 0.0.0.0 --port 8000
Access: http://YOUR_VPS_IP:8000

Install: pip install fastapi uvicorn[standard]
(already in requirements.txt)

SECURITY NOTE: this API has no authentication and binds 0.0.0.0 (all
interfaces) per deploy/hermes-dashboard.service. Anyone who can reach
port 8000 on the VPS can see full trade history, P&L, and open positions
(read-only — no route can modify state). If the VPS has a public IP,
firewall port 8000 to trusted IPs only, or put it behind a reverse proxy
with auth (nginx + basic auth, or a Cloudflare Tunnel/Access policy).
"""

import os
import time
import sqlite3
import datetime
import logging
import threading
import contextlib
from typing import Any, Dict, List

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core.model import ECMWF_ENSEMBLE_URL, WSSS_LAT, WSSS_LON

logger = logging.getLogger("hermes.dashboard_api")
logging.basicConfig(level=logging.INFO)

DB_PATH      = os.getenv("DB_PATH", "hermes.db")
VAULT_START  = float(os.getenv("MAX_VAULT_ALLOCATION", 200.0))
_DASHBOARD_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

# ── Upstream (Open-Meteo ECMWF) health check cache ───────────────────────────
# The dashboard polls this on a short client-side interval to show a live
# indicator, but the check itself hits the real Open-Meteo API — caching the
# result server-side for UPSTREAM_CACHE_TTL keeps N dashboard viewers/polls
# from turning into N x real requests against a third-party service.
UPSTREAM_CACHE_TTL = 30.0
_upstream_cache: Dict[str, Any] = {"result": None, "checked_at": 0.0}

# ── Wallet balance (deposit wallet, on-chain via CLOB balance-allowance) ─────
# Same read-only CLOB call core/execution.py now makes before every order
# (see PR that added the pre-order balance/allowance sync) — surfaced here so
# an operator can see the bot's own view of its funded balance without
# grepping journalctl for "balance: 0" after the fact. The CLOB client is
# built lazily (only once this endpoint is first hit) and reused, since
# construction does an auth round-trip; a lock keeps concurrent dashboard
# polls from racing to build it twice.
WALLET_BALANCE_CACHE_TTL = 20.0
_wallet_client = None
_wallet_client_lock = threading.Lock()
_wallet_balance_cache: Dict[str, Any] = {
    "result":        None,  # last successful {"balance_usd": ..., "synced_at": iso}
    "checked_at":    0.0,   # monotonic time of last attempt (success or fail)
}

app = FastAPI(title="Hermes Dashboard API", version="4.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Serve the single-file frontend ───────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_dashboard():
    # Absolute path anchored to this file's directory — a relative
    # "dashboard.html" would 404 if uvicorn is ever launched from a
    # different working directory than deploy/hermes-dashboard.service's
    # WorkingDirectory=/opt/hermes (e.g. manual local testing).
    return FileResponse(_DASHBOARD_HTML)

# ── DB helper ─────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _conn():
    """
    Same connection-leak fix as db/ledger.py: a bare sqlite3.Connection
    used as `with conn:` only wraps a transaction (commit/rollback) — it
    never calls close(). Confirmed: 200 calls leaked 13+ file descriptors
    relying on GC timing. This process runs continuously with the
    dashboard frontend polling every 30s across up to 9 endpoints per
    poll (kpis() alone makes 7 DB calls) — left unfixed this leaks far
    faster than the trading bot's own (already-fixed) instance of the
    same bug.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def _rows(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    if not os.path.exists(DB_PATH):
        return []
    try:
        with _conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        # Previously a bare `except Exception: return []` with no logging —
        # that silently swallows genuine bugs (a typo'd column name, a
        # missing table) as indistinguishable from "DB not ready yet",
        # making them invisible in journalctl. Log first, then degrade.
        logger.error(f"[DASHBOARD] Query failed: {query[:80]}... — {e}")
        return []

def _scalar(query: str, params: tuple = (), default: Any = 0.0) -> Any:
    if not os.path.exists(DB_PATH):
        return default
    try:
        with _conn() as conn:
            row = conn.execute(query, params).fetchone()
            return row[0] if row and row[0] is not None else default
    except Exception as e:
        logger.error(f"[DASHBOARD] Query failed: {query[:80]}... — {e}")
        return default

# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    """Health check + live/mock indicator."""
    live = os.path.exists(DB_PATH)
    sg   = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    return {
        "live":       live,
        "db_path":    DB_PATH,
        "vault_start": VAULT_START,
        "sgt_now":    sg.strftime("%Y-%m-%d %H:%M SGT"),
        "utc_now":    datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }

@app.get("/api/upstream_status")
def upstream_status():
    """
    Live health check for the Open-Meteo ECMWF ensemble API — the upstream
    forecast source core/model.py depends on for mu_ecmwf/sigma_ecmwf. If
    this is down, the bot falls back to GFS-only (or the hard prior), which
    is exactly the kind of degradation an operator wants surfaced on the
    dashboard rather than discovered later in the logs.

    ok=True requires BOTH a 200 response AND >=3 ensemble members in the
    body (the same threshold core/model.py._fetch_ensemble_members() gates
    on). A bare status-code check isn't enough: Open-Meteo can return 200
    with zero temperature_2m_member* keys (observed in production — the
    bot logged "ECMWF: only 0 members returned" and fell back to GFS-only
    while this endpoint, checking status code alone, would have reported
    "reachable"). n_members is returned so a degraded-but-200 response is
    visibly distinct from a fully healthy one, not just lumped into "ok".
    """
    now = time.monotonic()
    cached = _upstream_cache["result"]
    if cached is not None and (now - _upstream_cache["checked_at"]) < UPSTREAM_CACHE_TTL:
        return {**cached, "cached": True}

    url = ECMWF_ENSEMBLE_URL.format(lat=WSSS_LAT, lon=WSSS_LON)
    t0 = time.monotonic()
    try:
        r = requests.get(url, timeout=5)
        n_members = None
        if r.status_code == 200:
            try:
                hourly = r.json().get("hourly", {})
                n_members = sum(1 for k in hourly if k.startswith("temperature_2m_member"))
            except ValueError as e:
                logger.error(f"[DASHBOARD] ECMWF response not valid JSON: {e}")

        result = {
            "ok":           r.status_code == 200 and (n_members or 0) >= 3,
            "status_code":  r.status_code,
            "n_members":    n_members,
            "latency_ms":   round((time.monotonic() - t0) * 1000),
            "checked_at":   datetime.datetime.utcnow().isoformat(),
            "source":       "ecmwf_ensemble",
        }
    except requests.RequestException as e:
        logger.error(f"[DASHBOARD] Upstream ECMWF check failed: {e}")
        result = {
            "ok":           False,
            "status_code":  None,
            "n_members":    None,
            "latency_ms":   round((time.monotonic() - t0) * 1000),
            "checked_at":   datetime.datetime.utcnow().isoformat(),
            "source":       "ecmwf_ensemble",
            "error":        str(e),
        }

    _upstream_cache["result"]     = result
    _upstream_cache["checked_at"] = now
    return {**result, "cached": False}

def _get_wallet_client():
    """Build the CLOB client once and reuse it. Read-only usage here — this
    endpoint never signs or posts orders, only queries balance/allowance."""
    global _wallet_client
    if _wallet_client is not None:
        return _wallet_client
    with _wallet_client_lock:
        if _wallet_client is None:
            from core.execution import build_client
            _wallet_client = build_client()
        return _wallet_client

@app.get("/api/wallet_balance")
def wallet_balance():
    """
    Live deposit-wallet COLLATERAL balance, straight from the same CLOB
    balance-allowance endpoint core/execution.py now syncs immediately
    before every order (see execute()'s pre-order sync). Surfacing it here
    lets an operator see "does the bot currently believe it has funds?"
    without waiting for a trade attempt to fail with "balance: 0" in the logs.

    Cached for WALLET_BALANCE_CACHE_TTL so dashboard polling doesn't spam
    the CLOB API. On failure, returns the last known-good balance (if any)
    alongside ok=False so the frontend can show a stale-but-informative
    value instead of blanking out.
    """
    now = time.monotonic()
    cached = _wallet_balance_cache["result"]
    if cached is not None and (now - _wallet_balance_cache["checked_at"]) < WALLET_BALANCE_CACHE_TTL:
        return {**cached, "ok": True, "cached": True}

    try:
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType

        client = _get_wallet_client()
        client.update_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        raw = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))

        raw_balance = raw.get("balance") if isinstance(raw, dict) else getattr(raw, "balance", None)
        balance_usd = round(float(raw_balance) / 1_000_000, 2) if raw_balance is not None else None

        synced_at = datetime.datetime.utcnow().isoformat()
        result = {
            "balance_usd": balance_usd,
            "synced_at":   synced_at,
        }
        _wallet_balance_cache["result"]     = result
        _wallet_balance_cache["checked_at"] = now
        return {**result, "ok": True, "cached": False}

    except Exception as e:
        logger.error(f"[DASHBOARD] Wallet balance sync failed: {e}")
        _wallet_balance_cache["checked_at"] = now
        if cached is not None:
            return {**cached, "ok": False, "cached": True, "error": str(e)}
        return {"balance_usd": None, "synced_at": None, "ok": False, "cached": False, "error": str(e)}

@app.get("/api/kpis")
def kpis():
    """Headline KPI numbers."""
    net_pnl      = _scalar("SELECT COALESCE(SUM(realised_pnl),0) FROM exit_log")
    total_trades = _scalar("SELECT COUNT(*) FROM exit_log", default=0)
    wins         = _scalar("SELECT COUNT(*) FROM exit_log WHERE realised_pnl > 0", default=0)
    trail_bias   = _scalar("""
        SELECT AVG(residual) FROM (
            SELECT residual FROM calibration_logs ORDER BY id DESC LIMIT 10
        )
    """, default=0.0)
    mae          = _scalar("SELECT AVG(ABS(residual)) FROM calibration_logs", default=0.0)
    n_calib      = _scalar("SELECT COUNT(*) FROM calibration_logs", default=0)
    n_open       = _scalar("SELECT COUNT(*) FROM open_positions", default=0)
    actionable   = _scalar(
        "SELECT COUNT(*) FROM signal_log WHERE action IN ('SIGNAL_BUY','SIGNAL_SELL_NO')",
        default=0,
    )
    avg_edge     = _scalar(
        "SELECT AVG(ABS(edge)) FROM signal_log WHERE action IN ('SIGNAL_BUY','SIGNAL_SELL_NO')",
        default=0.0,
    )
    losses       = total_trades - wins
    win_rate     = (wins / total_trades * 100) if total_trades else 0.0
    roi          = (net_pnl / VAULT_START * 100) if VAULT_START else 0.0
    return {
        "vault_current":  round(VAULT_START + net_pnl, 2),
        "vault_start":    VAULT_START,
        "net_pnl":        round(net_pnl, 4),
        "roi_pct":        round(roi, 2),
        "total_trades":   int(total_trades),
        "wins":           int(wins),
        "losses":         int(losses),
        "win_rate_pct":   round(win_rate, 1),
        "trailing_bias":  round(trail_bias, 4),
        "mae_celsius":    round(mae, 4),
        "n_calibrations": int(n_calib),
        "open_positions": int(n_open),
        "actionable_signals": int(actionable),
        "avg_edge_pct":   round(avg_edge * 100, 2),
    }

@app.get("/api/equity")
def equity():
    """Cumulative vault equity per closed trade."""
    rows = _rows(
        "SELECT id, bracket_label, direction, reason, "
        "entry_price, exit_price, size_usd, realised_pnl, closed_at "
        "FROM exit_log ORDER BY id ASC"
    )
    running = VAULT_START
    for r in rows:
        running += r["realised_pnl"]
        r["vault"] = round(running, 4)
    return {"vault_start": VAULT_START, "trades": rows}

@app.get("/api/signals")
def signals(limit: int = 80):
    """Recent signal scan results — ALL brackets including non-actionable."""
    rows = _rows(
        "SELECT id, date, bracket_label, model_prob, market_price, "
        "edge, action, settled_outcome, COALESCE(gate_reason,'') as gate_reason "
        "FROM signal_log ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows.reverse()
    return {"signals": rows}


@app.get("/api/signal_summary")
def signal_summary():
    """
    Breakdown of signal action labels across all time.
    Lets the dashboard show a full scan funnel:
      Total priced → Edge threshold met → EV/sizing passed → Executed
    """
    rows = _rows(
        "SELECT action, COUNT(*) as count, "
        "AVG(ABS(edge)) as avg_abs_edge "
        "FROM signal_log GROUP BY action ORDER BY count DESC"
    )
    # Map to display-friendly labels
    LABEL_MAP = {
        "SIGNAL_BUY":     "BUY signal",
        "SIGNAL_SELL_NO": "SELL signal",
        "HOLD_EDGE":      "Held — edge below 5%",
        "SKIP_ILLIQUID":  "Skipped — illiquid",
        "SKIP_SPREAD":    "Skipped — wide spread",
        "SKIP_EXTREME_EDGE": "Skipped — implausible edge",
        "NO_PRICE":       "No price fetched",
    }
    for r in rows:
        r["display_label"] = LABEL_MAP.get(r["action"], r["action"])
        r["avg_abs_edge"]  = round(r["avg_abs_edge"] or 0.0, 4)
    return {"breakdown": rows}

@app.get("/api/latest_scan")
def latest_scan():
    """
    Every bracket from the MOST RECENT scan cycle — passing AND non-passing.
    This is the live snapshot: for the current market, which brackets cleared
    the edge gate, which were held below threshold, and which were gated on
    liquidity/spread — each with its model prob, market price, and edge.
    """
    latest = _rows(
        "SELECT id, date, bracket_label, model_prob, market_price, edge, "
        "action, COALESCE(gate_reason,'') as gate_reason "
        "FROM signal_log ORDER BY id DESC LIMIT 40"
    )
    if not latest:
        return {"scan": [], "scan_date": None, "n_brackets": 0,
                "n_passed": 0, "n_blocked": 0, "edge_threshold": 0.05}

    # Keep the most recent row per bracket (highest id)
    seen = {}
    for r in latest:
        if r["bracket_label"] not in seen:
            seen[r["bracket_label"]] = r
    scan = sorted(seen.values(), key=lambda r: r["bracket_label"])

    THRESH = float(os.getenv("EDGE_THRESHOLD", 0.05))
    STATUS = {
        "SIGNAL_BUY":     ("PASS",  "BUY YES",            True),
        "SIGNAL_SELL_NO": ("PASS",  "SELL NO",            True),
        "HOLD_EDGE":      ("HOLD",  f"edge < {THRESH*100:.0f}%", False),
        "SKIP_ILLIQUID":  ("GATED", "illiquid book",      False),
        "SKIP_SPREAD":    ("GATED", "spread > 8c",        False),
        "SKIP_EXTREME_EDGE": ("GATED", "implausible edge — likely miscalibration", False),
        "NO_PRICE":       ("GATED", "no price",           False),
    }
    for r in scan:
        status, label, passed = STATUS.get(r["action"], ("HOLD", "hold", False))
        r["status"]       = status
        r["status_label"] = label
        r["passed"]       = passed
        r["edge_pct"]     = round(r["edge"] * 100, 2)
        r["model_pct"]    = round(r["model_prob"] * 100, 1)
        r["market_pct"]   = round(r["market_price"] * 100, 1)

    n_pass = sum(1 for r in scan if r["passed"])
    return {
        "scan":           scan,
        "scan_date":      latest[0]["date"],
        "edge_threshold": THRESH,
        "n_brackets":     len(scan),
        "n_passed":       n_pass,
        "n_blocked":      len(scan) - n_pass,
    }

@app.get("/api/calibration")
def calibration():
    """
    All calibration residuals, in the same id-ordered sequence the live bot
    itself processes them in (db/ledger.py:fetch_trailing_bias orders by id,
    not market_date — id is "the order calibration rows were written in",
    which is usually but not always the same as calendar-date order: Job 4
    checks both today's and yesterday's date every cycle, so a backfilled
    prior-day entry can land right after a same-day entry, e.g. an id=N+1
    row can have an EARLIER market_date than id=N. Rather than silently
    reorder by market_date (which would also disagree with what the bot
    actually used at trade time), this returns market_date alongside each
    row so the frontend can label bars with real dates and make any such
    out-of-order backfill visible instead of hiding it behind a generic
    sequence number.

    trailing_bias here is a ROLLING last-10 window, matching
    fetch_trailing_bias(icao, n=10) exactly (same id order, same window
    size) — NOT an expanding all-time average. An expanding mean smooths
    over the bot's entire history forever and increasingly diverges from
    what the bot is actually using to calibrate live trades as more data
    accumulates; this makes the dashboard line track the real number.
    """
    rows = _rows(
        "SELECT id, timestamp, market_date, icao_code, model_mu, actual_settled, residual "
        "FROM calibration_logs ORDER BY id ASC"
    )
    window = []
    for r in rows:
        window.append(r["residual"])
        if len(window) > 10:
            window.pop(0)
        r["trailing_bias"] = round(sum(window) / len(window), 4)
    return {"calibrations": rows}

@app.get("/api/pnl_by_bracket")
def pnl_by_bracket():
    """P&L grouped by bracket + direction."""
    rows = _rows(
        "SELECT bracket_label, direction, "
        "SUM(realised_pnl) AS total_pnl, COUNT(*) AS n_trades, "
        "SUM(CASE WHEN realised_pnl > 0 THEN 1 ELSE 0 END) AS wins "
        "FROM exit_log GROUP BY bracket_label, direction ORDER BY bracket_label"
    )
    return {"groups": rows}

@app.get("/api/exit_reasons")
def exit_reasons():
    """Exit reason counts."""
    rows = _rows(
        "SELECT reason, COUNT(*) AS count, "
        "SUM(realised_pnl) AS total_pnl "
        "FROM exit_log GROUP BY reason ORDER BY count DESC"
    )
    return {"reasons": rows}

@app.get("/api/open_positions")
def open_positions():
    """All currently open positions with live trail/stop levels."""
    rows = _rows("SELECT * FROM open_positions ORDER BY opened_at ASC")
    trail_pct   = float(os.getenv("TRAIL_PCT", 0.20))
    edge_thresh = float(os.getenv("EDGE_THRESHOLD", 0.05))
    for r in rows:
        label     = r["bracket_label"]
        direction = "NO" if ":NO" in label else "YES"
        entry     = float(r["entry_price"])
        peak_raw  = r.get("peak_price")
        peak      = float(peak_raw) if peak_raw is not None else entry
        r["direction"]   = direction
        r["trail_pct"]   = trail_pct
        if direction == "YES":
            r["trail_level"]  = round(peak * (1 - trail_pct), 5) if peak > entry else None
            r["trail_armed"]  = peak > entry
            r["stop_level"]   = round(entry - edge_thresh, 5)
        else:
            r["trail_level"]  = round(peak * (1 + trail_pct), 5) if peak < entry else None
            r["trail_armed"]  = peak < entry
            r["stop_level"]   = round(entry + edge_thresh, 5)
        # Hold duration
        try:
            opened = datetime.datetime.fromisoformat(r["opened_at"])
            delta  = datetime.datetime.utcnow() - opened
            hours  = delta.total_seconds() / 3600
            r["hold_hours"] = round(hours, 1)
        except Exception:
            r["hold_hours"] = 0.0
    return {"positions": rows}

@app.get("/api/trades")
def trades(limit: int = 100):
    """Recent trade history."""
    rows = _rows(
        "SELECT id, closed_at, bracket_label, direction, reason, "
        "entry_price, exit_price, size_usd, realised_pnl, opened_at "
        "FROM exit_log ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return {"trades": rows}
