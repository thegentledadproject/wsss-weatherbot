"""
scheduler.py — S2: APScheduler round-the-clock job architecture

v4.6 REDESIGN — round-the-clock coverage to match how Polymarket WSSS
markets actually trade. Confirmed from live observation across this
session: markets for date D launch the evening BEFORE D (~23:00 SGT),
trade continuously through D, and can resolve/settle late in the evening
or after midnight D+1 depending on when the final METAR/Wunderground
reading posts. The old schedule (daytime-only windows, single daily
discovery trigger) had three consequences, all now fixed:

  1. Job 1 fired once at 07:30 SGT with no retry. If Gamma was down or the
     slug format didn't match that single time (exactly what happened in
     production on Jul 2-3 — two full days of zero trades), the bot had
     no token matrix until the NEXT day's 07:30 trigger — a ~24h dead zone.
     FIX: Job 1 now runs every 20 min, all day, self-healing within
     minutes instead of waiting a full day.

  2. Jobs 2/3/5 only ran 08:00-17:xx SGT, missing the evening launch window
     and any edge that appears overnight as forecasts update or the market
     re-prices. FIX: Jobs 2/3/5 now run every 15/15/5 min, 24/7. Market
     quality gates (liquidity floor, spread cap) already suppress bad
     signals on thin overnight books — a clock-based window isn't needed
     for that, and was excluding legitimate daytime-equivalent trading
     hours in other time zones' liquidity providers.

  3. Job 5's old window (hour="8-15") never overlapped position_monitor.py's
     own HARD_EXIT_HOUR_SGT=16 check, meaning the 16:00 SGT hard time-exit
     could never actually fire — confirmed by re-reading both files
     side by side. FIX: Job 5 now runs 24/7, so force_time_exit's own
     internal logic (unchanged) can actually execute.

  Consequence of #3's fix requiring a companion fix: once Jobs 2/3 can open
  NEW positions at any hour (including after 16:00 SGT), a purely
  wall-clock-hour force-exit would immediately close a position opened at,
  say, 20:00 SGT on its very next Job 5 tick — a self-defeating trade.
  position_monitor.py's force_time_exit is therefore now scoped to
  positions whose opened_at date is STRICTLY BEFORE today's SGT calendar
  date (i.e. genuinely stale, carried over from a prior market_date),
  not simply "it's currently past 16:00."

  4. Job 4 only settled _state["market_date"] (today). A position opened
     late yesterday could still be resolving in the early hours of today,
     after market_date has already rolled over — that outcome was never
     checked. FIX: Job 4 now settles BOTH today's and yesterday's date
     every cycle. Safe because settlement.py's has_calibration_for_date
     guard (db/ledger.py) makes this idempotent regardless of how many
     times or how many dates are checked per cycle.

Jobs (all times SGT, all 24/7 unless noted):
  Job 1 — market_discovery   : every 20 min — self-healing token matrix
  Job 2 — signal_scan        : every 15 min — forecast + edge scan
  Job 3 — order_execution    : every 15 min (offset +2 min from Job 2)
  Job 4 — settlement_check   : every 15 min — checks today AND yesterday
  Job 5 — position_monitor   : every 5 min  — trailing stop/stop loss/time exit

State shared across jobs (in-process dict, not DB):
  _state = {
      "token_matrix":  {label: {"yes": token_id, "no": no_token_id}},
      "signals":       {label: EdgeSignal},
      "forecast":      ForecastResult,
      "model_probs":   {label: float},
      "model_mu":      float,
      "market_date":   str,   # refreshed every 20 min by Job 1 — rolls over
                               # across SGT midnight automatically, no special-
                               # casing needed elsewhere.
  }

APScheduler runs jobs in a thread pool (max_workers=3).
The CLOB client is instantiated once at startup and shared.
"""

import os
import logging
import datetime
import signal
import sys

from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("hermes.log"),
    ],
)
logger = logging.getLogger("hermes.scheduler")

# ── Lazy imports (so missing deps fail loudly at runtime, not import time) ────
from db.ledger       import Ledger
from core.discovery  import MarketDiscovery
from core.model      import BracketModel, fetch_gfs_forecast
from core.edge       import scan_all_brackets
from core.sizing     import compute_size, compute_validation_size, check_sizing_config
from core.execution  import ExecutionEngine, build_client
from core.settlement    import SettlementEngine
from core.position_monitor import PositionMonitor

# ── Config from environment ────────────────────────────────────────────────────
DB_PATH       = os.getenv("DB_PATH",              "hermes.db")
VAULT_USD     = float(os.getenv("MAX_VAULT_ALLOCATION") or 200.0)
EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", 0.05))
MAX_EDGE_MAGNITUDE = float(os.getenv("MAX_EDGE_MAGNITUDE", 0.50))
TRAIL_PCT      = float(os.getenv("TRAIL_PCT", 0.20))

# VALIDATION_MODE: forces $1 trades on any actionable signal, bypassing
# Kelly sizing and EV hurdles entirely. Full lifecycle (entry, trailing
# stop, stop loss, settlement) still runs for real — only sizing/gating
# is bypassed. Use to prove mechanics work before deploying real capital.
# Set VALIDATION_MODE=false in .env once validation run is complete.
VALIDATION_MODE = os.getenv("VALIDATION_MODE", "false").lower() == "true"
ICAO          = "WSSS"

# ── Shared state ───────────────────────────────────────────────────────────────
_state: dict = {
    "token_matrix": {},
    "signals":      {},
    "forecast":     None,
    "model_probs":  {},
    "model_mu":     31.5,
    "market_date":  "",
}

# ── Shared singletons ──────────────────────────────────────────────────────────
_ledger    = Ledger(DB_PATH)
_client    = None   # initialised in main() to catch auth errors early


def _sg_now() -> datetime.datetime:
    return datetime.datetime.utcnow() + datetime.timedelta(hours=8)


# ══════════════════════════════════════════════════════════════════════════════
# JOB 1 — Market Discovery (every 20 min, 24/7 — self-healing)
# ══════════════════════════════════════════════════════════════════════════════
def job_market_discovery():
    sg_now     = _sg_now()
    today      = sg_now.strftime("%Y-%m-%d")
    tomorrow   = (sg_now + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    prior_date = _state.get("market_date", "")

    logger.info(f"[JOB1] ── Market Discovery @ {sg_now.strftime('%H:%M SGT')} ──────")

    discovery = MarketDiscovery(_ledger)

    # Prefer tomorrow's market once it's live — Polymarket launches date D's
    # market the evening of D-1 (~23:00 SGT), well before the SGT calendar
    # date itself rolls over to D (see module docstring). quiet=True since a
    # miss here is the normal, expected outcome for ~20 hours a day, not
    # something to log as a warning/error every 20-min tick.
    date   = today
    matrix = discovery.run(tomorrow, quiet=True)
    if matrix:
        logger.info(f"[JOB1] Tomorrow's market ({tomorrow}) is already live — trading it now.")
        date = tomorrow
    else:
        matrix = discovery.run(today)

    is_rollover = bool(prior_date) and prior_date != date
    if is_rollover:
        logger.info(f"[JOB1] Date rollover detected: {prior_date} → {date}")

    if not matrix:
        # "Keep the cached matrix and retry in 20 min" is only safe WITHIN
        # the same calendar day (a transient fetch miss). Once the date has
        # rolled over, a cached matrix is for YESTERDAY's already-settled
        # market — reusing it would have Jobs 2/3 silently scanning/trading
        # a dead market indefinitely (confirmed: this happened for real,
        # discovery never advanced past a stale prior-day matrix). So a
        # rollover always clears the cache and updates market_date below,
        # even when today's real event hasn't been found yet — Jobs 2/3
        # correctly see "no matrix" and skip until discovery catches up.
        if _state.get("token_matrix") and not is_rollover:
            logger.warning(
                "[JOB1] No fresh tokens found this cycle — keeping previously "
                "cached matrix until next retry."
            )
            return
        if is_rollover:
            logger.error(
                f"[JOB1] Date rolled over to {date} but today's market isn't "
                f"discoverable yet — clearing stale {prior_date} matrix. "
                f"Jobs 2/3 will skip until discovery succeeds for {date}."
            )
        else:
            logger.error("[JOB1] No tokens found and no cached matrix — Jobs 2/3 will skip.")
        matrix = {}
    else:
        # Validate discovered tokens against live Gamma, for whichever date
        # we actually ended up trading (today or tomorrow via lookahead).
        valid = discovery.validate_against_live(matrix, date=date)
        if not valid:
            logger.warning("[JOB1] Validation failed — re-running discovery.")
            matrix = discovery.run(date)

    _state["token_matrix"] = matrix
    _state["market_date"]  = date
    logger.info(f"[JOB1] Token matrix: {list(matrix.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
# JOB 2 — Signal Scan (every 15 min, 08:00–17:30 SGT)
# ══════════════════════════════════════════════════════════════════════════════
def job_signal_scan():
    sg_now_str = _sg_now().strftime("%H:%M SGT")
    logger.info(f"[JOB2] ── Signal Scan @ {sg_now_str} ────────────────────────────────")

    token_matrix = _state.get("token_matrix", {})
    if not token_matrix:
        logger.warning("[JOB2] No token matrix — skipping scan. Run Job 1 first.")
        # Clear any signals left over from a prior cycle — without this,
        # Job 3 would keep reading yesterday's already-actionable signals
        # (dead token ids on a now-settled market) straight through a date
        # rollover, since it trusts _state["signals"] with no cross-check
        # against token_matrix/market_date. Matches the other two early
        # returns below, which already clear signals for the same reason.
        _state["signals"] = {}
        return

    # Fetch live GFS forecast (ensemble sigma)
    forecast = fetch_gfs_forecast()
    _state["forecast"] = forecast

    if forecast.source == "fallback":
        logger.error("[JOB2] Forecast on hard prior — aborting scan.")
        _state["signals"] = {}
        return

    # Compute bracket probabilities with trailing bias + historical sigma
    # (see BracketModel.compute()'s docstring — ensemble spread alone has
    # repeatedly understated real forecast error by ~3x in this bot's own
    # calibration history).
    trailing_bias   = _ledger.fetch_trailing_bias(ICAO)
    historical_sigma = _ledger.fetch_residual_std(ICAO)
    model         = BracketModel(
        trailing_bias=trailing_bias, icao=ICAO, historical_sigma=historical_sigma,
    )
    model_probs   = model.compute(forecast)

    if not model_probs:
        logger.error("[JOB2] Model returned empty probs — aborting scan.")
        _state["signals"] = {}
        return

    _state["model_probs"] = model_probs
    _state["model_mu"]    = forecast.mu

    # Edge scan: model prob vs live market mid-price
    signals = scan_all_brackets(
        token_matrix       = token_matrix,
        model_probs        = model_probs,
        edge_threshold     = EDGE_THRESHOLD,
        max_edge_magnitude = MAX_EDGE_MAGNITUDE,
    )
    _state["signals"] = signals

    # Log ALL signals to DB — including non-actionable, gated, and held
    # This is the change that lets the dashboard show the full scan picture.
    date = _state.get("market_date", _sg_now().strftime("%Y-%m-%d"))
    for label, sig in signals.items():
        mid = sig.market_price.mid_price if sig.market_price else 0.0
        _ledger.log_signal(
            date          = date,
            bracket_label = label,
            model_prob    = sig.model_prob,
            market_price  = mid,
            edge          = sig.edge,
            action        = sig.action_label,   # includes HOLD_EDGE, SKIP_*, NO_PRICE
        )

    buys  = [l for l, s in signals.items() if s.direction == "BUY"  and s.actionable]
    sells = [l for l, s in signals.items() if s.direction == "SELL" and s.actionable]
    held  = [l for l, s in signals.items() if not s.actionable and not s.gate_reason]
    gated = [l for l, s in signals.items() if s.gate_reason]
    logger.info(
        f"[JOB2] Scan complete — "
        f"BUY:{buys or 'none'} SELL:{sells or 'none'} "
        f"HOLD_EDGE:{held or 'none'} GATED:{gated or 'none'}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# JOB 3 — Order Execution (every 15 min, 08:00–17:30 SGT)
# ══════════════════════════════════════════════════════════════════════════════
def job_order_execution():
    sg_now_str = _sg_now().strftime("%H:%M SGT")
    logger.info(f"[JOB3] ── Order Execution @ {sg_now_str} ─────────────────────────────")

    if _client is None:
        logger.error("[JOB3] CLOB client not initialised — skipping.")
        return

    signals = _state.get("signals", {})
    if not signals:
        logger.info("[JOB3] No signals from Job 2 — nothing to execute.")
        return

    # Only pass signals that are genuinely actionable — not gated, not held
    actionable = {l: s for l, s in signals.items() if s.actionable}
    if not actionable:
        logger.info("[JOB3] No actionable signals this cycle.")
        return

    # Alert on (never silently delete) positions open far longer than
    # expected. Job 5's own per-position force_time_exit logic already
    # retries a REAL close every cycle for anything carried over from a
    # prior market_date — this just makes a position stuck despite those
    # retries loud instead of invisible. See db/ledger.py's
    # find_stuck_positions() docstring for the incident this replaced
    # (silent deletion of two still-open, unmanaged real-money positions).
    for stuck in _ledger.find_stuck_positions(ttl_hours=28):
        logger.critical(
            f"[JOB3] STUCK POSITION: {stuck['bracket_label']} ({stuck['token_id']}) "
            f"opened_at={stuck['opened_at']} has been open >28h. Job 5 should be "
            f"retrying its exit every cycle — if this keeps recurring, investigate "
            f"manually (book depth? repeated FOK rejections? a bug?). Polymarket's "
            f"own wallet UI still shows these shares as held."
        )

    trailing_bias = _ledger.fetch_trailing_bias(ICAO)
    engine        = ExecutionEngine(_client, _ledger, VAULT_USD, ICAO)

    for label, signal in actionable.items():
        direction = signal.direction  # "BUY" or "SELL"

        # For BUY YES: Kelly uses best_ask (cost to buy)
        # For SELL YES (NO): Kelly uses effective_ask = 1 - best_bid
        #   because buying NO at implied price (1 - bid) is what we're sizing.
        #   e.g. 33°C bid=0.20 → effective_ask for NO = 1 - 0.20 = 0.80
        #   Kelly then sizes correctly for a position that pays $1 if 33°C does NOT occur.
        if direction == "BUY":
            effective_ask = signal.market_price.best_ask
        else:
            effective_ask = 1.0 - signal.market_price.best_bid

        # signal.model_prob is always P(bracket occurs) — i.e. P(YES).
        # Kelly's p must be the win probability of the SIDE BEING SIZED:
        #   BUY YES → wins if the bracket occurs         → p = model_prob
        #   SELL/NO → wins if the bracket does NOT occur  → p = 1 - model_prob
        # Previously model_prob was passed unflipped for SELL too, which fed
        # Kelly/EV the probability of the side we're betting AGAINST — e.g. a
        # NO trade priced favorably (effective_ask cheap) with a small
        # model_prob scored as strongly net-negative instead of net-positive,
        # so genuinely strong NO edges could be silently HOLD'd at the net-EV
        # hurdle before ever reaching execution.
        win_prob = signal.model_prob if direction == "BUY" else 1.0 - signal.model_prob

        if VALIDATION_MODE:
            sizing = compute_validation_size(
                model_prob = win_prob,
                market_ask = effective_ask,
                direction  = direction,
            )
            logger.warning(f"[JOB3] ⚠️  VALIDATION_MODE — {label} [{direction}]: {sizing}")
        else:
            sizing = compute_size(
                model_prob    = win_prob,
                market_ask    = effective_ask,
                vault_usd     = VAULT_USD,
                direction     = direction,
                trailing_bias = trailing_bias,
            )
            logger.info(f"[JOB3] {label} [{direction}]: {sizing}")

        if sizing.verdict == "EXECUTE":
            market_date_for_entry = _state.get("market_date", _sg_now().strftime("%Y-%m-%d"))
            filled = engine.execute(signal, sizing, market_date=market_date_for_entry)
            if filled:
                logger.info(
                    f"[JOB3] ✓ Position opened: {label} "
                    f"{'YES' if direction == 'BUY' else 'NO'} ${sizing.size_usd:.2f}"
                )
            else:
                logger.warning(f"[JOB3] ✗ Execution failed or rejected: {label} [{direction}]")
        else:
            logger.info(f"[JOB3] Sizing HOLD for {label} [{direction}]: {sizing.reason}")


# ══════════════════════════════════════════════════════════════════════════════
# JOB 4 — Settlement Check (every 15 min, 24/7)
# ══════════════════════════════════════════════════════════════════════════════
def job_settlement_check():
    sg_now_str = _sg_now().strftime("%H:%M SGT")
    logger.info(f"[JOB4] ── Settlement Check @ {sg_now_str} ───────────────────")

    engine       = SettlementEngine(_ledger, ICAO)
    model_mu     = _state.get("model_mu", 31.5)
    today        = _state.get("market_date", _sg_now().strftime("%Y-%m-%d"))
    yesterday    = (datetime.datetime.strptime(today, "%Y-%m-%d")
                     - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # Check BOTH dates every cycle. Now that trading runs 24/7, a position
    # opened late yesterday can still be resolving in the early hours of
    # today, after market_date has already rolled over — that outcome was
    # previously never re-checked once "today" moved on. Safe to run both
    # every cycle: has_calibration_for_date (db/ledger.py) makes each write
    # idempotent regardless of how many times or how many dates we check.
    # Minor accepted tradeoff: Task A (resolution polling) inside engine.run()
    # checks ALL open positions regardless of market_date, so this doubles
    # that polling per cycle — negligible cost, correctness unaffected.
    for market_date in (today, yesterday):
        results = engine.run(model_mu=model_mu, market_date=market_date)
        logger.info(
            f"[JOB4] date={results['date']} "
            f"checked={results['positions_checked']} "
            f"settled={results['positions_settled']} "
            f"actual_temp={results['actual_temp']} "
            f"calibration={'✓' if results['calibration_logged'] else '✗'}"
        )



# ══════════════════════════════════════════════════════════════════════════════
# JOB 5 — Position Monitor (every 5 min, 08:05–15:55 SGT)
# ══════════════════════════════════════════════════════════════════════════════
def job_position_monitor():
    logger.info("[JOB5] ── Position Monitor ───────────────────────────────────")

    if _client is None:
        logger.error("[JOB5] CLOB client not initialised — skipping.")
        return

    open_positions = _ledger.get_open_positions()
    if not open_positions:
        logger.info("[JOB5] No open positions.")
        return

    model_probs = _state.get("model_probs", {})
    if not model_probs:
        logger.warning("[JOB5] No model probs in state — Job 2 may not have run yet.")
        return

    monitor = PositionMonitor(
        client         = _client,
        ledger         = _ledger,
        edge_threshold = EDGE_THRESHOLD,
        trail_pct      = TRAIL_PCT,
        icao           = ICAO,
    )
    market_date = _state.get("market_date", _sg_now().strftime("%Y-%m-%d"))
    results = monitor.run(model_probs, market_date=market_date)

    exits_filled   = [r for r in results if r["filled"]]
    exits_failed   = [r for r in results if not r["filled"]]
    total_pnl      = sum(r["realised_pnl"] for r in exits_filled)
    date           = _state.get("market_date", _sg_now().strftime("%Y-%m-%d"))
    daily_pnl      = _ledger.daily_pnl(date)

    logger.info(
        f"[JOB5] Monitor complete: {len(open_positions)} checked | "
        f"{len(exits_filled)} exited | {len(exits_failed)} failed | "
        f"Cycle P&L={total_pnl:+.4f} | Day P&L={daily_pnl:+.4f}"
    )

    for r in exits_filled:
        pnl_pct    = r["realised_pnl"] / r["size_usd"] * 100
        trail_str  = f" peak={r['peak_price']:.4f}" if r.get("peak_price") else ""
        logger.info(
            f"[JOB5]   ✓ {r['label']} [{r['direction']}] {r['reason']}"
            f"{trail_str} exit={r['exit_price']:.4f} "
            f"P&L={r['realised_pnl']:+.4f} ({pnl_pct:+.1f}%)"
        )
    for r in exits_failed:
        logger.warning(f"[JOB5]   ✗ {r['label']} exit failed: {r['reason']}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global _client

    logger.info("═" * 60)
    logger.info("  HERMES v4.6 — WSSS Weather Bracket Trader (round-the-clock)")
    logger.info(
        f"  Vault: ${VAULT_USD:.0f} | Edge threshold: {EDGE_THRESHOLD*100:.0f}% | "
        f"Max edge magnitude: {MAX_EDGE_MAGNITUDE*100:.0f}%"
    )
    if VALIDATION_MODE:
        logger.warning("  ⚠️  VALIDATION_MODE ACTIVE — all trades forced to $1, EV gating bypassed")
        logger.warning("  ⚠️  Set VALIDATION_MODE=false in .env to resume normal Kelly sizing")
    logger.info("═" * 60)

    # Warn if vault/cap/floor collapse the sizing range (see sizing.py)
    check_sizing_config(VAULT_USD)

    # Initialise CLOB client once — shared across all jobs
    try:
        _client = build_client()
        logger.info("[INIT] CLOB client authenticated ✓")
    except Exception as e:
        logger.error(f"[INIT] CLOB client failed: {e}")
        logger.error("[INIT] Check POLYMARKET_PRIVATE_KEY / CLOB_* env vars.")
        sys.exit(1)

    # Sync the CLOB's internal balance/allowance cache for the deposit wallet
    # against actual on-chain state. Without this, order placement can fail
    # with "not enough balance / allowance ... balance: 0" even when the
    # wallet is genuinely funded — confirmed live: the cache does not track
    # on-chain changes automatically and has gone stale across ordinary bot
    # restarts (deploys, VALIDATION_MODE toggles), not just after funding.
    # Previously required a manual `python sync_balance.py` run after every
    # restart; doing it here means that's no longer something an operator
    # has to remember. Best-effort: log and continue on failure rather than
    # blocking startup, since sync_balance.py remains available as a manual
    # fallback and a transient failure here shouldn't take the whole bot down.
    try:
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
        _client.update_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        logger.info("[INIT] Balance/allowance cache synced ✓")
    except Exception as e:
        logger.warning(
            f"[INIT] Balance/allowance sync failed: {e} — orders may fail with "
            f"a stale-balance error until 'python sync_balance.py' is run manually."
        )

    # Run discovery immediately on startup so jobs 2/3 have tokens from the start
    logger.info("[INIT] Running initial market discovery...")
    job_market_discovery()

    # ── Scheduler setup ────────────────────────────────────────────────────────
    executors = {"default": ThreadPoolExecutor(max_workers=3)}
    scheduler = BlockingScheduler(executors=executors, timezone="Asia/Singapore")

    # Explicit timezone for all jobs — avoids UTC fallback on VPS without pytz
    _SGT = "Asia/Singapore"

    # Job 1 — Market discovery: every 20 min, 24/7 — self-healing.
    # Old design fired once at 07:30 SGT with no retry; a single Gamma
    # hiccup at that exact minute left the bot with zero tokens for a full
    # day (confirmed root cause of the Jul 2-3 zero-trade incident).
    scheduler.add_job(
        job_market_discovery,
        trigger   = "cron",
        minute    = "0,20,40",
        timezone  = _SGT,
        id        = "market_discovery",
        name      = "Market Discovery",
        max_instances = 1,
    )

    # Job 2 — Signal scan: every 15 min, 24/7.
    # Market quality gates (liquidity floor, spread cap in core/edge.py)
    # already suppress bad signals on thin overnight books — a clock
    # window isn't needed for that, and was excluding legitimate trading
    # hours (markets launch the evening before and trade continuously).
    scheduler.add_job(
        job_signal_scan,
        trigger   = "cron",
        minute    = "0,15,30,45",
        timezone  = _SGT,
        id        = "signal_scan",
        name      = "Signal Scan",
        max_instances = 1,
    )

    # Job 3 — Execution: every 15 min, 24/7, offset +2 min from Job 2
    # so _state["signals"] is always freshly computed before Job 3 reads it.
    scheduler.add_job(
        job_order_execution,
        trigger   = "cron",
        minute    = "2,17,32,47",
        timezone  = _SGT,
        id        = "order_execution",
        name      = "Order Execution",
        max_instances = 1,
    )

    # Job 4 — Settlement: every 15 min, 24/7. Checks both today's and
    # yesterday's market_date every cycle (see job_settlement_check).
    scheduler.add_job(
        job_settlement_check,
        trigger   = "cron",
        minute    = "5,20,35,50",
        timezone  = _SGT,
        id        = "settlement_check",
        name      = "Settlement Check",
        max_instances = 1,
    )

    # Job 5 — Position monitor: every 5 min, 24/7.
    # Previously windowed to hour="8-15" SGT, which never overlapped
    # position_monitor.py's own HARD_EXIT_HOUR_SGT=16 check — the 16:00
    # hard time-exit could never fire. Running 24/7 fixes that AND gives
    # continuous trailing-stop/stop-loss protection to positions opened
    # at any hour under Jobs 2/3's new round-the-clock schedule.
    scheduler.add_job(
        job_position_monitor,
        trigger   = "cron",
        minute    = "*/5",
        timezone  = _SGT,
        id        = "position_monitor",
        name      = "Position Monitor",
        max_instances = 1,
    )

    # Graceful shutdown on SIGTERM / SIGINT
    def _shutdown(signum, frame):
        logger.info("[SHUTDOWN] Signal received — stopping scheduler.")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    logger.info("[INIT] Scheduler armed. Jobs registered:")
    for job in scheduler.get_jobs():
        logger.info(f"  {job.name}: {job.trigger}")

    # Verify timezone resolution — fail loudly if SGT can't be loaded
    try:
        import pytz
        sgt = pytz.timezone("Asia/Singapore")
        now_sgt = datetime.datetime.now(sgt)
        logger.info(f"[INIT] Timezone verified: Asia/Singapore = {now_sgt.strftime('%H:%M %Z')}")
    except Exception as e:
        logger.error(
            f"[INIT] ⚠️  pytz timezone error: {e} — "
            "job schedules may run in UTC instead of SGT. "
            "Fix: pip install pytz tzdata"
        )

    logger.info("[INIT] Starting. Ctrl+C to stop.")
    scheduler.start()


if __name__ == "__main__":
    main()
