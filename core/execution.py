"""
core/execution.py — H3: CLOB order execution
Ported from Hermes v4.3 with all BUG fixes applied.

Execution flow per bracket:
  1. Compute-time VWAP from the order book's ask side (both directions are
     a BUY — see below)
  2. Kelly sizing decision
  3. Pre-execution VWAP revalidation (re-fetch book)
  4. Drift check: abort if book moved > VWAP_DRIFT_TOLERANCE
  5. Build and sign MarketOrderArgs (FOK)
  6. post_order() → parse fill status
  7. Record position in DB only on confirmed fill

BUY (YES):  Side.BUY on the YES token — fills from its ask side.
SELL (NO):  Side.BUY on the NO token — fills from its ask side.
            Polymarket's CLOB requires holding a token to sell it (no naked
            short-sell endpoint — confirmed by the UI only exposing Buy Yes/
            Buy No). A "SELL" signal therefore opens a NO position the same
            way a "BUY" signal opens a YES position: a plain market BUY, just
            on the NO token's own token_id. Kelly math still uses
            effective_ask = 1 - best_bid (from the YES book) as a sizing
            estimate; the actual VWAP/order here comes from the NO token's
            own live book.

All CLOB operations are synchronous (py-clob-client is not async).
They are called from APScheduler jobs running in a thread pool.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import ApiCreds, MarketOrderArgs, OrderType, BalanceAllowanceParams, AssetType
from py_clob_client_v2.order_builder.constants import BUY as _CLOB_BUY
from py_clob_client_v2.constants import POLYGON

from db.ledger import Ledger
from core.edge import EdgeSignal
from core.sizing import SizingResult

logger = logging.getLogger("hermes.execution")

VWAP_DRIFT_TOLERANCE = 0.03   # Abort if book moves > 3% between compute and exec

# Pre-order balance-propagation poll: how many times to re-check
# get_balance_allowance() after update_balance_allowance() before giving up
# and posting anyway. 3 attempts * 0.5s = up to 1.5s added latency in the
# worst case, negligible next to the FOK's own worst-price limit protection.
BALANCE_POLL_ATTEMPTS   = 3
BALANCE_POLL_DELAY_SEC  = 0.5


def build_client() -> ClobClient:
    """
    Instantiate py-clob-client-v2 with Level 2 auth.
    Called once at scheduler startup and reused across jobs.

    Migrated from py-clob-client (v1) after Polymarket's CLOB V2 go-live on
    2026-04-28 made v1-signed orders rejected in production ("invalid order
    version, please use the latest clob-client" / order_version_mismatch —
    the v1 SDK is archived with no fix forthcoming). v2 is close to a
    drop-in replacement for this codebase's usage:
      - ClobClient constructor, BUY/SELL constants, MarketOrderArgs fields,
        create_market_order()/post_order() two-step flow — all unchanged.
      - Only real rename: create_or_derive_api_creds() → create_or_derive_api_key().
      - funder = the Polymarket deposit wallet address (proxy wallet address,
        visible at polymarket.com/settings — NOT the same as private key address
        for email/Magic wallet accounts)
      - signature_type = 1 for email/Magic proxy wallets (most common),
                         0 for EOA/MetaMask (direct wallet)
      - set_api_creds() binds L2 credentials after construction

    POLYMARKET_FUNDER env var = your deposit wallet address from polymarket.com/settings
    POLYMARKET_SIGNATURE_TYPE = 0 (EOA/MetaMask) or 1 (email/Magic proxy) — default 1
    """
    host        = os.getenv("CLOB_API_HOST", "https://clob.polymarket.com")
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    funder      = os.getenv("POLYMARKET_FUNDER")          # deposit wallet address

    sig_type_raw = os.getenv("POLYMARKET_SIGNATURE_TYPE", "1")
    try:
        signature_type = int(sig_type_raw)
    except ValueError:
        # Same guard as generate_creds.py's _parse_signature_type(). Without
        # this, a bad value here raised an uncaught ValueError with no clue
        # as to the cause. The most likely real-world trigger: this value
        # was corrupted by systemd's EnvironmentFile= parser, which does NOT
        # strip inline "# comment" text the way python-dotenv does — if
        # .env ever has "POLYMARKET_SIGNATURE_TYPE=1 # some comment" on one
        # line, systemd passes the ENTIRE remainder (comment included) as
        # the literal value. .env.example is written to avoid this, but a
        # manual edit could reintroduce it.
        raise ValueError(
            f"POLYMARKET_SIGNATURE_TYPE must be an integer, got: '{sig_type_raw}'. "
            f"If this looks like it has trailing text after the number, check "
            f".env for an inline '# comment' on that line — systemd's "
            f"EnvironmentFile parser does not strip those (unlike python-dotenv), "
            f"so the comment text gets appended to the value. Put comments on "
            f"their own line instead."
        )

    creds = ApiCreds(
        api_key        = os.getenv("CLOB_API_KEY",     ""),
        api_secret     = os.getenv("CLOB_SECRET",      ""),
        api_passphrase = os.getenv("CLOB_PASS_PHRASE", ""),
    )

    client = ClobClient(
        host           = host,
        chain_id       = POLYGON,
        key            = private_key,
        creds          = creds,
        signature_type = signature_type,
        funder         = funder,
    )
    client.set_api_creds(client.create_or_derive_api_key())
    return client


def _best_bid_ask(orderbook: Any) -> Optional[tuple]:
    """Extract raw best bid / best ask from an order book (SDK object or dict)."""
    raw_bids = getattr(orderbook, "bids", None)
    if raw_bids is None:
        raw_bids = orderbook.get("bids", []) if isinstance(orderbook, dict) else []
    raw_asks = getattr(orderbook, "asks", None)
    if raw_asks is None:
        raw_asks = orderbook.get("asks", []) if isinstance(orderbook, dict) else []
    if not raw_bids or not raw_asks:
        return None

    def _price(x):
        return float(x.price) if hasattr(x, "price") else float(x.get("price", 0.0))

    best_bid = max(_price(b) for b in raw_bids)
    best_ask = min(_price(a) for a in raw_asks)
    return best_bid, best_ask


def _is_ghost_book(orderbook: Any) -> bool:
    """
    Detect the stale "ghost market" snapshot documented in
    github.com/Polymarket/py-clob-client issue #180: get_order_book()
    intermittently returns bid=0.01 / ask=0.99 for active, liquid markets
    while get_price() stays accurate. The py-clob-client repo is archived
    (May 2026) with no fix forthcoming, so this must be handled here.

    A 0.01/0.99 book is almost never real for a weather bracket mid-cycle —
    treat it as unusable rather than trading against it. Critically, since
    execution fetches the book TWICE (compute + revalidation) and a ghost
    snapshot can persist unchanged across both calls, the VWAP drift check
    alone would NOT catch this (drift=0% when both reads return the same
    stale numbers) — a dedicated check is required.
    """
    ba = _best_bid_ask(orderbook)
    if ba is None:
        return False
    best_bid, best_ask = ba
    return best_bid <= 0.02 and best_ask >= 0.98


def _extract_vwap_ask(orderbook: Any, max_spend_usd: float) -> Optional[float]:
    """
    Walk ask side of order book to compute VWAP for a given USD budget.
    BUG-4: handles both SDK OrderBookSummary objects and plain dicts.
    size = shares (contracts), price = USD/share.
    """
    raw_asks = getattr(orderbook, "asks", None)
    if raw_asks is None:
        raw_asks = orderbook.get("asks", []) if isinstance(orderbook, dict) else []

    if not raw_asks:
        return None

    # Sort ascending (cheapest ask first) before walking — the API does not
    # guarantee ask ordering. This function was missing that sort, so on an
    # unsorted book it could walk into a high-priced ask before the genuine
    # best one. Confirmed against a real fill: this produced a computed
    # VWAP of 0.999 for a BUY that actually filled at ~0.07 (the real best
    # ask) — the worst-price limit derived from that bad VWAP happened to be
    # generous enough that the fill still executed at the true, better
    # price, but the bug corrupted every downstream EV/edge number logged
    # for that trade and could, for a SELL, produce an unsafe (too-lenient)
    # worst-price limit instead.
    def _ask_price(a):
        return float(a.price) if hasattr(a, "price") else float(a.get("price", 0.0))

    sorted_asks = sorted(raw_asks, key=_ask_price)

    accumulated_usd    = 0.0
    accumulated_shares = 0.0

    for ask in sorted_asks:
        if hasattr(ask, "price"):
            price = float(ask.price)
            size  = float(ask.size)
        else:
            price = float(ask.get("price", 0.0))
            size  = float(ask.get("size",  0.0))

        if price <= 0:
            continue

        level_cost = price * size
        if accumulated_usd + level_cost >= max_spend_usd:
            remaining           = max_spend_usd - accumulated_usd
            accumulated_shares += remaining / price
            accumulated_usd    += remaining
            break
        accumulated_shares += size
        accumulated_usd    += level_cost

    if accumulated_shares == 0:
        return None
    return round(accumulated_usd / accumulated_shares, 5)


def _parse_fill_status(response: Any, label: str) -> bool:
    """
    BUG-3: Inspect post_order() response for confirmed fill.
    Never calls record_position() on a rejected FOK.

    Confirmed against a REAL matched order (see the PR that fixed this):
    py_clob_client_v2's post_order() response for a filled FOK has NO
    "size_matched" key at all — e.g.
        {'status': 'matched', 'success': True,
         'takingAmount': '14.285713', 'makingAmount': '0.999999',
         'transactionsHashes': ['0x...'], 'orderID': '0x...', 'errorMsg': ''}
    The old code read response.get("size_matched", 0.0), which always
    defaulted to 0.0 for this real shape, so `matched > 0` was always False
    — meaning EVERY successful fill was logged and treated as a rejection,
    and record_position() never ran for a single real trade in this bot's
    history (confirmed: exit_log/open_positions were both completely empty
    despite a verified on-chain fill). Fixed to also recognize
    takingAmount/makingAmount and the "success" flag, and to fall back to
    the old size_matched key if a different response shape ever supplies it.
    """
    if response is None:
        logger.warning(f"[EXEC] {label}: None response from post_order()")
        return False

    if isinstance(response, dict):
        data = response
    elif hasattr(response, "status"):
        data = {
            "status":             getattr(response, "status", None),
            "success":            getattr(response, "success", None),
            "size_matched":       getattr(response, "size_matched", None),
            "takingAmount":       getattr(response, "takingAmount", None),
            "makingAmount":       getattr(response, "makingAmount", None),
            "transactionsHashes": getattr(response, "transactionsHashes", None),
        }
    else:
        logger.error(f"[EXEC] {label}: unknown response type {type(response)}: {response}")
        return False

    status  = str(data.get("status", "")).upper()
    success = data.get("success")

    matched = data.get("size_matched")
    if matched is None:
        matched = data.get("takingAmount") or data.get("makingAmount")
    try:
        matched = float(matched) if matched is not None else 0.0
    except (TypeError, ValueError):
        matched = 0.0

    if status in ("MATCHED", "FILLED") and success is not False and matched > 0:
        logger.info(f"[EXEC] ✓ {label}: filled status={status} matched={matched:.4f}")
        return True

    logger.warning(
        f"[EXEC] ✗ {label}: FOK rejected status={status} success={success} matched={matched:.4f}"
    )
    return False


class ExecutionEngine:
    def __init__(self, client: ClobClient, ledger: Ledger, vault_usd: float, icao: str = "WSSS"):
        self.client    = client
        self.ledger    = ledger
        self.vault_usd = vault_usd
        self.icao      = icao

    def execute(
        self,
        signal: EdgeSignal,
        sizing: SizingResult,
        market_date: str = "",
    ) -> bool:
        """
        Execute a single bracket trade — BUY YES or BUY NO.

        BUY:  signal.direction == "BUY"
              → Side.BUY on the YES token, fills from its ask side
              → position records entry as long YES

        SELL: signal.direction == "SELL"  (opens a NO position)
              → Side.BUY on the NO token (its own token_id), fills from its
                ask side — Polymarket has no naked short-sell; opening NO
                exposure means actually buying the NO token, same as the UI's
                "Buy No" button
              → position records entry as long NO
              → entry_price stored as the NO token's own ask VWAP

        market_date: SGT calendar date this trade is being opened under.
        Passed through to record_position() so position_monitor.py's
        per-position time-exit logic can compare SGT dates directly
        (see db/ledger.py record_position docstring for why this matters).

        Returns True on confirmed fill, False on rejection or abort.
        """
        label     = signal.bracket_label
        direction = signal.direction  # "BUY" or "SELL"

        if direction not in ("BUY", "SELL"):
            logger.error(f"[EXEC] {label}: unknown direction '{direction}' — abort")
            return False

        # Both directions are a plain BUY now — just on different tokens.
        exec_token_id = signal.token_id if direction == "BUY" else signal.no_token_id
        if not exec_token_id:
            logger.error(
                f"[EXEC] {label} {direction}: no NO token_id available for this "
                f"bracket (stale/pre-migration token matrix?) — abort"
            )
            return False

        if self.ledger.is_position_open(exec_token_id):
            logger.info(f"[EXEC] {label}: position already open — skip")
            return False

        if sizing.verdict != "EXECUTE":
            logger.info(f"[EXEC] {label}: sizing says HOLD — skip")
            return False

        # ── Compute-time VWAP ─────────────────────────────────────────────────
        book_1 = self.client.get_order_book(exec_token_id)

        if _is_ghost_book(book_1):
            logger.error(
                f"[EXEC] {label} {direction}: ghost book detected at compute-time "
                f"(get_order_book issue #180) — abort, will retry next cycle"
            )
            return False

        vwap_compute = _extract_vwap_ask(book_1, sizing.size_usd)

        if not vwap_compute:
            logger.warning(f"[EXEC] {label} {direction}: no book depth at compute — abort")
            return False

        # ── PM-4: Pre-execution VWAP revalidation ─────────────────────────────
        book_2 = self.client.get_order_book(exec_token_id)

        if _is_ghost_book(book_2):
            logger.error(
                f"[EXEC] {label} {direction}: ghost book detected at revalidation "
                f"(get_order_book issue #180) — abort, will retry next cycle"
            )
            return False

        vwap_exec = _extract_vwap_ask(book_2, sizing.size_usd)

        if not vwap_exec:
            logger.warning(f"[EXEC] {label} {direction}: book gone before execution — abort")
            return False

        drift = abs(vwap_exec - vwap_compute) / vwap_compute
        if drift > VWAP_DRIFT_TOLERANCE:
            logger.warning(
                f"[EXEC] {label} {direction}: VWAP drifted {drift:.1%} "
                f"({vwap_compute:.4f}→{vwap_exec:.4f}) — abort"
            )
            return False

        logger.info(
            f"[EXEC] 🔥 {direction} {label} | VWAP={vwap_exec:.4f} | "
            f"${sizing.size_usd:.2f} | net EV={sizing.net_ev*100:+.2f}%"
        )

        # ── Order amount: both directions are a market BUY now, so amount is
        # always the USD notional to spend (confirmed from Polymarket CLOB
        # docs: market BUY amount = quote notional in USD).
        order_amount = round(sizing.size_usd, 2)

        # ── Worst-price limit (slippage protection), NOT a target price ───────
        # Polymarket docs: "The price field on market orders acts as a worst-price
        # limit, not a target execution price." Passing the exact VWAP means any
        # adverse tick between revalidation and match rejects the whole FOK.
        # Pad by the drift tolerance so a small move still fills — allow paying
        # UP TO vwap*(1+tol) (a higher ask is worse for a buyer).
        limit_price = round(min(0.99, vwap_exec * (1 + VWAP_DRIFT_TOLERANCE)), 2)

        # ── Build order — FOK passed to post_order, not the constructor ───────
        order_args = MarketOrderArgs(
            token_id = exec_token_id,
            amount   = order_amount,
            price    = limit_price,
            side     = _CLOB_BUY,
        )
        signed_order = self.client.create_market_order(order_args)

        # ── Refresh balance/allowance cache immediately before posting ────────
        # The CLOB's internal balance/allowance cache does not stay in sync with
        # on-chain state on its own — it was going stale mid-process (not just
        # across restarts), causing "balance: 0" rejections on a long-running
        # process even after the startup sync (see scheduler.py [INIT] sync,
        # PR #12) succeeded hours earlier.
        #
        # A sync call alone is NOT enough: production logs showed
        # update_balance_allowance() return 200 OK, immediately followed
        # (~250ms later) by post_order() still rejecting with "balance: 0" —
        # while a separate get_balance_allowance() call made ~10s later (via
        # the dashboard) read the real, nonzero, on-chain-correct balance.
        # update_balance_allowance() only TRIGGERS a re-check on Polymarket's
        # side; its 200 response doesn't guarantee the result has propagated
        # to what post_order() actually reads by the time it returns. Poll
        # get_balance_allowance() briefly afterward and wait for a genuinely
        # nonzero read (or give up after a bounded number of attempts) instead
        # of racing straight into post_order() on the sync call's response alone.
        try:
            self.client.update_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            for attempt in range(1, BALANCE_POLL_ATTEMPTS + 1):
                check = self.client.get_balance_allowance(
                    BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                )
                raw_balance = check.get("balance") if isinstance(check, dict) else getattr(check, "balance", None)
                if raw_balance is not None and float(raw_balance) > 0:
                    break
                logger.warning(
                    f"[EXEC] {label}: balance still reads 0 after sync "
                    f"(propagation check {attempt}/{BALANCE_POLL_ATTEMPTS}) — waiting"
                )
                if attempt < BALANCE_POLL_ATTEMPTS:
                    time.sleep(BALANCE_POLL_DELAY_SEC)
        except Exception as e:
            logger.warning(f"[EXEC] {label}: pre-order balance/allowance sync failed: {e}")

        # ── Post order — FOK: fill entirely at/inside limit, or reject ────────
        response = self.client.post_order(signed_order, OrderType.FOK)

        # ── Parse fill and record ──────────────────────────────────────────────
        if _parse_fill_status(response, label):
            # Store direction in label suffix so DB distinguishes YES/NO positions
            position_label = f"{label}:{'YES' if direction == 'BUY' else 'NO'}"
            self.ledger.record_position(
                token_id    = exec_token_id,
                label       = position_label,
                icao        = self.icao,
                entry_price = vwap_exec,
                size_usd    = sizing.size_usd,
                market_date = market_date,
            )
            return True

        logger.warning(
            f"[EXEC] {label} {direction}: FOK rejected — no position recorded. "
            f"Raw: {response}"
        )
        return False
