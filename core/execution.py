"""
core/execution.py — H3: CLOB order execution
Ported from Hermes v4.3 with all BUG fixes applied.

Execution flow per bracket:
  1. Compute-time VWAP from order book (ask side for BUY, bid side for SELL/NO)
  2. Kelly sizing decision
  3. Pre-execution VWAP revalidation (re-fetch book)
  4. Drift check: abort if book moved > VWAP_DRIFT_TOLERANCE
  5. Build and sign MarketOrderArgs (FOK)
  6. post_order() → parse fill status
  7. Record position in DB only on confirmed fill

BUY (YES):  Side.BUY  on YES token — fills from ask side
SELL (NO):  Side.SELL on YES token — fills into bid side
            Economically equivalent to buying NO shares.
            Kelly math uses effective_ask = 1 - best_bid.

All CLOB operations are synchronous (py-clob-client is not async).
They are called from APScheduler jobs running in a thread pool.
"""

import logging
import os
from typing import Any, Dict, Optional

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import ApiCreds, MarketOrderArgs, OrderType
from py_clob_client_v2.order_builder.constants import BUY as _CLOB_BUY, SELL as _CLOB_SELL
from py_clob_client_v2.constants import POLYGON

from db.ledger import Ledger
from core.edge import EdgeSignal
from core.sizing import SizingResult

logger = logging.getLogger("hermes.execution")

VWAP_DRIFT_TOLERANCE = 0.03   # Abort if book moves > 3% between compute and exec


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

    accumulated_usd    = 0.0
    accumulated_shares = 0.0

    for ask in raw_asks:
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


def _extract_vwap_bid(orderbook: Any, max_receive_usd: float) -> Optional[float]:
    """
    Walk bid side of order book to compute VWAP for SELL (NO) execution.
    For a SELL order we fill into bids (highest first).
    Returns the bid-side VWAP — the average price we receive per share sold.

    max_receive_usd: target USD proceeds from selling YES shares.
    BUG-4 handling: same dual SDK-object/dict support as _extract_vwap_ask.
    """
    raw_bids = getattr(orderbook, "bids", None)
    if raw_bids is None:
        raw_bids = orderbook.get("bids", []) if isinstance(orderbook, dict) else []

    if not raw_bids:
        return None

    # Sort bids descending (highest price first = best fills)
    def _bid_price(b):
        return float(b.price) if hasattr(b, "price") else float(b.get("price", 0.0))

    sorted_bids = sorted(raw_bids, key=_bid_price, reverse=True)

    accumulated_usd    = 0.0
    accumulated_shares = 0.0

    for bid in sorted_bids:
        if hasattr(bid, "price"):
            price = float(bid.price)
            size  = float(bid.size)
        else:
            price = float(bid.get("price", 0.0))
            size  = float(bid.get("size",  0.0))

        if price <= 0:
            continue

        level_proceeds = price * size
        if accumulated_usd + level_proceeds >= max_receive_usd:
            remaining           = max_receive_usd - accumulated_usd
            accumulated_shares += remaining / price
            accumulated_usd    += remaining
            break
        accumulated_shares += size
        accumulated_usd    += level_proceeds

    if accumulated_shares == 0:
        return None
    return round(accumulated_usd / accumulated_shares, 5)


def _parse_fill_status(response: Any, label: str) -> bool:
    """
    BUG-3: Inspect post_order() response for confirmed fill.
    Returns True only on MATCHED/FILLED with size_matched > 0.
    Never calls record_position() on a rejected FOK.
    """
    if response is None:
        logger.warning(f"[EXEC] {label}: None response from post_order()")
        return False

    if isinstance(response, dict):
        status   = str(response.get("status", "")).upper()
        matched  = float(response.get("size_matched", 0.0) or 0.0)
        if status in ("MATCHED", "FILLED") and matched > 0:
            logger.info(f"[EXEC] ✓ {label}: filled status={status} matched={matched:.4f}")
            return True
        logger.warning(f"[EXEC] ✗ {label}: FOK rejected status={status} matched={matched:.4f}")
        return False

    if hasattr(response, "status"):
        status  = str(getattr(response, "status", "")).upper()
        matched = float(getattr(response, "size_matched", 0.0) or 0.0)
        if status in ("MATCHED", "FILLED") and matched > 0:
            logger.info(f"[EXEC] ✓ {label}: filled status={status}")
            return True
        logger.warning(f"[EXEC] ✗ {label}: FOK rejected status={status}")
        return False

    logger.error(f"[EXEC] {label}: unknown response type {type(response)}: {response}")
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
        Execute a single bracket trade — BUY YES or SELL YES (NO).

        BUY:  signal.direction == "BUY"
              → Side.BUY  on YES token, fills from ask side
              → position records entry as long YES

        SELL: signal.direction == "SELL"
              → Side.SELL on YES token, fills into bid side
              → economically equivalent to buying NO
              → position records entry as short YES / long NO
              → entry_price stored as bid VWAP (what we receive per share)

        market_date: SGT calendar date this trade is being opened under.
        Passed through to record_position() so position_monitor.py's
        per-position time-exit logic can compare SGT dates directly
        (see db/ledger.py record_position docstring for why this matters).

        Returns True on confirmed fill, False on rejection or abort.
        """
        label     = signal.bracket_label
        token_id  = signal.token_id
        direction = signal.direction  # "BUY" or "SELL"

        if direction not in ("BUY", "SELL"):
            logger.error(f"[EXEC] {label}: unknown direction '{direction}' — abort")
            return False

        if self.ledger.is_position_open(token_id):
            logger.info(f"[EXEC] {label}: position already open — skip")
            return False

        if sizing.verdict != "EXECUTE":
            logger.info(f"[EXEC] {label}: sizing says HOLD — skip")
            return False

        # ── Compute-time VWAP ─────────────────────────────────────────────────
        book_1 = self.client.get_order_book(token_id)

        if _is_ghost_book(book_1):
            logger.error(
                f"[EXEC] {label} {direction}: ghost book detected at compute-time "
                f"(get_order_book issue #180) — abort, will retry next cycle"
            )
            return False

        if direction == "BUY":
            vwap_compute = _extract_vwap_ask(book_1, sizing.size_usd)
        else:
            # For SELL: size_usd is the USD value of shares to sell
            vwap_compute = _extract_vwap_bid(book_1, sizing.size_usd)

        if not vwap_compute:
            logger.warning(f"[EXEC] {label} {direction}: no book depth at compute — abort")
            return False

        # ── PM-4: Pre-execution VWAP revalidation ─────────────────────────────
        book_2 = self.client.get_order_book(token_id)

        if _is_ghost_book(book_2):
            logger.error(
                f"[EXEC] {label} {direction}: ghost book detected at revalidation "
                f"(get_order_book issue #180) — abort, will retry next cycle"
            )
            return False

        if direction == "BUY":
            vwap_exec = _extract_vwap_ask(book_2, sizing.size_usd)
        else:
            vwap_exec = _extract_vwap_bid(book_2, sizing.size_usd)

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

        # ── Order amount semantics (confirmed from Polymarket CLOB docs) ──────
        # Market BUY:  amount = quote notional in USD (dollars to spend)
        # Market SELL: amount = number of SHARES to sell (base units)
        # Passing a USD figure as the SELL amount would sell the wrong quantity
        # (e.g. amount=10 sells 10 shares ≈ $2.90, not $10 of exposure).
        # For SELL we convert: shares = size_usd / vwap_exec (bid VWAP = $/share).
        if direction == "BUY":
            order_amount = round(sizing.size_usd, 2)         # USD
        else:
            order_amount = round(sizing.size_usd / vwap_exec, 2)  # shares

        # ── Worst-price limit (slippage protection), NOT a target price ───────
        # Polymarket docs: "The price field on market orders acts as a worst-price
        # limit, not a target execution price." Passing the exact VWAP means any
        # adverse tick between revalidation and match rejects the whole FOK.
        # Pad by the drift tolerance so a small move still fills:
        #   BUY  → allow paying UP TO vwap*(1+tol)  (higher ask is worse)
        #   SELL → allow receiving DOWN TO vwap*(1-tol)  (lower bid is worse)
        if direction == "BUY":
            limit_price = round(min(0.99, vwap_exec * (1 + VWAP_DRIFT_TOLERANCE)), 2)
        else:
            limit_price = round(max(0.01, vwap_exec * (1 - VWAP_DRIFT_TOLERANCE)), 2)

        # ── Build order — FOK passed to post_order, not the constructor ───────
        clob_side  = _CLOB_BUY if direction == "BUY" else _CLOB_SELL
        order_args = MarketOrderArgs(
            token_id = token_id,
            amount   = order_amount,
            price    = limit_price,
            side     = clob_side,
        )
        signed_order = self.client.create_market_order(order_args)

        # ── Post order — FOK: fill entirely at/inside limit, or reject ────────
        response = self.client.post_order(signed_order, OrderType.FOK)

        # ── Parse fill and record ──────────────────────────────────────────────
        if _parse_fill_status(response, label):
            # Store direction in label suffix so DB distinguishes YES/NO positions
            position_label = f"{label}:{'YES' if direction == 'BUY' else 'NO'}"
            self.ledger.record_position(
                token_id    = token_id,
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
