"""
core/edge.py — N2: Edge calculation
The most important file in the system.

Edge = model_prob - market_implied_probability

market_implied_probability = mid-price of the YES token in the order book
  mid = (best_bid + best_ask) / 2

This is NOT the VWAP. VWAP is used for execution sizing.
Mid-price is used for edge signal.

Direction logic:
  edge > +threshold → BUY YES  (market under-pricing outcome)
  edge < -threshold → SELL YES = BUY NO  (market over-pricing outcome)
  |edge| < threshold → no trade

For NO trades (direction="SELL"):
  Polymarket's CLOB requires holding a token to sell it — there is no naked
  short-sell endpoint (confirmed: the UI itself only exposes Buy Yes/Buy No).
  A NO position is opened and closed by trading the NO outcome token
  directly (its own separate token_id), via a normal BUY to open / SELL to
  close — mechanically identical to a YES position, just on the other token.
  Sizing and quality gates use the NO token's own book.
  NO position pays $1 if outcome does NOT occur.

Edge threshold: 8% (0.08) — set in .env as EDGE_THRESHOLD (raised from 5%
for win-rate selectivity). Stop-loss distance is a separate config
(STOP_LOSS_PCT, core/position_monitor.py) — no longer coupled to this.

Max edge magnitude: 50% (0.50) — set in .env as MAX_EDGE_MAGNITUDE. An edge
this large means the model claims near-certainty against a market pricing
the opposite near-certainty — in practice this has meant the model's stated
uncertainty (sigma) was too tight relative to its own historical accuracy,
not that the bot found free money (see core/model.py's historical_sigma
blending, added alongside this gate for the same underlying issue). Treated
as a data/calibration red flag and gated from execution rather than sized
and traded, even though it would otherwise look like the best signal on the
board.
"""

import os
import datetime
import math
import logging
import requests
from typing import Dict, Optional, Tuple

logger = logging.getLogger("hermes.edge")

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_BOOK_URL     = "https://clob.polymarket.com/book"

EDGE_THRESHOLD     = 0.08  # overridden by env in scheduler; raised from 0.05 for win-rate selectivity
MAX_EDGE_MAGNITUDE = 0.50  # overridden by env in scheduler

# Cheap long-shot brackets (e.g. a 30c YES) need proportionally more shares
# to fill the same $ notional, so the same top-of-book $ liquidity absorbs a
# BUY/entry fine but leaves an exit walking much deeper into the book once
# the position needs to close — and that book can (and does) thin out further
# in the minutes/hours between entry and exit. Confirmed in production: a
# 30°C:YES position entered at 0.05 passed the standard $10 liquidity floor,
# then its STOP_LOSS exit filled at 0.03 — a 40% realized loss against a 10%
# stop-loss target, purely from thin-book slippage on the way out.
MIN_ENTRY_PRICE           = float(os.getenv("MIN_ENTRY_PRICE", "0.20"))
# LOW_PRICE_THRESHOLD (0.25) sits ABOVE MIN_ENTRY_PRICE (0.20), so the band
# between them is live: a token priced 0.20-0.25 clears the hard floor but
# still needs to clear LOW_PRICE_LIQUIDITY_FLOOR ($50, vs the standard $10)
# to be tradeable — a graduated second check, not just a binary cutoff.
LOW_PRICE_THRESHOLD       = float(os.getenv("LOW_PRICE_THRESHOLD", "0.25"))
LOW_PRICE_LIQUIDITY_FLOOR = float(os.getenv("LOW_PRICE_LIQUIDITY_FLOOR", "50.0"))


class MarketPrice:
    """Container for live market price data."""
    def __init__(
        self,
        token_id: str,
        mid_price: float,
        best_bid: float,
        best_ask: float,
        spread: float,
        liquidity_usd: float,
    ):
        self.token_id      = token_id
        self.mid_price     = mid_price    # implied probability
        self.best_bid      = best_bid
        self.best_ask      = best_ask
        self.spread        = spread       # ask - bid
        self.liquidity_usd = liquidity_usd  # estimated from top-of-book
        # Full depth + UTC fetch time, for book_snapshots; empty for Gamma prices.
        self.bids: list = []
        self.asks: list = []
        self.fetched_at = ""

    def __repr__(self):
        return (
            f"MarketPrice(mid={self.mid_price:.4f}, "
            f"bid={self.best_bid:.4f}, ask={self.best_ask:.4f}, "
            f"spread={self.spread:.4f})"
        )


# ── Signal action labels (written to signal_log.action) ──────────────────────
ACTION_BUY       = "SIGNAL_BUY"       # edge >=  threshold → BUY YES
ACTION_SELL      = "SIGNAL_SELL_NO"   # edge <= -threshold → SELL YES / BUY NO
ACTION_HOLD_EDGE = "HOLD_EDGE"        # priced + liquid, but |edge| < threshold
ACTION_SKIP_LIQ  = "SKIP_ILLIQUID"   # top-of-book liquidity too low
ACTION_SKIP_SPRD = "SKIP_SPREAD"     # bid/ask spread > 8c
ACTION_SKIP_EXTREME = "SKIP_EXTREME_EDGE"  # |edge| > sanity cap — likely miscalibration, not opportunity
ACTION_SKIP_LOW_PRICE = "SKIP_LOW_PRICE"  # token priced below MIN_ENTRY_PRICE — thin-book exit slippage risk
ACTION_NO_PRICE  = "NO_PRICE"        # price fetch failed entirely


class EdgeSignal:
    """
    Result of an edge scan for one bracket.
    Always instantiated — never None — so every bracket is logged every cycle.
    gate_reason is non-empty when quality gates blocked the signal.
    action_label is the string written to signal_log.action.

    direction: "BUY"  → buy YES  (model > market by >= threshold)
               "SELL" → sell YES / buy NO  (model < market by >= threshold)
               "NONE" → below threshold or gated
    """
    def __init__(
        self,
        bracket_label:  str,
        token_id:       str,
        model_prob:     float,
        market_price:   Optional[MarketPrice],
        edge:           float,
        edge_threshold: float,
        gate_reason:    str = "",
        no_token_id:    Optional[str] = None,
    ):
        self.bracket_label  = bracket_label
        self.token_id       = token_id
        self.no_token_id    = no_token_id
        self.model_prob     = model_prob
        self.market_price   = market_price
        self.execution_price = market_price  # price of the token actually bought
        self.market_date = ""
        self.scanned_at = None
        self.edge           = edge
        self.edge_threshold = edge_threshold
        self.gate_reason    = gate_reason

        # Direction + actionability
        if gate_reason or market_price is None:
            self.direction  = "NONE"
            self.actionable = False
        elif edge >= edge_threshold:
            self.direction  = "BUY"
            self.actionable = True
        elif edge <= -edge_threshold:
            self.direction  = "SELL"
            self.actionable = True
        else:
            self.direction  = "NONE"
            self.actionable = False

    @property
    def action_label(self) -> str:
        """String written to signal_log.action — describes outcome of this scan."""
        if self.gate_reason:
            return self.gate_reason
        if self.direction == "BUY":
            return ACTION_BUY
        if self.direction == "SELL":
            return ACTION_SELL
        return ACTION_HOLD_EDGE

    def __repr__(self):
        mid  = self.market_price.mid_price if self.market_price else float("nan")
        flag = (f"✓ {self.direction}" if self.actionable
                else f"✗ {self.gate_reason or 'HOLD_EDGE'}")
        return (
            f"EdgeSignal({self.bracket_label}: model={self.model_prob:.3f} "
            f"market={mid:.3f} edge={self.edge:+.3f} [{flag}])"
        )


def book_levels(book, side: str):
    """[(price, size), ...] for one side of a REST dict or SDK book."""
    raw = book.get(side, []) if isinstance(book, dict) else getattr(book, side, None) or []
    return [(float(x["price"]), float(x["size"])) if isinstance(x, dict)
            else (float(x.price), float(x.size)) for x in raw]


def market_price_from_book(token_id: str, book) -> Optional[MarketPrice]:
    """Normalize REST dictionaries and SDK books for the same entry gates."""
    bids, asks = book_levels(book, "bids"), book_levels(book, "asks")
    if not bids or not asks:
        return None
    if any(not math.isfinite(p) or not math.isfinite(s) or not 0 < p < 1 or s <= 0
           for p, s in bids + asks):
        return None
    bid, ask = max(p for p, s in bids), min(p for p, s in asks)
    if ask <= bid or (bid <= 0.02 and ask >= 0.98):
        return None
    liquidity = min(sum(p * s for p, s in sorted(bids, reverse=True)[:3]),
                    sum(p * s for p, s in sorted(asks)[:3]))
    price = MarketPrice(token_id, (bid + ask) / 2, bid, ask, round(ask - bid, 5), liquidity)
    price.bids, price.asks = bids, asks
    price.fetched_at = datetime.datetime.utcnow().isoformat()
    return price


def entry_gate_reason(price: Optional[MarketPrice], min_liquidity_usd: float = 10.0) -> str:
    """Check both entry and potential exit depth on the selected outcome token."""
    if price is None:
        return ACTION_NO_PRICE
    if price.mid_price < MIN_ENTRY_PRICE:
        return ACTION_SKIP_LOW_PRICE
    floor = max(min_liquidity_usd, LOW_PRICE_LIQUIDITY_FLOOR) if price.mid_price < LOW_PRICE_THRESHOLD else min_liquidity_usd
    if price.liquidity_usd < floor:
        return ACTION_SKIP_LIQ
    if price.spread > 0.08:
        return ACTION_SKIP_SPRD
    return ""


def fetch_market_price(token_id: str, timeout: int = 10) -> Optional[MarketPrice]:
    """
    Fetch live order book from Polymarket CLOB and extract:
      - best bid (highest buy order)
      - best ask (lowest sell order)
      - mid price = (bid + ask) / 2 = market implied probability
      - spread = ask - bid
      - rough liquidity estimate from top 3 levels each side

    Falls back to Gamma API outcomePrices if CLOB book is unavailable.
    """

    # ── Primary: CLOB order book ──────────────────────────────────────────────
    try:
        resp = requests.get(
            CLOB_BOOK_URL,
            params={"token_id": token_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        book = resp.json()

        price = market_price_from_book(token_id, book)
        if price is None:
            logger.warning(f"[EDGE] Unusable book for {token_id[:12]} — trying Gamma")
            return _fetch_price_from_gamma(token_id, timeout)
        return price

    except Exception as e:
        logger.warning(f"[EDGE] CLOB book fetch failed for {token_id[:12]}: {e}")
        return _fetch_price_from_gamma(token_id, timeout)


def _fetch_price_from_gamma(token_id: str, timeout: int) -> Optional[MarketPrice]:
    """
    Fallback: use Gamma API outcomePrices as the market implied probability.
    This is less precise than the order book mid but reliable for a rough signal.
    outcomePrices[0] = YES price ≈ market probability of outcome.
    """
    import json

    try:
        resp = requests.get(
            GAMMA_MARKETS_URL,
            params={"clob_token_ids": token_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        data    = resp.json()
        markets = data if isinstance(data, list) else data.get("markets", [])

        if not markets:
            return None

        raw_prices = markets[0].get("outcomePrices")
        if not raw_prices:
            return None

        prices  = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        yes_price = float(prices[0])

        logger.info(f"[EDGE] Gamma fallback price for {token_id[:12]}: {yes_price:.4f}")

        # liquidity_usd = -1.0 is a sentinel meaning "depth unknown — Gamma fallback".
        # The liquidity gate treats this as a hard block for actionable trading:
        # we have a price for the signal/dashboard, but no order-book depth, so we
        # must not execute against a fabricated ±0.01 spread. (0.0 would mean
        # "measured zero"; -1.0 distinguishes "never measured".)
        return MarketPrice(
            token_id      = token_id,
            mid_price     = yes_price,
            best_bid      = yes_price - 0.01,
            best_ask      = yes_price + 0.01,
            spread        = 0.02,
            liquidity_usd = -1.0,  # sentinel: depth unknown
        )

    except Exception as e:
        logger.error(f"[EDGE] Gamma fallback also failed for {token_id[:12]}: {e}")
        return None


def compute_edge(
    bracket_label: str,
    token_id: str,
    model_prob: float,
    edge_threshold: float = EDGE_THRESHOLD,
    min_liquidity_usd: float = 10.0,
    max_edge_magnitude: float = MAX_EDGE_MAGNITUDE,
    no_token_id: Optional[str] = None,
) -> EdgeSignal:
    """Find direction from YES pricing, then gate the token actually bought."""
    price = fetch_market_price(token_id)
    edge = model_prob - price.mid_price if price else 0.0
    selected_price = price
    if price is None:
        reason = ACTION_NO_PRICE
    elif abs(edge) > max_edge_magnitude:
        reason = ACTION_SKIP_EXTREME
    elif price.liquidity_usd < 0:
        reason = ACTION_SKIP_LIQ  # Gamma prices are display-only.
    else:
        if edge <= -edge_threshold:
            selected_price = fetch_market_price(no_token_id) if no_token_id else None
        reason = entry_gate_reason(selected_price, min_liquidity_usd)

    signal = EdgeSignal(bracket_label, token_id, model_prob, price, edge,
                        edge_threshold, reason, no_token_id)
    signal.execution_price = selected_price
    logger.info(str(signal))
    return signal


def scan_all_brackets(
    token_matrix: Dict[str, Dict[str, str]],
    model_probs: Dict[str, float],
    edge_threshold: float = EDGE_THRESHOLD,
    max_edge_magnitude: float = MAX_EDGE_MAGNITUDE,
) -> Dict[str, EdgeSignal]:
    """
    Run edge calculation across all brackets in token_matrix.
    token_matrix: {bracket_label: {"yes": token_id, "no": no_token_id}}.
    Returns {bracket_label: EdgeSignal} for every bracket
    where a price was successfully fetched.
    """
    signals: Dict[str, EdgeSignal] = {}

    for label, ids in token_matrix.items():
        model_prob = model_probs.get(label)
        if model_prob is None:
            logger.warning(f"[EDGE] No model prob for {label} — skipping")
            continue

        signals[label] = compute_edge(
            bracket_label       = label,
            token_id            = ids["yes"],
            model_prob          = model_prob,
            edge_threshold      = edge_threshold,
            max_edge_magnitude  = max_edge_magnitude,
            no_token_id         = ids.get("no") or None,
        )

    actionable = [l for l, s in signals.items() if s.actionable]
    buys  = [l for l in actionable if signals[l].direction == "BUY"]
    sells = [l for l in actionable if signals[l].direction == "SELL"]
    gated = [l for l, s in signals.items() if s.gate_reason]
    held  = [l for l, s in signals.items()
             if not s.actionable and not s.gate_reason]
    logger.info(
        f"[EDGE] Scan: {len(signals)} brackets | "
        f"BUY={buys or 'none'} SELL={sells or 'none'} "
        f"HOLD_EDGE={held or 'none'} GATED={gated or 'none'}"
    )
    return signals
