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
  Kelly math for NO still uses effective_ask = 1 - best_bid (from the YES
  book) as a sizing estimate; execution fetches the NO token's own live book
  for the actual order.
  NO position pays $1 if outcome does NOT occur.

Edge threshold: 5% (0.05) — set in .env as EDGE_THRESHOLD

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

import logging
import requests
from typing import Dict, Optional, Tuple

logger = logging.getLogger("hermes.edge")

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_BOOK_URL     = "https://clob.polymarket.com/book"

EDGE_THRESHOLD     = 0.05  # overridden by env in scheduler
MAX_EDGE_MAGNITUDE = 0.50  # overridden by env in scheduler


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

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if not bids or not asks:
            logger.warning(f"[EDGE] Empty book for {token_id[:12]}... — trying Gamma")
            return _fetch_price_from_gamma(token_id, timeout)

        # Best bid = highest price in bids, best ask = lowest price in asks
        best_bid = max(float(b["price"]) for b in bids)
        best_ask = min(float(a["price"]) for a in asks)

        if best_ask <= best_bid:
            logger.warning(
                f"[EDGE] Crossed book for {token_id[:12]}: "
                f"bid={best_bid:.4f} ask={best_ask:.4f} — skipping"
            )
            return None

        # Ghost-book detection (Polymarket py-clob-client issue #180):
        # The /book REST endpoint intermittently returns a stale "ghost" snapshot
        # of bid=0.01 / ask=0.99 for active, liquid markets, while /price stays
        # accurate. A 0.01/0.99 book is almost never real for a weather bracket
        # mid-day. Treat it as unavailable and fall back to Gamma rather than
        # trading against a fake 0.50 mid with a 0.98 spread.
        if best_bid <= 0.02 and best_ask >= 0.98:
            logger.warning(
                f"[EDGE] Ghost book for {token_id[:12]} (bid={best_bid:.2f} "
                f"ask={best_ask:.2f}) — issue #180, falling back to Gamma"
            )
            return _fetch_price_from_gamma(token_id, timeout)

        mid_price = (best_bid + best_ask) / 2.0
        spread    = best_ask - best_bid

        # Liquidity on BOTH sides: BUY YES fills into asks, SELL YES (NO) fills
        # into bids. Measuring only the ask side (the old behaviour) let a book
        # with fat asks but thin bids pass the liquidity gate, after which a
        # SELL/NO execution would fail to fill. We store the min of the two so
        # the gate reflects whichever side a trade would actually hit.
        top_asks     = sorted(asks, key=lambda x: float(x["price"]))[:3]
        top_bids     = sorted(bids, key=lambda x: float(x["price"]), reverse=True)[:3]
        ask_liq_usd  = sum(float(a["price"]) * float(a["size"]) for a in top_asks)
        bid_liq_usd  = sum(float(b["price"]) * float(b["size"]) for b in top_bids)
        liquidity_usd = min(ask_liq_usd, bid_liq_usd)

        return MarketPrice(
            token_id      = token_id,
            mid_price     = round(mid_price, 5),
            best_bid      = round(best_bid, 5),
            best_ask      = round(best_ask, 5),
            spread        = round(spread, 5),
            liquidity_usd = round(liquidity_usd, 2),
        )

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
    """
    Compute edge for one bracket. Always returns EdgeSignal (never None).
    Non-actionable signals carry gate_reason explaining why they were blocked.

    Liquidity gate: skip if top-of-book liquidity < min_liquidity_usd.
    This blocks entry into markets where even a $25 order would move the price.

    Extreme-edge gate: skip if |edge| > max_edge_magnitude. A well-liquidity,
    tight-spread market implying near-certainty of the OPPOSITE of what the
    model says is more often evidence the model's stated uncertainty is too
    tight than evidence of a huge mispriced opportunity — see module
    docstring. Checked before the liquidity/spread gates since it's a
    data-sanity concern independent of market microstructure.
    """
    market_price = fetch_market_price(token_id)

    if market_price is None:
        logger.warning(f"[EDGE] {bracket_label}: price fetch failed")
        return EdgeSignal(
            bracket_label=bracket_label, token_id=token_id,
            model_prob=model_prob, market_price=None,
            edge=0.0, edge_threshold=edge_threshold,
            gate_reason=ACTION_NO_PRICE,
            no_token_id=no_token_id,
        )

    edge = model_prob - market_price.mid_price

    # Extreme-edge sanity gate — see compute_edge()'s docstring.
    if abs(edge) > max_edge_magnitude:
        logger.warning(
            f"[EDGE] {bracket_label}: |edge|={abs(edge):.3f} > cap {max_edge_magnitude:.2f} "
            f"(model={model_prob:.3f} market={market_price.mid_price:.3f}) — "
            f"gated as likely miscalibration, not traded"
        )
        return EdgeSignal(
            bracket_label=bracket_label, token_id=token_id,
            model_prob=model_prob, market_price=market_price,
            edge=edge, edge_threshold=edge_threshold,
            gate_reason=ACTION_SKIP_EXTREME,
            no_token_id=no_token_id,
        )

    # Liquidity gate — blocks two cases:
    #   1. Gamma-fallback prices (liquidity_usd == -1.0): depth unknown, so we
    #      have a signal for the dashboard but must not execute against a
    #      fabricated spread.
    #   2. Real books too thin to absorb a min-size order without moving price
    #      (0 <= liquidity_usd < floor).
    if market_price.liquidity_usd < 0:
        logger.info(
            f"[EDGE] {bracket_label}: price via Gamma fallback (depth unknown) — "
            f"gated from execution"
        )
        return EdgeSignal(
            bracket_label=bracket_label, token_id=token_id,
            model_prob=model_prob, market_price=market_price,
            edge=edge, edge_threshold=edge_threshold,
            gate_reason=ACTION_SKIP_LIQ,
            no_token_id=no_token_id,
        )
    if market_price.liquidity_usd < min_liquidity_usd:
        logger.info(
            f"[EDGE] {bracket_label}: liquidity ${market_price.liquidity_usd:.2f} "
            f"< floor ${min_liquidity_usd} — gated"
        )
        return EdgeSignal(
            bracket_label=bracket_label, token_id=token_id,
            model_prob=model_prob, market_price=market_price,
            edge=edge, edge_threshold=edge_threshold,
            gate_reason=ACTION_SKIP_LIQ,
            no_token_id=no_token_id,
        )

    # Spread gate
    if market_price.spread > 0.08:
        logger.info(
            f"[EDGE] {bracket_label}: spread={market_price.spread:.3f} > 0.08 — gated"
        )
        return EdgeSignal(
            bracket_label=bracket_label, token_id=token_id,
            model_prob=model_prob, market_price=market_price,
            edge=edge, edge_threshold=edge_threshold,
            gate_reason=ACTION_SKIP_SPRD,
            no_token_id=no_token_id,
        )

    signal = EdgeSignal(
        bracket_label  = bracket_label,
        token_id       = token_id,
        model_prob     = model_prob,
        market_price   = market_price,
        edge           = edge,
        edge_threshold = edge_threshold,
        gate_reason    = "",
        no_token_id    = no_token_id,
    )
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
