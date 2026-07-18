"""
core/sizing.py — H2: Fractional Kelly sizing with net EV fee model

Kelly formula:
  b = net odds = (1 - ask_price) / ask_price
  f = (p*b - q) / b = edge / b   (raw Kelly fraction)
  size = f * 0.50 * vault        (half Kelly)

Net EV (what the hurdle checks):
  gross_ev = p * b - q
  fee_load = taker_fee + gas_usd / position_size
  net_ev   = gross_ev - fee_load

Anti-favorite compression:
  If market_ask > 0.65, compress model_prob by a factor derived from
  trailing ledger bias. Bounds: [0.70, 0.95].
  Rationale: high-price markets on Polymarket tend to have liquidity-driven
  price distortion. Compression reflects model uncertainty at the tail.

Minimum floor: $10.
  Compensated by a tiered net EV hurdle: positions between $10–$25
  require net EV > 8% to clear (lowered from 15% — gas=$0.05 on $10 is
  0.5% additional load; 8% still ensures meaningful net edge after fees).
  Positions above $25 use the standard 2.5% hurdle.

Changes from v4.5:
  KELLY_FRACTION: 0.25 → 0.50 (half Kelly)
    Rationale: at $200 vault, quarter Kelly cannot produce $10+ positions
    on 40–50c assets with moderate edge. Half Kelly keeps sizing conservative
    relative to full Kelly while clearing the execution floor.
    Risk: larger per-trade exposure. Mitigated by trailing stop + stop loss.

  _SMALL_POSITION_EV_HURDLE: 0.15 → 0.08 (8%)
    Rationale: 15% blocked all valid sub-$25 signals including the 33°C SELL
    NO (8.34% net EV). 8% still provides meaningful fee buffer while allowing
    well-edged small positions through.
    Risk: smaller net margin on sub-$25 trades. Acceptable given gas is only
    $0.05 per order — the hurdle exists for gas drag, not taker fee.
"""

import os
import logging
from typing import Any, Optional

logger = logging.getLogger("hermes.sizing")

TAKER_FEE_RATE    = 0.02    # Polymarket maker/taker fee
GAS_COST_USD      = 0.05    # Polygon gas estimate
MAX_POSITION_PCT  = 0.05    # Cap at 5% of vault per trade

# Runtime-configurable via .env — defaults match v4.5 option 2 & 3 fix
MIN_POSITION_USD  = float(os.getenv("MIN_POSITION_USD", "2.0"))  # Hard floor
KELLY_FRACTION    = float(os.getenv("KELLY_FRACTION",  "0.50"))  # half Kelly
_SMALL_POSITION_THRESHOLD = 25.0
_SMALL_POSITION_EV_HURDLE = float(os.getenv("SMALL_EV_HURDLE", "0.08"))


def check_sizing_config(vault_usd: float) -> None:
    """
    Warn at startup if the vault/cap/floor parameters collapse the sizing
    range to a single value, which makes Kelly and edge irrelevant.

    At vault=$200 with MAX_POSITION_PCT=0.05, the per-trade cap is $10 —
    exactly the MIN_POSITION_USD floor. Every executable position is then
    forced to exactly $10 regardless of edge or Kelly fraction. This isn't
    wrong, but it's almost certainly not what the operator intends, so we
    surface it loudly. Called once from scheduler.main().
    """
    cap = vault_usd * MAX_POSITION_PCT
    if cap <= MIN_POSITION_USD:
        logger.warning(
            f"[SIZING] ⚠️  Per-trade cap (${cap:.2f} = {MAX_POSITION_PCT:.0%} of "
            f"${vault_usd:.0f} vault) <= floor (${MIN_POSITION_USD:.0f}). "
            f"Every position will be forced to ${MIN_POSITION_USD:.0f} — Kelly "
            f"sizing and edge magnitude have NO effect. To restore a sizing "
            f"range, raise vault above ${MIN_POSITION_USD / MAX_POSITION_PCT:.0f} "
            f"or lower MIN_POSITION_USD."
        )
    elif cap < MIN_POSITION_USD * 2:
        logger.info(
            f"[SIZING] Note: per-trade cap ${cap:.2f} is within 2× of the "
            f"${MIN_POSITION_USD:.0f} floor — sizing range is narrow."
        )


class SizingResult:
    def __init__(
        self,
        verdict: str,         # "EXECUTE" | "HOLD"
        direction: str,       # "BUY" | "SELL" — passed through from EdgeSignal
        size_usd: float,
        net_ev: float,
        gross_ev: float,
        kelly_raw: float,
        reason: str,
    ):
        self.verdict   = verdict
        self.direction = direction
        self.size_usd  = size_usd
        self.net_ev    = net_ev
        self.gross_ev  = gross_ev
        self.kelly_raw = kelly_raw
        self.reason    = reason

    def __repr__(self):
        return (
            f"SizingResult({self.verdict} {self.direction} ${self.size_usd:.2f} "
            f"net_ev={self.net_ev*100:+.2f}% [{self.reason}])"
        )


def compute_size(
    model_prob: float,
    market_ask: float,
    vault_usd: float,
    direction: str = "BUY",
    trailing_bias: float = 0.0,
    net_ev_hurdle: float = 0.025,
) -> SizingResult:
    """
    Compute position size for a bracket trade.

    Args:
        model_prob:    P(outcome) from skewnorm model (bias-corrected)
        market_ask:    best ask price in order book (YES execution price).
                       For NO trades: caller passes (1 - best_bid) as the
                       effective ask so Kelly math is symmetric.
        vault_usd:     total capital allocated to bot
        direction:     "BUY" (YES) or "SELL" (NO)
        trailing_bias: ledger trailing bias (for compression factor)
        net_ev_hurdle: minimum net EV at standard position size (2.5%).
                       Small positions ($10–$25) apply a higher tiered
                       hurdle (15%) to account for proportionally larger
                       gas drag. Overridden by caller only in tests.

    Returns:
        SizingResult with verdict EXECUTE or HOLD
    """
    _hold = lambda reason, ev=0.0: SizingResult(
        "HOLD", direction, 0.0, ev, 0.0, 0.0, reason
    )

    if market_ask <= 0.01 or market_ask >= 0.99:
        return _hold("ask_price_out_of_range")

    # Anti-favorite compression — applies to any favorite, BUY or SELL.
    # market_ask here is the effective ask for whichever side we're sizing:
    #   BUY YES  → best_ask
    #   SELL/NO  → 1 - best_bid  (cost to buy NO)
    # When that effective price is high (> 0.65) the position is a favorite,
    # and Polymarket favorites carry liquidity-driven price distortion, so we
    # shrink model_prob toward the market. The old code only compressed BUY,
    # leaving NO trades on favorites (e.g. NO effective_ask 0.90) uncompressed.
    adjusted_prob = model_prob
    if market_ask > 0.65:
        raw_factor    = 0.85 + (trailing_bias * 0.02)
        factor        = max(0.70, min(0.95, raw_factor))
        adjusted_prob = model_prob * factor
        logger.info(
            f"[SIZING] Anti-fav [{direction}]: model={model_prob:.3f} × {factor:.3f} "
            f"→ {adjusted_prob:.3f} (eff_ask={market_ask:.3f} bias={trailing_bias:+.3f})"
        )

    p = adjusted_prob
    q = 1.0 - p
    b = (1.0 - market_ask) / market_ask   # net odds per dollar risked

    gross_ev = (p * b) - q

    # First-pass net EV using estimated size (vault * max_pct)
    est_size = vault_usd * MAX_POSITION_PCT
    gas_frac = GAS_COST_USD / max(est_size, 1.0)
    net_ev   = gross_ev - (TAKER_FEE_RATE + gas_frac)

    if net_ev <= net_ev_hurdle:
        return _hold(
            f"net_ev={net_ev*100:.2f}% < hurdle={net_ev_hurdle*100:.2f}%",
            ev=net_ev,
        )

    # Kelly sizing — growth-optimal fraction uses GROSS edge, not net_ev.
    # Classic Kelly: f* = edge/b where edge = p*b - q (the gross expectation).
    # Fees are handled separately by the net_ev hurdle above; folding them into
    # the Kelly numerator (the old `net_ev/b`) double-penalises fees — once at
    # the gate, again by shrinking size ~5%. Kelly should size on the true edge.
    kelly_raw = gross_ev / b if b > 0 else 0.0
    kelly_pct = kelly_raw * KELLY_FRACTION
    size_usd  = min(kelly_pct * vault_usd, vault_usd * MAX_POSITION_PCT)

    # Hard floor check
    if size_usd < MIN_POSITION_USD:
        return _hold(
            f"size=${size_usd:.2f} < floor=${MIN_POSITION_USD}",
            ev=net_ev,
        )

    # Refine net EV with actual size (gas fraction changes materially for small positions)
    gas_frac_final = GAS_COST_USD / max(size_usd, 1.0)
    net_ev_final   = gross_ev - (TAKER_FEE_RATE + gas_frac_final)

    # Tiered net EV hurdle: small positions ($10–$25) need 15% net EV
    # to absorb proportionally heavier gas drag
    effective_hurdle = (
        _SMALL_POSITION_EV_HURDLE
        if size_usd < _SMALL_POSITION_THRESHOLD
        else net_ev_hurdle
    )
    if net_ev_final <= effective_hurdle:
        return _hold(
            f"net_ev={net_ev_final*100:.2f}% < tiered_hurdle={effective_hurdle*100:.2f}% "
            f"(size=${size_usd:.2f})",
            ev=net_ev_final,
        )

    return SizingResult(
        "EXECUTE",
        direction,
        round(size_usd, 2),
        round(net_ev_final, 5),
        round(gross_ev, 5),
        round(kelly_raw, 5),
        "ok",
    )


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION MODE — forced $1 sizing, no Kelly/EV gating
# ══════════════════════════════════════════════════════════════════════════════
VALIDATION_POSITION_USD = float(os.getenv("VALIDATION_POSITION_USD", "1.00"))


def compute_validation_size(
    model_prob: float,
    market_ask: float,
    direction: str = "BUY",
) -> SizingResult:
    """
    Forced $1 sizing for end-to-end mechanics validation.

    Bypasses Kelly fraction, the $10 floor, and both EV hurdles entirely.
    Trades on ANY actionable edge signal regardless of net EV after fees.
    Real gross_ev and net_ev are still computed and attached to the result
    for logging/research — only the gating is skipped.

    At $1 size, gas ($0.05) alone is 5% drag and taker fee adds another 2%,
    so net_ev will frequently print negative even on genuine edge signals.
    This is expected and accepted — VALIDATION_MODE exists to prove the
    full lifecycle (entry → trailing stop → stop loss → settlement) works
    mechanically, not to be profitable. Never run with real capital deployed
    at scale; use only to confirm fills, exits, and DB writes are correct.

    Returns SizingResult with verdict="EXECUTE" unconditionally (caller in
    scheduler.py only invokes this for already-actionable EdgeSignals, so
    there is always a real edge — just not necessarily a profitable one
    after $1-scale fees).
    """
    if market_ask <= 0.01 or market_ask >= 0.99:
        return SizingResult(
            "HOLD", direction, 0.0, 0.0, 0.0, 0.0,
            "ask_price_out_of_range (validation mode still respects this)",
        )

    p = model_prob
    q = 1.0 - p
    b = (1.0 - market_ask) / market_ask
    gross_ev = (p * b) - q

    gas_frac = GAS_COST_USD / VALIDATION_POSITION_USD
    net_ev   = gross_ev - (TAKER_FEE_RATE + gas_frac)

    logger.info(
        f"[SIZING] VALIDATION_MODE: forcing ${VALIDATION_POSITION_USD:.2f} "
        f"(gross_ev={gross_ev*100:+.2f}% net_ev={net_ev*100:+.2f}% "
        f"— gating skipped, fee drag at this size is {gas_frac*100:.1f}% from gas alone)"
    )

    return SizingResult(
        "EXECUTE",
        direction,
        VALIDATION_POSITION_USD,
        round(net_ev, 5),
        round(gross_ev, 5),
        0.0,   # kelly_raw not applicable in validation mode
        "validation_mode_forced",
    )
