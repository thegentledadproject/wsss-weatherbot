"""
core/position_monitor.py — Job 5: Active position monitor
v4.5: Trailing stop replaces static profit target.
v4.6: SELL/NO positions are now tracked by the NO token's own token_id
      (core/execution.py opens them via a real BUY on the NO token, not a
      synthetic short on YES — see core/edge.py's module docstring). Once
      the position's own price feed is the token actually held, a NO
      position behaves exactly like a BUY position (price up = good), so
      the old direction-specific inverted peak/trail/stop-loss math is no
      longer needed — `direction` is kept only for labeling/logs.

EXIT CONDITIONS (priority order):

  EXIT 1 — TRAILING STOP (replaces static profit target)
    Mechanism (applies uniformly — a NO position is a long position on the
    NO token, same as a BUY position is a long position on the YES token):
      - Each cycle, if current_mid > stored peak_price → update peak_price
      - Trailing stop level = peak_price * (1 - TRAIL_PCT)
      - Exit trigger: current_mid <= trail_level
      - Meaning: price rose (good), then pulled back TRAIL_PCT from its peak
      - Default TRAIL_PCT = 0.20 (20% drawdown from peak)

    Why trailing stop > static profit target:
      Static target exits at a fixed price. Trailing stop follows the position
      upward and only exits on reversal — capturing more upside on strong moves.
      On Jun 28, the 32°C BUY hit the static target at 58c (3hr hold, +28.7%).
      A 20% trailing stop from that peak would have exited at 58c × 0.80 = 46c
      only if the market reversed — otherwise it would have ridden to 64c+ at
      16:00 time exit, capturing the full move.

  EXIT 2 — STOP LOSS
    mid <= entry_price * (1 - STOP_LOSS_PCT)  (market moved against entry)
    Percentage-of-entry, NOT a fixed absolute price delta. Previously this
    subtracted a flat EDGE_THRESHOLD (0.05) from entry_price regardless of
    entry level — fine for a ~0.50-0.90 entry (5-10% real tolerance), but a
    cheap long-shot entry like 0.089 got a stop at 0.039, a 56% drawdown
    tolerance before firing. Same nominal $0.05 distance, wildly different
    real risk exposure depending on price. Confirmed in production: a
    30°C:YES position entered at 0.08929 sat at -49.6% unrealized, still
    nowhere near triggering. Percentage-of-entry gives consistent relative
    protection regardless of entry price.

  EXIT 3 — TIME EXIT (unchanged)
    16:00 SGT hard close, all positions, best available price.

TRAIL_PCT is configurable via env var TRAIL_PCT (default 0.20).
Set lower (e.g. 0.10) for tighter stops on thin books.
Set higher (e.g. 0.30) to tolerate more intraday noise.

STOP_LOSS_PCT is configurable via env var STOP_LOSS_PCT (default 0.10).
Decoupled from EDGE_THRESHOLD (core/edge.py's separate edge-detection gate) —
these are two different concerns that used to share one config value.

peak_price is persisted in the DB open_positions table (added in v4.5).
Updated every Job 5 cycle when a new high/low is observed.
"""

import os
import math
import logging
import datetime
from typing import Any, Dict, List, Optional

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import MarketOrderArgs, OrderType
from py_clob_client_v2.order_builder.constants import SELL as _CLOB_SELL

from db.ledger import Ledger
from core.edge import fetch_market_price, MarketPrice
from core.execution import _is_ghost_book, VWAP_DRIFT_TOLERANCE

logger = logging.getLogger("hermes.monitor")

HARD_EXIT_HOUR_SGT = 16
TRAIL_PCT          = float(os.getenv("TRAIL_PCT", 0.20))
STOP_LOSS_PCT      = float(os.getenv("STOP_LOSS_PCT", 0.10))  # % of entry, not a fixed $ delta


class ExitReason:
    TRAILING_STOP = "TRAILING_STOP"
    STOP_LOSS     = "STOP_LOSS"
    TIME_EXIT     = "TIME_EXIT"
    NONE          = "NONE"


class ExitDecision:
    def __init__(
        self,
        token_id:     str,
        label:        str,
        direction:    str,
        reason:       str,
        entry_price:  float,
        peak_price:   float,
        trail_level:  Optional[float],
        current_mid:  float,
        model_prob:   float,
        market_price: Optional[MarketPrice],
    ):
        self.token_id     = token_id
        self.label        = label
        self.direction    = direction
        self.reason       = reason
        self.entry_price  = entry_price
        self.peak_price   = peak_price
        self.trail_level  = trail_level
        self.current_mid  = current_mid
        self.model_prob   = model_prob
        self.market_price = market_price
        self.should_exit  = reason != ExitReason.NONE

    def __repr__(self):
        trail = f" trail_lvl={self.trail_level:.4f}" if self.trail_level else ""
        return (
            f"ExitDecision({self.label} [{self.direction}] "
            f"reason={self.reason} entry={self.entry_price:.4f} "
            f"peak={self.peak_price:.4f}{trail} mid={self.current_mid:.4f})"
        )


def _parse_direction(position_label: str) -> str:
    if ":NO" in position_label:
        return "SELL"
    return "BUY"


def _parse_bracket(position_label: str) -> str:
    return position_label.split(":")[0]


def evaluate_exit(
    token_id:        str,
    position_label:  str,
    entry_price:     float,
    peak_price:      float,
    model_prob:      float,
    stop_loss_pct:   float,
    trail_pct:       float,
    ledger:          Ledger,
    force_time_exit: bool = False,
) -> ExitDecision:
    """
    Evaluate whether to exit a position using trailing stop logic.

    Args:
        peak_price:    best price seen since entry (max mid for BUY, min mid for SELL).
                       Loaded from DB, updated here if new peak observed.
        trail_pct:     fractional drawdown from peak that triggers exit.
        stop_loss_pct: fractional drawdown from ENTRY (not peak) that triggers
                       exit — e.g. 0.10 = stop fires at 90% of entry price,
                       regardless of what that entry price actually is.
    """
    direction = _parse_direction(position_label)
    label     = _parse_bracket(position_label)

    # ── Time exit: no price check needed ──────────────────────────────────────
    if force_time_exit:
        market_price = fetch_market_price(token_id)
        return ExitDecision(
            token_id=token_id, label=label, direction=direction,
            reason=ExitReason.TIME_EXIT,
            entry_price=entry_price, peak_price=peak_price,
            trail_level=None,
            current_mid=market_price.mid_price if market_price else entry_price,
            model_prob=model_prob, market_price=market_price,
        )

    # ── Fetch live price ───────────────────────────────────────────────────────
    market_price = fetch_market_price(token_id)
    if market_price is None:
        logger.warning(f"[MONITOR] {label}: price fetch failed — skipping")
        return ExitDecision(
            token_id=token_id, label=label, direction=direction,
            reason=ExitReason.NONE, entry_price=entry_price,
            peak_price=peak_price, trail_level=None,
            current_mid=entry_price, model_prob=model_prob, market_price=None,
        )

    mid = market_price.mid_price

    # A NO position is tracked by the NO token's own token_id (see module
    # docstring), so `mid` here already behaves like a normal long position
    # for both BUY and SELL/NO — no direction-specific inversion needed.

    # ── Update peak if new high ────────────────────────────────────────────
    if mid > peak_price:
        ledger.update_peak_price(token_id, mid)
        peak_price = mid
        logger.info(f"[MONITOR] {label}: new peak = {peak_price:.4f}")

    # ── Trailing stop level ────────────────────────────────────────────────
    # Only armed once peak is above entry — avoids stopping out immediately
    # on a position that's never moved in our favour.
    trail_level = peak_price * (1.0 - trail_pct) if peak_price > entry_price else None
    trail_trigger = (trail_level is not None) and (mid <= trail_level)

    # ── Stop loss ─────────────────────────────────────────────────────────
    # Independent of trailing stop — fires if market moves against entry
    # before the trailing stop is armed. Percentage-of-entry (not a fixed
    # absolute delta) so a cheap long-shot entry gets the same relative
    # protection as an expensive one — see module docstring.
    stop_level_price = entry_price * (1.0 - stop_loss_pct)
    stop_trigger = mid <= stop_level_price

    trail_str = f"{trail_level:.4f}" if trail_level is not None else "not_armed"
    logger.info(
        f"[MONITOR] {label} {direction} | mid={mid:.4f} peak={peak_price:.4f} "
        f"trail_lvl={trail_str} "
        f"stop_lvl={stop_level_price:.4f}"
    )

    if trail_trigger:
        reason = ExitReason.TRAILING_STOP
    elif stop_trigger:
        reason = ExitReason.STOP_LOSS
    else:
        reason = ExitReason.NONE

    if reason != ExitReason.NONE:
        trail_str = f"{trail_level:.4f}" if trail_level is not None else "N/A"
        logger.info(
            f"[MONITOR] {label} [{direction}] → {reason} | "
            f"entry={entry_price:.4f} peak={peak_price:.4f} "
            f"trail={trail_str} mid={mid:.4f}"
        )

    return ExitDecision(
        token_id=token_id, label=label, direction=direction,
        reason=reason, entry_price=entry_price, peak_price=peak_price,
        trail_level=trail_level, current_mid=mid,
        model_prob=model_prob, market_price=market_price,
    )


def _extract_vwap_bid_by_shares(orderbook: Any, target_shares: float) -> Optional[float]:
    """
    Walk bid side accumulating SHARES (not USD) until target_shares is reached.
    Used for exits: closing a BUY position means selling the EXACT number of
    shares originally bought, not walking to a dollar-proceeds target — the
    old _extract_vwap_bid(book, size_usd) would stop early/late depending on
    how far price has moved from entry, producing a VWAP for the wrong quantity.
    """
    raw_bids = getattr(orderbook, "bids", None)
    if raw_bids is None:
        raw_bids = orderbook.get("bids", []) if isinstance(orderbook, dict) else []
    if not raw_bids:
        return None

    def _p(b): return float(b.price) if hasattr(b, "price") else float(b.get("price", 0))
    def _s(b): return float(b.size)  if hasattr(b, "size")  else float(b.get("size",  0))

    acc_usd = acc_shares = 0.0
    for bid in sorted(raw_bids, key=_p, reverse=True):
        price, size = _p(bid), _s(bid)
        if price <= 0:
            continue
        if acc_shares + size >= target_shares:
            remaining   = target_shares - acc_shares
            acc_usd    += remaining * price
            acc_shares += remaining
            break
        acc_shares += size
        acc_usd    += price * size

    return round(acc_usd / acc_shares, 5) if acc_shares > 0 else None


def _parse_fill(response: Any, label: str) -> bool:
    """
    Same real-response-shape bug core/execution.py's _parse_fill_status()
    had before it was fixed: post_order()'s response for a matched FOK has
    NO "size_matched" key — it reports takingAmount/makingAmount plus a
    "success" flag instead. Confirmed against a real exit fill (status=
    'matched', success=True, real transactionsHashes) that this function
    logged as "FOK rejected" — the position was actually closed on-chain
    while open_positions kept showing it open. Fixed to match.
    """
    if response is None:
        return False
    data = response if isinstance(response, dict) else {
        "status":       getattr(response, "status", None),
        "success":      getattr(response, "success", None),
        "size_matched": getattr(response, "size_matched", None),
        "takingAmount": getattr(response, "takingAmount", None),
        "makingAmount": getattr(response, "makingAmount", None),
    }
    if not isinstance(data, dict):
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

    return status in ("MATCHED", "FILLED") and success is not False and matched > 0


class PositionMonitor:
    """
    Evaluates all open positions and executes exits where triggered.
    Called by Job 5 every 5 minutes, 08:05–15:55 SGT.
    v4.5: uses trailing stop instead of static profit target.
    """

    def __init__(
        self,
        client:         ClobClient,
        ledger:         Ledger,
        stop_loss_pct:  float = STOP_LOSS_PCT,
        trail_pct:      float = TRAIL_PCT,
        icao:           str   = "WSSS",
    ):
        self.client        = client
        self.ledger        = ledger
        self.stop_loss_pct = stop_loss_pct
        self.trail_pct     = trail_pct
        self.icao          = icao

    def run(self, model_probs: Dict[str, float], market_date: str = "") -> List[Dict]:
        """
        market_date: today's SGT calendar date (from _state["market_date"]).
        Used to scope the 16:00 hard time-exit to genuinely STALE positions
        only — see per-position force_time_exit logic below.
        """
        sg_now         = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        hour_cutoff    = sg_now.hour >= HARD_EXIT_HOUR_SGT
        open_positions = self.ledger.get_open_positions()
        results        = []

        if not open_positions:
            logger.info("[MONITOR] No open positions.")
            return results

        for pos in open_positions:
            token_id       = pos["token_id"]
            position_label = pos["bracket_label"]
            entry_price    = float(pos["entry_price"])
            size_usd       = float(pos["size_usd"])
            opened_at      = pos["opened_at"]
            pos_market_date = pos["market_date"] if pos["market_date"] else None
            # peak_price: stored in DB, updated each cycle. `is not None` (not
            # truthiness) — a stored 0.0 is only possible before the first
            # position ever recorded a peak, and truthiness would treat a
            # theoretically valid 0.0 the same as missing.
            peak_price     = float(pos["peak_price"]) if pos["peak_price"] is not None else entry_price

            bracket_label  = _parse_bracket(position_label)
            model_prob     = model_probs.get(bracket_label, 0.5)

            # ── Per-position force_time_exit ────────────────────────────────
            # Required by the round-the-clock scheduler redesign: Jobs 2/3 can
            # now open positions at ANY hour, including after 16:00 SGT. A
            # single global "wall-clock hour >= 16" check (the old behaviour)
            # would immediately force-close a position opened at, say, 20:00
            # SGT on its very next Job 5 tick — a self-defeating trade.
            #
            # A DATE-only comparison ("same calendar day" vs "prior day") is
            # NOT sufficient to fix this — a position opened at 20:00 SGT
            # today is still dated "today", so a same-day + hour>=16 rule
            # would still force-close it immediately. The correct check needs
            # actual TIME granularity: only force-exit a same-day position if
            # it existed BEFORE today's 16:00 SGT cutoff moment AND that
            # cutoff has now passed. A position opened at 20:00 didn't exist
            # at 16:00, so the cutoff doesn't apply to it — normal trailing
            # stop / stop loss protection applies until either it closes
            # naturally or tomorrow's stale-position rule catches it.
            try:
                opened_at_utc = datetime.datetime.fromisoformat(opened_at)
                opened_at_sgt = opened_at_utc + datetime.timedelta(hours=8)
            except (ValueError, TypeError):
                opened_at_sgt = None

            today_cutoff_sgt = sg_now.replace(
                hour=HARD_EXIT_HOUR_SGT, minute=0, second=0, microsecond=0
            )

            if pos_market_date and market_date and pos_market_date < market_date:
                # Carried over from an earlier calendar day — definitely
                # stale, force-exit regardless of current time of day.
                force_time_exit = True
                logger.warning(
                    f"[MONITOR] {position_label}: stale position from {pos_market_date} "
                    f"(today is {market_date}) — forcing exit regardless of hour"
                )
            elif opened_at_sgt is not None:
                # Same day (or unknown market_date) — only force-exit if this
                # position existed before today's cutoff AND the cutoff has
                # now passed. A position opened after 16:00 is exempt today.
                force_time_exit = (opened_at_sgt < today_cutoff_sgt) and (sg_now >= today_cutoff_sgt)
            else:
                # opened_at unparseable — fall back to the old global
                # behaviour rather than silently never force-exiting.
                force_time_exit = hour_cutoff

            # Per-position isolation: one bad API response or unexpected value
            # must not abort monitoring for every OTHER open position in this
            # cycle. Previously an unhandled exception here (e.g. the f-string
            # bug) silently broke the entire loop after the first position.
            try:
                decision = evaluate_exit(
                    token_id        = token_id,
                    position_label  = position_label,
                    entry_price     = entry_price,
                    peak_price      = peak_price,
                    model_prob      = model_prob,
                    stop_loss_pct   = self.stop_loss_pct,
                    trail_pct       = self.trail_pct,
                    ledger          = self.ledger,
                    force_time_exit = force_time_exit,
                )
            except Exception as e:
                logger.error(
                    f"[MONITOR] {position_label}: evaluate_exit raised {type(e).__name__}: {e} "
                    f"— skipping this position this cycle, will retry next cycle"
                )
                continue

            if not decision.should_exit:
                trail_str = (
                    f"trail_lvl={decision.trail_level:.4f}"
                    if decision.trail_level is not None else "trail_not_armed"
                )
                logger.info(
                    f"[MONITOR] {position_label}: HOLD | "
                    f"mid={decision.current_mid:.4f} peak={decision.peak_price:.4f} "
                    f"{trail_str}"
                )
                continue

            try:
                exit_result = self._execute_exit(decision, size_usd, opened_at, market_date)
            except Exception as e:
                logger.error(
                    f"[MONITOR] {position_label}: _execute_exit raised {type(e).__name__}: {e} "
                    f"— position remains open, will retry next cycle"
                )
                continue
            results.append(exit_result)

        return results

    def _execute_exit(
        self,
        decision:    ExitDecision,
        size_usd:    float,
        opened_at:   str,
        market_date: str = "",
    ) -> Dict:
        token_id  = decision.token_id
        label     = decision.label
        direction = decision.direction
        reason    = decision.reason

        logger.info(
            f"[MONITOR] → EXIT {label} [{direction}] {reason} "
            f"peak={decision.peak_price:.4f} mid={decision.current_mid:.4f} "
            f"size=${size_usd:.2f}"
        )

        # ── Exact share count to close — NOT size_usd (the original dollar
        # notional). Confirmed bug: walking the book to a size_usd proceeds/
        # cost target (the old behaviour) only approximates the right depth
        # when current price ≈ entry price. Once price has moved — which is
        # the whole point of a trailing stop or stop loss firing — it either
        # leaves a naked partial position (BUY exit) or massively over-buys
        # into a net-long position (SELL/NO exit). Closing a position means
        # trading the EXACT quantity originally opened, at whatever price
        # the market offers now — so we must walk the book by SHARE target.
        shares_held = size_usd / decision.entry_price
        clob_side   = _CLOB_SELL

        # ── Compute-time VWAP + revalidation, mirroring core/execution.py's
        # entry-side pattern. Without this, a ghost-book snapshot (bid=0.01/
        # ask=0.99, py-clob-client issue #180) would produce a wildly wrong
        # exit_vwap that gets recorded as the real exit_price/realised_pnl
        # even if the actual FOK fills fine at the true market price —
        # corrupting P&L bookkeeping rather than just risking a bad fill.
        try:
            book_1 = self.client.get_order_book(token_id)
        except Exception as e:
            logger.error(f"[MONITOR] {label}: book fetch failed: {e}")
            return self._result(decision, size_usd, opened_at, None, False, "book_fetch_failed")

        if _is_ghost_book(book_1):
            logger.error(
                f"[MONITOR] {label}: ghost book detected at compute-time "
                f"(get_order_book issue #180) — abort, will retry next cycle"
            )
            return self._result(decision, size_usd, opened_at, None, False, "ghost_book_compute")

        # Every open position (BUY/YES or SELL/NO) is a plain long on the
        # token stored in open_positions.token_id (see module docstring) —
        # closing it always means selling that exact share count into the bid.
        exit_vwap_compute = _extract_vwap_bid_by_shares(book_1, shares_held)
        if exit_vwap_compute is None:
            logger.warning(f"[MONITOR] {label}: no book depth at compute — retry next cycle")
            return self._result(decision, size_usd, opened_at, None, False, "no_depth")

        try:
            book_2 = self.client.get_order_book(token_id)
        except Exception as e:
            logger.error(f"[MONITOR] {label}: revalidation book fetch failed: {e}")
            return self._result(decision, size_usd, opened_at, None, False, "book_fetch_failed")

        if _is_ghost_book(book_2):
            logger.error(
                f"[MONITOR] {label}: ghost book detected at revalidation "
                f"(get_order_book issue #180) — abort, will retry next cycle"
            )
            return self._result(decision, size_usd, opened_at, None, False, "ghost_book_revalidation")

        exit_vwap = _extract_vwap_bid_by_shares(book_2, shares_held)
        if exit_vwap is None:
            logger.warning(f"[MONITOR] {label}: book gone before exit — retry next cycle")
            return self._result(decision, size_usd, opened_at, None, False, "no_depth")

        drift = abs(exit_vwap - exit_vwap_compute) / exit_vwap_compute
        if drift > VWAP_DRIFT_TOLERANCE:
            logger.warning(
                f"[MONITOR] {label}: exit VWAP drifted {drift:.1%} "
                f"({exit_vwap_compute:.4f}→{exit_vwap:.4f}) — abort, retry next cycle"
            )
            return self._result(decision, size_usd, opened_at, None, False, "vwap_drift")

        logger.info(
            f"[MONITOR] {label}: exit VWAP={exit_vwap:.4f} side={clob_side} "
            f"shares_held={shares_held:.4f}"
        )

        # Order amount: market SELL amount = shares (Polymarket docs).
        # shares_held = size_usd / entry_price is a re-derived approximation
        # of what was actually filled at entry (the real matched quantity
        # isn't stored) — round-to-nearest could round UP past the true
        # owned balance. Confirmed against real rejections: e.g. balance=
        # 1538460, order_amount=1540000 — a rounding-up excess of exactly
        # the kind math.floor prevents. Always round DOWN so a sell order
        # never requests more shares than are actually held.
        order_amount = math.floor(shares_held * 100) / 100

        try:
            signed = self.client.create_market_order(MarketOrderArgs(
                token_id=token_id, amount=order_amount, price=exit_vwap, side=clob_side,
            ))
            response = self.client.post_order(signed, OrderType.FOK)
            filled   = _parse_fill(response, label)
        except Exception as e:
            logger.error(f"[MONITOR] {label}: exit order error: {e}")
            return self._result(decision, size_usd, opened_at, exit_vwap, False, f"order_error:{e}")

        if filled:
            realised_pnl = (exit_vwap - decision.entry_price) * shares_held

            logger.info(
                f"[MONITOR] ✓ {label} [{direction}] FILLED "
                f"entry={decision.entry_price:.4f} exit={exit_vwap:.4f} "
                f"P&L={realised_pnl:+.4f} ({realised_pnl/size_usd*100:+.1f}%) "
                f"peak_captured={decision.peak_price:.4f}"
            )
            self.ledger.close_position(token_id)
            self.ledger.log_exit(
                token_id=token_id, bracket_label=label, direction=direction,
                reason=reason, entry_price=decision.entry_price,
                exit_price=exit_vwap, size_usd=size_usd,
                realised_pnl=realised_pnl, opened_at=opened_at,
                market_date=market_date,
            )
            return self._result(decision, size_usd, opened_at, exit_vwap, True, reason, realised_pnl)

        else:
            logger.warning(f"[MONITOR] ✗ {label}: FOK rejected — retry next cycle. Raw: {response}")
            return self._result(decision, size_usd, opened_at, exit_vwap, False, "fok_rejected")

    @staticmethod
    def _result(
        decision: ExitDecision, size_usd: float, opened_at: str,
        exit_price: Optional[float], filled: bool, reason: str,
        realised_pnl: float = 0.0,
    ) -> Dict:
        return {
            "label":        decision.label,
            "direction":    decision.direction,
            "reason":       reason,
            "entry_price":  decision.entry_price,
            "peak_price":   decision.peak_price,
            "trail_level":  decision.trail_level,
            "exit_price":   exit_price,
            "size_usd":     size_usd,
            "realised_pnl": realised_pnl,
            "filled":       filled,
            "opened_at":    opened_at,
        }
