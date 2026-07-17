"""
manual_trigger.py — One-off manual trade trigger, bypassing edge/Kelly gating.

PLACES A REAL ORDER WITH REAL FUNDS. This does not simulate, paper-trade, or
ask Polymarket for a quote-only response — it signs and posts a live FOK
market order through the same ExecutionEngine the scheduler uses
(core/execution.py), then records a real open_positions row on a fill.

Use this to manually validate a specific bracket/date after a code change
(e.g. confirming the SELL/NO-token-BUY fix from execution.py actually posts
successfully) without waiting for the scheduler's own cron ticks.

WHAT IT DOES:
  1. Fetches the Gamma event for the given date (same slug logic as
     core/discovery.py) and extracts the given bracket's YES/NO token ids.
  2. Builds an authenticated CLOB client (core/execution.py's build_client()).
  3. Runs the exact same ExecutionEngine.execute() path the scheduler calls
     from job_order_execution() — same VWAP/ghost-book/drift checks, same
     fill parsing, same DB position recording — just with a fixed $ size
     and direction instead of a computed EdgeSignal/SizingResult.
  4. Requires typed confirmation before signing/posting (skip with --yes,
     e.g. for a scripted/cron re-run — use with care).

Usage:
    python manual_trigger.py --bracket 31C --date 2026-07-18 --side YES --size 1.00
    python manual_trigger.py                      # defaults: 31C, 2026-07-18, YES, $1.00

side YES → buys the YES token for the bracket (bracket occurs)
side NO  → buys the NO token for the bracket  (bracket does not occur)

Internally this still maps to direction="BUY"/"SELL" for ExecutionEngine/
SizingResult (that's the codebase-wide convention — see core/edge.py), but
both are a real BUY order now: --side is exposed here instead of --direction
so this script doesn't repeat the exact "SELL" wording that caused the
original naked-short bug this tool exists to validate the fix for.
"""

import sys
import argparse
from types import SimpleNamespace

from dotenv import load_dotenv


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bracket", default="31°C", help='Bracket label, e.g. "31C" or "31°C" (default: 31°C)')
    p.add_argument("--date", default="2026-07-18", help="Market date YYYY-MM-DD (default: 2026-07-18)")
    p.add_argument("--side", default="YES", choices=["YES", "NO"],
                   help="YES = buy the YES token, NO = buy the NO token (default: YES)")
    p.add_argument("--size", type=float, default=1.00, help="USD size to spend (default: 1.00)")
    p.add_argument("--icao", default="WSSS", help="Station code for position bookkeeping (default: WSSS)")
    p.add_argument("--yes", action="store_true", help="Skip the typed confirmation prompt")
    return p.parse_args()


def _normalise_bracket(label: str) -> str:
    """Accept '31C' or '31°C' and normalise to the '31°C' form BRACKET_LABELS uses."""
    label = label.strip().upper().replace(" ", "")
    if label.endswith("C") and "°" not in label:
        label = label[:-1] + "°C"
    return label


def main():
    args = _parse_args()
    bracket   = _normalise_bracket(args.bracket)
    direction = "BUY" if args.side == "YES" else "SELL"  # codebase convention — see module docstring

    load_dotenv()

    from db.ledger import Ledger
    from core.discovery import MarketDiscovery, BRACKET_LABELS
    from core.execution import build_client, ExecutionEngine
    from core.sizing import SizingResult

    if bracket not in BRACKET_LABELS:
        print(f"ERROR: '{bracket}' is not a tradeable bracket label. Valid: {BRACKET_LABELS}")
        sys.exit(1)

    ledger = Ledger()
    discovery = MarketDiscovery(ledger)

    print(f"Fetching token matrix for {args.date} ...")
    matrix = discovery._fetch_from_gamma(args.date)
    if not matrix:
        print(f"ERROR: no event/markets found for {args.date} (Gamma slug/browse fetch returned nothing).")
        sys.exit(1)
    if bracket not in matrix:
        print(f"ERROR: bracket {bracket} not found in {args.date}'s event. Found brackets: {list(matrix.keys())}")
        sys.exit(1)

    ids = matrix[bracket]
    yes_id, no_id = ids["yes"], ids["no"]
    print(f"  {bracket}: yes={yes_id} no={no_id or 'MISSING'}")

    if args.side == "NO" and not no_id:
        print(f"ERROR: no NO token id available for {bracket} on {args.date} — cannot buy NO.")
        sys.exit(1)

    # Persist the matrix so record_position()/position_monitor.py's normal
    # lookups (and any later manual re-runs) see this bracket too.
    ledger.upsert_token_matrix(bracket, yes_id, no_id, args.date)

    print(f"\nAbout to place a REAL order: BUY {args.side} {bracket} (date={args.date}) "
          f"for ${args.size:.2f}")
    print(f"  {args.side} token_id: {yes_id if args.side == 'YES' else no_id}")
    if not args.yes:
        confirm = input('Type "yes" to confirm and post this order: ').strip().lower()
        if confirm != "yes":
            print("Aborted — no order placed.")
            sys.exit(0)

    print("\nAuthenticating CLOB client...")
    try:
        client = build_client()
    except Exception as e:
        print(f"ERROR: could not build/authenticate CLOB client: {e}")
        sys.exit(1)

    engine = ExecutionEngine(client, ledger, vault_usd=args.size, icao=args.icao)

    # ExecutionEngine.execute() only reads bracket_label/direction/token_id/
    # no_token_id off the signal — a SimpleNamespace avoids fighting
    # EdgeSignal's own auto-computed direction/actionable gating logic.
    signal = SimpleNamespace(
        bracket_label=bracket,
        direction=direction,
        token_id=yes_id,
        no_token_id=no_id or None,
    )
    sizing = SizingResult(
        verdict="EXECUTE",
        direction=direction,
        size_usd=round(args.size, 2),
        net_ev=0.0,
        gross_ev=0.0,
        kelly_raw=0.0,
        reason="manual_trigger",
    )

    print(f"\nExecuting BUY {args.side} {bracket} ${sizing.size_usd:.2f} ...")
    filled = engine.execute(signal, sizing, market_date=args.date)

    if filled:
        print(f"\n✓ FILLED — position recorded for {bracket} [{args.side}].")
    else:
        print(f"\n✗ NOT FILLED — order rejected, aborted, or already had an open position. "
              f"Check the [EXEC] log lines above for the reason.")
        sys.exit(1)


if __name__ == "__main__":
    main()
