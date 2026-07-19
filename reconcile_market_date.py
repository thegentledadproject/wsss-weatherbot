"""
reconcile_market_date.py — Backfill exit_log.market_date against on-chain
trade history.

BACKGROUND: core/position_monitor.py's _execute_exit() used to log every
exit under the CURRENT cycle's market_date (scheduler.py's run()-level
"today") rather than the position's own market_date read from
open_positions at close time (fixed in PR #36). For same-day exits this is
invisible — both values are equal. But a position force-closed by the
stale-position/day-rollover path (opened under yesterday's market_date,
Job 1 discovers tomorrow's market and rolls the scheduler's "today"
forward, then the position gets force-exited) was logged under the ROLLED
date instead of its own, mislabeling exit_log's market_date, the trade's
displayed "market_detail" question, and any market_date-keyed P&L rollup.

PR #36 only fixes writes going forward — it doesn't correct rows already in
the DB. This script finds and corrects those historical rows by treating
Polymarket's own on-chain trade data as ground truth: each exit_log row's
token_id + opened_at is matched against the Polymarket Data API's trade
history for the bot's wallet (that endpoint's "title" field embeds the
bracket's real date, e.g. "...on July 19?"), and market_date is corrected
wherever it disagrees.

Confirmed in production (2026-07-19): cross-checking all 17 exit_log rows
against on-chain data found exactly 2 mismatches (both day-rollover
TIME_EXITs that predate the PR #36 deploy) — ids 4 and 11 in that run.
Every other row already matched on-chain reality.

Read-only (dry-run) by default — prints every row's DB vs on-chain date and
lists mismatches, but writes nothing unless --apply is passed. Always back
up the DB first regardless (this script does not do that for you):
    cp hermes.db hermes.db.bak-$(date +%Y%m%d-%H%M)

Usage:
    python reconcile_market_date.py --wallet 0x...              # dry run
    python reconcile_market_date.py --wallet 0x... --apply       # write fixes
"""

import argparse
import datetime
import re
import sqlite3

import requests

DATA_API_URL = "https://data-api.polymarket.com/trades"
DATE_RE = re.compile(r"on (\w+ \d+)\?")
MATCH_TOLERANCE_SEC = 30  # opened_at vs on-chain trade timestamp, both converted to SGT


def fetch_onchain_trades(wallet: str, limit: int = 500):
    resp = requests.get(DATA_API_URL, params={"user": wallet, "limit": limit}, timeout=15)
    resp.raise_for_status()
    trades = resp.json()

    by_asset = {}
    for t in trades:
        # Data API timestamps are unix epoch UTC; ledger's opened_at is naive
        # UTC too — convert both to SGT for comparison so this doesn't depend
        # on the DB's or this script's local timezone.
        ts_sgt = datetime.datetime.utcfromtimestamp(t["timestamp"]) + datetime.timedelta(hours=8)
        by_asset.setdefault(t["asset"], []).append((ts_sgt, t["price"], t["title"]))
    return by_asset


def onchain_date_for_row(row, by_asset, year: int):
    """Returns (onchain_date_str, time_diff_seconds) for the best-matching
    on-chain trade, or (None, None) if no candidate exists for this token."""
    candidates = by_asset.get(row["token_id"])
    if not candidates:
        return None, None

    try:
        opened_sgt = datetime.datetime.fromisoformat(row["opened_at"]) + datetime.timedelta(hours=8)
    except (ValueError, TypeError):
        return None, None

    best_ts, best_price, best_title, best_diff = None, None, None, None
    for ts, price, title in candidates:
        diff = abs((ts - opened_sgt).total_seconds())
        if best_diff is None or diff < best_diff:
            best_ts, best_price, best_title, best_diff = ts, price, title, diff

    m = DATE_RE.search(best_title or "")
    if not m:
        return None, best_diff
    try:
        onchain_dt = datetime.datetime.strptime(f"{m.group(1)} {year}", "%B %d %Y")
    except ValueError:
        return None, best_diff
    return onchain_dt.strftime("%Y-%m-%d"), best_diff


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wallet", required=True, help="POLYMARKET_FUNDER address (public, not a secret)")
    p.add_argument("--db", default="hermes.db")
    p.add_argument("--year", type=int, default=datetime.datetime.utcnow().year,
                   help="Year to assume when parsing on-chain title dates (they carry no year).")
    p.add_argument("--apply", action="store_true",
                   help="Write corrections. Without this flag, only prints what would change.")
    args = p.parse_args()

    by_asset = fetch_onchain_trades(args.wallet)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, market_date, token_id, bracket_label, opened_at, closed_at "
        "FROM exit_log ORDER BY id"
    ).fetchall()

    mismatches = []
    for r in rows:
        onchain_date, diff = onchain_date_for_row(r, by_asset, args.year)
        if onchain_date is None:
            print(f"id={r['id']} {r['bracket_label']}: no on-chain match found (token={r['token_id'][:12]}...) — skipped")
            continue
        matched = diff is not None and diff < MATCH_TOLERANCE_SEC
        flag = "MISMATCH" if onchain_date != r["market_date"] else ""
        print(f"id={r['id']} {r['bracket_label']}: db={r['market_date']} onchain={onchain_date} "
              f"diff={diff:.0f}s match={matched} {flag}")
        if not matched:
            print(f"  ⚠️  time diff {diff:.0f}s exceeds tolerance — on-chain match may be wrong, verify manually")
        elif onchain_date != r["market_date"]:
            mismatches.append((r["id"], r["market_date"], onchain_date))

    print()
    if not mismatches:
        print("No mismatches found — exit_log.market_date already matches on-chain data.")
        conn.close()
        return

    print(f"{len(mismatches)} mismatch(es) found:")
    for rid, old, new in mismatches:
        print(f"  id={rid}: {old} -> {new}")

    if not args.apply:
        print("\nDry run only — re-run with --apply to write these corrections.")
        conn.close()
        return

    cur = conn.cursor()
    for rid, _old, new in mismatches:
        cur.execute("UPDATE exit_log SET market_date = ? WHERE id = ?", (new, rid))
    conn.commit()
    print(f"\nApplied {len(mismatches)} correction(s).")
    conn.close()


if __name__ == "__main__":
    main()
