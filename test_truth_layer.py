"""M1 truth-layer gates (settlement, point-in-time data, freshness). Run: python test_truth_layer.py"""
import datetime
import logging
import os
import sqlite3
import tempfile
import time
from unittest.mock import patch

from core import discovery, execution
from db.ledger import Ledger
from test_entry_flow import book, client, signal

# Wording copied from the live Gamma event for 2026-10-06.
DESC = ("This market will resolve to the temperature range that contains the highest temperature "
        "recorded by NOAA at the Singapore Changi Airport Station in degrees Celsius on 6 Oct '26. "
        "The resolution source for this market measures temperatures to whole degrees Celsius (eg, 9°C).")
SOURCE = "https://www.weather.gov/wrh/timeseries?site=wsss"


def market(title, yes, q=None):
    return {"question": q or f"Will the highest temperature in Singapore be {title} on October 6?",
            "groupItemTitle": title, "clobTokenIds": f'["{yes}", "{yes}-no"]'}


def event(**overrides):
    e = {"slug": "highest-temperature-in-singapore-on-october-6-2026", "description": DESC,
         "resolutionSource": SOURCE,
         "markets": [market("28°C or below", "lo"), market("31°C", "y31"), market("32°C", "y32"),
                     market("33°C or higher", "hi")]}
    e.update(overrides)
    return e


def main():
    logging.basicConfig(level=logging.CRITICAL)
    day = "2026-10-06"

    # ── P1: settlement ────────────────────────────────────────────────────────
    found = discovery._extract_markets_from_event(event(), day)
    assert found == {"31°C": {"yes": "y31", "no": "y31-no"}, "32°C": {"yes": "y32", "no": "y32-no"}}, found
    assert "33°C" not in found  # ">= 33" tail must never price as the [33, 34) bracket
    for bad, reason in [(event(description=DESC.replace("Changi Airport", "Paya Lebar")), "station"),
                        (event(resolutionSource="https://wunderground.com/x",
                               description=DESC), "source"),
                        (event(description=DESC.replace("whole degrees", "tenths of degrees")), "precision"),
                        (event(description=DESC.replace("6 Oct", "7 Oct")), "date")]:
        assert discovery.settlement_problem(bad, day) == reason, reason
        try:
            discovery._extract_markets_from_event(bad, day)
            raise AssertionError(f"{reason} mismatch must be INVALID")
        except discovery.SettlementAmbiguous:
            pass
    for bad_markets in ([market("31°C", "a"), market("31°C", "b")],                 # duplicate
                        [market("32°C", "a", q="Will ... be 31°C on October 6?")]):  # title/question disagree
        try:
            discovery._extract_markets_from_event(event(markets=bad_markets), day)
            raise AssertionError("ambiguous bracket mapping must be INVALID")
        except discovery.SettlementAmbiguous:
            pass

    with tempfile.TemporaryDirectory() as temp:
        ledger = Ledger(os.path.join(temp, "t.db"))
        # INVALID never falls back to a matrix cached while the event was valid.
        ledger.upsert_token_matrix("31°C", "cached", "cached-no", day)
        finder = discovery.MarketDiscovery(ledger)
        with patch.object(discovery.requests, "get") as get:
            get.return_value.json.return_value = [event(description="")]
            assert finder.run(day) == {} and "station" in finder.invalid_reason
            get.return_value.json.return_value = [event()]
            assert set(finder.run(day)) == {"31°C", "32°C"} and not finder.invalid_reason

        # ── P2: point-in-time snapshots ───────────────────────────────────────
        fc = type("F", (), dict(source="ensemble_blend", mu=31.2, sigma=0.6, mu_gfs=31.0,
                                mu_ecmwf=31.3, sigma_gfs=0.5, sigma_ecmwf=0.6))()
        s1 = ledger.log_scan("2026-10-06T06:00:00", day, fc, 0.1, 0.9, None, {"31°C": 0.4})
        ledger.log_book(s1, "y31", "scan", [(0.39, 100)], [(0.41, 50), (0.43, 10)], "2026-10-06T06:00:01")
        s2 = ledger.log_scan("2026-10-06T06:15:00", day, fc, 0.1, 0.9,
                             {"high_c": 30.0, "observed_at": "2026-10-05T22:00:00+00:00"}, {"31°C": 0.5})
        assert ledger.scan_as_of(day, "2026-10-06T05:59:59") is None
        known = ledger.scan_as_of(day, "2026-10-06T06:14:59")
        assert known["id"] == s1 and known["model_probs"] == {"31°C": 0.4}
        assert known["books"][0]["asks"] == [[0.41, 50], [0.43, 10]]
        assert ledger.scan_as_of(day, "2026-10-06T07:00:00")["observed_high"] == 30.0
        assert ledger.scan_as_of(day, "2026-10-06T07:00:00")["id"] == s2

        # Execution records the book it decided on and links the trade to its scan.
        sig = signal()
        sig.scan_id = s2
        forced = execution.SizingResult("EXECUTE", "BUY", 1, 0, 0, 0, "test")
        assert execution.ExecutionEngine(client(), ledger, 0).execute(sig, forced, sig.market_date)
        assert ledger.get_open_positions()[0]["scan_id"] == s2
        assert [b["purpose"] for b in ledger.scan_as_of(day, "9999")["books"]] == ["exec"]

        # Pre-M1 databases gain scan_id columns on startup.
        old = os.path.join(temp, "old.db")
        conn = sqlite3.connect(old)
        conn.execute("CREATE TABLE exit_log (id INTEGER PRIMARY KEY, timestamp TEXT, token_id TEXT, "
                     "bracket_label TEXT, direction TEXT, reason TEXT, entry_price REAL, exit_price REAL, "
                     "size_usd REAL, realised_pnl REAL, opened_at TEXT, closed_at TEXT)")
        conn.commit()
        conn.close()
        Ledger(old)
        conn = sqlite3.connect(old)
        assert "scan_id" in {r[1] for r in conn.execute("PRAGMA table_info(exit_log)")}
        conn.close()

    # ── P3: freshness / information integrity ────────────────────────────────
    now = time.time()
    f = execution.freshness
    assert f("scan", now - 60, now) == execution.FRESH
    assert f("scan", now - 901, now) == execution.STALE
    assert f("scan", now + 5, now) == execution.INVALID      # clock mismatch
    assert f("scan", None, now) == execution.INVALID
    assert f("observation", now - 61 * 60, now) == execution.DEGRADED
    assert f("observation", now - 91 * 60, now) == execution.STALE

    def obs_ago(minutes):
        return (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes)).isoformat()

    sig = signal()
    sig.observed_at = obs_ago(70)
    assert execution.signal_freshness(sig, sig.market_date) == execution.DEGRADED
    assert execution.signal_is_current(sig, sig.market_date)
    # Observation fresh at scan (85 min) but stale by execution time → refuse.
    sig.observed_at = obs_ago(95)
    assert execution.signal_freshness(sig, sig.market_date) == execution.STALE
    sig.observed_at = "garbage"
    assert execution.signal_freshness(sig, sig.market_date) == execution.INVALID
    with tempfile.TemporaryDirectory() as temp:
        fake = client()
        sig.observed_at = obs_ago(95)
        forced = execution.SizingResult("EXECUTE", "BUY", 1, 0, 0, 0, "test")
        assert not execution.ExecutionEngine(fake, Ledger(os.path.join(temp, "t.db")), 0).execute(
            sig, forced, sig.market_date)
        fake.post_order.assert_not_called()

    print("Truth-layer gates passed: settlement, point-in-time snapshots, freshness.")


if __name__ == "__main__":
    main()
