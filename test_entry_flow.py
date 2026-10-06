"""Offline entry regressions. Run: python test_entry_flow.py (no live orders)."""
import datetime
import logging
import math
import os
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core import edge, execution, model, sizing
from db.ledger import Ledger


def book(bid=0.39, ask=0.41, depth=1000):
    return {"bids": [{"price": str(bid), "size": str(depth)}],
            "asks": [{"price": str(ask), "size": str(depth)}]}


def signal(prob=0.60, direction="BUY"):
    price = edge.market_price_from_book("yes", book())
    result = edge.EdgeSignal("31°C", "yes", prob, price, prob - price.mid_price,
                             0.08, no_token_id="no")
    result.direction = direction
    result.market_date = "2026-10-07"
    result.scanned_at = time.time()
    return result


def client(orderbook=None, response=None):
    result = Mock()
    result.get_order_book.return_value = orderbook or book()
    result.get_balance_allowance.return_value = {"balance": "1000000"}
    result.create_market_order.side_effect = lambda args: args
    result.post_order.return_value = response or {
        "status": "matched", "success": True, "makingAmount": "0.99",
        "takingAmount": "2.5", "orderID": "mock-order",
    }
    return result


def main():
    logging.basicConfig(level=logging.ERROR)
    # Block accidental HTTP anywhere in the regression run.
    with patch("requests.get", side_effect=AssertionError("unexpected HTTP")), tempfile.TemporaryDirectory() as temp:
        target = "2026-10-07"
        payload = {"hourly": {
            "time": ["2026-10-06T12:00", target + "T12:00"],
            **{f"temperature_2m_member{i:02}": [99, 30 + i] for i in range(3)},
        }}
        with patch.object(model.requests, "get", return_value=Mock(json=lambda: payload)) as get:
            forecast = model.fetch_gfs_forecast(market_date=target)
            assert forecast.mu == 31 and forecast.source == "ensemble_blend"
            assert all(call.kwargs["params"] == {"start_date": target, "end_date": target}
                       for call in get.call_args_list)
        # Wrong-date ensemble and deterministic data must never be used.
        wrong_day = {"hourly": {"time": ["2026-10-06T12:00"],
                               **{f"temperature_2m_member{i}": [40] for i in range(3)}},
                     "daily": {"time": ["2026-10-06"], "temperature_2m_max": [40],
                               "temperature_2m_min": [25]}}
        with patch.object(model.requests, "get", return_value=Mock(json=lambda: wrong_day)):
            assert model.fetch_gfs_forecast(market_date=target).source == "fallback"
        # Deterministic fallback selects the requested date, not index zero.
        daily = {"daily": {"time": ["2026-10-06", target],
                           "temperature_2m_max": [99, 32], "temperature_2m_min": [90, 28]}}
        with patch.object(model.requests, "get", return_value=Mock(json=lambda: daily)):
            forecast = model.fetch_gfs_forecast(market_date=target)
            assert forecast.mu == 32 and forecast.source == "forecast_spread"

        yes = edge.market_price_from_book("yes", book(0.94, 0.96))
        cheap_no = edge.market_price_from_book("no", book(0.04, 0.06))
        with patch.object(edge, "fetch_market_price", side_effect=[yes, cheap_no]):
            candidate = edge.compute_edge("31°C", "yes", 0.80, no_token_id="no")
            assert candidate.gate_reason == edge.ACTION_SKIP_LOW_PRICE
        yes = edge.market_price_from_book("yes", book(0.59, 0.61))
        for no_book, reason in [(book(0.39, 0.41, 1), edge.ACTION_SKIP_LIQ),
                                (book(0.30, 0.42), edge.ACTION_SKIP_SPRD),
                                (book(0.21, 0.23, 100), edge.ACTION_SKIP_LIQ)]:
            no = edge.market_price_from_book("no", no_book)
            with patch.object(edge, "fetch_market_price", side_effect=[yes, no]):
                assert edge.compute_edge("31°C", "yes", 0.40, no_token_id="no").gate_reason == reason
        no = edge.market_price_from_book("no", book())
        with patch.object(edge, "fetch_market_price", side_effect=[yes, no]):
            candidate = edge.compute_edge("31°C", "yes", 0.40, no_token_id="no")
            assert candidate.direction == "SELL" and candidate.execution_price.token_id == "no"
        sdk_book = SimpleNamespace(bids=[SimpleNamespace(price="0.39", size="1000")],
                                   asks=[SimpleNamespace(price="0.41", size="1000")])
        assert not edge.entry_gate_reason(edge.market_price_from_book("no", sdk_book))

        assert sizing.compute_validation_size(0.97, 0.93).verdict == "HOLD"
        assert sizing.compute_validation_size(0.60, 0.41).verdict == "EXECUTE"
        forced_size = sizing.SizingResult("EXECUTE", "BUY", 1, 0, 0, 0, "test")
        ledger = Ledger(os.path.join(temp, "test.db"))
        # A stable execution book can still have lost the original edge.
        for candidate, current_book in [(signal(0.50), book(0.47, 0.49)),
                                         (signal(0.981), book(0.86, 0.875)),
                                         (signal(0.40, "SELL"), book(0.04, 0.06)),
                                         (signal(0.40, "SELL"), book(0.39, 0.41, 1)),
                                         (signal(0.40, "SELL"), book(0.30, 0.42))]:
            fake = client(current_book)
            assert not execution.ExecutionEngine(fake, ledger, 0).execute(candidate, forced_size, target)
            fake.post_order.assert_not_called()
        assert execution._extract_vwap_ask(book(depth=1), 1) is None
        # Mismatched dates, expired scans, and untagged signals cannot open trades.
        expired, mismatched, untagged = signal(), signal(), signal()
        expired.scanned_at = time.time() - 901
        mismatched.market_date = "2026-10-08"
        untagged.scanned_at = None
        for candidate in [expired, mismatched, untagged]:
            fake = client()
            assert not execution.ExecutionEngine(fake, ledger, 0).execute(candidate, forced_size, target)
            fake.post_order.assert_not_called()
        assert not execution.signal_is_current(signal(), "2026-10-08")
        assert not execution.signal_is_current(untagged, target)
        # A signal that expires while balance synchronization runs is also rejected.
        candidate = signal()
        fake = client()
        fake.update_balance_allowance.side_effect = lambda _: setattr(candidate, "scanned_at", time.time() - 901)
        assert not execution.ExecutionEngine(fake, ledger, 0).execute(candidate, forced_size, target)
        fake.post_order.assert_not_called()

        # Confirmed amounts, not the quoted 0.41 VWAP or requested $1, reach SQLite.
        for direction, prob, token in [("BUY", 0.60, "yes"), ("SELL", 0.40, "no")]:
            fake = client()
            assert execution.ExecutionEngine(fake, ledger, 0).execute(signal(prob, direction), forced_size, target)
            row = next(row for row in ledger.get_open_positions() if row["token_id"] == token)
            assert math.isclose(row["entry_price"], 0.99 / 2.5)
            assert math.isclose(row["size_usd"] / row["entry_price"], 2.5)
            assert row["market_date"] == target
            assert fake.post_order.call_args.args[0].token_id == token
            ledger.close_position(token)
        rejected = client(response={"status": "unmatched", "success": False})
        assert not execution.ExecutionEngine(rejected, ledger, 0).execute(signal(), forced_size, target)
        assert not ledger.get_open_positions()
        malformed = client(response={"status": "matched", "success": True, "size_matched": "2.5"})
        try:
            execution.ExecutionEngine(malformed, ledger, 0).execute(signal(), forced_size, target)
        except RuntimeError as exc:
            assert "lacks valid amounts" in str(exc)
        else:
            raise AssertionError("Missing fill amounts must never become estimated positions")
        assert not ledger.get_open_positions()
        # The explicit manual bypass still uses selected-token checks and actual fills.
        manual = SimpleNamespace(bracket_label="31°C", direction="BUY", token_id="yes", no_token_id="no")
        assert execution.ExecutionEngine(client(), ledger, 0).execute(manual, forced_size, target, bypass_edge_checks=True)
        ledger.close_position("yes")

        # Import scheduler with isolated DB/log settings; never authenticate a client.
        with patch.dict(os.environ, {"DB_PATH": os.path.join(temp, "scheduler.db")}), \
                patch("dotenv.load_dotenv"), patch("logging.FileHandler", return_value=logging.NullHandler()):
            import scheduler
        matrix = {"31°C": {"yes": "yes", "no": "no"}}
        scheduler._state.update(token_matrix=matrix, market_date=target, signals={"31°C": signal()})
        discovery = Mock()
        discovery.run.return_value = matrix
        discovery.validate_against_live.return_value = True
        with patch.object(scheduler, "MarketDiscovery", return_value=discovery), \
                patch.object(scheduler, "_sg_now", return_value=datetime.datetime(2026, 10, 7, 23)):
            scheduler.job_market_discovery()
        assert scheduler._state["market_date"] == "2026-10-08" and not scheduler._state["signals"]
        scheduler._state.update(market_date=target, token_matrix=matrix)
        with patch.object(scheduler, "fetch_gfs_forecast", return_value=model.ForecastResult(31, 1, "gfs_only")) as fetch, \
                patch.object(scheduler, "scan_all_brackets", return_value={"31°C": signal()}):
            scheduler.job_signal_scan()
            fetch.assert_called_once_with(market_date=target)
        assert scheduler._state["signals"]["31°C"].market_date == target
        assert execution.signal_is_current(scheduler._state["signals"]["31°C"], target)
        scheduler._client = Mock()
        scheduler._state["market_date"] = "2026-10-08"
        with patch.object(scheduler, "ExecutionEngine") as engine:
            scheduler.job_order_execution()
            engine.return_value.execute.assert_not_called()
        with patch.object(scheduler, "fetch_gfs_forecast", side_effect=RuntimeError("forecast failed")):
            try:
                scheduler.job_signal_scan()
            except RuntimeError:
                pass
        assert not scheduler._state["signals"]
    print("All five entry regressions passed (mocked HTTP/orders, temporary SQLite).")


if __name__ == "__main__":
    main()
