"""Offline NOAA-observations-to-edge checks. Run: python test_weather_refresh.py."""
import datetime
import logging
import os
import tempfile
from unittest.mock import Mock, patch

from core.model import BracketModel, ForecastResult
from core.observations import SGT, parse_daily_observations
from core import observations
from db.ledger import Ledger


def metar(sgt_hhmm, temp, station="WSSS", day=7):
    stamp = datetime.datetime(2026, 10, day, *sgt_hhmm, tzinfo=SGT)
    return {"icaoId": station, "obsTime": int(stamp.timestamp()), "temp": temp}


def history(station="WSSS", high=33):
    # 23:30 SGT on the 6th is the 7th in UTC: must be excluded by the SGT-day window.
    return [metar((23, 30), 40, station, day=6), metar((10, 0), high, station),
            metar((11, 0), None, station), metar((12, 0), 30, station)]


def main():
    logging.basicConfig(level=logging.ERROR)
    date = "2026-10-07"
    now = datetime.datetime(2026, 10, 7, 12, 10, tzinfo=SGT)
    observation = parse_daily_observations(history(), date, now)
    assert observation["high_c"] == 33  # Current temperature is only 30; prior SGT day's 40 ignored.
    assert observation["observed_at"] == "2026-10-07T04:00:00+00:00"
    with patch.object(observations.requests, "get", return_value=Mock(json=history)) as get, \
            patch.object(observations, "parse_daily_observations", side_effect=lambda data, day, clock: parse_daily_observations(data, day, now)):
        assert observations.fetch_daily_observations(date) == observation
        assert get.call_args.kwargs["params"]["ids"] == "WSSS"
    for reports, clock in [(history(station="WRONG"), now), ([metar((10, 0), 33, day=6)], now),
                           ([metar((10, 0), 99)], now), (history(), now - datetime.timedelta(hours=1)),
                           (history(), now + datetime.timedelta(hours=2)), ([], now)]:
        try:
            parse_daily_observations(reports, date, clock)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid, stale, or wrong-date observations were accepted")

    # Settlement calibration: whole SGT day, only once closed and complete.
    full_day = [metar((h, m), 33 if h == 14 else 30) for h in range(24) for m in (0, 30)]
    next_day = [metar((0, 0), 28, day=8)]
    later = datetime.datetime(2026, 10, 8, 1, tzinfo=SGT)
    assert observations.parse_settled_high(full_day + next_day, date, later) == 33
    assert observations.parse_settled_high(full_day, date, later) is None             # day not closed yet
    assert observations.parse_settled_high(full_day[::2] + next_day, date, later) is None  # 24 reports: gappy
    from core.settlement import SettlementEngine
    with patch.object(observations.requests, "get", return_value=Mock(json=lambda: full_day + next_day)), \
            patch.object(observations.datetime, "datetime", Mock(wraps=datetime.datetime, now=lambda tz: later)):
        assert SettlementEngine(None)._fetch_actual_temperature(date) == 33.5  # bracket [33, 34) midpoint

    forecast = ForecastResult(32, 1, "gfs_only")
    model = BracketModel()
    baseline = model.compute(forecast, month=10)
    updated = model.compute(forecast, month=10, observed_high=33)
    assert all(updated[label] == 0 for label in ("29°C", "30°C", "31°C", "32°C"))
    assert updated["33°C"] > baseline["33°C"]
    assert sum(updated.values()) <= 1
    assert all(p == 0 for p in model.compute(forecast, month=10, observed_high=34).values())
    assert model.compute(forecast, month=10, observed_high=None) == baseline

    with patch("requests.get", side_effect=AssertionError("unexpected HTTP")), tempfile.TemporaryDirectory() as temp:
        with patch.dict(os.environ, {"DB_PATH": os.path.join(temp, "weather.db")}), \
                patch("dotenv.load_dotenv"), patch("logging.FileHandler", return_value=logging.NullHandler()):
            import scheduler
            import dashboard_api
        matrix = {label: {"yes": label + "-yes", "no": label + "-no"} for label in baseline}
        scheduler._state.update(market_date=date, token_matrix=matrix)
        scanned = []

        def capture(token_matrix, model_probs, **kwargs):
            scanned.append(model_probs)
            from core.edge import EdgeSignal, MarketPrice
            price = MarketPrice("33°C-yes", 0.4, 0.39, 0.41, 0.02, 100)
            return {"33°C": EdgeSignal("33°C", "33°C-yes", model_probs["33°C"], price,
                                      model_probs["33°C"] - 0.4, 0.08, no_token_id="33°C-no")}

        with patch.object(scheduler, "_sg_now", return_value=now.replace(tzinfo=None)), \
                patch.object(scheduler, "fetch_gfs_forecast", return_value=forecast), \
                patch.object(scheduler, "fetch_daily_observations", return_value=observation) as fetch, \
                patch.object(scheduler, "scan_all_brackets", side_effect=capture):
            scheduler.job_signal_scan()
            fetch.assert_called_once_with(date)
        assert scanned[-1]["32°C"] == 0 and scanned[-1]["33°C"] > baseline["33°C"]
        # Migration is repeatable and scan provenance reaches the dashboard.
        Ledger(scheduler.DB_PATH)
        result = dashboard_api.latest_scan()
        assert result["observed_high"] == 33 and result["observed_at"] == observation["observed_at"]
        assert result["scanned_at"].endswith("Z") and result["scan_stale"] is False
        with scheduler._ledger._conn() as conn:
            conn.execute("UPDATE signal_log SET timestamp = '2000-01-01T00:00:00'")
        assert dashboard_api.latest_scan()["scan_stale"] is True

        # Unavailable observations clear old actionable entries instead of reusing them.
        with patch.object(scheduler, "_sg_now", return_value=now.replace(tzinfo=None)), \
                patch.object(scheduler, "fetch_gfs_forecast", return_value=forecast), \
                patch.object(scheduler, "fetch_daily_observations", side_effect=ValueError("stale")), \
                patch.object(scheduler, "scan_all_brackets") as scan:
            scheduler.job_signal_scan()
            scan.assert_not_called()
        assert not scheduler._state["signals"]
        # Today's observed high must never condition tomorrow's probabilities.
        scheduler._state["market_date"] = "2026-10-08"
        with patch.object(scheduler, "_sg_now", return_value=now.replace(tzinfo=None)), \
                patch.object(scheduler, "fetch_gfs_forecast", return_value=forecast), \
                patch.object(scheduler, "fetch_daily_observations") as fetch, \
                patch.object(scheduler, "scan_all_brackets", side_effect=capture):
            scheduler.job_signal_scan()
            fetch.assert_not_called()
        assert scanned[-1] == baseline
    print("Weather refresh checks passed: observations -> probabilities -> edge scan -> dashboard.")


if __name__ == "__main__":
    main()
