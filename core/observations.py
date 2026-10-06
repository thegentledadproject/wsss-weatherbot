"""
Read WSSS METARs from NOAA's Aviation Weather Center, without API keys.

Markets settle on NOAA's weather.gov/wrh/timeseries?site=wsss "Temp" column.
That page renders METARs via a Synoptic key issued to weather.gov itself, so
we read the same reports from NOAA's public METAR API instead. Whole-degree
°C, matching the settlement precision.
"""
import datetime
import math
from typing import Optional

import requests

SGT = datetime.timezone(datetime.timedelta(hours=8))
NOAA_METAR_URL = "https://aviationweather.gov/api/data/metar"
MAX_LOOKBACK_HOURS = 168  # AWC serves ~7 days of METARs
# WSSS reports every 30 min (48/day). A day with fewer temperature reports may
# be missing its peak, so it is not used to settle calibration.
MIN_SETTLED_REPORTS = 40


def _day_readings(reports: list, market_date: str, now: datetime.datetime):
    """[(utc stamp, °C)] for the SGT calendar day `market_date`; raises on bad data."""
    readings = []
    for report in reports:
        if report.get("icaoId") != "WSSS":
            raise ValueError("NOAA METAR station mismatch")
        stamp = datetime.datetime.fromtimestamp(report["obsTime"], datetime.timezone.utc)
        if stamp.astimezone(SGT).date().isoformat() != market_date:
            continue
        temp = report.get("temp")
        if temp is None:
            continue  # METAR without a temperature group; others still settle the day
        value = float(temp)
        if not math.isfinite(value) or not -50 <= value <= 60:
            raise ValueError("Invalid observed temperature")
        if stamp > now:
            raise ValueError("Future observation timestamp")
        readings.append((stamp, value))
    return readings


def parse_daily_observations(reports: list, market_date: str, now: datetime.datetime) -> dict:
    """Max temp so far over the SGT calendar day `market_date`, from AWC JSON METARs."""
    readings = _day_readings(reports, market_date, now)
    if not readings:
        raise ValueError("No NOAA WSSS observations for the market date")
    latest = max(stamp for stamp, value in readings)
    if now - latest > datetime.timedelta(minutes=90):
        raise ValueError("NOAA WSSS observations are more than 90 minutes old")
    return {"high_c": max(value for stamp, value in readings),
            "observed_at": latest.isoformat()}


def parse_settled_high(reports: list, market_date: str, now: datetime.datetime) -> Optional[float]:
    """
    Whole-day max once the day is over, else None (retry later). Like the
    market, the day counts as closed once a report for the next SGT day exists.
    """
    readings = _day_readings(reports, market_date, now)
    next_day = (datetime.date.fromisoformat(market_date) + datetime.timedelta(days=1)).isoformat()
    if not _day_readings(reports, next_day, now) or len(readings) < MIN_SETTLED_REPORTS:
        return None
    return max(value for stamp, value in readings)


def _fetch(hours: int, timeout: int) -> list:
    response = requests.get(NOAA_METAR_URL, params={"ids": "WSSS", "format": "json", "hours": hours},
                            timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_daily_observations(market_date: str, timeout: int = 15) -> dict:
    # 36h back from now always covers the whole SGT day when scanning it live.
    return parse_daily_observations(_fetch(36, timeout), market_date, datetime.datetime.now(datetime.timezone.utc))


def fetch_settled_high(market_date: str, timeout: int = 15) -> Optional[float]:
    now = datetime.datetime.now(datetime.timezone.utc)
    day_start = datetime.datetime.fromisoformat(market_date).replace(tzinfo=SGT)
    hours = math.ceil((now - day_start).total_seconds() / 3600) + 1
    if hours > MAX_LOOKBACK_HOURS:
        return None  # older than AWC keeps
    return parse_settled_high(_fetch(max(hours, 1), timeout), market_date, now)
