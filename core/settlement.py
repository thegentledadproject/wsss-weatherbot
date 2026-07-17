"""
core/settlement.py — S1: Settlement detection + calibration write-back

Two sub-tasks run in Job 4 (every 15 min, 24/7 as of the round-the-clock
scheduler redesign — checks both today's and yesterday's market_date
every cycle, since a position opened late one day can still resolve in
the early hours of the next):

  Task A — Resolution detection:
    Poll Gamma API outcomePrices for each open position's token_id.
    Terminal: outcomePrices[0] > 0.99 (YES) or < 0.01 (NO).
    On terminal state → fetch actual temperature from NEA/Open-Meteo
    historical archive → write calibration residual to ledger.

  Task B — Actual temperature fetch (separate from resolution):
    Uses Open-Meteo historical archive API to get the true observed
    daily max at WSSS. This is written to calibration_logs regardless
    of whether we had an open position — it feeds the trailing bias.

    THIS IS THE FIX for the settlement inference bug:
    We do NOT infer actual temp from the bracket midpoint.
    We fetch it directly from a meteorological archive.
"""

import json
import logging
import datetime
import requests
from typing import Dict, Optional

from db.ledger import Ledger

logger = logging.getLogger("hermes.settlement")

GAMMA_MARKETS_URL   = "https://gamma-api.polymarket.com/markets"
OPEN_METEO_HIST_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=1.3644&longitude=103.9915"
    "&daily=temperature_2m_max"
    "&timezone=Asia%2FSingapore"
    "&start_date={date}&end_date={date}"
)

# NEA Singapore observed weather API (backup for Open-Meteo archive)
# Returns daily station observations including Changi (station S24)
NEA_READINGS_URL = (
    "https://api.data.gov.sg/v1/environment/air-temperature"
    "?date={date}"
)
CHANGI_STATION_ID = "S24"

# Must match the fallback mu in core/model.py's fetch_gfs_forecast().
# Used to detect and skip calibration writes when Job 2 never ran that day.
FALLBACK_MU = 31.5


class SettlementEngine:
    def __init__(self, ledger: Ledger, icao: str = "WSSS", timeout: int = 15):
        self.ledger  = ledger
        self.icao    = icao
        self.timeout = timeout

    def run(self, model_mu: float, market_date: Optional[str] = None) -> Dict:
        """
        Main entry point for Job 4.
        model_mu: the GFS mu used in today's signal (passed from scheduler state)
        market_date: date string "YYYY-MM-DD", defaults to today SGT
        """
        if market_date is None:
            sg_now      = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
            market_date = sg_now.strftime("%Y-%m-%d")

        results = {
            "date":            market_date,
            "positions_checked": 0,
            "positions_settled": 0,
            "actual_temp":     None,
            "calibration_logged": False,
        }

        # Task A: Check all open positions for resolution
        open_positions = self.ledger.get_open_positions()
        results["positions_checked"] = len(open_positions)

        resolved_any = False
        for pos in open_positions:
            # Use the position's own stored SGT market_date (added when this
            # scheduler redesign threaded market_date through record_position)
            # rather than parsing Gamma's `endDate` field — that field's
            # timezone/format isn't guaranteed to match our SGT calendar-date
            # convention, and could silently fail to match any signal_log row
            # (mark_signal_settled's WHERE clause just matches zero rows,
            # no error raised). Falls back to this run()'s market_date for
            # legacy rows recorded before the market_date column existed.
            pos_market_date = pos["market_date"] if pos["market_date"] else market_date
            settled = self._check_resolution(
                token_id      = pos["token_id"],
                bracket_label = pos["bracket_label"],
                market_date   = pos_market_date,
            )
            if settled:
                results["positions_settled"] += 1
                resolved_any = True

        # Task B: Fetch actual observed temperature and write calibration log ONCE per day
        # Two guards added after Jul 2 incident:
        #   1. Idempotency — skip if this ICAO already has a row for today's date.
        #      Without this, Job 4 (every 10 min, 17:00-23:50 SGT) writes a fresh
        #      row on every cycle — 15-20+ duplicate rows per real trading day.
        #   2. Fallback rejection — skip if model_mu == FALLBACK_MU (31.5), which
        #      means Job 2 never ran that day (e.g. discovery found no token
        #      matrix). Logging a residual against a hard prior is meaningless
        #      and corrupts fetch_trailing_bias() for every future day.
        if self.ledger.has_calibration_for_date(self.icao, market_date):
            logger.info(
                f"[SETTLE] Calibration already logged for {market_date} — skipping Task B."
            )
        elif abs(model_mu - FALLBACK_MU) < 0.01:
            logger.warning(
                f"[SETTLE] model_mu={model_mu:.2f} is the fallback prior — "
                f"Job 2 likely never ran today (check discovery/token_matrix). "
                f"Skipping calibration write for {market_date} to avoid corrupting trailing bias."
            )
        else:
            actual_temp = self._fetch_actual_temperature(market_date)
            results["actual_temp"] = actual_temp

            if actual_temp is not None:
                self.ledger.log_outcome(self.icao, model_mu, actual_temp, market_date=market_date)
                results["calibration_logged"] = True
                logger.info(
                    f"[SETTLE] Calibration written: date={market_date} "
                    f"model_mu={model_mu:.2f} actual={actual_temp:.2f} "
                    f"residual={actual_temp - model_mu:+.2f}°C"
                )
            else:
                logger.warning(
                    f"[SETTLE] Could not fetch actual temp for {market_date} — "
                    f"calibration not written. Will retry next cycle."
                )

        return results

    def _check_resolution(self, token_id: str, bracket_label: str, market_date: str = "") -> bool:
        """
        S1: Check Gamma API outcomePrices for terminal state.
        Returns True if the market has resolved and position was closed.

        market_date: the position's own SGT calendar date (from open_positions),
        used for mark_signal_settled instead of parsing Gamma's endDate field.
        """
        url = f"{GAMMA_MARKETS_URL}?clob_token_ids={token_id}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data    = resp.json()
            markets = data if isinstance(data, list) else data.get("markets", [])

            if not markets:
                return False

            market = markets[0]

            # Check resolution flags first
            if not (market.get("closed") or market.get("resolved")):
                return False

            raw_prices = market.get("outcomePrices")
            if not raw_prices:
                return False

            prices    = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            yes_price = float(prices[0])

            if yes_price > 0.99:
                outcome = "YES"
            elif yes_price < 0.01:
                outcome = "NO"
            else:
                return False  # Not yet terminal

            logger.info(
                f"[SETTLE] {bracket_label} resolved {outcome} "
                f"(outcomePrices[0]={yes_price:.4f})"
            )

            self.ledger.close_position(token_id)
            self.ledger.mark_signal_settled(
                date          = market_date or market.get("endDate", "")[:10],
                bracket_label = bracket_label,
                outcome       = outcome,
            )
            return True

        except Exception as e:
            logger.error(f"[SETTLE] Resolution check failed for {bracket_label}: {e}")
            return False

    def _fetch_actual_temperature(self, date: str) -> Optional[float]:
        """
        Fetch the true observed daily maximum temperature at WSSS.

        Primary: Open-Meteo historical archive (reanalysis, reliable after ~6h lag).
        Fallback: NEA data.gov.sg air temperature readings for Changi S24.

        This is the critical fix over v4.2's settlement inference:
        We get the real number, not a bracket midpoint proxy.
        """
        # ── Primary: Open-Meteo archive ───────────────────────────────────────
        try:
            url  = OPEN_METEO_HIST_URL.format(date=date)
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            t_max_arr = data.get("daily", {}).get("temperature_2m_max", [])
            if t_max_arr and t_max_arr[0] is not None:
                actual = float(t_max_arr[0])
                logger.info(f"[SETTLE] Open-Meteo archive: actual max = {actual:.2f}°C")
                return actual

        except Exception as e:
            logger.warning(f"[SETTLE] Open-Meteo archive failed: {e} — trying NEA")

        # ── Fallback: NEA data.gov.sg ─────────────────────────────────────────
        try:
            url  = NEA_READINGS_URL.format(date=date)
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            readings = data.get("items", [])
            changi_max = None

            for item in readings:
                for reading in item.get("readings", []):
                    if reading.get("station_id") == CHANGI_STATION_ID:
                        val = reading.get("value")
                        if val is not None:
                            changi_max = max(changi_max or 0.0, float(val))

            if changi_max is not None:
                logger.info(f"[SETTLE] NEA Changi actual max = {changi_max:.2f}°C")
                return changi_max

        except Exception as e:
            logger.error(f"[SETTLE] NEA fallback also failed: {e}")

        return None

    def count_stuck_positions(self) -> int:
        """
        PM-5: count positions open longer than 28h — for alerting only,
        does not close or delete anything (see db/ledger.py's
        find_stuck_positions() docstring for why deleting was removed).

        NOTE: currently dead code — scheduler.py's Job 3 calls
        ledger.find_stuck_positions() directly, bypassing this wrapper
        entirely. Harmless (just unused), kept here in case a future
        caller wants it via SettlementEngine rather than the raw ledger.
        """
        return len(self.ledger.find_stuck_positions(ttl_hours=28))
