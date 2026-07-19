"""
check_skew_calibration.py — Sanity-check whether SKEW_ALPHA_TABLE's monsoon
values (esp. July's α=-2.2) are supported by actual settled-day forecast
residuals, or whether they're over-tuning the model into an unrealistically
thin right (warm) tail.

BACKGROUND: a live scan on 2026-07-17 showed the model assigning ~93%
probability to WSSS's July 18 high landing BELOW 29°C, with the 30-33°C
brackets floored to near-zero (see core/model.py's probs[label] =
max(0.001, p)) — driven by α=-2.2 combined with σ_effective=0.866. That's
either a genuine forecast edge, or the skew parameter making the model
falsely certain the actual high can't land a couple degrees warmer than
its mean. This script checks which, using db/ledger.py's calibration_logs
table (real settled outcomes vs what the model predicted that day).

This is read-only — no API calls, no orders, just a query against the
local hermes.db and some scipy stats.

Usage:
    python check_skew_calibration.py --icao WSSS
    python check_skew_calibration.py --icao WSSS --month 7   # July-only rows
"""

import argparse
import sqlite3

import numpy as np
from scipy.stats import skewnorm
from scipy.stats import skew as scipy_skew


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--icao", default="WSSS")
    p.add_argument("--month", type=int, default=None,
                    help="Filter to calibration_logs rows whose market_date falls in this "
                         "month (1-12). Default: all months (more data, less monsoon-specific).")
    p.add_argument("--db", default="hermes.db")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT market_date, model_mu, actual_settled, residual "
            "FROM calibration_logs WHERE icao_code = ? ORDER BY id",
            (args.icao.upper(),),
        ).fetchall()
    finally:
        conn.close()

    if args.month:
        rows = [r for r in rows if r["market_date"] and len(r["market_date"]) >= 7
                and int(r["market_date"][5:7]) == args.month]

    n = len(rows)
    scope = f"month={args.month}" if args.month else "all months"
    print(f"{n} calibration rows for {args.icao} ({scope})")
    if n == 0:
        print("No data — nothing to check yet.")
        return
    if n < 8:
        print("Fewer than 8 rows — treat any skew estimate below as very tentative; "
              "this is a rough sanity check, not a robust refit.")

    for r in rows:
        print(f"  {r['market_date']}: model_mu={r['model_mu']:.2f} "
              f"actual={r['actual_settled']:.2f} residual={r['residual']:+.2f}")

    residuals = np.array([r["residual"] for r in rows], dtype=float)
    mean = residuals.mean()
    std  = residuals.std(ddof=1) if n > 1 else float("nan")
    emp_skew = scipy_skew(residuals) if n > 2 else float("nan")

    print(f"\nresidual (actual - model_mu): mean={mean:+.3f}  std={std:.3f}  "
          f"empirical_skewness={emp_skew:+.3f}")
    print("(negative empirical_skewness = real busts skew toward COLDER than predicted, "
          "supporting a negative alpha; near-zero or positive = the negative skew assumption "
          "may be overstated for this data)")

    warm_busts = int((residuals > 2.0).sum())
    cold_busts = int((residuals < -2.0).sum())
    print(f"\nwarm busts (actual > model_mu + 2°C): {warm_busts}/{n}   "
          f"cold busts (actual < model_mu - 2°C): {cold_busts}/{n}")

    if n >= 8:
        try:
            fit_alpha, fit_loc, fit_scale = skewnorm.fit(residuals)
            print(f"\nEmpirical skewnorm fit to residuals: alpha={fit_alpha:.2f} "
                  f"loc={fit_loc:.2f} scale={fit_scale:.2f}")
            print("Compare against configured values: July=-2.2, DEFAULT=-1.5 — "
                  "if the fit alpha is notably less negative (or positive), the "
                  "configured table is over-tuned toward cold-skew for this station/period.")
        except Exception as e:
            print(f"skewnorm.fit failed (often just needs more data points): {e}")

    if args.month and std == std and std > 0:
        from core.model import SKEW_ALPHA_TABLE, DEFAULT_SKEW_ALPHA
        configured_alpha = SKEW_ALPHA_TABLE.get((args.icao.upper(), args.month), DEFAULT_SKEW_ALPHA)
        p_warm_bust_model = 1 - skewnorm.cdf(2.0, configured_alpha, loc=0, scale=std)
        print(f"\nConfigured alpha={configured_alpha} (at the observed residual std={std:.3f}) "
              f"implies P(residual > +2°C) = {p_warm_bust_model*100:.3f}%")
        print(f"Actual observed rate this period: {warm_busts/n*100:.1f}% ({warm_busts}/{n})")
        if n >= 8 and warm_busts / n > p_warm_bust_model * 3:
            print(
                "WARNING: actual warm-bust rate is several times higher than what alpha implies — "
                "this is evidence the configured skew is over-tuned (too confident the "
                "high won't land warmer than expected)."
            )


if __name__ == "__main__":
    main()
