"""
core/discovery.py — N1: Dynamic market discovery
Job 1 fires daily at 07:30 SGT.

FIXED after Jul 2-3 production incident (0 trades, 2 consecutive days):
  ROOT CAUSE 1: _fetch_by_slug() queried /markets?slug=... — WRONG resource.
    Confirmed via curl: this always returns [] for grouped multi-outcome
    events, because the event-level slug is not a market-level slug.
    Per Polymarket's own agent-skills repo (github.com/Polymarket/agent-skills,
    market-data.md): "Specific market: fetch by slug — GET
    https://gamma-api.polymarket.com/events?slug=..." — the correct
    resource is /events, not /markets.

  ROOT CAUSE 2: _search_gamma() used ?q=<text> against /events, assuming
    Gamma supported full-text search. Confirmed via curl: this parameter
    is silently ignored — a search for "highest temperature Singapore
    July 3" returned an unrelated 2021 NBA market (the API's default
    listing, unfiltered). There is no documented server-side text search
    on the public Gamma API. This entire fallback path never worked.

FIXED WORKFLOW:
  1. Fetch event directly by slug: GET /events?slug=<exact-slug>
     Slug format confirmed from live Polymarket URLs:
     "highest-temperature-in-singapore-on-july-2-2026"
  2. If that returns empty, try the path-style variant: GET /events/slug/<slug>
     (some Gamma deployments expose both forms; confirmed working in
     community docs even when the query-param form is inconsistent).
  3. If both slug forms fail (e.g. Polymarket changed the date format,
     used a slightly different phrasing, or the market hasn't launched
     yet), fall back to PAGINATED BROWSE + client-side filter:
     GET /events?active=true&closed=false&limit=100&offset=N
     and filter events client-side where the title contains "Singapore"
     and "temperature" and a date string matching today. This is slower
     but does not depend on any non-existent search endpoint.
  4. Markets are embedded in the event response by default (confirmed via
     curl — the NBA event response included a full "markets": [...] array
     with no special include parameter needed).
"""

import re
import json
import logging
import datetime
import requests
from typing import Dict, List, Optional

from db.ledger import Ledger

logger = logging.getLogger("hermes.discovery")

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

# Bracket labels we trade — must match what appears in market question text
BRACKET_LABELS = ["29°C", "30°C", "31°C", "32°C", "33°C"]

# Regex patterns to extract temperature from market question
_TEMP_PATTERNS = [
    re.compile(r'\b(2[89]|3[0-9])°?[Cc]\b'),      # "32°C" or "32C"
    re.compile(r'\b(2[89]|3[0-9])\s*degrees\b'),   # "32 degrees"
]

def _extract_temp_label(question: str) -> Optional[str]:
    """Extract a bracket label like '32°C' from a market question string."""
    for pat in _TEMP_PATTERNS:
        m = pat.search(question)
        if m:
            label = f"{m.group(1)}°C"
            if label in BRACKET_LABELS:
                return label
    return None

def _parse_clob_token_ids(market: dict) -> List[str]:
    """
    Extract clobTokenIds from a Gamma API market object.
    Field is a JSON-encoded string in most responses, e.g. '["123...", "456..."]'.
    Index 0 is conventionally the YES token.
    """
    raw = market.get("clobTokenIds")
    if not raw:
        return []
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
        return [str(i) for i in ids if i]
    except (json.JSONDecodeError, TypeError):
        return []

def _today_str() -> str:
    sg_now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    return sg_now.strftime("%Y-%m-%d")

def _build_slugs(date_str: str) -> List[str]:
    """
    Build candidate Polymarket event slugs for the date.
    Confirmed format from live URLs: "highest-temperature-in-singapore-on-july-2-2026"

    We can't be certain whether Polymarket zero-pads the day ("july-2" vs "july-04"),
    so we generate BOTH and let the caller try each. str(dt.day) is used instead of
    strftime("%-d") because %-d is glibc-only and raises ValueError on non-glibc
    platforms (macOS/BSD/Windows), which would crash discovery entirely.
    """
    dt      = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    mon_l   = dt.strftime("%B").lower()     # "july"
    year    = dt.strftime("%Y")             # "2026"
    day_pad = f"{dt.day:02d}"               # "04"
    day_raw = str(dt.day)                   # "4"

    base = "highest-temperature-in-singapore-on"
    slugs = [f"{base}-{mon_l}-{day_raw}-{year}"]
    if day_pad != day_raw:
        slugs.append(f"{base}-{mon_l}-{day_pad}-{year}")
    return slugs

def _extract_markets_from_event(event: dict) -> Dict[str, Dict[str, str]]:
    """
    Given one event object, extract {bracket_label: {"yes": yes_id, "no": no_id}}
    from its embedded markets. clobTokenIds is conventionally [yes_id, no_id] —
    the NO id is needed to open/close NO positions via a real BUY on the NO
    token, rather than a naked (and unsupported) sell of YES.
    """
    found: Dict[str, Dict[str, str]] = {}
    for market in event.get("markets", []):
        question = market.get("question", "") or market.get("title", "") \
                   or market.get("groupItemTitle", "")
        label = _extract_temp_label(question)
        if not label:
            continue
        token_ids = _parse_clob_token_ids(market)
        if len(token_ids) >= 2:
            found[label] = {"yes": token_ids[0], "no": token_ids[1]}
        elif token_ids:
            logger.warning(
                f"[DISCOVERY] {label}: only {len(token_ids)} clobTokenIds — "
                f"no NO token id available, SELL/NO trades will be skipped for it"
            )
            found[label] = {"yes": token_ids[0], "no": ""}
    return found


class MarketDiscovery:
    def __init__(self, ledger: Ledger, timeout: int = 15):
        self.ledger  = ledger
        self.timeout = timeout

    def run(self, date: Optional[str] = None, quiet: bool = False) -> Dict[str, Dict[str, str]]:
        """
        Main entry point. Returns {bracket_label: {"yes": token_id, "no": no_token_id}}
        for the given date (defaults to today's SGT date). Writes results to
        DB. Falls back to last known DB matrix if API fails entirely.

        quiet: suppress the WARNING/ERROR fallback logging. Used for
        speculative "is tomorrow's market live yet?" lookahead probes, where
        a miss is the normal, expected outcome for ~20 hours a day — not
        something to surface loudly every 20-min Job 1 tick.
        """
        if date is None:
            date = _today_str()
        logger.info(f"[DISCOVERY] Running market discovery for {date}")

        token_matrix = self._fetch_from_gamma(date)

        if token_matrix:
            logger.info(
                f"[DISCOVERY] Found {len(token_matrix)} brackets: "
                + ", ".join(f"{k}=yes:{v['yes'][:10]}.. no:{v['no'][:10] or 'MISSING'}.."
                            for k, v in token_matrix.items())
            )
            for label, ids in token_matrix.items():
                self.ledger.upsert_token_matrix(label, ids["yes"], ids["no"], date)
        else:
            log_miss = logger.info if quiet else logger.warning
            log_miss(
                "[DISCOVERY] Gamma API returned no results via slug or browse — "
                "falling back to last known token matrix in DB."
            )
            token_matrix = self.ledger.get_token_matrix(date)
            if not token_matrix:
                slugs = _build_slugs(date)
                log_fail = logger.info if quiet else logger.error
                log_fail(
                    f"[DISCOVERY] No token matrix available for {date}. "
                    f"Tried event slugs {slugs} (query + path form) and paginated "
                    f"browse+filter. Market may not have launched yet, or the slug "
                    f"format has changed. Job 2 will skip until discovery succeeds."
                )

        return token_matrix

    def _fetch_from_gamma(self, today: str) -> Dict[str, Dict[str, str]]:
        """
        Three-stage fetch, in order of speed/reliability:
          1. GET /events?slug=<slug>   (query-param form — confirmed correct resource)
          2. GET /events/slug/<slug>   (path form — fallback if query form is empty)
          3. Paginated browse + client-side title filter (last resort, no search dependency)
        """
        slugs = _build_slugs(today)

        # ── Stage 1: query-param slug fetch (try each slug variant) ──────────
        for slug in slugs:
            try:
                result = self._fetch_by_slug_query(slug)
                if result:
                    logger.info(f"[DISCOVERY] Found via /events?slug={slug}: {list(result.keys())}")
                    return result
            except Exception as e:
                logger.warning(f"[DISCOVERY] /events?slug={slug} failed: {e}")

        # ── Stage 2: path-style slug fetch (try each slug variant) ───────────
        for slug in slugs:
            try:
                result = self._fetch_by_slug_path(slug)
                if result:
                    logger.info(f"[DISCOVERY] Found via /events/slug/{slug}: {list(result.keys())}")
                    return result
            except Exception as e:
                logger.warning(f"[DISCOVERY] /events/slug/{slug} failed: {e}")

        # ── Stage 3: paginated browse + client-side filter ─────────────────
        try:
            result = self._browse_and_filter(today)
            if result:
                logger.info(f"[DISCOVERY] Found via browse+filter: {list(result.keys())}")
                return result
        except Exception as e:
            logger.warning(f"[DISCOVERY] Browse+filter failed: {e}")

        return {}

    def _fetch_by_slug_query(self, slug: str) -> Dict[str, Dict[str, str]]:
        """GET /events?slug=<slug> — the documented, confirmed-correct resource."""
        logger.info(f"[DISCOVERY] Trying /events?slug={slug}")
        resp = requests.get(GAMMA_EVENTS_URL, params={"slug": slug}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        events = data if isinstance(data, list) else data.get("events", [data] if data else [])
        found: Dict[str, Dict[str, str]] = {}
        for event in events:
            found.update(_extract_markets_from_event(event))
        return found

    def _fetch_by_slug_path(self, slug: str) -> Dict[str, Dict[str, str]]:
        """GET /events/slug/<slug> — path-style variant, single event object returned."""
        url = f"{GAMMA_EVENTS_URL}/slug/{slug}"
        logger.info(f"[DISCOVERY] Trying {url}")
        resp = requests.get(url, timeout=self.timeout)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        event = resp.json()
        if not isinstance(event, dict):
            return {}
        return _extract_markets_from_event(event)

    def _browse_and_filter(self, today: str, max_pages: int = 4) -> Dict[str, Dict[str, str]]:
        """
        Last resort: paginate active events and filter client-side.
        No server-side search dependency — Gamma's ?q= parameter is not
        a real filter (confirmed: it's silently ignored), so this is the
        only reliable fallback when the slug doesn't match.

        Filters on title containing "singapore" + "temperature" (case-insensitive).
        With ~50-100 events per page and weather markets created daily, today's
        market is almost always within the first 1-2 pages when active=true.
        """
        dt = datetime.datetime.strptime(today, "%Y-%m-%d")
        # Date fragments to disambiguate which Singapore-temp event is "today's".
        # str(dt.day) not %-d (glibc portability). Include the year so the slug
        # match is anchored: "july-4-2026" cannot be a substring of "july-14-2026".
        mon_l = dt.strftime("%B").lower()   # "july"
        mon_s = dt.strftime("%b").lower()   # "jul"
        day   = str(dt.day)                 # "4"
        year  = dt.strftime("%Y")           # "2026"
        # Slug-form fragments anchored by trailing year — boundary-safe
        slug_fragments = [
            f"{mon_l}-{day}-{year}",        # "july-4-2026"
            f"{mon_l}-{int(day):02d}-{year}",  # "july-04-2026"
        ]
        # Title-form fragments (human text in event.title)
        title_fragments = [
            f"{mon_l} {day}",               # "july 4"
            f"{mon_s} {day}",               # "jul 4"
        ]

        limit = 100
        for page in range(max_pages):
            offset = page * limit
            logger.info(f"[DISCOVERY] Browsing events page {page} (offset={offset})")
            resp = requests.get(
                GAMMA_EVENTS_URL,
                params={"active": "true", "closed": "false", "limit": limit, "offset": offset},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data   = resp.json()
            events = data if isinstance(data, list) else data.get("events", [])

            if not events:
                break  # no more pages

            for event in events:
                title = (event.get("title") or event.get("ticker") or "").lower()
                slug  = (event.get("slug") or "").lower()
                haystack = title + " " + slug

                if "singapore" not in haystack or "temperature" not in haystack:
                    continue

                # Slug match is year-anchored (boundary-safe by construction).
                slug_hit = any(frag in slug for frag in slug_fragments)
                # Title match needs a trailing boundary so "july 4" != "july 14".
                # Require the day to be followed by non-digit or end-of-string.
                title_hit = any(
                    re.search(rf"{re.escape(frag)}(?!\d)", title)
                    for frag in title_fragments
                )
                if not (slug_hit or title_hit):
                    continue

                found = _extract_markets_from_event(event)
                if found:
                    return found

            has_more = data.get("has_more") if isinstance(data, dict) else len(events) == limit
            if not has_more:
                break

        return {}

    def validate_against_live(
        self, token_matrix: Dict[str, Dict[str, str]], date: Optional[str] = None,
    ) -> bool:
        """
        Confirm stored token IDs still appear in the live event for `date`
        (defaults to today's SGT date — pass the actual date being traded,
        e.g. tomorrow's, if Job 1's lookahead is already trading it).
        Returns False and logs ALERT if any token has been removed or changed.
        Only the YES ids are compared — this is just a "market still live"
        sanity check, not a full reconciliation of both sides.
        """
        if not token_matrix:
            return False

        if date is None:
            date = _today_str()
        slugs = _build_slugs(date)
        live: Dict[str, Dict[str, str]] = {}
        try:
            for slug in slugs:
                live = self._fetch_by_slug_query(slug)
                if live:
                    break
            if not live:
                for slug in slugs:
                    live = self._fetch_by_slug_path(slug)
                    if live:
                        break
        except Exception as e:
            logger.warning(f"[DISCOVERY] Validation fetch failed: {e} — skipping check.")
            return True  # non-fatal: don't block trading on a validation network hiccup

        if not live:
            logger.warning("[DISCOVERY] Validation found no live event — skipping check.")
            return True

        live_ids  = {ids["yes"] for ids in live.values()}
        local_ids = {ids["yes"] for ids in token_matrix.values()}

        if not local_ids.issubset(live_ids):
            missing = local_ids - live_ids
            logger.error(
                f"[DISCOVERY] ⚠️  Token mismatch — these IDs are no longer live: {missing}. "
                f"Re-running discovery."
            )
            return False

        return True
