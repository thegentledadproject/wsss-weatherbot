"""
generate_creds.py — One-time CLOB API credential generator.

Run this once to generate your CLOB_API_KEY, CLOB_SECRET, CLOB_PASS_PHRASE.
Copy the output into your .env file. You do not need to run this again unless
you want to rotate credentials.

Usage:
    python generate_creds.py

Requirements:
    POLYMARKET_PRIVATE_KEY in your .env (or set as env var directly)
    POLYMARKET_FUNDER in your .env (deposit wallet address from polymarket.com/settings)
    POLYMARKET_SIGNATURE_TYPE in your .env (1 for email/Magic, 0 for MetaMask/EOA)

How to find your FUNDER address:
    1. Go to polymarket.com and sign in
    2. Click your profile → Settings
    3. Your deposit address (starts with 0x) is the FUNDER

How to export your PRIVATE_KEY:
    1. polymarket.com → Settings → Export Private Key
    2. The key that appears is your POLYMARKET_PRIVATE_KEY

SECURITY: Never share your private key. This script reads it from .env and
uses it only to sign a one-time credential derivation message locally.
"""

import os
import re
import sys
from dotenv import load_dotenv


def _validate_private_key(key: str) -> str:
    """
    Pre-validate private key format before attempting any network call.
    A malformed key (missing 0x, wrong length, stray whitespace from a
    copy-paste) previously surfaced as a cryptic low-level error from deep
    inside py-clob-client's signing code — this catches the common cases
    immediately with a clear, actionable message instead.
    Returns the cleaned key, or exits with guidance on failure.
    """
    cleaned = key.strip()
    if not cleaned.startswith("0x"):
        print(f"ERROR: POLYMARKET_PRIVATE_KEY should start with '0x' (got: {cleaned[:6]}...)")
        print("Copy it again from polymarket.com → Settings → Export Private Key")
        sys.exit(1)
    hex_part = cleaned[2:]
    if len(hex_part) != 64 or not re.fullmatch(r"[0-9a-fA-F]+", hex_part):
        print(
            f"ERROR: POLYMARKET_PRIVATE_KEY should be '0x' followed by exactly 64 "
            f"hex characters (got {len(hex_part)} characters after 0x)."
        )
        print("Check for extra whitespace, truncation, or a copy-paste error.")
        sys.exit(1)
    return cleaned


def _parse_signature_type(raw: str) -> int:
    """
    int(os.getenv(...)) with no error handling previously crashed with a raw
    ValueError traceback on any non-numeric value (e.g. a stray "auto" or
    quoted string left in .env) — before any of the script's own friendly
    error messages could ever print. Confirmed via direct test.
    """
    try:
        value = int(raw)
    except ValueError:
        print(f"ERROR: POLYMARKET_SIGNATURE_TYPE must be an integer (got: '{raw}')")
        print("Use 1 for email/Magic wallet, 0 for MetaMask/EOA.")
        sys.exit(1)
    if value not in (0, 1):
        print(f"WARNING: POLYMARKET_SIGNATURE_TYPE={value} is unusual — "
              f"this project's flows are documented for 0 (MetaMask/EOA) and "
              f"1 (email/Magic proxy). Proceeding, but double-check this is intentional.")
    return value


def main():
    load_dotenv()

    private_key_raw = os.getenv("POLYMARKET_PRIVATE_KEY")
    funder           = os.getenv("POLYMARKET_FUNDER")
    sig_type_raw     = os.getenv("POLYMARKET_SIGNATURE_TYPE", "1")

    if not private_key_raw:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set in .env")
        sys.exit(1)

    if not funder:
        print("ERROR: POLYMARKET_FUNDER not set in .env")
        print("Find it at polymarket.com → Settings (your deposit wallet address)")
        sys.exit(1)

    private_key    = _validate_private_key(private_key_raw)
    signature_type = _parse_signature_type(sig_type_raw)

    try:
        from py_clob_client_v2.client import ClobClient
        from py_clob_client_v2.constants import POLYGON
    except ImportError:
        print("ERROR: py-clob-client-v2 not installed.")
        print("Run: pip install py-clob-client-v2")
        sys.exit(1)

    print("Connecting to Polymarket CLOB...")
    print(f"  Funder:         {funder}")
    print(f"  Signature type: {signature_type} "
          f"({'email/Magic proxy' if signature_type == 1 else 'MetaMask/EOA'})")
    print()

    try:
        client = ClobClient(
            host           = "https://clob.polymarket.com",
            chain_id       = POLYGON,
            key            = private_key,
            signature_type = signature_type,
            funder         = funder,
        )
        creds = client.create_or_derive_api_key()

        # Defensive check: confirm all three fields actually came back
        # non-empty before declaring success — a malformed API response
        # could otherwise print "SUCCESS" with blank values that silently
        # fail later when pasted into .env and used for real trading.
        missing = [
            name for name, val in [
                ("api_key", creds.api_key),
                ("api_secret", creds.api_secret),
                ("api_passphrase", creds.api_passphrase),
            ] if not val
        ]
        if missing:
            print(f"ERROR: API returned empty value(s) for: {', '.join(missing)}")
            print("This is unexpected — try again, or check Polymarket API status.")
            sys.exit(1)

        print("SUCCESS — add these to your .env:\n")
        print(f"CLOB_API_KEY={creds.api_key}")
        print(f"CLOB_SECRET={creds.api_secret}")
        print(f"CLOB_PASS_PHRASE={creds.api_passphrase}")
        print()
        print("Done. You do not need to run this again unless rotating credentials.")

    except Exception as e:
        print(f"ERROR: {e}")
        print()
        print("Common causes:")
        print("  - Private key is wrong or missing the 0x prefix")
        print("  - Funder address does not match the private key's associated account")
        print("  - System clock is out of sync (Polymarket requires NTP accuracy)")
        print("    Fix: sudo ntpdate pool.ntp.org")
        print("  - POLYMARKET_SIGNATURE_TYPE mismatch (try 0 if you use MetaMask)")
        sys.exit(1)


if __name__ == "__main__":
    # Guard against import-time side effects: the old bare module-level
    # script executed the entire flow (network calls, printing credentials)
    # the instant anything did `import generate_creds` — e.g. from a future
    # test file or refactor. Wrapping in main() makes this safe to import.
    main()
