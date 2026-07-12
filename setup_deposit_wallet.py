"""
setup_deposit_wallet.py — One-time Polymarket "deposit wallet" provisioning.

BACKGROUND: Polymarket's CLOB V2 (live 2026-04-28) rejects orders from both
Magic/email proxy wallets (signature_type=1) and plain EOAs (signature_type=0)
with "maker address not allowed, please use the deposit wallet flow" — a
requirement change that isn't reflected in older SDK examples. The only
working path now is a "deposit wallet": an ERC-1967 proxy contract, traded
with signature_type=3, where maker and signer are both the deposit wallet's
own address.

This script does three things, gated behind explicit confirmation for any
step that costs gas or touches the chain:
  1. Derive your wallet's deterministic deposit-wallet address (read-only).
  2. Deploy that contract on-chain (--deploy; real gas cost).
  3. Approve the CTF Exchange to pull pUSD from it during order settlement
     (--approve; real gas cost; run this only AFTER you've funded the wallet).

THIS SCRIPT NEVER MOVES YOUR FUNDS. Funding the deposit wallet with USDC.e/
pUSD between steps 2 and 3 is on you — via your own wallet or Polymarket's UI.

Required in .env (or as env vars):
  POLYMARKET_PRIVATE_KEY   your EOA private key (the same one Hermes trades from)
  BUILDER_API_KEY          from polymarket.com/settings?tab=builder
  BUILDER_SECRET           from polymarket.com/settings?tab=builder
  BUILDER_PASS_PHRASE      from polymarket.com/settings?tab=builder
  RELAYER_URL              optional — defaults to production relayer below

Contract addresses (Polygon mainnet, chain_id=137):
  - Deposit wallet factory/implementation: NOT hardcoded here — sourced from
    py_builder_relayer_client's own bundled config (Polymarket/py-builder-
    relayer-client, config.py), so this script inherits whatever Polymarket
    ships in that package rather than a second-hand copy.
  - PUSD_ADDRESS / CTF_EXCHANGE below: cross-checked against PolygonScan's own
    "Polymarket:" contract tags, independent of any single blog/guide.
    VERIFY THEM YOURSELF at polygonscan.com before running --approve — a
    wrong address in an approve() call is real money and cannot be undone.

Usage:
    python setup_deposit_wallet.py            # step 1 only — show the address
    python setup_deposit_wallet.py --deploy    # steps 1+2 — derive and deploy
    python setup_deposit_wallet.py --approve   # step 3 — build + submit approve
                                                #   (run after funding it)

After --approve succeeds, update Hermes's own .env:
    POLYMARKET_FUNDER=<the deposit wallet address printed below>
    POLYMARKET_SIGNATURE_TYPE=3
then re-run: python generate_creds.py
"""

import os
import sys
import time
import argparse

from dotenv import load_dotenv

# Cross-checked against PolygonScan's "Polymarket: pUSD Token" and
# "Polymarket: CTF Exchange" contract tags — verify independently before
# running --approve.
PUSD_ADDRESS  = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
CTF_EXCHANGE  = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"

MAX_UINT256          = 2**256 - 1
CHAIN_ID             = 137
DEFAULT_RELAYER_URL  = "https://relayer-v2.polymarket.com"


def _confirm(prompt: str) -> bool:
    resp = input(f"{prompt} Type 'yes' to continue: ").strip().lower()
    return resp == "yes"


def _build_relayer():
    from py_builder_relayer_client.client import RelayClient
    from py_builder_signing_sdk.config import BuilderConfig
    from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds

    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    if not private_key:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set in .env")
        sys.exit(1)

    builder_key    = os.getenv("BUILDER_API_KEY")
    builder_secret = os.getenv("BUILDER_SECRET")
    builder_pass   = os.getenv("BUILDER_PASS_PHRASE")
    if not (builder_key and builder_secret and builder_pass):
        print("ERROR: BUILDER_API_KEY / BUILDER_SECRET / BUILDER_PASS_PHRASE not set.")
        print("Get these at polymarket.com/settings?tab=builder")
        sys.exit(1)

    relayer_url = os.getenv("RELAYER_URL", DEFAULT_RELAYER_URL)

    builder_config = BuilderConfig(
        local_builder_creds=BuilderApiKeyCreds(
            key=builder_key, secret=builder_secret, passphrase=builder_pass,
        )
    )
    return RelayClient(relayer_url, CHAIN_ID, private_key, builder_config)


def cmd_derive(relayer) -> str:
    from py_builder_relayer_client.models import TransactionType

    addr = relayer.get_expected_deposit_wallet()
    # get_deployed()'s "deployed" flag has been observed to under-report for
    # deposit wallets when queried without a type — pass WALLET_CREATE (the
    # same type used to actually deploy one) to match. Even so, treat this as
    # a best-effort hint, not authoritative: cmd_deploy() below additionally
    # catches the relayer's own "wallet already deployed" error as the real
    # source of truth, since that comes directly from the deploy attempt
    # itself rather than a separate (and evidently sometimes-wrong) lookup.
    already_deployed = relayer.get_deployed(addr, TransactionType.WALLET_CREATE.value)
    print(f"Expected deposit wallet address: {addr}")
    print(f"Already deployed on-chain (best-effort check): {already_deployed}")
    print()
    print("This is a deterministic, counterfactual address. Verify it")
    print("independently (e.g. on polygonscan.com) before funding or")
    print("deploying anything.")
    return addr, already_deployed


def cmd_deploy(relayer):
    from py_builder_relayer_client.exceptions import RelayerApiException

    addr, already_deployed = cmd_derive(relayer)
    if already_deployed:
        print()
        print("Already deployed — nothing to do. Skip to funding it, then run --approve.")
        return

    print()
    print(f"About to DEPLOY a deposit wallet contract at {addr} on Polygon mainnet.")
    print("This is a real on-chain transaction and costs real gas (MATIC).")
    if not _confirm("Proceed?"):
        print("Aborted.")
        return

    try:
        resp = relayer.deploy_deposit_wallet()
    except RelayerApiException as e:
        error_text = str(getattr(e, "error_msg", "")).lower()
        if "already deployed" in error_text:
            print(f"Already deployed at {addr} (confirmed by the deploy endpoint itself —")
            print("the earlier best-effort check above just didn't catch it). No gas spent.")
            print()
            print("NEXT STEP (you do this yourself — this script will not move funds):")
            print(f"  Fund {addr} with USDC.e or pUSD via your own wallet / Polymarket's UI.")
            print("Once funded, run: python setup_deposit_wallet.py --approve")
            return
        raise

    print(f"Submitted. transaction_id={resp.transaction_id} hash={resp.transaction_hash}")
    print("Waiting for confirmation...")
    resp.wait()
    print(f"Deposit wallet deployed at: {addr}")
    print()
    print("NEXT STEP (you do this yourself — this script will not move funds):")
    print(f"  Fund {addr} with USDC.e or pUSD via your own wallet / Polymarket's UI.")
    print("Once funded, run: python setup_deposit_wallet.py --approve")


def cmd_approve(relayer):
    from eth_abi import encode
    from py_builder_relayer_client.models import DepositWalletCall, TransactionType

    addr, already_deployed = cmd_derive(relayer)
    if not already_deployed:
        # get_deployed()'s flag has been observed to under-report (see
        # cmd_derive's comment) -- don't hard-block on it here, since that
        # would wrongly refuse to run --approve against a wallet that IS
        # actually deployed. Warn instead; if it's truly not deployed yet,
        # the batch submission below will fail with a clear error anyway.
        print()
        print("WARNING: the best-effort deployed-check above says 'not deployed'.")
        print("If you already ran --deploy (including an 'already deployed' response),")
        print("this is likely a false negative -- proceeding anyway. If --approve below")
        print("fails with a deployment-related error, run --deploy first for real.")

    print()
    print("Building approve() call:")
    print(f"  pUSD({PUSD_ADDRESS}).approve(spender={CTF_EXCHANGE}, amount=MAX_UINT256)")
    print()
    print("VERIFY these addresses yourself on polygonscan.com before continuing:")
    print(f"  pUSD token:   {PUSD_ADDRESS}")
    print(f"  CTF Exchange: {CTF_EXCHANGE}")
    if not _confirm("Addresses verified, proceed with on-chain approve?"):
        print("Aborted.")
        return

    selector      = bytes.fromhex("095ea7b3")  # approve(address,uint256)
    encoded_args  = encode(["address", "uint256"], [CTF_EXCHANGE, MAX_UINT256])
    calldata      = "0x" + (selector + encoded_args).hex()

    nonce_resp = relayer.get_nonce(relayer.signer.address(), TransactionType.WALLET.value)
    nonce      = str(nonce_resp["nonce"])
    deadline   = str(int(time.time()) + 240)

    resp = relayer.execute_deposit_wallet_batch(
        calls          = [DepositWalletCall(target=PUSD_ADDRESS, value="0", data=calldata)],
        wallet_address = addr,
        nonce          = nonce,
        deadline       = deadline,
    )
    print(f"Submitted. transaction_id={resp.transaction_id} hash={resp.transaction_hash}")
    print("Waiting for confirmation...")
    resp.wait()
    print("Approve confirmed.")
    print()
    print("Deposit wallet setup complete. Update Hermes's .env:")
    print(f"  POLYMARKET_FUNDER={addr}")
    print("  POLYMARKET_SIGNATURE_TYPE=3")
    print("Then re-run: python generate_creds.py")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy",  action="store_true", help="Derive and deploy the deposit wallet")
    parser.add_argument("--approve", action="store_true", help="Approve the CTF Exchange to spend pUSD (run after funding)")
    args = parser.parse_args()

    relayer = _build_relayer()

    if args.approve:
        cmd_approve(relayer)
    elif args.deploy:
        cmd_deploy(relayer)
    else:
        cmd_derive(relayer)


if __name__ == "__main__":
    main()
