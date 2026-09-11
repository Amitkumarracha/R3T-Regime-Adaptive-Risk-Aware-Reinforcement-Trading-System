#!/usr/bin/env python3
"""
scripts/fyers_auth.py — Fyers OAuth Token Generator

Run this ONCE to get your access token:
    python scripts/fyers_auth.py

It will:
1. Open the Fyers authorization URL in your browser
2. Ask you to paste the auth_code from the redirect URL
3. Generate and save the access_token to your .env file

Prerequisites:
- Fill FYERS_APP_ID and FYERS_SECRET_ID in your .env file first
- pip install fyers-apiv3 python-dotenv
"""

import os
import sys
import hashlib
from pathlib import Path

# Load .env from project root
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"

try:
    from dotenv import load_dotenv, set_key
    load_dotenv(env_file)
except ImportError:
    print("[ERROR] python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

try:
    from fyers_apiv3 import fyersModel
except ImportError:
    print("[ERROR] fyers-apiv3 not installed. Run: pip install fyers-apiv3")
    sys.exit(1)


def generate_token():
    app_id = os.getenv("FYERS_APP_ID")
    secret_id = os.getenv("FYERS_SECRET_ID")
    redirect_uri = os.getenv(
        "FYERS_REDIRECT_URI",
        "https://trade.fyers.in/api-login/redirect-uri/index.html",
    )

    if not app_id or not secret_id:
        print("[ERROR] FYERS_APP_ID or FYERS_SECRET_ID not set in .env")
        print(f"  Edit: {env_file}")
        sys.exit(1)

    # ── Step 1: Build Auth URL ────────────────────────────────────────────────
    session = fyersModel.SessionModel(
        client_id=app_id,
        secret_key=secret_id,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )

    auth_url = session.generate_authcode()
    print("\n" + "=" * 60)
    print("  FYERS AUTHORIZATION")
    print("=" * 60)
    print("\nOpen this URL in your browser and log in:\n")
    print(f"  {auth_url}\n")

    try:
        import webbrowser
        webbrowser.open(auth_url)
        print("  (Browser opened automatically)\n")
    except Exception:
        pass

    print("After login, you will be redirected to a URL like:")
    print("  https://trade.fyers.in/...?auth_code=XXXX&state=...")
    print("\nCopy the auth_code value from the URL.")

    # ── Step 2: Get auth code from user ───────────────────────────────────────
    auth_code = input("\nPaste auth_code here: ").strip()

    if not auth_code:
        print("[ERROR] No auth code entered. Exiting.")
        sys.exit(1)

    # ── Step 3: Generate access token ────────────────────────────────────────
    # Fyers uses SHA-256 of (app_id:secret) as the app hash
    app_hash = hashlib.sha256(f"{app_id}:{secret_id}".encode()).hexdigest()

    session.set_token(auth_code)
    response = session.generate_token()

    if response.get("code") == 200 or "access_token" in response:
        access_token = response["access_token"]
        print(f"\n[SUCCESS] Access token generated.")

        # Save to .env
        set_key(str(env_file), "FYERS_ACCESS_TOKEN", access_token)
        print(f"[SAVED] FYERS_ACCESS_TOKEN written to {env_file}")
        print("\nYou can now use the Fyers data provider.")
        return access_token
    else:
        print(f"\n[ERROR] Token generation failed: {response}")
        sys.exit(1)


if __name__ == "__main__":
    generate_token()
