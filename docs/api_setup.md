# Fyers API Setup

To use the Fyers API data provider in R3T:

1. **Create an App**: Go to https://fyers.in/web/api-dashboard/user-apps and create a new App. Set the redirect URI to `https://trade.fyers.in/api-login/redirect-uri/index.html`.
2. **Copy Credentials**: Add your App ID and Secret ID to the `.env` file in the project root:
   ```env
   FYERS_APP_ID=WTIW7EKAM0-100
   FYERS_SECRET_ID=JPTDYBYWMZ
   ```
3. **Generate Token**: Run the auth script:
   ```bash
   python scripts/fyers_auth.py
   ```
   Follow the prompts. It will open your browser to log in, and you'll paste the auth code back into the terminal. The access token will be saved to your `.env` file automatically.

