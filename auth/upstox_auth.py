# auth/upstox_auth.py
import os, json, time, threading, webbrowser
from pathlib import Path
from urllib.parse import urlencode
import requests
from flask import Flask, request, redirect, Response
from dotenv import load_dotenv

load_dotenv()
CLIENT_ID = os.getenv("UPSTOX_CLIENT_ID") or ""
CLIENT_SECRET = os.getenv("UPSTOX_CLIENT_SECRET") or ""
REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI") or "http://localhost:8000/callback"
SCOPES = os.getenv("UPSTOX_SCOPES") or "orders read portfolio read marketdata"

AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"

SECRETS_DIR = Path(".secrets")
TOKEN_FILE = SECRETS_DIR / "token.json"

HOST = "127.0.0.1"
PORT = int(os.getenv("UPSTOX_AUTH_PORT") or "8000")

app = Flask(__name__)

def save_token(data: dict):
    SECRETS_DIR.mkdir(exist_ok=True)
    data["_saved_at"] = int(time.time())
    TOKEN_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def exchange_code_for_token(code: str) -> dict:
    payload = {
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(TOKEN_URL, data=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Token exchange failed: {resp.status_code} - {resp.text}")
    return resp.json()

@app.route("/")
def start_auth():
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
    }
    return redirect(f"{AUTH_URL}?{urlencode(params)}")

@app.route("/callback")
def got_code():
    code = request.args.get("code")
    if not code:
        return Response("No `code` in query string.", status=400)
    try:
        token_data = exchange_code_for_token(code)
        save_token(token_data)
        access = token_data.get("access_token", "")
        html = (
            "<h3>✅ Access token saved to <code>.secrets/token.json</code>.</h3>"
            f"<p>Token starts with: <code>{access[:20]}...</code></p>"
            "<p>You can close this tab.</p>"
        )
    except Exception as e:
        html = f"<h3>❌ Error exchanging code:</h3><pre>{e}</pre>"

    # Graceful exit shortly after sending the response (no Flask request objects used here)
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return html

def prompt_if_missing():
    global CLIENT_ID, CLIENT_SECRET, REDIRECT_URI
    if not CLIENT_ID:
        CLIENT_ID = input("Enter UPSTOX_CLIENT_ID: ").strip()
    if not CLIENT_SECRET:
        CLIENT_SECRET = input("Enter UPSTOX_CLIENT_SECRET: ").strip()
    print(f"Using REDIRECT_URI: {REDIRECT_URI}  (set UPSTOX_REDIRECT_URI in .env to change)")

if __name__ == "__main__":
    print("== Upstox One-Click Auth ==")
    print("• Make sure your Upstox app's Redirect URI EXACTLY matches:", REDIRECT_URI)
    prompt_if_missing()
    url = f"http://{HOST}:{PORT}/"
    print(f"• Starting local server on {url}")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    app.run(host=HOST, port=PORT, debug=False)
