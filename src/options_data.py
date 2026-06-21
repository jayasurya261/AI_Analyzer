import os
import json
import time
import requests

CACHE_DIR          = os.path.join("cache", "options")
CACHE_TTL          = 3600  # 1 hour — options data changes intraday
OPTIONS_CACHE_FILE = os.path.join(CACHE_DIR, "nifty_pcr.json")

NSE_BASE        = "https://www.nseindia.com"
NSE_OPTIONS_URL = f"{NSE_BASE}/api/option-chain-indices?symbol=NIFTY"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}


def _get_nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        # Visit home page
        session.get(NSE_BASE, timeout=10)
        time.sleep(1)
        # Visit option chain page to get relevant cookies
        session.get(f"{NSE_BASE}/market-data/option-chain", timeout=10)
        time.sleep(1)
    except Exception:
        pass
    return session


def fetch_nifty_pcr():
    """Return Nifty Put/Call Ratio by total open interest, or None on failure. Cached 1h."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(OPTIONS_CACHE_FILE):
        if (time.time() - os.path.getmtime(OPTIONS_CACHE_FILE)) < CACHE_TTL:
            with open(OPTIONS_CACHE_FILE) as f:
                cached = json.load(f).get("pcr")
                return float(cached) if cached is not None else None

    try:
        data    = _get_nse_session().get(NSE_OPTIONS_URL, timeout=15).json()
        records = data.get("records", {}).get("data", [])

        put_oi  = sum(r["PE"].get("openInterest", 0) for r in records if "PE" in r)
        call_oi = sum(r["CE"].get("openInterest", 0) for r in records if "CE" in r)
        if call_oi <= 0:
            return None

        pcr = put_oi / call_oi
        with open(OPTIONS_CACHE_FILE, "w") as f:
            json.dump({"pcr": pcr}, f)
        return round(pcr, 4)

    except Exception:
        return None


def get_pcr_signal(pcr):
    """Contrarian PCR signal in -1..+1, or None if PCR unavailable.

    High PCR (>=1.5) = extreme bearish hedging = contrarian bullish  = +1.0
    Low  PCR (<=0.5) = extreme complacency     = contrarian bearish  = -1.0
    Linear interpolation between 0.5 and 1.5.
    """
    if pcr is None:
        return None
    if pcr >= 1.5:
        return 1.0
    if pcr <= 0.5:
        return -1.0
    return round((pcr - 1.0) * 2.0, 4)
