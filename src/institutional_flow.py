import os
import json
import time
import requests

CACHE_DIR       = os.path.join("cache", "flow")
CACHE_TTL       = 86400  # 24 hours
FLOW_CACHE_FILE = os.path.join(CACHE_DIR, "fii_dii.json")

NSE_BASE    = "https://www.nseindia.com"
NSE_FII_URL = f"{NSE_BASE}/api/fiidiiTradeReact"

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
    """Open NSE homepage to get session cookies required by data API endpoints."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get(NSE_BASE, timeout=10)
        time.sleep(1)
    except Exception:
        pass
    return session


def fetch_fii_dii_data() -> dict:
    """Return last 30 days of FII/DII net buy/sell (crore). Cached for 24 h."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(FLOW_CACHE_FILE):
        if (time.time() - os.path.getmtime(FLOW_CACHE_FILE)) < CACHE_TTL:
            with open(FLOW_CACHE_FILE) as f:
                return json.load(f)

    try:
        resp = _get_nse_session().get(NSE_FII_URL, timeout=15)
        resp.raise_for_status()
        raw  = resp.json()
    except Exception:
        return {}

    result = {}
    for entry in raw:
        dt = entry.get("date", "")
        if not dt:
            continue
        if dt not in result:
            result[dt] = {"fii_net": 0.0, "dii_net": 0.0}
        
        try:
            cat = entry.get("category", "")
            val = float(str(entry.get("netValue", "0")).replace(",", ""))
            if "FII" in cat:
                result[dt]["fii_net"] = val
            elif "DII" in cat:
                result[dt]["dii_net"] = val
        except (ValueError, TypeError):
            continue

    with open(FLOW_CACHE_FILE, "w") as f:
        json.dump(result, f)
    return result


def compute_flow_features(flow_data: dict):
    """Compute 5-day aggregated FII/DII flow features, normalized to ±1 scale.

    Returns None when flow_data is empty so the caller can flag the run as
    having degraded market-flow context (rather than silently using zeros).
    """
    if not flow_data:
        return None

    from datetime import datetime
    try:
        sorted_dates = sorted(flow_data.keys(), 
                              key=lambda x: datetime.strptime(x, "%d-%b-%Y"), 
                              reverse=True)
    except Exception:
        # Fallback to string sort if date format is unexpected
        sorted_dates = sorted(flow_data.keys(), reverse=True)

    fii_series   = [flow_data[d]["fii_net"] for d in sorted_dates]
    dii_series   = [flow_data[d]["dii_net"] for d in sorted_dates]

    fii_5d      = sum(fii_series[:5])
    dii_5d      = sum(dii_series[:5])
    fii_20d_avg = (sum(fii_series[:20]) / 20) if len(fii_series) >= 20 \
                  else (sum(fii_series) / len(fii_series))
    fii_trend   = (fii_5d / 5) - fii_20d_avg

    scale = 10_000.0  # normalize by ₹10,000 crore → roughly -1 to +1
    return {
        "FII_flow_5d": round(fii_5d   / scale, 4),
        "DII_flow_5d": round(dii_5d   / scale, 4),
        "FII_trend":   round(fii_trend / scale, 4),
    }
