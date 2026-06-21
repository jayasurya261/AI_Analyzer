import os
import json
import time
import statistics
import logging
import requests
from bs4 import BeautifulSoup

CACHE_DIR = os.path.join("cache", "fundamentals")
CACHE_TTL = 7 * 86400  # 7 days — fundamentals change quarterly
LOGGER = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _cache_path(symbol: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{symbol}.json")


def _load_cache(symbol: str):
    path = _cache_path(symbol)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < CACHE_TTL:
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(symbol: str, data: dict):
    with open(_cache_path(symbol), "w") as f:
        json.dump(data, f)


def _parse_number(text: str) -> float:
    """Convert screener.in ratio text like '24.5', '1,234.5', '12.3%' to float."""
    if not text:
        return float('nan')
    try:
        return float(text.replace(',', '').replace('%', '').strip())
    except ValueError:
        return float('nan')


def fetch_fundamentals(symbol: str) -> dict:
    """Fetch key ratios from screener.in. Returns dict or {} on failure."""
    cached = _load_cache(symbol)
    if cached is not None:
        return cached

    try:
        resp = requests.get(f"https://www.screener.in/company/{symbol}/",
                            headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return {}

        soup   = BeautifulSoup(resp.text, "html.parser")
        ratios = {}

        top_ratios = soup.find(id="top-ratios")
        if top_ratios:
            for li in top_ratios.find_all("li"):
                name_el  = li.find("span", class_="name")
                value_el = li.find("span", class_="nowrap") or li.find("span", class_="number")
                if name_el and value_el:
                    ratios[name_el.get_text(strip=True)] = value_el.get_text(strip=True)

        promoter = float('nan')
        sh_table = soup.find("table", class_="data-table")
        if sh_table:
            for row in sh_table.find_all("tr"):
                cells = row.find_all("td")
                if cells and "promoters" in cells[0].get_text(strip=True).lower():
                    vals = [c.get_text(strip=True) for c in cells[1:] if c.get_text(strip=True)]
                    if vals:
                        promoter = _parse_number(vals[-1])
                    break

        result = {
            "pe":           _parse_number(ratios.get("Stock P/E", "")),
            "eps_growth":   _parse_number(ratios.get("EPS last year", "")),
            "promoter_pct": promoter,
            "de_ratio":     _parse_number(ratios.get("Debt to equity", "")),
        }
        _save_cache(symbol, result)
        return result

    except Exception as exc:
        LOGGER.warning("Failed to fetch fundamentals for %s: %s", symbol, exc)
        return {}


def load_fundamentals_for_stocks(symbols: list) -> dict:
    """Batch fetch fundamentals for all symbols. Returns symbol -> dict."""
    results = {}
    for i, sym in enumerate(symbols):
        results[sym] = fetch_fundamentals(sym)
        time.sleep(2.0 if i % 10 == 9 else 0.5)
    return results


def compute_sector_medians(all_fund: dict, sym_sector: dict) -> dict:
    """Compute median P/E per sector for relative valuation."""
    sector_pes = {}
    for sym, data in all_fund.items():
        pe  = data.get('pe', float('nan'))
        sec = sym_sector.get(sym, 'Unknown')
        if pe == pe:  # not NaN
            sector_pes.setdefault(sec, []).append(pe)
    return {sec: statistics.median(pes) for sec, pes in sector_pes.items() if pes}
