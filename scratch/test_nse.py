import requests
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

def test_nse():
    session = requests.Session()
    session.headers.update(HEADERS)
    print("Visiting NSE homepage...")
    session.get("https://www.nseindia.com", timeout=10)
    time.sleep(1)
    print("Visiting Option Chain page...")
    session.get("https://www.nseindia.com/market-data/option-chain", timeout=10)
    time.sleep(1)
    
    print("Fetching FII/DII data...")
    fii_url = "https://www.nseindia.com/api/fiidiiTradeReact"
    resp = session.get(fii_url, timeout=15)
    print(f"FII/DII Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"FII/DII Data: {resp.text[:500]}")
    
    print("\nFetching PCR data...")
    pcr_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    resp = session.get(pcr_url, timeout=15)
    print(f"PCR Status: {resp.status_code}")
    print(f"PCR Data snippet: {resp.text[:200]}")
    if resp.status_code == 200:
        try:
            js = resp.json()
            print(f"PCR Data keys: {list(js.keys())}")
        except:
            print("PCR is not JSON")

if __name__ == "__main__":
    test_nse()
