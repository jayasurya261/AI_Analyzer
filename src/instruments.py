import os
import gzip
import shutil
import time
import logging
import requests
import pandas as pd

CACHE_DIR        = os.path.join("cache", "instruments")
CACHE_TTL_SECONDS = 86400  # 24 hours
LOGGER = logging.getLogger(__name__)

TRAINING_STOCKS = [
    # Energy (9)
    {"name": "Reliance Industries", "instrument_key": "NSE_EQ|INE002A01018",  "tradingsymbol": "RELIANCE",   "sector": "Energy"},
    {"name": "ONGC",                "instrument_key": "NSE_EQ|INE213A01029",  "tradingsymbol": "ONGC",       "sector": "Energy"},
    {"name": "NTPC",                "instrument_key": "NSE_EQ|INE733E01010",  "tradingsymbol": "NTPC",       "sector": "Energy"},
    {"name": "Power Grid",          "instrument_key": "NSE_EQ|INE752E01010",  "tradingsymbol": "POWERGRID",  "sector": "Energy"},
    {"name": "Adani Green Energy",  "instrument_key": "NSE_EQ|INE364U01010",  "tradingsymbol": "ADANIGREEN", "sector": "Energy"},
    {"name": "Tata Power",          "instrument_key": "NSE_EQ|INE245A01021",  "tradingsymbol": "TATAPOWER",  "sector": "Energy"},
    {"name": "NHPC",                "instrument_key": "NSE_EQ|INE848E01016",  "tradingsymbol": "NHPC",       "sector": "Energy"},
    {"name": "CESC",                "instrument_key": "NSE_EQ|INE486A01013",  "tradingsymbol": "CESC",       "sector": "Energy"},
    {"name": "Torrent Power",       "instrument_key": "NSE_EQ|INE813H01021",  "tradingsymbol": "TORNTPOWER", "sector": "Energy"},
    # IT (11)
    {"name": "TCS",                 "instrument_key": "NSE_EQ|INE467B01029",  "tradingsymbol": "TCS",        "sector": "IT"},
    {"name": "Infosys",             "instrument_key": "NSE_EQ|INE009A01021",  "tradingsymbol": "INFY",       "sector": "IT"},
    {"name": "Wipro",               "instrument_key": "NSE_EQ|INE075A01022",  "tradingsymbol": "WIPRO",      "sector": "IT"},
    {"name": "HCL Technologies",    "instrument_key": "NSE_EQ|INE860A01027",  "tradingsymbol": "HCLTECH",    "sector": "IT"},
    {"name": "Tech Mahindra",       "instrument_key": "NSE_EQ|INE669C01036",  "tradingsymbol": "TECHM",      "sector": "IT"},
    {"name": "LTIMindtree",         "instrument_key": "NSE_EQ|INE214T01019",  "tradingsymbol": "LTIM",       "sector": "IT"},
    {"name": "Mphasis",             "instrument_key": "NSE_EQ|INE356A01018",  "tradingsymbol": "MPHASIS",    "sector": "IT"},
    {"name": "Persistent Systems",  "instrument_key": "NSE_EQ|INE262H01021",  "tradingsymbol": "PERSISTENT", "sector": "IT"},
    {"name": "Coforge",             "instrument_key": "NSE_EQ|INE591G01017",  "tradingsymbol": "COFORGE",    "sector": "IT"},
    {"name": "L&T Technology",      "instrument_key": "NSE_EQ|INE010V01017",  "tradingsymbol": "LTTS",       "sector": "IT"},
    {"name": "Oracle Financial",    "instrument_key": "NSE_EQ|INE881D01027",  "tradingsymbol": "OFSS",       "sector": "IT"},
    # Banking & Finance (13)
    {"name": "HDFC Bank",           "instrument_key": "NSE_EQ|INE040A01034",  "tradingsymbol": "HDFCBANK",   "sector": "Banking"},
    {"name": "ICICI Bank",          "instrument_key": "NSE_EQ|INE090A01021",  "tradingsymbol": "ICICIBANK",  "sector": "Banking"},
    {"name": "SBI",                 "instrument_key": "NSE_EQ|INE062A01020",  "tradingsymbol": "SBIN",       "sector": "Banking"},
    {"name": "Kotak Mahindra Bank", "instrument_key": "NSE_EQ|INE237A01028",  "tradingsymbol": "KOTAKBANK",  "sector": "Banking"},
    {"name": "Axis Bank",           "instrument_key": "NSE_EQ|INE238A01034",  "tradingsymbol": "AXISBANK",   "sector": "Banking"},
    {"name": "Bajaj Finance",       "instrument_key": "NSE_EQ|INE296A01024",  "tradingsymbol": "BAJFINANCE", "sector": "Banking"},
    {"name": "Bajaj Finserv",       "instrument_key": "NSE_EQ|INE918I01026",  "tradingsymbol": "BAJAJFINSV", "sector": "Banking"},
    {"name": "IndusInd Bank",       "instrument_key": "NSE_EQ|INE095A01012",  "tradingsymbol": "INDUSINDBK", "sector": "Banking"},
    {"name": "Canara Bank",         "instrument_key": "NSE_EQ|INE476A01014",  "tradingsymbol": "CANBK",      "sector": "Banking"},
    {"name": "PNB",                 "instrument_key": "NSE_EQ|INE160A01022",  "tradingsymbol": "PNB",        "sector": "Banking"},
    {"name": "Bank of Baroda",      "instrument_key": "NSE_EQ|INE028A01039",  "tradingsymbol": "BANKBARODA", "sector": "Banking"},
    {"name": "Federal Bank",        "instrument_key": "NSE_EQ|INE171A01029",  "tradingsymbol": "FEDERALBNK", "sector": "Banking"},
    {"name": "IDFC First Bank",     "instrument_key": "NSE_EQ|INE092T01019",  "tradingsymbol": "IDFCFIRSTB", "sector": "Banking"},
    # FMCG (10)
    {"name": "Hindustan Unilever",  "instrument_key": "NSE_EQ|INE030A01027",  "tradingsymbol": "HINDUNILVR", "sector": "FMCG"},
    {"name": "ITC",                 "instrument_key": "NSE_EQ|INE154A01025",  "tradingsymbol": "ITC",        "sector": "FMCG"},
    {"name": "Nestle India",        "instrument_key": "NSE_EQ|INE239A01016",  "tradingsymbol": "NESTLEIND",  "sector": "FMCG"},
    {"name": "Britannia",           "instrument_key": "NSE_EQ|INE216A01030",  "tradingsymbol": "BRITANNIA",  "sector": "FMCG"},
    {"name": "Dabur India",         "instrument_key": "NSE_EQ|INE016A01026",  "tradingsymbol": "DABUR",      "sector": "FMCG"},
    {"name": "Marico",              "instrument_key": "NSE_EQ|INE196A01026",  "tradingsymbol": "MARICO",     "sector": "FMCG"},
    {"name": "Colgate",             "instrument_key": "NSE_EQ|INE259A01022",  "tradingsymbol": "COLPAL",     "sector": "FMCG"},
    {"name": "Godrej Consumer",     "instrument_key": "NSE_EQ|INE102D01028",  "tradingsymbol": "GODREJCP",   "sector": "FMCG"},
    {"name": "Tata Consumer",       "instrument_key": "NSE_EQ|INE192A01025",  "tradingsymbol": "TATACONSUM", "sector": "FMCG"},
    {"name": "Emami",               "instrument_key": "NSE_EQ|INE548C01032",  "tradingsymbol": "EMAMILTD",   "sector": "FMCG"},
    # Auto (10)
    {"name": "Maruti Suzuki",       "instrument_key": "NSE_EQ|INE585B01010",  "tradingsymbol": "MARUTI",     "sector": "Auto"},
    {"name": "Tata Motors",         "instrument_key": "NSE_EQ|INE155A01022",  "tradingsymbol": "TATAMOTORS", "sector": "Auto"},
    {"name": "Mahindra & Mahindra", "instrument_key": "NSE_EQ|INE101A01026",  "tradingsymbol": "M&M",        "sector": "Auto"},
    {"name": "Bajaj Auto",          "instrument_key": "NSE_EQ|INE917I01010",  "tradingsymbol": "BAJAJ-AUTO", "sector": "Auto"},
    {"name": "Hero MotoCorp",       "instrument_key": "NSE_EQ|INE158A01026",  "tradingsymbol": "HEROMOTOCO", "sector": "Auto"},
    {"name": "Eicher Motors",       "instrument_key": "NSE_EQ|INE066A01021",  "tradingsymbol": "EICHERMOT",  "sector": "Auto"},
    {"name": "TVS Motor",           "instrument_key": "NSE_EQ|INE494B01023",  "tradingsymbol": "TVSMOTOR",   "sector": "Auto"},
    {"name": "Ashok Leyland",       "instrument_key": "NSE_EQ|INE208A01029",  "tradingsymbol": "ASHOKLEY",   "sector": "Auto"},
    {"name": "MRF",                 "instrument_key": "NSE_EQ|INE883A01011",  "tradingsymbol": "MRF",        "sector": "Auto"},
    {"name": "Cummins India",       "instrument_key": "NSE_EQ|INE298A01020",  "tradingsymbol": "CUMMINSIND", "sector": "Auto"},
    # Pharma (10)
    {"name": "Sun Pharma",          "instrument_key": "NSE_EQ|INE044A01036",  "tradingsymbol": "SUNPHARMA",  "sector": "Pharma"},
    {"name": "Dr. Reddy's",         "instrument_key": "NSE_EQ|INE089A01031",  "tradingsymbol": "DRREDDY",    "sector": "Pharma"},
    {"name": "Cipla",               "instrument_key": "NSE_EQ|INE059A01026",  "tradingsymbol": "CIPLA",      "sector": "Pharma"},
    {"name": "Divis Laboratories",  "instrument_key": "NSE_EQ|INE361B01024",  "tradingsymbol": "DIVISLAB",   "sector": "Pharma"},
    {"name": "Apollo Hospitals",    "instrument_key": "NSE_EQ|INE437A01024",  "tradingsymbol": "APOLLOHOSP", "sector": "Pharma"},
    {"name": "Lupin",               "instrument_key": "NSE_EQ|INE326A01037",  "tradingsymbol": "LUPIN",      "sector": "Pharma"},
    {"name": "Torrent Pharma",      "instrument_key": "NSE_EQ|INE685A01028",  "tradingsymbol": "TORNTPHARM", "sector": "Pharma"},
    {"name": "Alkem Laboratories",  "instrument_key": "NSE_EQ|INE540L01014",  "tradingsymbol": "ALKEM",      "sector": "Pharma"},
    {"name": "Mankind Pharma",      "instrument_key": "NSE_EQ|INE634S01028",  "tradingsymbol": "MANKIND",    "sector": "Pharma"},
    {"name": "Abbott India",        "instrument_key": "NSE_EQ|INE358A01014",  "tradingsymbol": "ABBOTINDIA", "sector": "Pharma"},
    # Telecom (4)
    {"name": "Bharti Airtel",       "instrument_key": "NSE_EQ|INE397D01024",  "tradingsymbol": "BHARTIARTL", "sector": "Telecom"},
    {"name": "Vodafone Idea",       "instrument_key": "NSE_EQ|INE669E01016",  "tradingsymbol": "IDEA",       "sector": "Telecom"},
    {"name": "Indus Towers",        "instrument_key": "NSE_EQ|INE121J01017",  "tradingsymbol": "INDUSTOWER", "sector": "Telecom"},
    {"name": "HFCL",                "instrument_key": "NSE_EQ|INE045A01017",  "tradingsymbol": "HFCL",       "sector": "Telecom"},
    # Metals & Mining (9)
    {"name": "Tata Steel",          "instrument_key": "NSE_EQ|INE081A01012",  "tradingsymbol": "TATASTEEL",  "sector": "Metals"},
    {"name": "JSW Steel",           "instrument_key": "NSE_EQ|INE019A01038",  "tradingsymbol": "JSWSTEEL",   "sector": "Metals"},
    {"name": "Hindalco",            "instrument_key": "NSE_EQ|INE038A01020",  "tradingsymbol": "HINDALCO",   "sector": "Metals"},
    {"name": "Coal India",          "instrument_key": "NSE_EQ|INE522F01014",  "tradingsymbol": "COALINDIA",  "sector": "Metals"},
    {"name": "Vedanta",             "instrument_key": "NSE_EQ|INE205A01025",  "tradingsymbol": "VEDL",       "sector": "Metals"},
    {"name": "SAIL",                "instrument_key": "NSE_EQ|INE114A01011",  "tradingsymbol": "SAIL",       "sector": "Metals"},
    {"name": "NMDC",                "instrument_key": "NSE_EQ|INE584A01023",  "tradingsymbol": "NMDC",       "sector": "Metals"},
    {"name": "Jindal Steel",        "instrument_key": "NSE_EQ|INE749A01030",  "tradingsymbol": "JINDALSTEL", "sector": "Metals"},
    {"name": "APL Apollo Tubes",    "instrument_key": "NSE_EQ|INE702C01027",  "tradingsymbol": "APLAPOLLO",  "sector": "Metals"},
    # Cement (6)
    {"name": "UltraTech Cement",    "instrument_key": "NSE_EQ|INE481G01011",  "tradingsymbol": "ULTRACEMCO", "sector": "Cement"},
    {"name": "Grasim Industries",   "instrument_key": "NSE_EQ|INE047A01021",  "tradingsymbol": "GRASIM",     "sector": "Cement"},
    {"name": "Shree Cement",        "instrument_key": "NSE_EQ|INE070A01015",  "tradingsymbol": "SHREECEM",   "sector": "Cement"},
    {"name": "ACC",                 "instrument_key": "NSE_EQ|INE012A01025",  "tradingsymbol": "ACC",        "sector": "Cement"},
    {"name": "Ambuja Cements",      "instrument_key": "NSE_EQ|INE079A01024",  "tradingsymbol": "AMBUJACEM",  "sector": "Cement"},
    {"name": "Dalmia Bharat",       "instrument_key": "NSE_EQ|INE00R701025",  "tradingsymbol": "DALBHARAT",  "sector": "Cement"},
    # Infra (6)
    {"name": "Larsen & Toubro",     "instrument_key": "NSE_EQ|INE018A01030",  "tradingsymbol": "LT",         "sector": "Infra"},
    {"name": "Adani Ports",         "instrument_key": "NSE_EQ|INE742F01042",  "tradingsymbol": "ADANIPORTS", "sector": "Infra"},
    {"name": "DLF",                 "instrument_key": "NSE_EQ|INE271C01023",  "tradingsymbol": "DLF",        "sector": "Infra"},
    {"name": "Godrej Properties",   "instrument_key": "NSE_EQ|INE484J01027",  "tradingsymbol": "GODREJPROP", "sector": "Infra"},
    {"name": "Siemens India",       "instrument_key": "NSE_EQ|INE003A01024",  "tradingsymbol": "SIEMENS",    "sector": "Infra"},
    {"name": "ABB India",           "instrument_key": "NSE_EQ|INE117A01022",  "tradingsymbol": "ABB",        "sector": "Infra"},
    # Consumer & Retail (9)
    {"name": "Titan Company",       "instrument_key": "NSE_EQ|INE280A01028",  "tradingsymbol": "TITAN",      "sector": "Consumer"},
    {"name": "Asian Paints",        "instrument_key": "NSE_EQ|INE021A01026",  "tradingsymbol": "ASIANPAINT", "sector": "Consumer"},
    {"name": "Zomato",              "instrument_key": "NSE_EQ|INE758T01015",  "tradingsymbol": "ZOMATO",     "sector": "Consumer"},
    {"name": "Avenue Supermarts",   "instrument_key": "NSE_EQ|INE192R01011",  "tradingsymbol": "DMART",      "sector": "Consumer"},
    {"name": "Nykaa",               "instrument_key": "NSE_EQ|INE388Y01029",  "tradingsymbol": "NYKAA",      "sector": "Consumer"},
    {"name": "PB Fintech",          "instrument_key": "NSE_EQ|INE417T01026",  "tradingsymbol": "POLICYBZR",  "sector": "Consumer"},
    {"name": "Delhivery",           "instrument_key": "NSE_EQ|INE148O01028",  "tradingsymbol": "DELHIVERY",  "sector": "Consumer"},
    {"name": "Info Edge",           "instrument_key": "NSE_EQ|INE663F01024",  "tradingsymbol": "NAUKRI",     "sector": "Consumer"},
    {"name": "Pidilite Industries", "instrument_key": "NSE_EQ|INE318A01026",  "tradingsymbol": "PIDILITIND", "sector": "Consumer"},
    # Diversified (5)
    {"name": "Muthoot Finance",     "instrument_key": "NSE_EQ|INE414G01012",  "tradingsymbol": "MUTHOOTFIN", "sector": "Diversified"},
    {"name": "Cholamandalam",       "instrument_key": "NSE_EQ|INE121A01024",  "tradingsymbol": "CHOLAFIN",   "sector": "Diversified"},
    {"name": "SRF",                 "instrument_key": "NSE_EQ|INE647A01010",  "tradingsymbol": "SRF",        "sector": "Diversified"},
    {"name": "Bajaj Holdings",      "instrument_key": "NSE_EQ|INE118A01012",  "tradingsymbol": "BAJAJHLDNG", "sector": "Diversified"},
    {"name": "Havells India",       "instrument_key": "NSE_EQ|INE176B01034",  "tradingsymbol": "HAVELLS",    "sector": "Diversified"},
]


def download_instruments(exchange: str, cache_dir: str = CACHE_DIR) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, f"{exchange}.csv")
    gz_path  = csv_path + ".gz"

    if os.path.exists(csv_path) and (time.time() - os.path.getmtime(csv_path)) < CACHE_TTL_SECONDS:
        return csv_path

    url = f"https://assets.upstox.com/market-quote/instruments/exchange/{exchange}.csv.gz"
    print(f"Downloading {exchange} instruments list...")
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()

    with open(gz_path, "wb") as f:
        shutil.copyfileobj(resp.raw, f)
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print(f"  Saved to {csv_path}")
    return csv_path


def load_equity_instruments(cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    frames = []
    for exchange in ("NSE", "BSE"):
        path = download_instruments(exchange, cache_dir)
        df   = pd.read_csv(path, low_memory=False)
        if 'instrument_key' not in df.columns or 'tradingsymbol' not in df.columns:
            raise ValueError(f"Unexpected instruments schema in {path}")
        # Indian equity shares have ISINs starting with INE.
        df = df[df['instrument_key'].astype(str).str.contains("INE", regex=False)].copy()
        # Filter out malformed symbols and non-equity rows.
        df = df[
            df['tradingsymbol'].astype(str).str.match(r'^[A-Z][A-Z0-9&-]*$') &
            ~df['tradingsymbol'].astype(str).str.contains(r'\d{4,}')
        ].copy()
        df['exchange_label'] = exchange
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset='instrument_key')
    return combined[['instrument_key', 'tradingsymbol', 'name',
                     'exchange_label']].reset_index(drop=True)


def get_training_stocks() -> list:
    return TRAINING_STOCKS


if __name__ == "__main__":
    df = load_equity_instruments()
    print(f"\nTotal equity instruments loaded: {len(df)}")
    print(f"  NSE: {(df['exchange_label'] == 'NSE').sum()}")
    print(f"  BSE: {(df['exchange_label'] == 'BSE').sum()}")
    print(df.head(5).to_string(index=False))
