import os
import json
import datetime
import logging
import requests
import pandas as pd
import urllib.parse
from xml.etree import ElementTree
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

try:
    with open("token.txt", "r") as f:
        UPSTOX_TOKEN = f.read().strip()
except FileNotFoundError:
    UPSTOX_TOKEN = os.getenv("UPSTOX_TOKEN")

if not UPSTOX_TOKEN or not HF_API_TOKEN:
    print("WARNING: Missing API tokens. Check .env or token.txt before running.")

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    # Technical
    'RSI_14',
    'MACD_pct', 'MACDh_pct', 'MACDs_pct',
    'BB_Position',
    'Price_vs_EMA20_%', 'Price_vs_EMA50_%',
    'Daily_Return_%', 'High_Low_Spread_%',
    'Volume_Ratio',
    'Return_5d_%', 'Return_10d_%',
    'ATR_pct',
    'Stoch_K', 'Stoch_D',
    # Market context
    'Nifty_Return_5d_%', 'Nifty_Return_20d_%', 'Nifty_RSI',
    # Phase 1 — breakout / momentum / gap
    'High_52w_%', 'Low_52w_%', 'OBV_momentum', 'Gap_pct',
    # Phase 2 — candlestick patterns
    'Bullish_Engulf', 'Hammer', 'Doji', 'Gap_Volume',
]

# Fundamentals + institutional flow are NOT trained on (look-ahead bias —
# screener.in / NSE only expose current values, not historical snapshots).
# They are fetched at inference and folded into composite_score post-model.
NON_TRAINED_CONTEXT_COLUMNS = [
    'PE_vs_sector', 'EPS_growth_%', 'Promoter_pct', 'DE_ratio',
    'FII_flow_5d', 'DII_flow_5d', 'Nifty_PCR_signal',
]

NIFTY_KEY          = "NSE_INDEX|Nifty 50"
SENTIMENT_CACHE_DIR = os.path.join("cache", "sentiment")


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def fetch_nifty_data(upstox_token=None, from_date="2020-01-01", to_date=None):
    return fetch_technical_data(NIFTY_KEY, upstox_token, from_date, to_date)


def compute_nifty_features(nifty_df: pd.DataFrame) -> pd.DataFrame:
    """Return a date-indexed DataFrame of Nifty market context features."""
    n = nifty_df.copy()
    n['date'] = n['timestamp'].dt.date
    n = n.set_index('date').sort_index()

    n['Nifty_Return_5d_%']  = n['close'].pct_change(5) * 100
    n['Nifty_Return_20d_%'] = n['close'].pct_change(20) * 100

    delta    = n['close'].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    n['Nifty_RSI'] = 100 - (100 / (1 + avg_gain / avg_loss))

    return n[['Nifty_Return_5d_%', 'Nifty_Return_20d_%', 'Nifty_RSI']].dropna()


def fetch_technical_data(instrument_key, upstox_token=None,
                         from_date="2025-01-01", to_date=None):
    if upstox_token is None:
        upstox_token = UPSTOX_TOKEN
    if to_date is None:
        to_date = datetime.date.today().isoformat()

    url = (f"https://api.upstox.com/v3/historical-candle"
           f"/{instrument_key}/days/1/{to_date}/{from_date}")
    headers = {'Accept': 'application/json',
               'Authorization': f'Bearer {upstox_token}'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code in (429, 503):
            response.raise_for_status()
        if response.status_code == 200 and response.json().get('status') == 'success':
            candles = response.json()['data']['candles']
            df = pd.DataFrame(candles,
                              columns=['timestamp', 'open', 'high', 'low',
                                       'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp').reset_index(drop=True)
    except requests.exceptions.HTTPError:
        raise
    except Exception as exc:
        LOGGER.warning("Failed to fetch technical data for %s: %s", instrument_key, exc)
    return None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_indicators(df, nifty_features=None):
    df = df.copy()

    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    delta    = df['close'].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + avg_gain / avg_loss))

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9']  = ema_12 - ema_26
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']

    df['BBM_20_2.0'] = df['close'].rolling(20).mean()
    std_dev = df['close'].rolling(20).std()
    df['BBU_20_2.0'] = df['BBM_20_2.0'] + std_dev * 2
    df['BBL_20_2.0'] = df['BBM_20_2.0'] - std_dev * 2

    df['Daily_Return_%']   = df['close'].pct_change() * 100
    df['High_Low_Spread_%']= ((df['high'] - df['low']) / df['open']) * 100
    df['Volume_Ratio']     = df['volume'] / df['volume'].rolling(20).mean()
    df['Return_5d_%']      = df['close'].pct_change(5) * 100
    df['Return_10d_%']     = df['close'].pct_change(10) * 100

    band_width = df['BBU_20_2.0'] - df['BBL_20_2.0']
    df['BB_Position']     = (df['close'] - df['BBL_20_2.0']) / band_width.replace(0, float('nan'))
    df['Price_vs_EMA20_%']= ((df['close'] - df['EMA_20']) / df['EMA_20']) * 100
    df['Price_vs_EMA50_%']= ((df['close'] - df['EMA_50']) / df['EMA_50']) * 100
    df['MACD_pct']        = (df['MACD_12_26_9']  / df['close']) * 100
    df['MACDs_pct']       = (df['MACDs_12_26_9'] / df['close']) * 100
    df['MACDh_pct']       = (df['MACDh_12_26_9'] / df['close']) * 100

    high_low        = df['high'] - df['low']
    high_prev_close = (df['high'] - df['close'].shift()).abs()
    low_prev_close  = (df['low']  - df['close'].shift()).abs()
    true_range      = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['ATR_pct']   = (true_range.rolling(14).mean() / df['close']) * 100

    low14        = df['low'].rolling(14).min()
    high14       = df['high'].rolling(14).max()
    stoch_range  = (high14 - low14).replace(0, float('nan'))
    df['Stoch_K']= ((df['close'] - low14) / stoch_range) * 100
    df['Stoch_D']= df['Stoch_K'].rolling(3).mean()

    # Phase 1 — 52-week proximity
    h52 = df['high'].rolling(252).max()
    l52 = df['low'].rolling(252).min()
    df['High_52w_%'] = ((h52 - df['close']) / h52) * 100
    df['Low_52w_%']  = ((df['close'] - l52) / l52) * 100

    # Phase 1 — OBV momentum
    sign = df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv  = (df['volume'] * sign).cumsum()
    obv_ema = obv.ewm(span=20, adjust=False).mean()
    df['OBV_momentum'] = ((obv - obv_ema) / obv_ema.abs().replace(0, float('nan'))) * 100

    # Phase 1 — overnight gap
    df['Gap_pct'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * 100

    # Phase 2 — candlestick patterns
    body       = (df['close'] - df['open']).abs()
    crange     = df['high'] - df['low']
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    df['Bullish_Engulf'] = (
        (df['close'] > df['open']) &
        (df['open']  < df['close'].shift(1)) &
        (df['close'] > df['open'].shift(1)) &
        (df['close'].shift(1) < df['open'].shift(1))
    ).astype(int)

    df['Hammer'] = (
        (lower_wick > 2 * body.replace(0, float('nan'))) &
        (body > 0) &
        (df['close'] > df['open'])
    ).astype(int)

    df['Doji']      = (body < 0.1 * crange.replace(0, float('nan'))).astype(int)
    df['Gap_Volume']= ((df['Gap_pct'] > 0.5) & (df['Volume_Ratio'] > 1.5)).astype(int)

    if nifty_features is not None:
        df['date'] = df['timestamp'].dt.date
        df = df.merge(nifty_features.reset_index(), on='date', how='left')
        df = df.drop(columns=['date'])
    else:
        df['Nifty_Return_5d_%']  = 0.0
        df['Nifty_Return_20d_%'] = 0.0
        df['Nifty_RSI']          = 50.0

    return df.dropna()


def calculate_technicals_full(df, nifty_features=None):
    """Return all rows with computed indicators — used for model training."""
    return _compute_indicators(df, nifty_features)


def calculate_technicals_latest(df, nifty_features=None):
    """Return the most recent row as a Series — used for inference."""
    result = _compute_indicators(df, nifty_features)
    return result.iloc[-1] if not result.empty else None


# ---------------------------------------------------------------------------
# FinBERT news sentiment
# ---------------------------------------------------------------------------

def get_news_sentiment(company_name: str, hf_token=None,
                       symbol: str = None) -> tuple:
    """Return (sentiment_score, confidence).

    sentiment_score : -1.0 to +1.0
    confidence      :  0.0 to  1.0
    Fetches from NewsAPI, analyzes with HuggingFace FinBERT.
    Results are cached per (symbol, date) under cache/sentiment/YYYY-MM-DD/.
    """
    import datetime as _dt
    if hf_token is None:
        hf_token = HF_API_TOKEN

    today = _dt.date.today().isoformat()

    if symbol:
        cache_path = os.path.join(SENTIMENT_CACHE_DIR, today, f"{symbol}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            return cached["score"], cached["confidence"]

    if not NEWSAPI_KEY:
        return 0.0, 0.0

    # Fetch news from NewsAPI
    try:
        newsapi_url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{company_name} OR {symbol}" if symbol else company_name,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": NEWSAPI_KEY,
            "pageSize": 10,
        }
        resp = requests.get(newsapi_url, params=params, timeout=10)
        if resp.status_code != 200:
            return 0.0, 0.0
        data = resp.json()
        articles = data.get("articles", [])
        headlines = [a.get("title", "") for a in articles if a.get("title")][:10]
    except Exception as exc:
        LOGGER.warning("News fetch failed for %s: %s", company_name, exc)
        return 0.0, 0.0

    if not headlines:
        return 0.0, 0.0

    # Analyze sentiment with TextBlob when available.
    try:
        from textblob import TextBlob
    except Exception as exc:
        LOGGER.warning("TextBlob unavailable for %s: %s", company_name, exc)
        return 0.0, 0.0

    score_sum = confidence_sum = scored = 0

    try:
        for headline in headlines:
            blob = TextBlob(headline)
            polarity = blob.sentiment.polarity  # -1.0 to +1.0
            subjectivity = blob.sentiment.subjectivity  # 0.0 to 1.0

            score_sum      += polarity
            confidence_sum += (1.0 - subjectivity)  # confidence = objectivity
            scored         += 1
    except Exception as exc:
        LOGGER.warning("Sentiment scoring failed for %s: %s", company_name, exc)
        return 0.0, 0.0

    if scored == 0:
        return 0.0, 0.0

    final_score = round(score_sum / scored, 4)
    final_conf  = round(confidence_sum / scored, 4)

    if symbol:
        cache_dir = os.path.join(SENTIMENT_CACHE_DIR, today)
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, f"{symbol}.json"), "w") as f:
            json.dump({"score": final_score, "confidence": final_conf,
                       "headlines": headlines[:5]}, f)

    return final_score, final_conf
