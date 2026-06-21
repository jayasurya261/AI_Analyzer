# Blue Stock AI - Indian Equity Market Analyzer

A daily stock-ranking system for NSE and BSE equities that combines machine learning, fundamental analysis, and sentiment scoring to surface the strongest buy candidates each trading day.

---

## How It Works

Every day the system runs three sequential passes over the full NSE + BSE universe:

```text
Pass 1 - Technical Scan (all stocks)
    -> Fetches OHLCV from Upstox API
    -> Computes 33 technical features
    -> Scores each stock with a LightGBM + XGBoost ensemble

Pass 2 - Fundamental Enrichment (top 500 by technical score)
    -> Scrapes P/E, EPS growth, promoter %, D/E from screener.in
    -> Computes relative valuation vs sector median

Pass 3 - Sentiment Analysis (top 200 by technical score)
    -> Fetches latest headlines from NewsAPI
    -> Uses TextBlob polarity as a lightweight sentiment proxy
    -> Confidence-weighted positive/negative scoring
```

Composite score = technical 55% + fundamental 25% + sentiment 20%.

---

## Project Structure

```text
AI_Analyzer/
|-- main.py
|-- train.py
|-- README.md
|-- requirements.txt
|-- .env.example
|-- src/
|   |-- ai.py
|   |-- instruments.py
|   |-- fundamentals.py
|   |-- institutional_flow.py
|   |-- options_data.py
|   |-- sentiment_logger.py
|-- tests/
|   `-- test_core_logic.py
|-- models/
|-- cache/
|-- checkpoints/
|-- output/
|-- data/
`-- logs/
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

Copy `.env.example` to `.env` and fill in your tokens:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `UPSTOX_TOKEN` | JWT from Upstox developer console |
| `HF_API_TOKEN` | Optional compatibility token; current sentiment path does not use it |
| `NEWSAPI_KEY` | NewsAPI key used for headline retrieval |

Alternatively, place the Upstox JWT in `token.txt` at the project root.

### 3. Train models

```bash
cd AI_Analyzer
python train.py
```

Training also writes a walk-forward validation report to `output/walk_forward_validation_YYYY-MM-DD.json`.

### 4. Run daily scan

```bash
cd AI_Analyzer
python main.py
```

Results are saved to `output/ranked_stocks_YYYY-MM-DD.csv`.

---

## Output Format

| Column | Description |
|---|---|
| `rank` | Composite rank (1 = best) |
| `tradingsymbol` | NSE/BSE ticker |
| `name` | Company name |
| `exchange` | NSE or BSE |
| `close` | Last closing price |
| `rsi` | 14-day RSI |
| `technical_score` | ML ensemble probability x 100 |
| `fundamental_score` | Valuation quality score |
| `sentiment_score` | Sentiment score |
| `sentiment_confidence` | Average headline confidence |
| `composite_score` | Final weighted score |
| `buy_signal` | STRONG BUY / HOLD / SELL/WAIT |
| `model_used` | Which sector model scored the stock |

---

## Tests

Run the analyzer tests with:

```bash
python -m unittest discover -s tests
```

---

## APIs Used

| Service | Purpose |
|---|---|
| Upstox v3 | OHLCV historical candles |
| screener.in | Fundamental ratios |
| NSE India | FII/DII flow + option chain PCR |
| NewsAPI | Headlines for sentiment |
| TextBlob | Sentiment scoring |
