# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Train all sector models (run once, or after adding stocks/features)
python train.py

# Full daily market scan
python main.py

# Check instrument counts (NSE + BSE equity filter)
python instruments.py
```

**Dependencies:** `pip install lightgbm xgboost scikit-learn pandas numpy requests python-dotenv joblib beautifulsoup4`

**Credentials:** Upstox JWT token in `token.txt` (refreshed daily) or `.env` as `UPSTOX_TOKEN`. HuggingFace token in `.env` as `HF_API_TOKEN`.

## Architecture

Two-script design: `train.py` builds models offline; `main.py` runs the daily scan.

**`ai.py`** is the shared engine imported by both. Key entry points:
- `FEATURE_COLUMNS` — the 26-element list of leak-free features the model trains on
- `NON_TRAINED_CONTEXT_COLUMNS` — fundamentals + FII/DII flow + PCR. Fetched only at inference (screener.in/NSE expose only current values, so training on them would leak today's snapshot into 2020 rows). Folded into composite_score post-model in `main.py`.
- `_compute_indicators(df, nifty_features)` — computes all technical features; called by both `calculate_technicals_full` (training) and `calculate_technicals_latest` (inference)

**Training target** (`train.py`): market-neutral — stock 5-day return must beat Nifty 5-day return by `ALPHA_PCT = 1.0%`. Nifty forward return is aligned by date merge then shifted `-FORWARD_DAYS`.

**Model format**: LightGBM booster saved as `model_{sector}_lgb.txt`, XGBoost as `model_{sector}_xgb.json`, scaler as `scaler_{sector}.pkl`. General model is always trained; sector models only when ≥2 stocks in sector. At inference, sector model is used only if `sector_acc >= general_acc` (from `models/metrics.json`).

**Inference ensemble** (`main.py → _ensemble_prob`): loads raw `lgb.Booster` (not LGBMClassifier) — `.predict()` returns P(class=1) directly for binary models.

**Checkpointing**: `main.py` saves progress to `checkpoints/progress_YYYY-MM-DD.json` after every 100-stock batch, enabling resume on crash.

**Feature sync rule**: Any feature added to `FEATURE_COLUMNS` must be computable from OHLCV history alone (no current-snapshot data). `_compute_indicators` must produce non-NaN values for every column or `dropna()` at the end will silently discard all rows. External-source signals belong in `NON_TRAINED_CONTEXT_COLUMNS` — fetched at inference and combined post-model.

## Key gotchas

- `_compute_indicators` calls `df.merge(nifty_features, on='date')` which resets the integer index. Do not try to reindex back to the pre-merge df — Target and other columns are correctly carried through the merge.
- `load_equity_instruments()` filters `instrument_key.str.contains('|INE')` + tradingsymbol regex to exclude bonds/SDLs from the ~21k raw NSE+BSE rows.
- Sector models with small training sets (Telecom, Diversified) often underperform general — `get_model_for_stock` falls back automatically via metrics comparison.
