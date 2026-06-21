import os
import sys
import json
import time
import warnings
import datetime
import concurrent.futures
import logging

warnings.filterwarnings("ignore", message="X does not have valid feature names")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import ai
import instruments
import fundamentals as fund
import institutional_flow as iflow
import options_data as opt
import db

CHECKPOINT_DIR    = "checkpoints"
MODELS_DIR        = "models"
BATCH_SIZE        = 100
MAX_WORKERS       = 5
BATCH_DELAY       = 2.0
SENTIMENT_TOP_N   = 80
FUNDAMENTAL_TOP_N = 500
MAX_RETRIES       = 3

# Minimum 20-day avg traded value (₹) to consider a stock tradable in retail size.
# 5 Cr = ₹50,000,000 — below this, slippage on a ₹2L position becomes meaningful.
MIN_AVG_TRADED_VALUE = 2.5e8

# Minimum fraction of attempted stocks (ok + illiquid) that must complete for
# the run to publish. Below this we treat the run as broken (Upstox outage,
# token expiry, network issues) and exit non-zero so the scheduler sees failure.
MIN_RUN_COMPLETION = 0.80


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_sector_models() -> dict:
    """Load all sector ensemble models → sector: (lgb_booster, xgb_model, scaler)."""
    sector_models = {}
    if not os.path.exists(MODELS_DIR):
        return sector_models
    for fname in os.listdir(MODELS_DIR):
        if not (fname.startswith("model_") and fname.endswith("_lgb.txt")):
            continue
        sector      = fname[len("model_"):-len("_lgb.txt")]
        lgb_path    = os.path.join(MODELS_DIR, fname)
        xgb_path    = os.path.join(MODELS_DIR, f"model_{sector}_xgb.json")
        scaler_path = os.path.join(MODELS_DIR, f"scaler_{sector}.pkl")
        if os.path.exists(xgb_path) and os.path.exists(scaler_path):
            lgb_m = lgb.Booster(model_file=lgb_path)
            xgb_m = xgb.XGBClassifier()
            xgb_m.load_model(xgb_path)
            sector_models[sector] = (lgb_m, xgb_m, joblib.load(scaler_path))
    return sector_models


def build_instrument_sector_map() -> dict:
    return {s['instrument_key']: s['sector'] for s in instruments.get_training_stocks()}


def load_model_metrics() -> dict:
    path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_model_for_stock(instrument_key: str, sector_models: dict,
                        instrument_sector_map: dict, metrics: dict):
    general_acc = metrics.get('general', {}).get('accuracy', 0.0)
    sector = instrument_sector_map.get(instrument_key)
    if sector and sector in sector_models:
        if metrics.get(sector, {}).get('accuracy', 0.0) >= general_acc:
            lgb_m, xgb_m, s = sector_models[sector]
            return lgb_m, xgb_m, s, sector
    if 'general' in sector_models:
        lgb_m, xgb_m, s = sector_models['general']
        return lgb_m, xgb_m, s, 'general'
    raise FileNotFoundError("No model found. Run train.py first.")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint(date_str: str) -> dict:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"progress_{date_str}.json")
    if os.path.exists(path):
        with open(path) as f:
            state = json.load(f)
        print(f"Resuming from checkpoint: {len(state['processed_keys'])} stocks already done.")
        return state
    return {"run_date": date_str, "total_stocks": 0, "processed_keys": [], "results": []}


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        return obj.item() if hasattr(obj, 'item') else super().default(obj)


def save_checkpoint(state: dict, date_str: str):
    path = os.path.join(CHECKPOINT_DIR, f"progress_{date_str}.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2, cls=_NumpyEncoder)


# ---------------------------------------------------------------------------
# Per-stock analysis — ensemble inference with retry
# ---------------------------------------------------------------------------

def _ensemble_prob(lgb_booster, xgb_model, features_scaled) -> float:
    lgb_prob = float(lgb_booster.predict(features_scaled)[0])
    xgb_prob = float(xgb_model.predict_proba(features_scaled)[0][1])
    return lgb_prob * 0.6 + xgb_prob * 0.4


def analyze_single_stock(row, upstox_token: str, lgb_m, xgb_m, scaler,
                          nifty_features=None, sector_label="general") -> dict:
    instrument_key = row['instrument_key']
    base = {
        "instrument_key":  instrument_key,
        "tradingsymbol":   row['tradingsymbol'],
        "name":            row['name'],
        "exchange":        row['exchange_label'],
        "close":           None,
        "rsi":             None,
        "technical_score": None,
        "sentiment_score": None,
        "status":          "error",
    }

    for attempt in range(MAX_RETRIES):
        try:
            df = ai.fetch_technical_data(instrument_key, upstox_token)
            if df is None or len(df) < 60:
                base["status"] = "insufficient_data"
                return base

            # Liquidity gate — reject illiquid stocks before scoring so they
            # never appear as picks. close * volume averaged over last 20 days.
            recent = df.tail(20)
            avg_tv = float((recent['close'] * recent['volume']).mean())
            if avg_tv < MIN_AVG_TRADED_VALUE:
                base["status"]               = "illiquid"
                base["avg_traded_value_20d"] = round(avg_tv, 0)
                return base

            latest = ai.calculate_technicals_latest(df, nifty_features)
            if latest is None:
                base["status"] = "insufficient_data"
                return base

            features = latest[ai.FEATURE_COLUMNS].values.reshape(1, -1)
            if scaler is not None:
                features = scaler.transform(features)

            prob   = _ensemble_prob(lgb_m, xgb_m, features) * 100
            close  = float(latest['close'])
            atr_pct= float(latest['ATR_pct'])  # daily true range as % of close
            atr_rs = close * atr_pct / 100.0   # ATR in rupees

            base.update({
                "close":                round(close, 2),
                "rsi":                  round(float(latest['RSI_14']), 2),
                "atr_pct":              round(atr_pct, 2),
                "stop_loss":            round(close - 1.5 * atr_rs, 2),
                "target":               round(close + 3.0 * atr_rs, 2),
                "risk_per_share":       round(1.5 * atr_rs, 2),
                "technical_score":      round(prob, 2),
                "avg_traded_value_20d": round(avg_tv, 0),
                "model_used":           sector_label,
                "status":               "ok",
            })
            return base

        except Exception as e:
            err = str(e)
            if "429" in err or "503" in err:
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, 10))
                continue
            base["status"] = f"error: {err[:80]}"
            return base

    base["status"] = "rate_limited"
    return base


# ---------------------------------------------------------------------------
# Pass 1 — Technical scan (all instruments)
# ---------------------------------------------------------------------------

def run_batch_analysis(instruments_df: pd.DataFrame, upstox_token: str,
                       sector_models: dict, instrument_sector_map: dict,
                       metrics: dict, state: dict, date_str: str,
                       nifty_features=None) -> list:
    processed = set(state['processed_keys'])
    todo      = instruments_df[~instruments_df['instrument_key'].isin(processed)].reset_index(drop=True)
    total     = len(todo)
    print(f"\nStocks to process: {total} (skipping {len(processed)} already done)")

    batches = [todo.iloc[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        print(f"  Batch {batch_idx + 1}/{len(batches)} ({len(batch)} stocks)...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {}
            for _, row in batch.iterrows():
                lgb_m, xgb_m, scaler, label = get_model_for_stock(
                    row['instrument_key'], sector_models, instrument_sector_map, metrics)
                f = ex.submit(analyze_single_stock, row, upstox_token,
                              lgb_m, xgb_m, scaler, nifty_features, label)
                futures[f] = row['instrument_key']

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    key    = futures[future]
                    result = {"instrument_key": key, "status": f"error: {str(e)[:80]}"}
                state['results'].append(result)
                state['processed_keys'].append(result['instrument_key'])

        save_checkpoint(state, date_str)
        ok_count = sum(1 for r in state['results'] if r.get('status') == 'ok')
        print(f"    Done. Total ok so far: {ok_count}")

        if batch_idx < len(batches) - 1:
            time.sleep(BATCH_DELAY)

    return state['results']


# ---------------------------------------------------------------------------
# Pass 2 — Fundamental enrichment (top N by technical score)
# ---------------------------------------------------------------------------

def run_fundamental_enrichment(results: list, top_n: int = FUNDAMENTAL_TOP_N) -> list:
    ok  = [r for r in results if r.get('status') == 'ok' and r.get('technical_score') is not None]
    top = sorted(ok, key=lambda r: r['technical_score'], reverse=True)[:top_n]

    print(f"\nFetching fundamentals for top {len(top)} stocks...")
    all_fund = fund.load_fundamentals_for_stocks([r['tradingsymbol'] for r in top])

    training   = {s['tradingsymbol']: s['sector'] for s in instruments.get_training_stocks()}
    sym_sector = {r['tradingsymbol']: training.get(r['tradingsymbol'], 'Unknown') for r in top}
    sector_medians = fund.compute_sector_medians(all_fund, sym_sector)

    for r in top:
        sym = r['tradingsymbol']
        f   = all_fund.get(sym, {})
        sec = sym_sector.get(sym, 'Unknown')
        med = sector_medians.get(sec, 1.0)

        # A stock with no scraped data at all = unknown, not "average".
        # Tag it so composite_score re-normalizes instead of pricing in fake info.
        has_real_pe  = f.get('pe') is not None and not np.isnan(f.get('pe', float('nan')))
        has_real_eps = f.get('eps_growth') is not None and not np.isnan(f.get('eps_growth', float('nan')))
        if not (has_real_pe or has_real_eps):
            r['fundamental_score']      = None
            r['fundamental_data_ok']    = False
            print(f"  {sym:<12} fundamentals: no data")
            continue

        pe     = f.get('pe', float('nan'))
        pe_rel = (pe / med) if (med and med != 0 and not np.isnan(pe)) else 1.0

        pe_score   = max(0, min(100, (2.0 - pe_rel) * 50))
        eps_score  = max(0, min(100, 50 + f.get('eps_growth', 0.0)))
        prom_score = max(0, min(100, f.get('promoter_pct', 50.0)))
        de_score   = max(0, min(100, (2.0 - f.get('de_ratio', 1.0)) * 50))

        r['fundamental_score']   = round((pe_score + eps_score + prom_score + de_score) / 4, 2)
        r['fundamental_data_ok'] = True
        print(f"  {sym:<12} fundamentals: {r['fundamental_score']:5.1f}  (PE={pe_score:.0f} EPS={eps_score:.0f} Prom={prom_score:.0f} D/E={de_score:.0f})")

    return results


# ---------------------------------------------------------------------------
# Pass 3 — Sentiment enrichment (top N by technical score)
# ---------------------------------------------------------------------------

def run_sentiment_top_n(results: list, hf_token: str, top_n: int = SENTIMENT_TOP_N) -> list:
    ok  = [r for r in results if r.get('status') == 'ok' and r.get('technical_score') is not None]
    top = sorted(ok, key=lambda r: (
        (r.get('technical_score') or 0) * 0.55 +
        (r.get('fundamental_score') or 0) * 0.25
    ), reverse=True)[:top_n]

    print(f"\nFetching sentiment for top {len(top)} stocks...")
    for i, r in enumerate(top):
        score, conf = ai.get_news_sentiment(r['name'], hf_token, symbol=r['tradingsymbol'])
        sym = r['tradingsymbol']
        # confidence == 0.0 means NewsAPI failed or no articles found — no real signal.
        if conf == 0.0:
            r['sentiment_score']      = None
            r['sentiment_confidence'] = None
            r['sentiment_data_ok']    = False
            print(f"  {sym:<12} sentiment:    no data")
        else:
            # Normalize -1..+1 polarity to 0..100 for consistent display with
            # technical/fundamental scores. Weighted by confidence so low-confidence
            # headlines don't swing the bar far from neutral (50).
            r['sentiment_score']      = round((score * conf) * 50 + 50, 2)
            r['sentiment_confidence'] = conf
            r['sentiment_data_ok']    = True
            label = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
            print(f"  {sym:<12} sentiment:    {score:+.3f}  ({label}, conf={conf:.2f})")

        cache_today = os.path.join(ai.SENTIMENT_CACHE_DIR,
                                   datetime.date.today().isoformat(),
                                   f"{r['tradingsymbol']}.json")
        if not os.path.exists(cache_today):
            time.sleep(0.5)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

W_TECH, W_FUND, W_SENT = 0.55, 0.25, 0.20


def _attach_composite_fields(results: list) -> None:
    """In-place: add composite_score + buy_signal + confidence + position size
    to each ok row, so both the CSV writer and the prediction logger see the
    same values."""
    for r in results:
        if r.get('status') != 'ok' or r.get('technical_score') is None:
            continue
        r.setdefault('fundamental_data_ok', False)
        r.setdefault('sentiment_data_ok',   False)
        r['composite_score'] = _composite(r)
        r['buy_signal']      = _signal_label(r.get('technical_score'))
        sources              = 1 + int(bool(r['fundamental_data_ok'])) \
                                 + int(bool(r['sentiment_data_ok']))
        r['confidence']      = {1: 'TECH-ONLY', 2: 'TECH+1', 3: 'FULL'}[sources]
        r['position_size_pct'] = _position_size_pct(r)


def _position_size_pct(row) -> float:
    """Suggested capital % for this position.

    Scales linearly with (composite - 50) so a 50-score gets 0% and a 100-score
    hits the cap. Divided by ATR%/3 so a high-vol stock (ATR 9%) gets 1/3 the
    size of a tame one (ATR 3%). Capped at 5% per name. Returns 0 for HOLD/SELL.
    """
    score = row.get('composite_score')
    atr   = row.get('atr_pct') or 3.0
    if score is None or score < 50 or row.get('buy_signal') in ('SELL/WAIT', 'NO DATA'):
        return 0.0
    raw = (score - 50) / 10.0          # 50→0, 100→5
    vol_adj = raw * (3.0 / max(atr, 1.0))  # halve when ATR=6%, etc.
    return round(min(vol_adj, 5.0), 2)


def _signal_label(score):
    if score is None: return 'NO DATA'
    if score > 65:    return 'STRONG BUY'
    if score < 35:    return 'SELL/WAIT'
    return 'HOLD'


def _composite(row) -> float:
    """Weighted composite, re-normalized over only the components with real data.

    A stock with no fundamentals available scores on technical alone — we
    don't substitute a neutral 50 (which would mask the missing signal).
    """
    tech = row.get('technical_score')
    if tech is None:
        return None

    parts  = [(W_TECH, float(tech))]

    if row.get('fundamental_data_ok'):
        parts.append((W_FUND, float(row['fundamental_score'])))

    if row.get('sentiment_data_ok'):
        parts.append((W_SENT, float(row['sentiment_score'])))

    total_w = sum(w for w, _ in parts)
    return round(sum(w * v for w, v in parts) / total_w, 2)


def _build_picks_df(results: list) -> pd.DataFrame:
    """Build the ranked picks DataFrame (sorted, deduped, ranked, columns ordered).

    The same company often trades on both NSE and BSE under the same
    tradingsymbol but with different instrument_keys. The DB primary key is
    (run_date, tradingsymbol), so we keep the higher-scoring listing per
    symbol — usually NSE because of liquidity and tighter spreads.
    """
    ok = [r for r in results
          if r.get('status') == 'ok' and r.get('technical_score') is not None]
    if not ok:
        return pd.DataFrame()

    df = pd.DataFrame(ok)
    if 'fundamental_data_ok' not in df.columns:
        df['fundamental_data_ok'] = False
    else:
        df['fundamental_data_ok'] = df['fundamental_data_ok'].fillna(False)
    if 'sentiment_data_ok' not in df.columns:
        df['sentiment_data_ok'] = False
    else:
        df['sentiment_data_ok'] = df['sentiment_data_ok'].fillna(False)
    df = df.sort_values('composite_score', ascending=False, na_position='last')

    # Drop duplicate symbols, keeping first (= highest composite_score).
    before = len(df)
    df = df.drop_duplicates(subset='tradingsymbol', keep='first').reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Deduped {dropped} dual-listed rows (NSE/BSE same symbol).")

    df.insert(0, 'rank', df.index + 1)
    return df


def _build_prediction_rows(df: pd.DataFrame, date_str: str,
                           forward_days: int = 5,
                           min_composite: float = 60.0,
                           top_n: int = 30) -> list:
    """Today's actionable picks for the shadow log (mirrors the old logic)."""
    if df.empty:
        return []

    eligible = df[df['composite_score'].notna() & df['close'].notna()]
    if eligible.empty:
        return []

    above = eligible[eligible['composite_score'] >= min_composite]
    picks = above if len(above) >= top_n else eligible.head(top_n)

    pred_date = datetime.date.fromisoformat(date_str)
    target    = _next_trading_day(pred_date, forward_days).isoformat()

    rows = []
    for _, r in picks.iterrows():
        rows.append({
            "predicted_at":      date_str,
            "tradingsymbol":     r.get('tradingsymbol'),
            "instrument_key":    r.get('instrument_key', ''),
            "name":              r.get('name'),
            "exchange":          r.get('exchange'),
            "sector_model":      r.get('model_used'),
            "buy_signal":        r.get('buy_signal'),
            "technical_score":   r.get('technical_score'),
            "fundamental_score": r.get('fundamental_score'),
            "sentiment_score":   r.get('sentiment_score'),
            "composite_score":   r.get('composite_score'),
            "confidence":        r.get('confidence'),
            "entry_price":       r.get('close'),
            "target_date":       target,
        })
    return rows


def _trend_label(nifty_return_5d, nifty_return_20d, nifty_rsi) -> str:
    if nifty_return_5d is None or nifty_return_20d is None or nifty_rsi is None:
        return "Unknown"
    if nifty_return_5d > 1.0 and nifty_return_20d > 2.0 and 45 <= nifty_rsi <= 75:
        return "Bullish"
    if nifty_return_5d < -1.0 and nifty_return_20d < -2.0:
        return "Bearish"
    if nifty_rsi >= 75:
        return "Overheated"
    if nifty_rsi <= 35:
        return "Weak"
    return "Neutral"


def _environment_label(score: float) -> str:
    if score >= 70:
        return "Favorable"
    if score >= 55:
        return "Constructive"
    if score >= 40:
        return "Mixed"
    return "Risk-off"


def _build_sector_leaderboard(df: pd.DataFrame) -> list:
    if df.empty:
        return []

    data = df.copy()
    data["sector_model"] = data.get("model_used", "unknown").fillna("unknown")
    for col in ("composite_score", "technical_score", "fundamental_score", "sentiment_score"):
        data[col] = pd.to_numeric(data.get(col), errors="coerce")

    rows = []
    for sector, group in data.groupby("sector_model", dropna=False):
        group = group.sort_values("composite_score", ascending=False, na_position="last")
        best = group.iloc[0]
        rows.append({
            "sector_model": sector or "unknown",
            "stock_count": int(len(group)),
            "strong_buys": int((group["buy_signal"] == "STRONG BUY").sum()),
            "avg_composite_score": round(float(group["composite_score"].mean()), 2),
            "avg_technical_score": round(float(group["technical_score"].mean()), 2),
            "avg_fundamental_score": (
                round(float(group["fundamental_score"].mean()), 2)
                if group["fundamental_score"].notna().any() else None
            ),
            "avg_sentiment_score": (
                round(float(group["sentiment_score"].mean()), 4)
                if group["sentiment_score"].notna().any() else None
            ),
            "best_symbol": best.get("tradingsymbol"),
            "best_stock_name": best.get("name"),
            "best_composite_score": best.get("composite_score"),
        })

    rows.sort(
        key=lambda r: (
            r["avg_composite_score"] if r["avg_composite_score"] is not None else -1,
            r["strong_buys"],
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def _build_market_dashboard(df: pd.DataFrame, nifty_features, flow_context,
                            pcr, pcr_signal, sector_rows: list) -> dict:
    latest = nifty_features.iloc[-1] if nifty_features is not None else {}
    nifty_return_5d = latest.get("Nifty_Return_5d_%") if nifty_features is not None else None
    nifty_return_20d = latest.get("Nifty_Return_20d_%") if nifty_features is not None else None
    nifty_rsi = latest.get("Nifty_RSI") if nifty_features is not None else None

    fii_flow_5d = flow_context.get("FII_flow_5d") if flow_context else None
    dii_flow_5d = flow_context.get("DII_flow_5d") if flow_context else None
    net_flow_5d = (
        round(float(fii_flow_5d) + float(dii_flow_5d), 4)
        if fii_flow_5d is not None and dii_flow_5d is not None else None
    )

    strong_buys = int((df["buy_signal"] == "STRONG BUY").sum()) if not df.empty else 0
    ranked_stocks = int(len(df))
    avg_composite = float(df["composite_score"].mean()) if not df.empty else 0.0

    trend = _trend_label(nifty_return_5d, nifty_return_20d, nifty_rsi)
    score = 50.0
    if nifty_return_5d is not None:
        score += max(-15, min(15, float(nifty_return_5d) * 3))
    if nifty_return_20d is not None:
        score += max(-15, min(15, float(nifty_return_20d) * 1.5))
    if nifty_rsi is not None:
        rsi = float(nifty_rsi)
        if 45 <= rsi <= 65:
            score += 8
        elif rsi > 75 or rsi < 35:
            score -= 8
    if net_flow_5d is not None:
        score += max(-10, min(10, float(net_flow_5d) * 10))
    if pcr_signal is not None:
        score += float(pcr_signal) * 5
    score += max(-10, min(10, (avg_composite - 50) / 2))
    score += max(-5, min(5, strong_buys / 5))
    score = round(max(0, min(100, score)), 2)

    leading = sector_rows[0] if sector_rows else {}
    return {
        "market_score": score,
        "environment": _environment_label(score),
        "nifty_trend": trend,
        "nifty_return_5d": round(float(nifty_return_5d), 2) if nifty_return_5d is not None else None,
        "nifty_return_20d": round(float(nifty_return_20d), 2) if nifty_return_20d is not None else None,
        "nifty_rsi": round(float(nifty_rsi), 2) if nifty_rsi is not None else None,
        "fii_flow_5d": fii_flow_5d,
        "dii_flow_5d": dii_flow_5d,
        "net_flow_5d": net_flow_5d,
        "pcr": pcr,
        "pcr_signal": pcr_signal,
        "strong_buys": strong_buys,
        "ranked_stocks": ranked_stocks,
        "leading_sector": leading.get("sector_model"),
        "leading_sector_score": leading.get("avg_composite_score"),
    }


def _next_trading_day(d: datetime.date, n: int) -> datetime.date:
    """Skip weekends only — the outcome filler tolerates holidays by walking
    forward to the next available candle."""
    cur, left = d, n
    while left > 0:
        cur += datetime.timedelta(days=1)
        if cur.weekday() < 5:
            left -= 1
    return cur


def write_to_db(conn, results: list, date_str: str, run_health: dict,
                nifty_features=None, flow_context=None,
                pcr=None, pcr_signal=None) -> None:
    """Strict-mode DB write. Any psycopg2 error here propagates to main()
    which exits 4 — caller must NOT have done file writes by this point."""
    df = _build_picks_df(results)
    if df.empty:
        print("No valid picks to write to DB.")
        # Still record run_health so we can diagnose empty runs.
        db.upsert_run_health(conn, date_str, run_health)
        return

    n_ranked = db.upsert_ranked_stocks(conn, date_str, df)
    print(f"\nDB: wrote {n_ranked} rows to ranked_stocks for {date_str}")

    sector_rows = _build_sector_leaderboard(df)
    n_sectors = db.upsert_sector_leaderboard(conn, date_str, sector_rows)
    print(f"DB: wrote {n_sectors} rows to sector_leaderboard")

    market_dashboard = _build_market_dashboard(
        df, nifty_features, flow_context, pcr, pcr_signal, sector_rows)
    db.upsert_market_dashboard(conn, date_str, market_dashboard)
    print(f"DB: wrote market_dashboard for {date_str}  {market_dashboard}")

    pred_rows = _build_prediction_rows(df, date_str)
    n_pred    = db.upsert_predictions(conn, pred_rows)
    print(f"DB: wrote {n_pred} actionable picks to predictions "
          f"(target_date 5 trading days out)")

    db.upsert_run_health(conn, date_str, run_health)
    print(f"DB: wrote run_health for {date_str}  {run_health}")

    live = db.live_accuracy(conn, window_days=30)
    if not live["filled"]:
        print("Live accuracy: no outcomes filled yet — run fill_outcomes.py "
              "after first prediction's target_date passes.")
    else:
        print(f"Live accuracy (30d, n={live['filled']}): "
              f"precision={live['precision']*100:.1f}%  "
              f"avg_alpha={live['avg_alpha_pct']:.2f}%")

    print("\nTop 10 picks:")
    cols = ['rank', 'tradingsymbol', 'close', 'stop_loss', 'target',
            'position_size_pct', 'composite_score', 'buy_signal', 'confidence']
    print(df.head(10)[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    date_str = datetime.date.today().isoformat()
    print(f"Blue Stock AI — Full Market Scan — {date_str}\n")

    if not ai.UPSTOX_TOKEN:
        print("FATAL: Upstox token missing. Refresh token.txt or .env UPSTOX_TOKEN.")
        sys.exit(2)
    if not ai.HF_API_TOKEN:
        print("WARNING: HF_API_TOKEN missing — sentiment will be unavailable.")

    # Strict mode: prove DB is reachable BEFORE we spend an hour fetching
    # data we can't ship anywhere. Fail fast with exit 4 if Supabase is down.
    print("Checking Supabase connection...")
    try:
        with db.connection() as conn:
            db.ping(conn)
        print("  Supabase OK")
    except Exception as e:
        print(f"FATAL: cannot reach Supabase ({type(e).__name__}: {e}). "
              f"Check SUPABASE_DB_URL in .env. Aborting before fetch.")
        sys.exit(4)

    state          = load_checkpoint(date_str)
    instruments_df = instruments.load_equity_instruments()
    if state['total_stocks'] == 0:
        state['total_stocks'] = len(instruments_df)
    print(f"Instruments loaded: {len(instruments_df)}")

    print("Fetching Nifty 50 market context...")
    nifty_df       = ai.fetch_nifty_data(ai.UPSTOX_TOKEN)
    nifty_features = ai.compute_nifty_features(nifty_df) if nifty_df is not None else None
    if nifty_features is not None:
        latest = nifty_features.iloc[-1]
        print(f"  Nifty 5d: {latest['Nifty_Return_5d_%']:.2f}%  "
              f"20d: {latest['Nifty_Return_20d_%']:.2f}%  RSI: {latest['Nifty_RSI']:.1f}")
    else:
        print("  Warning: Nifty data unavailable, using neutral market context.")

    print("Fetching market flow context (FII/DII + PCR)...")
    flow_data    = iflow.fetch_fii_dii_data()
    flow_context = iflow.compute_flow_features(flow_data)
    pcr          = opt.fetch_nifty_pcr()
    pcr_signal   = opt.get_pcr_signal(pcr)
    if flow_context is not None:
        print(f"  FII 5d: {flow_context.get('FII_flow_5d', 0):.2f}  "
              f"DII 5d: {flow_context.get('DII_flow_5d', 0):.2f}  "
              f"PCR: {pcr if pcr is not None else 'unavailable'}")
    else:
        print("  Warning: FII/DII flow unavailable.")
    if pcr is None:
        print("  Warning: Nifty PCR unavailable.")

    sector_models = load_sector_models()
    if not sector_models:
        print("FATAL: No models found in models/. Run train.py first.")
        sys.exit(2)
    metrics               = load_model_metrics()
    instrument_sector_map = build_instrument_sector_map()
    general_acc = metrics.get('general', {}).get('accuracy', 0)
    trusted = [s for s in sector_models if s != 'general'
               and metrics.get(s, {}).get('accuracy', 0) >= general_acc]
    print(f"Loaded {len(sector_models)} models. Sectors beating general "
          f"({general_acc*100:.1f}%): {trusted or 'none'}\n")

    results = run_batch_analysis(instruments_df, ai.UPSTOX_TOKEN, sector_models,
                                 instrument_sector_map, metrics, state, date_str, nifty_features)

    # Deduplicate before enrichment so fundamentals/sentiment aren't fetched twice
    # for NSE+BSE dual-listed stocks (keep highest technical_score copy).
    seen: dict[str, dict] = {}
    for r in results:
        sym = r.get('tradingsymbol')
        if sym and r.get('status') == 'ok':
            if sym not in seen or (r.get('technical_score') or 0) > (seen[sym].get('technical_score') or 0):
                seen[sym] = r
        elif sym not in seen:
            seen[sym] = r
    pre_dedup = len(results)
    results = list(seen.values())
    deduped = pre_dedup - len(results)
    if deduped:
        print(f"Pre-enrichment dedup: removed {deduped} duplicate symbols.\n")

    results = run_fundamental_enrichment(results)
    results = run_sentiment_top_n(results, ai.HF_API_TOKEN)

    ok_count        = sum(1 for r in results if r.get('status') == 'ok')
    illiquid_count  = sum(1 for r in results if r.get('status') == 'illiquid')
    fund_count      = sum(1 for r in results if r.get('fundamental_data_ok'))
    sent_count      = sum(1 for r in results if r.get('sentiment_data_ok'))
    completed       = ok_count + illiquid_count
    completion_rate = completed / max(len(results), 1)

    run_health = {
        "date":             date_str,
        "stocks_total":     len(results),
        "stocks_ok":        ok_count,
        "stocks_illiquid":  illiquid_count,
        "stocks_with_fund": fund_count,
        "stocks_with_sent": sent_count,
        "completion_rate":  round(completion_rate, 4),
        "nifty_ok":         nifty_features is not None,
        "fii_dii_ok":       flow_context is not None,
        "pcr_ok":           pcr is not None,
        "min_avg_traded_value": MIN_AVG_TRADED_VALUE,
        "min_completion_rate":  MIN_RUN_COMPLETION,
        "publish":          completion_rate >= MIN_RUN_COMPLETION and nifty_features is not None,
    }

    # Score every ok row first — composite/signal/confidence/position_size
    # must exist before we ship anything to DB.
    _attach_composite_fields(results)

    if not run_health["publish"]:
        # Still record the failed-run health so we can diagnose later. But no
        # picks: a degraded run must not look like a successful one downstream.
        try:
            with db.connection() as conn:
                db.upsert_run_health(conn, date_str, run_health)
        except Exception as e:
            print(f"  (also failed to record run_health to DB: {e})")
        reason = []
        if completion_rate < MIN_RUN_COMPLETION:
            reason.append(f"completion {completion_rate*100:.1f}% < {MIN_RUN_COMPLETION*100:.0f}%")
        if nifty_features is None:
            reason.append("no Nifty data")
        print(f"\nFATAL: run too degraded to publish ({'; '.join(reason)}). "
              f"Picks NOT written. run_health row recorded for diagnostics.")
        sys.exit(3)

    try:
        with db.connection() as conn:
            write_to_db(conn, results, date_str, run_health,
                        nifty_features, flow_context, pcr, pcr_signal)
    except Exception as e:
        print(f"\nFATAL: DB write failed ({type(e).__name__}: {e}). "
              f"Run aborted. No partial publish.")
        sys.exit(4)


if __name__ == "__main__":
    main()
