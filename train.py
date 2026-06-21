import os
import sys
import json
import warnings
import datetime
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import ai
import instruments

FORWARD_DAYS = 5

# Round-trip transaction cost on Indian equity delivery trades.
# 0.10% brokerage + 0.10% STT (sell-side) + 0.05% slippage + small SEBI/exchange fees.
TRANSACTION_COST_PCT = 0.30

# A "buy" must beat Nifty AND clear costs to be worth taking.
NOMINAL_ALPHA_PCT = 1.0
ALPHA_PCT         = NOMINAL_ALPHA_PCT + TRANSACTION_COST_PCT

MODELS_DIR = "models"
VALIDATION_DIR = "output"
WALK_FORWARD_WINDOWS = 6
WALK_FORWARD_MIN_TRAIN_DAYS = 420
WALK_FORWARD_TEST_DAYS = 63
WALK_FORWARD_EMBARGO_DAYS = FORWARD_DAYS
PRECISION_AT_N = (10, 20, 30)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_all_stock_data(stocks: list, upstox_token: str,
                         from_date: str = "2020-01-01", to_date: str = None):
    if to_date is None:
        to_date = datetime.date.today().isoformat()

    print("  Fetching Nifty 50 market context...")
    nifty_df       = ai.fetch_nifty_data(upstox_token, from_date, to_date)
    nifty_features = ai.compute_nifty_features(nifty_df) if nifty_df is not None else None
    if nifty_features is None:
        print("    Warning: Nifty data unavailable, market context disabled.")

    # NOTE: fundamentals + FII/DII flow are deliberately NOT fetched here.
    # Training only on technical features avoids look-ahead bias from
    # current-only sources. They re-enter via composite_score at inference.

    sector_train = {}
    sector_test  = {}

    for stock in stocks:
        sym = stock.get('tradingsymbol', stock['name'])
        print(f"  Fetching {stock['name']} ({stock['sector']})...")
        df = ai.fetch_technical_data(stock['instrument_key'], upstox_token, from_date, to_date)
        if df is None or len(df) < 100:
            print("    Skipped (insufficient data)")
            continue

        # Market-neutral target: stock 5d return must exceed Nifty 5d return by ALPHA_PCT
        stock_ret5d = (df['close'].shift(-FORWARD_DAYS) - df['close']) / df['close'] * 100

        if nifty_features is not None:
            df['date'] = df['timestamp'].dt.date
            nifty_fwd  = (nifty_features['Nifty_Return_5d_%']
                          .reset_index()
                          .rename(columns={'Nifty_Return_5d_%': '_nifty_fwd'}))
            nifty_fwd['_nifty_fwd'] = nifty_fwd['_nifty_fwd'].shift(-FORWARD_DAYS)
            df    = df.merge(nifty_fwd, on='date', how='left')
            alpha = stock_ret5d - df['_nifty_fwd'].fillna(0)
            df    = df.drop(columns=['date', '_nifty_fwd'])
        else:
            alpha = stock_ret5d

        df['Target'] = (alpha >= ALPHA_PCT).astype(int)
        df = df.iloc[:-FORWARD_DAYS]

        feat_df = ai.calculate_technicals_full(df, nifty_features)
        feat_df = feat_df[ai.FEATURE_COLUMNS + ['Target', 'timestamp']].copy().dropna()
        feat_df['date'] = feat_df['timestamp'].dt.date
        feat_df['symbol'] = stock['tradingsymbol']
        feat_df = feat_df.drop(columns=['timestamp'])
        feat_df['sector'] = stock['sector']

        split  = _chronological_split_idx(feat_df)
        sector = stock['sector']
        sector_train.setdefault(sector, []).append(feat_df.iloc[:split])
        sector_test.setdefault(sector,  []).append(feat_df.iloc[split:])

    all_train = [f for frames in sector_train.values() for f in frames]
    all_test  = [f for frames in sector_test.values()  for f in frames]

    result = {'general': (
        pd.concat(all_train, ignore_index=True).dropna(),
        pd.concat(all_test,  ignore_index=True).dropna(),
    )}
    for sector in sector_train:
        if len(sector_train[sector]) >= 2:
            result[sector] = (
                pd.concat(sector_train[sector], ignore_index=True).dropna(),
                pd.concat(sector_test[sector],  ignore_index=True).dropna(),
            )
    return result, nifty_features


def _chronological_split_idx(df: pd.DataFrame, train_frac: float = 0.8) -> int:
    ordered_dates = sorted(df['date'].unique())
    if len(ordered_dates) < 2:
        return int(len(df) * train_frac)
    split_date = ordered_dates[int(len(ordered_dates) * train_frac)]
    split = int((df['date'] < split_date).sum())
    return max(1, min(split, len(df) - 1))


# ---------------------------------------------------------------------------
# Ensemble training — LightGBM 60% + XGBoost 40%
# ---------------------------------------------------------------------------

def train_ensemble(train_data: pd.DataFrame, test_data: pd.DataFrame, label: str):
    X_train = train_data[ai.FEATURE_COLUMNS]
    y_train = train_data['Target']
    X_test  = test_data[ai.FEATURE_COLUMNS]
    y_test  = test_data['Target']

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, min_child_samples=20,
        random_state=42, verbose=-1,
    )
    lgb_model.fit(X_tr_s, y_train,
                  eval_set=[(X_te_s, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)])

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, scale_pos_weight=spw,
        eval_metric='logloss', random_state=42,
    )
    xgb_model.fit(X_tr_s, y_train, eval_set=[(X_te_s, y_test)], verbose=False)

    lgb_prob = lgb_model.predict_proba(X_te_s)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_te_s)[:, 1]
    ens_prob = lgb_prob * 0.6 + xgb_prob * 0.4
    preds    = (ens_prob >= 0.5).astype(int)

    accuracy   = float(accuracy_score(y_test, preds))
    buy_prec   = float(precision_score(y_test, preds, zero_division=0))
    buy_recall = float(recall_score(y_test, preds, zero_division=0))
    ranking_metrics = precision_at_n(y_test.to_numpy(), ens_prob, PRECISION_AT_N)

    # AUC on the ensemble probability — measures ranking quality, which is
    # what main.py actually uses (it ranks by score, not by 0.5 threshold).
    # AUC is meaningful only when both classes are present in y_test.
    try:
        auc = float(roc_auc_score(y_test, ens_prob)) if y_test.nunique() > 1 else float('nan')
    except ValueError:
        auc = float('nan')

    pos_rate = float(y_test.mean())
    p_at_10 = ranking_metrics.get("precision_at_10")
    print(f"  [{label}] rows={len(train_data)}  acc={accuracy*100:.1f}%  "
          f"AUC={auc:.3f}  buy_prec={buy_prec:.2f}  buy_recall={buy_recall:.2f}  "
          f"p@10={p_at_10:.2f}  pos_rate={pos_rate*100:.1f}%")

    return lgb_model, xgb_model, scaler, {
        "accuracy":      round(accuracy, 4),
        "auc":           round(auc, 4) if auc == auc else None,
        "buy_precision": round(buy_prec, 4),
        "buy_recall":    round(buy_recall, 4),
        **ranking_metrics,
        "pos_rate":      round(pos_rate, 4),
        "train_rows":    len(train_data),
    }


# ---------------------------------------------------------------------------
# Walk-forward validation (measurement only — does not change saved model)
# ---------------------------------------------------------------------------

def precision_at_n(y_true, probabilities, top_ns=PRECISION_AT_N) -> dict:
    if len(y_true) == 0:
        return {f"precision_at_{n}": None for n in top_ns}
    order = np.argsort(probabilities)[::-1]
    metrics = {}
    for n in top_ns:
        top = order[:min(n, len(order))]
        metrics[f"precision_at_{n}"] = (
            round(float(np.mean(np.asarray(y_true)[top])), 4) if len(top) else None
        )
    return metrics


def _train_window_model(train: pd.DataFrame, test: pd.DataFrame):
    X_train = train[ai.FEATURE_COLUMNS]
    y_train = train['Target']
    X_test = test[ai.FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        min_child_samples=25,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr_s, y_train)
    return model.predict_proba(X_te_s)[:, 1]


def walk_forward_validate(data: pd.DataFrame, label: str,
                          n_windows: int = WALK_FORWARD_WINDOWS) -> dict:
    data = data.sort_values(['date', 'symbol']).reset_index(drop=True)
    dates = sorted(data['date'].unique())
    if len(dates) < WALK_FORWARD_MIN_TRAIN_DAYS + WALK_FORWARD_TEST_DAYS:
        print(f"  [{label}] Too little calendar history for walk-forward "
              f"({len(dates)} dates)")
        return {"label": label, "windows": []}

    max_start = len(dates) - WALK_FORWARD_TEST_DAYS
    min_start = WALK_FORWARD_MIN_TRAIN_DAYS
    starts = (
        [min_start] if max_start <= min_start
        else np.linspace(min_start, max_start, n_windows, dtype=int).tolist()
    )

    windows = []
    for idx, start in enumerate(dict.fromkeys(starts), start=1):
        train_end_idx = start - WALK_FORWARD_EMBARGO_DAYS
        if train_end_idx <= 0:
            continue

        train_end = dates[train_end_idx - 1]
        test_start = dates[start]
        test_end = dates[min(start + WALK_FORWARD_TEST_DAYS - 1, len(dates) - 1)]

        train = data[data['date'] <= train_end]
        test = data[(data['date'] >= test_start) & (data['date'] <= test_end)]
        if len(train) < 200 or len(test) < 30 or test['Target'].nunique() < 2:
            continue

        proba = _train_window_model(train, test)
        y_test = test['Target'].to_numpy()
        preds = (proba >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y_test, proba))
        except ValueError:
            auc = float('nan')

        windows.append({
            "window": idx,
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "pos_rate": round(float(np.mean(y_test)), 4),
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "auc": round(auc, 4) if auc == auc else None,
            "buy_precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "buy_recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            **precision_at_n(y_test, proba, PRECISION_AT_N),
        })

    if not windows:
        print(f"  [{label}] No valid walk-forward windows")
        return {"label": label, "windows": []}

    aucs = [w["auc"] for w in windows if w["auc"] is not None]
    p10s = [w["precision_at_10"] for w in windows if w["precision_at_10"] is not None]
    accs = [w["accuracy"] for w in windows]
    print(f"  [{label}] Walk-forward ({len(windows)} windows): "
          f"acc avg={np.mean(accs)*100:.1f}%  "
          f"AUC avg={(np.mean(aucs) if aucs else float('nan')):.3f}  "
          f"p@10 avg={(np.mean(p10s) if p10s else float('nan')):.2f}")

    return {
        "label": label,
        "config": {
            "windows": n_windows,
            "min_train_days": WALK_FORWARD_MIN_TRAIN_DAYS,
            "test_days": WALK_FORWARD_TEST_DAYS,
            "embargo_days": WALK_FORWARD_EMBARGO_DAYS,
            "precision_at_n": list(PRECISION_AT_N),
        },
        "summary": {
            "window_count": len(windows),
            "accuracy_avg": round(float(np.mean(accs)), 4),
            "auc_avg": round(float(np.mean(aucs)), 4) if aucs else None,
            "precision_at_10_avg": round(float(np.mean(p10s)), 4) if p10s else None,
        },
        "windows": windows,
    }


def save_validation_report(reports: list) -> str:
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    path = os.path.join(
        VALIDATION_DIR,
        f"walk_forward_validation_{datetime.date.today().isoformat()}.json",
    )
    with open(path, "w") as f:
        json.dump({
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "target": {
                "forward_days": FORWARD_DAYS,
                "alpha_pct": ALPHA_PCT,
                "transaction_cost_pct": TRANSACTION_COST_PCT,
            },
            "reports": reports,
        }, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_sector_artifacts(lgb_model, xgb_model, scaler, sector: str, metrics: dict):
    import json
    os.makedirs(MODELS_DIR, exist_ok=True)
    lgb_model.booster_.save_model(os.path.join(MODELS_DIR, f"model_{sector}_lgb.txt"))
    xgb_model.save_model(os.path.join(MODELS_DIR,           f"model_{sector}_xgb.json"))
    joblib.dump(scaler,          os.path.join(MODELS_DIR,   f"scaler_{sector}.pkl"))

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    all_metrics  = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    all_metrics[sector] = metrics
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    stocks = instruments.get_training_stocks()
    print(f"Blue Stock AI — Training — {len(stocks)} stocks\n")
    print(f"Target: stock alpha vs Nifty >= {ALPHA_PCT:.2f}% over {FORWARD_DAYS} days "
          f"(nominal {NOMINAL_ALPHA_PCT:.2f}% + costs {TRANSACTION_COST_PCT:.2f}%)\n")

    sector_data, _ = fetch_all_stock_data(stocks, ai.UPSTOX_TOKEN)

    print(f"\nTraining {len(sector_data)} models: {list(sector_data.keys())}\n")
    validation_reports = []
    for sector, (train_df, test_df) in sector_data.items():
        lgb_m, xgb_m, scaler, metrics = train_ensemble(train_df, test_df, sector)
        save_sector_artifacts(lgb_m, xgb_m, scaler, sector, metrics)
        report = walk_forward_validate(pd.concat([train_df, test_df], ignore_index=True), sector)
        validation_reports.append(report)

    report_path = save_validation_report(validation_reports)

    print(f"\nAll models saved to {MODELS_DIR}/")
    print(f"Walk-forward report saved to {report_path}")
    print("Sectors trained:", [s for s in sector_data if s != 'general'])
    print("Done. Run main.py for the full market scan.")


if __name__ == "__main__":
    main()
