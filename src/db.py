"""Supabase Postgres sink for AI-Stock.

Strict mode: callers should let DB errors propagate so the pipeline aborts
rather than publishing a half-written result. The pipeline runs through
Supavisor's transaction-mode pooler (port 6543), so:

  * autocommit=True — no long-running transactions across statements
  * one psycopg2 connection per run, reused across all writes
  * retries only on transient errors (network, deadlock, server restart);
    schema/constraint errors fail loudly because retrying won't fix them
"""
import os
import math
import datetime
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type,
)

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Errors worth retrying — connection blips, server restarts, deadlocks.
# Schema / constraint / data-type errors are NOT retried.
TRANSIENT = (
    psycopg2.OperationalError,
    psycopg2.InterfaceError,
    psycopg2.errors.DeadlockDetected,
    psycopg2.errors.SerializationFailure,
)

_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(TRANSIENT),
)


def _require_url() -> str:
    if not SUPABASE_DB_URL:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set. Add it to AI_Analyzer/.env. "
            "Format: postgresql://postgres.<ref>:<password>@aws-0-<region>"
            ".pooler.supabase.com:6543/postgres?sslmode=require"
        )
    return SUPABASE_DB_URL


@_retry
def open_conn():
    """Open a single pooled connection. Caller is responsible for close()."""
    conn = psycopg2.connect(
        _require_url(),
        connect_timeout=10,
        application_name="ai_stock_main",
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        # Cap any single statement at 60s so a stuck query can't hang the run.
        cur.execute("SET statement_timeout = '60s'")
    return conn


@contextmanager
def connection():
    conn = open_conn()
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def ping(conn) -> None:
    """Sanity check the connection at run start so we fail fast if DB is down."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        cur.fetchone()


# ---------------------------------------------------------------------------
# Cleaning helpers — psycopg2 won't accept NaN/Inf for numeric columns.
# ---------------------------------------------------------------------------

def _clean(v):
    if v is None or v == "":
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):  # numpy scalars
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# ---------------------------------------------------------------------------
# Writers — each is idempotent (ON CONFLICT DO UPDATE).
# ---------------------------------------------------------------------------

@_retry
def upsert_run_health(conn, run_date: str, h: dict) -> None:
    sql = """
        insert into run_health (
            run_date, stocks_total, stocks_ok, stocks_illiquid,
            stocks_with_fund, stocks_with_sent, completion_rate,
            nifty_ok, fii_dii_ok, pcr_ok,
            min_avg_traded_value, min_completion_rate, publish
        ) values (
            %(run_date)s, %(stocks_total)s, %(stocks_ok)s, %(stocks_illiquid)s,
            %(stocks_with_fund)s, %(stocks_with_sent)s, %(completion_rate)s,
            %(nifty_ok)s, %(fii_dii_ok)s, %(pcr_ok)s,
            %(min_avg_traded_value)s, %(min_completion_rate)s, %(publish)s
        )
        on conflict (run_date) do update set
            stocks_total         = excluded.stocks_total,
            stocks_ok            = excluded.stocks_ok,
            stocks_illiquid      = excluded.stocks_illiquid,
            stocks_with_fund     = excluded.stocks_with_fund,
            stocks_with_sent     = excluded.stocks_with_sent,
            completion_rate      = excluded.completion_rate,
            nifty_ok             = excluded.nifty_ok,
            fii_dii_ok           = excluded.fii_dii_ok,
            pcr_ok               = excluded.pcr_ok,
            min_avg_traded_value = excluded.min_avg_traded_value,
            min_completion_rate  = excluded.min_completion_rate,
            publish              = excluded.publish
    """
    params = {
        "run_date":             run_date,
        "stocks_total":         int(h.get("stocks_total", 0)),
        "stocks_ok":            int(h.get("stocks_ok", 0)),
        "stocks_illiquid":      int(h.get("stocks_illiquid", 0)),
        "stocks_with_fund":     int(h.get("stocks_with_fund", 0)),
        "stocks_with_sent":     int(h.get("stocks_with_sent", 0)),
        "completion_rate":      float(h.get("completion_rate", 0)),
        "nifty_ok":             bool(h.get("nifty_ok", False)),
        "fii_dii_ok":           bool(h.get("fii_dii_ok", False)),
        "pcr_ok":               bool(h.get("pcr_ok", False)),
        "min_avg_traded_value": int(h.get("min_avg_traded_value", 0)),
        "min_completion_rate":  float(h.get("min_completion_rate", 0)),
        "publish":              bool(h.get("publish", False)),
    }
    with conn.cursor() as cur:
        cur.execute(sql, params)


@_retry
def upsert_market_dashboard(conn, run_date: str, m: dict) -> None:
    sql = """
        insert into market_dashboard (
            run_date, market_score, environment, nifty_trend,
            nifty_return_5d, nifty_return_20d, nifty_rsi,
            fii_flow_5d, dii_flow_5d, net_flow_5d,
            pcr, pcr_signal, strong_buys, ranked_stocks,
            leading_sector, leading_sector_score
        ) values (
            %(run_date)s, %(market_score)s, %(environment)s, %(nifty_trend)s,
            %(nifty_return_5d)s, %(nifty_return_20d)s, %(nifty_rsi)s,
            %(fii_flow_5d)s, %(dii_flow_5d)s, %(net_flow_5d)s,
            %(pcr)s, %(pcr_signal)s, %(strong_buys)s, %(ranked_stocks)s,
            %(leading_sector)s, %(leading_sector_score)s
        )
        on conflict (run_date) do update set
            market_score         = excluded.market_score,
            environment          = excluded.environment,
            nifty_trend          = excluded.nifty_trend,
            nifty_return_5d      = excluded.nifty_return_5d,
            nifty_return_20d     = excluded.nifty_return_20d,
            nifty_rsi            = excluded.nifty_rsi,
            fii_flow_5d          = excluded.fii_flow_5d,
            dii_flow_5d          = excluded.dii_flow_5d,
            net_flow_5d          = excluded.net_flow_5d,
            pcr                  = excluded.pcr,
            pcr_signal           = excluded.pcr_signal,
            strong_buys          = excluded.strong_buys,
            ranked_stocks        = excluded.ranked_stocks,
            leading_sector       = excluded.leading_sector,
            leading_sector_score = excluded.leading_sector_score,
            updated_at           = now()
    """
    params = {
        "run_date": run_date,
        "market_score": _clean(m.get("market_score")),
        "environment": _clean(m.get("environment")),
        "nifty_trend": _clean(m.get("nifty_trend")),
        "nifty_return_5d": _clean(m.get("nifty_return_5d")),
        "nifty_return_20d": _clean(m.get("nifty_return_20d")),
        "nifty_rsi": _clean(m.get("nifty_rsi")),
        "fii_flow_5d": _clean(m.get("fii_flow_5d")),
        "dii_flow_5d": _clean(m.get("dii_flow_5d")),
        "net_flow_5d": _clean(m.get("net_flow_5d")),
        "pcr": _clean(m.get("pcr")),
        "pcr_signal": _clean(m.get("pcr_signal")),
        "strong_buys": int(m.get("strong_buys", 0)),
        "ranked_stocks": int(m.get("ranked_stocks", 0)),
        "leading_sector": _clean(m.get("leading_sector")),
        "leading_sector_score": _clean(m.get("leading_sector_score")),
    }
    with conn.cursor() as cur:
        cur.execute(sql, params)


_SECTOR_COLS = (
    "run_date", "rank", "sector_model", "stock_count", "strong_buys",
    "avg_composite_score", "avg_technical_score", "avg_fundamental_score",
    "avg_sentiment_score", "best_symbol", "best_stock_name",
    "best_composite_score",
)

_SECTOR_UPDATE = ", ".join(
    f"{c} = excluded.{c}" for c in _SECTOR_COLS
    if c not in ("run_date", "sector_model")
) + ", updated_at = now()"


@_retry
def upsert_sector_leaderboard(conn, run_date: str, rows: list) -> int:
    if not rows:
        return 0
    tuples = [
        (
            run_date,
            int(r.get("rank", 0)),
            r.get("sector_model") or "unknown",
            int(r.get("stock_count", 0)),
            int(r.get("strong_buys", 0)),
            _clean(r.get("avg_composite_score")),
            _clean(r.get("avg_technical_score")),
            _clean(r.get("avg_fundamental_score")),
            _clean(r.get("avg_sentiment_score")),
            _clean(r.get("best_symbol")),
            _clean(r.get("best_stock_name")),
            _clean(r.get("best_composite_score")),
        )
        for r in rows
    ]
    sql = (
        f"insert into sector_leaderboard ({', '.join(_SECTOR_COLS)}) values %s "
        f"on conflict (run_date, sector_model) do update set {_SECTOR_UPDATE}"
    )
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, tuples, page_size=100)
    return len(tuples)


_RANKED_COLS = (
    "run_date", "tradingsymbol", "instrument_key", "name", "exchange",
    "sector_model", "rank", "close", "rsi", "atr_pct", "stop_loss",
    "target", "risk_per_share", "position_size_pct",
    "technical_score", "fundamental_score", "sentiment_score",
    "sentiment_confidence", "composite_score", "buy_signal", "confidence",
    "fundamental_data_ok", "sentiment_data_ok", "avg_traded_value_20d",
)

_RANKED_UPDATE = ", ".join(
    f"{c} = excluded.{c}" for c in _RANKED_COLS
    if c not in ("run_date", "tradingsymbol")
)


@_retry
def upsert_ranked_stocks(conn, run_date: str, df) -> int:
    """Bulk upsert ranked picks. df is a pandas DataFrame from main.py."""
    if df is None or df.empty:
        return 0

    rows = []
    for _, r in df.iterrows():
        rows.append((
            run_date,
            r.get("tradingsymbol"),
            r.get("instrument_key") or "",
            _clean(r.get("name")),
            _clean(r.get("exchange")),
            _clean(r.get("model_used")),
            int(r.get("rank")),
            _clean(r.get("close")),
            _clean(r.get("rsi")),
            _clean(r.get("atr_pct")),
            _clean(r.get("stop_loss")),
            _clean(r.get("target")),
            _clean(r.get("risk_per_share")),
            _clean(r.get("position_size_pct")),
            _clean(r.get("technical_score")),
            _clean(r.get("fundamental_score")),
            _clean(r.get("sentiment_score")),
            _clean(r.get("sentiment_confidence")),
            _clean(r.get("composite_score")),
            _clean(r.get("buy_signal")),
            _clean(r.get("confidence")),
            bool(r.get("fundamental_data_ok") or False),
            bool(r.get("sentiment_data_ok")   or False),
            _clean(r.get("avg_traded_value_20d")),
        ))

    sql = (
        f"insert into ranked_stocks ({', '.join(_RANKED_COLS)}) values %s "
        f"on conflict (run_date, tradingsymbol) do update set {_RANKED_UPDATE}"
    )
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
    return len(rows)


_PRED_COLS = (
    "predicted_at", "tradingsymbol", "instrument_key", "name", "exchange",
    "sector_model", "buy_signal", "technical_score", "fundamental_score",
    "sentiment_score", "composite_score", "confidence",
    "entry_price", "target_date",
)

_PRED_UPDATE = ", ".join(
    f"{c} = excluded.{c}" for c in _PRED_COLS
    if c not in ("predicted_at", "tradingsymbol")
)


@_retry
def upsert_predictions(conn, rows: list) -> int:
    """Insert today's actionable picks into the shadow log. Idempotent on
    (predicted_at, tradingsymbol) so re-running the same date is safe."""
    if not rows:
        return 0
    tuples = [
        (
            r["predicted_at"],
            r["tradingsymbol"],
            r.get("instrument_key") or "",
            _clean(r.get("name")),
            _clean(r.get("exchange")),
            _clean(r.get("sector_model")),
            _clean(r.get("buy_signal")),
            _clean(r.get("technical_score")),
            _clean(r.get("fundamental_score")),
            _clean(r.get("sentiment_score")),
            _clean(r.get("composite_score")),
            _clean(r.get("confidence")),
            _clean(r.get("entry_price")),
            r["target_date"],
        )
        for r in rows
    ]
    sql = (
        f"insert into predictions ({', '.join(_PRED_COLS)}) values %s "
        f"on conflict (predicted_at, tradingsymbol) do update set {_PRED_UPDATE}"
    )
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, tuples, page_size=500)
    return len(tuples)


@_retry
def fetch_pending_predictions(conn, on_or_before: str) -> list:
    """Return predictions whose target_date has passed and are not yet filled."""
    sql = """
        select predicted_at, tradingsymbol, instrument_key, entry_price, target_date
          from predictions
         where hit is null
           and target_date <= %s
         order by predicted_at, tradingsymbol
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (on_or_before,))
        return [dict(r) for r in cur.fetchall()]


@_retry
def update_outcome(conn, predicted_at, tradingsymbol,
                   exit_price, actual_return_pct, nifty_return_pct,
                   alpha_pct, hit) -> None:
    sql = """
        update predictions set
            exit_price        = %s,
            actual_return_pct = %s,
            nifty_return_pct  = %s,
            alpha_pct         = %s,
            hit               = %s,
            filled_at         = now()
         where predicted_at = %s and tradingsymbol = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            _clean(exit_price), _clean(actual_return_pct),
            _clean(nifty_return_pct), _clean(alpha_pct),
            int(hit), predicted_at, tradingsymbol,
        ))


@_retry
def live_accuracy(conn, window_days: int = 30) -> dict:
    """Live precision over predictions whose outcomes are filled in the window."""
    cutoff = (datetime.date.today() - datetime.timedelta(days=window_days)).isoformat()
    sql = """
        select count(*)::int                       as filled,
               coalesce(sum(hit), 0)::int          as hits,
               avg(alpha_pct)::numeric(8,4)        as avg_alpha
          from predictions
         where hit is not null
           and predicted_at >= %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (cutoff,))
        row = cur.fetchone()
    filled    = row["filled"]
    hits      = row["hits"]
    avg_alpha = float(row["avg_alpha"]) if row["avg_alpha"] is not None else None
    return {
        "window_days":   window_days,
        "filled":        filled,
        "hits":          hits,
        "precision":     round(hits / filled, 4) if filled else None,
        "avg_alpha_pct": round(avg_alpha, 4) if avg_alpha is not None else None,
    }
