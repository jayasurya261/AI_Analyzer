"""
Backfill May 13 predictions from checkpoint into Supabase.

The May 13 main.py run saved a checkpoint but never wrote to the DB.
This script reconstructs the composite_score / buy_signal / confidence
using the same weights as main.py, then inserts top picks into predictions.

Usage:
    python backfill_may13.py [--dry-run]
"""
import sys
import json
import datetime

sys.path.insert(0, "src")
import db

# ── weights (mirror main.py) ─────────────────────────────────────────────────
W_TECH, W_FUND, W_SENT = 0.55, 0.25, 0.20

FORWARD_DAYS   = 5
MIN_COMPOSITE  = 60.0
TOP_N          = 30
CHECKPOINT     = "checkpoints/progress_2026-05-13.json"
PRED_DATE      = "2026-05-13"


# ── helpers (mirrors main.py) ─────────────────────────────────────────────────

def _composite(row) -> float:
    tech = row.get("technical_score")
    if tech is None:
        return None
    parts = [(W_TECH, float(tech))]
    if row.get("fundamental_data_ok") and row.get("fundamental_score") is not None:
        parts.append((W_FUND, float(row["fundamental_score"])))
    if row.get("sentiment_data_ok") and row.get("sentiment_score") is not None:
        sent = float(row["sentiment_score"])
        conf = float(row.get("sentiment_confidence", 1.0))
        parts.append((W_SENT, (sent * conf) * 50 + 50))
    total_w = sum(w for w, _ in parts)
    return round(sum(w * v for w, v in parts) / total_w, 2)


def _signal_label(score):
    if score is None: return "NO DATA"
    if score > 65:    return "STRONG BUY"
    if score < 35:    return "SELL/WAIT"
    return "HOLD"


def _next_trading_day(d: datetime.date, n: int) -> datetime.date:
    cur, left = d, n
    while left > 0:
        cur += datetime.timedelta(days=1)
        if cur.weekday() < 5:
            left -= 1
    return cur


# ── main ─────────────────────────────────────────────────────────────────────

def build_rows(results: list) -> list:
    pred_date = datetime.date.fromisoformat(PRED_DATE)
    target    = _next_trading_day(pred_date, FORWARD_DAYS).isoformat()

    enriched = []
    for r in results:
        if r.get("status") != "ok" or r.get("technical_score") is None:
            continue
        r["fundamental_data_ok"] = False
        r["sentiment_data_ok"]   = r.get("sentiment_score") is not None
        r["composite_score"]     = _composite(r)
        r["buy_signal"]          = _signal_label(r.get("technical_score"))
        sources = 1 + int(r["sentiment_data_ok"])
        r["confidence"]          = {1: "TECH-ONLY", 2: "TECH+1", 3: "FULL"}[sources]
        enriched.append(r)

    # Sort by composite_score descending, take top picks
    enriched.sort(key=lambda r: r.get("composite_score") or 0, reverse=True)

    above  = [r for r in enriched if (r.get("composite_score") or 0) >= MIN_COMPOSITE]
    picks  = above if len(above) >= TOP_N else enriched[:TOP_N]

    seen = set()
    rows = []
    for r in picks:
        sym = r.get("tradingsymbol")
        if sym in seen:
            continue
        seen.add(sym)
        rows.append({
            "predicted_at":      PRED_DATE,
            "tradingsymbol":     sym,
            "instrument_key":    r.get("instrument_key", ""),
            "name":              r.get("name"),
            "exchange":          r.get("exchange"),
            "sector_model":      r.get("model_used"),
            "buy_signal":        r.get("buy_signal"),
            "technical_score":   r.get("technical_score"),
            "fundamental_score": None,
            "sentiment_score":   r.get("sentiment_score"),
            "composite_score":   r.get("composite_score"),
            "confidence":        r.get("confidence"),
            "entry_price":       r.get("close"),
            "target_date":       target,
        })
    return rows


def main():
    dry_run = "--dry-run" in sys.argv

    print(f"Loading checkpoint: {CHECKPOINT}")
    with open(CHECKPOINT) as f:
        data = json.load(f)

    results = data["results"]
    print(f"  {len(results)} stocks in checkpoint")

    rows = build_rows(results)
    print(f"  {len(rows)} picks to insert  (target_date = {rows[0]['target_date'] if rows else 'N/A'})")

    if not rows:
        print("Nothing to insert.")
        return

    # Preview top 5
    print("\nTop 5 picks:")
    for r in rows[:5]:
        print(f"  {r['tradingsymbol']:12s}  composite={r['composite_score']}  signal={r['buy_signal']}")

    if dry_run:
        print("\n[dry-run] Skipping DB insert.")
        return

    confirm = input(f"\nInsert {len(rows)} rows into predictions for {PRED_DATE}? [y/N] ")
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return

    conn = db.open_conn()
    inserted = db.upsert_predictions(conn, rows)
    conn.close()
    print(f"\nDone. {inserted} rows upserted for {PRED_DATE}.")


if __name__ == "__main__":
    main()
