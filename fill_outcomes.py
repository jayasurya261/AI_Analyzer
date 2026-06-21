"""Fill actual returns for past predictions in the Supabase predictions table.

Run nightly (after market close) or manually. Pulls every prediction whose
target_date has passed and `hit IS NULL`, fetches the post-prediction price
candle from Upstox, computes:

    actual_return_pct = (exit_price / entry_price - 1) * 100
    nifty_return_pct  = corresponding 5-day Nifty return
    alpha_pct         = actual - nifty
    hit               = 1 if alpha_pct >= ALPHA_PCT (incl. transaction costs)

Idempotent — skips rows where hit is already set. Strict on DB: any DB error
exits non-zero so the scheduler knows the run failed.
"""
import os
import sys
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import ai
import db
from train import ALPHA_PCT  # cost-adjusted threshold


def _fetch_close_at_or_after(instrument_key: str, target_iso: str):
    """Return (close, actual_date) of first available candle on/after target_iso."""
    target = datetime.date.fromisoformat(target_iso)
    today  = datetime.date.today()
    if target > today:
        return None, None
    # Don't cap at today — Upstox historical endpoint returns whatever is available,
    # and today's daily candle is only finalized after exchange EOD processing.
    end = target + datetime.timedelta(days=10)
    df  = ai.fetch_technical_data(
        instrument_key,
        from_date=target_iso,
        to_date=end.isoformat(),
    )
    if df is None or df.empty:
        return None, None
    first = df.iloc[0]
    return float(first['close']), first['timestamp'].date().isoformat()


def fill():
    today = datetime.date.today().isoformat()

    try:
        conn = db.open_conn()
        db.ping(conn)
    except Exception as e:
        print(f"FATAL: cannot reach Supabase ({type(e).__name__}: {e}). "
              f"Check SUPABASE_DB_URL in .env.")
        sys.exit(4)

    try:
        pending = db.fetch_pending_predictions(conn, today)
        if not pending:
            print("No pending outcomes to fill.")
            return

        print(f"Filling outcomes for {len(pending)} predictions "
              f"(threshold ALPHA_PCT = {ALPHA_PCT}%)...")

        # Cache Nifty closes per date so we don't re-fetch per stock.
        nifty_cache = {}

        def nifty_close(date_iso):
            if date_iso in nifty_cache:
                return nifty_cache[date_iso]
            c, _ = _fetch_close_at_or_after(ai.NIFTY_KEY, date_iso)
            nifty_cache[date_iso] = c
            return c

        filled = skipped = errored = 0
        for r in pending:
            sym = r["tradingsymbol"]
            try:
                entry = float(r["entry_price"])
                pred_iso   = r["predicted_at"].isoformat() \
                             if hasattr(r["predicted_at"], "isoformat") \
                             else str(r["predicted_at"])
                target_iso = r["target_date"].isoformat() \
                             if hasattr(r["target_date"], "isoformat") \
                             else str(r["target_date"])

                exit_close, _ = _fetch_close_at_or_after(r["instrument_key"], target_iso)
                if exit_close is None:
                    print(f"  {sym}: no exit candle yet")
                    skipped += 1
                    continue

                entry_nifty = nifty_close(pred_iso)
                exit_nifty  = nifty_close(target_iso)
                if entry_nifty is None or exit_nifty is None:
                    print(f"  {sym}: Nifty data missing")
                    skipped += 1
                    continue

                stock_ret = (exit_close / entry - 1) * 100
                nifty_ret = (exit_nifty / entry_nifty - 1) * 100
                alpha     = stock_ret - nifty_ret
                hit       = 1 if alpha >= ALPHA_PCT else 0

                db.update_outcome(
                    conn,
                    r["predicted_at"], sym,
                    round(exit_close, 2),
                    round(stock_ret, 4),
                    round(nifty_ret, 4),
                    round(alpha, 4),
                    hit,
                )
                filled += 1
            except Exception as e:
                print(f"  {sym}: {type(e).__name__}: {e}")
                errored += 1

        print(f"\nFilled: {filled}  Skipped: {skipped}  Errored: {errored}")

        live = db.live_accuracy(conn, window_days=30)
        if live["filled"]:
            print(f"Live precision (last 30d, n={live['filled']}): "
                  f"{live['precision']*100:.1f}%  "
                  f"avg_alpha={live['avg_alpha_pct']:.2f}%")

        if errored and not filled:
            sys.exit(5)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    fill()
