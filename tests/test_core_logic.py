import unittest

import pandas as pd

import main
import instruments


class CoreLogicTests(unittest.TestCase):
    def test_signal_label_thresholds(self):
        self.assertEqual(main._signal_label(None), "NO DATA")
        self.assertEqual(main._signal_label(66), "STRONG BUY")
        self.assertEqual(main._signal_label(65), "HOLD")
        self.assertEqual(main._signal_label(34), "SELL/WAIT")

    def test_composite_uses_only_available_signals(self):
        row = {
            "technical_score": 80,
            "fundamental_data_ok": True,
            "fundamental_score": 60,
            "sentiment_data_ok": False,
            "sentiment_score": None,
        }
        self.assertEqual(main._composite(row), 73.75)

    def test_position_size_is_zero_for_low_score(self):
        row = {"composite_score": 49.9, "atr_pct": 3.0, "buy_signal": "HOLD"}
        self.assertEqual(main._position_size_pct(row), 0.0)

    def test_build_picks_df_prefers_higher_score_for_duplicate_symbol(self):
        results = [
            {
                "status": "ok",
                "technical_score": 60,
                "composite_score": 60,
                "tradingsymbol": "ABC",
                "name": "ABC NSE",
                "exchange": "NSE",
                "close": 100,
                "instrument_key": "NSE_EQ|1",
            },
            {
                "status": "ok",
                "technical_score": 72,
                "composite_score": 72,
                "tradingsymbol": "ABC",
                "name": "ABC BSE",
                "exchange": "BSE",
                "close": 101,
                "instrument_key": "BSE_EQ|2",
            },
        ]
        df = main._build_picks_df(results)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["exchange"], "BSE")
        self.assertEqual(df.iloc[0]["rank"], 1)

    def test_instrument_filter_keeps_plain_equities(self):
        raw = pd.DataFrame(
            [
                {"instrument_key": "NSE_EQ|INE123456789", "tradingsymbol": "TCS", "name": "TCS"},
                {"instrument_key": "NSE_EQ|ABC123", "tradingsymbol": "NFO1234", "name": "Bad"},
            ]
        )
        filtered = raw[
            raw["instrument_key"].astype(str).str.contains("INE", regex=False)
            & raw["tradingsymbol"].astype(str).str.match(r"^[A-Z][A-Z0-9&-]*$")
            & ~raw["tradingsymbol"].astype(str).str.contains(r"\d{4,}")
        ]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["tradingsymbol"], "TCS")


if __name__ == "__main__":
    unittest.main()
