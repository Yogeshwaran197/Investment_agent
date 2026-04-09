"""
Technical Agent — RSI · MACD · Bollinger Bands
Analyses price history and returns a signal + conviction score per ticker.
"""

import numpy as np
import pandas as pd


class TechnicalAgent:
    """Computes RSI-14, MACD(12,26,9), Bollinger Bands(20,2σ) and produces
    a BUY / HOLD / SELL signal with a 0-100 conviction score per stock."""

    NAME = "Technical"

    # ── RSI ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    @staticmethod
    def _bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return sma, upper, lower

    # ── Score a single ticker ────────────────────────────────────────────────
    def _score_ticker(self, prices: pd.Series) -> dict:
        if prices.dropna().shape[0] < 30:
            return {"signal": "HOLD", "score": 50,
                    "reasoning": "Insufficient price history for technical analysis."}

        reasoning_parts = []
        scores = []

        # RSI
        rsi = self._rsi(prices)
        latest_rsi = float(rsi.iloc[-1]) if not rsi.iloc[-1] != rsi.iloc[-1] else 50.0
        if latest_rsi < 30:
            rsi_score = 80  # oversold → bullish
            reasoning_parts.append(f"RSI={latest_rsi:.1f} (oversold → bullish)")
        elif latest_rsi > 70:
            rsi_score = 20  # overbought → bearish
            reasoning_parts.append(f"RSI={latest_rsi:.1f} (overbought → bearish)")
        else:
            rsi_score = 50 + (50 - latest_rsi)  # neutral zone, slight tilt
            reasoning_parts.append(f"RSI={latest_rsi:.1f} (neutral)")
        scores.append(rsi_score)

        # MACD
        macd_line, signal_line, histogram = self._macd(prices)
        latest_hist = float(histogram.iloc[-1]) if not np.isnan(histogram.iloc[-1]) else 0.0
        prev_hist = float(histogram.iloc[-2]) if len(histogram) > 1 and not np.isnan(histogram.iloc[-2]) else 0.0
        if latest_hist > 0 and latest_hist > prev_hist:
            macd_score = 75
            reasoning_parts.append("MACD histogram rising above zero (bullish momentum)")
        elif latest_hist > 0:
            macd_score = 60
            reasoning_parts.append("MACD histogram positive but weakening")
        elif latest_hist < 0 and latest_hist < prev_hist:
            macd_score = 25
            reasoning_parts.append("MACD histogram falling below zero (bearish momentum)")
        elif latest_hist < 0:
            macd_score = 40
            reasoning_parts.append("MACD histogram negative but recovering")
        else:
            macd_score = 50
            reasoning_parts.append("MACD neutral")
        scores.append(macd_score)

        # Bollinger Bands
        sma, upper, lower = self._bollinger(prices)
        latest_price = float(prices.iloc[-1])
        latest_upper = float(upper.iloc[-1]) if not np.isnan(upper.iloc[-1]) else latest_price
        latest_lower = float(lower.iloc[-1]) if not np.isnan(lower.iloc[-1]) else latest_price
        latest_sma = float(sma.iloc[-1]) if not np.isnan(sma.iloc[-1]) else latest_price

        if latest_price <= latest_lower:
            bb_score = 78
            reasoning_parts.append(f"Price at lower Bollinger Band ₹{latest_lower:.0f} (potential bounce)")
        elif latest_price >= latest_upper:
            bb_score = 22
            reasoning_parts.append(f"Price at upper Bollinger Band ₹{latest_upper:.0f} (potential pullback)")
        else:
            # Position within band (0 = lower, 1 = upper)
            band_width = latest_upper - latest_lower
            if band_width > 0:
                position = (latest_price - latest_lower) / band_width
                bb_score = int(50 + (0.5 - position) * 40)
            else:
                bb_score = 50
            reasoning_parts.append(f"Price within Bollinger Bands (SMA=₹{latest_sma:.0f})")
        scores.append(bb_score)

        # Aggregate
        final_score = int(np.mean(scores))
        final_score = max(0, min(100, final_score))

        if final_score >= 65:
            signal = "BUY"
        elif final_score <= 35:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "score": final_score,
            "reasoning": " | ".join(reasoning_parts),
            "indicators": {
                "rsi": round(latest_rsi, 2),
                "macd_histogram": round(latest_hist, 4),
                "bb_position": round(latest_price, 2),
            }
        }

    # ── Public API ───────────────────────────────────────────────────────────
    def analyze(self, prices: pd.DataFrame) -> dict:
        """
        Analyse all tickers in the price DataFrame.

        Returns:
            { ticker: { "signal", "score", "reasoning", "indicators" } }
        """
        results = {}
        for col in prices.columns:
            results[col] = self._score_ticker(prices[col])
        return results
