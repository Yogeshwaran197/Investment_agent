"""
Fundamental Agent — P/E · EPS · Debt-to-Equity
Fetches fundamental data from yfinance and scores each stock's value.
"""

import yfinance as yf
import numpy as np


class FundamentalAgent:
    """Evaluates stocks on P/E ratio, EPS growth, and debt-to-equity.
    Returns STRONG / FAIR / WEAK signal with 0-100 score."""

    NAME = "Fundamental"

    # Indian market benchmarks
    NIFTY_AVG_PE = 22.0         # Long-run Nifty 50 average P/E
    HEALTHY_DEBT_EQUITY = 1.0   # Debt/Equity below this is considered healthy

    def _fetch_fundamentals(self, ticker_ns: str) -> dict:
        """Fetch key fundamental metrics from yfinance for an NSE ticker."""
        try:
            stock = yf.Ticker(ticker_ns)
            info = stock.info
            return {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "forward_eps": info.get("forwardEps"),
                "debt_to_equity": info.get("debtToEquity"),
                "roe": info.get("returnOnEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "profit_margins": info.get("profitMargins"),
                "market_cap": info.get("marketCap"),
                "sector": info.get("sector", "N/A"),
                "name": info.get("longName", ticker_ns),
            }
        except Exception:
            return {}

    def _score_ticker(self, ticker: str) -> dict:
        """Compute a fundamental score for a single ticker."""
        ticker_ns = f"{ticker}.NS"
        data = self._fetch_fundamentals(ticker_ns)

        if not data:
            return {
                "signal": "FAIR",
                "score": 50,
                "reasoning": f"Could not fetch fundamental data for {ticker}.",
                "metrics": {},
            }

        scores = []
        reasoning_parts = []
        metrics = {}

        # ── P/E Ratio Score ──────────────────────────────────────────────────
        pe = data.get("pe_ratio")
        if pe and isinstance(pe, (int, float)) and pe > 0:
            metrics["P/E"] = round(pe, 2)
            if pe < self.NIFTY_AVG_PE * 0.7:
                pe_score = 85  # undervalued
                reasoning_parts.append(f"P/E={pe:.1f} (undervalued vs Nifty avg {self.NIFTY_AVG_PE})")
            elif pe < self.NIFTY_AVG_PE:
                pe_score = 70  # fairly valued, leaning cheap
                reasoning_parts.append(f"P/E={pe:.1f} (below Nifty avg, reasonable)")
            elif pe < self.NIFTY_AVG_PE * 1.5:
                pe_score = 50  # slightly expensive
                reasoning_parts.append(f"P/E={pe:.1f} (above Nifty avg, growth priced in)")
            else:
                pe_score = 25  # expensive
                reasoning_parts.append(f"P/E={pe:.1f} (expensive, high expectations)")
            scores.append(pe_score)
        else:
            reasoning_parts.append("P/E data unavailable")

        # ── EPS Score ────────────────────────────────────────────────────────
        eps = data.get("eps")
        forward_eps = data.get("forward_eps")
        if eps and forward_eps and isinstance(eps, (int, float)) and isinstance(forward_eps, (int, float)):
            metrics["EPS"] = round(eps, 2)
            metrics["Forward EPS"] = round(forward_eps, 2)
            if eps > 0:
                eps_growth = (forward_eps - eps) / abs(eps) * 100
                metrics["EPS Growth %"] = round(eps_growth, 1)
                if eps_growth > 15:
                    eps_score = 85
                    reasoning_parts.append(f"EPS growth {eps_growth:.0f}% (strong)")
                elif eps_growth > 5:
                    eps_score = 65
                    reasoning_parts.append(f"EPS growth {eps_growth:.0f}% (moderate)")
                elif eps_growth > 0:
                    eps_score = 50
                    reasoning_parts.append(f"EPS growth {eps_growth:.0f}% (slow)")
                else:
                    eps_score = 30
                    reasoning_parts.append(f"EPS declining {eps_growth:.0f}% (concerning)")
                scores.append(eps_score)
            else:
                reasoning_parts.append("EPS negative — company not profitable")
                scores.append(20)
        elif eps and isinstance(eps, (int, float)):
            metrics["EPS"] = round(eps, 2)
            if eps > 0:
                scores.append(55)
                reasoning_parts.append(f"EPS=₹{eps:.1f} (profitable, no forward data)")
            else:
                scores.append(25)
                reasoning_parts.append(f"EPS=₹{eps:.1f} (not profitable)")

        # ── Debt-to-Equity Score ─────────────────────────────────────────────
        de = data.get("debt_to_equity")
        if de is not None and isinstance(de, (int, float)):
            de_val = de / 100 if de > 10 else de  # yfinance sometimes returns as percentage
            metrics["Debt/Equity"] = round(de_val, 2)
            if de_val < 0.3:
                de_score = 90
                reasoning_parts.append(f"D/E={de_val:.2f} (very low debt)")
            elif de_val < self.HEALTHY_DEBT_EQUITY:
                de_score = 70
                reasoning_parts.append(f"D/E={de_val:.2f} (healthy)")
            elif de_val < 2.0:
                de_score = 45
                reasoning_parts.append(f"D/E={de_val:.2f} (moderately leveraged)")
            else:
                de_score = 20
                reasoning_parts.append(f"D/E={de_val:.2f} (highly leveraged)")
            scores.append(de_score)
        else:
            reasoning_parts.append("Debt/Equity data unavailable")

        # ── ROE bonus ────────────────────────────────────────────────────────
        roe = data.get("roe")
        if roe is not None and isinstance(roe, (int, float)):
            metrics["ROE %"] = round(roe * 100, 1)
            if roe > 0.20:
                scores.append(80)
                reasoning_parts.append(f"ROE={roe*100:.0f}% (excellent capital efficiency)")
            elif roe > 0.12:
                scores.append(60)
                reasoning_parts.append(f"ROE={roe*100:.0f}% (good)")
            else:
                scores.append(40)
                reasoning_parts.append(f"ROE={roe*100:.0f}% (below average)")

        # ── Aggregate ────────────────────────────────────────────────────────
        if scores:
            final_score = int(np.mean(scores))
        else:
            final_score = 50

        final_score = max(0, min(100, final_score))

        if final_score >= 65:
            signal = "STRONG"
        elif final_score <= 35:
            signal = "WEAK"
        else:
            signal = "FAIR"

        return {
            "signal": signal,
            "score": final_score,
            "reasoning": " | ".join(reasoning_parts) if reasoning_parts else "Insufficient data for scoring.",
            "metrics": metrics,
        }

    # ── Public API ───────────────────────────────────────────────────────────
    def analyze(self, tickers: list[str]) -> dict:
        """
        Analyse fundamentals for a list of tickers (without .NS suffix).

        Returns:
            { ticker: { "signal", "score", "reasoning", "metrics" } }
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self._score_ticker(ticker)
        return results
