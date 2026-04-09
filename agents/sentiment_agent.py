"""
Sentiment Agent — News headlines · Groq LLM scoring
Fetches recent news via Google News RSS and uses Groq to score sentiment.
"""

import os
import re
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.parse import quote

from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)


# Mapping of clean ticker names to company names for better news search
COMPANY_NAMES = {
    "TCS": "TCS Tata Consultancy",
    "INFY": "Infosys",
    "WIPRO": "Wipro",
    "HCLTECH": "HCL Technologies",
    "TECHM": "Tech Mahindra",
    "SUNPHARMA": "Sun Pharma",
    "DRREDDY": "Dr Reddys",
    "CIPLA": "Cipla",
    "DIVISLAB": "Divis Laboratories",
    "RELIANCE": "Reliance Industries",
    "ONGC": "ONGC",
    "NTPC": "NTPC",
    "POWERGRID": "Power Grid Corporation",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "SBI State Bank India",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "AXISBANK": "Axis Bank",
    "HINDUNILVR": "Hindustan Unilever",
    "NESTLEIND": "Nestle India",
    "TITAN": "Titan Company",
    "ASIANPAINT": "Asian Paints",
    "MARUTI": "Maruti Suzuki",
    "TATAMOTORS": "Tata Motors",
    "M&M": "Mahindra and Mahindra",
    "BAJAJ-AUTO": "Bajaj Auto",
    "NIFTYBEES": "Nifty BeES ETF",
    "JUNIORBEES": "Junior BeES ETF",
    "SETFNN50": "Nifty Next 50 ETF",
}


class SentimentAgent:
    """Fetches recent news headlines and scores market sentiment per stock
    using Groq LLM. Returns BULLISH / NEUTRAL / BEARISH with 0-100 score."""

    NAME = "Sentiment"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    # ── Fetch headlines from Google News RSS ─────────────────────────────────
    @staticmethod
    def _fetch_headlines(query: str, max_results: int = 5) -> list[str]:
        """Fetch news headlines from Google News RSS feed."""
        try:
            url = f"https://news.google.com/rss/search?q={quote(query + ' stock India')}&hl=en-IN&gl=IN&ceid=IN:en"
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as response:
                xml_data = response.read().decode("utf-8")
            root = ET.fromstring(xml_data)
            items = root.findall(".//item/title")
            headlines = [item.text for item in items[:max_results] if item.text]
            return headlines
        except Exception:
            return []

    # ── LLM-based sentiment scoring ──────────────────────────────────────────
    def _score_with_llm(self, ticker: str, headlines: list[str]) -> dict:
        """Use Groq LLM to score sentiment from headlines."""
        if not self.client:
            return {
                "signal": "NEUTRAL",
                "score": 50,
                "reasoning": "GROQ_API_KEY not set; cannot score sentiment.",
                "headlines": headlines,
            }

        headlines_text = "\n".join([f"  - {h}" for h in headlines])

        prompt = f"""Analyse the following recent news headlines for {ticker} (Indian stock market / NSE).
Rate the overall sentiment on a scale of 0 to 100 where:
- 0-30 = Very Bearish (negative outlook)
- 31-45 = Bearish
- 46-55 = Neutral
- 56-70 = Bullish
- 71-100 = Very Bullish (positive outlook)

Headlines:
{headlines_text}

Respond in EXACTLY this format (no extra text):
SCORE: <number>
SIGNAL: <BULLISH|NEUTRAL|BEARISH>
REASON: <one-line reason>"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyst for Indian equities. Be concise."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.3,
            )
            text = response.choices[0].message.content.strip()

            # Parse response
            score_match = re.search(r"SCORE:\s*(\d+)", text)
            signal_match = re.search(r"SIGNAL:\s*(BULLISH|NEUTRAL|BEARISH)", text)
            reason_match = re.search(r"REASON:\s*(.+)", text)

            score = int(score_match.group(1)) if score_match else 50
            score = max(0, min(100, score))
            signal = signal_match.group(1) if signal_match else "NEUTRAL"
            reason = reason_match.group(1).strip() if reason_match else "Unable to parse LLM response."

            return {
                "signal": signal,
                "score": score,
                "reasoning": reason,
                "headlines": headlines,
            }
        except Exception as e:
            return {
                "signal": "NEUTRAL",
                "score": 50,
                "reasoning": f"LLM sentiment scoring failed: {e}",
                "headlines": headlines,
            }

    # ── Fallback: keyword-based scoring ──────────────────────────────────────
    @staticmethod
    def _keyword_score(headlines: list[str]) -> dict:
        """Simple keyword-based fallback when LLM is not available."""
        positive = ["surge", "rally", "gain", "profit", "growth", "rise", "up",
                     "record", "high", "strong", "buy", "upgrade", "bullish", "beat"]
        negative = ["fall", "drop", "loss", "crash", "decline", "down", "low",
                     "weak", "sell", "downgrade", "bearish", "miss", "debt", "fraud"]

        pos_count = 0
        neg_count = 0
        joined = " ".join(headlines).lower()
        for word in positive:
            pos_count += joined.count(word)
        for word in negative:
            neg_count += joined.count(word)

        total = pos_count + neg_count
        if total == 0:
            return {"signal": "NEUTRAL", "score": 50,
                    "reasoning": "No strong sentiment keywords found.", "headlines": headlines}

        score = int(50 + (pos_count - neg_count) / total * 40)
        score = max(0, min(100, score))

        if score >= 60:
            signal = "BULLISH"
        elif score <= 40:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {"signal": signal, "score": score,
                "reasoning": f"Keyword analysis: {pos_count} positive, {neg_count} negative mentions.",
                "headlines": headlines}

    # ── Public API ───────────────────────────────────────────────────────────
    def analyze(self, tickers: list[str]) -> dict:
        """
        Analyse sentiment for a list of tickers.

        Returns:
            { ticker: { "signal", "score", "reasoning", "headlines" } }
        """
        results = {}
        for ticker in tickers:
            company = COMPANY_NAMES.get(ticker, ticker)
            headlines = self._fetch_headlines(company)

            if not headlines:
                # No headlines found — use a neutral default
                results[ticker] = {
                    "signal": "NEUTRAL",
                    "score": 50,
                    "reasoning": f"No recent news found for {ticker}.",
                    "headlines": [],
                }
                continue

            if self.client:
                results[ticker] = self._score_with_llm(ticker, headlines)
            else:
                results[ticker] = self._keyword_score(headlines)

        return results
