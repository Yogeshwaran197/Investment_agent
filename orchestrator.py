"""
Signal Orchestrator - Conviction Score per Stock
Collects signals from all 5 specialist agents and computes a weighted
conviction score for each stock. Also builds a full reasoning trace.
"""

import pandas as pd
from agents.technical_agent import TechnicalAgent
from agents.sentiment_agent import SentimentAgent
from agents.fundamental_agent import FundamentalAgent
from agents.macro_agent import MacroAgent
from agents.risk_profiler import RiskProfiler


# Agent weight in the final conviction score
AGENT_WEIGHTS = {
    "Technical":    0.25,
    "Sentiment":    0.20,
    "Fundamental":  0.30,
    "Macro":        0.15,
    "Risk Profiler": 0.10,
}


def orchestrate(
    prices: pd.DataFrame,
    tickers: list[str],
    age: int,
    horizon: int,
    risk_level: str,
    goal: str,
) -> dict:
    """
    Run all specialist agents and compute conviction scores.

    Parameters:
        prices     - DataFrame of adjusted close prices (columns = clean ticker names)
        tickers    - list of clean ticker names (without .NS)
        age, horizon, risk_level, goal - user profile inputs

    Returns:
        {
            "conviction_scores": { ticker: 0-100 },
            "agent_results": {
                "Technical":    { ticker: { ... } },
                "Sentiment":    { ticker: { ... } },
                "Fundamental":  { ticker: { ... } },
                "Macro":        { ... },
                "Risk Profiler": { ... },
            },
            "reasoning_trace": [
                { "agent": str, "ticker": str, "signal": str, "score": int, "reasoning": str },
                ...
            ],
            "macro_adjustment": float,
        }
    """
    clean_tickers = list(prices.columns)

    # -- Run each agent -------------------------------------------------------
    technical_agent = TechnicalAgent()
    sentiment_agent = SentimentAgent()
    fundamental_agent = FundamentalAgent()
    macro_agent = MacroAgent()
    risk_profiler = RiskProfiler()

    tech_results = technical_agent.analyze(prices)
    sent_results = sentiment_agent.analyze(clean_tickers)
    fund_results = fundamental_agent.analyze(clean_tickers)
    macro_result = macro_agent.analyze()
    risk_result = risk_profiler.analyze(age, horizon, risk_level, goal)

    # -- Build reasoning trace -------------------------------------------------
    reasoning_trace = []

    # Macro and Risk are portfolio-level, not per-stock
    reasoning_trace.append({
        "agent": "Macro",
        "ticker": "---",
        "signal": macro_result["environment"],
        "score": macro_result["score"],
        "reasoning": macro_result["reasoning"],
    })
    reasoning_trace.append({
        "agent": "Risk Profiler",
        "ticker": "---",
        "signal": risk_result["category"],
        "score": risk_result["risk_score"],
        "reasoning": risk_result["reasoning"],
    })

    # -- Compute conviction per stock ------------------------------------------
    macro_tilt = macro_result["adjustments"]["equity_tilt"]
    conviction_scores = {}

    for ticker in clean_tickers:
        per_stock_scores = {}

        # Technical
        if ticker in tech_results:
            t = tech_results[ticker]
            per_stock_scores["Technical"] = t["score"]
            reasoning_trace.append({
                "agent": "Technical",
                "ticker": ticker,
                "signal": t["signal"],
                "score": t["score"],
                "reasoning": t["reasoning"],
            })

        # Sentiment
        if ticker in sent_results:
            s = sent_results[ticker]
            per_stock_scores["Sentiment"] = s["score"]
            reasoning_trace.append({
                "agent": "Sentiment",
                "ticker": ticker,
                "signal": s["signal"],
                "score": s["score"],
                "reasoning": s["reasoning"],
            })

        # Fundamental
        if ticker in fund_results:
            f = fund_results[ticker]
            per_stock_scores["Fundamental"] = f["score"]
            reasoning_trace.append({
                "agent": "Fundamental",
                "ticker": ticker,
                "signal": f["signal"],
                "score": f["score"],
                "reasoning": f["reasoning"],
            })

        # Macro (same score for all stocks, but tilts conviction)
        per_stock_scores["Macro"] = macro_result["score"]

        # Risk Profiler (same for all stocks)
        per_stock_scores["Risk Profiler"] = risk_result["risk_score"]

        # -- Weighted conviction -----------------------------------------------
        weighted_sum = 0
        weight_total = 0
        for agent_name, weight in AGENT_WEIGHTS.items():
            if agent_name in per_stock_scores:
                weighted_sum += weight * per_stock_scores[agent_name]
                weight_total += weight

        if weight_total > 0:
            raw_conviction = weighted_sum / weight_total
        else:
            raw_conviction = 50

        # Apply macro tilt
        tilted_conviction = raw_conviction * macro_tilt
        conviction_scores[ticker] = max(0, min(100, int(tilted_conviction)))

    return {
        "conviction_scores": conviction_scores,
        "agent_results": {
            "Technical": tech_results,
            "Sentiment": sent_results,
            "Fundamental": fund_results,
            "Macro": macro_result,
            "Risk Profiler": risk_result,
        },
        "reasoning_trace": reasoning_trace,
        "macro_adjustment": macro_tilt,
    }