"""
Portfolio Optimizer — Conviction-Weighted MVO · SIP Mode · Benchmark
Uses PyPortfolioOpt with conviction scores from the orchestrator to
tilt expected returns before optimisation.
"""

from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import pandas as pd
import numpy as np
import yfinance as yf

# Realistic return targets for Indian market (annualised)
RETURN_TARGETS = {
    "Conservative": 0.10,   # ~10% — close to Nifty 50 long-run avg for low-risk
    "Moderate":     0.15,   # ~15% — balanced equity/bluechip
    "Aggressive":   None,   # max_sharpe — let optimiser decide
}

# Benchmark tickers
BENCHMARKS = {
    "Nifty 50": "^NSEI",
    "Sensex":   "^BSESN",
}


def _apply_conviction_views(
    mu: pd.Series,
    conviction_scores: dict[str, int],
    strength: float = 0.3
) -> pd.Series:
    """
    Tilt expected returns based on conviction scores.
    Conviction > 60 → boost expected return, < 40 → reduce.

    Parameters:
        mu               – base expected returns from historical data
        conviction_scores – { ticker: 0-100 } from orchestrator
        strength          – how strongly convictions tilt returns (0-1)
    """
    adjusted = mu.copy()
    for ticker in adjusted.index:
        if ticker in conviction_scores:
            conv = conviction_scores[ticker]
            # Map conviction 0-100 to multiplier [1-strength, 1+strength]
            # 50 → 1.0 (no change), 100 → 1+strength, 0 → 1-strength
            multiplier = 1.0 + strength * (conv - 50) / 50
            adjusted[ticker] = adjusted[ticker] * multiplier
    return adjusted


def optimize_portfolio(
    prices: pd.DataFrame,
    risk_level: str,
    investment_amount: float,
    conviction_scores: dict[str, int] | None = None,
    sip_monthly: float | None = None,
):
    """
    Runs Conviction-Weighted Mean-Variance Optimisation.

    Parameters:
        prices            – DataFrame of adjusted close prices
        risk_level        – "Conservative" / "Moderate" / "Aggressive"
        investment_amount – lump sum in ₹
        conviction_scores – optional { ticker: 0-100 } from orchestrator
        sip_monthly       – optional monthly SIP amount in ₹

    Returns:
        weights      – dict of {ticker: weight}
        performance  – (expected_return, volatility, sharpe_ratio)
        allocation   – dict of {ticker: num_shares}
        leftover_inr – uninvested cash in ₹
        sip_schedule – list of monthly allocations (if SIP mode) or None
    """
    if prices.shape[1] < 2:
        raise ValueError("Need at least 2 valid tickers to optimise a portfolio.")

    mu = expected_returns.mean_historical_return(prices)
    S  = risk_models.sample_cov(prices)

    # Apply conviction tilts if provided
    if conviction_scores:
        mu = _apply_conviction_views(mu, conviction_scores)

    ef = EfficientFrontier(mu, S)

    if risk_level == "Conservative":
        ef.min_volatility()
    elif risk_level == "Aggressive":
        ef.max_sharpe()
    else:  # Moderate
        # Guard: if target return exceeds the highest individual stock return, fall back
        target = RETURN_TARGETS["Moderate"]
        if target > float(mu.max()):
            ef.max_sharpe()
        else:
            ef.efficient_return(target_return=target)

    weights     = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=False)

    latest_prices = get_latest_prices(prices)

    # ── Lump-sum allocation ──────────────────────────────────────────────────
    try:
        da = DiscreteAllocation(
            weights, latest_prices,
            total_portfolio_value=investment_amount
        )
        allocation, leftover = da.lp_portfolio()
    except Exception:
        da = DiscreteAllocation(
            weights, latest_prices,
            total_portfolio_value=investment_amount
        )
        allocation, leftover = da.greedy_portfolio()

    # ── SIP schedule ─────────────────────────────────────────────────────────
    sip_schedule = None
    if sip_monthly and sip_monthly > 0:
        sip_schedule = _compute_sip_schedule(weights, latest_prices, sip_monthly)

    return weights, performance, allocation, leftover, sip_schedule


def _compute_sip_schedule(
    weights: dict,
    latest_prices: pd.Series,
    monthly_amount: float,
    months: int = 12,
) -> list[dict]:
    """
    Compute a 12-month SIP allocation schedule.

    Returns a list of dicts, one per month, each containing:
        { "month": int, "stocks": { ticker: shares }, "invested": float, "cash": float }
    """
    schedule = []
    for month in range(1, months + 1):
        month_alloc = {}
        invested = 0
        for ticker, weight in weights.items():
            if weight < 0.001:
                continue
            amount_for_stock = monthly_amount * weight
            price = float(latest_prices.get(ticker, 0))
            if price > 0:
                shares = int(amount_for_stock / price)
                if shares > 0:
                    month_alloc[ticker] = shares
                    invested += shares * price

        cash = monthly_amount - invested
        schedule.append({
            "month": month,
            "stocks": month_alloc,
            "invested": round(invested, 2),
            "cash": round(cash, 2),
        })

    return schedule


def fetch_benchmark_data(period: str = "1y") -> pd.DataFrame:
    """
    Fetch Nifty 50 and Sensex price data for benchmark comparison.

    Returns:
        DataFrame with columns ["Nifty 50", "Sensex"] of adjusted close prices.
    """
    bench_data = {}
    for name, ticker in BENCHMARKS.items():
        try:
            data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if data.empty:
                continue
            # Handle multi-level columns from newer yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten: grab the "Close" column for this ticker
                if "Close" in data.columns.get_level_values(0):
                    close_series = data["Close"]
                    # If still a DataFrame (multi-ticker), take first column
                    if isinstance(close_series, pd.DataFrame):
                        close_series = close_series.iloc[:, 0]
                    bench_data[name] = close_series
            else:
                if "Close" in data.columns:
                    bench_data[name] = data["Close"]
        except Exception:
            pass

    if bench_data:
        df = pd.DataFrame(bench_data)
        df.dropna(inplace=True)
        return df
    return pd.DataFrame()


def compute_benchmark_comparison(
    portfolio_weights: dict,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
) -> dict:
    """
    Compare portfolio returns against benchmarks.

    Returns:
        {
            "portfolio_return": float,
            "benchmarks": { name: { "return": float, "alpha": float } },
            "normalized": DataFrame  (portfolio + benchmarks normalised to 100)
        }
    """
    # Portfolio daily returns (weighted)
    daily_returns = prices.pct_change().dropna()
    weight_series = pd.Series(portfolio_weights)
    # Only include stocks that exist in both
    common = weight_series.index.intersection(daily_returns.columns)
    if len(common) == 0:
        return {"portfolio_return": 0, "benchmarks": {}, "normalized": pd.DataFrame()}

    weight_series = weight_series[common]
    weight_series = weight_series / weight_series.sum()  # renormalize

    portfolio_daily = (daily_returns[common] * weight_series).sum(axis=1)
    portfolio_cumulative = (1 + portfolio_daily).cumprod()

    result = {
        "portfolio_return": float(portfolio_cumulative.iloc[-1] - 1),
        "benchmarks": {},
    }

    # Build normalised comparison
    normalized = pd.DataFrame(index=portfolio_cumulative.index)
    normalized["Your Portfolio"] = portfolio_cumulative / portfolio_cumulative.iloc[0] * 100

    for bench_name in benchmark_prices.columns:
        bench_series = benchmark_prices[bench_name]
        # Align dates
        bench_aligned = bench_series.reindex(portfolio_cumulative.index, method="ffill").dropna()
        if len(bench_aligned) > 1:
            bench_return = float(bench_aligned.iloc[-1] / bench_aligned.iloc[0] - 1)
            bench_normalized = bench_aligned / bench_aligned.iloc[0] * 100
            normalized[bench_name] = bench_normalized

            result["benchmarks"][bench_name] = {
                "return": bench_return,
                "alpha": result["portfolio_return"] - bench_return,
            }

    result["normalized"] = normalized
    return result