import yfinance as yf
import pandas as pd
import streamlit as st

# Indian stocks use .NS suffix (NSE) on yfinance
TICKERS = {
    "Tech": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS"],
    "Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "Consumer": ["HINDUNILVR.NS", "NESTLEIND.NS", "TITAN.NS", "ASIANPAINT.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS"],
    "Index ETF": ["NIFTYBEES.NS", "JUNIORBEES.NS", "SETFNN50.NS"],
}


def display_name(ticker: str) -> str:
    """Strip .NS suffix for cleaner display."""
    return ticker.replace(".NS", "")


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(tickers_tuple: tuple, period: str = "1y") -> pd.DataFrame:
    """
    Downloads adjusted Close prices for given tickers from NSE via yfinance.
    Returns a clean DataFrame with ticker symbols as columns (without .NS suffix).

    NOTE: accepts a *tuple* of tickers (not list) so Streamlit can hash it for caching.
    """
    tickers = list(tickers_tuple)

    if not tickers:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers, period=period, auto_adjust=True,
            group_by="ticker", progress=False, threads=True
        )
    except Exception as e:
        st.error(f"yfinance download failed: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if len(tickers) == 1:
        data = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        close_data = {}
        for ticker in tickers:
            try:
                close_data[ticker] = raw[ticker]["Close"]
            except Exception:
                pass  # Skip tickers that failed to download
        data = pd.DataFrame(close_data)

    if data.empty:
        return pd.DataFrame()

    # Remove columns with >10 % missing data, then drop remaining NaN rows
    data.dropna(axis=1, thresh=int(len(data) * 0.9), inplace=True)
    data.dropna(inplace=True)

    # Rename columns: remove .NS suffix for cleaner display
    data.rename(columns=display_name, inplace=True)

    print(f"[OK] Loaded {data.shape[1]} tickers: {list(data.columns)}")
    return data


def get_tickers_for_sectors(sectors: list[str]) -> list[str]:
    """Return flat list of NSE ticker symbols for selected sectors."""
    tickers = []
    for sector in sectors:
        tickers.extend(TICKERS.get(sector, []))
    return tickers


def get_stock_info(ticker: str) -> dict:
    """Fetch fundamental data for a single NSE ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except Exception:
        return {"name": ticker, "sector": "N/A", "pe_ratio": "N/A",
                "market_cap_cr": "N/A", "52w_high": "N/A",
                "52w_low": "N/A", "dividend_yield": "N/A"}

    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "market_cap_cr": round(info.get("marketCap", 0) / 1e7, 2),  # in Crores
        "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        "dividend_yield": info.get("dividendYield", "N/A"),
    }