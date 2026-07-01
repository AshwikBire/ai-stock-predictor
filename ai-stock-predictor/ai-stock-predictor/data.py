"""
data.py
--------
Handles fetching and cleaning stock market data using yfinance.
Keeping this logic separate from app.py and model.py makes the
codebase modular and easy to test/extend (e.g., swapping data sources later).
"""

import pandas as pd
import yfinance as yf


class InvalidTickerError(Exception):
    """Raised when a stock ticker does not return any usable data."""
    pass


def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker symbol.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL", "TSLA").
    period : str
        How far back to fetch data. Default is "1y" (1 year),
        which satisfies the "at least 1 year of data" requirement.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least 'Date' and 'Close' columns, sorted by date.

    Raises
    ------
    InvalidTickerError
        If the ticker is invalid, empty, or no data could be retrieved
        (e.g., due to a typo or an API/network failure).
    """
    ticker = (ticker or "").strip().upper()

    if not ticker:
        raise InvalidTickerError("Ticker symbol cannot be empty.")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
    except Exception as exc:
        # Catches network errors, API errors, etc.
        raise InvalidTickerError(
            f"Failed to fetch data for '{ticker}'. "
            f"Please check your internet connection or try again later. "
            f"Details: {exc}"
        ) from exc

    if df is None or df.empty:
        raise InvalidTickerError(
            f"No data found for ticker '{ticker}'. "
            f"It may be an invalid or delisted symbol."
        )

    # yfinance returns the date as the index; bring it into a column
    df = df.reset_index()

    # Keep only the columns we need, and make sure they exist
    required_cols = {"Date", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise InvalidTickerError(
            f"Unexpected data format for ticker '{ticker}'. Missing columns: {missing}"
        )

    df = df[["Date", "Close"]].copy()

    # Ensure Date is a proper datetime type and data is sorted chronologically
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)

    # Drop any rows with missing close prices (can happen on holidays/gaps)
    df = df.dropna(subset=["Close"])

    if len(df) < 30:
        raise InvalidTickerError(
            f"Not enough historical data for '{ticker}' to train a reliable model "
            f"(found only {len(df)} rows)."
        )

    return df


def get_company_name(ticker: str) -> str:
    """
    Best-effort fetch of the company's display name for nicer chart titles.
    Falls back to the raw ticker symbol if the lookup fails for any reason.
    """
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker.upper()
    except Exception:
        return ticker.upper()
