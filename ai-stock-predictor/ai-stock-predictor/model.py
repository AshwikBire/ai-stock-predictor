"""
model.py
--------
Handles training a machine learning model (Linear Regression) on historical
stock closing prices, and predicting the next N days into the future.

The approach: treat each trading day as an integer index (0, 1, 2, ...) and
fit a Linear Regression model mapping day-index -> closing price. This is a
simple, interpretable baseline model well suited for demonstrating an
end-to-end ML pipeline (not intended as a real trading strategy).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class PredictionResult:
    """Container for everything the UI needs after training + predicting."""
    future_dates: pd.DatetimeIndex
    future_prices: np.ndarray
    model_r2: float
    model_mae: float
    trend: str  # "Uptrend" or "Downtrend"
    trend_change_pct: float


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a DataFrame with 'Date' and 'Close' columns into model-ready
    feature (X) and target (y) arrays.

    X = day index (0, 1, 2, ...) reshaped for sklearn
    y = closing price
    """
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values
    return X, y


def train_and_predict(df: pd.DataFrame, days_to_predict: int = 30) -> PredictionResult:
    """
    Train a Linear Regression model on historical closing prices and
    forecast the next `days_to_predict` trading days.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Date' and 'Close' columns, sorted chronologically.
    days_to_predict : int
        Number of future days to forecast (default: 30).

    Returns
    -------
    PredictionResult
        Dataclass containing future dates, predicted prices, basic model
        evaluation metrics, and a simple trend insight.
    """
    X, y = _build_features(df)

    # Split data to get a rough sense of model performance (evaluation only;
    # the final model used for forecasting is refit on ALL available data
    # so the forecast benefits from the most recent information).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # shuffle=False preserves time order
    )

    eval_model = LinearRegression()
    eval_model.fit(X_train, y_train)
    y_pred_test = eval_model.predict(X_test)

    model_r2 = r2_score(y_test, y_pred_test)
    model_mae = mean_absolute_error(y_test, y_pred_test)

    # Refit on the full dataset for the actual forward-looking prediction
    final_model = LinearRegression()
    final_model.fit(X, y)

    last_index = len(df) - 1
    future_indices = np.arange(last_index + 1, last_index + 1 + days_to_predict).reshape(-1, 1)
    future_prices = final_model.predict(future_indices)

    # Generate future business-day dates following the last known date
    last_date = df["Date"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)

    # Simple trend insight: compare first vs last predicted price
    trend_change_pct = ((future_prices[-1] - future_prices[0]) / future_prices[0]) * 100
    trend = "Uptrend" if trend_change_pct >= 0 else "Downtrend"

    return PredictionResult(
        future_dates=future_dates,
        future_prices=future_prices,
        model_r2=model_r2,
        model_mae=model_mae,
        trend=trend,
        trend_change_pct=trend_change_pct,
    )
