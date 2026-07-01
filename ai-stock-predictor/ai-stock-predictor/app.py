"""
app.py
------
Streamlit UI for the AI Stock Market Predictor.

Run with:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data import InvalidTickerError, fetch_stock_data, get_company_name
from model import train_and_predict

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Market Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI Stock Market Predictor")
st.markdown(
    "Predict near-term stock price trends using historical data and a "
    "Linear Regression model. Enter a ticker symbol below to get started."
)

# ----------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------
st.sidebar.header("Settings")

ticker_input = st.sidebar.text_input(
    "Stock Ticker Symbol",
    value="AAPL",
    help="Example tickers: AAPL, TSLA, MSFT, GOOGL, AMZN",
).strip().upper()

period_option = st.sidebar.selectbox(
    "Historical Data Range",
    options=["1y", "2y", "5y"],
    index=0,
    help="How much historical data to use for training the model.",
)

days_to_predict = st.sidebar.slider(
    "Days to Predict",
    min_value=7,
    max_value=60,
    value=30,
    step=1,
)

predict_button = st.sidebar.button("🔮 Predict Stock Price", type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ This tool is for educational purposes only and is **not** "
    "financial advice. Stock markets are influenced by many factors "
    "a simple Linear Regression model cannot capture."
)

# ----------------------------------------------------------------------
# Main logic — triggered when the user clicks the Predict button
# ----------------------------------------------------------------------
if predict_button:
    if not ticker_input:
        st.error("Please enter a valid stock ticker symbol.")
    else:
        try:
            # --- Step 1: Fetch data, with a loading indicator ---
            with st.spinner(f"Fetching historical data for {ticker_input}..."):
                df = fetch_stock_data(ticker_input, period=period_option)
                company_name = get_company_name(ticker_input)

            # --- Step 2: Train model and predict, with a loading indicator ---
            with st.spinner("Training model and generating predictions..."):
                result = train_and_predict(df, days_to_predict=days_to_predict)

            st.success(f"Prediction complete for {company_name} ({ticker_input})")

            # --- Step 3: Key metrics row ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Latest Close Price", f"${df['Close'].iloc[-1]:.2f}")
            col2.metric(
                f"Predicted Price ({days_to_predict}d)",
                f"${result.future_prices[-1]:.2f}",
                f"{result.trend_change_pct:+.2f}%",
            )
            col3.metric("Trend", f"{'📈' if result.trend == 'Uptrend' else '📉'} {result.trend}")
            col4.metric("Model R² Score", f"{result.model_r2:.3f}")

            # --- Step 4: Chart — historical + predicted prices ---
            st.subheader("Price History & Forecast")

            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(df["Date"], df["Close"], label="Historical Close Price", color="#1f77b4")
            ax.plot(
                result.future_dates,
                result.future_prices,
                label=f"Predicted Price (next {days_to_predict} days)",
                color="#ff7f0e",
                linestyle="--",
            )

            # Mark the transition point between historical and predicted data
            ax.axvline(df["Date"].iloc[-1], color="gray", linestyle=":", alpha=0.7)

            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.set_title(f"{company_name} ({ticker_input}) — Historical Prices & {days_to_predict}-Day Forecast")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()

            st.pyplot(fig)

            # --- Step 5: Predicted values table ---
            st.subheader(f"Predicted Closing Prices — Next {days_to_predict} Days")

            forecast_df = pd.DataFrame(
                {
                    "Date": result.future_dates.strftime("%Y-%m-%d"),
                    "Predicted Close ($)": result.future_prices.round(2),
                }
            )
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            # --- Step 6: Model performance note ---
            with st.expander("ℹ️ Model Performance Details"):
                st.write(
                    f"""
                    - **Model type:** Linear Regression
                    - **Training period:** {period_option} of historical data ({len(df)} trading days)
                    - **R² score (test split):** {result.model_r2:.4f}
                    - **Mean Absolute Error (test split):** ${result.model_mae:.2f}

                    A higher R² (closer to 1.0) indicates the model fits the
                    historical trend more closely. Note that stock prices are
                    highly volatile and influenced by many real-world factors
                    this simple model does not account for (news, earnings,
                    macroeconomic events, etc.). Use predictions as a rough
                    trend indicator only, not as financial advice.
                    """
                )

        except InvalidTickerError as e:
            # Handles invalid tickers and data-fetch failures gracefully
            st.error(f"❌ {e}")
        except Exception as e:
            # Catch-all for any unexpected errors (e.g., network/API issues)
            st.error(f"❌ An unexpected error occurred: {e}")
            st.info("Please check your internet connection and try again.")
else:
    st.info("👈 Enter a stock ticker in the sidebar and click **Predict Stock Price** to begin.")
