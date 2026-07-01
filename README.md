# 📈 AI Stock Market Predictor (ML + Visualization)

A complete, modular Streamlit application that fetches real stock market
data, trains a Linear Regression model on historical closing prices, and
forecasts near-term future prices — with interactive charts and a
predicted-values table.

---

## ✨ Features

- **Real stock data** — fetches at least 1 year of historical data via `yfinance`
- **Machine learning** — trains a Linear Regression model on closing prices
- **30-day forecast** (adjustable from 7–60 days) of future stock prices
- **Visualization** — Matplotlib chart showing historical prices and the
  predicted forecast line on the same plot
- **Interactive UI** — enter any ticker symbol, choose historical range and
  forecast horizon, click a button to run the prediction
- **Trend insight** — automatically labels the forecast as an Uptrend 📈 or
  Downtrend 📉, with percentage change
- **Model performance metrics** — R² score and Mean Absolute Error (MAE) on
  a held-out test split
- **Robust error handling** — invalid tickers, empty data, and API/network
  failures are all caught and shown as clear messages instead of crashing
- **Loading indicators** — spinners while data is fetched and the model trains

---

## 🧩 Tech Stack

| Purpose            | Library        |
|--------------------|----------------|
| UI                 | Streamlit      |
| Stock data         | yfinance       |
| Data handling      | Pandas, NumPy  |
| ML model           | Scikit-learn   |
| Charts             | Matplotlib     |

---

## 🗂️ Project Structure

```
ai-stock-predictor/
│
├── app.py              # Streamlit UI — entry point of the app
├── data.py              # Fetches & cleans stock data via yfinance
├── model.py             # Trains Linear Regression model & generates forecasts
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

**Why this structure?**
- `data.py` isolates all data-fetching logic, so the data source could later
  be swapped (e.g., a different API) without touching the model or UI.
- `model.py` isolates the ML logic, so the model (e.g., upgrading to
  Polynomial Regression or an LSTM) could be swapped without touching the UI.
- `app.py` only handles UI rendering and orchestration — it calls into
  `data.py` and `model.py` rather than doing data/ML work itself.

---

## ⚙️ Installation

### 1. Clone or download this project

```bash
git clone <your-repo-url>
cd ai-stock-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

This will open the app in your default browser at `http://localhost:8501`.

---

## 🧪 Example Usage

1. In the sidebar, enter a stock ticker, e.g. `AAPL`
2. Choose a historical data range (`1y`, `2y`, or `5y`)
3. Choose how many days to forecast (default: 30)
4. Click **🔮 Predict Stock Price**

**Output:**
- Key metrics: latest close price, predicted price, trend (up/down), R² score
- A chart showing historical closing prices and the predicted forecast line
- A table listing the predicted closing price for each of the next N days
- An expandable section with model performance details (R², MAE)

Try other tickers like `TSLA`, `MSFT`, `GOOGL`, or `AMZN`. If you enter an
invalid or delisted ticker (e.g., `XYZ123`), the app will show a clear error
message instead of crashing.

---

## ⚠️ Disclaimer

This project is built for **educational and demonstration purposes only**.
It uses a simple Linear Regression model on historical closing prices alone —
it does **not** account for news, earnings reports, macroeconomic events, or
other real-world factors that drive actual stock prices. **Do not use this
tool for real financial or investment decisions.**

---

## 🛠️ Possible Extensions

- Swap Linear Regression for a more advanced model (Random Forest, LSTM, Prophet)
- Add additional features (volume, moving averages, RSI, etc.)
- Add a comparison view for multiple tickers at once
- Cache fetched data with `st.cache_data` to reduce repeated API calls
- Deploy to Streamlit Community Cloud for free public hosting

---

## 📄 License

MIT — free to use and modify for learning or portfolio purposes.
