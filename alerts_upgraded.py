import os
import json
import requests
import yfinance as yf
import pandas as pd

# === Load Environment Variables ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === Config ===
TICKERS = ["AAPL", "TSLA", "QQQ", "NVDA", "NFLX"]  # You can add more here
DATA_FILE = "state.json"

# === Telegram Function ===
def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Telegram error:", e)

# === Helper Indicators ===
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_ema(series: pd.Series, period: int = 20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

# === Strategy Checks ===
def bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    last = df.iloc[-1]
    return (
        prev["Close"] < prev["Open"]
        and last["Close"] > last["Open"]
        and last["Close"] > prev["Open"]
        and last["Open"] < prev["Close"]
    )

def hammer(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    body = abs(last["Close"] - last["Open"])
    candle_range = last["High"] - last["Low"]
    lower_shadow = min(last["Open"], last["Close"]) - last["Low"]
    return lower_shadow > 2 * body and body / candle_range < 0.3

def ema_cross(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return None
    prev = df.iloc[-2]
    last = df.iloc[-1]
    if last["EMA20"] > last["EMA50"] and prev["EMA20"] <= prev["EMA50"]:
        return "bullish"
    elif last["EMA20"] < last["EMA50"] and prev["EMA20"] >= prev["EMA50"]:
        return "bearish"
    return None

def breakout(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 1:
        return False
    close_series = df["Close"]
    return float(close_series.iloc[-1]) > float(close_series.iloc[-lookback-1])

# === Load / Save State ===
def load_state():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(DATA_FILE, "w") as f:
        json.dump(state, f)

# === Main Analysis ===
def analyze_ticker(ticker: str):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        df.dropna(inplace=True)
        df["RSI"] = compute_rsi(df["Close"])
        df["EMA20"] = compute_ema(df["Close"], 20)
        df["EMA50"] = compute_ema(df["Close"], 50)

        signals = []

        # RSI conditions
        if df["RSI"].iloc[-1] < 30:
            signals.append("RSI oversold (possible CALL)")
        elif df["RSI"].iloc[-1] > 70:
            signals.append("RSI overbought (possible PUT)")

        # EMA cross
        ema_sig = ema_cross(df)
        if ema_sig == "bullish":
            signals.append("Bullish EMA cross (20 > 50)")
        elif ema_sig == "bearish":
            signals.append("Bearish EMA cross (20 < 50)")

        # Candle patterns
        if bullish_engulfing(df):
            signals.append("Bullish engulfing pattern")
        if hammer(df):
            signals.append("Hammer pattern (possible reversal)")

        # Breakout
        if breakout(df):
            signals.append("Breakout above resistance")

        if signals:
            last = df.iloc[-1]
            message = (
                f"⭐ ALERT for {ticker} ⭐\n"
                f"Price: {float(last['Close']):.2f}\n"
                f"RSI: {float(df['RSI'].iloc[-1]):.2f}\n"
                f"Signals: \n- " + "\n- ".join(signals)
            )
            return message
    except Exception as e:
        print(f"analyze_ticker error for {ticker}", e)
    return None

# === Runner ===
def run():
    state = load_state()
    alerts = []

    for ticker in TICKERS:
        msg = analyze_ticker(ticker)
        if msg and state.get(ticker) != msg:
            alerts.append(msg)
            state[ticker] = msg

    if alerts:
        for alert in alerts:
            send_telegram_message(alert)
        save_state(state)
        print("Alerts sent for:", [a.split()[2] for a in alerts])
    else:
        print("No new alerts.")

if __name__ == "__main__":
    run()
