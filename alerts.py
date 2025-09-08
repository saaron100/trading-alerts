import requests
import yfinance as yf
import ta
import os

# Telegram bot setup
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        r = requests.post(url, json=payload)
        r.raise_for_status()
    except Exception as e:
        print("Error sending message:", e)

# Tick list
TICKERS = ["SPY","AAPL","CVX","AMZN","QQQ","GLD","SLV","PLTR",
           "USO","NFLX","TNA","XOM","NVDA","BAC","TSLA","META"]

def analyze_ticker(ticker):
    df = yf.download(ticker, period="6mo", interval="15m")
    if df.empty:
        return None

    # Indicators
    df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    last = df.iloc[-1]

    signal = None
    score = 0

    # Conditions
    if last["EMA20"] > last["EMA50"] and last["RSI"] > 55:
        signal = "CALL"
        score += 1
    if last["EMA20"] < last["EMA50"] and last["RSI"] < 45:
        signal = "PUT"
        score += 1

    if signal:
        return f"{ticker}: {signal} ⭐ Score: {score}"
    return None

def main():
    messages = []
    for ticker in TICKERS:
        result = analyze_ticker(ticker)
        if result:
            messages.append(result)

    if messages:
        send_message("📊 New Signals:\n" + "\n".join(messages))
    else:
        print("No signals this run.")

# ---------- TEST LINE ----------
if __name__ == "__main__":
    main()
    send_message("✅ TEST MESSAGE: Signals system is working with ⭐ scoring")
