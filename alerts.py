import os, requests, yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TICKERS = ["SPY","AAPL","CVX","AMZN","QQQ","GLD","SLV","PLTR","USO",
           "NFLX","TNA","XOM","NVDA","BAC","TSLA","META"]

STATE_FILE = "last_signals.txt"

def send_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.get(url, params={"chat_id": CHAT_ID, "text": text})

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def next_friday():
    today = datetime.today()
    days_ahead = 4 - today.weekday()  # 4 = Friday
    if days_ahead <= 0:
        days_ahead += 7
    return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

def round_strike(price):
    if price < 50:
        return round(price)  # $1 steps
    elif price < 200:
        return int(round(price / 2.5) * 2.5)  # $2.5 steps
    else:
        return int(round(price / 5) * 5)  # $5 steps

def analyze_ticker(ticker):
    df = yf.download(ticker, period="3mo", interval="30m")
    if df.empty: return None
    df["RSI"] = rsi(df["Close"])
    df["EMA20"] = ema(df["Close"], 20)

    last = df.iloc[-1]
    price = last["Close"]
    expiry = next_friday()
    strike = round_strike(price)

    if last["RSI"] < 30 and last["Close"] > last["EMA20"]:
        return f"🚨 {ticker}: CALL {strike}c exp {expiry} | Entry ~ {price:.2f}"
    elif last["RSI"] > 70 and last["Close"] < last["EMA20"]:
        return f"🚨 {ticker}: PUT {strike}p exp {expiry} | Entry ~ {price:.2f}"
    return None

def load_state():
    if not os.path.exists(STATE_FILE):
        return set()
    with open(STATE_FILE, "r") as f:
        return set(line.strip() for line in f)

def save_state(signals):
    with open(STATE_FILE, "w") as f:
        for s in signals:
            f.write(s + "\n")

def main():
    last_signals = load_state()
    new_signals = set()

    for t in TICKERS:
        sig = analyze_ticker(t)
        if sig and sig not in last_signals:
            send_message(f"{sig}")
            new_signals.add(sig)

    if new_signals:
        save_state(last_signals.union(new_signals))

if __name__ == "__main__":
    main()
