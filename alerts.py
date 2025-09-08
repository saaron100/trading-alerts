# alerts.py
# Full scanner: RSI + EMA signals -> pick option (ATM) -> choose expiry based on strength -> Telegram alert
# State is saved in state.json so you do not get spam; workflow will commit state.json back to repo.

import os
import json
import math
import requests
from datetime import datetime, date, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# ----------------- CONFIG -----------------
TICKERS = ["SPY","AAPL","CVX","AMZN","QQQ","GLD","SLV","PLTR",
           "USO","NFLX","TNA","XOM","NVDA","BAC","TSLA","META"]

STATE_FILE = "state.json"   # persisted between runs (workflow will commit it)
RSI_PERIOD = 14
EMA_SHORT = 20
EMA_LONG = 50

# Profit / stop guidance (not automatically traded)
TAKE_PROFIT_PCT = 0.30  # +30% on option mid
STOP_LOSS_PCT   = -0.20 # -20% on option mid
# ------------------------------------------

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")

# ---------- helpers ----------
def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text}
        r = requests.post(url, json=payload, timeout=15)
        print("telegram:", r.status_code, r.text[:200])
    except Exception as e:
        print("Telegram send error:", e)

def compute_rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Wilder smoothing (ewm with alpha = 1/period)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_ema(series: pd.Series, period=20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def parse_expirations(exp_list):
    exps = []
    for s in exp_list:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d").date()
            exps.append((dt, s))
        except Exception:
            continue
    exps.sort()
    return exps

def choose_expiry_by_score(ticker_obj, score):
    # score: 1 weak, 2 medium, 3 strong
    exps = ticker_obj.options
    parsed = parse_expirations(exps)
    if not parsed:
        return None
    today = date.today()

    # Score 3 -> prefer within 0-7 days (this week / this Friday)
    if score == 3:
        candidates = [s for (d,s) in parsed if 0 <= (d - today).days <= 7]
        if candidates:
            return candidates[0]

    # Score 2 -> prefer 4-10 days
    if score == 2:
        candidates = [s for (d,s) in parsed if 4 <= (d - today).days <= 10]
        if candidates:
            return candidates[0]

    # Score 1 -> prefer 11-45 days
    if score == 1:
        candidates = [s for (d,s) in parsed if 11 <= (d - today).days <= 45]
        if candidates:
            return candidates[0]

    # fallback: nearest future expiry
    for (d,s) in parsed:
        if (d - today).days >= 0:
            return s
    return parsed[-1][1]

def pick_atm_contract(ticker, expiry, is_call, spot):
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        table = chain.calls if is_call else chain.puts
        if table.empty:
            return None
        # choose strike closest to spot
        table = table.copy()
        table["dist"] = (table["strike"] - spot).abs()
        row = table.sort_values("dist").iloc[0]
        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        last = float(row.get("lastPrice") or 0)
        if bid > 0 and ask > 0:
            mid = round((bid + ask)/2, 2)
        else:
            mid = round(last if last > 0 else max(bid, ask), 2)
        return {
            "contract": row.get("contractSymbol", ""),
            "strike": float(row["strike"]),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "inTheMoney": bool(row.get("inTheMoney", False))
        }
    except Exception as e:
        print("pick_atm_contract error:", e)
        return None

def score_signal(signal_type, rsi_val, close, ema_short, ema_long, candle_green):
    # base 1 star, add points for strong evidence; cap 3 stars
    score = 1
    if signal_type == "CALL":
        if rsi_val < 25: score += 1
        if ema_short > ema_long: score += 1
        if candle_green: score += 1
    elif signal_type == "PUT":
        if rsi_val > 75: score += 1
        if ema_short < ema_long: score += 1
        if not candle_green: score += 1
    return min(score, 3)

def expiry_label(expiry_str):
    try:
        e = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = date.today()
        days = (e - today).days
        if days <= 7:
            return f"{expiry_str} (this week)"
        if days <= 14:
            return f"{expiry_str} (next week)"
        return expiry_str
    except Exception:
        return expiry_str

# ---------- state persistence ----------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_signals": {}, "positions": {}}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"last_signals": {}, "positions": {}}

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print("save_state error:", e)

# ---------- analysis ----------
def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="30m", progress=False)
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < RSI_PERIOD + 2:
            return None

        rsi_series = compute_rsi(close, RSI_PERIOD)
        ema_short_s = compute_ema(close, EMA_SHORT)
        ema_long_s  = compute_ema(close, EMA_LONG)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        rsi_val = float(rsi_series.iloc[-1])
        close_price = float(last["Close"])
        ema_s = float(ema_short_s.iloc[-1])
        ema_l = float(ema_long_s.iloc[-1])
        candle_green = float(last["Close"]) > float(last["Open"])

        # Basic directional rules
        signal_type = None
        if rsi_val < 30 and close_price > ema_s and ema_s > ema_l:
            signal_type = "CALL"
        elif rsi_val > 70 and close_price < ema_s and ema_s < ema_l:
            signal_type = "PUT"

        if not signal_type:
            return None

        # Score
        score = score_signal(signal_type, rsi_val, close_price, ema_s, ema_l, candle_green)

        # Determine expiry based on score
        tkr = yf.Ticker(ticker)
        expiry = choose_expiry_by_score(tkr, score)
        if not expiry:
            return None

        # pick ATM contract
        is_call = (signal_type == "CALL")
        opt = pick_atm_contract(ticker, expiry, is_call, close_price)
        if not opt:
            return None

        # Format instruction for Schwab: Buy to Open 1 TICKER 260C exp 09/12/25
        exp_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
        exp_short = exp_dt.strftime("%m/%d/%y")
        cp_letter = "C" if is_call else "P"
        instruction = f"Buy to Open 1 {ticker} {int(opt['strike'])}{cp_letter} exp {exp_short}"
        reason = (f"Strong bullish setup (RSI {rsi_val:.1f} < 30, EMA{EMA_SHORT} > EMA{EMA_LONG})"
                  if is_call else
                  f"Strong bearish setup (RSI {rsi_val:.1f} > 70, EMA{EMA_SHORT} < EMA{EMA_LONG})")

        # human readable expiry label
        exp_lab = expiry_label(expiry)

        # Build message
        msg = (
            f"📊 {ticker} SIGNAL\n"
            f"Type: {signal_type}\n"
            f"Expiration: {exp_lab}\n"
            f"Strike: {int(opt['strike'])}\n"
            f"Entry (spot): {close_price:.2f}\n"
            f"Option mid≈ {opt['mid']:.2f} (bid {opt['bid']:.2f} / ask {opt['ask']:.2f})\n"
            f"Contract: {opt['contract']}\n"
            f"Instruction: {instruction}\n"
            f"Reason: {reason}\n"
            f"Confidence: {score}⭐  | TP {int(TAKE_PROFIT_PCT*100)}% / SL {int(abs(STOP_LOSS_PCT)*100)}%\n"
            f"Note: adjust qty to your risk. Use Limit orders on Schwab."
        )

        signature = f"{signal_type}|{expiry}|{int(opt['strike'])}"
        return {"ticker": ticker, "message": msg, "signature": signature}
    except Exception as e:
        print("analyze_ticker error for", ticker, e)
        return None

# ---------- main ----------
def main():
    state = load_state()
    last_signals = state.get("last_signals", {})

    new_signals = []
    for t in TICKERS:
        res = analyze_ticker(t)
        if not res:
            continue
        sig = res["signature"]
        if last_signals.get(t) != sig:
            # new or changed signal -> alert + track
            send_telegram(res["message"])
            last_signals[t] = sig
            new_signals.append(t)
        else:
            print(f"{t}: same signal as before, skipping.")

    # save state back
    state["last_signals"] = last_signals
    save_state(state)

    print("Done. Alerts sent for:", new_signals)

if __name__ == "__main__":
    main()

