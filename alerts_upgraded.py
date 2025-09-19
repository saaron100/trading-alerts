#!/usr/bin/env python3
# alerts_upgraded.py
# Cloud-ready trading alerts for GitHub Actions (run once per schedule).
# - Sends Telegram alerts
# - Detects RSI, EMA, Hammer/Hanging Man/Doji, Bollinger breakout, volume spike
# - Picks ATM option (yfinance), builds recommendation message
# - Processes simple Telegram commands via getUpdates
#
# Requirements (put in requirements.txt):
# yfinance
# pandas
# numpy
# requests
#
# Environment variables (set these in GitHub Secrets):
# TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
#
# Usage:
# - Run once per schedule (GitHub Actions). It will load/save state.json in repo workspace.
# - To persist state between runs, configure your GitHub Actions workflow to commit state.json back (optional).
#
# Keep it simple and test with DEBUG_FORCE_TEST=True first.

import os
import json
import time
import traceback
from datetime import datetime, date, timedelta

import requests
import yfinance as yf
import pandas as pd
import numpy as np

# ---------------- CONFIG (tweak these) ----------------
TICKERS = ["SPY","AAPL","CVX","AMZN","QQQ","GLD","SLV","PLTR",
           "USO","NFLX","TNA","XOM","NVDA","BAC","TSLA","META"]

STATE_FILE = "state.json"

# Indicator settings
RSI_PERIOD = 14
EMA_SHORT = 20
EMA_LONG = 50

# Looser RSI thresholds (we agreed to test these)
RSI_BUY_THRESHOLD = 40
RSI_SELL_THRESHOLD = 60

# Option & trade helpers
DEFAULT_QTY = 1
TAKE_PROFIT_PCT = 0.50   # notify when +50% (0.50 => 50%)
STOP_LOSS_PCT   = -0.30  # notify when -30%

# Data fetch settings
INTRADAY_TRY_PERIODS = ["7d", "30d", "60d"]
INTRADAY_INTERVAL = "15m"
FALLBACK_DAILY_PERIOD = "3mo"
FALLBACK_DAILY_INTERVAL = "1d"

# Debug/test
DEBUG_FORCE_TEST = False   # set True to send one fake alert (turn off after)
DEBUG_TEST_TICKER = "TSLA"

# ---------------- Environment (Telegram) ----------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ---------------- Helpers ----------------
def send_telegram(text):
    """Send text message to configured Telegram chat."""
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=15)
        try:
            j = r.json()
        except Exception:
            j = None
        print("Telegram send:", r.status_code, (j if j else str(r.text))[:200])
        return j
    except Exception as e:
        print("send_telegram error:", e)
        return None

def get_updates(offset=None, timeout=5):
    """Poll Telegram getUpdates (used when running scheduled jobs)."""
    if not BOT_TOKEN:
        return []
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        params = {"timeout": timeout}
        if offset:
            params["offset"] = offset
        r = requests.get(url, params=params, timeout=timeout+5)
        j = r.json()
        if not j.get("ok"):
            print("getUpdates not ok:", j)
            return []
        return j.get("result", [])
    except Exception as e:
        print("get_updates error:", e)
        return []

# ---------------- Data fetch (robust) ----------------
def get_data(ticker):
    # Try intraday windows first, fallback to daily
    for p in INTRADAY_TRY_PERIODS:
        try:
            df = yf.download(ticker, period=p, interval=INTRADAY_INTERVAL, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"{ticker} intraday attempt period={p} failed: {e}")
            continue
    try:
        df = yf.download(ticker, period=FALLBACK_DAILY_PERIOD, interval=FALLBACK_DAILY_INTERVAL, progress=False, auto_adjust=True)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as e:
        print("get_data fallback failed for", ticker, e)
        return pd.DataFrame()

# ---------------- Technical indicators ----------------
def compute_rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_ema(series: pd.Series, period=20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

# ---------------- Candlestick patterns ----------------
def detect_candlestick_pattern(df):
    """
    Detect Hammer, Hanging Man, Doji on the last candle.
    Returns 'HAMMER','HANGING_MAN','DOJI' or None.
    """
    if len(df) < 1:
        return None
    last = df.iloc[-1]
    o = float(last["Open"])
    h = float(last["High"])
    l = float(last["Low"])
    c = float(last["Close"])

    body = abs(c - o)
    rng = h - l
    if rng == 0:
        return None
    upper = h - max(o, c)
    lower = min(o, c) - l

    # Hammer/hanging-man shape: small body, long lower wick
    if body <= rng * 0.35 and lower >= body * 2 and upper <= body:
        # need context to decide hammer vs hanging man; return generic and let analyze decide
        return "HAMMER"

    # Doji: very small body
    if body <= rng * 0.1:
        return "DOJI"

    return None

def recent_trend_up(close_series, lookback=5):
    try:
        if len(close_series) < lookback + 1:
            return False
        return float(close_series.iloc[-1]) > float(close_series.iloc[-lookback-1])
    except Exception:
        return False

# ---------------- Bollinger & Volume helpers ----------------
def bollinger_breakout(df, period=20, nbdev=2):
    close = df["Close"]
    if len(close) < period + 1:
        return None
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + nbdev * std
    lower = ma - nbdev * std
    last = float(close.iloc[-1])
    if last > upper.iloc[-1]:
        return "upper"
    if last < lower.iloc[-1]:
        return "lower"
    return None

def is_volume_spike(df, lookback=20, spike_factor=1.5):
    if "Volume" not in df.columns or len(df["Volume"]) < lookback + 1:
        return False
    mean_vol = df["Volume"].iloc[-(lookback+1):-1].mean()
    return df["Volume"].iloc[-1] >= mean_vol * spike_factor

# ---------------- Option helpers (yfinance) ----------------
def choose_expiry_by_score(ticker_obj, score):
    exps = ticker_obj.options
    parsed = []
    for s in exps:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d").date()
            parsed.append((dt, s))
        except Exception:
            continue
    parsed.sort()
    if not parsed:
        return None
    today = date.today()
    ranges = {
        5: (0, 7),
        4: (0, 14),
        3: (7, 30),
        2: (14, 60),
        1: (30, 120)
    }
    low, high = ranges.get(max(1,min(5,score)), (7,30))
    candidates = [s for (d,s) in parsed if low <= (d - today).days <= max(1, high)]
    if candidates:
        return candidates[0]
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
        table = table.copy()
        table["dist"] = (table["strike"] - spot).abs()
        row = table.sort_values("dist").iloc[0]
        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        lastp = float(row.get("lastPrice") or 0)
        if bid>0 and ask>0:
            mid = round((bid+ask)/2, 2)
        else:
            mid = round(lastp if lastp>0 else max(bid,ask), 2)
        return {
            "contract": row.get("contractSymbol",""),
            "strike": float(row["strike"]),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "inTheMoney": bool(row.get("inTheMoney", False))
        }
    except Exception as e:
        print("pick_atm_contract error:", e)
        return None

def current_option_mid_by_contract(ticker, contract_symbol, expiry, opt_type, strike):
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        tab = chain.calls if opt_type == "CALL" else chain.puts
        row = tab.loc[tab["contractSymbol"] == contract_symbol]
        if row.empty:
            row = tab.loc[abs(tab["strike"] - strike) < 1e-6]
            if row.empty:
                return None
        bid = float(row.iloc[0].get("bid") or 0)
        ask = float(row.iloc[0].get("ask") or 0)
        lastp = float(row.iloc[0].get("lastPrice") or 0)
        if bid>0 and ask>0:
            return round((bid+ask)/2, 2)
        return round(lastp if lastp>0 else max(bid,ask), 2)
    except Exception as e:
        print("current_option_mid error:", e)
        return None

# ---------------- Scoring function (1-5 stars) ----------------
def score_signals(df, signal_type, rsi_val, ema_s_val, ema_l_val):
    score = 1
    reasons = []

    # Price confirmation with EMAs
    if signal_type == "CALL":
        if ema_s_val > ema_l_val:
            score += 1
            reasons.append("EMA trend")
        if rsi_val is not None and rsi_val < RSI_BUY_THRESHOLD:
            score += 1
            reasons.append(f"RSI{int(rsi_val)}")
        if bollinger_breakout(df) == "upper":
            score += 1
            reasons.append("Bollinger upper")
        if is_volume_spike(df):
            score += 1
            reasons.append("Volume spike")
    else:  # PUT
        if ema_s_val < ema_l_val:
            score += 1
            reasons.append("EMA trend")
        if rsi_val is not None and rsi_val > RSI_SELL_THRESHOLD:
            score += 1
            reasons.append(f"RSI{int(rsi_val)}")
        if bollinger_breakout(df) == "lower":
            score += 1
            reasons.append("Bollinger lower")
        if is_volume_spike(df):
            score += 1
            reasons.append("Volume spike")

    score = max(1, min(5, score))
    return int(score), reasons

# ---------------- State persistence ----------------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "last_update_id": None,
            "last_signals": {},
            "last_suggestions": {},
            "positions": {},
            "alerts_enabled": True
        }
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print("load_state error:", e)
        return {
            "last_update_id": None,
            "last_signals": {},
            "last_suggestions": {},
            "positions": {},
            "alerts_enabled": True
        }

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print("save_state error:", e)

# ---------------- Analyze ticker (main logic) ----------------
def analyze_ticker(ticker):
    """
    Returns dict with keys:
      - ticker
      - message
      - signature
      - suggestion (contract info)
    Or None if no signal.
    """
    try:
        df = get_data(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        close_series = df["Close"].dropna()
        if len(close_series) < max(RSI_PERIOD, EMA_LONG) + 2:
            return None

        # Indicators
        rsi_series = compute_rsi(close_series, RSI_PERIOD)
        rsi_val = float(rsi_series.iloc[-1]) if len(rsi_series)>0 else None
        ema_s = compute_ema(close_series, EMA_SHORT)
        ema_l = compute_ema(close_series, EMA_LONG)
        ema_s_val = float(ema_s.iloc[-1])
        ema_l_val = float(ema_l.iloc[-1])

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close_price = float(last["Close"])

        # Candle pattern & context
        pattern = detect_candlestick_pattern(df)
        uptrend = recent_trend_up(close_series, lookback=5)

        # Decide direction with looser RSI thresholds + candle confirmations
        signal_type = None
        # CALL logic
        if (rsi_val is not None and rsi_val < RSI_BUY_THRESHOLD and ema_s_val > ema_l_val and close_price > ema_s_val) or (pattern == "HAMMER" and not uptrend):
            signal_type = "CALL"
        # PUT logic
        elif (rsi_val is not None and rsi_val > RSI_SELL_THRESHOLD and ema_s_val < ema_l_val and close_price < ema_s_val) or (pattern == "HAMMER" and uptrend):
            # treat hammer after uptrend as hanging/man -> bearish
            signal_type = "PUT"

        if not signal_type:
            return None

        # Score & reasons
        score, reasons = score_signals(df, signal_type, rsi_val, ema_s_val, ema_l_val)

        # Option expiry and contract
        tkr = yf.Ticker(ticker)
        expiry = choose_expiry_by_score(tkr, score)
        if not expiry:
            return None

        is_call = (signal_type == "CALL")
        opt = pick_atm_contract(ticker, expiry, is_call, close_price)
        if not opt:
            return None

        est_cost = round(opt["mid"] * 100, 2) if opt.get("mid") else None
        days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
        if days_to_expiry <= 3:
            horizon = "Daytrade (very short)"
        elif days_to_expiry <= 14:
            horizon = "Short-term / swing"
        else:
            horizon = "Medium/long-term"

        stars = "⭐" * score

        message = (
            f"📊 {ticker} SIGNAL {stars} ({score}/5)\n"
            f"Type: {signal_type}\n"
            f"Expiration: {expiry} ({days_to_expiry}d)\n"
            f"Strike: {int(opt['strike'])}\n"
            f"Instruction: Buy to Open {DEFAULT_QTY} {ticker} {int(opt['strike'])}{'C' if is_call else 'P'} exp {datetime.strptime(expiry,'%Y-%m-%d').strftime('%m/%d/%y')}\n"
            f"Entry (spot): {close_price:.2f}\n"
            f"Option mid≈ {opt['mid']:.2f} (bid {opt['bid']:.2f}/ask {opt['ask']:.2f})\n"
            + (f"Est. Cost: ${est_cost:.2f}\n" if est_cost else "")
            + f"Reason: {', '.join(reasons)} | Candle: {pattern if pattern else 'None'}\n"
            f"Horizon: {horizon}\n"
            f"Confidence: {score}⭐\n"
            f"Schwab: Buy to Open {DEFAULT_QTY} {ticker} {int(opt['strike'])}{'C' if is_call else 'P'} exp {datetime.strptime(expiry,'%Y-%m-%d').strftime('%m/%d/%y')}\n"
            f"Note: Use limit orders; this is idea only."
        )

        signature = f"{signal_type}|{expiry}|{int(opt['strike'])}|{ticker}"
        suggestion = {
            "contract": opt.get("contract"),
            "strike": int(opt["strike"]),
            "mid": opt["mid"],
            "bid": opt["bid"],
            "ask": opt["ask"],
            "expiry": expiry,
            "type": signal_type,
            "horizon": horizon
        }

        return {"ticker": ticker, "message": message, "signature": signature, "suggestion": suggestion}
    except Exception as e:
        print("analyze_ticker error for", ticker, e)
        traceback.print_exc()
        return None

# ----------------- Telegram command processing -----------------
def process_telegram_messages(state):
    last_id = state.get("last_update_id")
    updates = get_updates(offset=(last_id + 1) if last_id else None, timeout=3)
    if not updates:
        return state
    for upd in updates:
        state['last_update_id'] = upd.get("update_id")
        msg = upd.get("message") or upd.get("channel_post") or {}
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        sender = msg.get("from", {}).get("username") or msg.get("from", {}).get("first_name") or "user"
        low = text.lower()
        print("Got Telegram:", low)
        # Help
        if low in ("help", "/help"):
            send_telegram(
                "Commands:\n"
                "- help\n"
                "- test\n"
                "- track TICKER  (or: I bought TICKER)\n"
                "- stop TICKER\n"
                "- status TICKER\n"
                "- portfolio\n"
                "- close TICKER\n"
                "- explain stars\n"
                "- alerts on / alerts off"
            )
            continue
        if low in ("explain stars", "stars"):
            send_telegram("Stars 1-5: 5=very strong, 4=strong, 3=ok, 2=weak, 1=noise.")
            continue
        if low in ("test", "/test"):
            send_telegram("🚨 TEST SIGNAL: (Ticker) CALL 100C exp 09/12/25 | Est. Cost: $100 | Confidence: 4⭐")
            continue
        if low == "portfolio":
            pos = state.get("positions", {})
            if not pos:
                send_telegram("📌 Portfolio empty.")
            else:
                lines = ["📌 Tracked positions:"]
                for t,p in pos.items():
                    lines.append(f"- {t}: {p['type']} {p['strike']} exp {p['expiry']} entry ${p['entry_mid']:.2f}")
                send_telegram("\n".join(lines))
            continue
        if low.startswith("alerts off"):
            state["alerts_enabled"] = False
            send_telegram("Alerts paused (tracking still monitored).")
            continue
        if low.startswith("alerts on"):
            state["alerts_enabled"] = True
            send_telegram("Alerts resumed.")
            continue

        # track or "i bought ticker"
        matched = None
        for t in TICKERS:
            if f" {t.lower()}" in " " + low or low == t.lower() or low.startswith(f"i bought {t.lower()}") or low.startswith(f"track {t.lower()}"):
                matched = t
                break
        if matched:
            last_sugg = state.get("last_suggestions", {}).get(matched)
            if not last_sugg:
                send_telegram(f"No recent suggestion for {matched}. Wait for next signal then 'track {matched}'.")
                continue
            pos = state.get("positions", {})
            pos[matched] = {
                "contract": last_sugg["contract"],
                "strike": last_sugg["strike"],
                "mid": last_sugg["mid"],
                "entry_mid": last_sugg["mid"],
                "qty": DEFAULT_QTY,
                "opened_at": datetime.utcnow().isoformat(),
                "expiry": last_sugg["expiry"],
                "type": last_sugg["type"],
                "tp_notified": False,
                "sl_notified": False,
                "tracked_by": sender
            }
            state["positions"] = pos
            send_telegram(f"✅ Tracking {matched} {pos[matched]['type']} {pos[matched]['strike']} exp {pos[matched]['expiry']}. Entry est ${pos[matched]['entry_mid']:.2f}. I’ll alert you if/when conditions change.")
            continue

        # stop tracking
        if low.startswith("stop ") or low.startswith("stoptracking "):
            for t in TICKERS:
                if t.lower() in low:
                    if t in state.get("positions", {}):
                        state["positions"].pop(t, None)
                        send_telegram(f"🛑 Stopped tracking {t}.")
                    else:
                        send_telegram(f"I wasn't tracking {t}.")
                    break
            continue

        # status TICKER
        if low.startswith("status "):
            for t in TICKERS:
                if t.lower() in low:
                    p = state.get("positions", {}).get(t)
                    if not p:
                        send_telegram(f"Not tracking {t}.")
                        break
                    cur_mid = current_option_mid_by_contract(t, p["contract"], p["expiry"], p["type"], p["strike"])
                    if cur_mid is None:
                        send_telegram(f"{t}: current option price unknown right now.")
                        break
                    entry = p["entry_mid"]
                    qty = p.get("qty",1)
                    val_now = cur_mid * 100 * qty
                    entry_val = entry * 100 * qty
                    pnl = val_now - entry_val
                    pct = (pnl / entry_val) * 100 if entry_val else 0
                    send_telegram(f"{t} status: Now ${cur_mid:.2f} -> Value ${val_now:.0f} | P/L ${pnl:.0f} ({pct:+.1f}%)")
                    break
            continue

        # close TICKER
        if low.startswith("close ") or low.startswith("sell "):
            for t in TICKERS:
                if t.lower() in low:
                    p = state.get("positions", {}).get(t)
                    if not p:
                        send_telegram(f"Not tracking {t}.")
                        break
                    cp = "C" if p["type"] == "CALL" else "P"
                    exp_dt = datetime.strptime(p["expiry"], "%Y-%m-%d").date()
                    exp_short = exp_dt.strftime("%m/%d/%y")
                    instruction = f"Sell to Close {p.get('qty',1)} {t} {int(p['strike'])}{cp} exp {exp_short}"
                    send_telegram(f"🔒 Closing guidance for {t}:\n{instruction}\nUse Limit near bid to help fill.")
                    state["positions"].pop(t, None)
                    break
            continue

        # fallback unknown
        send_telegram("I didn't understand that. Send 'help' for commands.")
    return state

# ---------------- Monitor tracked positions ----------------
def monitor_positions(state):
    positions = state.get("positions", {})
    if not positions:
        return state
    for t, p in list(positions.items()):
        try:
            cur_mid = current_option_mid_by_contract(t, p["contract"], p["expiry"], p["type"], p["strike"])
            if cur_mid is None:
                print(f"{t}: current mid unknown")
                continue
            entry_mid = float(p["entry_mid"])
            qty = p.get("qty", 1)
            val_now = cur_mid * 100 * qty
            entry_val = entry_mid * 100 * qty
            pnl = val_now - entry_val
            pct = (pnl / entry_val) if entry_val else 0

            # TP
            if pct >= TAKE_PROFIT_PCT and not p.get("tp_notified"):
                send_telegram(
                    f"🎯 PROFIT ALERT for {t} {p['type']} {p['expiry']} strike {p['strike']}\n"
                    f"Now: ${cur_mid:.2f} x100 = ${val_now:.0f}\n"
                    f"Est. Profit: ${pnl:.0f} ({pct*100:+.1f}%)\n"
                    f"Recommendation: Sell to Close to lock profit.\n"
                    f"Schwab: Sell to Close {qty} {t} {int(p['strike'])}{'C' if p['type']=='CALL' else 'P'} exp {datetime.strptime(p['expiry'],'%Y-%m-%d').strftime('%m/%d/%y')}\n"
                    f"Use Limit near bid."
                )
                p["tp_notified"] = True

            # SL
            if pct <= STOP_LOSS_PCT and not p.get("sl_notified"):
                send_telegram(
                    f"⚠️ STOP-LOSS ALERT for {t} {p['type']} {p['expiry']} strike {p['strike']}\n"
                    f"Now: ${cur_mid:.2f} x100 = ${val_now:.0f}\n"
                    f"Est. Loss: ${pnl:.0f} ({pct*100:+.1f}%)\n"
                    f"Recommendation: Consider Sell to Close to cut losses.\n"
                    f"Schwab: Sell to Close {qty} {t} {int(p['strike'])}{'C' if p['type']=='CALL' else 'P'} exp {datetime.strptime(p['expiry'],'%Y-%m-%d').strftime('%m/%d/%y')}\n"
                    f"Use Limit near bid."
                )
                p["sl_notified"] = True

            # Save update
            state["positions"][t] = p
        except Exception as e:
            print("monitor_positions error for", t, e)
    return state

# ---------------- Main run flow (single scheduled run) ----------------
def main_run_once():
    state = load_state()

    # Process incoming Telegram commands first
    try:
        state = process_telegram_messages(state)
    except Exception as e:
        print("Error processing telegram messages:", e, traceback.format_exc())

    # Debug forced test message (quick end-to-end check)
    if DEBUG_FORCE_TEST:
        send_telegram(f"🚨 DEBUG TEST: {DEBUG_TEST_TICKER} fake signal ⭐⭐⭐")
        return state

    # If alerts enabled, scan tickers
    if state.get("alerts_enabled", True):
        last_signals = state.get("last_signals", {})
        last_suggestions = state.get("last_suggestions", {})
        new_signals = []
        for t in TICKERS:
            try:
                res = analyze_ticker(t)
                if not res:
                    # no signal for this ticker
                    continue
                # save last suggestion for convenience (track)
                last_suggestions[t] = res["suggestion"]
                sig = res["signature"]
                # dedupe by signature per ticker
                if last_signals.get(t) != sig:
                    # send message
                    send_telegram(res["message"])
                    last_signals[t] = sig
                    new_signals.append(t)
                else:
                    print(f"{t}: no change (same signature)")
            except Exception as e:
                print("Scan error for", t, e, traceback.format_exc())
        state["last_signals"] = last_signals
        state["last_suggestions"] = last_suggestions
        print("Alerts sent for:", new_signals if new_signals else [])
    else:
        print("Alerts disabled in state (alerts_enabled=False)")

    # Monitor tracked positions for TP/SL
    try:
        state = monitor_positions(state)
    except Exception as e:
        print("monitor_positions failed:", e, traceback.format_exc())

    # Save state
    save_state(state)
    print("Run complete.")
    return state

# ---------------- Entry point ----------------
if __name__ == "__main__":
    main_run_once()
