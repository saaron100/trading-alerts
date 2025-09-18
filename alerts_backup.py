# alerts.py
# Full Trading Alerts + Telegram command handler
# - Uses yfinance for prices and option chains (with intraday -> daily fallback)
# - Sends Telegram messages via bot API
# - Processes Telegram messages via getUpdates (suitable for scheduled runs)
# - Optional local immediate listener (python-telegram-bot) if you run locally
#
# Requirements:
#   pip install yfinance pandas numpy requests
# Optional (for local mode):
#   pip install python-telegram-bot==13.15
#
# Environment:
#   TELEGRAM_BOT_TOKEN  (required)
#   TELEGRAM_CHAT_ID    (required)
#
# State file: state.json (committed back by workflow so state persists)

import os
import json
import math
import time
import requests
import traceback
from datetime import datetime, date, timedelta

import yfinance as yf
import pandas as pd
import numpy as np

# Optional local listener import (only required if you run with --local)
try:
    from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
    TELEGRAM_BOTLIB_AVAILABLE = True
except Exception:
    TELEGRAM_BOTLIB_AVAILABLE = False

# ---------------- CONFIG ----------------
TICKERS = ["SPY","AAPL","CVX","AMZN","QQQ","GLD","SLV","PLTR",
           "USO","NFLX","TNA","XOM","NVDA","BAC","TSLA","META"]

STATE_FILE = "state.json"
RSI_PERIOD = 14
EMA_SHORT = 20
EMA_LONG = 50

DEFAULT_QTY = 1
TAKE_PROFIT_PCT = 0.50   # notify when +50%
STOP_LOSS_PCT   = -0.30  # notify when -30%

# Environment variables (required)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")

# Some internal settings
INTRADAY_TRY_PERIODS = [ "7d", "30d", "60d" ]  # attempt intraday periods in order
INTRADAY_INTERVAL = "15m"
FALLBACK_DAILY_PERIOD = "3mo"
FALLBACK_DAILY_INTERVAL = "1d"

# ----------------- Helpers -----------------
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
        print("Telegram send:", r.status_code, (j if j else r.text)[:200])
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

# ----------------- Data fetch (robust) -----------------
def get_data(ticker):
    """
    Try intraday first (several short periods). If that fails or yields empty,
    fallback to daily data.
    """
    # Prefer smaller intraday window to reduce Yahoo errors
    for p in INTRADAY_TRY_PERIODS:
        try:
            df = yf.download(ticker, period=p, interval=INTRADAY_INTERVAL, progress=False)
            if df is not None and not df.empty:
                # if index is timezone-aware, convert to naive to avoid future confusion
                return df
        except Exception as e:
            # log and try next period
            print(f"{ticker} intraday attempt period={p} failed: {e}")
            continue
    # fallback to daily
    try:
        df = yf.download(ticker, period=FALLBACK_DAILY_PERIOD, interval=FALLBACK_DAILY_INTERVAL, progress=False)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as e:
        print("get_data fallback failed for", ticker, e)
        return pd.DataFrame()

# ----------------- Technicals -----------------
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

# ----------------- Option expiry & strike helpers -----------------
def parse_expirations(exp_list):
    exps = []
    for s in exp_list:
        try:
            dtobj = datetime.strptime(s, "%Y-%m-%d").date()
            exps.append((dtobj, s))
        except Exception:
            continue
    exps.sort()
    return exps

def choose_expiry_by_score(ticker_obj, score):
    """
    Choose expiry (string 'YYYY-MM-DD') based on 5-star score.
    Higher score -> prefer nearer expirations.
    """
    exps = ticker_obj.options
    parsed = parse_expirations(exps)
    if not parsed:
        return None
    today = date.today()
    # ranges in days (min,max)
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
    # fallback nearest future
    for (d,s) in parsed:
        if (d - today).days >= 0:
            return s
    return parsed[-1][1]

def pick_atm_contract(ticker, expiry, is_call, spot):
    """Return best ATM contract dict for the given expiry or None."""
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
        last = float(row.get("lastPrice") or 0)
        if bid > 0 and ask > 0:
            mid = round((bid + ask) / 2, 2)
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
        print("pick_atm_contract error for", ticker, expiry, e)
        return None

def current_option_mid_by_contract(ticker, contract_symbol, expiry, opt_type, strike):
    """Return current mid price (float) for a given contract symbol/strike."""
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
        last = float(row.iloc[0].get("lastPrice") or 0)
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2, 2)
        return round(last if last > 0 else max(bid, ask), 2)
    except Exception as e:
        print("current_option_mid error:", e)
        return None

# ----------------- Scoring logic (5-star) -----------------
def score_signal_5star(signal_type, rsi_val, close, ema_s, ema_l, volume_series):
    score = 1
    vol_spike = False
    try:
        if len(volume_series) >= 20:
            mean_vol = np.nanmean(volume_series[-20:])
            if mean_vol > 0 and volume_series.iloc[-1] > mean_vol * 1.5:
                vol_spike = True
    except Exception:
        vol_spike = False

    if signal_type == "CALL":
        if rsi_val < 15:
            score += 2
        elif rsi_val < 22:
            score += 1
        if ema_s > ema_l:
            score += 1
        if (close > ema_s) and vol_spike:
            score += 1
    else:  # PUT
        if rsi_val > 85:
            score += 2
        elif rsi_val > 78:
            score += 1
        if ema_s < ema_l:
            score += 1
        if (close < ema_s) and vol_spike:
            score += 1

    return max(1, min(5, score))

def expiry_label(expiry_str):
    try:
        e = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = date.today()
        days = (e - today).days
        if days <= 7:
            return f"{expiry_str} (this week)"
        elif days <= 14:
            return f"{expiry_str} (next week)"
        elif days <= 30:
            return f"{expiry_str} (1–4 weeks)"
        return expiry_str
    except Exception:
        return expiry_str

# ----------------- State persistence -----------------
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

# ----------------- Analysis per ticker -----------------
def analyze_ticker(ticker):
    try:
        df = get_data(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        close_series = df["Close"].dropna()
        if len(close_series) < RSI_PERIOD + 2:
            return None

        rsi_series = compute_rsi(close_series, RSI_PERIOD)
        ema_s = compute_ema(close_series, EMA_SHORT)
        ema_l = compute_ema(close_series, EMA_LONG)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        rsi_val = float(rsi_series.iloc[-1])
        close_price = float(last["Close"])
        ema_short_val = float(ema_s.iloc[-1])
        ema_long_val = float(ema_l.iloc[-1])
        candle_green = float(last["Close"]) > float(last.get("Open", last["Close"]))
        volume_series = df["Volume"] if "Volume" in df.columns else pd.Series([])

        signal_type = None
        # Strict directional rules
        if rsi_val < 30 and ema_short_val > ema_long_val and close_price > ema_short_val:
            signal_type = "CALL"
        elif rsi_val > 70 and ema_short_val < ema_long_val and close_price < ema_short_val:
            signal_type = "PUT"

        if not signal_type:
            return None

        score = score_signal_5star(signal_type, rsi_val, close_price, ema_short_val, ema_long_val, volume_series)

        tkr = yf.Ticker(ticker)
        expiry = choose_expiry_by_score(tkr, score)
        if not expiry:
            return None

        is_call = (signal_type == "CALL")
        opt = pick_atm_contract(ticker, expiry, is_call, close_price)
        if not opt:
            return None

        est_cost = opt["mid"] * 100 if opt["mid"] and opt["mid"] > 0 else None

        days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
        if days_to_expiry <= 3:
            horizon = "Daytrade (very short)"
        elif days_to_expiry <= 14:
            horizon = "Short-term / swing"
        else:
            horizon = "Medium/long-term"

        message = (
            f"📊 {ticker} SIGNAL {'⭐'*score} ({score}/5)\n"
            f"Type: {signal_type}\n"
            f"Expiration: {expiry_label(expiry)}\n"
            f"Strike: {int(opt['strike'])}\n"
            f"Instruction: Buy to Open {DEFAULT_QTY} {ticker} {int(opt['strike'])}{'C' if is_call else 'P'} exp {datetime.strptime(expiry, '%Y-%m-%d').strftime('%m/%d/%y')}\n"
            f"Entry (spot): {close_price:.2f}\n"
            f"Option mid≈ {opt['mid']:.2f} (bid {opt['bid']:.2f}/ask {opt['ask']:.2f})\n"
            + (f"Est. Cost: ${est_cost:.2f}\n" if est_cost else "")
            + f"Reason: RSI {rsi_val:.1f}, EMA{EMA_SHORT} {'>' if ema_short_val>ema_long_val else '<'} EMA{EMA_LONG}\n"
            f"Horizon: {horizon}\n"
            f"Confidence: {score}⭐\n"
            f"Note: Use limit orders; this is idea only."
        )

        signature = f"{signal_type}|{expiry}|{int(opt['strike'])}"
        suggestion = {
            "contract": opt["contract"],
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

# ----------------- Command processing via getUpdates -----------------
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
            # immediate fake alert
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

        # track or "i bought ticker" convenience
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

# ----------------- Monitor tracked positions -----------------
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

# ----------------- Main run flow -----------------
def main_run_once():
    state = load_state()

    # 1) Process any incoming Telegram commands/messages
    try:
        state = process_telegram_messages(state)
    except Exception as e:
        print("Error processing telegram messages:", e, traceback.format_exc())

    # 2) If alerts enabled, scan tickers
    if state.get("alerts_enabled", True):
        last_signals = state.get("last_signals", {})
        last_suggestions = state.get("last_suggestions", {})
        new_signals = []
        for t in TICKERS:
            try:
                res = analyze_ticker(t)
                if not res:
                    continue
                # save last suggestion for track convenience
                last_suggestions[t] = res["suggestion"]
                sig = res["signature"]
                if last_signals.get(t) != sig:
                    send_telegram(res["message"])
                    last_signals[t] = sig
                    new_signals.append(t)
                else:
                    print(f"{t}: no change")
            except Exception as e:
                print("Scan error for", t, e)
        state["last_signals"] = last_signals
        state["last_suggestions"] = last_suggestions
        print("Alerts sent for:", new_signals if new_signals else [])
    else:
        print("Alerts disabled (alerts_enabled=False)")

    # 3) Monitor tracked positions (TP/SL)
    try:
        state = monitor_positions(state)
    except Exception as e:
        print("monitor_positions failed:", e)

    # 4) Save state
    save_state(state)
    print("Run complete.")
    return state

# ----------------- Optional local immediate listener (if you run locally) -----------------
def local_listener_mode():
    """If you run locally and want instant replies, run script with --local.
       This requires python-telegram-bot to be installed.
    """
    if not TELEGRAM_BOTLIB_AVAILABLE:
        print("Local listener requires python-telegram-bot installed (pip install python-telegram-bot==13.15)")
        return

    # Simple handlers that call the same logic used by process_telegram_messages.
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    def help_cmd(update, context):
        update.message.reply_text("Commands: help, test, track TICKER, stop TICKER, portfolio, status TICKER, close TICKER")
    def test_cmd(update, context):
        update.message.reply_text("🚨 TEST ALERT: (Ticker) CALL 100C exp 09/12/25 | Est. Cost: $100 | Confidence: 4⭐")
    def track_cmd(update, context):
        args = context.args
        if not args:
            update.message.reply_text("Usage: /track TSLA")
            return
        t = args[0].upper()
        state = load_state()
        last_sugg = state.get("last_suggestions", {}).get(t)
        if not last_sugg:
            update.message.reply_text(f"No suggestion for {t} yet.")
            return
        pos = state.get("positions", {})
        pos[t] = {
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
            "tracked_by": update.message.from_user.username if update.message.from_user else "user"
        }
        state["positions"] = pos
        save_state(state)
        update.message.reply_text(f"✅ Now tracking {t} {pos[t]['type']} {pos[t]['strike']} exp {pos[t]['expiry']}")
    def stop_cmd(update, context):
        args = context.args
        if not args:
            update.message.reply_text("Usage: /stop TSLA")
            return
        t = args[0].upper()
        state = load_state()
        if t in state.get("positions", {}):
            state["positions"].pop(t, None)
            save_state(state)
            update.message.reply_text(f"🛑 Stopped tracking {t}")
        else:
            update.message.reply_text(f"I wasn't tracking {t}")

    dp.add_handler(CommandHandler("help", help_cmd))
    dp.add_handler(CommandHandler("test", test_cmd))
    dp.add_handler(CommandHandler("track", track_cmd))
    dp.add_handler(CommandHandler("stop", stop_cmd))
    dp.add_handler(CommandHandler("start", lambda u,c: u.message.reply_text("Bot running locally.")))

    print("Starting local listener (polling). Press Ctrl+C to stop.")
    updater.start_polling()
    updater.idle()

# ----------------- Entry point -----------------
if __name__ == "__main__":
    import sys
    # If user runs with "--local" argument, start local listener mode (instant).
    if len(sys.argv) > 1 and sys.argv[1] in ("--local", "-l"):
        local_listener_mode()
    else:
        # normal scheduled-run mode: run once (good for GitHub Actions schedule)
        main_run_once()
