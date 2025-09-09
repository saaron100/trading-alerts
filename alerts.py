# alerts.py
# Full scanner + Telegram command processor
# Requires: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as env vars
# State persisted to state.json

import os
import json
import math
import requests
from datetime import datetime, date, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import time

# --------- USER CONFIG ----------
TICKERS = ["SPY","AAPL","CVX","AMZN","QQQ","GLD","SLV","PLTR",
           "USO","NFLX","TNA","XOM","NVDA","BAC","TSLA","META"]

STATE_FILE = "state.json"
RSI_PERIOD = 14
EMA_SHORT = 20
EMA_LONG = 50

# P/L thresholds (you can tune)
TAKE_PROFIT_PCT = 0.50   # +50% = strong profit alert
STOP_LOSS_PCT   = -0.30  # -30% = strong stop-loss alert

# How many contracts default when adding a tracked position
DEFAULT_QTY = 1

# --------------------------------

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")

# ---------- Helpers ----------
def send_telegram(text):
    """Send a message to your Telegram chat"""
    if not BOT_TOKEN or not CHAT_ID:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text}
        r = requests.post(url, json=payload, timeout=15)
        print("telegram:", r.status_code, r.text[:200])
    except Exception as e:
        print("Telegram send error:", str(e))

def get_updates(offset=None, timeout=5):
    """Poll Telegram updates (messages)"""
    if not BOT_TOKEN:
        return []
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {"timeout": timeout}
    if offset:
        params["offset"] = offset
    try:
        r = requests.get(url, params=params, timeout=timeout+2)
        j = r.json()
        if not j.get("ok"):
            print("getUpdates not ok:", j)
            return []
        return j.get("result", [])
    except Exception as e:
        print("get_updates error:", e)
        return []

def compute_rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
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
    """Choose expiry string (yyyy-mm-dd) based on 5-star score (1..5)"""
    exps = ticker_obj.options
    parsed = parse_expirations(exps)
    if not parsed:
        return None
    today = date.today()

    # mapping: higher score = shorter preferred expiries
    # score 5 -> within 0-7 days
    # score 4 -> within 0-14 days
    # score 3 -> 7-30 days
    # score 2 -> 14-60 days
    # score 1 -> 30-90 days
    ranges = {
        5: (0, 7),
        4: (0, 14),
        3: (7, 30),
        2: (14, 60),
        1: (30, 120)
    }
    low, high = ranges.get(max(1, min(5, score)), (7,30))
    candidates = [s for (d,s) in parsed if low <= (d - today).days <= max(1, high)]
    if candidates:
        return candidates[0]
    # fallback nearest future expiry
    for (d,s) in parsed:
        if (d - today).days >= 0:
            return s
    return parsed[-1][1]

def pick_atm_contract(ticker, expiry, is_call, spot):
    """Return one ATM contract dict or None"""
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
        print("pick_atm_contract error:", e)
        return None

def score_signal_5star(signal_type, rsi_val, close, ema_s, ema_l, volume_series):
    """Return score 1..5 (5 strongest)"""
    score = 1
    # volume spike detection: current volume > mean(last 20) * 1.5
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

# ---------- State persistence ----------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "last_update_id": None,
            "last_signals": {},         # ticker -> signature
            "last_suggestions": {},     # ticker -> option info suggested last run
            "positions": {},            # ticker -> tracked position
            "alerts_enabled": True
        }
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
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

# ---------- Analysis ----------
def analyze_ticker(ticker):
    """Return dict with message & signature & suggestion dict, or None"""
    try:
        df = yf.download(ticker, period="3mo", interval="30m", progress=False)
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < RSI_PERIOD + 2:
            return None

        rsi_series = compute_rsi(close, RSI_PERIOD)
        ema_s = compute_ema(close, EMA_SHORT)
        ema_l = compute_ema(close, EMA_LONG)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        rsi_val = float(rsi_series.iloc[-1])
        close_price = float(last["Close"])
        ema_short_val = float(ema_s.iloc[-1])
        ema_long_val = float(ema_l.iloc[-1])
        candle_green = float(last["Close"]) > float(last["Open"])
        volume_series = df["Volume"] if "Volume" in df.columns else pd.Series([])

        # Directional rules (strict)
        signal_type = None
        if rsi_val < 30 and ema_short_val > ema_long_val and close_price > ema_short_val:
            signal_type = "CALL"
        elif rsi_val > 70 and ema_short_val < ema_long_val and close_price < ema_short_val:
            signal_type = "PUT"

        if not signal_type:
            return None

        # Score 1..5
        score = score_signal_5star(signal_type, rsi_val, close_price, ema_short_val, ema_long_val, volume_series)

        # choose expiry based on score
        tkr = yf.Ticker(ticker)
        expiry = choose_expiry_by_score(tkr, score)
        if not expiry:
            return None

        is_call = (signal_type == "CALL")
        opt = pick_atm_contract(ticker, expiry, is_call, close_price)
        if not opt:
            return None

        # Build message
        exp_lab = expiry_label(expiry)
        est_cost = None
        if opt["mid"] and opt["mid"] > 0:
            est_cost = opt["mid"] * 100

        # Daytrade/swing/long categorization
        days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
        if days_to_expiry <= 3:
            horizon = "Daytrade (very short)"
        elif days_to_expiry <= 14:
            horizon = "Short-term / swing"
        else:
            horizon = "Medium/long-term"

        message = (
            f"📊 {ticker} SIGNAL { '⭐'*score } ({score}/5)\n"
            f"Type: {signal_type}\n"
            f"Expiration: {exp_lab}\n"
            f"Strike: {int(opt['strike'])}\n"
            f"Instruction: Buy to Open {DEFAULT_QTY} {ticker} {int(opt['strike'])}{'C' if is_call else 'P'} exp {datetime.strptime(expiry, '%Y-%m-%d').strftime('%m/%d/%y')}\n"
            f"Entry (spot): {close_price:.2f}\n"
            f"Option mid≈ {opt['mid']:.2f} (bid {opt['bid']:.2f}/ask {opt['ask']:.2f})\n"
            + (f"Est. Cost: ${est_cost:.2f}\n" if est_cost else "")
            + f"Reason: RSI {rsi_val:.1f}, EMA{EMA_SHORT} {'>' if ema_short_val>ema_long_val else '<'} EMA{EMA_LONG}\n"
            f"Horizon: {horizon}\n"
            f"Confidence: {score}⭐\n"
            f"Note: Use limit orders; this is an idea — do your own checks."
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
        return None

def current_option_mid_by_contract(ticker, contract_symbol, expiry, opt_type, strike):
    """Return mid or None for a tracked contract"""
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        tab = chain.calls if opt_type == "CALL" else chain.puts
        row = tab.loc[tab["contractSymbol"] == contract_symbol]
        if row.empty:
            # fallback by strike match
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

# ---------- Command processing ----------
def process_telegram_messages(state):
    """Read Telegram updates and handle commands. Update state accordingly."""
    last_id = state.get("last_update_id")
    updates = get_updates(offset=(last_id + 1) if last_id else None, timeout=3)
    if not updates:
        return state
    for upd in updates:
        # store latest update_id
        update_id = upd.get("update_id")
        state["last_update_id"] = update_id
        msg = upd.get("message") or upd.get("channel_post") or {}
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        user = msg.get("from", {}).get("username") or msg.get("from", {}).get("first_name") or "user"
        low = text.lower().strip()
        print("Got message:", low)
        # Simple parsing
        # HELP
        if low in ("help", "/help"):
            help_text = (
                "Commands:\n"
                "- tracking -> show tracked positions\n"
                "- track TICKER or I bought TICKER -> track the last suggested contract for TICKER\n"
                "- stop TICKER -> stop tracking TICKER\n"
                "- status TICKER -> current value & P/L for tracked TICKER\n"
                "- close TICKER -> show sell instruction and stop tracking\n"
                "- portfolio -> summary\n"
                "- explain stars -> describe star ratings\n"
                "- alerts off / alerts on -> pause/resume new signals\n"
            )
            send_telegram(help_text)
            continue
        if low in ("explain stars", "stars", "explain"):
            txt = ("Stars (1-5): 5=very strong, 4=strong, 3=ok, 2=weak, 1=noise.\n"
                   "Higher stars = shorter expiries recommended and higher conviction.")
            send_telegram(txt)
            continue
        if low == "tracking":
            pos = state.get("positions", {})
            if not pos:
                send_telegram("📌 Not tracking anything right now.")
            else:
                lines = ["📌 Currently tracking:"]
                for tiker, p in pos.items():
                    lines.append(f"- {tiker} {p['type']} {p['strike']} exp {p['expiry']} | entry ${p['entry_mid']:.2f} | qty {p.get('qty',1)}")
                send_telegram("\n".join(lines))
            continue
        if low.startswith("alerts off"):
            state["alerts_enabled"] = False
            send_telegram("Alerts paused. Tracking still runs.")
            continue
        if low.startswith("alerts on"):
            state["alerts_enabled"] = True
            send_telegram("Alerts resumed.")
            continue
        if low.startswith("portfolio"):
            pos = state.get("positions", {})
            if not pos:
                send_telegram("Portfolio empty.")
            else:
                lines = ["Portfolio:"]
                for tiker,p in pos.items():
                    # try to compute current mid
                    cur_mid = current_option_mid_by_contract(tiker, p["contract"], p["expiry"], p["type"], p["strike"])
                    if cur_mid is None:
                        lines.append(f"- {tiker}: contract {p['contract']} - current price unknown")
                        continue
                    entry = p["entry_mid"]
                    qty = p.get("qty",1)
                    val_now = cur_mid * 100 * qty
                    entry_val = entry * 100 * qty
                    pnl = val_now - entry_val
                    pct = (pnl / entry_val) * 100 if entry_val else 0
                    lines.append(f"- {tiker}: Now ${cur_mid:.2f} | P/L ${pnl:.0f} ({pct:+.1f}%)")
                send_telegram("\n".join(lines))
            continue

        # Track / I bought
        matched = None
        for t in TICKERS:
            if f" {t.lower()}" in " " + low or low == t.lower() or low.startswith(f"i bought {t.lower()}") or low.startswith(f"track {t.lower()}"):
                matched = t
                break

        if matched:
            # find last suggestion for this ticker
            last_sugg = state.get("last_suggestions", {}).get(matched)
            if not last_sugg:
                send_telegram(f"I don't have a recent suggested contract for {matched}. Wait for next signal, then 'track {matched}'.")
                continue
            # Add to positions
            pos = state.get("positions", {})
            # default qty 1
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
                "tracked_by": user
            }
            state["positions"] = pos
            send_telegram(f"✅ Tracking {matched} {pos[matched]['type']} {pos[matched]['strike']} exp {pos[matched]['expiry']}. Entry est ${pos[matched]['entry_mid']:.2f}. I’ll alert you if/when conditions change.")
            continue

        # Stop tracking / stop TICKER
        if low.startswith("stop ") or low.startswith("stoptracking ") or low.startswith("stop "):
            # find ticker token
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

        # close TICKER -> provide sell instr and stop tracking
        if low.startswith("close ") or low.startswith("sell "):
            for t in TICKERS:
                if t.lower() in low:
                    p = state.get("positions", {}).get(t)
                    if not p:
                        send_telegram(f"Not tracking {t}.")
                        break
                    # compose short Schwab instruction
                    cp = "C" if p["type"] == "CALL" else "P"
                    exp_dt = datetime.strptime(p["expiry"], "%Y-%m-%d").date()
                    exp_short = exp_dt.strftime("%m/%d/%y")
                    instruction = f"Sell to Close {p.get('qty',1)} {t} {int(p['strike'])}{cp} exp {exp_short}"
                    send_telegram(f"🔒 Closing guidance for {t}:\n{instruction}\nUse Limit near bid to help fill.")
                    # remove from tracked positions
                    state["positions"].pop(t, None)
                    break
            continue

        # unknown command -> hint
        send_telegram("I didn't understand that. Send 'help' for commands.")
    return state

# ---------- Position monitoring ----------
def monitor_positions(state):
    """Check tracked positions and send alerts on TP/SL or reversal"""
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
                    f"🎯 PROFIT ALERT for {t} {p['type']} {p['expiry']} {p['strike']}\n"
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
                    f"⚠️ STOP-LOSS ALERT for {t} {p['type']} {p['expiry']} {p['strike']}\n"
                    f"Now: ${cur_mid:.2f} x100 = ${val_now:.0f}\n"
                    f"Est. Loss: ${pnl:.0f} ({pct*100:+.1f}%)\n"
                    f"Recommendation: Consider Sell to Close to cut losses.\n"
                    f"Schwab: Sell to Close {qty} {t} {int(p['strike'])}{'C' if p['type']=='CALL' else 'P'} exp {datetime.strptime(p['expiry'],'%Y-%m-%d').strftime('%m/%d/%y')}\n"
                    f"Use Limit near bid."
                )
                p["sl_notified"] = True

            # Save possible updated flags
            state["positions"][t] = p

        except Exception as e:
            print("monitor_positions error for", t, e)
    return state

# ---------- Main ----------
def main():
    state = load_state()

    # 1) Read Telegram messages (commands) and process them
    try:
        state = process_telegram_messages(state)
    except Exception as e:
        print("process_telegram_messages failed:", e)

    # 2) If alerts are enabled, scan tickers for new signals
    if state.get("alerts_enabled", True):
        last_signals = state.get("last_signals", {})
        last_suggestions = state.get("last_suggestions", {})
        new_signals = []
        for t in TICKERS:
            try:
                res = analyze_ticker(t)
                if not res:
                    continue
                sig = res["signature"]
                # Save last suggestion unconditionally (so 'track TICKER' works even if same)
                last_suggestions[t] = res["suggestion"]
                if last_signals.get(t) != sig:
                    # new or changed signal -> alert + track suggestion
                    send_telegram(res["message"])
                    last_signals[t] = sig
                    new_signals.append(t)
                else:
                    print(f"{t}: same signal as before")
            except Exception as e:
                print("Scan error for", t, e)
        state["last_signals"] = last_signals
        state["last_suggestions"] = last_suggestions
        if new_signals:
            print("Alerts sent for:", new_signals)
        else:
            print("Done. Alerts sent for: []")

    # 3) Monitor positions being tracked (P/L & TP/SL)
    try:
        state = monitor_positions(state)
    except Exception as e:
        print("monitor_positions error:", e)

    # 4) Save state back
    save_state(state)
    print("Run complete.")

if __name__ == "__main__":
    main()
