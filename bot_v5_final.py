#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mahdi v5 PRO - FINAL (LONG & SHORT)
This bot executes MARKET entries (BUY for long, SELL for short)
Note: No timeInForce for MARKET orders; workingType=CONTRACT_PRICE
"""

# =========================================================
# 📦 Import Required Libraries
# =========================================================
import os
import time
import hmac
import hashlib
import math
import requests
import urllib.parse as urlparse
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from requests.exceptions import RequestException

from typing import List, Dict, Tuple, Any
import requests

# -------- Logging --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("mahdi-v5-pro")

# -------- ENV --------
def env_bool(k, default=False):
    v = os.getenv(k, str(default)).strip().lower()
    return v in ("1","true","yes","y","on")

def env_float(k, default: float):
    try: return float(os.getenv(k, str(default)).strip())
    except: return default

def env_int(k, default: int):
    try: return int(float(os.getenv(k, str(default)).strip()))
    except: return default

RUN_MODE            = os.getenv("RUN_MODE","live").strip().lower()     # live | paper
USE_TESTNET         = env_bool("USE_TESTNET", False)

SYMBOLS_CSV         = os.getenv("SYMBOLS_CSV","universe.csv").strip()
MAX_SYMBOLS         = env_int("MAX_SYMBOLS", 25)

INTERVAL            = os.getenv("INTERVAL","5m").strip()
INTERVAL_CONFIRM    = os.getenv("INTERVAL_CONFIRM","15m").strip()
KLINES_LIMIT        = env_int("KLINES_LIMIT", 300)
SCAN_INTERVAL_SEC   = env_int("SCAN_INTERVAL_SEC", 120)
COOLDOWN_MIN        = env_int("COOLDOWN_MIN", 60)

CAPITAL_USE_PCT     = env_float("CAPITAL_USE_PCT", 0.40)
MAX_OPEN_POS        = env_int("MAX_OPEN_POS", 6)

SLOT_A_PCT          = env_float("SLOT_A_PCT", 0.06)   # long bucket
SLOT_A_LEV          = env_int("SLOT_A_LEV", 10)
SLOT_B_PCT          = env_float("SLOT_B_PCT", 0.05)   # short bucket
SLOT_B_LEV          = env_int("SLOT_B_LEV", 5)

TP1_PCT             = env_float("TP1_PCT", 0.0035)
TP2_PCT             = env_float("TP2_PCT", 0.0070)
TP3_PCT             = env_float("TP3_PCT", 0.0120)
SL_PCT              = env_float("SL_PCT", 0.0075)

CONSENSUS_MIN       = env_float("CONSENSUS_MIN", 0.60)
ADX_MIN             = env_float("ADX_MIN", 20.0)

API_KEY             = (os.getenv("API_KEY","") or "").strip()
API_SECRET          = (os.getenv("API_SECRET","") or "").strip()

TG_ENABLED          = env_bool("TG_ENABLED", True)
TG_TOKEN            = (os.getenv("TELEGRAM_TOKEN","") or "").strip()
TG_CHAT_ID          = (os.getenv("TELEGRAM_CHAT_ID","") or "").strip()
TG_HEARTBEAT_MIN    = env_int("TG_HEARTBEAT_MIN", 15)
TG_NOTIFY_UNIVERSE  = env_bool("TG_NOTIFY_UNIVERSE", False)

BASE = "https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"
SESSION = requests.Session()
if API_KEY:
    SESSION.headers.update({"X-MBX-APIKEY": API_KEY})

# -------- Telegram --------
def tg(msg: str):
    if not TG_ENABLED or not TG_TOKEN or not TG_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        SESSION.post(url, data={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode":"HTML"}, timeout=8)
    except Exception as e:
        log.warning(f"[TG] {e}")

# -------- Time sync --------
_server_off_ms = 0
def sync_time():
    global _server_off_ms
    try:
        r = SESSION.get(f"{BASE}/fapi/v1/time", timeout=5); r.raise_for_status()
        srv = int(r.json().get("serverTime"))
        _server_off_ms = srv - int(time.time()*1000)
    except Exception as e:
        log.warning(f"time sync failed: {e}")
def now_ms():
    return int(time.time()*1000) + _server_off_ms
sync_time()

# -------- Signed helpers --------
def _sign(params: Dict[str, Any]) -> str:
    qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"

def get_public(path: str, params: Dict[str, Any]=None):
    url = f"{BASE}{path}"
    r = SESSION.get(url, params=params, timeout=12); r.raise_for_status()
    return r.json()

def get_signed(path: str, params: Dict[str, Any]):
    p = dict(params); p.setdefault("timestamp", now_ms()); p.setdefault("recvWindow", 5000)
    url = f"{BASE}{path}?{_sign(p)}"
    r = SESSION.get(url, timeout=12); r.raise_for_status(); return r.json()

def post_signed(path: str, params: Dict[str, Any]):
    p = dict(params); p.setdefault("timestamp", now_ms()); p.setdefault("recvWindow", 5000)
    url = f"{BASE}{path}"
    body = _sign(p)
    r = SESSION.post(url, data=body, timeout=12); r.raise_for_status(); return r.json()

# -------- Universe --------
def read_symbols(path: str, maxn: int) -> List[str]:
    syms = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().upper().replace("/", "")
                if not s or s.startswith("#"): continue
                if not s.endswith("USDT"): s += "USDT"
                syms.append(s)
                if len(syms) >= maxn: break
    except Exception as e:
        log.error(f"read_symbols: {e}")
    return syms

UNIVERSE = read_symbols(SYMBOLS_CSV, MAX_SYMBOLS)
if TG_NOTIFY_UNIVERSE: tg(f"📊 Universe ({len(UNIVERSE)}): {', '.join(UNIVERSE)}")
tg(f"♻️ Mahdi v5 PRO: {RUN_MODE.upper()} | Testnet: {'On' if USE_TESTNET else 'Off'}")

# -------- Exchange info & rounding --------
_exinfo = None
_filter_cache: Dict[str, Tuple[float,float,float]] = {}

def exchange_info():
    global _exinfo
    if _exinfo is None:
        _exinfo = get_public("/fapi/v1/exchangeInfo")
    return _exinfo

def symbol_filters(symbol: str) -> Tuple[float,float,float]:
    if symbol in _filter_cache: return _filter_cache[symbol]
    info = exchange_info()
    step=tick=min_not=0.0
    for s in info.get("symbols", []):
        if s["symbol"] == symbol:
            for f in s["filters"]:
                if f["filterType"]=="LOT_SIZE":     step=float(f["stepSize"])
                elif f["filterType"]=="PRICE_FILTER": tick=float(f["tickSize"])
                elif f["filterType"]=="MIN_NOTIONAL": min_not=float(f.get("notional", 0))
            break
    if step==0: step=0.001
    if tick==0: tick=0.01
    _filter_cache[symbol]=(step,tick,min_not)
    return _filter_cache[symbol]

def round_step(x: float, step: float) -> float:
    if step<=0: return x
    return math.floor(x/step)*step

# -------- Market data --------
def klines(symbol: str, interval: str, limit: int=200):
    return get_public("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

def last_price(symbol: str) -> float:
    j = get_public("/fapi/v1/ticker/price", {"symbol": symbol})
    return float(j["price"])

# -------- Indicators (5) --------
def ema(arr: List[float], n: int) -> List[float]:
    if not arr: return []
    k = 2/(n+1)
    out=[]; e=arr[0]; out.append(e)
    for v in arr[1:]:
        e = v*k + e*(1-k); out.append(e)
    return out

def rsi(arr: List[float], n: int=14) -> float:
    if len(arr) < n+1: return 50.0
    gains=losses=0.0
    for i in range(-n, 0):
        ch = arr[i]-arr[i-1]
        if ch>=0: gains += ch
        else: losses -= ch
    if losses==0: return 100.0
    rs = (gains/n)/(losses/n)
    return 100 - 100/(1+rs)

def macd(arr: List[float], fast=12, slow=26, signal=9):
    efast = ema(arr, fast)
    eslow = ema(arr, slow)
    macd_line = [(a-b) for a,b in zip(efast, eslow)]
    sig = ema(macd_line, signal)
    hist = [a-b for a,b in zip(macd_line, sig)]
    return macd_line[-1], sig[-1], hist[-1]

def true_range(h,l,c):
    trs=[]; pc=None
    for i in range(len(c)):
        if i==0 or pc is None: trs.append(h[i]-l[i])
        else: trs.append(max(h[i]-l[i], abs(h[i]-pc), abs(l[i]-pc)))
        pc=c[i]
    return trs

def adx_val(h,l,c, period=14):
    if len(c)<period+2: return 0.0
    plus_dm=[0.0]; minus_dm=[0.0]
    for i in range(1,len(c)):
        up = h[i]-h[i-1]; dn = l[i-1]-l[i]
        plus_dm.append(up if (up>dn and up>0) else 0.0)
        minus_dm.append(dn if (dn>up and dn>0) else 0.0)
    tr = true_range(h,l,c)
    def rma(x,n):
        if len(x)<n: return 0.0
        v = sum(x[:n])/n
        for t in x[n:]: v = (v*(n-1)+t)/n
        return v
    atr = rma(tr, period)
    if atr==0: return 0.0
    pdi = (rma(plus_dm,period)/atr)*100
    ndi = (rma(minus_dm,period)/atr)*100
    if pdi+ndi==0: return 0.0
    dx = 100*abs(pdi-ndi)/(pdi+ndi)
    # smooth once
    return dx

def bb_percent_b(arr: List[float], length=20):
    if len(arr)<length: return 0.5
    w = arr[-length:]
    ma = sum(w)/length
    var = sum((x-ma)**2 for x in w)/length
    sd = var**0.5
    up = ma + 2*sd; lo = ma - 2*sd
    if up==lo: return 0.5
    return (arr[-1]-lo)/(up-lo)

def vote_signal(symbol: str, interval: str) -> Tuple[str, Dict[str, Any]]:
    k = klines(symbol, interval, min(200, KLINES_LIMIT))
    c = [float(x[4]) for x in k]
    h = [float(x[2]) for x in k]
    l = [float(x[3]) for x in k]
    if len(c) < 30: return "none", {}
    r = rsi(c,14)
    m_line, m_sig, _ = macd(c,12,26,9)
    e9 = ema(c,9)[-1]; e21 = ema(c,21)[-1]
    adxv = adx_val(h,l,c,14)
    bbp = bb_percent_b(c,20)
    votes_long = 0; votes_short = 0; det = {}

    # 1) RSI
    if r < 30: votes_long += 1; det["RSI"]="long"
    elif r > 70: votes_short += 1; det["RSI"]="short"
    else: det["RSI"]="neutral"

    # 2) MACD
    if m_line > m_sig: votes_long += 1; det["MACD"]="long"
    elif m_line < m_sig: votes_short += 1; det["MACD"]="short"
    else: det["MACD"]="neutral"

    # 3) EMA Cross
    if e9 > e21: votes_long += 1; det["EMA"]="long"
    elif e9 < e21: votes_short += 1; det["EMA"]="short"
    else: det["EMA"]="neutral"

    # 4) ADX as filter (trend strength)
    det["ADX"]= round(adxv,2)
    strong = adxv >= ADX_MIN

    # 5) Bollinger %B
    if bbp < 0.1: votes_long += 1; det["BB%"]="long"
    elif bbp > 0.9: votes_short += 1; det["BB%"]="short"
    else: det["BB%"]="neutral"

    total = 5
    if strong and (votes_long/total >= CONSENSUS_MIN): return "long", det
    if strong and (votes_short/total >= CONSENSUS_MIN): return "short", det
    return "none", det

# -------- Balance & sizing --------
def get_usdt_equity() -> float:
    try:
        j = get_signed("/fapi/v2/balance", {})
        for a in j:
            if a.get("asset")=="USDT":
                return float(a.get("availableBalance", a.get("balance", 0.0)))
    except Exception as e:
        log.error(f"balance: {e}")
    return 0.0

def compute_qty(symbol: str, price: float, slot_pct: float, lev: int) -> float:
    step, tick, min_not = symbol_filters(symbol)
    total = get_usdt_equity()
    if total <= 0: total = 500.0  # fallback
    alloc = total * CAPITAL_USE_PCT
    per_trade = alloc / max(1, MAX_OPEN_POS)
    usd = per_trade * slot_pct
    notional = usd * lev
    raw_qty = notional / price
    qty = round_step(raw_qty, step)
    if qty*price < min_not:
        qty = round_step((min_not/price), step)
    return max(qty, float(step))

# -------- Account setup --------
def set_isolated(symbol: str):
    try:
        post_signed("/fapi/v1/marginType", {"symbol": symbol, "marginType":"ISOLATED"})
    except requests.HTTPError as e:
        t = e.response.text
        if "-4046" in t:  # No need to change
            return
        log.warning(f"isolated {symbol}: {t}")
    except Exception as e:
        log.warning(f"isolated {symbol}: {e}")

def set_leverage(symbol: str, lev: int):
    try:
        post_signed("/fapi/v1/leverage", {"symbol": symbol, "leverage": int(lev)})
    except Exception as e:
        log.warning(f"leverage {symbol}: {e}")

# -------- Orders --------
def entry_market(symbol: str, side: str, qty: float):
    # MARKET entry (no timeInForce)
    return post_signed("/fapi/v1/order", {
        "symbol": symbol, "side": side, "type":"MARKET", "quantity": f"{qty}"
    })

def tp_partial(symbol: str, exit_side: str, qty: float, price: float):
    # TAKE_PROFIT_MARKET partial reduceOnly (no closePosition)
    return post_signed("/fapi/v1/order", {
        "symbol": symbol,
        "side": exit_side,
        "type": "TAKE_PROFIT_MARKET",
        "stopPrice": f"{price}",
        "quantity": f"{qty}",
        "reduceOnly": "true",
        "workingType": "CONTRACT_PRICE",
    })

def tp_close_all(symbol: str, exit_side: str, price: float):
    # TAKE_PROFIT_MARKET closePosition (no quantity, no reduceOnly)
    return post_signed("/fapi/v1/order", {
        "symbol": symbol,
        "side": exit_side,
        "type": "TAKE_PROFIT_MARKET",
        "stopPrice": f"{price}",
        "closePosition": "true",
        "workingType": "CONTRACT_PRICE",
    })

def sl_close_all(symbol: str, exit_side: str, price: float):
    # STOP_MARKET closePosition (no quantity, no reduceOnly)
    return post_signed("/fapi/v1/order", {
        "symbol": symbol,
        "side": exit_side,
        "type": "STOP_MARKET",
        "stopPrice": f"{price}",
        "closePosition": "true",
        "workingType": "CONTRACT_PRICE",
    })

# -------- Cooldown --------
_last_trade_min: Dict[str, int] = {}
def can_trade(symbol: str) -> bool:
    last = _last_trade_min.get(symbol, 0)
    return (int(time.time()//60) - last) >= COOLDOWN_MIN
def mark_traded(symbol: str):
    _last_trade_min[symbol] = int(time.time()//60)

# -------- Processing --------
def process_symbol(sym: str):
    if not can_trade(sym): return
    sig1, d1 = vote_signal(sym, INTERVAL)
    if sig1 == "none": return
    sig2, d2 = vote_signal(sym, INTERVAL_CONFIRM)
    if sig2 != sig1: return

    px = last_price(sym)
    step, tick, _ = symbol_filters(sym)
    # choose slot by direction
    if sig1 == "long":
        slot_pct = SLOT_A_PCT; lev = SLOT_A_LEV
        side_entry = "BUY"; side_exit = "SELL"
    else:
        slot_pct = SLOT_B_PCT; lev = SLOT_B_LEV
        side_entry = "SELL"; side_exit = "BUY"

    # account setup
    set_isolated(sym)
    set_leverage(sym, lev)

    qty = compute_qty(sym, px, slot_pct, lev)
    if qty <= 0: return

    if RUN_MODE == "paper":
        mark_traded(sym)
        tg(f"📝 PAPER {sym} {sig1} qty≈{qty} px≈{px:.6f}\n{json.dumps(d1)}")
        return

    # live path
    try:
        entry_market(sym, side_entry, qty)
    except requests.HTTPError as e:
        tg(f"⚠️ ENTRY {sym}: {e.response.text}")
        return
    except Exception as e:
        tg(f"⚠️ ENTRY {sym}: {e}")
        return

    # targets & stop
    if sig1=="long":
        tp1 = px*(1+TP1_PCT); tp2 = px*(1+TP2_PCT); tp3 = px*(1+TP3_PCT); sl = px*(1-SL_PCT)
    else:
        tp1 = px*(1-TP1_PCT); tp2 = px*(1-TP2_PCT); tp3 = px*(1-TP3_PCT); sl = px*(1+SL_PCT)

    # split qty for partials
    q1 = round_step(qty*0.4, step)
    q2 = round_step(qty*0.3, step)
    # remainder will be closed by tp3 closePosition
    try:
        if q1 > 0: tp_partial(sym, side_exit, q1, f"{tp1:.10f}")
        if q2 > 0: tp_partial(sym, side_exit, q2, f"{tp2:.10f}")
        tp_close_all(sym, side_exit, f"{tp3:.10f}")
        sl_close_all(sym, side_exit, f"{sl:.10f}")
        tg(f"✅ {sym} {sig1} qty={qty} px≈{px:.6f}\nTP1={tp1:.6f} TP2={tp2:.6f} TP3={tp3:.6f} SL={sl:.6f}")
    except requests.HTTPError as e:
        tg(f"⚠️ TP/SL {sym}: {e.response.text}")
    except Exception as e:
        tg(f"⚠️ TP/SL {sym}: {e}")

    mark_traded(sym)

# -------- Main loop --------
_last_hb = 0.0
def heartbeat():
    global _last_hb
    if time.time()-_last_hb >= TG_HEARTBEAT_MIN*60:
        _last_hb = time.time()
        tg(f"[HB] alive {int(time.time()*1000)} symbols={len(UNIVERSE)}")

def main():
    while True:
        try:
            for s in UNIVERSE:
                process_symbol(s)
                time.sleep(0.2)
            heartbeat()
        except Exception as e:
            tg(f"Loop error: {e}\n{traceback.format_exc()[:400]}")
        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    main()
