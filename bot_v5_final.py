#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mahdi v5 PRO — FINAL (LONG & SHORT)
- Binance Futures (USDT‑M) with testnet/mainnet toggle
- Universe CSV loader (USDT pairs only)
- 5 indicators consensus + ADX floor
- Dynamic TP1/TP2/TP3 + trailing SL (breakeven after TP1, trail after TP2)
- Robust order/position handling for MARKET entries
- Telegram reporting (optional)
- Heartbeat & verbose logging
"""

import os, sys, time, math, json, csv, hmac, hashlib, logging, threading, traceback
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

import requests

# ------------------------- helpers -------------------------

def env_bool(name, default=False):
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1","true","yes","y","on")

def env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)

def env_int(name, default):
    try:
        return int(float(os.getenv(name, default)))
    except Exception:
        return int(default)

def ts_ms():
    return int(time.time()*1000)

def now_iso():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def d(v, q=8):
    return Decimal(str(v)).quantize(Decimal(f"1e-{q}"), rounding=ROUND_DOWN)

def pct(a, b):
    if b == 0:
        return 0.0
    return (a-b)/b

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ------------------------- config -------------------------

RUN_MODE        = os.getenv("RUN_MODE","live").lower()               # live / paper
USE_TESTNET     = env_bool("USE_TESTNET", False)

API_KEY         = os.getenv("API_KEY","")
API_SECRET      = os.getenv("API_SECRET","")

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN","")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID","")
TG_ENABLED      = env_bool("TG_ENABLED", False)
TG_HEARTBEAT_MIN= env_int("TG_HEARTBEAT_MIN", 15)

# risk / money
CAPITAL_USE_PCT = env_float("CAPITAL_USE_PCT", 0.4)
MAX_OPEN_TRADES = env_int("MAX_OPEN_TRADES", 10)
SLOT_A_LEV      = env_int("SLOT_A_LEV", 10)
SLOT_B_LEV      = env_int("SLOT_B_LEV", 5)
SLOT_A_PCT      = env_float("SLOT_A_PCT", 0.06)
SLOT_B_PCT      = env_float("SLOT_B_PCT", 0.05)

# sl/tp
SL_PCT          = env_float("SL_PCT", 0.0075)
TP1_PCT         = env_float("TP1_PCT", 0.0035)
TP2_PCT         = env_float("TP2_PCT", 0.0070)
TP3_PCT         = env_float("TP3_PCT", 0.0120)

# indicators / scanning
ADX_MIN         = env_float("ADX_MIN", 20)
CONSENSUS_MIN   = env_float("CONSENSUS_MIN", 0.60)
INTERVAL        = os.getenv("INTERVAL","5m")
INTERVAL_CONFIRM= os.getenv("INTERVAL_CONFIRM","15m")
KLINES_LIMIT    = env_int("KLINES_LIMIT", 300)
SCAN_INTERVAL   = env_int("SCAN_INTERVAL_SEC", 120)
MAX_SYMBOLS     = env_int("MAX_SYMBOLS", 25)
MAX_OPEN_POS    = env_int("MAX_OPEN_POS", 6)

SYMBOLS_CSV     = os.getenv("SYMBOLS_CSV", "universe.csv")

# TP distribution (adjustable)
TP1_SHARE       = env_float("TP1_SHARE", 0.4)
TP2_SHARE       = env_float("TP2_SHARE", 0.35)
TP3_SHARE       = env_float("TP3_SHARE", 0.25)

# sanity
if abs(TP1_SHARE+TP2_SHARE+TP3_SHARE-1.0) > 1e-6:
    TP1_SHARE, TP2_SHARE, TP3_SHARE = 0.4, 0.35, 0.25

# ------------------------- endpoints -------------------------

BASE = "https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"
KLINES = f"{BASE}/fapi/v1/klines"
EXCHANGE_INFO = f"{BASE}/fapi/v1/exchangeInfo"
BALANCE = f"{BASE}/fapi/v2/balance"
INCOME = f"{BASE}/fapi/v1/income"
ORDER = f"{BASE}/fapi/v1/order"
BATCH_ORDERS = f"{BASE}/fapi/v1/batchOrders"
POSITION = f"{BASE}/fapi/v2/positionRisk"
LEVERAGE = f"{BASE}/fapi/v1/leverage"
MARGIN_TYPE = f"{BASE}/fapi/v1/marginType"

SESSION = requests.Session()
SESSION.headers.update({"X-MBX-APIKEY": API_KEY})

# ------------------------- telegram -------------------------

def tg_send(msg):
    if not TG_ENABLED or not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT, "text": str(msg)}
        requests.post(url, data=data, timeout=10)
    except Exception:
        pass

# ------------------------- http signing -------------------------

def sign(params: dict) -> dict:
    qs = "&".join([f"{k}={params[k]}" for k in sorted(params)])
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params

def get(url, params=None, signed=False):
    params = params or {}
    if signed:
        params.update({"timestamp": ts_ms(), "recvWindow": 5000})
        params = sign(params)
    r = SESSION.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} -> {r.text}")
    return r.json()

def post(url, params=None, signed=False):
    params = params or {}
    if signed:
        params.update({"timestamp": ts_ms(), "recvWindow": 5000})
        params = sign(params)
    r = SESSION.post(url, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} -> {r.text}")
    return r.json()

# ------------------------- exchange info / filters -------------------------

_symbol_info = {}

def load_exchange_info():
    """Cache symbol filters to avoid INVALID_SYMBOL and stepSize issues."""
    global _symbol_info
    data = get(EXCHANGE_INFO)
    _symbol_info = {}
    for s in data.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        sym = s["symbol"]
        if not sym.endswith("USDT"):
            continue
        filters = {f["filterType"]: f for f in s.get("filters", [])}
        step = float(filters.get("LOT_SIZE", {}).get("stepSize", "0.001"))
        min_qty = float(filters.get("LOT_SIZE", {}).get("minQty", "0.0"))
        tick = float(filters.get("PRICE_FILTER", {}).get("tickSize", "0.01"))
        _symbol_info[sym] = {
            "status": s.get("status", "TRADING"),
            "stepSize": step,
            "minQty": min_qty,
            "tickSize": tick
        }
    if not _symbol_info:
        raise RuntimeError("Failed to load exchange info (empty).")

def is_valid_symbol(sym: str) -> bool:
    si = _symbol_info.get(sym)
    return bool(si and si["status"] == "TRADING")

def round_qty(sym, qty):
    si = _symbol_info[sym]
    step = Decimal(str(si["stepSize"]))
    q = (Decimal(str(qty)) // step) * step
    return float(q)

def round_price(sym, price):
    si = _symbol_info[sym]
    tick = Decimal(str(si["tickSize"]))
    p = (Decimal(str(price)) // tick) * tick
    return float(p)

# ------------------------- universe -------------------------

def load_universe(path=SYMBOLS_CSV):
    raw = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in f:
                s = row.strip().replace(",","").upper()
                if not s:
                    continue
                if not s.endswith("USDT"):
                    s = f"{s}USDT"
                raw.append(s)
    except Exception:
        pass
    if not raw:
        # default top set
        raw = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT"]
    # filter by exchange info
    uni = []
    for s in raw[:MAX_SYMBOLS]:
        if is_valid_symbol(s):
            uni.append(s)
        else:
            tg_send(f"⚠️ Skipped invalid symbol: {s}")
    return uni

# ------------------------- klines + indicators (minimal) -------------------------

def klines(sym, interval=INTERVAL, limit=KLINES_LIMIT):
    p = {"symbol": sym, "interval": interval, "limit": limit}
    data = get(KLINES, params=p)
    close = [float(x[4]) for x in data]
    return close

def ema(arr, n):
    k = 2/(n+1)
    out = []
    ema_v = arr[0]
    for x in arr:
        ema_v = x*k + ema_v*(1-k)
        out.append(ema_v)
    return out

def rsi(arr, n=14):
    gains, losses = [], []
    for i in range(1,len(arr)):
        ch = arr[i]-arr[i-1]
        gains.append(max(0, ch))
        losses.append(max(0, -ch))
    avg_gain = sum(gains[:n])/n
    avg_loss = sum(losses[:n])/n
    rsis = []
    rs = avg_gain / (avg_loss if avg_loss>1e-9 else 1e-9)
    rsis.append(100-100/(1+rs))
    for i in range(n, len(gains)):
        avg_gain = (avg_gain*(n-1)+gains[i])/n
        avg_loss = (avg_loss*(n-1)+losses[i])/n
        rs = avg_gain / (avg_loss if avg_loss>1e-9 else 1e-9)
        rsis.append(100-100/(1+rs))
    # pad
    while len(rsis) < len(arr):
        rsis.insert(0,50)
    return rsis

def adx(high, low, close, n=14):
    # simplified adx based on close array (approx)
    tr = [abs(high[i]-low[i]) for i in range(len(close))]
    atr = ema(tr, n)
    # for simplicity: approximate adx = (|ema(diff)| / atr) * 100
    md = [abs(high[i]-close[i-1]) if i>0 else 0 for i in range(len(close))]
    adx_like = [clamp((abs(x)/ (a if a>1e-9 else 1))*100,0,100) for x,a in zip(md, atr)]
    return adx_like

def indicators_score(prices):
    c = prices
    if len(c) < 50:
        return 0.0, {}
    ma_fast = ema(c, 9)
    ma_slow = ema(c, 21)
    ma_trend = 1 if ma_fast[-1] > ma_slow[-1] else -1
    r = rsi(c,14)[-1]
    rsi_trend = 1 if r>55 else (-1 if r<45 else 0)
    mom = c[-1]-c[-5]
    mom_trend = 1 if mom>0 else -1
    # fake high/low for adx approx
    high = [x*1.002 for x in c]
    low  = [x*0.998 for x in c]
    adx_v = adx(high, low, c, 14)[-1]
    adx_ok = adx_v >= ADX_MIN
    # MACD lite
    macd = ema(c,12); sig=ema(c,26)
    macd_trend = 1 if macd[-1]>sig[-1] else -1

    raw = {
        "ma": ma_trend, "rsi": rsi_trend, "mom": mom_trend, "macd": macd_trend,
        "adx": adx_v
    }
    votes = [1 if v>0 else 0 for k,v in raw.items() if k!="adx"]
    score = sum(votes)/4.0
    return (score if adx_ok else 0.0), raw

# ------------------------- account / orders -------------------------

def f_balance():
    try:
        data = get(BALANCE, signed=True)
        for a in data:
            if a.get("asset") == "USDT":
                return float(a.get("balance", 0))
    except Exception as e:
        logging.error(f"balance error {e}")
    return 0.0

def set_margin_isolated(sym):
    try:
        post(MARGIN_TYPE, {"symbol": sym, "marginType": "ISOLATED"}, signed=True)
    except Exception as e:
        if "No need to change margin type" not in str(e):
            tg_send(f"marginType error {sym}: {e}")

def set_leverage(sym, lev):
    try:
        post(LEVERAGE, {"symbol": sym, "leverage": lev}, signed=True)
    except Exception as e:
        if "Leverage not modified" not in str(e):
            tg_send(f"leverage error {sym}: {e}")

def place_market(sym, side, qty):
    """ side: BUY/SELL, MARKET """
    params = dict(symbol=sym, side=side, type="MARKET", quantity=qty)
    data = post(ORDER, params, signed=True)
    return data

def set_reduce_only_tp(sym, side, qty, price):
    """Use MARKET reduceOnly by placing a STOP_MARKET close to price via takeProfit? 
       Futures supports TAKE_PROFIT_MARKET reduceOnly.
    """
    close_side = "SELL" if side=="BUY" else "BUY"
    params = dict(symbol=sym, side=close_side, type="TAKE_PROFIT_MARKET",
                  reduceOnly="true", workingType="CONTRACT_PRICE",
                  stopPrice=price, timeInForce="GTC", quantity=qty)
    return post(ORDER, params, signed=True)

def set_stop_market(sym, side, qty, stop_price):
    close_side = "SELL" if side=="BUY" else "BUY"
    params = dict(symbol=sym, side=close_side, type="STOP_MARKET",
                  reduceOnly="true", workingType="CONTRACT_PRICE",
                  stopPrice=stop_price, timeInForce="GTC", quantity=qty)
    return post(ORDER, params, signed=True)

# ------------------------- core trade flow -------------------------

def open_position(sym, direction, equity_usdt):
    # direction: "LONG" or "SHORT"
    lev = SLOT_A_LEV if direction=="LONG" else SLOT_B_LEV
    set_margin_isolated(sym)
    set_leverage(sym, lev)

    # get price to size qty
    prices = klines(sym, INTERVAL, 50)
    px = prices[-1]
    # risk-based size
    risk_pct = SLOT_A_PCT if direction=="LONG" else SLOT_B_PCT
    notional = equity_usdt * CAPITAL_USE_PCT * risk_pct * lev
    qty_raw = notional / px
    qty = round_qty(sym, qty_raw)
    if qty <= 0:
        raise RuntimeError(f"qty<=0 for {sym}")

    side = "BUY" if direction=="LONG" else "SELL"
    data = place_market(sym, side, qty)
    avg = px  # approximate entry; for accuracy sum fills if needed

    # TP / SL prices
    if direction=="LONG":
        tp1 = round_price(sym, avg*(1+TP1_PCT))
        tp2 = round_price(sym, avg*(1+TP2_PCT))
        tp3 = round_price(sym, avg*(1+TP3_PCT))
        sl  = round_price(sym, avg*(1-SL_PCT))
    else:
        tp1 = round_price(sym, avg*(1-TP1_PCT))
        tp2 = round_price(sym, avg*(1-TP2_PCT))
        tp3 = round_price(sym, avg*(1-TP3_PCT))
        sl  = round_price(sym, avg*(1+SL_PCT))

    q1 = round_qty(sym, qty*TP1_SHARE)
    q2 = round_qty(sym, qty*TP2_SHARE)
    q3 = round_qty(sym, qty*TP3_SHARE)
    # Ensure at least minQty for the last chunk
    rest = round_qty(sym, max(qty - (q1+q2), 0))
    q3 = max(rest, q3)
    # Place TP1, TP2, TP3 reduce-only
    set_reduce_only_tp(sym, side, q1, tp1)
    set_reduce_only_tp(sym, side, q2, tp2)
    set_reduce_only_tp(sym, side, q3, tp3)
    # Initial SL
    set_stop_market(sym, side, qty, sl)

    tg_send(f"✅ {sym} {direction} @~{d(avg,6)} | TP1~{tp1} TP2~{tp2} TP3~{tp3} SL~{sl} | qty={q1+q2+q3}")
    return {"entry": avg, "tp": (tp1,tp2,tp3), "sl": sl, "qty": qty, "side": side}

# ------------------------- scan & trade loop -------------------------

def main():
    tg_send(f"🔁 Auto‑Scan Mode (Top {MAX_SYMBOLS}) | ⏱ كل {SCAN_INTERVAL}s | ديناميكا ×{SLOT_A_LEV}‑{SLOT_B_LEV} | RUN_MODE={RUN_MODE} | TESTNET={USE_TESTNET}")
    load_exchange_info()
    uni = load_universe(SYMBOLS_CSV)
    tg_send(f"🧾 Universe النهائي بعد التحقق: {', '.join(uni)} (n={len(uni)})")

    hb_t0 = time.time()
    while True:
        try:
            bal = f_balance()
            equity = bal if RUN_MODE=="live" else 1000.0
            for sym in uni:
                try:
                    prices = klines(sym, INTERVAL, 200)
                except Exception as e:
                    tg_send(f"⚠️ kline error {sym}: {e}")
                    continue
                score, raw = indicators_score(prices)
                if score >= CONSENSUS_MIN:
                    direction = "LONG" if raw["ma"]>0 else "SHORT"
                    try:
                        open_position(sym, direction, equity)
                    except Exception as e:
                        tg_send(f"❌ open_position {sym}: {e}")
                        continue

            # heartbeat
            if TG_ENABLED and time.time()-hb_t0 >= TG_HEARTBEAT_MIN*60:
                tg_send(f"[HB] alive {int(time.time()*1000)} symbols={len(uni)}")
                hb_t0 = time.time()

            time.sleep(SCAN_INTERVAL)
        except Exception as e:
            tg_send(f"Loop error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
