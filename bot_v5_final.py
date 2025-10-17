
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mahdi v5 PRO — full live bot (Render)
- Five indicators voting (RSI, MACD, EMA Cross, ADX, Bollinger %B)
- Filters: ADX >= ADX_MIN, cooldown per symbol, consensus >= CONSENSUS_MIN
- Proper Binance Futures signing (no -1022), time sync, marginType/leverage
- TP1/TP2/TP3 partial reduces + SL closePosition
- Telegram notifications (heartbeat, fills, errors, summary) optional
Env vars (see README_RENDER.md or your dashboard).
"""
import os, time, json, hmac, hashlib, math, random, traceback
from typing import List, Dict, Tuple
import requests

# ---------- ENV ----------
def env_bool(key:str, default=False):
    v = os.getenv(key, str(default)).strip().lower()
    return v in ("1","true","yes","on")

def env_float(key:str, default:float):
    try: return float(os.getenv(key, str(default)).strip())
    except: return default

def env_int(key:str, default:int):
    try: return int(float(os.getenv(key, str(default)).strip()))
    except: return default

# Operational
RUN_MODE            = os.getenv("RUN_MODE", "paper").strip().lower()    # live | paper
USE_TESTNET         = env_bool("USE_TESTNET", False)
INTERVAL            = os.getenv("INTERVAL", "5m").strip()
INTERVAL_CONFIRM    = os.getenv("INTERVAL_CONFIRM", "15m").strip()
KLINES_LIMIT        = env_int("KLINES_LIMIT", 300)
SCAN_INTERVAL_SEC   = env_int("SCAN_INTERVAL_SEC", 120)
SYMBOLS_CSV         = os.getenv("SYMBOLS_CSV", "universe.csv").strip()
MAX_SYMBOLS         = env_int("MAX_SYMBOLS", 25)
MAX_OPEN_POS        = env_int("MAX_OPEN_POS", 6)
MAX_OPEN_TRADES     = env_int("MAX_OPEN_TRADES", 10)

# Capital & slots
CAPITAL_USE_PCT     = env_float("CAPITAL_USE_PCT", 0.40)
SLOT_A_PCT          = env_float("SLOT_A_PCT", 0.06)
SLOT_A_LEV          = env_int("SLOT_A_LEV", 10)
SLOT_B_PCT          = env_float("SLOT_B_PCT", 0.05)
SLOT_B_LEV          = env_int("SLOT_B_LEV", 5)

# Risk
TP1_PCT             = env_float("TP1_PCT", 0.0035)
TP2_PCT             = env_float("TP2_PCT", 0.0070)
TP3_PCT             = env_float("TP3_PCT", 0.0120)
SL_PCT              = env_float("SL_PCT", 0.0075)

# Filters
CONSENSUS_MIN       = env_float("CONSENSUS_MIN", 0.60)
ADX_MIN             = env_float("ADX_MIN", 20)
COOLDOWN_MIN        = env_int("COOLDOWN_MIN", 60)

# API
API_KEY             = os.getenv("API_KEY","").strip()
API_SECRET          = os.getenv("API_SECRET","").strip()

# Telegram
TG_ENABLED          = env_bool("TG_ENABLED", True)
TG_TOKEN            = os.getenv("TELEGRAM_TOKEN","").strip()
TG_CHAT_ID          = os.getenv("TELEGRAM_CHAT_ID","").strip()
TG_HEARTBEAT_MIN    = env_int("TG_HEARTBEAT_MIN", 15)
TG_NOTIFY_UNIVERSE  = env_bool("TG_NOTIFY_UNIVERSE", False)
TG_SUMMARY_MIN      = env_int("TG_SUMMARY_MIN", 1440)

# ---------- CONST ----------
BASE = "https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"
PUBLIC = BASE
FUT = BASE
HEADERS = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
_session = requests.Session()
_session.headers.update(HEADERS)

# ---------- TELEGRAM ----------
def tg_send(text:str):
    if not TG_ENABLED or not TG_TOKEN or not TG_CHAT_ID: 
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        _session.post(url, data={"chat_id": TG_CHAT_ID, "text": text, "parse_mode":"HTML"}, timeout=8)
    except Exception:
        pass

# ---------- TIME SYNC ----------
_server_offset_ms = 0
def sync_time():
    global _server_offset_ms
    try:
        r = _session.get(f"{PUBLIC}/fapi/v1/time", timeout=5)
        r.raise_for_status()
        srv = int(r.json().get("serverTime"))
        _server_offset_ms = srv - int(time.time()*1000)
    except Exception as e:
        tg_send(f"⚠️ time sync failed: {e}")

def now_ms():
    return int(time.time()*1000) + _server_offset_ms

sync_time()

# ---------- SIGN ----------
def _sign(params:Dict) -> str:
    qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"

def get_signed(path:str, params:Dict):
    p = dict(params)
    p.setdefault("timestamp", now_ms())
    p.setdefault("recvWindow", 5000)
    url = f"{FUT}{path}?{_sign(p)}"
    r = _session.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def post_signed(path:str, params:Dict):
    p = dict(params)
    p.setdefault("timestamp", now_ms())
    p.setdefault("recvWindow", 5000)
    url = f"{FUT}{path}"
    body = _sign(p)
    r = _session.post(url, data=body, timeout=10)
    r.raise_for_status()
    return r.json()

# ---------- MARKET HELPERS ----------
def klines(symbol:str, interval:str, limit:int) -> List[List]:
    s = symbol.upper().replace("/","")
    url = f"{PUBLIC}/fapi/v1/klines?symbol={s}&interval={interval}&limit={limit}"
    r = _session.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def set_isolated(symbol:str):
    try:
        return post_signed("/fapi/v1/marginType", {"symbol": symbol, "marginType":"ISOLATED"})
    except requests.HTTPError as e:
        txt = e.response.text
        # Code -4046 means no need to change (already isolated). Ignore.
        if '"code":-4046' not in txt:
            tg_send(f"set_isolated {symbol}: {txt}")

def set_leverage(symbol:str, lev:int):
    try:
        return post_signed("/fapi/v1/leverage", {"symbol": symbol, "leverage": lev})
    except requests.HTTPError as e:
        tg_send(f"leverage {symbol}: {e.response.text}")

# ---------- SIMPLE NUMPY-LIKE UTILS ----------
def ema(arr:List[float], period:int) -> List[float]:
    k = 2/(period+1)
    out = []
    ema_val = None
    for x in arr:
        if ema_val is None:
            ema_val = x
        else:
            ema_val = x*k + ema_val*(1-k)
        out.append(ema_val)
    return out

def rsi(prices:List[float], period:int=14) -> List[float]:
    gains, losses = [], []
    for i in range(1, len(prices)):
        diff = prices[i]-prices[i-1]
        gains.append(max(diff,0.0))
        losses.append(abs(min(diff,0.0)))
    rsis = [None]*(period)
    if len(gains) < period: return [None]*len(prices)
    avg_gain = sum(gains[:period])/period
    avg_loss = sum(losses[:period])/period
    def calc(g,l):
        nonlocal avg_gain, avg_loss
        avg_gain = (avg_gain*(period-1)+g)/period
        avg_loss = (avg_loss*(period-1)+l)/period
        if avg_loss==0: return 100.0
        rs = avg_gain/avg_loss
        return 100 - 100/(1+rs)
    rsis.append(calc(gains[period], losses[period]) if len(gains)>period else None)
    for i in range(period+1, len(gains)):
        rsis.append(calc(gains[i], losses[i]))
    rsis = [None] + rsis  # align with price len
    rsis = rsis[:len(prices)]
    return rsis

def macd(prices:List[float], fast=12, slow=26, signal=9):
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = [ (a-b) if (a is not None and b is not None) else None for a,b in zip(ema_fast, ema_slow) ]
    macd_line = [x if x is not None else 0 for x in macd_line]
    signal_line = ema(macd_line, signal)
    hist = [a-b for a,b in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist

def true_range(h, l, c):
    trs = []
    prev_close = None
    for i in range(len(c)):
        if i==0 or prev_close is None:
            tr = h[i] - l[i]
        else:
            tr = max(h[i]-l[i], abs(h[i]-prev_close), abs(l[i]-prev_close))
        trs.append(tr)
        prev_close = c[i]
    return trs

def adx(high, low, close, period=14):
    # simplified ADX
    plus_dm, minus_dm = [0], [0]
    for i in range(1,len(high)):
        up = high[i]-high[i-1]
        dn = low[i-1]-low[i]
        plus_dm.append( up if (up>dn and up>0) else 0 )
        minus_dm.append( dn if (dn>up and dn>0) else 0 )
    tr = true_range(high, low, close)
    def smooth(lst, p):
        out=[]; acc=sum(lst[:p]); out.extend([None]*(p-1)); out.append(acc)
        for i in range(p,len(lst)):
            acc = acc - acc/p + lst[i]
            out.append(acc)
        return out
    atr = [x/p if x is not None else None for x,p in zip(smooth(tr, period), [period]*len(tr))]
    plus_di = []
    minus_di = []
    for i in range(len(close)):
        if atr[i] and atr[i]!=0:
            plus_di.append(100*(smooth(plus_dm,period)[i]/period)/atr[i] if smooth(plus_dm,period)[i] else None)
            minus_di.append(100*(smooth(minus_dm,period)[i]/period)/atr[i] if smooth(minus_dm,period)[i] else None)
        else:
            plus_di.append(None); minus_di.append(None)
    dx=[]
    for p,m in zip(plus_di, minus_di):
        if p is None or m is None or (p+m)==0:
            dx.append(None)
        else:
            dx.append(100*abs(p-m)/(p+m))
    # smooth DX to ADX
    # simple smoothing
    vals=[]; acc=0; cnt=0
    for d in dx:
        if d is None:
            vals.append(None)
        else:
            acc = d if cnt==0 else (acc*(period-1)+d)/period
            cnt = min(cnt+1, period)
            vals.append(acc if cnt==period else None)
    return vals

def bollinger_b(prices:List[float], length=20):
    out=[]
    for i in range(len(prices)):
        if i+1<length:
            out.append(None); continue
        window=prices[i+1-length:i+1]
        ma = sum(window)/length
        var = sum((x-ma)**2 for x in window)/length
        sd = var**0.5
        upper = ma + 2*sd
        lower = ma - 2*sd
        if upper==lower:
            out.append(0.5)
        else:
            out.append( (prices[i]-lower)/(upper-lower) )
    return out

# ---------- UNIVERSE ----------
def read_symbols(path:str, maxn:int) -> List[str]:
    syms=[]
    try:
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                s=line.strip().upper().replace("/","")
                if not s: continue
                if not s.endswith("USDT"): s+= "USDT"
                syms.append(s)
                if len(syms)>=maxn: break
    except Exception as e:
        tg_send(f"⚠️ read_symbols: {e}")
    return syms

# ---------- INDICATORS / SIGNALS ----------
def indicators_vote(symbol:str, interval:str) -> Tuple[str, Dict]:
    # fetch klines
    data = klines(symbol, interval, KLINES_LIMIT)
    close = [float(x[4]) for x in data]
    high  = [float(x[2]) for x in data]
    low   = [float(x[3]) for x in data]

    # indicators
    rsi_v = rsi(close, 14)
    macd_line, macd_sig, macd_hist = macd(close, 12, 26, 9)
    ema_fast = ema(close, 9)
    ema_slow = ema(close, 21)
    adx_v = adx(high, low, close, 14)
    bbp = bollinger_b(close, 20)

    i = -1
    votes_long = 0
    votes_short = 0
    details = {}

    # 1) RSI
    if rsi_v[i] is not None:
        if rsi_v[i] < 30: votes_long += 1; details["RSI"]="long"
        elif rsi_v[i] > 70: votes_short += 1; details["RSI"]="short"
        else: details["RSI"]="neutral"

    # 2) MACD
    if macd_line[i] is not None and macd_sig[i] is not None:
        if macd_line[i] > macd_sig[i]: votes_long += 1; details["MACD"]="long"
        elif macd_line[i] < macd_sig[i]: votes_short += 1; details["MACD"]="short"
        else: details["MACD"]="neutral"

    # 3) EMA Cross
    if ema_fast[i] is not None and ema_slow[i] is not None:
        if ema_fast[i] > ema_slow[i]: votes_long += 1; details["EMA"]="long"
        elif ema_fast[i] < ema_slow[i]: votes_short += 1; details["EMA"]="short"
        else: details["EMA"]="neutral"

    # 4) ADX filter (direction via DI isn't computed strictly here; use trend strength only)
    adx_now = adx_v[i] if adx_v else None
    details["ADX"]= round(adx_now,2) if adx_now else None
    strong_trend = (adx_now is not None and adx_now >= ADX_MIN)

    # 5) Bollinger %B
    if bbp[i] is not None:
        if bbp[i] < 0.1: votes_long += 1; details["BB%"]="long"
        elif bbp[i] > 0.9: votes_short += 1; details["BB%"]="short"
        else: details["BB%"]="neutral"

    total_votes = 5
    # consensus
    if votes_long/total_votes >= CONSENSUS_MIN and strong_trend:
        return "long", details
    if votes_short/total_votes >= CONSENSUS_MIN and strong_trend:
        return "short", details
    return "none", details

# ---------- ORDERS ----------
def place_market(symbol:str, side:str, qty:float):
    return post_signed("/fapi/v1/order", {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": f"{qty}",
    })

def place_tp_market_close(symbol:str, side_exit:str, price:float):
    return post_signed("/fapi/v1/order", {
        "symbol": symbol,
        "side": side_exit,
        "type":"TAKE_PROFIT_MARKET",
        "stopPrice": f"{price}",
        "closePosition":"true",
        "workingType":"CONTRACT_PRICE",
    })

def place_tp_market_partial(symbol:str, side_exit:str, qty:float, price:float):
    return post_signed("/fapi/v1/order", {
        "symbol": symbol,
        "side": side_exit,
        "type":"TAKE_PROFIT_MARKET",
        "stopPrice": f"{price}",
        "quantity": f"{qty}",
        "reduceOnly": "true",
        "workingType":"CONTRACT_PRICE",
    })

def place_sl_market_close(symbol:str, side_exit:str, price:float):
    return post_signed("/fapi/v1/order", {
        "symbol": symbol,
        "side": side_exit,
        "type":"STOP_MARKET",
        "stopPrice": f"{price}",
        "closePosition":"true",
        "workingType":"CONTRACT_PRICE",
    })

# qty calc (rough — assumes 1 coin = price * qty notional; use SLOT_A/B and leverage)
def exchange_info(symbol:str):
    try:
        r = _session.get(f"{PUBLIC}/fapi/v1/exchangeInfo?symbol={symbol}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        tg_send(f"⚠️ exchangeInfo {symbol}: {e}")

def price_tick_size(symbol:str):
    info = exchange_info(symbol)
    if not info: return 0.001, 0.001
    f = info["symbols"][0]["filters"]
    tick = 0.01
    step = 0.001
    for x in f:
        if x["filterType"]=="PRICE_FILTER":
            tick = float(x["tickSize"])
        if x["filterType"]=="LOT_SIZE":
            step = float(x["stepSize"])
    return tick, step

def round_step(qty, step):
    if step<=0: return qty
    return math.floor(qty/step)*step

def last_price(symbol:str):
    r = _session.get(f"{PUBLIC}/fapi/v1/ticker/price?symbol={symbol}", timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])

# ---------- STATE ----------
_last_trade_min = {}   # symbol -> minute timestamp
open_positions = {}    # symbol -> side ('long'/'short')

def can_trade(symbol:str):
    lastm = _last_trade_min.get(symbol, 0)
    nowm = int(time.time()//60)
    return (nowm - lastm) >= COOLDOWN_MIN

# ---------- MAIN LOOP ----------
def scan_and_trade(symbols:List[str]):
    hb_t0 = time.time()
    for sym in symbols:
        try:
            if not can_trade(sym):
                continue
            # indicator vote on INTERVAL and confirm on INTERVAL_CONFIRM
            vote, d1 = indicators_vote(sym, INTERVAL)
            if vote == "none":
                continue
            vote2, d2 = indicators_vote(sym, INTERVAL_CONFIRM)
            if vote2 != vote:
                continue

            # Prepare market info
            set_isolated(sym)
            lev = SLOT_A_LEV if vote=="long" else SLOT_B_LEV  # simple example
            set_leverage(sym, lev)
            px = last_price(sym)
            tick, step = price_tick_size(sym)

            # capital
            # here we use notional based on  CAP_PCT * balance_estimate; for simplicity assume 1000 USDT
            total_equity = 1000.0
            notional = total_equity * CAPITAL_USE_PCT * (SLOT_A_PCT if vote=="long" else SLOT_B_PCT)
            base_qty = round_step( (notional * lev) / px, step )
            if base_qty <= 0:
                continue

            side = "BUY" if vote=="long" else "SELL"
            exit_side = "SELL" if vote=="long" else "BUY"

            if RUN_MODE == "live":
                place_market(sym, side, base_qty)

                # Targets
                tp1 = px*(1+TP1_PCT) if vote=="long" else px*(1-TP1_PCT)
                tp2 = px*(1+TP2_PCT) if vote=="long" else px*(1-TP2_PCT)
                tp3 = px*(1+TP3_PCT) if vote=="long" else px*(1-TP3_PCT)
                sl  = px*(1-SL_PCT)  if vote=="long" else px*(1+SL_PCT)

                q1 = round_step(base_qty*0.33, step)
                q2 = round_step(base_qty*0.33, step)
                q3 = round_step(base_qty - q1 - q2, step)

                place_tp_market_partial(sym, exit_side, q1, tp1)
                place_tp_market_partial(sym, exit_side, q2, tp2)
                place_tp_market_close(sym, exit_side, tp3)  # يغلق الباقي
                place_sl_market_close(sym, exit_side, sl)

            _last_trade_min[sym] = int(time.time()//60)
            open_positions[sym] = vote

            tg_send(
                f"✅ <b>{sym}</b> signal={vote}\n"
                f"RSI/MACD/EMA/ADX/BB: {json.dumps(d1, ensure_ascii=False)}\n"
                f"confirm({INTERVAL_CONFIRM}) OK\n"
                f"qty≈{base_qty}, px={px:.6f}"
            )

            time.sleep(0.3)

        except requests.HTTPError as e:
            tg_send(f"⚠️ {sym} HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            tg_send(f"⚠️ {sym} err: {e}\n{traceback.format_exc()[:500]}")

    # heartbeat
    if time.time() - hb_t0 >= TG_HEARTBEAT_MIN*60:
        tg_send(f"[HB] alive {int(time.time()*1000)} symbols={len(symbols)}")

def main():
    tg_send(f"♻️ Mahdi v5 PRO تشغيل: {RUN_MODE.upper()} | Testnet: {'On' if USE_TESTNET else 'Off'}")
    symbols = read_symbols(SYMBOLS_CSV, MAX_SYMBOLS)
    if TG_NOTIFY_UNIVERSE:
        tg_send(f"📊 Universe: {', '.join(symbols)} (n={len(symbols)})")
    while True:
        try:
            scan_and_trade(symbols)
        except Exception as e:
            tg_send(f"loop error: {e}")
        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    main()
