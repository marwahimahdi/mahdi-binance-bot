# -*- coding: utf-8 -*-
"""
Mahdi v5 PRO — Final
Lightweight Binance Futures bot for Render (no pandas/numpy).
(see in-file docstring for details)
"""
import os, time, hmac, hashlib, requests, math, csv, json
from datetime import datetime, timezone

def env(name, default=None, cast=str):
    v = os.getenv(name, default)
    if v is None:
        return None
    if cast is bool:
        return str(v).lower() in ("1","true","yes","y")
    if cast is int:
        try:
            return int(str(v).strip())
        except:
            return int(float(str(v).strip()))
    if cast is float:
        return float(str(v).strip())
    return str(v).strip()

API_KEY         = env("API_KEY","")
API_SECRET      = env("API_SECRET","")
BASE_HOST       = "https://testnet.binancefuture.com" if env("USE_TESTNET","false",bool) else "https://fapi.binance.com"

RUN_MODE        = env("RUN_MODE","paper")
CAPITAL_USE_PCT = env("CAPITAL_USE_PCT","0.40", float)
TOTAL_CAPITAL_PCT = env("TOTAL_CAPITAL_PCT", str(CAPITAL_USE_PCT), float)
MAX_OPEN_TRADES = env("MAX_OPEN_TRADES","6", int)
MAX_OPEN_POS    = env("MAX_OPEN_POS", str(MAX_OPEN_TRADES), int)
MAX_SYMBOLS     = env("MAX_SYMBOLS","15", int)
INTERVAL        = env("INTERVAL","5m")
INTERVAL_CONFIRM= env("INTERVAL_CONFIRM","15m")
KLINES_LIMIT    = env("KLINES_LIMIT","300", int)
SCAN_INTERVAL   = env("SCAN_INTERVAL_SEC","120", int)

SLOT_A_PCT      = env("SLOT_A_PCT","0.06", float)
SLOT_B_PCT      = env("SLOT_B_PCT","0.05", float)
SLOT_A_LEV      = env("SLOT_A_LEV","10", int)
SLOT_B_LEV      = env("SLOT_B_LEV","5", int)

TP1_PCT         = env("TP1_PCT","0.0035", float)
TP2_PCT         = env("TP2_PCT","0.0070", float)
TP3_PCT         = env("TP3_PCT","0.0120", float)
SL_PCT          = env("SL_PCT","0.0075", float)

CONSENSUS_MIN   = env("CONSENSUS_MIN","0.60", float)
ADX_MIN         = env("ADX_MIN","20", float)
COOLDOWN_MIN    = env("COOLDOWN_MIN","60", int)

TG_ENABLED      = env("TG_ENABLED","true", bool)
TELEGRAM_TOKEN  = env("TELEGRAM_TOKEN","")
TELEGRAM_CHAT_ID= env("TELEGRAM_CHAT_ID","")
TG_HEARTBEAT_MIN= env("TG_HEARTBEAT_MIN","15", int)
TG_NOTIFY_UNIVERSE = env("TG_NOTIFY_UNIVERSE","false", bool)
TG_SUMMARY_MIN  = env("TG_SUMMARY_MIN","1440", int)

MARGIN_TYPE     = env("MARGIN_TYPE","ISOLATED")
SYMBOLS_CSV     = env("SYMBOLS_CSV","universe.csv")

session = requests.Session()
if API_KEY:
    session.headers.update({"X-MBX-APIKEY": API_KEY})

def ts():
    return int(time.time()*1000)

def sign(params: dict) -> dict:
    query = "&".join([f"{k}={params[k]}" for k in sorted(params)])
    signature = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    return params

def get(path, params=None):
    url = BASE_HOST + path
    r = session.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def post(path, params=None):
    url = BASE_HOST + path
    r = session.post(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def err_text(e):
    try:
        return e.response.text
    except:
        return str(e)

def tg(msg: str):
    if not TG_ENABLED or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        session.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg[:4096]}, timeout=10)
    except Exception as exc:
        print("[TG] failed:", exc)

def load_universe(path: str):
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in f:
                s = row.strip().upper()
                if s and not s.startswith("#"):
                    if not s.endswith("USDT"):
                        s += "USDT"
                    out.append(s)
    except Exception as exc:
        print("universe load error:", exc)
    return out[:MAX_SYMBOLS]

UNIVERSE = load_universe(SYMBOLS_CSV)

_symbol_cache = {}
def fetch_symbol_info(symbol):
    if symbol in _symbol_cache:
        return _symbol_cache[symbol]
    data = get("/fapi/v1/exchangeInfo")
    for s in data["symbols"]:
        if s["symbol"] == symbol:
            _symbol_cache[symbol] = s
            return s
    raise ValueError("symbol not found: "+symbol)

def step_precision(step):
    try:
        step = float(step)
        if step == 0:
            return 8
        return max(0, int(round(-math.log(step,10))))
    except:
        return 8

def round_step(value, step):
    p = step_precision(step)
    return float(f"{value:.{p}f}")

def symbol_filters(symbol):
    info = fetch_symbol_info(symbol)
    price_filter = next(f for f in info["filters"] if f["filterType"]=="PRICE_FILTER")
    lot_filter   = next(f for f in info["filters"] if f["filterType"]=="LOT_SIZE")
    tick = price_filter["tickSize"]
    step = lot_filter["stepSize"]
    return tick, step

def position_risk(symbol):
    p = {"symbol": symbol, "timestamp": ts()}
    p = sign(p)
    return get("/fapi/v2/positionRisk", params=p)

def set_isolated(symbol):
    try:
        p = {"symbol": symbol, "marginType": MARGIN_TYPE.upper(), "timestamp": ts()}
        p = sign(p)
        post("/fapi/v1/marginType", params=p)
    except requests.HTTPError as e:
        if "-4046" in err_text(e):
            pass
        else:
            print("set_isolated:", err_text(e))

def set_leverage(symbol, lev):
    try:
        p = {"symbol": symbol, "leverage": int(lev), "timestamp": ts()}
        p = sign(p)
        post("/fapi/v1/leverage", params=p)
    except requests.HTTPError as e:
        print("leverage:", err_text(e))

def klines(symbol, interval, limit):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    return get("/fapi/v1/klines", params=params)

def ema(values, n):
    k = 2/(n+1)
    out, v_ema = [], values[0]
    for v in values:
        v_ema = v*k + v_ema*(1-k)
        out.append(v_ema)
    return out

def rsi(values, n=14):
    gains=[0]; losses=[0]
    for i in range(1,len(values)):
        ch=values[i]-values[i-1]
        gains.append(max(ch,0)); losses.append(max(-ch,0))
    def sma(a,n):
        out=[]; s=sum(a[:n]); out.append(s/n)
        for i in range(n,len(a)):
            s+=a[i]-a[i-n]; out.append(s/n)
        return out
    if len(values)<n+1: return [50.0]
    rsg=sma(gains,n); rsl=sma(losses,n)
    rs=[g/l if l>0 else 100 for g,l in zip(rsg[-len(rsl):], rsl)]
    return [100-100/(1+max(x,1e-9)) for x in rs]

def get_signal(symbol):
    try:
        data = klines(symbol, INTERVAL, 60)
        closes = [float(k[4]) for k in data]
        if len(closes) < 30: 
            return None
        e9 = ema(closes, 9)
        e21= ema(closes, 21)
        r = rsi(closes, 14)[-1]
        last = closes[-1]
        buy_score = 0; sell_score=0
        if e9[-1] > e21[-1]: buy_score += 1
        if e9[-1] < e21[-1]: sell_score += 1
        if r < 35: buy_score += 1
        if r > 65: sell_score += 1
        if last > e9[-1]: buy_score += 1
        if last < e9[-1]: sell_score += 1
        if buy_score/3.0 >= float(CONSENSUS_MIN): return ("BUY", last)
        if sell_score/3.0 >= float(CONSENSUS_MIN): return ("SELL", last)
    except Exception as exc:
        print("signal error", symbol, exc)
    return None

def new_order(params):
    params["timestamp"] = ts()
    p = sign(params)
    return post("/fapi/v1/order", params=p)

def entry_market(symbol, side, qty):
    params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty}
    return new_order(params)

def exit_tp_market(symbol, side, stop_price):
    params = {
        "symbol": symbol,
        "side": "SELL" if side=="BUY" else "BUY",
        "type": "TAKE_PROFIT_MARKET",
        "stopPrice": stop_price,
        "workingType": "CONTRACT_PRICE",
        "closePosition": True
    }
    return new_order(params)

def exit_sl_market(symbol, side, stop_price):
    params = {
        "symbol": symbol,
        "side": "SELL" if side=="BUY" else "BUY",
        "type": "STOP_MARKET",
        "stopPrice": stop_price,
        "workingType": "CONTRACT_PRICE",
        "closePosition": True
    }
    return new_order(params)

def get_usdt_balance():
    try:
        p = {"timestamp": ts()}
        p = sign(p)
        bals = get("/fapi/v2/balance", params=p)
        for b in bals:
            if b["asset"] == "USDT":
                return float(b["balance"])
    except Exception as exc:
        print("balance err", exc)
    return 0.0

def compute_qty(symbol, entry_price, slot_pct, lev):
    bal = get_usdt_balance()
    capital = bal * float(slot_pct) * float(CAPITAL_USE_PCT)
    if capital <= 0: capital = 10
    notional = capital * int(lev)
    qty = notional / entry_price
    tick, step = symbol_filters(symbol)
    qty = max(round_step(qty, step), float(step))
    return qty

_last_trade_time = {}
def can_trade(symbol):
    last = _last_trade_time.get(symbol, 0)
    return (time.time() - last) >= (int(COOLDOWN_MIN)*60)
def mark_traded(symbol):
    _last_trade_time[symbol] = time.time()

last_hb = 0
def heartbeat():
    global last_hb
    if time.time()-last_hb >= int(TG_HEARTBEAT_MIN)*60:
        last_hb = time.time()
        tg(f"[HB] alive {int(time.time()*1000)} symbols={len(UNIVERSE)}")

def process_symbol(symbol):
    sig = get_signal(symbol)
    if not sig: return
    side, price = sig
    if not can_trade(symbol): return
    set_isolated(symbol)
    lev = int(SLOT_A_LEV)
    set_leverage(symbol, lev)
    tick, step = symbol_filters(symbol)
    price = round_step(price, tick)
    qty = compute_qty(symbol, price, SLOT_A_PCT, lev)
    try:
        entry_market(symbol, side, qty)
        tg(f"🟢 ENTRY {symbol} {side} qty={qty}")
    except requests.HTTPError as e:
        tg(f"⚠️ ENTRY error {symbol}: {err_text(e)}")
        return
    if side=="BUY":
        tp1 = round_step(price*(1+float(TP1_PCT)), tick)
        tp2 = round_step(price*(1+float(TP2_PCT)), tick)
        tp3 = round_step(price*(1+float(TP3_PCT)), tick)
        sl  = round_step(price*(1-float(SL_PCT)), tick)
    else:
        tp1 = round_step(price*(1-float(TP1_PCT)), tick)
        tp2 = round_step(price*(1-float(TP2_PCT)), tick)
        tp3 = round_step(price*(1-float(TP3_PCT)), tick)
        sl  = round_step(price*(1+float(SL_PCT)), tick)
    try:
        exit_tp_market(symbol, side, tp1)
        exit_tp_market(symbol, side, tp2)
        exit_tp_market(symbol, side, tp3)
        exit_sl_market(symbol, side, sl)
        tg(f"🎯 TP/SL set {symbol} tp1={tp1} tp2={tp2} tp3={tp3} sl={sl}")
    except requests.HTTPError as e:
        tg(f"⚠️ TP/SL error {symbol}: {err_text(e)}")
    mark_traded(symbol)

def main_loop():
    tg(f"🚀 Mahdi v5 PRO — تشغيل: {RUN_MODE.upper()} | Testnet: {'On' if 'testnet' in BASE_HOST else 'Off'}")
    tg(f"Symbols (n={len(UNIVERSE)}): {', '.join(UNIVERSE)}")
    while True:
        try:
            for s in UNIVERSE:
                process_symbol(s)
                time.sleep(0.2)
            heartbeat()
        except Exception as exc:
            tg(f"Loop error: {exc}")
        time.sleep(int(SCAN_INTERVAL))

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        tg("Bot stopped by user.")
