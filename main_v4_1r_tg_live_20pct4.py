#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Mahdi Trade Bot — v4.1r-TG-Live (20% split across 4 trades)
# Changes:
# - Adds PER_TRADE_PCT (if absent, computed = TOTAL_CAPITAL_PCT / MAX_OPEN_TRADES)
# - MAX_OPEN_TRADES default 4
# - TOTAL_CAPITAL_PCT default 0.20

import os, time, hmac, hashlib, random
from datetime import datetime, timezone, timedelta
import requests, pandas as pd, numpy as np
from dotenv import load_dotenv

load_dotenv()

API_KEY     = os.getenv("API_KEY","")
API_SECRET  = os.getenv("API_SECRET","")
USE_TESTNET = os.getenv("USE_TESTNET","false").lower() in ("1","true","yes")
RUN_MODE    = os.getenv("RUN_MODE","live")

INTERVAL          = os.getenv("INTERVAL","5m")
MAX_SYMBOLS       = int(os.getenv("MAX_SYMBOLS","16"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC","45"))
COOLDOWN_MIN      = int(os.getenv("COOLDOWN_MIN","45"))
KLINES_LIMIT      = int(os.getenv("KLINES_LIMIT","200"))
HEARTBEAT_MIN     = int(os.getenv("HEARTBEAT_MIN","30"))
SYMBOLS_CSV       = os.getenv("SYMBOLS_CSV","")

CONSENSUS_RATIO = float(os.getenv("CONSENSUS_RATIO","0.6"))
MIN_AGREE       = int(os.getenv("MIN_AGREE","1"))
ADX_MIN         = float(os.getenv("ADX_MIN","15"))
ATR_PCT_MIN     = float(os.getenv("ATR_PCT_MIN","0.002"))

# ===== Risk config =====
TOTAL_CAPITAL_PCT = float(os.getenv("TOTAL_CAPITAL_PCT","0.20"))
MAX_OPEN_TRADES   = int(os.getenv("MAX_OPEN_TRADES","4"))
PER_TRADE_PCT     = os.getenv("PER_TRADE_PCT", "").strip()
PER_TRADE_PCT     = float(PER_TRADE_PCT) if PER_TRADE_PCT else (TOTAL_CAPITAL_PCT / max(1, MAX_OPEN_TRADES))
LEVERAGE          = int(os.getenv("LEVERAGE","5"))
TAKE_PROFIT_PCT   = float(os.getenv("TAKE_PROFIT_PCT","0.006"))
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT","0.004"))

TG_TOKEN  = os.getenv("TELEGRAM_TOKEN","")
TG_CHATID = os.getenv("TELEGRAM_CHAT_ID","")

BASE = "https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"
KLINES = f"{BASE}/fapi/v1/klines"
TICKER_24H = f"{BASE}/fapi/v1/ticker/24hr"
EXCHANGE_INFO = f"{BASE}/fapi/v1/exchangeInfo"
ORDER_EP = f"{BASE}/fapi/v1/order"
BALANCE_EP = f"{BASE}/fapi/v2/balance"
POSITION_RISK_EP = f"{BASE}/fapi/v2/positionRisk"
LEVERAGE_EP = f"{BASE}/fapi/v1/leverage"

session = requests.Session()
session.headers.update({"X-MBX-APIKEY": API_KEY, "User-Agent":"MahdiTradeBot/4.1r-TG-Live"})

def now_utc(): return datetime.now(timezone.utc)

def send_tg(text: str):
    if not TG_TOKEN or not TG_CHATID: return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TG_CHATID, "text": text}, timeout=10)
    except Exception as e:
        print(f"[TG ERR] {e}")

def signed(params: dict):
    params["timestamp"] = int(time.time()*1000)
    q = "&".join([f"{k}={params[k]}" for k in sorted(params)])
    sig = hmac.new(API_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    return q + f"&signature={sig}"

def _request(method, url, *, params=None, data=None, signed_req=False, retries=4, timeout=20):
    if params is None: params = {}
    if data   is None: data   = {}
    backoff = 1.0
    for i in range(retries):
        try:
            if method == "GET":
                resp = session.get(url + ("?"+signed(params) if signed_req else ""), params=None if signed_req else params, timeout=timeout)
            else:
                resp = session.post(url, data=signed(data) if signed_req else data, timeout=timeout)
            if resp.status_code in (418,429):
                wait_s = 60 * (i+1) * 2
                print(f"[RATE LIMIT] {resp.status_code} -> cooling {wait_s}s"); time.sleep(wait_s); continue
            resp.raise_for_status(); return resp.json()
        except requests.exceptions.RequestException as e:
            if i == retries-1: raise
            sleep_s = backoff + random.random(); print(f"[NET WARN] {e} -> retry {sleep_s:.1f}s"); time.sleep(sleep_s); backoff *= 1.7

def f_get(url, params): return _request("GET", url, params=params)

# ===== data & indicators =====
def get_klines(symbol, interval="5m", limit=None):
    if limit is None: limit = KLINES_LIMIT
    data = f_get(KLINES, {"symbol":symbol, "interval":interval, "limit":limit})
    cols = ["open_time","open","high","low","close","volume","close_time","q","t","tb","tq","i"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]: df[c] = df[c].astype(float)
    return df

def ema(s,n): return s.ewm(span=n, adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.rolling(n).mean()/(dn.rolling(n).mean()+1e-9); return 100-(100/(1+rs))
def macd(s,fast=12,slow=26,signal=9):
    f=ema(s,fast); sl=ema(s,slow); m=f-sl; sig=ema(m,signal); return m,sig,m-sig
def atr(df,n=14):
    h,l,c=df["high"],df["low"],df["close"]; pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); return tr.rolling(n).mean()
def adx(df,n=14):
    h,l,c=df["high"],df["low"],df["close"]; plus=(h-h.shift(1)).clip(lower=0); minus=(l.shift(1)-l).clip(lower=0)
    tr=atr(df,n)*n; p=100*(plus.rolling(n).sum()/(tr+1e-9)); m=100*(minus.rolling(n).sum()/(tr+1e-9))
    dx=((p-m).abs()/((p+m)+1e-9))*100; return dx.rolling(n).mean()

def indicator_votes(df):
    close=df["close"]; votes={}
    f=ema(close,21); s=ema(close,50); votes["EMA"]="BUY" if f.iloc[-1]>s.iloc[-1] else "SELL" if f.iloc[-1]<s.iloc[-1] else "HOLD"
    m,sg,_=macd(close,12,26,9); votes["MACD"]="BUY" if m.iloc[-1]>sg.iloc[-1] else "SELL" if m.iloc[-1]<sg.iloc[-1] else "HOLD"
    r=rsi(close,14).iloc[-1]; votes["RSI"]="SELL" if r>70 else "BUY" if r<30 else "HOLD"
    atr_v=float(atr(df,14).iloc[-1]); adx_v=float(adx(df,14).iloc[-1]); last=float(close.iloc[-1])
    return votes,last,adx_v,atr_v

def soft_consensus(votes, price, adx_v, atr_v):
    w={k:(1.0 if any(x in k.lower() for x in ["ema","sma","macd","supertrend","vwap"]) else 0.8) for k in votes.keys()}
    if adx_v is not None and adx_v<ADX_MIN: return "HOLD"
    if atr_v is not None and price>0 and (atr_v/price)<ATR_PCT_MIN: return "HOLD"
    bw=sum(w[k] for k,v in votes.items() if v=="BUY"); sw=sum(w[k] for k,v in votes.items() if v=="SELL")
    hw=sum(w[k] for k,v in votes.items() if v=="HOLD"); tot=bw+sw+hw
    if tot<=0: return "HOLD"
    need=tot*CONSENSUS_RATIO; bn=sum(1 for v in votes.values() if v=="BUY"); sn=sum(1 for v in votes.values() if v=="SELL")
    if bw>=need and bn>=max(MIN_AGREE,1) and bw>sw: return "BUY"
    if sw>=need and sn>=max(MIN_AGREE,1) and sw>bw: return "SELL"
    return "HOLD"

# ===== exchange =====
_info_cache={}
def symbol_filters(symbol):
    if symbol in _info_cache: return _info_cache[symbol]
    data=f_get(EXCHANGE_INFO, {"symbol":symbol}); fs=data["symbols"][0]["filters"]
    lot=float([f for f in fs if f["filterType"]=="LOT_SIZE"][0]["stepSize"])
    minq=float([f for f in fs if f["filterType"]=="LOT_SIZE"][0]["minQty"])
    tick=float([f for f in fs if f["filterType"]=="PRICE_FILTER"][0]["tickSize"])
    _info_cache[symbol]={"lot_step":lot,"min_qty":minq,"tick_size":tick}; return _info_cache[symbol]

def round_step(qty, step): return float(np.floor(qty/step)*step + 1e-12)
def account_balance_usdt():
    data=_request("GET","%s/fapi/v2/balance"%("https://fapi.binance.com" if not USE_TESTNET else "https://testnet.binancefuture.com"), signed_req=True)
    for x in data:
        if x["asset"]=="USDT": return float(x["balance"])
    return 0.0

def current_positions():
    ep="%s/fapi/v2/positionRisk"%("https://fapi.binance.com" if not USE_TESTNET else "https://testnet.binancefuture.com")
    data=_request("GET", ep, signed_req=True); open_pos={}
    for p in data:
        qty=float(p["positionAmt"]); 
        if abs(qty)>1e-12: open_pos[p["symbol"]]=qty
    return open_pos

def ensure_leverage(symbol, lev):
    ep="%s/fapi/v1/leverage"%("https://fapi.binance.com" if not USE_TESTNET else "https://testnet.binancefuture.com")
    try: _request("POST", ep, signed_req=True, data={"symbol":symbol,"leverage":lev})
    except Exception as e: print(f"[LEV WARN] {e}")

def place_market(symbol, side, qty):
    return _request("POST", "%s/fapi/v1/order"%("https://fapi.binance.com" if not USE_TESTNET else "https://testnet.binancefuture.com"),
                    signed_req=True, data={"symbol":symbol,"side":side,"type":"MARKET","quantity":qty})

def calc_order_qty(symbol, price):
    bal = account_balance_usdt()
    per = PER_TRADE_PCT  # e.g., 0.05 when TOTAL=0.20 and MAX_OPEN_TRADES=4
    notional = bal * per * LEVERAGE
    raw_qty = notional / price
    f = symbol_filters(symbol)
    qty = max(round_step(raw_qty, f["lot_step"]), f["min_qty"])
    return qty

def maybe_trade(symbol, signal, price, open_pos, max_open=MAX_OPEN_TRADES):
    if len(open_pos) >= max_open and symbol not in open_pos:
        return
    if symbol in open_pos:
        return
    side = "BUY" if signal=="BUY" else "SELL"
    ensure_leverage(symbol, LEVERAGE)
    qty = calc_order_qty(symbol, price)
    if qty <= 0: return
    if RUN_MODE.lower()=="paper":
        send_tg(f"(PAPER) {symbol} {side} qty={qty:.6f} @ ~{price:.4f}")
        return
    try:
        order = place_market(symbol, side, qty)
        send_tg(f"✅ فتح {symbol} {side} | qty={qty} | px≈{price}")
    except Exception as e:
        send_tg(f"❌ فشل فتح {symbol}: {e}")

# ===== universe & scan =====
def load_universe():
    if SYMBOLS_CSV and os.path.exists(SYMBOLS_CSV):
        return pd.read_csv(SYMBOLS_CSV)["symbol"].tolist()[:MAX_SYMBOLS]
    data=f_get(KLINES.replace("/klines","/ticker/24hr"), {"type":"FULL"})
    df=pd.DataFrame(data); df=df[df["symbol"].str.endswith("USDT")]
    df["quoteVolume"]=df["quoteVolume"].astype(float)
    return df.sort_values("quoteVolume", ascending=False)["symbol"].head(60).tolist()[:MAX_SYMBOLS]

def scan_once(symbols):
    hits=0; errors=0; signals=[]
    for sym in symbols:
        try:
            df=get_klines(sym, INTERVAL, KLINES_LIMIT)
            if len(df)<60: time.sleep(0.3); continue
            votes, px, adx_v, atr_v = indicator_votes(df)
            sig=soft_consensus(votes, px, adx_v, atr_v)
            if sig in ("BUY","SELL"): hits+=1; signals.append((sym,sig,px))
            time.sleep(1.0 + random.random()*0.5)
        except Exception as e:
            errors+=1; print(f"[SYM ERR] {sym}: {e}")
    return hits, errors, signals

def heartbeat(h,e,sigs):
    ts=now_utc().strftime("%Y-%m-%d %H:%M:%SZ")
    msg=f"💗 Heartbeat | {ts}\nSignals:{h} | Errors:{e} | Open per-trade:{PER_TRADE_PCT*100:.1f}% of balance"
    print(msg); send_tg(msg)

def main():
    send_tg("🚀 تشغيل Mahdi v4.1r Live (20%/4 صفقات)")
    symbols=load_universe()
    last_hb=now_utc()-timedelta(minutes=HEARTBEAT_MIN+1)
    cooldown_until=now_utc()
    while True:
        start=now_utc()
        if start<cooldown_until: time.sleep(2); continue
        try: open_pos=current_positions()
        except Exception as e: open_pos={}; print(f"[POS WARN] {e}")
        h,e,sigs=scan_once(symbols)
        for sym,sig,px in sigs: maybe_trade(sym,sig,px,open_pos,MAX_OPEN_TRADES)
        if (now_utc()-last_hb)>=timedelta(minutes=HEARTBEAT_MIN):
            heartbeat(h,e,sigs); last_hb=now_utc()
        cooldown_until=now_utc()+timedelta(seconds=SCAN_INTERVAL_SEC if h==0 else COOLDOWN_MIN*60)
        print(f"دورة مسح: إشارات={h}, أخطاء={e}, وقت={(now_utc()-start).seconds}s")
        time.sleep(max(1, SCAN_INTERVAL_SEC))

if __name__=="__main__":
    main()
