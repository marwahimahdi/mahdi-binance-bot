#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MahdiBot v5 FINAL — Auto-Scan Mode (Top N)
— USDT-M Futures فقط (PERPETUAL/TRADING)
— مؤشرات: EMA(21/50), MACD(12,26,9), RSI(14), Supertrend(10,3), VWAP(50)
— فلاتر: ADX>=20, ATR/Close>=0.0025, وإجماع>=0.65 مع MIN_AGREE>=2
— رافعة ديناميكية: 5x/10x | حجم: 5%/6% (حد رأس المال 40%، أقصى صفقات 6)
— 3 أهداف: 0.35% / 0.7% / 1.2% (القوية: 0.5% / 1.0% / 1.8%)
— بعد TP1: SL→Breakeven | بعد TP2: قفل ربح (+0.20% / القوية +0.30%)
— Kill-Switch يومي عند -5% | Watchdog 10د + تذكير بعد 30د
— إجبار الهامش ISOLATED، وتنظيف الرموز غير الصالحة تلقائيًا
"""

import os, time, hmac, hashlib, math, requests
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone, timedelta

# ============ ENV ============
API_KEY     = os.getenv("API_KEY","")
API_SECRET  = os.getenv("API_SECRET","")
USE_TESTNET = os.getenv("USE_TESTNET","false").lower() in ("1","true","yes")
RUN_MODE    = os.getenv("RUN_MODE","live")  # live|paper|analysis

INTERVAL          = os.getenv("INTERVAL","5m")
MAX_SYMBOLS       = int(os.getenv("MAX_SYMBOLS","30"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC","75"))
COOLDOWN_MIN      = int(os.getenv("COOLDOWN_MIN","15"))
KLINES_LIMIT      = int(os.getenv("KLINES_LIMIT","200"))
# معدل تلطيف الطلبات لتفادي حظر 418/429 من Binance
REST_BACKOFF_BASE = float(os.getenv("REST_BACKOFF_BASE", "0.35"))  # ثوانٍ
REST_BACKOFF_MAX  = float(os.getenv("REST_BACKOFF_MAX",  "8"))     # حد أقصى للنوم التلقائي
HEARTBEAT_MIN     = int(os.getenv("TG_HEARTBEAT_MIN","15"))
SYMBOLS_CSV       = os.getenv("SYMBOLS_CSV","")  # اتركه فارغًا للوضع التلقائي

CONSENSUS_RATIO   = float(os.getenv("CONSENSUS_RATIO","0.65"))
MIN_AGREE         = int(os.getenv("MIN_AGREE","2"))
ADX_MIN           = float(os.getenv("ADX_MIN","20"))
ATR_PCT_MIN       = float(os.getenv("ATR_PCT_MIN","0.0025"))

TOTAL_CAPITAL_PCT = float(os.getenv("TOTAL_CAPITAL_PCT","0.40"))
MAX_OPEN_TRADES   = int(os.getenv("MAX_OPEN_TRADES","6"))
NORMAL_TRADE_PCT  = float(os.getenv("NORMAL_TRADE_PCT","0.05"))
STRONG_TRADE_PCT  = float(os.getenv("STRONG_TRADE_PCT","0.06"))

# أهداف عادية
STOP_LOSS_PCT_BASE = float(os.getenv("STOP_LOSS_PCT","0.009"))
TP1_PCT = float(os.getenv("TP1_PCT","0.0035"))
TP2_PCT = float(os.getenv("TP2_PCT","0.007"))
TP3_PCT = float(os.getenv("TP3_PCT","0.012"))
TP1_SHARE = float(os.getenv("TP1_SHARE","0.40"))
TP2_SHARE = float(os.getenv("TP2_SHARE","0.35"))
TP3_SHARE = float(os.getenv("TP3_SHARE","0.25"))
BREAKEVEN_AFTER_TP1 = os.getenv("BREAKEVEN_AFTER_TP1","true").lower() in ("1","true","yes")
LOCK_AFTER_TP2_PCT  = float(os.getenv("LOCK_AFTER_TP2_PCT","0.002"))  # 0.20%

# أهداف قوية
STOP_LOSS_PCT_STRONG = float(os.getenv("STOP_LOSS_PCT_STRONG","0.012"))
TP1_PCT_STRONG = float(os.getenv("TP1_PCT_STRONG","0.005"))
TP2_PCT_STRONG = float(os.getenv("TP2_PCT_STRONG","0.010"))
TP3_PCT_STRONG = float(os.getenv("TP3_PCT_STRONG","0.018"))
TP_SHARES_STRONG = os.getenv("TP_SHARES_STRONG","0.35,0.35,0.30")

# Telegram
TG_TOKEN  = os.getenv("TELEGRAM_TOKEN","")
TG_CHATID = os.getenv("TELEGRAM_CHAT_ID","")
TG_ENABLED = os.getenv("TG_ENABLED","true").lower() in ("1","true","yes")
TG_NOTIFY_WEAK = os.getenv("TG_NOTIFY_WEAK","false").lower() in ("1","true","yes")
TG_NOTIFY_UNIVERSE = os.getenv("TG_NOTIFY_UNIVERSE","true").lower() in ("1","true","yes")

# إدارة المخاطرة
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT","0.05"))
DEFAULT_MARGIN_TYPE = os.getenv("MARGIN_TYPE","ISOLATED").upper()

# ============ Endpoints ============
BASE = "https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"
KLINES = f"{BASE}/fapi/v1/klines"
TICKER_24H = f"{BASE}/fapi/v1/ticker/24hr"
EXCHANGE_INFO = f"{BASE}/fapi/v1/exchangeInfo"
PRICE_EP = f"{BASE}/fapi/v1/ticker/price"
BALANCE_EP = f"{BASE}/fapi/v2/balance"
POSITION_RISK_EP = f"{BASE}/fapi/v2/positionRisk"
LEVERAGE_EP = f"{BASE}/fapi/v1/leverage"
MARGIN_TYPE_EP = f"{BASE}/fapi/v1/marginType"
DUAL_SIDE_EP = f"{BASE}/fapi/v1/positionSide/dual"
ORDER_EP = f"{BASE}/fapi/v1/order"
ALL_OPEN_ORDERS = f"{BASE}/fapi/v1/allOpenOrders"
SERVER_TIME_EP = f"{BASE}/fapi/v1/time"
INCOME_EP = f"{BASE}/fapi/v1/income"
USER_TRADES_EP = f"{BASE}/fapi/v1/userTrades"

session = requests.Session()
session.headers.update({"X-MBX-APIKEY": API_KEY, "User-Agent":"MahdiBot/5.0-final"})

# ============ Time helpers ============
try:
    from zoneinfo import ZoneInfo
    TZ_RIYADH = ZoneInfo("Asia/Riyadh")
except Exception:
    TZ_RIYADH = timezone(timedelta(hours=3))

def now_utc(): return datetime.now(timezone.utc)

_last_action_time = now_utc()
_last_activity_desc = "Startup"
_last_activity_ts_utc = _last_action_time
_watchdog_stage = 0  # 0=ok,1=warned,2=reminded

def mark_activity(event, detail=""):
    global _last_action_time, _last_activity_desc, _last_activity_ts_utc
    _last_action_time = now_utc()
    _last_activity_ts_utc = _last_action_time
    _last_activity_desc = event if not detail else f"{event}: {detail}"

def fmt_both_times(ts_utc):
    ts_local = ts_utc.astimezone(TZ_RIYADH)
    return ts_utc.strftime("%Y-%m-%d %H:%M:%S UTC"), ts_local.strftime("%Y-%m-%d %H:%M:%S Asia/Riyadh")

# ============ Utils ============
def _D(x): return Decimal(str(x))
def floor_to_step(value, step):
    v=_D(value); s=_D(step)
    n=(v/s).to_integral_value(rounding=ROUND_DOWN)
    q=(n*s)
    if q<=0: q=s
    return str(q.normalize())
def price_to_tick(price, tick): return floor_to_step(price, tick)
def qty_to_step(qty, lot_step, min_qty):
    q=_D(floor_to_step(qty, lot_step))
    mq=_D(str(min_qty))
    if q < mq: q = mq
    return str(q.normalize())

# ============ Telegram ============
def send_tg(text):
    if not (TG_ENABLED and TG_TOKEN and TG_CHATID):
        print(f"[TG SKIP] enabled={TG_ENABLED} token={bool(TG_TOKEN)} chat_id={TG_CHATID}")
        return
    try:
        r = session.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHATID, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=10
        )
        if r.status_code >= 400:
            # اطبع الخطأ بوضوح
            try:
                print(f"[TG ERR] {r.status_code} {r.json()}")
            except Exception:
                print(f"[TG ERR] {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[TG EXC] {e}")

# ============ Requests/Signing ============
_time_offset_ms=0
def sync_server_time():
    global _time_offset_ms
    try:
        r=session.get(SERVER_TIME_EP, timeout=10); r.raise_for_status()
        srv=int(r.json()["serverTime"]); loc=int(time.time()*1000)
        _time_offset_ms=srv-loc
    except Exception as e:
        print(f"[TIME WARN] {e}")

def signed(params:dict):
    ts=int(time.time()*1000 + _time_offset_ms)
    params["timestamp"]=ts
    params.setdefault("recvWindow",60000)
    q="&".join([f"{k}={params[k]}" for k in sorted(params)])
    sig=hmac.new(API_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    return q+f"&signature={sig}"

import random

def _request(method, url, *, params=None, data=None, signed_req=False, timeout=20):
    if params is None: params={}
    if data is None: data={}
    attempt = 0
    backoff = REST_BACKOFF_BASE

    while True:
        try:
            if method == "GET":
                if signed_req:
                    u = url + "?" + signed(params); P = None
                else:
                    u = url; P = params
                resp = session.get(u, params=P, timeout=timeout)
            else:
                payload = signed(data) if signed_req else data
                resp = session.post(url, data=payload, timeout=timeout)

            # 418/429 = حظر/ضغط — نطبّق باك أوف ونكرّر
            if resp.status_code in (418, 429):
                attempt += 1
                try:
                    j = resp.json(); code = j.get("code"); msg = j.get("msg","")
                except Exception:
                    code = None; msg = ""
                if attempt == 3:
                    send_tg(f"⏳ Binance ضغط/حظر مؤقت ({resp.status_code} [{code}] {msg}). سيتم تخفيف الطلبات وإعادة المحاولة تلقائيًا.")
                sleep_s = min(REST_BACKOFF_MAX, backoff * (1.7 ** attempt)) + random.uniform(0, 0.3)
                time.sleep(sleep_s)
                continue

            if resp.status_code >= 400:
                try:
                    j = resp.json(); code = j.get("code"); msg = j.get("msg")
                    raise requests.HTTPError(f"{resp.status_code} [{code}] {msg}", response=resp)
                except ValueError:
                    resp.raise_for_status()

            return resp.json()

        except requests.Timeout:
            attempt += 1
            time.sleep(min(REST_BACKOFF_MAX, backoff * (1.5 ** attempt)) + random.uniform(0, 0.2))
            continue
        except requests.ConnectionError:
            attempt += 1
            time.sleep(min(REST_BACKOFF_MAX, backoff * (1.5 ** attempt)) + random.uniform(0, 0.2))
            continue

def f_get(url, params): return _request("GET", url, params=params)

# ============ Exchange & Account ============
_info_cache={}
def symbol_filters(symbol):
    if symbol in _info_cache: return _info_cache[symbol]
    data=f_get(EXCHANGE_INFO, {"symbol":symbol})
    fs=data["symbols"][0]["filters"]
    lot=float([f for f in fs if f["filterType"]=="LOT_SIZE"][0]["stepSize"])
    minq=float([f for f in fs if f["filterType"]=="LOT_SIZE"][0]["minQty"])
    tick=float([f for f in fs if f["filterType"]=="PRICE_FILTER"][0]["tickSize"])
    _info_cache[symbol]={"lot_step":lot,"min_qty":minq,"tick_size":tick}
    return _info_cache[symbol]

def account_balance_usdt():
    data=_request("GET", BALANCE_EP, signed_req=True)
    for x in data:
        if x["asset"]=="USDT": return float(x["balance"])
    return 0.0

def is_hedge_mode():
    try:
        j=_request("GET", DUAL_SIDE_EP, signed_req=True); return bool(j.get("dualSidePosition"))
    except Exception: return False

def ensure_leverage(symbol, lev):
    try: _request("POST", LEVERAGE_EP, signed_req=True, data={"symbol":symbol,"leverage":lev})
    except Exception: pass

def ensure_margin_type(symbol, margin_type):
    mt=margin_type.upper()
    if mt not in ("ISOLATED","CROSS"): return
    try:
        _request("POST", MARGIN_TYPE_EP, signed_req=True, data={"symbol":symbol,"marginType":mt})
    except requests.HTTPError as e:
        msg=str(e)
        if "4046" in msg:  # no need to change
            return
        elif "4048" in msg:
            send_tg(f"⚠️ لا يمكن تغيير هامش <b>{symbol}</b> إلى {mt} لوجود صفقات/أوامر مفتوحة. سيتم المتابعة بوضع الهامش الحالي.")
        else:
            print(f"[MARGIN WARN] {symbol}: {e}")
    except Exception as e:
        print(f"[MARGIN WARN] {symbol}: {e}")

# ============ Market Data ============
def get_klines(symbol, interval="5m", limit=200):
    data=f_get(KLINES, {"symbol":symbol,"interval":interval,"limit":limit})
    cols=["open_time","open","high","low","close","volume","close_time","q","t","tb","tq","i"]
    df=pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]: df[c]=df[c].astype(float)
    return df

def get_live_price(symbol):
    j=f_get(PRICE_EP, {"symbol":symbol})
    return float(j["price"])

# ============ Indicators ============
def ema(s,n): return s.ewm(span=n, adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.rolling(n).mean()/(dn.rolling(n).mean()+1e-9)
    return 100-(100/(1+rs))
def macd(s,fast=12,slow=26,signal=9):
    f=ema(s,fast); sl=ema(s,slow); m=f-sl; sg=ema(m,signal); return m,sg,m-sg
def atr(df,n=14):
    h,l,c=df["high"],df["low"],df["close"]; pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()
def adx(df,n=14):
    h,l,c=df["high"],df["low"],df["close"]
    plus=(h-h.shift(1)).clip(lower=0); minus=(l.shift(1)-l).clip(lower=0)
    tr=atr(df,n)*n
    p=100*(plus.rolling(n).sum()/(tr+1e-9))
    m=100*(minus.rolling(n).sum()/(tr+1e-9))
    dx=((p-m).abs()/((p+m)+1e-9))*100
    return dx.rolling(n).mean()

def supertrend(df, period=10, mult=3.0):
    hl2=(df["high"]+df["low"])/2.0
    _atr=atr(df, period)
    upper=hl2 + mult*_atr
    lower=hl2 - mult*_atr
    st=pd.Series(index=df.index, dtype=float)
    dir_up=True
    for i in range(len(df)):
        if i==0:
            st.iloc[i]=upper.iloc[i]; dir_up=True; continue
        if df["close"].iloc[i]>st.iloc[i-1]: dir_up=True
        elif df["close"].iloc[i]<st.iloc[i-1]: dir_up=False
        st.iloc[i]= lower.iloc[i] if dir_up else upper.iloc[i]
    sig=np.where(df["close"]>st, "BUY", "SELL")
    return pd.Series(sig, index=df.index)

def vwap(df, window=50):
    tp=(df["high"]+df["low"]+df["close"])/3.0
    vol=df["volume"]
    rv=(tp*vol).rolling(window).sum()/(vol.rolling(window).sum()+1e-9)
    close=df["close"]; diff=(close-rv)/rv
    return pd.Series(np.where(diff>0.0005,"BUY", np.where(diff<-0.0005,"SELL","HOLD")), index=df.index)

def indicator_votes(df):
    close=df["close"]; votes={}
    # EMA
    f=ema(close,21); s=ema(close,50)
    votes["EMA"]="BUY" if f.iloc[-1]>s.iloc[-1] else "SELL" if f.iloc[-1]<s.iloc[-1] else "HOLD"
    # MACD
    m,sg,_=macd(close); votes["MACD"]="BUY" if m.iloc[-1]>sg.iloc[-1] else "SELL" if m.iloc[-1]<sg.iloc[-1] else "HOLD"
    # RSI
    r=rsi(close,14).iloc[-1]; votes["RSI"]="SELL" if r>70 else "BUY" if r<30 else "HOLD"
    # Supertrend
    votes["SUPERTREND"]=supertrend(df).iloc[-1]
    # VWAP
    votes["VWAP"]=vwap(df).iloc[-1]
    # Filters
    _atr=atr(df,14).iloc[-1]; _adx=adx(df,14).iloc[-1]; last=close.iloc[-1]
    atr_pct = float(_atr/last) if last>0 else 0.0
    return votes, float(last), float(_adx), float(_atr), atr_pct

def soft_consensus(votes, adx_v, atr_pct):
    if adx_v < ADX_MIN or atr_pct < ATR_PCT_MIN: return "HOLD", 0.0
    w = {k:(1.0 if k in ("EMA","MACD","SUPERTREND","VWAP") else 0.8) for k in votes}
    bw=sum(w[k] for k,v in votes.items() if v=="BUY")
    sw=sum(w[k] for k,v in votes.items() if v=="SELL")
    bn=sum(1 for v in votes.values() if v=="BUY")
    sn=sum(1 for v in votes.values() if v=="SELL")
    if bw+sw==0: return "HOLD", 0.0
    strength = max(bw,sw)/(bw+sw)
    if bw>sw and bn>=MIN_AGREE and strength>=CONSENSUS_RATIO: return "BUY", strength
    if sw>bw and sn>=MIN_AGREE and strength>=CONSENSUS_RATIO: return "SELL", strength
    return "HOLD", strength

# ============ Universe (strict futures) ============
def fetch_valid_perpetual_usdt():
    """جلب جميع رموز USDT-M/ PERPETUAL/ TRADING."""
    data = _request("GET", EXCHANGE_INFO, params={}, signed_req=False)
    valid=set()
    for s in data.get("symbols", []):
        if s.get("status")=="TRADING" and s.get("quoteAsset")=="USDT" and s.get("contractType")=="PERPETUAL":
            valid.add(s["symbol"])
    return valid

def build_auto_universe():
    valid = fetch_valid_perpetual_usdt()
    tickers = f_get(TICKER_24H, {"type":"FULL"})
    df=pd.DataFrame(tickers)
    df=df[df["symbol"].isin(valid)].copy()
    df["quoteVolume"]=df["quoteVolume"].astype(float)
    syms = df.sort_values("quoteVolume", ascending=False)["symbol"].tolist()
    return syms[:MAX_SYMBOLS]

def load_universe():
    if SYMBOLS_CSV:
        import os
        if os.path.exists(SYMBOLS_CSV):
            df=pd.read_csv(SYMBOLS_CSV)
            syms=[s.strip().upper() for s in df["symbol"] if s.upper().endswith("USDT")]
        else:
            syms=[]
        valid = fetch_valid_perpetual_usdt()
        final=[s for s in syms if s in valid][:MAX_SYMBOLS]
        if TG_NOTIFY_UNIVERSE:
            send_tg(f"📊 Universe ثابت: {', '.join(final[:10])}... (n={len(final)})")
        return final
    # Auto
    syms = build_auto_universe()
    if TG_NOTIFY_UNIVERSE:
        send_tg(f"🔄 Auto-Scan Mode (Top {MAX_SYMBOLS})\n📊 Universe: {', '.join(syms[:10])}... (n={len(syms)})")
    return syms

# ============ Orders ============
def place_market(symbol, side, qty, positionSide=None):
    f=symbol_filters(symbol)
    qty_str=qty_to_step(qty, f["lot_step"], f["min_qty"])
    p={"symbol":symbol,"side":side,"type":"MARKET","quantity":qty_str}
    if positionSide: p["positionSide"]=positionSide
    return _request("POST", ORDER_EP, signed_req=True, data=p)

def cancel_all_orders(symbol):
    try: _request("DELETE", ALL_OPEN_ORDERS, signed_req=True, data={"symbol":symbol})
    except Exception: pass

def fmt_price(x):
    try: return f"{float(x):.8f}".rstrip("0").rstrip(".")
    except Exception: return str(x)

def send_entry_alert(symbol, side, entry, qty, lev, tps, sl):
    emoji = "🟢 LONG دخول ✅" if side=="BUY" else "🔴 SHORT دخول ✅"
    send_tg(f"{emoji}\n<b>{symbol}</b> | Entry {fmt_price(entry)} | SL {fmt_price(sl)}\n"
            f"TP1:{tps[0]*100:.2f}% | TP2:{tps[1]*100:.2f}% | TP3:{tps[2]*100:.2f}%\n"
            f"Qty {float(qty):.8f} | Lev {lev}x")

# ============ PnL & Trades ============
def user_trades(symbol, start_ms):
    out=[]; s=start_ms
    while True:
        data=_request("GET", USER_TRADES_EP, params={"symbol":symbol,"startTime":int(s),"limit":1000}, signed_req=True)
        if not isinstance(data, list) or not data: break
        out.extend(data)
        last=max(int(t["time"]) for t in data)
        if len(data)<1000: break
        s=last+1
    return out

def income_sum(symbol, start_ms, end_ms):
    realized=0.0; fees=0.0; s=start_ms
    while True:
        data=_request("GET", INCOME_EP, params={"symbol":symbol or None, "startTime":int(s), "endTime":int(end_ms), "limit":1000}, signed_req=True)
        if not isinstance(data, list) or not data: break
        for it in data:
            t=it.get("incomeType"); v=float(it.get("income",0.0))
            if t=="REALIZED_PNL": realized+=v
            elif t=="COMMISSION": fees+=v
        last=max(int(it["time"]) for it in data)
        if last>=end_ms or len(data)<1000: break
        s=last+1
    return realized, fees

# ============ State ============
state={}
_prev_open=set()

def open_positions():
    try:
        data=_request("GET", POSITION_RISK_EP, signed_req=True)
    except Exception:
        return {}
    pos={}
    for p in data:
        q=float(p["positionAmt"])
        if abs(q)>1e-12: pos[p["symbol"]]=q
    return pos

# ============ Sizing & leverage ============
def calc_order_qty(symbol, price, leverage, strong):
    bal=account_balance_usdt()
    cap_pct = STRONG_TRADE_PCT if (leverage==10 or strong) else NORMAL_TRADE_PCT
    notional = bal * cap_pct * leverage
    raw_qty = notional / price
    f=symbol_filters(symbol)
    qty_str=qty_to_step(raw_qty, f["lot_step"], f["min_qty"])
    return Decimal(qty_str)

# ============ TP/SL placement ============
def place_tp3_sl(symbol, side, entry, qty, posSide, strong):
    if strong:
        tps=(TP1_PCT_STRONG, TP2_PCT_STRONG, TP3_PCT_STRONG)
        try:
            s1,s2,s3=[float(x.strip()) for x in TP_SHARES_STRONG.split(",")]
        except Exception:
            s1,s2,s3=0.35,0.35,0.30
        shares=(s1,s2,s3)
        sl_pct=STOP_LOSS_PCT_STRONG
        lock_pct=max(LOCK_AFTER_TP2_PCT, 0.003)
    else:
        tps=(TP1_PCT, TP2_PCT, TP3_PCT)
        shares=(TP1_SHARE, TP2_SHARE, TP3_SHARE)
        sl_pct=STOP_LOSS_PCT_BASE
        lock_pct=LOCK_AFTER_TP2_PCT

    f=symbol_filters(symbol); tick=f["tick_size"]; lot=f["lot_step"]; minq=f["min_qty"]
    is_buy=(side=="BUY")
    def price_for(pct):
        raw = entry*(1+pct) if is_buy else entry*(1-pct)
        return price_to_tick(max(raw, float(tick)), tick)

    tp_prices=[price_for(tps[0]), price_for(tps[1]), price_for(tps[2])]
    sl_price = price_for(sl_pct)

    from decimal import Decimal as D
    tp_qtys=[
        qty_to_step(D(str(qty))*D(str(shares[0])), lot, minq),
        qty_to_step(D(str(qty))*D(str(shares[1])), lot, minq),
        qty_to_step(D(str(qty))*D(str(shares[2])), lot, minq),
    ]
    tp_side="SELL" if is_buy else "BUY"
    posS=("LONG" if is_buy else "SHORT") if posSide else None

    p1={"symbol":symbol,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp_prices[0],
        "quantity":tp_qtys[0], "reduceOnly":"true","workingType":"MARK_PRICE"}
    if posS: p1["positionSide"]=posS
    _request("POST", ORDER_EP, signed_req=True, data=p1)

    p2={"symbol":symbol,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp_prices[1],
        "quantity":tp_qtys[1], "reduceOnly":"true","workingType":"MARK_PRICE"}
    if posS: p2["positionSide"]=posS
    _request("POST", ORDER_EP, signed_req=True, data=p2)

    p3={"symbol":symbol,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp_prices[2],
        "closePosition":"true","workingType":"MARK_PRICE"}
    if posS: p3["positionSide"]=posS
    _request("POST", ORDER_EP, signed_req=True, data=p3)

    sp={"symbol":symbol,"side":tp_side,"type":"STOP_MARKET","stopPrice":sl_price,
        "closePosition":"true","workingType":"MARK_PRICE"}
    if posS: sp["positionSide"]=posS
    _request("POST", ORDER_EP, signed_req=True, data=sp)

    return tps, tp_prices, sl_pct, sl_price, lock_pct, shares

# ============ TP/SL fills detection (FIXED SYNTAX) ============
def detect_tp_fills(symbol):
    st = state.get(symbol)
    if not st:
        return

    start_ms = st["open_ts_ms"] - 60_000
    trades = user_trades(symbol, start_ms)
    if not trades:
        return

    close_side = "SELL" if st["side"] == "BUY" else "BUY"
    fills = [t for t in trades if int(t["time"]) >= st["open_ts_ms"] and t.get("side") == close_side]
    if not fills:
        return

    def fqty(t): return float(t["qty"])
    def fpr(t): return float(t["price"])

    lot = st["lot_step"]; minq = st["min_qty"]
    from decimal import Decimal as D
    tp1_target = float(qty_to_step(D(str(st["qty"])) * D(str(st["shares"][0])), lot, minq))
    tp2_target = float(qty_to_step(D(str(st["qty"])) * D(str(st["shares"][1])), lot, minq))
    total_closed = sum(fqty(t) for t in fills)

    # ---- TP1 ----
    if (not st.get("tp1_done")) and total_closed + 1e-12 >= tp1_target:
        acc = 0.0
        vwap = 0.0
        for t in sorted(fills, key=lambda x: x["time"]):
            q = fqty(t); p = fpr(t)
            take = min(q, tp1_target - acc)
            vwap += p * take
            acc  += take
            if acc >= tp1_target - 1e-12: break
        if acc > 0:
            exec_px = vwap / acc
            st["tp1_done"] = True
            st["tp1_price"] = exec_px
            send_tg(f"🎯 TP1 تنفيذ فعلي <b>{symbol}</b>\nسعر التنفيذ: {fmt_price(exec_px)} | كمية≈ {tp1_target:.8f}")
            mark_activity("TP1 filled", f"{symbol} exec≈{fmt_price(exec_px)}")

            if BREAKEVEN_AFTER_TP1:
                try:
                    cancel_all_orders(symbol)
                    is_buy = (st["side"] == "BUY")
                    f = symbol_filters(symbol); tick = f["tick_size"]

                    def price_for(pct):
                        raw = st["entry"] * (1 + pct) if is_buy else st["entry"] * (1 - pct)
                        return price_to_tick(max(raw, float(tick)), tick)

                    tp2_price = price_for(st["tps"][1])
                    tp3_price = price_for(st["tps"][2])
                    be_price  = price_to_tick(st["entry"], tick)
                    tp_side   = "SELL" if is_buy else "BUY"
                    common = {"symbol": symbol, "workingType": "MARK_PRICE"}
                    if st["positionSide"]: common["positionSide"] = st["positionSide"]

                    tp2_qty = qty_to_step(D(str(st["qty"])) * D(str(st["shares"][1])), f["lot_step"], f["min_qty"])

                    _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp2_price,"quantity":tp2_qty,"reduceOnly":"true"})
                    _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp3_price,"closePosition":"true"})
                    _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"STOP_MARKET","stopPrice":be_price,"closePosition":"true"})
                    send_tg("🛡️ تم تعديل SL إلى Breakeven ووضع TP2/TP3 لباقي الكمية")
                except Exception as e:
                    send_tg(f"⚠️ لم أستطع تعديل الأوامر بعد TP1: {e}")

    # ---- TP2 ----
    if st.get("tp1_done") and (not st.get("tp2_done")) and total_closed + 1e-12 >= (tp1_target + tp2_target):
        st["tp2_done"] = True
        try:
            cancel_all_orders(symbol)
            is_buy = (st["side"] == "BUY")
            f = symbol_filters(symbol); tick = f["tick_size"]

            def price_for(pct):
                raw = st["entry"] * (1 + pct) if is_buy else st["entry"] * (1 - pct)
                return price_to_tick(max(raw, float(tick)), tick)

            tp3_price = price_for(st["tps"][2])
            common = {"symbol": symbol, "workingType": "MARK_PRICE"}
            tp_side = "SELL" if is_buy else "BUY"
            if st["positionSide"]: common["positionSide"] = st["positionSide"]

            _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp3_price,"closePosition":"true"})

            lock_pct = st["lock_pct"]
            lock_price = price_for(lock_pct if is_buy else -lock_pct)
            _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"STOP_MARKET","stopPrice":lock_price,"closePosition":"true"})
            send_tg(f"🔒 تم تشديد SL بعد TP2 إلى {'+' if is_buy else '-'}{lock_pct*100:.2f}% من الدخول")
            mark_activity("TP2 filled", f"{symbol} lock={lock_pct*100:.2f}%")
        except Exception as e:
            send_tg(f"⚠️ لم أستطع تشديد SL بعد TP2: {e}")

def send_close_summary_real(symbol, entry, qty, side, open_ts_ms, lev):
    end_ms=int(time.time()*1000 + _time_offset_ms)
    realized, fees = income_sum(symbol, open_ts_ms-60_000, end_ms)
    pnl = realized + fees
    margin = float(entry)*float(qty)/max(1,lev)
    roi = (pnl/margin*100) if margin>0 else 0.0
    emoji="✅ ربح" if pnl>=0 else "❌ خسارة"
    send_tg(f"📘 إغلاق مركز (فعلي)\n<b>{symbol}</b> | {emoji}\nRealized P&L: {pnl:.4f} USDT (PnL {realized:.4f}, Fees {fees:.4f})\nQty {float(qty):.8f} | Lev {lev}x | ROI≈ {roi:.2f}%")
    mark_activity("Closed", f"{symbol} PnL={pnl:.4f}")

# ============ Scan & Trade ============
def scan_once(symbols):
    hits=0; errors=0; signals=[]; to_remove=[]
    for sym in list(symbols):
        try:
            df=get_klines(sym, INTERVAL, KLINES_LIMIT)
            if len(df)<60: time.sleep(0.1); continue
            votes, last, adx_v, atr_v, atr_pct = indicator_votes(df)
            sig, strength = soft_consensus(votes, adx_v, atr_pct)
            if sig in ("BUY","SELL"):
                hits+=1; signals.append((sym, sig, last, adx_v, strength))
            time.sleep(0.15)
        except requests.HTTPError as he:
            if "-1121" in str(he) or "Invalid symbol" in str(he):
                to_remove.append(sym)
            errors+=1
        except Exception:
            errors+=1
    if to_remove:
        for s in to_remove:
            try: symbols.remove(s)
            except ValueError: pass
        send_tg("⚠️ تمت إزالة أزواج غير صالحة أثناء المسح: " + ", ".join(to_remove))
    return hits, errors, signals

def detect_closes_and_notify():
    global _prev_open
    positions=open_positions()
    now_open=set(positions.keys())
    # detect TP fills
    for s in list(state.keys()):
        try: detect_tp_fills(s)
        except Exception: pass
    # closed
    closed=[s for s in _prev_open if s not in now_open]
    for s in closed:
        st=state.pop(s, None)
        if st:
            try: send_close_summary_real(s, st["entry"], st["qty"], st["side"], st["open_ts_ms"], st.get("leverage",5))
            except Exception as e: send_tg(f"🔔 تم إغلاق مركز {s} (تعذر حساب P&L بدقة: {e})")
        else:
            send_tg(f"🔔 تم إغلاق مركز {s}")
    _prev_open=now_open

def heartbeat(h,e,open_n, cap_used_pct):
    send_tg(f"💗 Heartbeat | Auto-Scan Mode (Top {MAX_SYMBOLS}) | إستراتيجية: Consensus+ST+VWAP\n"
            f"إشارات بالدورة: {h} | مراكز مفتوحة: {open_n} | أخطاء: {e}\n"
            f"استخدام رأس المال: {cap_used_pct:.0f}% (حدك {int(TOTAL_CAPITAL_PCT*100)}%)")
    mark_activity("Heartbeat", f"open={open_n}")

def capital_usage_pct():
    n=len(_prev_open)
    return min(100.0, n*6.0)  # تقديري

def maybe_trade(symbol, signal, price, adx_v, strength, hedge):
    positions=open_positions()
    if len(positions)>=MAX_OPEN_TRADES and symbol not in positions: return
    if symbol in positions: return

    strong=(adx_v>=28 or strength>=0.80)
    lev = 10 if strong else 5

    price=get_live_price(symbol)
    if not price or price<=0: return

    try:
        ensure_margin_type(symbol, DEFAULT_MARGIN_TYPE)
    except Exception: pass
    ensure_leverage(symbol, lev)

    qty=calc_order_qty(symbol, price, lev, strong)
    if qty<=0: return

    if strong:
        tps=(TP1_PCT_STRONG, TP2_PCT_STRONG, TP3_PCT_STRONG)
        try: s1,s2,s3=[float(x.strip()) for x in TP_SHARES_STRONG.split(",")]
        except Exception: s1,s2,s3=0.35,0.35,0.30
        shares=(s1,s2,s3); sl_pct=STOP_LOSS_PCT_STRONG; lock_pct=max(LOCK_AFTER_TP2_PCT,0.003)
    else:
        tps=(TP1_PCT, TP2_PCT, TP3_PCT)
        shares=(TP1_SHARE, TP2_SHARE, TP3_SHARE); sl_pct=STOP_LOSS_PCT_BASE; lock_pct=LOCK_AFTER_TP2_PCT

    side="BUY" if signal=="BUY" else "SELL"
    posSide=("LONG" if side=="BUY" else "SHORT") if hedge else None

    if RUN_MODE.lower() in ("paper","analysis"):
        f=symbol_filters(symbol)
        state[symbol]={"side":side,"entry":price,"qty":qty,"positionSide":posSide,
                       "open_ts_ms":int(time.time()*1000 + _time_offset_ms),
                       "tp1_done":False,"tp2_done":False,
                       "tick_size":f["tick_size"], "lot_step":f["lot_step"],
                       "min_qty":f["min_qty"], "tps":tps, "shares":shares,
                       "lock_pct":lock_pct, "leverage":lev}
        sl_price = price*(1-sl_pct) if side=="BUY" else price*(1+sl_pct)
        send_entry_alert(symbol, side, price, qty, lev, tps, sl_price)
        mark_activity("Entry", f"{symbol} {side} @ {fmt_price(price)}")
        return

    try:
        order=place_market(symbol, side, qty, posSide)
        entry=float(order.get("avgPrice") or price)
        f=symbol_filters(symbol)
        state[symbol]={"side":side,"entry":entry,"qty":qty,"positionSide":posSide,
                       "open_ts_ms":int(time.time()*1000 + _time_offset_ms),
                       "tp1_done":False,"tp2_done":False,
                       "tick_size":f["tick_size"],"lot_step":f["lot_step"],"min_qty":f["min_qty"],
                       "tps":tps,"shares":shares,"lock_pct":lock_pct,"leverage":lev}
        tps_, tp_prices, sl_pct_used, sl_price, lock_pct_used, shares_used = place_tp3_sl(symbol, side, entry, qty, posSide, strong)
        send_entry_alert(symbol, side, entry, qty, lev, tps, sl_price)
        mark_activity("Entry", f"{symbol} {side} @ {fmt_price(entry)}")
    except requests.HTTPError as he:
        if "-1121" in str(he) or "Invalid symbol" in str(he):
            send_tg(f"⚠️ {symbol}: رمز غير صالح — سيتم استبعاده من المسح.")
        else:
            send_tg(f"❌ فشل فتح {symbol}: {he}")
    except Exception as e:
        send_tg(f"❌ فشل فتح/تسعير {symbol}: {e}\nقد يكون المركز مفتوحًا بدون TP/SL — راجع يدويًا.")

# ============ Kill-Switch ============
_daily_loss_triggered=False
def check_daily_pnl_limit():
    global _daily_loss_triggered
    if _daily_loss_triggered: return True
    bal=account_balance_usdt()
    end_ms=int(time.time()*1000 + _time_offset_ms)
    start_of_day=datetime.now(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
    start_ms=int(start_of_day.timestamp()*1000)
    realized, fees = income_sum("", start_ms, end_ms)
    total=realized+fees
    if total < -bal*DAILY_LOSS_LIMIT_PCT:
        _daily_loss_triggered=True
        send_tg(f"🛑 تم إيقاف التداول حتى نهاية اليوم (UTC)\nخسارة اليوم: {total:.2f} USDT (> {DAILY_LOSS_LIMIT_PCT*100:.1f}%)")
        return True
    return False

# ============ Watchdog ============
WATCHDOG_MIN=int(os.getenv("WATCHDOG_MIN","10"))
WATCHDOG_REMINDER_MIN=int(os.getenv("WATCHDOG_REMINDER_MIN","30"))

def watchdog_check():
    global _watchdog_stage
    idle_min=(now_utc()-_last_action_time).total_seconds()/60.0
    if idle_min >= WATCHDOG_MIN and _watchdog_stage==0:
        utc_str, ry_str = fmt_both_times(_last_activity_ts_utc)
        send_tg("⚠️ <b>تنبيه:</b> لم يصدر البوت أي نشاط منذ "
                f"{WATCHDOG_MIN} دقيقة.\nآخر نشاط: {_last_activity_desc}\n- {utc_str}\n- {ry_str}\n"
                "تحقق من Logs في Render أو أعد التشغيل.")
        _watchdog_stage=1
    elif idle_min >= WATCHDOG_MIN+WATCHDOG_REMINDER_MIN and _watchdog_stage==1:
        utc_str, ry_str = fmt_both_times(_last_activity_ts_utc)
        send_tg("🚨 <b>تنبيه مستمر:</b> البوت ما زال متوقفًا منذ "
                f"{int(idle_min)} دقيقة.\nآخر نشاط: {_last_activity_desc}\n- {utc_str}\n- {ry_str}\n"
                "يرجى فحص الخدمة فورًا.")
        _watchdog_stage=2
    elif idle_min < WATCHDOG_MIN and _watchdog_stage>0:
        send_tg("✅ عاد البوت للعمل بعد توقف مؤقت."); _watchdog_stage=0

# ============ Main ============
def main():
    send_tg(f"🚀 تشغيل Mahdi v5 — وضع: {RUN_MODE} | Testnet: {'On' if USE_TESTNET else 'Off'}")
    send_tg(f"🔄 Auto-Scan Mode (Top {MAX_SYMBOLS})\n🧩 فحص كل {SCAN_INTERVAL_SEC} ث | حد أقصى {MAX_OPEN_TRADES} صفقات | رافعة 5-10× ديناميكية")
    mark_activity("Startup", f"mode={RUN_MODE}, testnet={USE_TESTNET}")
    sync_server_time()
    hedge=is_hedge_mode()
    symbols=load_universe()
    # إجبار الهامش المعزول قدر الإمكان عند الإقلاع
    for s in symbols:
        try: ensure_margin_type(s, DEFAULT_MARGIN_TYPE); time.sleep(0.03)
        except Exception: pass
    last_hb=now_utc()-timedelta(minutes=HEARTBEAT_MIN+1)
    cooldown_until=now_utc()

    while True:
        try:
            if check_daily_pnl_limit():
                time.sleep(60); watchdog_check(); continue
            if now_utc() < cooldown_until:
                time.sleep(1); watchdog_check(); continue

            h,e,sigs=scan_once(symbols)
            for sym,sig,px,adx_v,strength in sigs:
                if adx_v<ADX_MIN or strength<CONSENSUS_RATIO:
                    if TG_NOTIFY_WEAK:
                        send_tg(f"⚠️ تجاهل {sym}: إشارة ضعيفة (ADX {adx_v:.1f}, Ratio {strength:.2f})")
                    continue
                maybe_trade(sym,sig,px,adx_v,strength,hedge)

            detect_closes_and_notify()

            if (now_utc()-last_hb)>=timedelta(minutes=HEARTBEAT_MIN):
                heartbeat(h,e,len(_prev_open), capital_usage_pct()); last_hb=now_utc()

            cooldown_until=now_utc()+timedelta(seconds=SCAN_INTERVAL_SEC if h==0 else COOLDOWN_MIN*60)
            watchdog_check()
            time.sleep(1)
        except Exception as ex:
            send_tg(f"⚠️ Loop error: {ex}")
            time.sleep(3)
            watchdog_check()

if __name__=="__main__":
    main()
