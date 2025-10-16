#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mahdi Trade Bot — v5 FINAL
- Strategy: Consensus (EMA, MACD, RSI, Supertrend, VWAP) + ADX/ATR filters
- Dynamic leverage (5x normal / 10x strong), dynamic position sizing (5% / 6%)
- 3 Targets: TP1/TP2/TP3 with partial closes and SL->BE then lock after TP2
- Precision-safe price/qty rounding with Decimal; guards against zero/invalid prices
- Universe: reads universe.csv (25 symbols) or falls back to top-volume auto
- Kill-Switch: daily realized PnL limit (e.g., 5% of balance)
- Watchdog: inactivity alerts with reminder + last-activity summary
- Telegram: structured alerts for startup, entries, TP1, closes, heartbeats, errors
- Heartbeat every TG_HEARTBEAT_MIN minutes with capital usage & stats
"""
import os, time, hmac, hashlib, random, math
from datetime import datetime, timezone, timedelta
import requests, pandas as pd, numpy as np
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv

# ========= Env =========
load_dotenv()
API_KEY     = os.getenv("API_KEY","")
API_SECRET  = os.getenv("API_SECRET","")
USE_TESTNET = os.getenv("USE_TESTNET","false").lower() in ("1","true","yes")
RUN_MODE    = os.getenv("RUN_MODE","live")  # live | paper | analysis

INTERVAL          = os.getenv("INTERVAL","5m")
MAX_SYMBOLS       = int(os.getenv("MAX_SYMBOLS","25"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC","60"))
COOLDOWN_MIN      = int(os.getenv("COOLDOWN_MIN","15"))
KLINES_LIMIT      = int(os.getenv("KLINES_LIMIT","200"))
HEARTBEAT_MIN     = int(os.getenv("TG_HEARTBEAT_MIN","15"))
SYMBOLS_CSV       = os.getenv("SYMBOLS_CSV","universe.csv")

CONSENSUS_RATIO   = float(os.getenv("CONSENSUS_RATIO","0.65"))
MIN_AGREE         = int(os.getenv("MIN_AGREE","2"))
ADX_MIN           = float(os.getenv("ADX_MIN","20"))
ATR_PCT_MIN       = float(os.getenv("ATR_PCT_MIN","0.0025"))

TOTAL_CAPITAL_PCT = float(os.getenv("TOTAL_CAPITAL_PCT","0.40"))
MAX_OPEN_TRADES   = int(os.getenv("MAX_OPEN_TRADES","6"))
# Dynamic per-trade allocation by leverage tier:
STRONG_TRADE_PCT  = float(os.getenv("STRONG_TRADE_PCT","0.06"))  # 10x
NORMAL_TRADE_PCT  = float(os.getenv("NORMAL_TRADE_PCT","0.05"))  # 5x
# Stops & targets (balanced defaults); can be overridden per-strength inline
STOP_LOSS_PCT_BASE = float(os.getenv("STOP_LOSS_PCT","0.009"))   # 0.9%
TP1_PCT = float(os.getenv("TP1_PCT","0.0035"))
TP2_PCT = float(os.getenv("TP2_PCT","0.007"))
TP3_PCT = float(os.getenv("TP3_PCT","0.012"))
TP1_SHARE = float(os.getenv("TP1_SHARE","0.40"))
TP2_SHARE = float(os.getenv("TP2_SHARE","0.35"))
TP3_SHARE = float(os.getenv("TP3_SHARE","0.25"))
BREAKEVEN_AFTER_TP1 = os.getenv("BREAKEVEN_AFTER_TP1","true").lower() in ("1","true","yes")
LOCK_AFTER_TP2_PCT  = float(os.getenv("LOCK_AFTER_TP2_PCT","0.002"))  # 0.20% lock after TP2

# Strong-signal overrides
STOP_LOSS_PCT_STRONG = float(os.getenv("STOP_LOSS_PCT_STRONG","0.012"))
TP1_PCT_STRONG = float(os.getenv("TP1_PCT_STRONG","0.005"))
TP2_PCT_STRONG = float(os.getenv("TP2_PCT_STRONG","0.010"))
TP3_PCT_STRONG = float(os.getenv("TP3_PCT_STRONG","0.018"))
TP_SHARES_STRONG = os.getenv("TP_SHARES_STRONG","0.35,0.35,0.30")  # csv

# Telegram
TG_TOKEN  = os.getenv("TELEGRAM_TOKEN","")
TG_CHATID = os.getenv("TELEGRAM_CHAT_ID","")
TG_ENABLED = os.getenv("TG_ENABLED","true").lower() in ("1","true","yes")
TG_NOTIFY_WEAK = os.getenv("TG_NOTIFY_WEAK","false").lower() in ("1","true","yes")
TG_NOTIFY_UNIVERSE = os.getenv("TG_NOTIFY_UNIVERSE","true").lower() in ("1","true","yes")

# Margin type control
DEFAULT_MARGIN_TYPE = os.getenv("MARGIN_TYPE","ISOLATED").upper()  # ISOLATED or CROSS

# Kill-Switch
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT","0.05"))

# Watchdog
WATCHDOG_MIN = int(os.getenv("WATCHDOG_MIN","10"))
WATCHDOG_REMINDER_MIN = int(os.getenv("WATCHDOG_REMINDER_MIN","30"))

# Endpoints
BASE = "https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"
KLINES = f"{BASE}/fapi/v1/klines"
TICKER_24H = f"{BASE}/fapi/v1/ticker/24hr"
PRICE_EP = f"{BASE}/fapi/v1/ticker/price"
EXCHANGE_INFO = f"{BASE}/fapi/v1/exchangeInfo"
ORDER_EP = f"{BASE}/fapi/v1/order"
ALL_OPEN_ORDERS = f"{BASE}/fapi/v1/allOpenOrders"
BALANCE_EP = f"{BASE}/fapi/v2/balance"
POSITION_RISK_EP = f"{BASE}/fapi/v2/positionRisk"
LEVERAGE_EP = f"{BASE}/fapi/v1/leverage"
DUAL_SIDE_EP = f"{BASE}/fapi/v1/positionSide/dual"
SERVER_TIME_EP = f"{BASE}/fapi/v1/time"
INCOME_EP = f"{BASE}/fapi/v1/income"
USER_TRADES_EP = f"{BASE}/fapi/v1/userTrades"

session = requests.Session()
session.headers.update({"X-MBX-APIKEY": API_KEY, "User-Agent":"MahdiTradeBot/5.0-final"})

# ========= Timezones & Activity =========
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

def mark_activity(event: str, detail: str = ""):
    global _last_action_time, _last_activity_desc, _last_activity_ts_utc
    _last_action_time = now_utc()
    _last_activity_ts_utc = _last_action_time
    _last_activity_desc = (event if not detail else f"{event}: {detail}")

def fmt_both_times(ts_utc: datetime):
    ts_local = ts_utc.astimezone(TZ_RIYADH)
    return ts_utc.strftime("%Y-%m-%d %H:%M:%S UTC"), ts_local.strftime("%Y-%m-%d %H:%M:%S Asia/Riyadh")

# ========= Decimal helpers =========
def _D(x): return Decimal(str(x))

def floor_to_step(value, step):
    v = _D(value); s = _D(step)
    n = (v / s).to_integral_value(rounding=ROUND_DOWN)
    q = (n * s)
    if q <= 0: q = s  # never zero/negative
    return str(q.normalize())

def price_to_tick(price, tick): return floor_to_step(price, tick)

def qty_to_step(qty, lot_step, min_qty):
    q = _D(floor_to_step(qty, lot_step))
    mq = _D(str(min_qty))
    if q < mq: q = mq
    return str(q.normalize())

# ========= Telegram =========
def send_tg(text: str):
    if not TG_ENABLED or not TG_TOKEN or not TG_CHATID: return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TG_CHATID, "text": text, "parse_mode":"HTML", "disable_web_page_preview":True}, timeout=10)
    except Exception as e:
        print(f"[TG ERR] {e}")

# ========= Binance low-level =========
_time_offset_ms = 0
def sync_server_time():
    global _time_offset_ms
    try:
        r = session.get(SERVER_TIME_EP, timeout=10); r.raise_for_status()
        srv = int(r.json()["serverTime"]); loc = int(time.time()*1000)
        _time_offset_ms = srv - loc
    except Exception as e:
        print(f"[TIME WARN] {e}")

def signed(params: dict):
    ts = int(time.time()*1000 + _time_offset_ms)
    params["timestamp"] = ts
    params.setdefault("recvWindow", 60000)
    q = "&".join([f"{k}={params[k]}" for k in sorted(params)])
    sig = hmac.new(API_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    return q + f"&signature={sig}"

def _request(method, url, *, params=None, data=None, signed_req=False, retries=3, timeout=20):
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
                time.sleep(60*(i+1)*2); continue
            if resp.status_code >= 400:
                try:
                    j=resp.json(); code=j.get("code"); msg=j.get("msg")
                    raise requests.HTTPError(f"{resp.status_code} [{code}] {msg}", response=resp)
                except ValueError:
                    resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if i == retries-1: raise
            time.sleep(backoff); backoff *= 1.7

def f_get(url, params): return _request("GET", url, params=params)

# ========= Exchange info & account =========
_info_cache={}
def symbol_filters(symbol):
    if symbol in _info_cache: return _info_cache[symbol]
    data=f_get(EXCHANGE_INFO, {"symbol":symbol}); fs=data["symbols"][0]["filters"]
    lot=float([f for f in fs if f["filterType"]=="LOT_SIZE"][0]["stepSize"])
    minq=float([f for f in fs if f["filterType"]=="LOT_SIZE"][0]["minQty"])
    tick=float([f for f in fs if f["filterType"]=="PRICE_FILTER"][0]["tickSize"])
    _info_cache[symbol]={"lot_step":lot,"min_qty":minq,"tick_size":tick}; return _info_cache[symbol]

def account_balance_usdt():
    data=_request("GET", BALANCE_EP, signed_req=True)
    for x in data:
        if x["asset"]=="USDT": return float(x["balance"])
    return 0.0

def is_hedge_mode():
    try:
        j=_request("GET", DUAL_SIDE_EP, signed_req=True)
        return bool(j.get("dualSidePosition"))
    except Exception as e:
        return False

def ensure_leverage(symbol, lev):
    try: _request("POST", LEVERAGE_EP, signed_req=True, data={"symbol":symbol,"leverage":lev})
    except Exception: pass

def ensure_margin_type(symbol, margin_type):
    """Force symbol margin type (ISOLATED/CROSS). Ignores 'no need to change' error."""
    mt = margin_type.upper()
    if mt not in ("ISOLATED","CROSS"): return
    try:
        _request("POST", f"{BASE}/fapi/v1/marginType", signed_req=True, data={"symbol":symbol,"marginType":mt})
    except requests.HTTPError as e:
        # -4046 No need to change margin type.
        # -4048 Margin type cannot be changed if there exists open orders on the symbol or the leverage is lower than 1.
        msg = str(e)
        if "4046" in msg:
            return
        elif "4048" in msg:
            send_tg(f"⚠️ لا يمكن تغيير هامش {symbol} إلى {mt} لوجود صفقات/أوامر مفتوحة. سيتم المتابعة بوضع الهامش الحالي.")
        else:
            # Other errors, just log once.
            print(f"[MARGIN WARN] {symbol}: {e}")
    except Exception as e:
        print(f"[MARGIN WARN] {symbol}: {e}")


# ========= Market data =========
def get_klines(symbol, interval="5m", limit=200):
    data = f_get(KLINES, {"symbol":symbol, "interval":interval, "limit":limit})
    cols = ["open_time","open","high","low","close","volume","close_time","q","t","tb","tq","i"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]: df[c] = df[c].astype(float)
    return df

def get_live_price(symbol, fallback=None):
    try:
        live = f_get(PRICE_EP, {"symbol": symbol})
        p = float(live["price"])
        if p>0: return p
    except Exception: pass
    return fallback

# ========= Indicators =========
def ema(s,n): return s.ewm(span=n, adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.rolling(n).mean()/(dn.rolling(n).mean()+1e-9); return 100-(100/(1+rs))
def macd(s,fast=12,slow=26,signal=9):
    f=ema(s,fast); sl=ema(s,slow); m=f-sl; sg=ema(m,signal); return m,sg,m-sg
def atr(df,n=14):
    h,l,c=df["high"],df["low"],df["close"]; pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); return tr.rolling(n).mean()
def adx(df,n=14):
    h,l,c=df["high"],df["low"],df["close"]; plus=(h-h.shift(1)).clip(lower=0); minus=(l.shift(1)-l).clip(lower=0)
    tr=atr(df,n)*n; p=100*(plus.rolling(n).sum()/(tr+1e-9)); m=100*(minus.rolling(n).sum()/(tr+1e-9))
    dx=((p-m).abs()/((p+m)+1e-9))*100; return dx.rolling(n).mean()

def supertrend(df, period=10, mult=3.0):
    # Basic bands
    hl2 = (df["high"] + df["low"]) / 2.0
    _atr = atr(df, period)
    upper = hl2 + mult * _atr
    lower = hl2 - mult * _atr
    # Supertrend logic
    st = pd.Series(index=df.index, dtype=float)
    dir_up = True
    for i in range(len(df)):
        if i==0:
            st.iloc[i]=upper.iloc[i]; dir_up=True; continue
        if df["close"].iloc[i] > st.iloc[i-1]:
            dir_up=True
        elif df["close"].iloc[i] < st.iloc[i-1]:
            dir_up=False
        st.iloc[i] = lower.iloc[i] if dir_up else upper.iloc[i]
    # Signal: price above ST => uptrend
    sig = pd.Series(np.where(df["close"]>st, "BUY", "SELL"), index=df.index)
    return sig

def vwap(df, window=50):
    # rolling VWAP approximation for 5m candles
    tp = (df["high"]+df["low"]+df["close"])/3.0
    vol = df["volume"]
    rv = (tp*vol).rolling(window).sum()/(vol.rolling(window).sum()+1e-9)
    # signal: close above vwap => BUY ; below => SELL; else HOLD when very close
    close = df["close"]
    diff = (close - rv)/rv
    sig = pd.Series(np.where(diff>0.0005, "BUY", np.where(diff<-0.0005, "SELL", "HOLD")), index=df.index)
    return sig

def indicator_votes(df):
    close=df["close"]; votes={}
    # EMA
    f=ema(close,21); s=ema(close,50)
    votes["EMA"]="BUY" if f.iloc[-1]>s.iloc[-1] else "SELL" if f.iloc[-1]<s.iloc[-1] else "HOLD"
    # MACD
    m,sg,_=macd(close,12,26,9)
    votes["MACD"]="BUY" if m.iloc[-1]>sg.iloc[-1] else "SELL" if m.iloc[-1]<sg.iloc[-1] else "HOLD"
    # RSI
    r=rsi(close,14).iloc[-1]; votes["RSI"]="SELL" if r>70 else "BUY" if r<30 else "HOLD"
    # Supertrend
    st = supertrend(df, period=10, mult=3.0).iloc[-1]
    votes["SUPERTREND"]=st
    # VWAP
    vw = vwap(df, window=50).iloc[-1]
    votes["VWAP"]=vw
    # Filters
    atr_v=float(atr(df,14).iloc[-1]); adx_v=float(adx(df,14).iloc[-1]); last=float(close.iloc[-1])
    atr_pct = (atr_v/last) if last>0 else 0.0
    return votes,last,adx_v,atr_v,atr_pct

def soft_consensus(votes, price, adx_v, atr_pct):
    # Weights
    w = {}
    for k in votes.keys():
        if any(x in k.lower() for x in ["ema","macd","supertrend","vwap"]):
            w[k]=1.0
        else:
            w[k]=0.8
    # Filters
    if adx_v is not None and adx_v < ADX_MIN: return "HOLD", 0.0
    if atr_pct is not None and atr_pct < ATR_PCT_MIN: return "HOLD", 0.0
    # Consensus
    bw=sum(w[k] for k,v in votes.items() if v=="BUY")
    sw=sum(w[k] for k,v in votes.items() if v=="SELL")
    bn=sum(1 for v in votes.values() if v=="BUY")
    sn=sum(1 for v in votes.values() if v=="SELL")
    tot=bw+sw+sum(w[k] for k,v in votes.items() if v=="HOLD")
    if tot<=0: return "HOLD", 0.0
    # Normalize strength as ratio of dominant side over (bw+sw)
    dom = max(bw, sw)
    strength = dom / (bw+sw+1e-9)
    # Require minimum agreements
    if bw>sw and bn>=MIN_AGREE and strength>=CONSENSUS_RATIO: return "BUY", strength
    if sw>bw and sn>=MIN_AGREE and strength>=CONSENSUS_RATIO: return "SELL", strength
    return "HOLD", strength

# ========= Orders =========
def place_market(symbol, side, qty, positionSide=None):
    f = symbol_filters(symbol)
    qty_str = qty_to_step(qty, f["lot_step"], f["min_qty"])
    params={"symbol":symbol,"side":side,"type":"MARKET","quantity":qty_str}
    if positionSide: params["positionSide"]=positionSide
    return _request("POST", ORDER_EP, signed_req=True, data=params)

def cancel_all_orders(symbol):
    try: _request("DELETE", ALL_OPEN_ORDERS, signed_req=True, data={"symbol":symbol})
    except Exception: pass

def fmt_price(p):
    try: return f"{float(p):.8f}".rstrip('0').rstrip('.')
    except Exception: return str(p)

def send_entry_alert(symbol, side, entry, qty, lev, tp1, tp2, tp3, sl):
    if not entry or entry <= 0:
        send_tg(f"⚠️ إلغاء تنبيه {symbol}: Entry غير صالح"); return
    emoji = "🟢 LONG دخول ✅" if side=="BUY" else "🔴 SHORT دخول ✅"
    msg = (f"{emoji}\n<b>{symbol}</b> | Entry {fmt_price(entry)} | SL {fmt_price(sl)}\n"
           f"TP1:{tp1*100:.2f}% | TP2:{tp2*100:.2f}% | TP3:{tp3*100:.2f}%\n"
           f"Qty {float(qty):.8f} | Lev {lev}x")
    send_tg(msg)

# ========= User trades & PnL =========
def user_trades(symbol, start_ms):
    out=[]; s=start_ms
    while True:
        params={"symbol":symbol, "startTime":int(s), "limit":1000}
        data=_request("GET", USER_TRADES_EP, params=params, signed_req=True, timeout=20)
        if not isinstance(data, list) or not data: break
        out.extend(data)
        last_ts = max(int(t["time"]) for t in data)
        if len(data)<1000: break
        s = last_ts + 1
    return out

def income_sum(symbol, start_ms, end_ms):
    realized=0.0; fees=0.0; s=start_ms
    while True:
        params={"symbol":symbol or None,"startTime":int(s),"endTime":int(end_ms),"limit":1000}
        data=_request("GET", INCOME_EP, params=params, signed_req=True, timeout=20)
        if not isinstance(data, list) or not data: break
        for it in data:
            t=it.get("incomeType"); v=float(it.get("income",0.0))
            if t=="REALIZED_PNL": realized+=v
            elif t=="COMMISSION": fees+=v
        last_ts=max(int(it["time"]) for it in data)
        if last_ts>=end_ms or len(data)<1000: break
        s=last_ts+1
    return realized, fees

# ========= Position & state =========
def open_positions():
    try: data=_request("GET", POSITION_RISK_EP, signed_req=True)
    except Exception: return {}
    pos={}
    for p in data:
        qty=float(p["positionAmt"])
        if abs(qty)>1e-12: pos[p["symbol"]]=qty
    return pos

state = {}
_prev_open=set()

# ========= Universe =========

def validate_symbols(symbols):
    """Return only symbols that exist on USDT-M Futures; warn on invalid ones."""
    try:
        data = f_get(EXCHANGE_INFO, {"symbol": None})  # full list
        valid = set(s["symbol"] for s in data["symbols"])
    except Exception:
        return symbols
    out = []
    bad = []
    for s in symbols:
        if s in valid:
            out.append(s)
        else:
            bad.append(s)
    if bad:
        send_tg("⚠️ تم حذف أزواج غير متاحة على العقود الدائمة: " + ", ".join(bad))
    return out
def load_universe():
    syms=None
    if SYMBOLS_CSV and os.path.exists(SYMBOLS_CSV):
        try:
            df=pd.read_csv(SYMBOLS_CSV)
            syms=[s.strip().upper() for s in df["symbol"].tolist() if s.upper().endswith("USDT")]
            syms=validate_symbols(syms)[:MAX_SYMBOLS]
            if TG_NOTIFY_UNIVERSE:
                send_tg(f"📊 Universe ثابت: {', '.join(syms[:10])}... (n={len(syms)})")
            return syms
        except Exception as e:
            send_tg(f"⚠️ لم أستطع قراءة universe.csv: {e}")
    data = f_get(TICKER_24H, {"type":"FULL"})
    df=pd.DataFrame(data); df=df[df["symbol"].str.endswith("USDT")]
    df["quoteVolume"]=df["quoteVolume"].astype(float)
    syms=df.sort_values("quoteVolume", ascending=False)["symbol"].head(60).tolist()[:MAX_SYMBOLS]
    syms=validate_symbols(syms)
    if TG_NOTIFY_UNIVERSE:
        send_tg(f"📊 Universe تلقائي (Top Volume): {', '.join(syms[:10])}... (n={len(syms)})")
    return syms

# ========= Dynamic sizing & leverage =========
def calc_order_qty(symbol, price, leverage, strong_signal: bool):
    bal = account_balance_usdt()
    cap_pct = STRONG_TRADE_PCT if leverage==10 or strong_signal else NORMAL_TRADE_PCT
    notional = bal * cap_pct * leverage
    raw_qty = notional / price
    f = symbol_filters(symbol)
    qty_str = qty_to_step(raw_qty, f["lot_step"], f["min_qty"])
    return Decimal(qty_str)

# ========= TP/SL placement =========
def place_tp3_sl(symbol, side, entry_price, qty, positionSide, strong_signal: bool):
    # choose targets based on signal strength
    if strong_signal:
        tps = (TP1_PCT_STRONG, TP2_PCT_STRONG, TP3_PCT_STRONG)
        try:
            s1,s2,s3 = [float(x.strip()) for x in TP_SHARES_STRONG.split(",")]
        except Exception:
            s1,s2,s3 = 0.35,0.35,0.30
        tp_shares = (s1, s2, s3)
        sl_pct = STOP_LOSS_PCT_STRONG
        lock_pct = max(LOCK_AFTER_TP2_PCT, 0.003)
    else:
        tps = (TP1_PCT, TP2_PCT, TP3_PCT)
        tp_shares = (TP1_SHARE, TP2_SHARE, TP3_SHARE)
        sl_pct = STOP_LOSS_PCT_BASE
        lock_pct = LOCK_AFTER_TP2_PCT

    f=symbol_filters(symbol); tick=f["tick_size"]; lot=f["lot_step"]

    def side_price(pct, is_buy): 
        raw = entry_price*(1+pct) if is_buy else entry_price*(1-pct)
        return price_to_tick(max(raw, float(tick)), tick)

    is_buy = (side=="BUY")
    tp_prices = [ side_price(tps[0], is_buy), side_price(tps[1], is_buy), side_price(tps[2], is_buy) ]
    sl_price  = side_price(sl_pct, is_buy) if is_buy else side_price(sl_pct, is_buy)  # function handles

    # quantities
    from decimal import Decimal as D
    fmin = f["min_qty"]
    tp_qtys = [
        qty_to_step(D(str(qty))*D(str(tp_shares[0])), lot, fmin),
        qty_to_step(D(str(qty))*D(str(tp_shares[1])), lot, fmin),
        qty_to_step(D(str(qty))*D(str(tp_shares[2])), lot, fmin),
    ]

    tp_side = "SELL" if side=="BUY" else "BUY"
    posSide = ("LONG" if side=="BUY" else "SHORT") if positionSide else None

    # Create three TP MARKET orders
    # TP1 reduceOnly with fixed quantity
    p1={"symbol":symbol,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp_prices[0],
        "quantity": tp_qtys[0], "reduceOnly":"true","workingType":"MARK_PRICE"}
    if posSide: p1["positionSide"]=posSide
    _request("POST", ORDER_EP, signed_req=True, data=p1)

    # TP2 reduceOnly quantity
    p2={"symbol":symbol,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp_prices[1],
        "quantity": tp_qtys[1], "reduceOnly":"true","workingType":"MARK_PRICE"}
    if posSide: p2["positionSide"]=posSide
    _request("POST", ORDER_EP, signed_req=True, data=p2)

    # TP3 close remaining
    p3={"symbol":symbol,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp_prices[2],
        "closePosition":"true","workingType":"MARK_PRICE"}
    if posSide: p3["positionSide"]=posSide
    _request("POST", ORDER_EP, signed_req=True, data=p3)

    # Initial SL
    sp={"symbol":symbol,"side":tp_side,"type":"STOP_MARKET","stopPrice":sl_price,
        "closePosition":"true","workingType":"MARK_PRICE"}
    if posSide: sp["positionSide"]=posSide
    _request("POST", ORDER_EP, signed_req=True, data=sp)

    return tps, tp_prices, sl_pct, sl_price, lock_pct

# ========= TP detection and dynamic SL adjust =========
def detect_tp_fills(symbol):
    st = state.get(symbol)
    if not st: return
    start_ms = st["open_ts_ms"] - 60_000
    trades = user_trades(symbol, start_ms)
    if not trades: return
    close_side = "SELL" if st["side"]=="BUY" else "BUY"
    fills=[t for t in trades if int(t["time"])>=st["open_ts_ms"] and t.get("side")==close_side]
    if not fills: return

    def fqty(t): return float(t["qty"])
    def fpr(t): return float(t["price"])
    lot = st["lot_step"]
    from decimal import Decimal as D

    # Targets in qty terms
    tp1_qty_target = float(qty_to_step(D(str(st["qty"])) * D(str(st["tp_shares"][0])), lot, st["min_qty"]))
    tp2_qty_target = float(qty_to_step(D(str(st["qty"])) * D(str(st["tp_shares"][1])), lot, st["min_qty"]))
    # cumulative closed since entry
    total_closed = sum(fqty(t) for t in fills)

    # TP1 detection
    if (not st.get("tp1_done")) and total_closed + 1e-12 >= tp1_qty_target:
        acc=0.0; vwap=0.0
        for t in sorted(fills, key=lambda x: x["time"]):
            q=fqty(t); p=fpr(t)
            take=min(q, tp1_qty_target-acc)
            vwap += p*take; acc += take
            if acc >= tp1_qty_target-1e-12: break
        if acc>0:
            exec_px = vwap/acc
            st["tp1_done"]=True; st["tp1_price"]=exec_px
            send_tg(f"🎯 TP1 تنفيذ فعلي <b>{symbol}</b>\nسعر التنفيذ: {fmt_price(exec_px)} | كمية≈ {tp1_qty_target:.8f}")
            mark_activity("TP1 filled", f"{symbol} exec≈{fmt_price(exec_px)}")
            try:
                # Move SL to breakeven if enabled
                if BREAKEVEN_AFTER_TP1:
                    cancel_all_orders(symbol)
                    tp_side = "SELL" if st["side"]=="BUY" else "BUY"
                    posSide = st["positionSide"]
                    common = {"symbol":symbol,"workingType":"MARK_PRICE"}
                    if posSide: common["positionSide"]=posSide
                    # Recreate TP2 and TP3 and SL=BE
                    f=symbol_filters(symbol)
                    tick=f["tick_size"]
                    # prices
                    is_buy = (st["side"]=="BUY")
                    def price_for(pct):
                        raw = st["entry"]*(1+pct) if is_buy else st["entry"]*(1-pct)
                        return price_to_tick(max(raw, float(tick)), tick)
                    tp2_price = price_for(st["tps"][1])
                    tp3_price = price_for(st["tps"][2])
                    be_price  = price_to_tick(st["entry"], tick)
                    # quantities
                    # TP2 quantity fixed, TP3 closePosition
                    tp2_qty   = qty_to_step(D(str(st["qty"])) * D(str(st["tp_shares"][1])), f["lot_step"], f["min_qty"])
                    _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp2_price,"quantity":tp2_qty,"reduceOnly":"true"})
                    _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp3_price,"closePosition":"true"})
                    _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"STOP_MARKET","stopPrice":be_price,"closePosition":"true"})
                    send_tg("🛡️ تم تعديل SL إلى Breakeven ووضع TP2/TP3 لباقي الكمية")
            except Exception as e:
                send_tg(f"⚠️ لم أستطع تعديل الأوامر بعد TP1: {e}")

    # TP2 detection (cumulative >= tp1 + tp2 targets)
    if st.get("tp1_done") and (not st.get("tp2_done")) and total_closed + 1e-12 >= (tp1_qty_target + tp2_qty_target):
        st["tp2_done"]=True
        # tighten SL to lock profit
        try:
            cancel_all_orders(symbol)
            tp_side = "SELL" if st["side"]=="BUY" else "BUY"
            posSide = st["positionSide"]
            f=symbol_filters(symbol); tick=f["tick_size"]; lot=f["lot_step"]
            is_buy = (st["side"]=="BUY")
            def price_for(pct):
                raw = st["entry"]*(1+pct) if is_buy else st["entry"]*(1-pct)
                return price_to_tick(max(raw, float(tick)), tick)
            # remaining TP3
            tp3_price = price_for(st["tps"][2])
            common = {"symbol":symbol,"workingType":"MARK_PRICE"}
            if posSide: common["positionSide"]=posSide
            _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp3_price,"closePosition":"true"})
            # lock SL
            lock = price_for(st["lock_pct"] if is_buy else -st["lock_pct"])
            _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"STOP_MARKET","stopPrice":lock,"closePosition":"true"})
            send_tg(f"🔒 تم تشديد SL بعد TP2 إلى {'+' if is_buy else '-'}{st['lock_pct']*100:.2f}% من الدخول")
            mark_activity("TP2 filled", f"{symbol} lock={st['lock_pct']*100:.2f}%")
        except Exception as e:
            send_tg(f"⚠️ لم أستطع تشديد SL بعد TP2: {e}")

# ========= Close summary =========
def send_close_summary_real(symbol, entry, qty, side, open_ts_ms, leverage):
    end_ms=int(time.time()*1000 + _time_offset_ms)
    realized, fees = income_sum(symbol, open_ts_ms-60_000, end_ms)
    pnl = realized + fees
    margin = float(entry)*float(qty)/max(1,leverage)
    roi = (pnl/margin*100) if margin>0 else 0.0
    emoji = "✅ ربح" if pnl>=0 else "❌ خسارة"
    send_tg(f"📘 إغلاق مركز (فعلي)\n<b>{symbol}</b> | {emoji}\nRealized P&L: {pnl:.4f} USDT (PnL {realized:.4f}, Fees {fees:.4f})\nQty {float(qty):.8f} | Lev {leverage}x | ROI≈ {roi:.2f}%")
    mark_activity("Closed", f"{symbol} PnL={pnl:.4f}")

# ========= Trading loop helpers =========
def scan_once(symbols):
    hits=0; errors=0; signals=[]
    for sym in symbols:
        try:
            df=get_klines(sym, INTERVAL, KLINES_LIMIT)
            if len(df)<60: time.sleep(0.2); continue
            votes, px, adx_v, atr_v, atr_pct = indicator_votes(df)
            sig, strength = soft_consensus(votes, px, adx_v, atr_pct)
            if sig in ("BUY","SELL"): hits+=1; signals.append((sym,sig,px,adx_v,strength))
            time.sleep(0.5)
        except Exception as e:
            errors+=1
    return hits, errors, signals

def detect_closes_and_notify():
    global _prev_open
    positions=open_positions()
    now_open=set(positions.keys())
    for s in list(state.keys()):
        try: detect_tp_fills(s)
        except Exception: pass
    closed=[s for s in _prev_open if s not in now_open]
    for s in closed:
        st=state.pop(s, None)
        if st:
            try: send_close_summary_real(s, st["entry"], st["qty"], st["side"], st["open_ts_ms"], st.get("leverage",5))
            except Exception as e: send_tg(f"🔔 تم إغلاق مركز {s} (تعذر حساب P&L بدقة: {e})")
        else: send_tg(f"🔔 تم إغلاق مركز {s}")
    _prev_open=now_open

def heartbeat(h,e,open_n, cap_used_pct):
    send_tg(f"💗 Heartbeat | إستراتيجية: Consensus+ST+VWAP\nإشارات بالدورة: {h} | مراكز مفتوحة: {open_n} | أخطاء: {e}\nاستخدام رأس المال: {cap_used_pct:.0f}% (حدك {int(TOTAL_CAPITAL_PCT*100)}%)")
    mark_activity("Heartbeat", f"open={open_n}")

def capital_usage_pct():
    # approximate by number of open trades * avg per-trade pct (5~6%)
    n = len(_prev_open)
    if n==0: return 0.0
    # assume worst case 6%
    return min(100.0, n*6.0)

def maybe_trade(symbol, signal, price, adx_v, consensus_strength, hedge):
    # limit open trades
    positions = open_positions()
    if len(positions) >= MAX_OPEN_TRADES and symbol not in positions: return
    if symbol in positions: return

    # dynamic leverage decision
    strong = (adx_v >= 28 or consensus_strength >= 0.80)
    if strong:
        lev = 10
    elif adx_v >= 20 and consensus_strength >= 0.65:
        lev = 5
    else:
        if TG_NOTIFY_WEAK:
            send_tg(f"⚠️ تجاهل {symbol}: إشارة ضعيفة (ADX {adx_v:.1f}, Ratio {consensus_strength:.2f})")
        return

    # live price safeguard
    price = get_live_price(symbol, fallback=price)
    if not price or price<=0:
        send_tg(f"⚠️ تجاوز {symbol}: سعر غير صالح للفتح ({price})"); return

    # Force margin type per symbol before leverage
    try:
        ensure_margin_type(symbol, DEFAULT_MARGIN_TYPE)
    except Exception:
        pass
    ensure_leverage(symbol, lev)
    qty=calc_order_qty(symbol, price, lev, strong_signal=strong)
    if qty<=0: return

    # targets based on strength
    if strong:
        tps = (TP1_PCT_STRONG, TP2_PCT_STRONG, TP3_PCT_STRONG)
        try:
            s1,s2,s3 = [float(x.strip()) for x in TP_SHARES_STRONG.split(",")]
        except Exception:
            s1,s2,s3 = 0.35,0.35,0.30
        tp_shares=(s1,s2,s3)
        sl_pct = STOP_LOSS_PCT_STRONG
        lock_pct = max(LOCK_AFTER_TP2_PCT, 0.003)
    else:
        tps = (TP1_PCT, TP2_PCT, TP3_PCT)
        tp_shares=(TP1_SHARE, TP2_SHARE, TP3_SHARE)
        sl_pct = STOP_LOSS_PCT_BASE
        lock_pct = LOCK_AFTER_TP2_PCT

    side="BUY" if signal=="BUY" else "SELL"
    posSide=("LONG" if side=="BUY" else "SHORT") if hedge else None

    if RUN_MODE.lower()=="paper" or RUN_MODE.lower()=="analysis":
        # simulate entry only
        f=symbol_filters(symbol)
        state[symbol]={"side":side,"entry":price,"qty":qty,"positionSide":posSide,
                       "open_ts_ms": int(time.time()*1000 + _time_offset_ms),
                       "tp1_done": False, "tp2_done": False,
                       "tick_size": f["tick_size"], "lot_step": f["lot_step"], "min_qty": f["min_qty"],
                       "tps": tps, "tp_shares": tp_shares, "lock_pct": lock_pct, "leverage": lev}
        sl_price = price*(1-sl_pct) if side=="BUY" else price*(1+sl_pct)
        send_entry_alert(symbol, side, price, qty, lev, tps[0], tps[1], tps[2], sl_price)
        mark_activity("Entry", f"{symbol} {side} @ {fmt_price(price)}")
        return

    try:
        order=place_market(symbol, side, qty, posSide)
        entry=float(order.get("avgPrice") or price)
        f=symbol_filters(symbol)
        state[symbol]={"side":side,"entry":entry,"qty":qty,"positionSide":posSide,
                       "open_ts_ms": int(time.time()*1000 + _time_offset_ms),
                       "tp1_done": False, "tp2_done": False,
                       "tick_size": f["tick_size"], "lot_step": f["lot_step"], "min_qty": f["min_qty"],
                       "tps": tps, "tp_shares": tp_shares, "lock_pct": lock_pct, "leverage": lev}
        tps_, tp_prices, sl_pct_used, sl_price, lock_pct_used = place_tp3_sl(symbol, side, entry, qty, posSide, strong_signal=strong)
        send_entry_alert(symbol, side, entry, qty, lev, tps[0], tps[1], tps[2], sl_price)
        mark_activity("Entry", f"{symbol} {side} @ {fmt_price(entry)}")
    except Exception as e:
        send_tg(f"❌ فشل فتح/تسعير أوامر {symbol}: {e}\n"
                f"ملاحظة: قد يكون المركز مفتوحًا بدون TP/SL — راجع يدويًا.")
        # keep state so we can still detect close

# ========= Kill-Switch =========
_daily_loss_triggered = False
def check_daily_pnl_limit():
    global _daily_loss_triggered
    if _daily_loss_triggered: return True
    bal = account_balance_usdt()
    end_ms = int(time.time()*1000 + _time_offset_ms)
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_ms = int(start_of_day.timestamp()*1000)
    realized, fees = income_sum("", start_ms, end_ms)
    total_pnl = realized + fees
    if total_pnl < -bal * DAILY_LOSS_LIMIT_PCT:
        _daily_loss_triggered = True
        send_tg(f"🛑 تم إيقاف التداول حتى نهاية اليوم (UTC)\nخسارة اليوم: {total_pnl:.2f} USDT (> {DAILY_LOSS_LIMIT_PCT*100:.1f}% من الرصيد)")
        return True
    return False

# ========= Watchdog =========
def watchdog_check():
    global _watchdog_stage
    now = now_utc()
    idle_min = (now - _last_action_time).total_seconds() / 60.0

    if idle_min >= WATCHDOG_MIN and _watchdog_stage == 0:
        utc_str, ry_str = fmt_both_times(_last_activity_ts_utc)
        send_tg(
            "⚠️ <b>تنبيه:</b> لم يصدر البوت أي نشاط منذ "
            f"{WATCHDOG_MIN} دقيقة.\n"
            f"آخر نشاط: {_last_activity_desc}\n"
            f"- {utc_str}\n- {ry_str}\n"
            "تحقق من Logs في Render أو أعد التشغيل."
        )
        _watchdog_stage = 1

    elif idle_min >= WATCHDOG_MIN + WATCHDOG_REMINDER_MIN and _watchdog_stage == 1:
        utc_str, ry_str = fmt_both_times(_last_activity_ts_utc)
        send_tg(
            "🚨 <b>تنبيه مستمر:</b> البوت ما زال متوقفًا منذ "
            f"{int(idle_min)} دقيقة.\n"
            f"آخر نشاط: {_last_activity_desc}\n"
            f"- {utc_str}\n- {ry_str}\n"
            "يُرجى فحص الخدمة فورًا."
        )
        _watchdog_stage = 2

    elif idle_min < WATCHDOG_MIN and _watchdog_stage > 0:
        send_tg("✅ عاد البوت للعمل بعد توقف مؤقت.")
        _watchdog_stage = 0

# ========= Main =========
def main():
    send_tg(f"🚀 تشغيل Mahdi v5 — وضع: {RUN_MODE} | Testnet: {'On' if USE_TESTNET else 'Off'}")
    mark_activity("Startup", f"mode={RUN_MODE}, testnet={USE_TESTNET}")
    sync_server_time()
    hedge=is_hedge_mode()
    symbols=load_universe()
    # Pre-set margin type for all symbols (best-effort)
    for _s in symbols:
        try:
            ensure_margin_type(_s, DEFAULT_MARGIN_TYPE)
            time.sleep(0.05)
        except Exception:
            pass
    last_hb=now_utc()-timedelta(minutes=HEARTBEAT_MIN+1)
    cooldown_until=now_utc()

    while True:
        start=now_utc()
        try:
            if check_daily_pnl_limit():
                time.sleep(60); watchdog_check(); continue
            if start<cooldown_until:
                time.sleep(1); watchdog_check(); continue
            h,e,sigs=scan_once(symbols)
            for sym,sig,px,adx_v,strength in sigs:
                maybe_trade(sym,sig,px,adx_v,strength,hedge)
            detect_closes_and_notify()
            # heartbeat
            if (now_utc()-last_hb)>=timedelta(minutes=HEARTBEAT_MIN):
                hb_cap = capital_usage_pct()
                heartbeat(h,e,len(_prev_open), hb_cap); last_hb=now_utc()
            # cooldown
            cooldown_until=now_utc()+timedelta(seconds=SCAN_INTERVAL_SEC if h==0 else COOLDOWN_MIN*60)
            watchdog_check()
            time.sleep(1)
        except Exception as ex:
            send_tg(f"⚠️ Loop error: {ex}")
            time.sleep(3)
            watchdog_check()

if __name__=="__main__":
    main()
