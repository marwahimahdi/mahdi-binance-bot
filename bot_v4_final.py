
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mahdi Trade Bot — v4.1r (20%/4)
Partial TP (TP1 50% at mid), TP2 full, SL 1%, Exec Alerts, Breakeven SL, Realized PnL
-------------------------------------------------------------------------------------
- Entry alert (LONG/SHORT + Entry/SL/TP1/TP2/Qty/Lev)
- TP1: TAKE_PROFIT_MARKET @ mid target with quantity=50% (reduceOnly)
- TP2: TAKE_PROFIT_MARKET @ full target (closePosition=true)
- SL : STOP_MARKET @ 1% (closePosition=true)
- After TP1 fills => Cancel orders, set SL=Breakeven (Entry), re-place TP2 for remainder
- Exec prices for TP1/TP2 via /fapi/v1/userTrades; final Realized PnL via /fapi/v1/income
"""

import os, time, hmac, hashlib, random
from datetime import datetime, timezone, timedelta
import requests, pandas as pd, numpy as np
from dotenv import load_dotenv

load_dotenv()

# ===== Env =====
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

TOTAL_CAPITAL_PCT = float(os.getenv("TOTAL_CAPITAL_PCT","0.20"))
MAX_OPEN_TRADES   = int(os.getenv("MAX_OPEN_TRADES","4"))
PER_TRADE_PCT     = float(os.getenv("PER_TRADE_PCT", str(TOTAL_CAPITAL_PCT / max(1, int(MAX_OPEN_TRADES)))))
LEVERAGE          = int(os.getenv("LEVERAGE","5"))
TAKE_PROFIT_PCT   = float(os.getenv("TAKE_PROFIT_PCT","0.006"))  # TP2 = 0.6%
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT","0.01"))     # SL = 1% per request
TP1_SHARE         = float(os.getenv("TP1_SHARE", "0.5"))         # 50%

TG_TOKEN  = os.getenv("TELEGRAM_TOKEN","")
TG_CHATID = os.getenv("TELEGRAM_CHAT_ID","")

# ===== Endpoints =====
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
session.headers.update({"X-MBX-APIKEY": API_KEY, "User-Agent":"MahdiTradeBot/4.1r-Final"})

def now_utc(): return datetime.now(timezone.utc)

def send_tg(text: str):
    if not TG_TOKEN or not TG_CHATID: return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TG_CHATID, "text": text, "parse_mode":"HTML", "disable_web_page_preview":True}, timeout=10)
    except Exception as e:
        print(f"[TG ERR] {e}")

# ===== time sync/sign =====
_time_offset_ms = 0
def sync_server_time():
    global _time_offset_ms
    try:
        r = session.get(SERVER_TIME_EP, timeout=10); r.raise_for_status()
        srv = int(r.json()["serverTime"]); loc = int(time.time()*1000)
        _time_offset_ms = srv - loc
        print(f"[TIME] offset { _time_offset_ms } ms")
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
                wait_s = 60*(i+1)*2; print(f"[RATE LIMIT] {resp.status_code} sleep {wait_s}s"); time.sleep(wait_s); continue
            if resp.status_code >= 400:
                try:
                    j=resp.json(); code=j.get("code"); msg=j.get("msg")
                    raise requests.HTTPError(f"{resp.status_code} [{code}] {msg}", response=resp)
                except ValueError:
                    resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if i == retries-1: raise
            sleep_s = backoff + random.random(); print(f"[NET WARN] {e} -> retry {sleep_s:.1f}s"); time.sleep(sleep_s); backoff *= 1.7

def f_get(url, params): return _request("GET", url, params=params)

# ===== indicators =====
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

# ===== exchange helpers =====
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
    data=_request("GET", BALANCE_EP, signed_req=True)
    for x in data:
        if x["asset"]=="USDT": return float(x["balance"])
    return 0.0

def is_hedge_mode():
    try:
        j=_request("GET", DUAL_SIDE_EP, signed_req=True)
        return bool(j.get("dualSidePosition"))
    except Exception as e:
        print(f"[HEDGE WARN] {e}"); return False

def ensure_leverage(symbol, lev):
    try: _request("POST", LEVERAGE_EP, signed_req=True, data={"symbol":symbol,"leverage":lev})
    except Exception as e: print(f"[LEV WARN] {e}")

def place_market(symbol, side, qty, positionSide=None):
    params={"symbol":symbol,"side":side,"type":"MARKET","quantity":qty}
    if positionSide: params["positionSide"]=positionSide
    return _request("POST", ORDER_EP, signed_req=True, data=params)

def cancel_all_orders(symbol):
    try:
        _request("DELETE", ALL_OPEN_ORDERS, signed_req=True, data={"symbol":symbol})
    except Exception as e:
        print(f"[CANCEL WARN] {symbol}: {e}")

def place_tp1_tp2_sl(symbol, side, entry_price, qty, positionSide=None):
    """TP1 at mid-target (half of TAKE_PROFIT_PCT), TP2 full target, SL 1%."""
    f=symbol_filters(symbol); tick=f["tick_size"]; lot=f["lot_step"]
    # Targets
    tp2_price = entry_price*(1+TAKE_PROFIT_PCT) if side=="BUY" else entry_price*(1-TAKE_PROFIT_PCT)
    tp1_price = entry_price*(1+TAKE_PROFIT_PCT/2) if side=="BUY" else entry_price*(1-TAKE_PROFIT_PCT/2)
    sl_price  = entry_price*(1-STOP_LOSS_PCT) if side=="BUY" else entry_price*(1+STOP_LOSS_PCT)
    # Round
    tp2_price = round(tp2_price/tick)*tick
    tp1_price = round(tp1_price/tick)*tick
    sl_price  = round(sl_price/tick)*tick

    tp1_qty   = max(round_step(qty*TP1_SHARE, lot), f["min_qty"])

    tp_side = "SELL" if side=="BUY" else "BUY"
    # TP1 reduceOnly with quantity
    p1={"symbol":symbol,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp1_price,
        "quantity": tp1_qty, "reduceOnly":"true","workingType":"MARK_PRICE"}
    if positionSide: p1["positionSide"]=("LONG" if side=="BUY" else "SHORT")
    ok_tp1=ok_tp2=ok_sl=True
    try: _request("POST", ORDER_EP, signed_req=True, data=p1)
    except Exception as e: ok_tp1=False; send_tg(f"⚠️ TP1 Warn {symbol}: {e}")

    # TP2 closePosition for the rest
    p2={"symbol":symbol,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp2_price,
        "closePosition":"true","workingType":"MARK_PRICE"}
    if positionSide: p2["positionSide"]=("LONG" if side=="BUY" else "SHORT")
    try: _request("POST", ORDER_EP, signed_req=True, data=p2)
    except Exception as e: ok_tp2=False; send_tg(f"⚠️ TP2 Warn {symbol}: {e}")

    # SL closePosition
    sp={"symbol":symbol,"side":tp_side,"type":"STOP_MARKET","stopPrice":sl_price,
        "closePosition":"true","workingType":"MARK_PRICE"}
    if positionSide: sp["positionSide"]=("LONG" if side=="BUY" else "SHORT")
    try: _request("POST", ORDER_EP, signed_req=True, data=sp)
    except Exception as e: ok_sl=False; send_tg(f"⚠️ SL Warn {symbol}: {e}")

    if ok_tp1 and ok_tp2 and ok_sl:
        send_tg(f"🧷 تم تعيين TP1/TP2/SL لـ <b>{symbol}</b> ✅")

def fmt_price(p): return f"{p:.8f}".rstrip('0').rstrip('.')

def send_entry_alert(symbol, side, entry, qty, lev):
    tp2 = entry*(1+TAKE_PROFIT_PCT) if side=="BUY" else entry*(1-TAKE_PROFIT_PCT)
    tp1 = entry*(1+TAKE_PROFIT_PCT/2) if side=="BUY" else entry*(1-TAKE_PROFIT_PCT/2)
    sl  = entry*(1-STOP_LOSS_PCT) if side=="BUY" else entry*(1+STOP_LOSS_PCT)
    emoji = "🟢 LONG دخول ✅" if side=="BUY" else "🔴 SHORT دخول ✅"
    msg = (f"{emoji}\n"
           f"<b>{symbol}</b> | Entry {fmt_price(entry)} | SL {fmt_price(sl)}\n"
           f"TP1(50%): {fmt_price(tp1)} | TP2: {fmt_price(tp2)}\n"
           f"Qty {qty:.8f} | Lev {lev}x")
    send_tg(msg)

# ===== tracking & PnL =====
state = {}  # symbol -> dict(side, entry, qty, positionSide, open_ts_ms, tp1_done, tick_size, lot_step, min_qty)

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
    tp1_qty_target = max(round_step(st["qty"]*TP1_SHARE, lot), st["min_qty"])
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
            # Adjust to Breakeven: cancel and set new SL at entry, re-place TP2 closePosition
            try:
                cancel_all_orders(symbol)
                tp2_price = st["entry"]*(1+TAKE_PROFIT_PCT) if st["side"]=="BUY" else st["entry"]*(1-TAKE_PROFIT_PCT)
                tick = st["tick_size"]; tp2_price = round(tp2_price/tick)*tick
                tp_side = "SELL" if st["side"]=="BUY" else "BUY"
                common = {"symbol":symbol,"workingType":"MARK_PRICE"}
                if st["positionSide"]: common["positionSide"]=st["positionSide"]
                _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"TAKE_PROFIT_MARKET","stopPrice":tp2_price,"closePosition":"true"})
                _request("POST", ORDER_EP, signed_req=True, data={**common,"side":tp_side,"type":"STOP_MARKET","stopPrice":st["entry"],"closePosition":"true"})
                send_tg(f"🛡️ تم تعديل SL إلى Breakeven ووضع TP2 لباقي الكمية")
            except Exception as e:
                send_tg(f"⚠️ لم أستطع تعديل الأوامر بعد TP1: {e}")
    # TP2 execution alert (when fully closed we will also send PnL via income)
    if total_closed + 1e-12 >= st["qty"]:
        remaining = st["qty"] - tp1_qty_target if st.get("tp1_done") else st["qty"]
        if remaining > 0:
            acc=0.0; vwap=0.0
            sorted_f = sorted(fills, key=lambda x: x["time"])
            if st.get("tp1_done"):
                skip = tp1_qty_target
                buff=[]
                for t in sorted_f:
                    q=fqty(t); p=fpr(t)
                    if skip>0:
                        d=min(q, skip); skip-=d; left=q-d
                        if left>0: buff.append({"qty":left,"price":p})
                    else:
                        buff.append({"qty":q,"price":p})
                sorted_f = buff
            for t in sorted_f:
                q=float(t["qty"]); p=float(t["price"])
                take=min(q, remaining-acc)
                vwap += p*take; acc += take
                if acc>=remaining-1e-12: break
            if acc>0:
                exec_px = vwap/acc
                send_tg(f"🎯 TP2 تنفيذ فعلي <b>{symbol}</b>\nسعر التنفيذ: {fmt_price(exec_px)} | كمية≈ {remaining:.8f}")

def income_sum(symbol, start_ms, end_ms):
    realized=0.0; fees=0.0; s=start_ms
    while True:
        params={"symbol":symbol,"startTime":int(s),"endTime":int(end_ms),"limit":1000}
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

def send_close_summary_real(symbol, entry, qty, side, open_ts_ms):
    end_ms=int(time.time()*1000 + _time_offset_ms)
    realized, fees = income_sum(symbol, open_ts_ms-60_000, end_ms)
    pnl = realized + fees  # fees negative
    margin = entry*qty/max(1,LEVERAGE)
    roi = (pnl/margin*100) if margin>0 else 0.0
    emoji = "✅ ربح" if pnl>=0 else "❌ خسارة"
    send_tg(f"📘 إغلاق مركز (فعلي)\n<b>{symbol}</b> | {emoji}\nRealized P&L: {pnl:.4f} USDT (PnL {realized:.4f}, Fees {fees:.4f})\nQty {qty:.8f} | Lev {LEVERAGE}x | ROI≈ {roi:.2f}%")

# ===== execution =====
def calc_order_qty(symbol, price):
    bal = account_balance_usdt()
    notional = bal * PER_TRADE_PCT * LEVERAGE
    raw_qty = notional / price
    f = symbol_filters(symbol)
    qty = max(round_step(raw_qty, f["lot_step"]), f["min_qty"])
    return qty

def maybe_trade(symbol, signal, price, hedge):
    positions = open_positions()
    if len(positions) >= MAX_OPEN_TRADES and symbol not in positions: return
    if symbol in positions: return
    side="BUY" if signal=="BUY" else "SELL"
    ensure_leverage(symbol, LEVERAGE)
    qty=calc_order_qty(symbol, price)
    if qty<=0: return
    if RUN_MODE.lower()=="paper":
        send_entry_alert(symbol, side, price, qty, LEVERAGE)
        f=symbol_filters(symbol)
        state[symbol]={"side":side,"entry":price,"qty":qty,"positionSide":("LONG" if side=="BUY" else "SHORT") if hedge else None,
                       "open_ts_ms": int(time.time()*1000 + _time_offset_ms),
                       "tp1_done": False, "tick_size": f["tick_size"], "lot_step": f["lot_step"], "min_qty": f["min_qty"]}
        return
    try:
        posSide=("LONG" if side=="BUY" else "SHORT") if hedge else None
        order=place_market(symbol, side, qty, posSide)
        entry=float(order.get("avgPrice") or price)
        f=symbol_filters(symbol)
        state[symbol]={"side":side,"entry":entry,"qty":qty,"positionSide":posSide,
                       "open_ts_ms": int(time.time()*1000 + _time_offset_ms),
                       "tp1_done": False,"tick_size": f["tick_size"], "lot_step": f["lot_step"], "min_qty": f["min_qty"]}
        send_entry_alert(symbol, side, entry, qty, LEVERAGE)
        place_tp1_tp2_sl(symbol, side, entry, qty, posSide)
    except Exception as e:
        send_tg(f"❌ فشل فتح {symbol}: {e}")

# helpers
def open_positions():
    try:
        data=_request("GET", POSITION_RISK_EP, signed_req=True)
    except Exception:
        return {}
    pos={}
    for p in data:
        qty=float(p["positionAmt"])
        if abs(qty)>1e-12:
            pos[p["symbol"]]=qty
    return pos

def price_now(symbol):
    j=f_get(PRICE_EP, {"symbol":symbol}); return float(j["price"])

# ===== scanning & heartbeat =====
def load_universe():
    if SYMBOLS_CSV and os.path.exists(SYMBOLS_CSV):
        return pd.read_csv(SYMBOLS_CSV)["symbol"].tolist()[:MAX_SYMBOLS]
    data=f_get(TICKER_24H, {"type":"FULL"})
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

_prev_open=set()
def detect_closes_and_notify():
    global _prev_open
    positions=open_positions()
    now_open=set(positions.keys())
    # detect tp1/tp2 fills
    for s in list(state.keys()):
        try: detect_tp_fills(s)
        except Exception as e: print(f"[TP DETECT WARN] {s}: {e}")
    # full close
    closed=[s for s in _prev_open if s not in now_open]
    for s in closed:
        st=state.pop(s, None)
        if st:
            try:
                send_close_summary_real(s, st["entry"], st["qty"], st["side"], st["open_ts_ms"])
            except Exception as e:
                send_tg(f"🔔 تم إغلاق مركز {s} (تعذر حساب P&L بدقة: {e})")
        else:
            send_tg(f"🔔 تم إغلاق مركز {s}")
    _prev_open=now_open

def heartbeat(h,e):
    ts=now_utc().strftime("%Y-%m-%d %H:%M:%SZ")
    msg=f"💗 Heartbeat | إستراتيجية: Consensus\nإشارات: {h} | صفقات: {len(_prev_open)} | أخطاء: {e} | لكل صفقة: {PER_TRADE_PCT*100:.1f}%"
    send_tg(msg)

def main():
    send_tg("🚀 تشغيل Mahdi v4.1r — TP جزئي بأسعار تنفيذ فعلية + SL 1% + P&L فعلي")
    sync_server_time()
    hedge=is_hedge_mode()
    symbols=load_universe()
    last_hb=now_utc()-timedelta(minutes=HEARTBEAT_MIN+1)
    cooldown_until=now_utc()
    while True:
        start=now_utc()
        if start<cooldown_until: time.sleep(2); continue
        h,e,sigs=scan_once(symbols)
        for sym,sig,px in sigs: maybe_trade(sym,sig,px,hedge)
        detect_closes_and_notify()
        if (now_utc()-last_hb)>=timedelta(minutes=HEARTBEAT_MIN):
            heartbeat(h,e); last_hb=now_utc()
        cooldown_until=now_utc()+timedelta(seconds=SCAN_INTERVAL_SEC if h==0 else COOLDOWN_MIN*60)
        time.sleep(max(1, SCAN_INTERVAL_SEC))

if __name__=="__main__":
    main()
