# -*- coding: utf-8 -*-
# Mahdi v5 PRO — Full Live Edition
# (See chat for full feature list)
import os, time, hmac, hashlib, math, csv, re, unicodedata, traceback
from datetime import datetime, timezone, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd

API_KEY=os.getenv("API_KEY","").strip(); API_SECRET=os.getenv("API_SECRET","").strip().encode()
RUN_MODE=os.getenv("RUN_MODE","live").lower(); USE_TESTNET=os.getenv("USE_TESTNET","false").lower()=="true"
SYMBOLS_CSV=os.getenv("SYMBOLS_CSV","universe.csv").strip(); MAX_SYMBOLS=int(os.getenv("MAX_SYMBOLS","25"))
INTERVAL_MAIN=os.getenv("INTERVAL","5m"); INTERVAL_CONF=os.getenv("INTERVAL_CONFIRM","15m")
KLINES_LIMIT=int(os.getenv("KLINES_LIMIT","300")); SCAN_INTERVAL=int(os.getenv("SCAN_INTERVAL_SEC","120"))
COOLDOWN_MIN=int(os.getenv("COOLDOWN_MIN","60"))
CAPITAL_USE_PCT=float(os.getenv("CAPITAL_USE_PCT",os.getenv("TOTAL_CAPITAL_PCT","0.40")))
SLOT_A_PCT=float(os.getenv("SLOT_A_PCT","0.06")); SLOT_A_LEV=int(os.getenv("SLOT_A_LEV","10"))
SLOT_B_PCT=float(os.getenv("SLOT_B_PCT","0.05")); SLOT_B_LEV=int(os.getenv("SLOT_B_LEV","5"))
MAX_OPEN_TRADES=int(os.getenv("MAX_OPEN_TRADES","10"))
TP1_PCT=float(os.getenv("TP1_PCT","0.0035")); TP2_PCT=float(os.getenv("TP2_PCT","0.0070")); TP3_PCT=float(os.getenv("TP3_PCT","0.0120")); SL_PCT=float(os.getenv("SL_PCT","0.0075"))
CONSENSUS_MIN=float(os.getenv("CONSENSUS_MIN","0.60")); ADX_MIN=float(os.getenv("ADX_MIN","20"))
TG_ENABLED=os.getenv("TG_ENABLED","true").lower()=="true"; TG_TOKEN=os.getenv("TELEGRAM_TOKEN","").strip(); TG_CHATID=os.getenv("TELEGRAM_CHAT_ID","").strip()
TG_NOTIFY_UNIVERSE=os.getenv("TG_NOTIFY_UNIVERSE","false").lower()=="true"; TG_DAILY_SUMMARY=os.getenv("TG_DAILY_SUMMARY","true").lower()=="true"
TG_SUMMARY_MIN=int(os.getenv("TG_SUMMARY_MIN","1440")); TG_HEARTBEAT_MIN=int(os.getenv("TG_HEARTBEAT_MIN","15"))
REST_BACKOFF_BASE=float(os.getenv("REST_BACKOFF_BASE","0.5")); REST_BACKOFF_MAX=float(os.getenv("REST_BACKOFF_MAX","10"))
BASE="https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"
KLINES_EP=f"{BASE}/fapi/v1/klines"; PRICE_EP=f"{BASE}/fapi/v1/ticker/price"; EXINFO_EP=f"{BASE}/fapi/v1/exchangeInfo"; TIME_EP=f"{BASE}/fapi/v1/time"
BALANCE_EP=f"{BASE}/fapi/v2/balance"; ORDER_EP=f"{BASE}/fapi/v1/order"; LEV_EP=f"{BASE}/fapi/v1/leverage"; MARG_EP=f"{BASE}/fapi/v1/marginType"
SIDE_EP=f"{BASE}/fapi/v1/positionSide/dual"; RISK_EP=f"{BASE}/fapi/v1/positionRisk"; OPEN_ORDERS=f"{BASE}/fapi/v1/openOrders"
def build_session():
    s=requests.Session()
    retries=Retry(total=5,backoff_factor=0.4,status_forcelist=(418,429,500,502,503,504),allowed_methods=frozenset(['GET','POST','DELETE']))
    adp=HTTPAdapter(max_retries=retries,pool_connections=100,pool_maxsize=100)
    s.mount("https://",adp); s.mount("http://",adp)
    if API_KEY: s.headers.update({"X-MBX-APIKEY":API_KEY,"User-Agent":"Mahdi v5 PRO"})
    return s
session=build_session()
def send_tg(t):
    if not (TG_ENABLED and TG_TOKEN and TG_CHATID): return
    try: session.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",data={"chat_id":TG_CHATID,"text":t,"parse_mode":"HTML","disable_web_page_preview":True},timeout=12)
    except Exception: pass
def sign(params):
    q="&".join([f"{k}={params[k]}" for k in params]); params["signature"]=hmac.new(API_SECRET,q.encode(),hashlib.sha256).hexdigest(); return params
def _request(method,url,signed=False,params=None):
    delay=REST_BACKOFF_BASE
    while True:
        try:
            p=dict(params or {})
            if signed: p["timestamp"]=int(time.time()*1000); p=sign(p)
            r=session.request(method,url,params=p,timeout=12)
            if r.status_code in (418,429): time.sleep(delay); delay=min(delay*1.7,REST_BACKOFF_MAX); continue
            try: r.raise_for_status()
            except requests.HTTPError as he: he.response=r; raise
            try: return r.json()
            except: return r.text
        except requests.HTTPError: raise
        except Exception: time.sleep(min(delay,REST_BACKOFF_MAX)); delay=min(delay*1.7,REST_BACKOFF_MAX)
def normalize_symbol(s):
    if s is None: return ""
    s=str(s); s=unicodedata.normalize("NFKC",s)
    s=re.sub(r"[^\w/.-]","",s).upper().replace("/USDT","USDT").replace("-PERP","").replace("PERP","").replace("/","")
    return s
SYMF={}
def load_filters_cache():
    SYMF.clear(); ex=_request("GET",EXINFO_EP)
    for s in ex.get("symbols",[]):
        if s.get("status")!="TRADING": continue
        if s.get("contractType")!="PERPETUAL" or s.get("quoteAsset")!="USDT": continue
        sym=s.get("symbol",""); f={f["filterType"]:f for f in s.get("filters",[])}
        ts=float(f.get("PRICE_FILTER",{}).get("tickSize","0.01")); st=float(f.get("LOT_SIZE",{}).get("stepSize","0.001"))
        minq=float(f.get("LOT_SIZE",{}).get("minQty","0")); minnot=float(f.get("MIN_NOTIONAL",{}).get("notional","0"))
        SYMF[sym]={"tickSize":ts,"stepSize":st,"minQty":minq,"minNotional":minnot}
def _decimals_from_step(step): 
    if step<=0: return 8
    s=f"{step:.12f}".rstrip("0"); return len(s.split(".")[1]) if "." in s else 0
def _floor_to_step(x,step): 
    if step<=0: return x
    return math.floor(x/step)*step
def round_price(sym,p):
    f=SYMF.get(normalize_symbol(sym),{}); ts=float(f.get("tickSize",0.01)); v=_floor_to_step(float(p),ts); d=_decimals_from_step(ts); return float(f"{v:.{d}f}")
def round_qty(sym,q):
    f=SYMF.get(normalize_symbol(sym),{}); st=float(f.get("stepSize",0.001)); v=_floor_to_step(float(q),st); d=_decimals_from_step(st); return float(f"{v:.{d}f}")
def enforce_filters(sym,price,qty):
    f=SYMF.get(normalize_symbol(sym),{}); minq=float(f.get("minQty",0)); minnot=float(f.get("minNotional",0))
    if qty<minq: return None,f"qty<{minq}"
    if price*qty<minnot: return None,f"notional<{minnot}"
    return qty,None
def get_price(sym): j=_request("GET",PRICE_EP,params={"symbol":normalize_symbol(sym)}); return float(j["price"])
def get_klines_df(sym, interval, limit):
    j=_request("GET",KLINES_EP,params={"symbol":normalize_symbol(sym),"interval":interval,"limit":limit})
    cols=["t","o","h","l","c","v","ct","qv","tr","tb","tq","ig"]; df=pd.DataFrame(j,columns=cols)
    for c in ("o","h","l","c","v"): df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["open","high","low","close","volume"]].dropna()
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=np.where(d>0,d,0.0); dn=np.where(d<0,-d,0.0)
    up=pd.Series(up,index=s.index).ewm(span=n,adjust=False).mean()
    dn=pd.Series(dn,index=s.index).ewm(span=n,adjust=False).mean()
    rs=up/(dn+1e-12); return 100-(100/(1+rs))
def true_range(df):
    pc=df["close"].shift(1); return pd.concat([(df["high"]-df["low"]),(df["high"]-pc).abs(),(df["low"]-pc).abs()],axis=1).max(axis=1)
def adx(df,n=14):
    up=df["high"].diff(); dn=(-df["low"].diff())
    plus_dm=np.where((up>dn)&(up>0),up,0.0); minus_dm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=true_range(df); tr_n=tr.ewm(span=n,adjust=False).mean()
    plus_di=100*pd.Series(plus_dm,index=df.index).ewm(span=n,adjust=False).mean()/(tr_n+1e-12)
    minus_di=100*pd.Series(minus_dm,index=df.index).ewm(span=n,adjust=False).mean()/(tr_n+1e-12)
    dx=(abs(plus_di-minus_di)/(plus_di+minus_di+1e-12))*100; return dx.ewm(span=n,adjust=False).mean()
def vwap(df):
    tp=(df["high"]+df["low"]+df["close"])/3.0; vol=df["volume"]; return (tp*vol).cumsum()/(vol.cumsum()+1e-12)
def macd(close):
    e12=ema(close,12); e26=ema(close,26); line=e12-e26; signal=ema(line,9); hist=line-signal; return line,signal,hist
def votes_for_df(df5, df15):
    close=df5["close"]; ema200=ema(close,200); macd_line,macd_signal,macd_hist=macd(close); rsi_v=rsi(close,14); vwap_s=vwap(df5); adx_v=adx(df5,14).fillna(0.0)
    votes_buy=votes_sell=0
    # MACD hist
    if macd_hist.iloc[-1]>0: votes_buy+=1
    elif macd_hist.iloc[-1]<0: votes_sell+=1
    # RSI
    if rsi_v.iloc[-1]>=55: votes_buy+=1
    elif rsi_v.iloc[-1]<=45: votes_sell+=1
    # VWAP
    if close.iloc[-1]>vwap_s.iloc[-1]: votes_buy+=1
    elif close.iloc[-1]<vwap_s.iloc[-1]: votes_sell+=1
    # EMA200 trend vote
    if close.iloc[-1]>ema200.iloc[-1]: votes_buy+=1
    else: votes_sell+=1
    # Confirmation 15m MACD
    _,_,h15=macd(df15["close"])
    if h15.iloc[-1]>0: votes_buy+=1
    elif h15.iloc[-1]<0: votes_sell+=1
    meta={"votes_buy":int(votes_buy),"votes_sell":int(votes_sell),"adx":float(adx_v.iloc[-1]),"above_200": bool(close.iloc[-1]>ema200.iloc[-1]),"close":float(close.iloc[-1])}
    return meta
def decide(meta):
    need=math.ceil(CONSENSUS_MIN*5)
    buy_ok=meta["votes_buy"]>=need and meta["adx"]>=ADX_MIN and meta["above_200"]
    sell_ok=meta["votes_sell"]>=need and meta["adx"]>=ADX_MIN and (not meta["above_200"])
    if buy_ok and not sell_ok: return "BUY"
    if sell_ok and not buy_ok: return "SELL"
    return "NONE"
def get_usdt_balance():
    if RUN_MODE!="live": return 1000.0
    j=_request("GET",BALANCE_EP,signed=True)
    for it in j:
        if it.get("asset")=="USDT": return float(it.get("balance",0))
    return 0.0
def set_one_way():
    try: _request("POST",SIDE_EP,signed=True,params={"dualSidePosition":"false"})
    except Exception as e: send_tg(f"⚠️ لم أستطع فرض one-way: {e}")
def set_isolated(sym):
    try: _request("POST",MARG_EP,signed=True,params={"symbol":normalize_symbol(sym),"marginType":"ISOLATED"})
    except Exception: pass
def set_leverage(sym,lev):
    try: _request("POST",LEV_EP,signed=True,params={"symbol":normalize_symbol(sym),"leverage":int(lev)})
    except Exception as e: send_tg(f"⚠️ {sym} leverage: {e}")
def new_order(sym,side,qty,typ="MARKET",**extra):
    if RUN_MODE!="live": return {"paper":True,"symbol":sym,"side":side,"qty":qty,"type":typ,"extra":extra}
    p={"symbol":normalize_symbol(sym),"side":side,"type":typ}; p.update(extra)
    if typ=="MARKET": p["quantity"]=qty
    return _request("POST",ORDER_EP,signed=True,params=p)
def open_orders(sym): return _request("GET",OPEN_ORDERS,signed=True,params={"symbol":normalize_symbol(sym)})
def position_risk(): return _request("GET",RISK_EP,signed=True,params={})
def place_exit_order_close(sym,side,stop_price,is_tp,tag):
    p={"symbol":normalize_symbol(sym),"side":"SELL" if side=="BUY" else "BUY","type":"TAKE_PROFIT_MARKET" if is_tp else "STOP_MARKET","stopPrice":round_price(sym,float(stop_price)),"closePosition":"true","workingType":"CONTRACT_PRICE","priceProtect":"true","reduceOnly":"true","timeInForce":"GTC"}
    try:
        res=new_order(sym,p["side"],None,p["type"],**{k:v for k,v in p.items() if k not in ("side","type")}); send_tg(f"✅ [{tag}] {sym} @ {p['stopPrice']} ({'TP' if is_tp else 'SL'})"); return res
    except requests.HTTPError as he:
        body=he.response.text if getattr(he,"response",None) else str(he); send_tg(f"🚫 [{tag}] {sym} فشل: {body}"); raise
def place_brackets(sym,side,entry_price):
    if side=="BUY":
        tp1=entry_price*(1+TP1_PCT); tp2=entry_price*(1+TP2_PCT); tp3=entry_price*(1+TP3_PCT); sl=entry_price*(1-SL_PCT)
    else:
        tp1=entry_price*(1-TP1_PCT); tp2=entry_price*(1-TP2_PCT); tp3=entry_price*(1-TP3_PCT); sl=entry_price*(1+SL_PCT)
    tp1=round_price(sym,tp1); tp2=round_price(sym,tp2); tp3=round_price(sym,tp3); sl=round_price(sym,sl)
    place_exit_order_close(sym,side,tp1,True,"TP1"); place_exit_order_close(sym,side,tp2,True,"TP2"); place_exit_order_close(sym,side,tp3,True,"TP3"); place_exit_order_close(sym,side,sl,False,"SL")
def exits_present(sym):
    try:
        ods=open_orders(sym); types={o.get("type") for o in ods}
        return (("TAKE_PROFIT_MARKET" in types) or ("TAKE_PROFIT" in types)) and (("STOP_MARKET" in types) or ("STOP" in types))
    except Exception: return False
def reconcile_exits():
    try:
        poss=position_risk()
        for p in poss:
            sym=p.get("symbol"); amt=float(p.get("positionAmt","0"))
            if abs(amt)<1e-12: continue
            side="BUY" if amt>0 else "SELL"
            if not exits_present(sym):
                px=get_price(sym); place_brackets(sym,side,px); send_tg(f"🛠️ أعيد إنشاء SL/TP لـ {sym} (side={side})")
    except Exception as e: send_tg(f"⚠️ reconcile_exits: {e}")
def position_size_from_usdt(sym, usdt, price, lev): return float(f"{(usdt*lev)/price:.6f}")
def load_csv_symbols(path):
    try:
        with open(path,"r",encoding="utf-8") as f: rows=f.read().splitlines()
        out=[]
        for r in rows:
            r=r.strip()
            if not r: continue
            if "," in r: r=r.split(",")[0]
            s=normalize_symbol(r)
            if s and s not in out: out.append(s)
        return out
    except Exception as e: send_tg(f"⚠️ خطأ قراءة {path}: {e}"); return []
def verify_symbols(syms):
    ok=set(); ex=_request("GET",EXINFO_EP)
    for s in ex.get("symbols",[]):
        if s.get("status")=="TRADING" and s.get("contractType")=="PERPETUAL" and s.get("quoteAsset")=="USDT":
            ok.add(normalize_symbol(s.get("symbol","")))
    out=[]; removed=[]
    for s in syms:
        ns=normalize_symbol(s)
        if ns not in ok: removed.append(ns); continue
        try: _=_request("GET",PRICE_EP,params={"symbol":ns}); out.append(ns); time.sleep(0.03)
        except requests.HTTPError as he:
            body=he.response.text if getattr(he,"response",None) else str(he); removed.append(f"{ns} -> {body}")
    if removed and TG_NOTIFY_UNIVERSE: send_tg("⚠️ أزواج محذوفة: "+", ".join(removed))
    return out[:MAX_SYMBOLS]
stats={"started_at":datetime.now(timezone.utc).isoformat(),"last_summary":datetime.now(timezone.utc).isoformat(),"universe_n":0,"scans":0,"signals_buy":0,"signals_sell":0,"entries":0,"invalid_removed":0,"open_trades":0}
last_entry_time={}
def can_enter(sym):
    t=last_entry_time.get(sym); 
    return True if not t else (datetime.now(timezone.utc)-t)>=timedelta(minutes=COOLDOWN_MIN)
def mark_enter(sym): last_entry_time[normalize_symbol(sym)]=datetime.now(timezone.utc)
def maybe_daily_summary():
    if not TG_DAILY_SUMMARY: return
    try:
        last=datetime.fromisoformat(stats["last_summary"])
        if datetime.now(timezone.utc)-last>=timedelta(minutes=TG_SUMMARY_MIN):
            send_tg(f"🧾 <b>ملخص Mahdi v5 PRO</b>\n⏱️ منذ: {stats['started_at']}\n📊 Universe: {stats['universe_n']}\n🔁 فحوصات: {stats['scans']}\n🟢 BUY: {stats['signals_buy']} | 🔴 SELL: {stats['signals_sell']}\n✅ صفقات: {stats['entries']}\n📈 مراكز: {stats['open_trades']}")
            stats["last_summary"]=datetime.now(timezone.utc).isoformat()
    except Exception: pass
def scan_symbol(sym):
    df5=get_klines_df(sym,INTERVAL_MAIN,KLINES_LIMIT); df15=get_klines_df(sym,INTERVAL_CONF,max(150,KLINES_LIMIT//3))
    if len(df5)<210 or len(df15)<60: return "NONE", None
    meta=votes_for_df(df5,df15); sig=decide(meta); return sig, meta
def active_trades_count():
    try:
        prs=position_risk(); c=0
        for p in prs:
            if abs(float(p.get("positionAmt","0")))>0: c+=1
        return c
    except Exception: return 0
def enter_trade(sym,direction,meta):
    if not can_enter(sym): return
    open_now=active_trades_count(); stats["open_trades"]=open_now
    if open_now>=MAX_OPEN_TRADES: return
    price=get_price(sym); bal=get_usdt_balance(); budget_total=bal*CAPITAL_USE_PCT; side_mkt="BUY" if direction=="BUY" else "SELL"
    set_one_way(); set_isolated(sym)
    # Slot A
    set_leverage(sym,SLOT_A_LEV); price_r=round_price(sym,price)
    qtyA=position_size_from_usdt(sym,budget_total*SLOT_A_PCT,price_r,SLOT_A_LEV); qtyA=round_qty(sym,qtyA); ok,why=enforce_filters(sym,price_r,qtyA)
    if ok is None: send_tg(f"⛔ تخطي {sym}: {why} (p={price_r}, q={qtyA})"); return
    _=new_order(sym,side_mkt,qtyA,"MARKET")
    # Slot B
    set_leverage(sym,SLOT_B_LEV); qtyB=position_size_from_usdt(sym,budget_total*SLOT_B_PCT,price_r,SLOT_B_LEV); qtyB=round_qty(sym,qtyB); ok,why=enforce_filters(sym,price_r,qtyB)
    if ok is not None: _=new_order(sym,side_mkt,qtyB,"MARKET")
    time.sleep(0.3); place_brackets(sym,side_mkt,price_r)
    send_tg(f"🟢 تنفيذ {direction} {sym}\nسعر: {price_r}\nتصويت: BUY={meta['votes_buy']} SELL={meta['votes_sell']} | ADX={meta['adx']:.1f}\nوضع: {'LIVE' if RUN_MODE=='live' else 'PAPER'}")
    mark_enter(sym); stats["entries"]+=1; stats["open_trades"]=active_trades_count()
def scan_once(symbols):
    for sym in list(symbols):
        try:
            sig,meta=scan_symbol(sym)
            if sig=="BUY": stats["signals_buy"]+=1; enter_trade(sym,sig,meta)
            elif sig=="SELL": stats["signals_sell"]+=1; enter_trade(sym,sig,meta)
            stats["scans"]+=1; time.sleep(0.1)
        except requests.HTTPError as he:
            body=he.response.text if getattr(he,"response",None) else str(he); send_tg(f"⚠️ {repr(normalize_symbol(sym))}: HTTP {body}")
        except Exception as e:
            send_tg(f"⚠️ {repr(normalize_symbol(sym))}: Loop {e}")
    reconcile_exits(); maybe_daily_summary()
def main():
    try: t=_request("GET",TIME_EP); srv=t.get("serverTime")
    except Exception: srv="?"
    syms=verify_symbols(load_csv_symbols(SYMBOLS_CSV))
    if not syms: send_tg("⚠️ لا توجد رموز صالحة بعد التحقق."); return
    load_filters_cache(); stats["universe_n"]=len(syms)
    if TG_NOTIFY_UNIVERSE: send_tg(f"📊 Universe: {', '.join(syms[:10])}{'...' if len(syms)>10 else ''} (n={len(syms)})")
    send_tg(f"♻️ Mahdi v5 PRO تشغيل — وضع: {'LIVE' if RUN_MODE=='live' else 'PAPER'} | Testnet: {'On' if USE_TESTNET else 'Off'}\n⏱️ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\nBinanceTime: {srv}")
    last_hb=time.time()
    while True:
        try:
            scan_once(syms)
            if time.time()-last_hb>TG_HEARTBEAT_MIN*60: send_tg('💚 Heartbeat: يعمل بدون أخطاء'); last_hb=time.time()
            time.sleep(SCAN_INTERVAL)
        except KeyboardInterrupt: break
        except Exception as e: traceback.print_exc(); send_tg(f"⚠️ Loop exception: {e}"); time.sleep(2)
if __name__=="__main__": main()
