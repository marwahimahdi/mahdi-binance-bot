
# bot_v5_final_pro.py
# Mahdi v5 PRO — Futures USDT-M bot (5 indicators + 60% consensus)
# See header in previous attempt for full details.
import os, time, hmac, hashlib, csv, re, unicodedata, math, traceback
from datetime import datetime, timezone, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd

API_KEY         = os.getenv("API_KEY", "").strip()
API_SECRET      = os.getenv("API_SECRET", "").strip().encode()
RUN_MODE        = os.getenv("RUN_MODE", "paper").lower()
TESTNET         = os.getenv("USE_TESTNET", os.getenv("TESTNET","false")).lower()=="true"
TG_TOKEN        = os.getenv("TELEGRAM_TOKEN", "").strip()
TG_CHATID       = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TG_ENABLED      = os.getenv("TG_ENABLED", "true").lower()=="true"
TG_NOTIFY_UNIVERSE = os.getenv("TG_NOTIFY_UNIVERSE", "false").lower()=="true"
SYMBOLS_CSV     = os.getenv("SYMBOLS_CSV", "").strip()
MAX_SYMBOLS     = int(os.getenv("MAX_SYMBOLS", "25"))
INTERVAL        = os.getenv("INTERVAL", "5m")
KLINES_LIMIT    = int(os.getenv("KLINES_LIMIT", "300"))
SCAN_INTERVAL   = int(os.getenv("SCAN_INTERVAL_SEC", "120"))
CAPITAL_USE_PCT = float(os.getenv("CAPITAL_USE_PCT", os.getenv("TOTAL_CAPITAL_PCT","0.40")))
SLOT_A_PCT      = float(os.getenv("SLOT_A_PCT", "0.06"))
SLOT_A_LEV      = int(os.getenv("SLOT_A_LEV", "10"))
SLOT_B_PCT      = float(os.getenv("SLOT_B_PCT", "0.05"))
SLOT_B_LEV      = int(os.getenv("SLOT_B_LEV", "5"))
MAX_OPEN_POS    = int(os.getenv("MAX_OPEN_POS", os.getenv("MAX_OPEN_TRADES","6")))
TP1_PCT         = float(os.getenv("TP1_PCT", "0.0035"))
TP2_PCT         = float(os.getenv("TP2_PCT", "0.0070"))
TP3_PCT         = float(os.getenv("TP3_PCT", "0.0120"))
SL_PCT          = float(os.getenv("SL_PCT", "0.0075"))
CONSENSUS_MIN   = float(os.getenv("CONSENSUS_MIN", "0.60"))
ADX_MIN         = float(os.getenv("ADX_MIN", "20"))
COOLDOWN_MIN    = int(os.getenv("COOLDOWN_MIN", "60"))
REST_BACKOFF_BASE = float(os.getenv("REST_BACKOFF_BASE", "0.5"))
REST_BACKOFF_MAX  = float(os.getenv("REST_BACKOFF_MAX", "10"))

BASE = "https://testnet.binancefuture.com" if TESTNET else "https://fapi.binance.com"
KLINES_EP   = f"{BASE}/fapi/v1/klines"
PRICE_EP    = f"{BASE}/fapi/v1/ticker/price"
EXINFO_EP   = f"{BASE}/fapi/v1/exchangeInfo"
TIME_EP     = f"{BASE}/fapi/v1/time"
BALANCE_EP  = f"{BASE}/fapi/v2/balance"
MARGIN_TYPE_EP = f"{BASE}/fapi/v1/marginType"
LEVERAGE_EP    = f"{BASE}/fapi/v1/leverage"
ORDER_EP       = f"{BASE}/fapi/v1/order"

def build_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.4, status_forcelist=(418,429,500,502,503,504), allowed_methods=frozenset(['GET','POST','DELETE']))
    adp = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
    s.mount("https://", adp); s.mount("http://", adp)
    s.headers.update({"User-Agent":"MahdiBot/5 PRO","X-MBX-APIKEY":API_KEY})
    return s
session = build_session()

def now_utc(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def normalize_symbol(s: str) -> str:
    if s is None: return ""
    s=str(s); s=unicodedata.normalize("NFKC", s)
    s=re.sub(r"[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060-\u206f]","",s)
    s=s.replace("\xa0","").strip().upper()
    s=s.replace("-PERP","").replace("PERP","").replace("/USDT","USDT")
    s=re.sub(r"[^A-Z0-9]","",s)
    return s

def sign(params: dict)->dict:
    q="&".join([f"{k}={params[k]}" for k in params])
    sig = hmac.new(API_SECRET, q.encode(), hashlib.sha256).hexdigest()
    params["signature"]=sig; return params

def _request(method, url, signed=False, params=None):
    delay=REST_BACKOFF_BASE
    while True:
        try:
            if signed:
                if params is None: params={}
                params["timestamp"]=int(time.time()*1000)
                params=sign(params)
            r=session.request(method,url,params=params,timeout=12)
            if r.status_code in (418,429):
                time.sleep(delay); delay=min(delay*1.7, REST_BACKOFF_MAX); continue
            r.raise_for_status()
            try: return r.json()
            except: return r.text
        except requests.HTTPError: raise
        except Exception: time.sleep(min(delay, REST_BACKOFF_MAX)); delay=min(delay*1.7, REST_BACKOFF_MAX)

def send_tg(text:str):
    if not (TG_ENABLED and TG_TOKEN and TG_CHATID): return
    try:
        url=f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload={"chat_id":TG_CHATID,"text":text,"parse_mode":"HTML","disable_web_page_preview":True}
        session.post(url,data=payload,timeout=12)
    except Exception: pass

def get_price(symbol:str)->float:
    s=normalize_symbol(symbol)
    j=_request("GET", PRICE_EP, params={"symbol":s})
    return float(j["price"])

def get_klines_df(sym:str, interval:str, limit:int)->pd.DataFrame:
    s=normalize_symbol(sym)
    j=_request("GET", KLINES_EP, params={"symbol":s,"interval":interval,"limit":limit})
    cols=["open_time","open","high","low","close","volume","close_time","qv","trades","tb_base","tb_quote","ignore"]
    df=pd.DataFrame(j,columns=cols)
    for c in ("open","high","low","close","volume"):
        df[c]=pd.to_numeric(df[c], errors="coerce")
    return df[["open","high","low","close","volume"]].dropna()

def fetch_exchange_info(): return _request("GET", EXINFO_EP)

def load_universe_from_csv(path:str):
    try:
        with open(path,"r",encoding="utf-8-sig") as f:
            try: has_header=csv.Sniffer().has_header(f.read(2048))
            except Exception: has_header=True
            f.seek(0); syms=[]
            if has_header:
                reader=csv.DictReader(f)
                if reader.fieldnames:
                    lower=[h.lower() for h in reader.fieldnames]
                    if "symbol" in lower:
                        idx=lower.index("symbol")
                        for row in reader:
                            raw=row[reader.fieldnames[idx]]; s=normalize_symbol(raw)
                            if s: syms.append(s)
                    else:
                        f.seek(0); syms=[normalize_symbol(x) for x in f.read().splitlines() if normalize_symbol(x)]
                        if syms and syms[0]=="SYMBOL": syms=syms[1:]
            else:
                f.seek(0); syms=[normalize_symbol(x) for x in f.read().splitlines() if normalize_symbol(x)]
            out=[]; seen=set()
            for s in syms:
                if s and s not in seen: seen.add(s); out.append(s)
            return out
    except Exception as e:
        send_tg(f"⚠️ خطأ قراءة <code>{path}</code>: {e}"); return []

def build_universe():
    if SYMBOLS_CSV:
        syms=load_universe_from_csv(SYMBOLS_CSV); return syms[:MAX_SYMBOLS]
    return []

def verify_symbols(syms):
    out, removed=[], []
    try:
        ex=fetch_exchange_info(); ok=set()
        for s in ex.get("symbols",[]):
            if s.get("status")=="TRADING" and s.get("contractType")=="PERPETUAL" and s.get("quoteAsset")=="USDT":
                ok.add(normalize_symbol(s.get("symbol","")))
        for s in syms:
            ns=normalize_symbol(s)
            if ns not in ok: removed.append(ns); continue
            try:
                _=get_price(ns); out.append(ns); time.sleep(0.03)
            except requests.HTTPError as he:
                if "-1121" in str(he) or "Invalid symbol" in str(he): removed.append(ns)
                else: removed.append(f"{ns} -> {he}")
    except Exception as e:
        send_tg(f"⚠️ تعذر التحقق من الأزواج: {e}")
        return [normalize_symbol(x) for x in syms if normalize_symbol(x)]
    if removed and TG_NOTIFY_UNIVERSE:
        send_tg("⚠️ تم حذف أزواج غير متاحة: "+", ".join(removed))
    return out

# ---- Indicators ----
def ema(s:pd.Series, n:int)->pd.Series: return s.ewm(span=n, adjust=False).mean()
def rsi(s:pd.Series, n:int=14)->pd.Series:
    d=s.diff(); up=np.where(d>0,d,0.0); dn=np.where(d<0,-d,0.0)
    up=pd.Series(up,index=s.index).ewm(span=n,adjust=False).mean()
    dn=pd.Series(dn,index=s.index).ewm(span=n,adjust=False).mean()
    rs=up/(dn+1e-12); return 100-(100/(1+rs))
def true_range(df:pd.DataFrame)->pd.Series:
    pc=df["close"].shift(1)
    return pd.concat([(df["high"]-df["low"]),(df["high"]-pc).abs(),(df["low"]-pc).abs()],axis=1).max(axis=1)
def atr(df:pd.DataFrame, n:int=10)->pd.Series: return true_range(df).ewm(span=n, adjust=False).mean()
def supertrend(df:pd.DataFrame, n:int=10, m:float=3.0):
    hl2=(df["high"]+df["low"])/2.0; a=atr(df,n)
    ub=hl2+m*a; lb=hl2-m*a
    st=pd.Series(index=df.index,dtype=float); tr=pd.Series(index=df.index,dtype=int)
    st.iloc[0]=ub.iloc[0]; tr.iloc[0]=1
    for i in range(1,len(df)):
        tr.iloc[i]=1 if df["close"].iloc[i]>st.iloc[i-1] else (-1 if df["close"].iloc[i]<st.iloc[i-1] else tr.iloc[i-1])
        st.iloc[i]=min(ub.iloc[i], st.iloc[i-1]) if tr.iloc[i]==1 else max(lb.iloc[i], st.iloc[i-1])
    return st, tr, a
def adx(df:pd.DataFrame, n:int=14)->pd.Series:
    up=df["high"].diff(); dn=(-df["low"].diff())
    plus_dm=np.where((up>dn)&(up>0),up,0.0); minus_dm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=true_range(df); tr_n=tr.ewm(span=n,adjust=False).mean()
    plus_di=100*pd.Series(plus_dm,index=df.index).ewm(span=n,adjust=False).mean()/(tr_n+1e-12)
    minus_di=100*pd.Series(minus_dm,index=df.index).ewm(span=n,adjust=False).mean()/(tr_n+1e-12)
    dx=(abs(plus_di-minus_di)/(plus_di+minus_di+1e-12))*100; return dx.ewm(span=n,adjust=False).mean()
def vwap(df:pd.DataFrame)->pd.Series:
    tp=(df["high"]+df["low"]+df["close"])/3.0; vol=df["volume"]
    return (tp*vol).cumsum()/(vol.cumsum()+1e-12)

def indicator_votes(df:pd.DataFrame):
    close=df["close"]
    ema_f=ema(close,12); ema_s=ema(close,26); ema200=ema(close,200)
    macd_line=ema_f-ema_s; macd_sig=ema(macd_line,9); macd_hist=macd_line-macd_sig
    rsi_v=rsi(close,14); st, tr_dir, atr_v=supertrend(df,10,3.0); vwap_s=vwap(df); adx_v=adx(df,14).fillna(0.0)

    votes_buy=votes_sell=0
    # EMA cross
    if ema_f.iloc[-2]<=ema_s.iloc[-2] and ema_f.iloc[-1]>ema_s.iloc[-1]: votes_buy+=1
    elif ema_f.iloc[-2]>=ema_s.iloc[-2] and ema_f.iloc[-1]<ema_s.iloc[-1]: votes_sell+=1
    # MACD hist
    if macd_hist.iloc[-1]>0: votes_buy+=1
    elif macd_hist.iloc[-1]<0: votes_sell+=1
    # RSI
    if rsi_v.iloc[-1]>=55: votes_buy+=1
    elif rsi_v.iloc[-1]<=45: votes_sell+=1
    # SuperTrend
    if tr_dir.iloc[-1]==1: votes_buy+=1
    elif tr_dir.iloc[-1]==-1: votes_sell+=1
    # VWAP
    if close.iloc[-1]>vwap_s.iloc[-1]: votes_buy+=1
    elif close.iloc[-1]<vwap_s.iloc[-1]: votes_sell+=1

    meta={
        "votes_buy":int(votes_buy),
        "votes_sell":int(votes_sell),
        "adx":float(adx_v.iloc[-1]),
        "atr":float(atr_v.iloc[-1]),
        "atr_pct":float(atr_v.iloc[-1]/(close.iloc[-1]+1e-12)),
        "above_200": bool(close.iloc[-1]>ema200.iloc[-1]),
        "close":float(close.iloc[-1]),
        "vwap":float(vwap_s.iloc[-1]),
        "rsi":float(rsi_v.iloc[-1])
    }
    return meta

def decide_signal(meta:dict):
    need=math.ceil(CONSENSUS_MIN*5)
    buy_ok = meta["votes_buy"]>=need and meta["adx"]>=ADX_MIN and meta["above_200"]
    sell_ok = meta["votes_sell"]>=need and meta["adx"]>=ADX_MIN and (not meta["above_200"])
    if buy_ok and not sell_ok: return "BUY"
    if sell_ok and not buy_ok: return "SELL"
    return "NONE"

# ---- Account & orders ----
def get_usdt_balance():
    if RUN_MODE!="live": return 1000.0
    j=_request("GET", BALANCE_EP, signed=True)
    for it in j:
        if it.get("asset")=="USDT": return float(it.get("balance",0))
    return 0.0

def set_isolated(sym):
    if RUN_MODE!="live": return
    try: _request("POST", MARGIN_TYPE_EP, signed=True, params={"symbol":normalize_symbol(sym),"marginType":"ISOLATED"}); time.sleep(0.05)
    except Exception: pass

def set_leverage(sym, lev):
    if RUN_MODE!="live": return
    _request("POST", LEVERAGE_EP, signed=True, params={"symbol":normalize_symbol(sym),"leverage":int(lev)}); time.sleep(0.05)

def new_order(sym, side, qty, order_type="MARKET", **extra):
    if RUN_MODE!="live": return {"paper":True,"symbol":sym,"side":side,"qty":qty,"type":order_type,"extra":extra}
    params={"symbol":normalize_symbol(sym),"side":side,"type":order_type,"quantity":qty}; params.update(extra)
    return _request("POST", ORDER_EP, signed=True, params=params)

def reduce_order(sym, side, order_type, stopPrice=None, closePosition=True, **extra):
    if RUN_MODE!="live": return {"paper":True,"reduceOnly":True,"symbol":sym,"side":side,"type":order_type,"stopPrice":stopPrice}
    params={"symbol":normalize_symbol(sym),"side":side,"type":order_type,"reduceOnly":"true"}
    if stopPrice is not None:
        params["stopPrice"]=stopPrice; params["workingType"]=extra.get("workingType","CONTRACT_PRICE"); params["priceProtect"]="true"
    if closePosition: params["closePosition"]="true"
    return _request("POST", ORDER_EP, signed=True, params=params)

def tick_size(sym): return 3
def fmt_price(p, sym): return float(f"{p:.{tick_size(sym)}f}")
def position_size_from_usdt(sym, usdt, price, lev): return float(f"{(usdt*lev)/price:.3f}")

_last_entry_at={}
def can_enter(sym):
    t=_last_entry_at.get(sym); 
    return True if not t else (datetime.now(timezone.utc)-t)>=timedelta(minutes=COOLDOWN_MIN)
def mark_entry(sym): _last_entry_at[normalize_symbol(sym)]=datetime.now(timezone.utc)

def dynamic_leverage(vb, vs, adx_value):
    strength=max(vb,vs)
    if strength>=4 or adx_value>=25: return SLOT_A_LEV, SLOT_B_LEV
    return 5,5

def enter_trade(sym, direction, meta):
    if not can_enter(sym): return None
    price=get_price(sym); bal=get_usdt_balance(); budget_total=bal*CAPITAL_USE_PCT
    side_mkt="BUY" if direction=="BUY" else "SELL"; side_close="SELL" if direction=="BUY" else "BUY"
    set_isolated(sym); levA,levB=dynamic_leverage(meta["votes_buy"], meta["votes_sell"], meta["adx"])

    results=[]
    for pct, lev in ((SLOT_A_PCT, levA),(SLOT_B_PCT, levB)):
        part=budget_total*pct; set_leverage(sym,lev)
        qty=position_size_from_usdt(sym, part, price, lev)
        if qty<=0: continue
        res=new_order(sym, side_mkt, qty, "MARKET"); results.append(("ENTRY",res))
        if direction=="BUY":
            tp1, tp2, tp3 = fmt_price(price*(1+TP1_PCT),sym), fmt_price(price*(1+TP2_PCT),sym), fmt_price(price*(1+TP3_PCT),sym)
            sl  = fmt_price(price*(1-SL_PCT),sym)
        else:
            tp1, tp2, tp3 = fmt_price(price*(1-TP1_PCT),sym), fmt_price(price*(1-TP2_PCT),sym), fmt_price(price*(1-TP3_PCT),sym)
            sl  = fmt_price(price*(1+SL_PCT),sym)
        for sp in (tp1,tp2,tp3):
            res=reduce_order(sym, side_close, "TAKE_PROFIT_MARKET", stopPrice=sp); results.append(("TP",res)); time.sleep(0.05)
        res=reduce_order(sym, side_close, "STOP_MARKET", stopPrice=sl); results.append(("SL",res)); time.sleep(0.05)

    send_tg(f"🟢 تنفيذ {direction} {sym}\n"
            f"سعر: {price}\n"
            f"أهداف: {tp1}, {tp2}, {tp3}\n"
            f"وقف: {sl}\n"
            f"تصويت: BUY={meta['votes_buy']} SELL={meta['votes_sell']} | ADX={meta['adx']:.1f}\n"
            f"وضع: {'LIVE' if RUN_MODE=='live' else 'PAPER'}")
    mark_entry(sym); return results

# ---- Scan ----
def scan_symbol(sym):
    s=normalize_symbol(sym)
    try:
        _=get_price(s)
    except requests.HTTPError as he:
        if "-1121" in str(he) or "Invalid symbol" in str(he):
            send_tg(f"⚠️ {repr(s)}: Invalid symbol (price-check) — تم حذفه."); return None,"INVALID"
        else: raise
    df=get_klines_df(s, INTERVAL, KLINES_LIMIT)
    if len(df)<210: return None,"SHORT"   # need enough bars for EMA200
    meta=indicator_votes(df); sig=decide_signal(meta)
    return (sig, meta, float(df['close'].iloc[-1])),"OK"

def scan_once(symbols):
    for sym in list(symbols):
        try:
            res, status = scan_symbol(sym)
            if status=="INVALID":
                symbols.remove(normalize_symbol(sym)); continue
            if status!="OK": time.sleep(0.12); continue
            sig, meta, last = res
            if sig in ("BUY","SELL"):
                enter_trade(sym, sig, meta)
            time.sleep(0.12)
        except requests.HTTPError as he:
            msg=str(he); shown=repr(normalize_symbol(sym))
            if "-1121" in msg or "Invalid symbol" in msg:
                try: symbols.remove(normalize_symbol(sym))
                except ValueError: pass
                send_tg(f"⚠️ {shown}: Invalid symbol — تم حذف الزوج من القائمة.")
            else:
                send_tg(f"⚠️ {shown}: HTTP {msg}")
        except Exception as e:
            send_tg(f"⚠️ {repr(normalize_symbol(sym))}: Loop error {e}")

def boot_banner():
    try:
        j=_request("GET", TIME_EP); srv=j.get("serverTime")
        send_tg(f"♻️ Mahdi v5 PRO تشغيل — وضع: {'LIVE' if RUN_MODE=='live' else 'PAPER'} | Testnet: {'On' if TESTNET else 'Off'}\n⏱️ {now_utc()}\nBinanceTime: {srv}")
    except Exception: pass

def main():
    if TG_ENABLED and TG_TOKEN:
        try:
            url=f"https://api.telegram.org/bot{TG_TOKEN}/getMe"
            r=session.get(url,timeout=10)
            print(f"[BOOT] getMe -> {r.status_code} {r.text[:200]}")
        except Exception: pass

    boot_banner()

    symbols=build_universe()
    if not symbols:
        send_tg("⚠️ لا توجد قائمة رموز. تأكد من <code>SYMBOLS_CSV=universe.csv</code>."); return

    symbols=[normalize_symbol(s) for s in symbols if normalize_symbol(s)]
    symbols=list(dict.fromkeys(symbols))[:MAX_SYMBOLS]

    try: send_tg("🧪 DEBUG symbols (repr):\n" + "\n".join(repr(x) for x in symbols))
    except Exception: pass

    symbols=verify_symbols(symbols)
    if not symbols:
        send_tg("⚠️ بعد التحقق لم يتبقَّ أي زوج صالح."); return

    if TG_NOTIFY_UNIVERSE:
        preview=", ".join(symbols[:10]) + ("..." if len(symbols)>10 else "")
        send_tg(f"📊 Universe النهائي (بعد التحقق): {preview} (n={len(symbols)})")

    while True:
        try:
            scan_once(symbols); time.sleep(SCAN_INTERVAL)
        except Exception as e:
            traceback.print_exc(); send_tg(f"⚠️ Loop exception: {e}"); time.sleep(3)

if __name__=="__main__":
    print(f"[BOOT] RUN_MODE={RUN_MODE} TESTNET={TESTNET} TG={TG_ENABLED} CSV={SYMBOLS_CSV}")
    main()
