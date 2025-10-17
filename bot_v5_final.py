
# bot_v5_final.py
# Mahdi v5 — Futures USDT-M bot (final)
import os, time, hmac, hashlib, csv, re, unicodedata, traceback
from datetime import datetime, timezone
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY         = os.getenv("API_KEY", "").strip()
API_SECRET      = os.getenv("API_SECRET", "").strip().encode()
RUN_MODE        = os.getenv("RUN_MODE", "paper").lower()
TESTNET         = os.getenv("USE_TESTNET", os.getenv("TESTNET","false")).lower() == "true"
TG_TOKEN        = os.getenv("TELEGRAM_TOKEN", "").strip()
TG_CHATID       = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TG_ENABLED      = os.getenv("TG_ENABLED", "true").lower() == "true"
TG_NOTIFY_UNIVERSE = os.getenv("TG_NOTIFY_UNIVERSE", "false").lower() == "true"
SYMBOLS_CSV     = os.getenv("SYMBOLS_CSV", "").strip()
MAX_SYMBOLS     = int(os.getenv("MAX_SYMBOLS", "15"))
INTERVAL        = os.getenv("INTERVAL", "5m")
KLINES_LIMIT    = int(os.getenv("KLINES_LIMIT", "150"))
SCAN_INTERVAL   = int(os.getenv("SCAN_INTERVAL_SEC", "120"))
CAPITAL_USE_PCT = float(os.getenv("CAPITAL_USE_PCT", os.getenv("TOTAL_CAPITAL_PCT", "0.40")))
SLOT_A_PCT      = float(os.getenv("SLOT_A_PCT", "0.06"))
SLOT_A_LEV      = int(os.getenv("SLOT_A_LEV", "10"))
SLOT_B_PCT      = float(os.getenv("SLOT_B_PCT", "0.05"))
SLOT_B_LEV      = int(os.getenv("SLOT_B_LEV", "5"))
MAX_OPEN_POS    = int(os.getenv("MAX_OPEN_POS", os.getenv("MAX_OPEN_TRADES", "6")))
TP1_PCT         = float(os.getenv("TP1_PCT", "0.0035"))
TP2_PCT         = float(os.getenv("TP2_PCT", "0.0070"))
TP3_PCT         = float(os.getenv("TP3_PCT", "0.0120"))
SL_PCT          = float(os.getenv("SL_PCT", "0.0075"))
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
    s.headers.update({"User-Agent":"MahdiBot/5", "X-MBX-APIKEY": API_KEY})
    return s
session = build_session()

def now_utc(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def normalize_symbol(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060-\u206f]", "", s)
    s = s.replace("\xa0", "").strip().upper()
    s = s.replace("-PERP","").replace("PERP","").replace("/USDT","USDT")
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

def sign(params: dict) -> dict:
    q = "&".join([f"{k}={params[k]}" for k in params])
    sig = hmac.new(API_SECRET, q.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params

def _request(method, url, signed=False, params=None):
    delay = REST_BACKOFF_BASE
    while True:
        try:
            if signed:
                if params is None: params = {}
                params["timestamp"] = int(time.time()*1000)
                params = sign(params)
            r = session.request(method, url, params=params, timeout=12)
            if r.status_code in (418,429):
                time.sleep(delay); delay = min(delay*1.7, REST_BACKOFF_MAX); continue
            r.raise_for_status()
            try: return r.json()
            except Exception: return r.text
        except requests.HTTPError: raise
        except Exception: time.sleep(min(delay, REST_BACKOFF_MAX)); delay = min(delay*1.7, REST_BACKOFF_MAX)

def send_tg(text: str):
    if not (TG_ENABLED and TG_TOKEN and TG_CHATID): return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHATID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        session.post(url, data=payload, timeout=12)
    except Exception: pass

def get_price(symbol: str) -> float:
    s = normalize_symbol(symbol)
    j = _request("GET", PRICE_EP, params={"symbol": s})
    return float(j["price"])

def get_klines(sym: str, interval: str, limit: int):
    s = normalize_symbol(sym)
    j = _request("GET", KLINES_EP, params={"symbol": s, "interval": interval, "limit": limit})
    return [(int(row[0]), float(row[4])) for row in j]

def fetch_exchange_info(): return _request("GET", EXINFO_EP)

def load_universe_from_csv(path: str):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            f.seek(0)
            try:
                has_header = csv.Sniffer().has_header(f.read(2048))
            except Exception:
                has_header = True
            f.seek(0)
            syms = []
            if has_header:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    lower = [h.lower() for h in reader.fieldnames]
                    if "symbol" in lower:
                        idx = lower.index("symbol")
                        for row in reader:
                            raw = row[reader.fieldnames[idx]]
                            s = normalize_symbol(raw)
                            if s: syms.append(s)
                    else:
                        f.seek(0)
                        syms = [normalize_symbol(x) for x in f.read().splitlines() if normalize_symbol(x)]
                        if syms and syms[0] == "SYMBOL": syms = syms[1:]
            else:
                syms = [normalize_symbol(x) for x in f.read().splitlines() if normalize_symbol(x)]
            out, seen = [], set()
            for s in syms:
                if s and s not in seen:
                    seen.add(s); out.append(s)
            return out
    except Exception as e:
        send_tg(f"⚠️ خطأ قراءة <code>{path}</code>: {e}")
        return []

def build_universe():
    if SYMBOLS_CSV:
        syms = load_universe_from_csv(SYMBOLS_CSV)
        return syms[:MAX_SYMBOLS]
    return []

def verify_symbols(syms):
    out, removed = [], []
    try:
        ex = fetch_exchange_info()
        ok = set()
        for s in ex.get("symbols", []):
            if s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT":
                ok.add(normalize_symbol(s.get("symbol","")))
        for s in syms:
            ns = normalize_symbol(s)
            if ns not in ok:
                removed.append(ns); continue
            try:
                _ = get_price(ns)
                out.append(ns); time.sleep(0.03)
            except requests.HTTPError as he:
                if "-1121" in str(he) or "Invalid symbol" in str(he):
                    removed.append(ns)
                else:
                    removed.append(f"{ns} -> {he}")
    except Exception as e:
        send_tg(f"⚠️ تعذر التحقق من الأزواج: {e}")
        return [normalize_symbol(x) for x in syms if normalize_symbol(x)]
    if removed and TG_NOTIFY_UNIVERSE:
        send_tg("⚠️ تم حذف أزواج غير متاحة: " + ", ".join(removed))
    return out

def indicator_signal(closes): return "NONE"   # disable entries while debugging

def get_usdt_balance():
    if RUN_MODE != "live": return 1000.0
    j = _request("GET", BALANCE_EP, signed=True)
    for it in j:
        if it.get("asset") == "USDT": return float(it.get("balance", 0))
    return 0.0

def set_isolated(sym):
    if RUN_MODE != "live": return
    try:
        _request("POST", MARGIN_TYPE_EP, signed=True, params={"symbol": normalize_symbol(sym), "marginType": "ISOLATED"})
        time.sleep(0.05)
    except Exception: pass

def set_leverage(sym, lev):
    if RUN_MODE != "live": return
    _request("POST", LEVERAGE_EP, signed=True, params={"symbol": normalize_symbol(sym), "leverage": int(lev)})
    time.sleep(0.05)

def new_order(sym, side, qty, order_type="MARKET", **extra):
    if RUN_MODE != "live":
        return {"paper": True, "symbol": sym, "side": side, "qty": qty, "type": order_type, "extra": extra}
    params = {"symbol": normalize_symbol(sym), "side": side, "type": order_type, "quantity": qty}
    params.update(extra)
    return _request("POST", ORDER_EP, signed=True, params=params)

def reduce_order(sym, side, order_type, stopPrice=None, closePosition=True, **extra):
    if RUN_MODE != "live":
        return {"paper": True, "reduceOnly": True, "symbol": sym, "side": side, "type": order_type, "stopPrice": stopPrice}
    params = {"symbol": normalize_symbol(sym), "side": side, "type": order_type, "reduceOnly": "true"}
    if stopPrice is not None:
        params["stopPrice"] = stopPrice
        params["workingType"] = extra.get("workingType", "CONTRACT_PRICE")
        params["priceProtect"] = "true"
    if closePosition: params["closePosition"] = "true"
    return _request("POST", ORDER_EP, signed=True, params=params)

def tick_size(sym): return 3
def fmt_price(p, sym): return float(f"{p:.{tick_size(sym)}f}")
def position_size_from_usdt(sym, usdt, price, lev): return float(f"{(usdt*lev)/price:.3f}")

def enter_trade(sym, direction):
    price = get_price(sym); bal = get_usdt_balance()
    budget_total = bal * CAPITAL_USE_PCT
    side_mkt = "BUY" if direction=="BUY" else "SELL"
    side_close = "SELL" if direction=="BUY" else "BUY"
    set_isolated(sym); results = []
    for pct, lev in ((SLOT_A_PCT, SLOT_A_LEV),(SLOT_B_PCT, SLOT_B_LEV)):
        part = budget_total * pct
        set_leverage(sym, lev)
        qty = position_size_from_usdt(sym, part, price, lev)
        if qty<=0: continue
        res = new_order(sym, side_mkt, qty, "MARKET"); results.append(("ENTRY",res))
        if direction=="BUY":
            tp1, tp2, tp3 = fmt_price(price*(1+TP1_PCT),sym), fmt_price(price*(1+TP2_PCT),sym), fmt_price(price*(1+TP3_PCT),sym)
            sl  = fmt_price(price*(1-SL_PCT),sym)
        else:
            tp1, tp2, tp3 = fmt_price(price*(1-TP1_PCT),sym), fmt_price(price*(1-TP2_PCT),sym), fmt_price(price*(1-TP3_PCT),sym)
            sl  = fmt_price(price*(1+SL_PCT),sym)
        for sp in (tp1,tp2,tp3):
            res = reduce_order(sym, side_close, "TAKE_PROFIT_MARKET", stopPrice=sp); results.append(("TP",res)); time.sleep(0.05)
        res = reduce_order(sym, side_close, "STOP_MARKET", stopPrice=sl); results.append(("SL",res)); time.sleep(0.05)
    send_tg(f"🟢 تنفيذ {direction} {sym}\nسعر: {price}\nأهداف: {tp1}, {tp2}, {tp3}\nوقف: {sl}\nوضع: {'LIVE' if RUN_MODE=='live' else 'PAPER'}")
    return results

def scan_once(symbols):
    for sym in list(symbols):
        s = normalize_symbol(sym)
        try:
            # pre-check price
            try:
                _ = get_price(s)
            except requests.HTTPError as he:
                if "-1121" in str(he) or "Invalid symbol" in str(he):
                    try: symbols.remove(s)
                    except ValueError: pass
                    send_tg(f"⚠️ {repr(s)}: Invalid symbol (price-check) — تم حذفه.")
                    continue
                else:
                    raise
            closes = [c for _, c in get_klines(s, INTERVAL, KLINES_LIMIT)]
            time.sleep(0.12)
            sig = indicator_signal(closes)
            if sig in ("BUY","SELL"): enter_trade(s, sig)
        except requests.HTTPError as he:
            msg = str(he); shown = repr(s)
            if "-1121" in msg or "Invalid symbol" in msg:
                try: symbols.remove(s)
                except ValueError: pass
                send_tg(f"⚠️ {shown}: Invalid symbol — تم حذف الزوج من القائمة.")
            else:
                send_tg(f"⚠️ {shown}: HTTP {msg}")
        except Exception as e:
            send_tg(f"⚠️ {repr(s)}: Loop error {e}")

def boot_banner():
    try:
        j = _request("GET", TIME_EP); srv = j.get("serverTime")
        send_tg(f"♻️ Mahdi v5 تشغيل — وضع: {'LIVE' if RUN_MODE=='live' else 'PAPER'} | Testnet: {'On' if TESTNET else 'Off'}\n⏱️ {now_utc()}\nBinanceTime: {srv}")
    except Exception: pass

def main():
    if TG_ENABLED and TG_TOKEN:
        try:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/getMe"
            r = session.get(url, timeout=10)
            print(f"[BOOT] getMe -> {r.status_code} {r.text[:200]}")
        except Exception: pass

    boot_banner()

    symbols = build_universe()
    if not symbols:
        send_tg("⚠️ لا توجد قائمة رموز. تأكد من <code>SYMBOLS_CSV=universe.csv</code>."); return

    symbols = [normalize_symbol(s) for s in symbols if normalize_symbol(s)]
    symbols = list(dict.fromkeys(symbols))[:MAX_SYMBOLS]

    try:
        dbg = "\n".join(repr(x) for x in symbols)
        send_tg("🧪 DEBUG symbols (repr):\n" + dbg)
    except Exception: pass

    symbols = verify_symbols(symbols)
    if not symbols:
        send_tg("⚠️ بعد التحقق لم يتبقَّ أي زوج صالح."); return

    if TG_NOTIFY_UNIVERSE:
        preview = ", ".join(symbols[:10]) + ("..." if len(symbols)>10 else "")
        send_tg(f"📊 Universe النهائي (بعد التحقق): {preview} (n={len(symbols)})")

    while True:
        try:
            scan_once(symbols); time.sleep(SCAN_INTERVAL)
        except Exception as e:
            traceback.print_exc(); send_tg(f"⚠️ Loop exception: {e}"); time.sleep(3)

if __name__ == "__main__":
    print(f"[BOOT] RUN_MODE={RUN_MODE} TESTNET={TESTNET} TG={TG_ENABLED} CSV={SYMBOLS_CSV}")
    main()
