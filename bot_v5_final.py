# MahdiBot v5 PRO — Conservative + Dual TF + Hourly Summary + Advanced Risk Add-ons (ALL Enabled)
# ----------------------------------------------------------------------------------
# هذا الإصدار يدمج كل الإضافات المقترحة:
# - Auto-TopN بدون CSV + فلتر سيولة دنيا (MIN_QUOTEVOL_USDT)
# - فلتر اتجاه بالفريم الأعلى 15m (EMA50/MACD-hist) + خيار EMA200
# - 5 مؤشرات على 5m + إجماع 60%
# - 3 أهداف TP (50%/30%/20%) + وقف متحرك ATR + Breakeven بعد TP1
# - Trailing أقوى بعد TP2 (POST_TP2_TRAIL_MULT)
# - حد أقصى للصفقات المتزامنة (MAX_CONCURRENT_TRADES)
# - فترة تبريد بعد SL (SYMBOL_COOLDOWN_MIN) + منع إعادة الدخول (REENTRY_LOCK_BARS)
# - فلتر تمويل (FUNDING_ABS_MAX)
# - حد خسارة يومي (DAILY_LOSS_STOP_USDT أو DAILY_LOSS_STOP_PCT)
# - ملخص كل ساعة + عدّادات + جدول Top-N
# ----------------------------------------------------------------------------------

import os, time, hmac, hashlib, math
from datetime import datetime, timezone, date
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ===================== تحميل البيئة =====================
load_dotenv()

API_KEY  = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

USE_TESTNET = os.getenv("USE_TESTNET", "false").lower() == "true"
BASE_URL = os.getenv("BASE_URL") or ("https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com")

RUN_MODE = os.getenv("RUN_MODE", "paper").lower()  # paper | real
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "15"))
INTERVAL = os.getenv("INTERVAL", "5m")              # الفريم الرئيسي
HTF_INTERVAL = os.getenv("HTF_INTERVAL", "15m")      # الفريم الأعلى
DEFAULT_MARGIN_TYPE = os.getenv("DEFAULT_MARGIN_TYPE", "ISOLATED").upper()
LEVERAGE = int(os.getenv("LEVERAGE", "5"))

# محافظ + نسب الإغلاق
RISK_PROFILE = os.getenv("RISK_PROFILE", "conservative").lower()
MAX_RISK_PCT   = float(os.getenv("MAX_RISK_PCT", "0.0035"))   # 0.35% من الرصيد
STOP_ATR_MULT   = float(os.getenv("STOP_ATR_MULT", "1.3"))
TRAIL_ATR_MULT  = float(os.getenv("TRAIL_ATR_MULT", "0.8"))
TP1_R_MULT      = float(os.getenv("TP1_R_MULT", "0.9"))
TP2_R_MULT      = float(os.getenv("TP2_R_MULT", "1.6"))
TP3_R_MULT      = float(os.getenv("TP3_R_MULT", "2.3"))
TP1_PCT_CLOSE   = float(os.getenv("TP1_PCT_CLOSE", "0.50"))
TP2_PCT_CLOSE   = float(os.getenv("TP2_PCT_CLOSE", "0.30"))
TP3_PCT_CLOSE   = float(os.getenv("TP3_PCT_CLOSE", "0.20"))

# فلاتر إشارة
CONSENSUS_MIN   = float(os.getenv("CONSENSUS_MIN", "0.6"))
ADX_MIN         = float(os.getenv("ADX_MIN", "15"))
RSI_BUY_MAX     = float(os.getenv("RSI_BUY_MAX", "70"))
RSI_SELL_MIN    = float(os.getenv("RSI_SELL_MIN", "30"))
MIN_ATR_PCT     = float(os.getenv("MIN_ATR_PCT", "0.20"))

# إضافات متقدمة (مفعّلة افتراضيًا)
BREAKEVEN_AFTER_TP1 = os.getenv("BREAKEVEN_AFTER_TP1", "true").lower() == "true"
BE_OFFSET_MULT      = float(os.getenv("BE_OFFSET_MULT", "0.05"))  # 0.05 * ATR فوق/تحت الدخول
POST_TP2_TRAIL_MULT = float(os.getenv("POST_TP2_TRAIL_MULT", "1.3"))
MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "4"))
SYMBOL_COOLDOWN_MIN = int(os.getenv("SYMBOL_COOLDOWN_MIN", "30"))
REENTRY_LOCK_BARS   = int(os.getenv("REENTRY_LOCK_BARS", "5"))
MIN_QUOTEVOL_USDT   = float(os.getenv("MIN_QUOTEVOL_USDT", "5e7"))  # 50M
EMA200_FILTER       = os.getenv("EMA200_FILTER", "true").lower() == "true"
FUNDING_ABS_MAX     = float(os.getenv("FUNDING_ABS_MAX", "0.0005"))  # 0.05%
DAILY_LOSS_STOP_USDT= float(os.getenv("DAILY_LOSS_STOP_USDT", "-999999"))  # عطّل إذا كبير جدًا
DAILY_LOSS_STOP_PCT = float(os.getenv("DAILY_LOSS_STOP_PCT", "-100.0"))   # عطّل إذا -100%

SYMBOLS_CSV = os.getenv("SYMBOLS_CSV", "").strip()

# ===================== مسارات Binance =====================
EXCHANGE_INFO = f"{BASE_URL}/fapi/v1/exchangeInfo"
TICKER_24H    = f"{BASE_URL}/fapi/v1/ticker/24hr"
KLINES_EP     = f"{BASE_URL}/fapi/v1/klines"
PRICE_EP      = f"{BASE_URL}/fapi/v1/ticker/price"
BALANCE_EP    = f"{BASE_URL}/fapi/v2/balance"
LEVERAGE_EP   = f"{BASE_URL}/fapi/v1/leverage"
MARGIN_TYPE_EP= f"{BASE_URL}/fapi/v1/marginType"
ORDER_EP      = f"{BASE_URL}/fapi/v1/order"
OPEN_ORDERS   = f"{BASE_URL}/fapi/v1/openOrders"
FUNDING_EP    = f"{BASE_URL}/fapi/v1/premiumIndex"

SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})

# ===================== عدّادات الجلسة =====================
METRICS = {"signals": 0, "trades": 0, "tp_hits": 0, "sl_hits": 0, "realized_pnl": 0.0}
_last_summary_ts = 0.0
SUMMARY_EVERY_SEC = int(os.getenv("SUMMARY_EVERY_SEC", str(60*60)))
_session_day = date.today()
_session_start_balance = None

# ====== عرض تفاصيل قائمة الأزواج المختارة (Top-N) إلى تيليجرام ======

def _fmt_qv(x: float) -> str:
    try:
        x = float(x)
        if x >= 1e9: return f"{x/1e9:.2f}B"
        if x >= 1e6: return f"{x/1e6:.2f}M"
        if x >= 1e3: return f"{x/1e3:.2f}K"
        return f"{x:.0f}"
    except Exception:
        return str(x)


def send_universe_details(symbols: List[str]) -> None:
    try:
        data = f_get(TICKER_24H)
        df = pd.DataFrame(data)
        if df.empty:
            return
        df["symbol"] = df["symbol"].astype(str).str.upper()
        for c in ["quoteVolume", "count", "volume", "lastPrice", "priceChangePercent"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["symbol"].isin([s.upper() for s in symbols])]
        if df.empty:
            return
        df = df.sort_values(["quoteVolume", "count"], ascending=[False, False])
        lines = ["📋 Top-N Details (24h)", "┌────────────┬──────────┬───────┐", "│ Symbol     │ QuoteVol │ Trades│", "├────────────┼──────────┼───────┤"]
        for _, r in df.iterrows():
            sym = f"{r['symbol']:<10}"
            qv  = _fmt_qv(r.get("quoteVolume", 0))
            cnt = int(r.get("count", 0)) if not pd.isna(r.get("count", np.nan)) else 0
            lines.append(f"│ {sym} │ {qv:>8} │ {cnt:>5} │")
        lines.append("└────────────┴──────────┴───────┘")
        send_tg("\n".join(lines))
    except Exception:
        pass

# ===================== أدوات عامة =====================
class HttpErr(requests.HTTPError):
    pass

def now_ts_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def send_tg(msg: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        print("[TG] Missing TG_TOKEN or TG_CHAT_ID — skipping send.")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = {"chat_id": TG_CHAT_ID, "text": msg}
        r = requests.post(url, json=data, timeout=10)
        if r.status_code != 200:
            try:
                body = r.text
            except Exception:
                body = "<no body>"
            print(f"[TG] sendMessage failed {r.status_code}: {body}")
    except Exception as e:
        print(f"[TG] Exception sending: {e}")

def _clean_symbol(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u200f", "").replace("\u200e", "")
    return s.strip().upper()

def f_get(url: str, params: Optional[Dict[str, Any]] = None):
    try:
        r = SESSION.get(url, params=params, timeout=15)
        if r.status_code >= 400:
            raise HttpErr(f"GET {url} -> {r.status_code} {r.text}")
        try:
            return r.json()
        except Exception:
            return r.text
    except requests.RequestException as e:
        raise HttpErr(str(e))

def f_signed(method: str, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not API_KEY or not API_SECRET:
        raise HttpErr("Missing API credentials")
    p = params.copy() if params else {}
    p["timestamp"] = now_ts_ms()
    query = "&".join([f"{k}={p[k]}" for k in sorted(p.keys())])
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    p["signature"] = sig
    headers = {"X-MBX-APIKEY": API_KEY}
    try:
        if method == "GET":
            r = SESSION.get(url, params=p, headers=headers, timeout=15)
        elif method == "POST":
            r = SESSION.post(url, params=p, headers=headers, timeout=15)
        else:
            r = SESSION.delete(url, params=p, headers=headers, timeout=15)
        if r.status_code >= 400:
            raise HttpErr(f"{method} {url} -> {r.status_code} {r.text}")
        return r.json()
    except requests.RequestException as e:
        raise HttpErr(str(e))

# ===================== اختيار الأزواج Auto-TopN =====================

def fetch_valid_perp_usdt() -> List[str]:
    info = f_get(EXCHANGE_INFO)
    out = []
    for s in info.get("symbols", []):
        if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING":
            out.append(_clean_symbol(s.get("symbol")))
    return sorted(set(out))

def build_auto_universe(top_n: Optional[int] = None) -> List[str]:
    top_n = int(top_n or MAX_SYMBOLS)
    tickers = f_get(TICKER_24H)
    df = pd.DataFrame(tickers)
    if df.empty:
        return []
    df["symbol"] = df["symbol"].astype(str).map(_clean_symbol)
    df = df[df["symbol"].isin(fetch_valid_perp_usdt())]
    for col in ["quoteVolume", "count", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # فلتر سيولة دنيا
    if "quoteVolume" in df.columns:
        df = df[df["quoteVolume"] >= MIN_QUOTEVOL_USDT]
    df = df.sort_values(["quoteVolume", "count"], ascending=[False, False]).head(top_n)
    return df["symbol"].tolist()

def verify_symbol(symbol: str) -> bool:
    s = _clean_symbol(symbol)
    try:
        data = f_get(EXCHANGE_INFO, {"symbol": s})
        if not data or not data.get("symbols"):
            return False
        _ = f_get(PRICE_EP, {"symbol": s})
        return True
    except HttpErr:
        return False

# ===================== بيانات السوق والمؤشرات =====================

def get_klines(symbol: str, interval: str, limit: int = 500) -> Optional[pd.DataFrame]:
    s = _clean_symbol(symbol)
    try:
        raw = f_get(KLINES_EP, {"symbol": s, "interval": interval, "limit": limit})
        cols = ["open_time","open","high","low","close","volume","close_time","qv","trades","taker_base","taker_quote","ignore"]
        df = pd.DataFrame(raw, columns=cols)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df
    except HttpErr:
        return None

def indicators(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    macds = macd.ewm(span=9, adjust=False).mean()
    hist  = macd - macds

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = (pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1)).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-9))
    minus_di= 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-9))
    dx = 100 * ( (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9) )
    adx = dx.ewm(alpha=1/14, adjust=False).mean()

    ma20 = close.rolling(20).mean()
    std20= close.rolling(20).std(ddof=0)
    bb_up = ma20 + 2*std20
    bb_dn = ma20 - 2*std20

    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200= close.ewm(span=200, adjust=False).mean()
    vwap  = (close*vol).cumsum() / (vol.cumsum() + 1e-9)

    return {
        "close": close, "ema12": ema12, "ema26": ema26, "macd": macd, "macds": macds, "hist": hist,
        "rsi": rsi, "adx": adx, "atr": atr, "ema50": ema50, "ema200": ema200, "bb_up": bb_up, "bb_dn": bb_dn, "vwap": vwap
    }

# فلتر الاتجاه بالفريم الأعلى

def htf_trend(df_htf: pd.DataFrame) -> str:
    if df_htf is None or df_htf.empty:
        return "NEUTRAL"
    ind = indicators(df_htf)
    c = float(ind["close"].iloc[-1])
    ema50 = float(ind["ema50"].iloc[-1])
    ema200 = float(ind["ema200"].iloc[-1])
    hist = float(ind["hist"].iloc[-1])
    bull = c > ema50 and hist > 0 and (not EMA200_FILTER or c > ema200)
    bear = c < ema50 and hist < 0 and (not EMA200_FILTER or c < ema200)
    if bull: return "BULLISH"
    if bear: return "BEARISH"
    return "NEUTRAL"

# تمويل

def funding_abs(symbol: str) -> float:
    try:
        data = f_get(FUNDING_EP, {"symbol": _clean_symbol(symbol)})
        rate = float(data.get("lastFundingRate", 0.0)) if isinstance(data, dict) else 0.0
        return abs(rate)
    except Exception:
        return 0.0

# ===================== أوامر وحجم صفقة =====================
_symbol_filters_cache: Dict[str, Dict[str, float]] = {}

def symbol_filters(symbol: str) -> Dict[str, float]:
    s = _clean_symbol(symbol)
    if s in _symbol_filters_cache:
        return _symbol_filters_cache[s]
    info = f_get(EXCHANGE_INFO, {"symbol": s})
    sym = info.get("symbols", [{}])[0]
    step = 1e-3; tick = 1e-2; min_qty = 0.0
    for flt in sym.get("filters", []):
        if flt.get("filterType") == "LOT_SIZE":
            step = float(flt.get("stepSize", "0.001"))
            min_qty = float(flt.get("minQty", "0.0"))
        if flt.get("filterType") == "PRICE_FILTER":
            tick = float(flt.get("tickSize", "0.01"))
    _symbol_filters_cache[s] = {"step": step, "tick": tick, "min_qty": min_qty}
    return _symbol_filters_cache[s]

def round_step(x: float, step: float) -> float:
    return math.floor(x / step) * step

def get_balance_usdt() -> float:
    try:
        bals = f_signed("GET", BALANCE_EP, {})
        for b in bals:
            if b.get("asset") == "USDT":
                return float(b.get("balance", 0.0))
    except HttpErr:
        pass
    return 0.0

def get_price(symbol: str) -> float:
    p = f_get(PRICE_EP, {"symbol": _clean_symbol(symbol)})
    return float(p.get("price"))

def ensure_margin_type(symbol: str, mtype: str = "ISOLATED") -> bool:
    s = _clean_symbol(symbol)
    try:
        f_signed("POST", MARGIN_TYPE_EP, {"symbol": s, "marginType": mtype})
        return True
    except HttpErr as he:
        if "-4046" in str(he): return True
        if "-1121" in str(he): return False
        return False

def ensure_leverage(symbol: str, lev: int) -> bool:
    s = _clean_symbol(symbol)
    try:
        f_signed("POST", LEVERAGE_EP, {"symbol": s, "leverage": lev})
        return True
    except HttpErr:
        return False

def calc_position_size(symbol: str, entry: float, stop: float) -> float:
    bal = get_balance_usdt()
    risk_cap = bal * MAX_RISK_PCT
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0: return 0.0
    qty = risk_cap / risk_per_unit
    flt = symbol_filters(symbol)
    qty = max(round_step(qty, flt["step"]), flt["min_qty"])
    return max(qty, 0.0)

def place_order(symbol: str, side: str, qty: float, order_type: str = "MARKET") -> Dict[str, Any]:
    s = _clean_symbol(symbol)
    if RUN_MODE != "real":
        price = get_price(s)
        return {"status": "FILLED", "symbol": s, "side": side, "origQty": qty, "price": price, "mode": "paper"}
    return f_signed("POST", ORDER_EP, {"symbol": s, "side": side, "quantity": qty, "type": order_type})

def place_conditional(symbol: str, side: str, trigger_type: str, trigger_price: float, qty: float) -> Dict[str, Any]:
    s = _clean_symbol(symbol)
    params = {"symbol": s, "side": side, "type": trigger_type, "stopPrice": trigger_price, "closePosition": False, "reduceOnly": True, "quantity": qty, "workingType": "MARK_PRICE"}
    if RUN_MODE != "real":
        return {"status": "NEW", "type": trigger_type, "stopPrice": trigger_price, "qty": qty, "mode": "paper"}
    return f_signed("POST", ORDER_EP, params)

def cancel_all(symbol: str):
    s = _clean_symbol(symbol)
    try:
        if RUN_MODE == "real":
            f_signed("DELETE", OPEN_ORDERS, {"symbol": s})
    except HttpErr:
        pass

# ===================== إدارة الحالة والتنبيهات =====================
open_positions: Dict[str, Dict[str, Any]] = {}
_symbol_cooldown_until: Dict[str, float] = {}
_reentry_lock_until: Dict[str, float] = {}


def _interval_minutes(iv: str) -> int:
    try:
        return int(iv.replace('m',''))
    except Exception:
        return 5

def fmt_votes(v: Dict[str,int]) -> str:
    return f"B={v.get('BUY',0)} S={v.get('SELL',0)} H={v.get('HOLD',0)}"

def compute_sl_tp(entry: float, atr: float, side: str) -> Tuple[float, float, float, float]:
    if side == "BUY":
        sl = entry - STOP_ATR_MULT * atr
        tp1 = entry + TP1_R_MULT * (entry - sl)
        tp2 = entry + TP2_R_MULT * (entry - sl)
        tp3 = entry + TP3_R_MULT * (entry - sl)
    else:
        sl = entry + STOP_ATR_MULT * atr
        tp1 = entry - TP1_R_MULT * (sl - entry)
        tp2 = entry - TP2_R_MULT * (sl - entry)
        tp3 = entry - TP3_R_MULT * (sl - entry)
    return sl, tp1, tp2, tp3

def manage_trailing(symbol: str, side: str, entry: float, atr: float, qty_left: float):
    price = get_price(symbol)
    sl = open_positions[symbol]["sl"]
    # عدّل قوة التريلينغ إذا تجاوز TP2
    trail_mult = open_positions[symbol].get("trail_mult", TRAIL_ATR_MULT)
    if side == "BUY":
        new_sl = max(sl, price - trail_mult * atr)
        if new_sl > sl:
            open_positions[symbol]["sl"] = new_sl
            place_conditional(symbol, "SELL", "STOP_MARKET", new_sl, qty_left)
            send_tg(f"🔁 Trailing SL {symbol} -> {new_sl:.4f} | rem={qty_left}")
    else:
        new_sl = min(sl, price + trail_mult * atr)
        if new_sl < sl:
            open_positions[symbol]["sl"] = new_sl
            place_conditional(symbol, "BUY", "STOP_MARKET", new_sl, qty_left)
            send_tg(f"🔁 Trailing SL {symbol} -> {new_sl:.4f} | rem={qty_left}")

def maybe_notify_tp_sl(symbol: str, price: float):
    st = open_positions.get(symbol)
    if not st: return
    side, entry, sl, atr = st["side"], st["entry"], st["sl"], st["atr"]
    tps = st["tps"]
    qty0 = st["qty"]
    remaining = st["qty_left"]

    def on_hit(label: str, trigger: float, pct_close: float):
        nonlocal remaining
        if remaining <= 0: return
        part = round_step(qty0 * pct_close, symbol_filters(symbol)["step"]) if label != "SL" else remaining
        remaining = max(0.0, remaining - part)
        open_positions[symbol]["qty_left"] = remaining
        pnl = (trigger - entry) * part if side == "BUY" else (entry - trigger) * part
        r_mult = abs(trigger - entry) / max(1e-9, abs(entry - sl))
        if label.startswith("TP"): METRICS["tp_hits"] += 1
        if label == "SL":
            METRICS["sl_hits"] += 1
            _symbol_cooldown_until[symbol] = time.time() + SYMBOL_COOLDOWN_MIN*60
        METRICS["realized_pnl"] += pnl
        send_tg(
            f"🎯 {label} {symbol} @ {trigger:.4f} | part={part} | rem={remaining} | ~PnL={pnl:.4f} | ~{r_mult:.2f}R"
        )
        # Breakeven بعد TP1
        if BREAKEVEN_AFTER_TP1 and label == "TP1":
            be = entry + (BE_OFFSET_MULT * atr if side=="BUY" else -BE_OFFSET_MULT * atr)
            open_positions[symbol]["sl"] = be
            place_conditional(symbol, ("SELL" if side=="BUY" else "BUY"), "STOP_MARKET", be, remaining)
            send_tg(f"🛡️ BE {symbol}: SL -> {be:.4f} بعد TP1")
        # Trailing أقوى بعد TP2
        if label == "TP2":
            open_positions[symbol]["trail_mult"] = POST_TP2_TRAIL_MULT
            send_tg(f"⚙️ Trail Mult {symbol} -> {POST_TP2_TRAIL_MULT}")
        if remaining <= 0:
            send_tg(f"📘 EXIT {symbol} | entry={entry:.4f} | SL={sl:.4f} | realized≈{METRICS['realized_pnl']:.4f}")
            cancel_all(symbol)
            open_positions.pop(symbol, None)
            # غلق إعادة الدخول لبضع شموع
            mins = _interval_minutes(INTERVAL) * REENTRY_LOCK_BARS
            _reentry_lock_until[symbol] = time.time() + mins*60

    if side == "BUY":
        if price <= sl: on_hit("SL", sl, 1.0)
        else:
            if (not tps[0][2]) and price >= tps[0][0]: tps[0][2] = True; on_hit("TP1", tps[0][0], tps[0][1])
            if (not tps[1][2]) and price >= tps[1][0]: tps[1][2] = True; on_hit("TP2", tps[1][0], tps[1][1])
            if (not tps[2][2]) and price >= tps[2][0]: tps[2][2] = True; on_hit("TP3", tps[2][0], tps[2][1])
    else:
        if price >= sl: on_hit("SL", sl, 1.0)
        else:
            if (not tps[0][2]) and price <= tps[0][0]: tps[0][2] = True; on_hit("TP1", tps[0][0], tps[0][1])
            if (not tps[1][2]) and price <= tps[1][0]: tps[1][2] = True; on_hit("TP2", tps[1][0], tps[1][1])
            if (not tps[2][2]) and price <= tps[2][0]: tps[2][2] = True; on_hit("TP3", tps[2][0], tps[2][1])

# ===================== إجماع الإشارة =====================

def consensus_signal(df: pd.DataFrame) -> Tuple[str, Dict[str,int]]:
    if df is None or df.empty:
        return "HOLD", {"BUY":0, "SELL":0, "HOLD":1}
    ind = indicators(df)
    c = ind["close"].iloc[-1]
    prev_hist = ind["hist"].iloc[-2]; curr_hist = ind["hist"].iloc[-1]

    votes = []
    bullish = c > ind["ema50"].iloc[-1]
    if curr_hist > 0 and prev_hist <= 0 and bullish: votes.append("BUY")
    elif curr_hist < 0 and prev_hist >= 0 and (not bullish): votes.append("SELL")
    else: votes.append("HOLD")

    r = float(ind["rsi"].iloc[-1])
    if r < RSI_SELL_MIN: votes.append("BUY")
    elif r > RSI_BUY_MAX: votes.append("SELL")
    else: votes.append("HOLD")

    if float(ind["adx"].iloc[-1]) >= ADX_MIN: votes.append("TREND_OK")
    else: votes.append("TREND_WEAK")

    if c <= float(ind["bb_dn"].iloc[-1]): votes.append("BUY")
    elif c >= float(ind["bb_up"].iloc[-1]): votes.append("SELL")
    else: votes.append("HOLD")

    if c > float(ind["vwap"].iloc[-1]) and bullish: votes.append("BUY")
    elif c < float(ind["vwap"].iloc[-1]) and (not bullish): votes.append("SELL")
    else: votes.append("HOLD")

    atr_pct = float(ind["atr"].iloc[-1]) / (c + 1e-9) * 100.0
    if atr_pct < MIN_ATR_PCT:
        return "HOLD", {"BUY":0, "SELL":0, "HOLD":1}

    buy_votes  = sum(1 for v in votes if v == "BUY")
    sell_votes = sum(1 for v in votes if v == "SELL")
    total_votes= buy_votes + sell_votes
    if total_votes == 0:
        return "HOLD", {"BUY":0, "SELL":0, "HOLD":1}

    if buy_votes / 5.0 >= CONSENSUS_MIN and "TREND_OK" in votes:
        METRICS["signals"] += 1
        return "BUY", {"BUY":buy_votes, "SELL":sell_votes, "HOLD":5 - total_votes}
    if sell_votes / 5.0 >= CONSENSUS_MIN and "TREND_OK" in votes:
        METRICS["signals"] += 1
        return "SELL", {"BUY":buy_votes, "SELL":sell_votes, "HOLD":5 - total_votes}
    return "HOLD", {"BUY":buy_votes, "SELL":sell_votes, "HOLD":5 - total_votes}

# ======= مسار الدخول مع كل الفلاتر =======

def try_enter(symbol: str):
    global _session_day, _session_start_balance
    # توقف يومي
    # إعادة ضبط عدادات اليوم عند تبدّل التاريخ
    if _session_day != date.today():
        _session_day = date.today()
        METRICS.update({"signals":0, "trades":0, "tp_hits":0, "sl_hits":0, "realized_pnl":0.0})
        _session_start_balance = None
    if DAILY_LOSS_STOP_USDT > -1e6 and METRICS['realized_pnl'] <= DAILY_LOSS_STOP_USDT:
        return
    if _session_start_balance is None:
        _session_start_balance = get_balance_usdt()
    if DAILY_LOSS_STOP_PCT > -100:
        if _session_start_balance > 0 and (METRICS['realized_pnl']/_session_start_balance*100) <= DAILY_LOSS_STOP_PCT:
            return

    # تبريد أو منع إعادة الدخول
    nowt = time.time()
    if _symbol_cooldown_until.get(symbol, 0) > nowt: return
    if _reentry_lock_until.get(symbol, 0) > nowt: return

    # حد الصفقات المتزامنة
    if len(open_positions) >= MAX_CONCURRENT_TRADES:
        return

    # فلتر تمويل
    if funding_abs(symbol) > FUNDING_ABS_MAX:
        return

    df = get_klines(symbol, INTERVAL, 300)
    if df is None or df.empty: return

    sig, votes = consensus_signal(df)
    if sig == "HOLD": return

    # فلتر الفريم الأعلى
    df_htf = get_klines(symbol, HTF_INTERVAL, 300)
    trend = htf_trend(df_htf)
    if (sig == "BUY" and trend != "BULLISH") or (sig == "SELL" and trend != "BEARISH"):
        return

    ind = indicators(df)
    price = float(ind["close"].iloc[-1])
    atr   = float(ind["atr"].iloc[-1])
    side  = "BUY" if sig == "BUY" else "SELL"

    sl, tp1, tp2, tp3 = compute_sl_tp(price, atr, side)
    qty = calc_position_size(symbol, price, sl)
    if qty <= 0: return

    ensure_margin_type(symbol, DEFAULT_MARGIN_TYPE)
    ensure_leverage(symbol, LEVERAGE)

    res = place_order(symbol, side, qty, "MARKET")
    METRICS["trades"] += 1
    cancel_all(symbol)

    if side == "BUY":
        place_conditional(symbol, "SELL", "STOP_MARKET", sl, qty)
        tp1q = round_step(qty*TP1_PCT_CLOSE, symbol_filters(symbol)["step"]) ; place_conditional(symbol, "SELL", "TAKE_PROFIT_MARKET", tp1, tp1q)
        tp2q = round_step(qty*TP2_PCT_CLOSE, symbol_filters(symbol)["step"]) ; place_conditional(symbol, "SELL", "TAKE_PROFIT_MARKET", tp2, tp2q)
        tp3q = round_step(qty*TP3_PCT_CLOSE, symbol_filters(symbol)["step"]) ; place_conditional(symbol, "SELL", "TAKE_PROFIT_MARKET", tp3, tp3q)
    else:
        place_conditional(symbol, "BUY", "STOP_MARKET", sl, qty)
        tp1q = round_step(qty*TP1_PCT_CLOSE, symbol_filters(symbol)["step"]) ; place_conditional(symbol, "BUY", "TAKE_PROFIT_MARKET", tp1, tp1q)
        tp2q = round_step(qty*TP2_PCT_CLOSE, symbol_filters(symbol)["step"]) ; place_conditional(symbol, "BUY", "TAKE_PROFIT_MARKET", tp2, tp2q)
        tp3q = round_step(qty*TP3_PCT_CLOSE, symbol_filters(symbol)["step"]) ; place_conditional(symbol, "BUY", "TAKE_PROFIT_MARKET", tp3, tp3q)

    open_positions[symbol] = {
        "side": side, "entry": price, "sl": sl, "atr": atr, "qty": qty,
        "qty_left": qty, "tps": [[tp1, TP1_PCT_CLOSE, False], [tp2, TP2_PCT_CLOSE, False], [tp3, TP3_PCT_CLOSE, False]],
        "realized": 0.0, "trail_mult": TRAIL_ATR_MULT
    }

    r_unit = abs(price - sl)
    send_tg(
        f"✅ ENTRY {symbol} {side} qty={qty} @~{price:.4f}\n"
        f"SL {sl:.4f} | R={r_unit:.4f} | TP1 {tp1:.4f}({TP1_PCT_CLOSE*100:.0f}%) TP2 {tp2:.4f}({TP2_PCT_CLOSE*100:.0f}%) TP3 {tp3:.4f}({TP3_PCT_CLOSE*100:.0f}%)\n"
        f"profile={RISK_PROFILE} | HTF={trend} | lev={LEVERAGE}x | margin={DEFAULT_MARGIN_TYPE}"
    )

# ============== ملخص كل ساعة (جدول نصّي بسيط) ==============

def hourly_summary():
    lines = [
        "🕐 Hourly Summary (Session)",
        "┌──────────────┬────────┐",
        f"│ Signals      │ {METRICS['signals']:>6} │",
        f"│ Trades       │ {METRICS['trades']:>6} │",
        f"│ TP hits      │ {METRICS['tp_hits']:>6} │",
        f"│ SL hits      │ {METRICS['sl_hits']:>6} │",
        f"│ Realized PnL │ {METRICS['realized_pnl']:>6.4f} │",
        "└──────────────┴────────┘",
        f"mode={RUN_MODE} | lev={LEVERAGE}x | margin={DEFAULT_MARGIN_TYPE} | profile={RISK_PROFILE} | TF={INTERVAL}->{HTF_INTERVAL}"
    ]
    send_tg("\n".join(lines))

# ===================== التحميل والحلقة =====================

def load_universe(top_n: Optional[int] = None) -> List[str]:
    top_n = int(top_n or MAX_SYMBOLS)
    final: List[str] = []

    if SYMBOLS_CSV:
        try:
            try:
                df = pd.read_csv(SYMBOLS_CSV)
            except Exception:
                df = pd.read_csv(SYMBOLS_CSV, header=None, names=["symbol"])
            syms = [_clean_symbol(s) for s in df["symbol"] if str(s).strip()]
            valid = set(fetch_valid_perp_usdt())
            for s in syms:
                if s not in valid: continue
                if verify_symbol(s): final.append(s)
                if len(final) >= top_n: break
        except Exception:
            final = []

    if not final:
        final = build_auto_universe(top_n)

    checked = []
    for s in final:
        if verify_symbol(s): checked.append(s)
    return checked[:top_n]


def main_loop():
    global _last_summary_ts, _session_start_balance, _session_day
    send_tg("🚀 MahdiBot v5 PRO — ALL Add-ons Enabled")
    symbols = load_universe(MAX_SYMBOLS)
    if not symbols:
        send_tg("❌ لم يتم العثور على أزواج صالحة — توقف")
        print("No valid symbols.")
        return
    print("Universe:", symbols)
    send_tg(
        f"📊 Universe (n={len(symbols)}): {', '.join(symbols[:12])}{'…' if len(symbols)>12 else ''}\n"
        f"mode={RUN_MODE} | lev={LEVERAGE}x | margin={DEFAULT_MARGIN_TYPE} | profile={RISK_PROFILE} | TF={INTERVAL}->{HTF_INTERVAL}"
    )
    send_universe_details(symbols)

    _last_summary_ts = time.time()
    _session_day = date.today()
    _session_start_balance = get_balance_usdt()

    SLEEP_SEC = int(os.getenv("SLEEP_SEC", "30"))
    while True:
        try:
            for sym in symbols:
                sym = _clean_symbol(sym)
                try:
                    if sym not in open_positions:
                        try_enter(sym)
                    else:
                        pos = open_positions[sym]
                        price = get_price(sym)
                        manage_trailing(sym, pos["side"], pos["entry"], pos["atr"], pos["qty_left"])
                        maybe_notify_tp_sl(sym, price)
                except HttpErr as he:
                    if "-1121" in str(he):
                        send_tg(f"⚠️ {sym}: Invalid symbol — شُطب.")
                    else:
                        send_tg(f"⚠️ {sym}: Error {he}")
                time.sleep(0.2)

            now_ts = time.time()
            if now_ts - _last_summary_ts >= SUMMARY_EVERY_SEC:
                hourly_summary()
                _last_summary_ts = now_ts

        except KeyboardInterrupt:
            break
        except Exception as e:
            send_tg(f"⚠️ Loop error: {e}")
        time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    print("MahdiBot v5 PRO — ALL Add-ons Enabled")
    main_loop()
