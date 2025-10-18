#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mahdi v5 PRO — FINAL (LONG & SHORT) with 3 TPs (40/35/25) + Dynamic SL updates

- Binance USDⓈ-M Futures (REST).
- Entry: MARKET (BUY for LONG, SELL for SHORT).
- Exits:
  * 3 × TAKE_PROFIT_MARKET (reduceOnly, with exact quantities via stepSize).
  * 1 × STOP_MARKET (reduceOnly + closePosition=true) يغلق الباقي دائمًا.
  * Dynamic SL manager:
      - بعد تحقق TP1: ينقل SL إلى نقطة الدخول Breakeven.
      - بعد تحقق TP2: ينقل SL إلى مستوى TP1 (قفل ربح).
- 5 indicators voting: RSI, MACD(hist slope), ADX, EMA200 trend, Stochastic.
- Filters: EMA200 direction, ADX>=ADX_MIN, cooldown, max open trades, universe.csv.
- Position sizing: from real USDT available balance:
    qty = (available * CAPITAL_USE_PCT * SLOT_A_PCT * SLOT_A_LEV) / price   → rounded to stepSize.

IMPORTANT:
- One-way mode (لا تفعّل hedge). وإلا يجب إضافة positionSide.
- لا timeInForce مع MARKET/TP/SL.
"""

import os, time, hmac, hashlib, json, math, csv
import urllib.parse as urlparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import numpy as np
from requests.exceptions import RequestException

# ---------------- Env ----------------
API_KEY     = os.getenv('API_KEY', '')
API_SECRET  = os.getenv('API_SECRET', '')
USE_TESTNET = os.getenv('USE_TESTNET', 'false').lower() == 'true'
BASE_URL    = 'https://testnet.binancefuture.com' if USE_TESTNET else 'https://fapi.binance.com'

RUN_MODE    = os.getenv('RUN_MODE', 'paper')   # live | paper
SYMBOLS_CSV = os.getenv('SYMBOLS_CSV', 'universe.csv')

INTERVAL         = os.getenv('INTERVAL', '5m')
INTERVAL_CONFIRM = os.getenv('INTERVAL_CONFIRM', '15m')   # محفوظ إذا أحببت تضيف تأكيد ثانٍ لاحقًا
KLINES_LIMIT     = int(os.getenv('KLINES_LIMIT', '300'))
SCAN_INTERVAL    = int(os.getenv('SCAN_INTERVAL_SEC', '120'))

MAX_SYMBOLS      = int(os.getenv('MAX_SYMBOLS', '16'))
MAX_OPEN_TRADES  = int(os.getenv('MAX_OPEN_TRADES', os.getenv('MAX_OPEN_POS', '6')))

CAPITAL_USE_PCT  = float(os.getenv('CAPITAL_USE_PCT', '0.40'))
SLOT_A_PCT       = float(os.getenv('SLOT_A_PCT', '0.06'))
SLOT_A_LEV       = int(float(os.getenv('SLOT_A_LEV', '10')))
SLOT_B_PCT       = float(os.getenv('SLOT_B_PCT', '0.05'))
SLOT_B_LEV       = int(float(os.getenv('SLOT_B_LEV', '5')))  # غير مستخدم الآن

TP1_PCT          = float(os.getenv('TP1_PCT', '0.0035'))
TP2_PCT          = float(os.getenv('TP2_PCT', '0.0070'))
TP3_PCT          = float(os.getenv('TP3_PCT', '0.0120'))
SL_PCT           = float(os.getenv('SL_PCT',  '0.0075'))

CONSENSUS_MIN    = float(os.getenv('CONSENSUS_MIN', '0.60'))
ADX_MIN          = float(os.getenv('ADX_MIN', '20'))
COOLDOWN_MIN     = int(os.getenv('COOLDOWN_MIN', '60'))

TG_ENABLED       = os.getenv('TG_ENABLED', 'false').lower() == 'true'
TG_TOKEN         = os.getenv('TELEGRAM_TOKEN', '')
TG_CHAT_ID       = os.getenv('TELEGRAM_CHAT_ID', '')
TG_HEARTBEAT_MIN = int(os.getenv('TG_HEARTBEAT_MIN', '15'))
TG_NOTIFY_UNIVERSE = os.getenv('TG_NOTIFY_UNIVERSE', 'false').lower() == 'true'
TG_SUMMARY_MIN   = int(os.getenv('TG_SUMMARY_MIN', '1440'))

# توزيع الكميات بين الأهداف (يجب أن يساوي 1.0)
TP_SPLIT = (0.40, 0.35, 0.25)

SESSION = requests.Session()
SESSION.headers.update({'X-MBX-APIKEY': API_KEY})

# ---------------- Utils ----------------
def now_ms() -> int:
    return int(time.time() * 1000)

def ts() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def sign_query(q: str) -> str:
    return hmac.new(API_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()

def _get(path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False, timeout: int = 15):
    params = params or {}
    if signed:
        params['timestamp'] = now_ms()
        q = urlparse.urlencode(params, doseq=True)
        params['signature'] = sign_query(q)
    try:
        r = SESSION.get(BASE_URL + path, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except RequestException as e:
        return {'error': str(e), 'text': getattr(e.response, 'text', None)}

def _post(path: str, data: Optional[Dict[str, Any]] = None, signed: bool = True, timeout: int = 15):
    data = data or {}
    if signed:
        data['timestamp'] = now_ms()
        q = urlparse.urlencode(data, doseq=True)
        data['signature'] = sign_query(q)
    try:
        r = SESSION.post(BASE_URL + path, data=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except RequestException as e:
        return {'error': str(e), 'text': getattr(e.response, 'text', None)}

def _delete(path: str, params: Optional[Dict[str, Any]] = None, signed: bool = True, timeout: int = 15):
    params = params or {}
    if signed:
        params['timestamp'] = now_ms()
        q = urlparse.urlencode(params, doseq=True)
        params['signature'] = sign_query(q)
    try:
        r = SESSION.delete(BASE_URL + path, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except RequestException as e:
        return {'error': str(e), 'text': getattr(e.response, 'text', None)}

def tg(text: str):
    if not TG_ENABLED or not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      data={'chat_id': TG_CHAT_ID, 'text': text, 'parse_mode': 'HTML'}, timeout=10)
    except Exception:
        pass

# ---------- Universe ----------
def load_universe(path: str) -> List[str]:
    out = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip().upper()
                if not s or s.startswith('#'): continue
                if not s.endswith('USDT'): s += 'USDT'
                out.append(s)
    except Exception:
        pass
    return out[:MAX_SYMBOLS] if MAX_SYMBOLS > 0 else out

# ---------- Klines & DF ----------
def klines(symbol: str, interval: str, limit: int):
    return _get('/fapi/v1/klines', {'symbol': symbol, 'interval': interval, 'limit': limit})

def df_from_klines(raw: list) -> pd.DataFrame:
    cols = ['open_time','open','high','low','close','volume',
            'close_time','qav','trades','taker_bav','taker_qav','ignore']
    df = pd.DataFrame(raw, columns=cols)
    for c in ('open','high','low','close','volume'):
        df[c] = df[c].astype(float)
    return df

# ---------- Indicators ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    ma_up = up.ewm(com=n-1, adjust=False).mean()
    ma_dn = dn.ewm(com=n-1, adjust=False).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return 100 - (100/(1+rs))

def macd(s: pd.Series, fast=12, slow=26, signal=9):
    ef, es = ema(s, fast), ema(s, slow)
    m = ef - es
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h,l,c = df['high'], df['low'], df['close']
    plus_dm  = (h.diff()).clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1/n, adjust=False).mean()

def stochastic(df: pd.DataFrame, k: int=14, d: int=3):
    low_min  = df['low'].rolling(k).min()
    high_max = df['high'].rolling(k).max()
    kperc = 100*(df['close']-low_min)/(high_max-low_min)
    dperc = kperc.rolling(d).mean()
    return kperc, dperc

# ---------- Exchange Info (precision) ----------
_symbol_info_cache: Dict[str, Dict[str, float]] = {}

def load_symbol_info(symbol: str) -> Dict[str, float]:
    """returns dict: {'stepSize':float, 'tickSize':float, 'minQty':float}"""
    if symbol in _symbol_info_cache:
        return _symbol_info_cache[symbol]
    info = _get('/fapi/v1/exchangeInfo')
    step = 0.001
    tick = 0.01
    minq = 0.001
    if isinstance(info, dict) and 'symbols' in info:
        for s in info['symbols']:
            if s.get('symbol') == symbol:
                for f in s.get('filters', []):
                    if f.get('filterType') == 'LOT_SIZE':
                        step = float(f.get('stepSize', step))
                        minq = float(f.get('minQty', minq))
                    if f.get('filterType') == 'PRICE_FILTER':
                        tick = float(f.get('tickSize', tick))
                break
    _symbol_info_cache[symbol] = {'stepSize': step, 'tickSize': tick, 'minQty': minq}
    return _symbol_info_cache[symbol]

def round_step(q: float, step: float) -> float:
    if step <= 0: return q
    return math.floor(q/step)*step

def round_tick(p: float, tick: float) -> float:
    if tick <= 0: return p
    return math.floor(p/tick)*tick

# ---------- Balance, Positions & Sizing ----------
def fetch_usdt_available() -> float:
    bal = _get('/fapi/v2/balance', signed=True)
    if isinstance(bal, list):
        for a in bal:
            if a.get('asset') == 'USDT':
                return float(a.get('availableBalance', a.get('balance', '0')))
    return 0.0

def get_position(symbol: str) -> Dict[str, Any]:
    """Return dict with positionAmt (abs float), entryPrice (float)."""
    pr = _get('/fapi/v2/positionRisk', {'symbol': symbol}, signed=True)
    if isinstance(pr, list) and pr:
        p = pr[0]
        amt = abs(float(p.get('positionAmt', '0')))
        ep  = float(p.get('entryPrice', '0'))
        return {'amt': amt, 'entryPrice': ep}
    return {'amt': 0.0, 'entryPrice': 0.0}

def open_orders(symbol: str) -> List[Dict[str, Any]]:
    oo = _get('/fapi/v1/openOrders', {'symbol': symbol}, signed=True)
    return oo if isinstance(oo, list) else []

def cancel_order(symbol: str, order_id: int):
    return _delete('/fapi/v1/order', {'symbol': symbol, 'orderId': order_id})

def compute_qty(symbol: str, price: float) -> float:
    info = load_symbol_info(symbol)
    step = info['stepSize']
    avail = fetch_usdt_available()
    slot_usdt = avail * CAPITAL_USE_PCT * SLOT_A_PCT
    notional = slot_usdt * SLOT_A_LEV
    raw_qty = max(0.0, notional / price)
    qty = round_step(raw_qty, step)
    # لا تقل عن minQty
    if qty < info['minQty']:
        qty = 0.0
    return float(qty)

# ---------- Orders ----------
def set_isolated(symbol: str):
    return _post('/fapi/v1/marginType', {'symbol': symbol, 'marginType': 'ISOLATED'})

def set_leverage(symbol: str, lev: int):
    return _post('/fapi/v1/leverage', {'symbol': symbol, 'leverage': lev})

def market_order(symbol: str, side: str, qty: float):
    return _post('/fapi/v1/order', {'symbol': symbol, 'side': side, 'type': 'MARKET', 'quantity': f"{qty:.10f}"})

def place_take_profit(symbol: str, side: str, stop_price: float, qty: float):
    # side هنا هو الجانب المُخفِّض (SELL للـ LONG, BUY للـ SHORT)
    return _post('/fapi/v1/order', {
        'symbol': symbol,
        'side': side,
        'type': 'TAKE_PROFIT_MARKET',
        'reduceOnly': 'true',
        'stopPrice': f"{stop_price:.10f}",
        'workingType': 'CONTRACT_PRICE',
        'quantity': f"{qty:.10f}"
    })

def place_stop(symbol: str, side: str, stop_price: float):
    # SL يغلق ما تبقى: closePosition=true
    return _post('/fapi/v1/order', {
        'symbol': symbol,
        'side': side,
        'type': 'STOP_MARKET',
        'reduceOnly': 'true',
        'closePosition': 'true',
        'stopPrice': f"{stop_price:.10f}",
        'workingType': 'CONTRACT_PRICE'
    })

def replace_stop(symbol: str, side: str, new_stop: float):
    """Cancel existing closePosition STOP then place a new one at new_stop."""
    # cancel existing stop orders
    for o in open_orders(symbol):
        if o.get('type') in ('STOP', 'STOP_MARKET') and o.get('reduceOnly') and str(o.get('closePosition', '')).lower() == 'true':
            try:
                cancel_order(symbol, int(o['orderId']))
            except Exception:
                pass
    # place new
    return place_stop(symbol, side, new_stop)

# ---------- Strategy (5 indicators + filters) ----------
def analyze_symbol(symbol: str):
    raw = klines(symbol, INTERVAL, KLINES_LIMIT)
    if not isinstance(raw, list) or len(raw) < 60:
        return None
    df = df_from_klines(raw)
    price = float(df['close'].iloc[-1])

    ema200 = ema(df['close'], 200)
    rsi14 = rsi(df['close'], 14)
    macd_line, macd_sig, macd_hist = macd(df['close'])
    adx14 = adx(df, 14)
    kperc, dperc = stochastic(df, 14, 3)
    ema20, ema50 = ema(df['close'], 20), ema(df['close'], 50)

    if adx14.iloc[-1] < ADX_MIN:
        return None

    votes_long = 0
    votes_short = 0

    # 1 RSI
    if rsi14.iloc[-1] > 55: votes_long += 1
    if rsi14.iloc[-1] < 45: votes_short += 1
    # 2 MACD histogram slope
    if macd_hist.iloc[-1] > macd_hist.iloc[-2]: votes_long += 1
    if macd_hist.iloc[-1] < macd_hist.iloc[-2]: votes_short += 1
    # 3 EMA200 direction
    above = price > ema200.iloc[-1]
    below = price < ema200.iloc[-1]
    if above: votes_long += 1
    if below: votes_short += 1
    # 4 Stochastic crossover
    if kperc.iloc[-1] > dperc.iloc[-1]: votes_long += 1
    if kperc.iloc[-1] < dperc.iloc[-1]: votes_short += 1
    # 5 EMA20 vs EMA50
    if ema20.iloc[-1] > ema50.iloc[-1]: votes_long += 1
    if ema20.iloc[-1] < ema50.iloc[-1]: votes_short += 1

    # كل الج票 5
    votes_long = min(votes_long, 5)
    votes_short = min(votes_short, 5)
    cons_long = votes_long / 5.0
    cons_short = votes_short / 5.0

    signal = None
    if cons_long >= CONSENSUS_MIN and above:
        signal = ('LONG', price, cons_long)
    elif cons_short >= CONSENSUS_MIN and below:
        signal = ('SHORT', price, cons_short)
    if not signal:
        return None

    side, entry, score = signal
    if side == 'LONG':
        tp1 = entry * (1 + TP1_PCT)
        tp2 = entry * (1 + TP2_PCT)
        tp3 = entry * (1 + TP3_PCT)
        sl  = entry * (1 - SL_PCT)
    else:
        tp1 = entry * (1 - TP1_PCT)
        tp2 = entry * (1 - TP2_PCT)
        tp3 = entry * (1 - TP3_PCT)
        sl  = entry * (1 + SL_PCT)

    return {'symbol': symbol, 'side': side, 'entry': entry, 'tp': [tp1, tp2, tp3], 'sl': sl, 'score': score, 'price': entry}

# ---------- Loop & State ----------
last_heartbeat_ts = 0.0
last_signal_ts_by_symbol: Dict[str, float] = {}

# حالة مراكزنا النشطة داخل الذاكرة
# state[symbol] = {
#   'side': 'LONG'|'SHORT',
#   'entry_price': float,
#   'entry_qty': float,
#   'tp_prices': [tp1,tp2,tp3],
#   'sl_initial': float,
#   'moved_be': bool,
#   'moved_tp1lock': bool
# }
active_state: Dict[str, Dict[str, Any]] = {}

def can_trade(symbol: str) -> bool:
    last = last_signal_ts_by_symbol.get(symbol, 0.0)
    return (time.time() - last) > (COOLDOWN_MIN * 60)

def heartbeat():
    global last_heartbeat_ts
    if not TG_ENABLED: return
    if time.time() - last_heartbeat_ts >= TG_HEARTBEAT_MIN * 60:
        syms = load_universe(SYMBOLS_CSV)
        tg(f"[HB] alive {int(time.time()*1000)} symbols={len(syms)}")
        last_heartbeat_ts = time.time()

def reconcile_positions():
    """يراقب تقدم الصفقة ويُعدل SL بعد TP1/TP2 تلقائيًا."""
    to_clear = []
    for sym, st in active_state.items():
        pos = get_position(sym)
        rem = pos['amt']
        if rem <= 0.0:
            to_clear.append(sym)
            continue
        entry_qty = st['entry_qty']
        if entry_qty <= 0: 
            continue

        # thresholds
        tp1_done_qty = entry_qty * (1.0 - TP_SPLIT[0])
        tp2_done_qty = entry_qty * (1.0 - (TP_SPLIT[0] + TP_SPLIT[1]))
        reduce_side = 'SELL' if st['side'] == 'LONG' else 'BUY'
        tick = load_symbol_info(sym)['tickSize']

        # بعد TP1 -> انقل SL إلى breakeven (سعر الدخول)
        if not st.get('moved_be', False) and rem <= tp1_done_qty + 1e-12:
            new_sl = st['entry_price']
            new_sl = round_tick(new_sl, tick)
            replace_stop(sym, reduce_side, new_sl)
            st['moved_be'] = True
            tg(f"🔒 {sym} SL → Breakeven @ {new_sl:.6f} (بعد TP1)")

        # بعد TP2 -> انقل SL إلى TP1 لقفل ربح
        if st.get('moved_be', False) and not st.get('moved_tp1lock', False) and rem <= tp2_done_qty + 1e-12:
            new_sl = st['tp_prices'][0]  # TP1 price
            new_sl = round_tick(new_sl, tick)
            replace_stop(sym, reduce_side, new_sl)
            st['moved_tp1lock'] = True
            tg(f"🧷 {sym} SL → TP1 @ {new_sl:.6f} (بعد TP2)")

    for s in to_clear:
        active_state.pop(s, None)

def scan_and_trade():
    # قبل المسح عدّل SL إن لزم
    reconcile_positions()

    syms = load_universe(SYMBOLS_CSV)
    if TG_ENABLED and TG_NOTIFY_UNIVERSE:
        tg("📊 Universe: " + ", ".join(syms[:12]) + ("..." if len(syms) > 12 else ""))

    open_trades = 0
    for s in syms:
        if open_trades >= MAX_OPEN_TRADES:
            break
        if not can_trade(s):
            continue

        sig = analyze_symbol(s)
        if not sig:
            continue

        info = load_symbol_info(s)
        step, tick = info['stepSize'], info['tickSize']

        side = sig['side']
        entry_price = float(sig['price'])
        qty_total = compute_qty(s, entry_price)
        if qty_total <= 0.0:
            tg(f"⚠️ {s}: الكمية صفر/أقل من الحد. رصيد غير كافٍ أو stepSize كبير.")
            continue

        # جهّز الأسعار بالدقة الصحيحة
        tp_prices = [round_tick(p, tick) for p in sig['tp']]
        sl_price  = round_tick(sig['sl'], tick)

        # دخول
        set_isolated(s)
        set_leverage(s, SLOT_A_LEV)
        side_entry = 'BUY' if side == 'LONG' else 'SELL'
        entry = market_order(s, side_entry, qty_total)
        if 'error' in entry:
            tg(f"⚠️ {s} دخول فشل: {entry.get('text')}")
            continue

        # كميات TPs (40/35/25) بدقة stepSize
        q1 = round_step(qty_total * TP_SPLIT[0], step)
        q2 = round_step(qty_total * TP_SPLIT[1], step)
        # الباقي يذهب لـ TP3
        q3 = qty_total - q1 - q2
        q3 = round_step(max(q3, 0.0), step)
        # تصحيح في حال ضاعت كسور
        if q1 + q2 + q3 > qty_total:
            q3 = round_step(qty_total - q1 - q2, step)

        reduce_side = 'SELL' if side == 'LONG' else 'BUY'

        # أوامر TP منفصلة بكميات محددة
        if q1 >= info['minQty']:
            place_take_profit(s, reduce_side, tp_prices[0], q1)
        if q2 >= info['minQty']:
            place_take_profit(s, reduce_side, tp_prices[1], q2)
        if q3 >= info['minQty']:
            place_take_profit(s, reduce_side, tp_prices[2], q3)

        # SL يُغلق المتبقي كله
        place_stop(s, reduce_side, sl_price)

        # خزّن الحالة للمتابعة الديناميكية للـ SL
        active_state[s] = {
            'side': side,
            'entry_price': entry_price,
            'entry_qty': qty_total,
            'tp_prices': tp_prices,
            'sl_initial': sl_price,
            'moved_be': False,
            'moved_tp1lock': False
        }

        tg(f"✅ {s} {side} @~{entry_price:.6f} | TP1~{tp_prices[0]:.6f} "
           f"TP2~{tp_prices[1]:.6f} TP3~{tp_prices[2]:.6f} SL~{sl_price:.6f} | qty={qty_total} | score={sig['score']:.2f}")

        last_signal_ts_by_symbol[s] = time.time()
        open_trades += 1

def main():
    tg(f"🚀 Mahdi v5 PRO — تشغيل: {('LIVE' if RUN_MODE=='live' else 'PAPER')} | Testnet: {'On' if USE_TESTNET else 'Off'} | BinanceTime: {int(time.time()*1000)}")
    while True:
        try:
            scan_and_trade()
            heartbeat()
        except Exception as e:
            tg(f"⚠️ loop error: {e}")
        time.sleep(SCAN_INTERVAL)

if __name__ == '__main__':
    main()
