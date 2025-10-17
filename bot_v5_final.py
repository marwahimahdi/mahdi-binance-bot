
# -*- coding: utf-8 -*-
'''
Mahdi v5 PRO — full live script (Binance USDT‑M Futures)

What this file does
-------------------
- Loads environment variables (see section "ENV Vars" below).
- Reads symbols list from a CSV file (SYMBOLS_CSV), one symbol per line, e.g. BTCUSDT.
- Caches exchange filters (tickSize / stepSize) and provides helpers to normalize price/qty.
- Ensures leverage is set once per symbol.
- Opens MARKET positions and immediately places exit brackets:
  * TAKE_PROFIT_MARKET with closePosition=true, reduceOnly=true
  * STOP_MARKET with closePosition=true, reduceOnly=true
  (Both use workingType="CONTRACT_PRICE".)
- Minimal Telegram notifications (optional).

✅ Works with python-binance 1.0.19

ENV Vars (set on Render -> Environment)
---------------------------------------
API_KEY=your_key
API_SECRET=your_secret

RUN_MODE=live              # live (default)
USE_TESTNET=false          # true to use testnet endpoints

SYMBOLS_CSV=universe.csv   # one symbol per line
TOTAL_CAPITAL_PCT=0.40
MAX_OPEN_TRADES=6

SLOT_A_PCT=0.06            # portion of total capital to use per trade
SLOT_A_LEV=10              # leverage to set and use

TP1_PCT=0.0035             # +0.35% target
SL_PCT=0.0075              # -0.75% stop

TG_ENABLED=true
TELEGRAM_TOKEN=xxxxx
TELEGRAM_CHAT_ID=xxxxxx

Start on Render
---------------
Start command:
    python -u bot_v5_final.py
'''
import math
import os
import sys
import time
import json
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, List, Optional

import requests
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, FUTURE_ORDER_TYPE_MARKET

# -------------------- Utilities --------------------

def getenv_bool(key: str, default: bool=False) -> bool:
    v = os.getenv(key, str(default)).strip().lower()
    return v in ("1", "true", "yes", "on")

def getenv_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default

def tg_enabled() -> bool:
    return getenv_bool("TG_ENABLED", False) and os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")

def tg(msg: str):
    """Send Telegram message if enabled. Safe no-op otherwise."""
    if not tg_enabled():
        return
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat_id, "text": msg})
    except Exception as e:
        print(f"[TG] send failed: {e}", file=sys.stderr)

def log(msg: str):
    print(msg, flush=True)
    if "ERROR" in msg or "HTTP" in msg:
        tg(msg)

# -------------------- Client --------------------

def make_client() -> Client:
    api_key = os.getenv("API_KEY", "").strip()
    api_secret = os.getenv("API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError("Missing API_KEY / API_SECRET")

    testnet = getenv_bool("USE_TESTNET", False)
    client = Client(api_key, api_secret, testnet=testnet)
    # Ensure futures endpoints
    if testnet:
        client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
        client.FUTURES_DATA_URL = "https://testnet.binancefuture.com/fapi"
    return client

# -------------------- Symbols & Filters --------------------

_filters_cache: Dict[str, Dict[str, Any]] = {}

def load_symbols_from_csv(path: str) -> List[str]:
    syms: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().upper()
                if not s:
                    continue
                if not s.endswith("USDT"):
                    s = s + "USDT"
                syms.append(s)
    except Exception as e:
        raise RuntimeError(f"Failed reading {path}: {e}")
    return syms

def fetch_symbol_filters(client: Client, symbol: str) -> Dict[str, Any]:
    if symbol in _filters_cache:
        return _filters_cache[symbol]
    ex = client.futures_exchange_info()
    for s in ex["symbols"]:
        if s["symbol"] == symbol:
            fs = {f["filterType"]: f for f in s["filters"]}
            _filters_cache[symbol] = fs
            return fs
    raise RuntimeError(f"Symbol {symbol} not found in exchange info.")

def _step_from_str(x: str) -> Decimal:
    d = Decimal(x)
    return d

def _quantize(value: Decimal, step: Decimal) -> Decimal:
    # Round DOWN to step
    q = (value / step).to_integral_value(rounding=ROUND_DOWN) * step
    # normalize trailing zeros
    return q.normalize()

def normalize_price(client: Client, symbol: str, price: float) -> str:
    fs = fetch_symbol_filters(client, symbol)
    tick = _step_from_str(fs["PRICE_FILTER"]["tickSize"])
    p = _quantize(Decimal(str(price)), tick)
    return format(p, "f")

def normalize_qty(client: Client, symbol: str, qty: float) -> str:
    fs = fetch_symbol_filters(client, symbol)
    step = _step_from_str(fs["LOT_SIZE"]["stepSize"])
    q = _quantize(Decimal(str(qty)), step)
    return format(q, "f")

# -------------------- Position & Leverage --------------------

def ensure_leverage(client: Client, symbol: str, lev: int):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=lev)
    except Exception as e:
        log(f"⚠️ leverage: {e}")

def get_position_info(client: Client, symbol: str) -> Dict[str, Any]:
    pos = client.futures_position_information(symbol=symbol)
    # API returns list with a single dict for the symbol
    if isinstance(pos, list) and pos:
        return pos[0]
    return {}

# -------------------- Orders --------------------

def place_market_entry(client: Client, symbol: str, side: str, qty: str) -> Dict[str, Any]:
    """Open market order with given qty (string)."""
    params = dict(symbol=symbol, side=side, type=FUTURE_ORDER_TYPE_MARKET, quantity=qty)
    log(f"▶️ ENTRY {symbol} {side} qty={qty}")
    return client.futures_create_order(**params)

def place_exit_brackets(client: Client, symbol: str, side_open: str,
                        tp_price_raw: float, sl_price_raw: float):
    """
    Create TP/SL exits that close the ENTIRE position using closePosition=true.
    We must NOT send quantity with closePosition=true.
    """
    opp_side = SIDE_SELL if side_open == SIDE_BUY else SIDE_BUY
    tp_price = normalize_price(client, symbol, tp_price_raw)
    sl_price = normalize_price(client, symbol, sl_price_raw)

    # TAKE_PROFIT_MARKET
    tp_params = dict(
        symbol=symbol,
        side=opp_side,
        type="TAKE_PROFIT_MARKET",
        closePosition=True,
        reduceOnly=True,
        stopPrice=tp_price,
        workingType="CONTRACT_PRICE",
        priceProtect=True,
    )
    # STOP_MARKET
    sl_params = dict(
        symbol=symbol,
        side=opp_side,
        type="STOP_MARKET",
        closePosition=True,
        reduceOnly=True,
        stopPrice=sl_price,
        workingType="CONTRACT_PRICE",
        priceProtect=True,
    )
    log(f"🧩 EXITS {symbol} -> TP@{tp_price} / SL@{sl_price}")
    r1 = client.futures_create_order(**tp_params)
    r2 = client.futures_create_order(**sl_params)
    return r1, r2

# -------------------- Sizing --------------------

def get_last_price(client: Client, symbol: str) -> float:
    p = client.futures_symbol_ticker(symbol=symbol)
    return float(p["price"])

def get_wallet_balance(client: Client) -> float:
    bal = client.futures_account_balance()
    for b in bal:
        if b["asset"] == "USDT":
            return float(b["balance"])
    return 0.0

def compute_entry_qty(client: Client, symbol: str, capital_usdt: float, price: float, leverage: int) -> str:
    notional = capital_usdt * leverage
    qty = notional / price
    return normalize_qty(client, symbol, qty)

# -------------------- Demo flow (single shot) --------------------

def demo_once():
    """
    Optional single-shot demo: enter a LONG on first symbol, place TP/SL, then exit.
    Enable by calling demo_once() in __main__.
    """
    client = make_client()
    csv_path = os.getenv("SYMBOLS_CSV", "universe.csv")
    symbols = load_symbols_from_csv(csv_path)
    if not symbols:
        raise RuntimeError("SYMBOLS_CSV is empty.")

    symbol = symbols[0]

    total_cap_pct = getenv_float("TOTAL_CAPITAL_PCT", 0.40)
    slot_pct = getenv_float("SLOT_A_PCT", 0.06)
    lev = int(getenv_float("SLOT_A_LEV", 10))
    tp_pct = getenv_float("TP1_PCT", 0.0035)
    sl_pct = getenv_float("SL_PCT", 0.0075)

    # balances & sizing
    balance = get_wallet_balance(client)
    capital = balance * total_cap_pct * slot_pct
    price = get_last_price(client, symbol)

    ensure_leverage(client, symbol, lev)
    qty = compute_entry_qty(client, symbol, capital, price, lev)

    # open LONG for demo
    place_market_entry(client, symbol, SIDE_BUY, qty)

    # compute TP/SL prices on entry price
    tp_price = price * (1.0 + tp_pct)
    sl_price = price * (1.0 - sl_pct)

    # place exits (closePosition=true)
    place_exit_brackets(client, symbol, SIDE_BUY, tp_price, sl_price)

    tg(f"✅ Demo entered {symbol} LONG qty={qty} @~{price:.4f}\nTP~{tp_price:.4f} / SL~{sl_price:.4f}")

# -------------------- Main (heartbeat only) --------------------

def main():
    """
    Minimal runner: at start, just prints loaded symbols and waits.
    Your own scanning/strategy logic can call place_market_entry() then place_exit_brackets().
    """
    client = make_client()
    csv_path = os.getenv("SYMBOLS_CSV", "universe.csv")
    symbols = load_symbols_from_csv(csv_path)
    tg(f"♻️ Mahdi v5 PRO — تشغيل: {os.getenv('RUN_MODE','live').upper()} | Testnet: {'On' if getenv_bool('USE_TESTNET') else 'Off'}\nSymbols: {symbols}")

    # Heartbeat loop
    while True:
        try:
            ts = int(time.time()*1000)
            log(f"[HB] alive {ts} symbols={len(symbols)}")
            time.sleep(60)
        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"⚠️ loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    # To run a single demo trade once, uncomment the next line:
    # demo_once(); sys.exit(0)
    main()
