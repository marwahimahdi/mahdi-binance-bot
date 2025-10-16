# MahdiBot v5 — Render Worker

## What’s new
- Consensus with EMA, MACD, RSI, Supertrend, VWAP
- Dynamic leverage (5x/10x) + dynamic position sizing (5%/6%)
- Three targets (TP1/TP2/TP3) + SL->BE after TP1 + lock after TP2
- Daily Kill-Switch (5% default) + Watchdog inactivity alerts with reminder
- Universe from `universe.csv` (25 symbols) or auto top-volume fallback
- Telegram rich alerts + heartbeat

## Files
- `bot_v5_final.py` — main worker
- `requirements.txt` — deps (same as v4)
- `runtime.txt` — Python 3.11.9
- `universe.csv` — 25 symbols
- `.env` — fill secrets and settings

## Quick deploy (Render)
1) Push files to a new GitHub repo.
2) On Render: **New+ → Blueprint** (if using render.yaml) or **New+ → Worker** and set:
   - Start Command: `python bot_v5_final.py`
   - Environment: add variables from `.env` (API_KEY/SECRET/TELEGRAM_* ...).
3) Click **Deploy**. Check Logs — expect a startup message + universe list.
4) To update: push a new commit — Render restarts the worker.

## Safe defaults
- RUN_MODE=live with USE_TESTNET=false for production. Use `paper`/`true` for safe testing.
- API keys: Futures only; withdrawals disabled; IP restricted.
- MAX_OPEN_TRADES=6 keeps capital under 40% total with 5–6% per trade.

## Notes
- If you edit targets/filters, prefer `.env` first; no code change needed.
- For performance logs or weekly reports, extend using CSV logging where commented.