# Deploy Mahdi Bot on Render (Worker)

## Files in repo
- `main_v4_1r_tg_live_20pct4.py` — main bot
- `requirements.txt` — Python deps
- `runtime.txt` — Python version
- `render.yaml` — Render IaC (worker)

## Quick steps
1) Push these files + your `main_v4_1r_tg_live_20pct4.py` to a **new GitHub repo**.
2) On Render: **New +** → **Blueprint** → connect the repo (it reads `render.yaml`).
3) In Render dashboard → Service **mahdi-bot-worker** → **Environment**:
   - Set `API_KEY` and `API_SECRET` (Add Secret).
   - Set `TELEGRAM_TOKEN` (Add Secret).
   - Check remaining vars (already have safe defaults).
4) Click **Deploy**. The worker starts and runs your bot continuously.
5) Logs: Render → the service → **Logs**. You should see heartbeat and any trades.
6) To update: push to GitHub → auto deploy will restart the worker.

> Notes
- No web port is required (type: worker).
- If you hit API rate limits, raise `SCAN_INTERVAL_SEC` to `60` and lower `MAX_SYMBOLS` to `12`.
- To test safely, set `USE_TESTNET=true` and `RUN_MODE=paper`.
