# ========= Universe (strict USDT-M PERP) =========
def fetch_valid_perpetual_usdt():
    # Cache daily
    try:
        if CACHE_PATH.exists():
            j = json.loads(CACHE_PATH.read_text())
            if time.time() - j.get("ts", 0) < CACHE_TTL_SEC:
                return set(j.get("symbols", []))
    except Exception:
        pass

    data = _request("GET", EXCHANGE_INFO, params={}, signed_req=False)
    valid = {
        s["symbol"]
        for s in data.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
        and s.get("contractType") == "PERPETUAL"
    }

    try:
        CACHE_PATH.write_text(json.dumps({"ts": time.time(), "symbols": sorted(valid)}))
    except Exception:
        pass
    return valid


def build_auto_universe():
    """Top-N by 24h quote volume (USDT-M PERP only) + pre-validate on futures price endpoint."""
    valid = fetch_valid_perpetual_usdt()
    tickers = f_get(TICKER_24H, {"type": "FULL"})
    df = pd.DataFrame(tickers)
    # احتفظ فقط بما هو في قائمة العقود الدائمة USDT-M
    df = df[df["symbol"].isin(valid)].copy()

    if df.empty:
        send_tg("⚠️ لم أستطع بناء Universe تلقائي (قائمة فارغة بعد الفلترة). سأجرّب لاحقًا.")
        return []

    df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce").fillna(0.0)
    candidates = df.sort_values("quoteVolume", ascending=False)["symbol"].tolist()

    # ✅ تحقّق مسبق على واجهة الأسعار الخاصة بالعقود (يمنع [-1121])
    final = []
    for s in candidates:
        if len(final) >= MAX_SYMBOLS:
            break
        try:
            _ = f_get(PRICE_EP, {"symbol": s})  # إن فشل هنا فهو ليس Futures صالح
            final.append(s)
            time.sleep(0.02)  # تهدئة خفيفة
        except requests.HTTPError as he:
            if "-1121" in str(he) or "Invalid symbol" in str(he):
                continue
            else:
                continue
        except Exception:
            continue

    if TG_NOTIFY_UNIVERSE:
        if final:
            send_tg(f"🔄 Auto-Scan Mode (Top {MAX_SYMBOLS})\n📊 Universe: {', '.join(final[:10])}... (n={len(final)})")
        else:
            send_tg("⚠️ تعذر التحقق من أي رمز صالح للعقود — ربما ضغط API. سأحاول لاحقًا.")
    return final


def load_universe():
    """يختار القائمة إما من CSV (إن وُجد) أو تلقائيًا من Top-N."""
    if SYMBOLS_CSV:
        if os.path.exists(SYMBOLS_CSV):
            df = pd.read_csv(SYMBOLS_CSV)
            syms = [s.strip().upper() for s in df["symbol"] if s.upper().endswith("USDT")]
        else:
            syms = []
        valid = fetch_valid_perpetual_usdt()
        final = [s for s in syms if s in valid][:MAX_SYMBOLS]
        if TG_NOTIFY_UNIVERSE:
            send_tg(f"📊 Universe ثابت: {', '.join(final[:10])}... (n={len(final)})")
        return final

    # تلقائي
    return build_auto_universe()
