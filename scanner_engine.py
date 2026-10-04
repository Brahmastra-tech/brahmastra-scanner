import os
import time
import duckdb
import pandas as pd
import numpy as np
import requests

DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"

LOCAL_CSV_PATH = r"D:\Scanner\fo_mktlots.csv"
REPO_CSV_PATH = "fo_mktlots.csv"

ATR_PERIOD = 14
RVOL_PERIOD = 20
LOCAL_LIQ_PERIOD = 10
HTF_PERIOD = 20

MIN_PRICE = 100.0
RR_RATIO = 2.0

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"


def load_fo_universe() -> frozenset:
    file_path = LOCAL_CSV_PATH if os.path.exists(LOCAL_CSV_PATH) else REPO_CSV_PATH
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, skipinitialspace=True)
            sym_col = next((c for c in df.columns if 'SYMBOL' in c.upper()), None)
            if sym_col:
                raw_symbols = df[sym_col].dropna().astype(str).str.strip().str.upper().unique()
                cleaned = {s for s in raw_symbols if s and not any(idx in s for idx in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])}
                print(f"✅ Loaded {len(cleaned)} F&O stocks from {file_path}")
                return frozenset(cleaned)
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
    return frozenset()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    high = df['High']
    low = df['Low']
    close = df['Close']
    close_prev = close.shift(1)

    # 1. ATR 14
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()

    # 2. RVOL (Volume / 20-day SMA)
    vol_sma20 = df['Volume'].rolling(window=RVOL_PERIOD, min_periods=5).mean()
    df['RVOL'] = (df['Volume'] / (vol_sma20 + 1e-5)).round(2)

    # 3. 10-Day Local Liquidity (Shifted by 1 so current bar is not counted)
    df['prior_low'] = low.shift(1).rolling(window=LOCAL_LIQ_PERIOD, min_periods=LOCAL_LIQ_PERIOD).min()
    df['prior_high'] = high.shift(1).rolling(window=LOCAL_LIQ_PERIOD, min_periods=LOCAL_LIQ_PERIOD).max()

    # 4. 20-Day HTF Structure
    df['major_demand'] = low.shift(1).rolling(window=HTF_PERIOD, min_periods=HTF_PERIOD).min()
    df['major_supply'] = high.shift(1).rolling(window=HTF_PERIOD, min_periods=HTF_PERIOD).max()

    return df


def run_scanner():
    print("🚀 Running Daily Liquidity Sweep & Displacement Scanner...")
    if not os.path.exists(DB_PATH):
        print("❌ candles.duckdb not found.")
        return

    conn = duckdb.connect(DB_PATH)
    cols = [c[0].lower() for c in conn.execute("DESCRIBE ohlcv_candles").fetchall()]
    
    select_parts = [
        "symbol AS Symbol", "CAST(timestamp AS DATE) AS Date",
        "open AS Open", "high AS High", "low AS Low", "close AS Close", "volume AS Volume"
    ]
    if "delivery_pct" in cols:
        select_parts.append("delivery_pct AS DeliveryPct")
    else:
        select_parts.append("40.0 AS DeliveryPct")

    where_parts = [f"close > {MIN_PRICE}"]
    if "series" in cols:
        where_parts.append("UPPER(series) = 'EQ'")

    df_raw = conn.execute(f"SELECT {', '.join(select_parts)} FROM ohlcv_candles WHERE {' AND '.join(where_parts)} ORDER BY symbol, timestamp ASC").df()
    conn.close()

    if df_raw.empty:
        return

    universe = load_fo_universe()
    df_raw["Symbol_Clean"] = df_raw["Symbol"].astype(str).str.upper().str.strip().str.replace(r"-EQ$", "", regex=True)
    if universe:
        df_raw = df_raw[df_raw["Symbol_Clean"].isin(universe)].copy()

    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])
    latest_date_str = df_raw["Date_DT"].max().strftime("%d-%m-%Y")

    signals = []

    for symbol, df_sym in df_raw.groupby("Symbol_Clean"):
        if len(df_sym) < 25:
            continue

        df = compute_indicators(df_sym.copy().sort_values("Date_DT").reset_index(drop=True))
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        atr = curr['ATR']
        rvol = curr['RVOL']
        o, h, l, c = curr['Open'], curr['High'], curr['Low'], curr['Close']
        c_range = max(h - l, 1e-5)
        body = abs(c - o)

        prior_low = curr['prior_low']
        prior_high = curr['prior_high']
        major_supply = curr['major_supply']
        major_demand = curr['major_demand']

        if pd.isna(atr) or pd.isna(prior_low) or pd.isna(major_supply):
            continue

        # Volume threshold: at least 1.0x (average volume) or spike >= 1.15x
        vol_confirmed = rvol >= 1.05

        # ----------------------------------------------------
        # 1. DAILY BUY SETUP (Sweep 10-day low + Bullish Reclaim)
        # ----------------------------------------------------
        sweep_buy = (l < prior_low) and (c > prior_low)
        lower_wick = min(o, c) - l
        # Rejection: lower wick should show buying tail (at least 20% of range or 30% of body)
        rejection_buy = (c > o) and ((lower_wick >= body * 0.30) or (lower_wick >= c_range * 0.20))
        # Daily displacement: close in upper 40% of range
        close_strong_buy = c >= (h - (c_range * 0.40))
        supply_safe_buy = (c >= major_supply) or ((major_supply - c) >= (1.0 * atr))

        if sweep_buy and rejection_buy and close_strong_buy and vol_confirmed and supply_safe_buy:
            sl = round(l - (1.0 * atr), 2)
            risk = max(c - sl, c * 0.005)
            target = round(c + (risk * RR_RATIO), 2)

            signals.append({
                "Date": latest_date_str,
                "Symbol": symbol,
                "Type": "LONG",
                "Pattern": "LIQUIDITY_SWEEP_BUY",
                "Entry": round(c, 2),
                "SL": sl,
                "Target": target,
                "Close": round(c, 2),
                "Volume": int(curr['Volume']),
                "DeliveryPct": round(float(curr.get('DeliveryPct', 0.0)), 1),
                "DelivSpikeRatio": round(rvol, 2),
                "ATR": round(atr, 2)
            })

        # ----------------------------------------------------
        # 2. DAILY SELL SETUP (Grab 10-day high + Bearish Rejection)
        # ----------------------------------------------------
        grab_sell = (h > prior_high) and (c < prior_high)
        upper_wick = h - max(o, c)
        rejection_sell = (c < o) and ((upper_wick >= body * 0.30) or (upper_wick >= c_range * 0.20))
        close_weak_sell = c <= (l + (c_range * 0.40))
        demand_safe_sell = (c <= major_demand) or ((c - major_demand) >= (1.0 * atr))

        if grab_sell and rejection_sell and close_weak_sell and vol_confirmed and demand_safe_sell:
            sl = round(h + (1.0 * atr), 2)
            risk = max(sl - c, c * 0.005)
            target = round(c - (risk * RR_RATIO), 2)

            signals.append({
                "Date": latest_date_str,
                "Symbol": symbol,
                "Type": "SHORT",
                "Pattern": "LIQUIDITY_GRAB_SELL",
                "Entry": round(c, 2),
                "SL": sl,
                "Target": target,
                "Close": round(c, 2),
                "Volume": int(curr['Volume']),
                "DeliveryPct": round(float(curr.get('DeliveryPct', 0.0)), 1),
                "DelivSpikeRatio": round(rvol, 2),
                "ATR": round(atr, 2)
            })

    # Save to data/signals.csv
    os.makedirs("data", exist_ok=True)
    today_df = pd.DataFrame(signals)
    clean_cols = ["Date", "Symbol", "Type", "Pattern", "Entry", "SL", "Target", "Close", "Volume", "DeliveryPct", "DelivSpikeRatio", "ATR"]

    if os.path.exists(SIGNALS_CSV):
        try:
            existing = pd.read_csv(SIGNALS_CSV)
            existing = existing[existing['Date'] != latest_date_str]
            combined = pd.concat([today_df, existing], ignore_index=True) if not today_df.empty else existing
        except Exception:
            combined = today_df
    else:
        combined = today_df

    if not combined.empty:
        for c_name in clean_cols:
            if c_name not in combined.columns:
                combined[c_name] = 0.0
        combined = combined[clean_cols]
        combined['Parsed_Date'] = pd.to_datetime(combined['Date'], dayfirst=True, errors='coerce')
        combined = combined.sort_values(by=['Parsed_Date'], ascending=False).drop(columns=['Parsed_Date'])
    else:
        combined = pd.DataFrame(columns=clean_cols)

    combined.to_csv(SIGNALS_CSV, index=False)
    print(f"✅ Generated {len(today_df)} Daily Liquidity Setups for {latest_date_str}.")

    # Dispatch to Telegram
    records = today_df.to_dict('records') if not today_df.empty else []
    try:
        for item in records:
            send_telegram_alert(item)
            time.sleep(0.3)
    finally:
        send_summary_telegram(records, latest_date_str)


def send_telegram_alert(sig: dict):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    sym, stype, entry, sl, target = sig['Symbol'], sig['Type'], sig['Entry'], sig['SL'], sig['Target']
    rvol, atr, date = sig['DelivSpikeRatio'], sig['ATR'], sig['Date']
    is_buy = stype == "LONG"
    header = "🟢 <b>DAILY LIQUIDITY SWEEP (BUY)</b>" if is_buy else "🔴 <b>DAILY LIQUIDITY GRAB (SHORT)</b>"
    action = "Buy at Close" if is_buy else "Sell at Close"
    chart_url = f"https://in.tradingview.com/chart/?symbol=NSE:{sym}"

    msg = (
        f"{header}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {sym} (NSE F&O)\n"
        f"⏱ <b>Date:</b> {date} | <b>Timeframe:</b> Daily\n"
        f"⚡ <b>RVOL:</b> {rvol:.2f}x | <b>ATR(14):</b> ₹{atr:.2f}\n\n"
        f"📊 <b>TRADE LEVELS (1:2 R:R)</b>\n"
        f"• <b>Entry :</b> ₹{entry:.2f} ({action})\n"
        f"• <b>SL    :</b> ₹{sl:.2f} (1.0x ATR buffer)\n"
        f"• <b>Target:</b> ₹{target:.2f} (1:2)\n\n"
        f"🛡️ <b>CONFIRMATIONS</b>\n"
        f"• 10-Day Liquidity Swept & Reclaimed: ✅\n"
        f"• Supply/Demand Runway Clear: ✅\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <a href='{chart_url}'>View {sym} Chart</a>"
    )
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass


def send_summary_telegram(records: list, date_str: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    buys = sum(1 for r in records if r['Type'] == "LONG")
    sells = sum(1 for r in records if r['Type'] == "SHORT")
    msg = (
        f"🏁 <b>DAILY LIQUIDITY SWEEP SCAN COMPLETE ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Total Setups Triggered:</b> {len(records)}\n"
        f"🟢 <b>Buy Sweeps:</b> {buys}  |  🔴 <b>Sell Grabs:</b> {sells}\n"
        f"🌐 <a href='{DASHBOARD_URL}'>Open Terminal Dashboard</a>\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    run_scanner()
