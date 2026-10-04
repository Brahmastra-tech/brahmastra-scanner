import os
import io
import re
import time
import duckdb
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
DB_PATH = "data/candles.duckdb"
SIGNALS_CSV = "data/signals.csv"

LOCAL_CSV_PATH = r"D:\Scanner\fo_mktlots.csv"
REPO_CSV_PATH = "fo_mktlots.csv"

ATR_PERIOD = 14
RVOL_PERIOD = 20
LOCAL_LIQ_PERIOD = 10
HTF_PERIOD = 20

MIN_PRICE = 100.0
RR_RATIO = 2.0  # Strict 1:2 Risk to Reward

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
DASHBOARD_URL = "https://brahmastra-tech.github.io/brahmastra-scanner/"


def load_fo_universe() -> frozenset:
    """Robust extractor for NSE fo_mktlots.csv that captures all 210+ equity symbols."""
    file_path = LOCAL_CSV_PATH if os.path.exists(LOCAL_CSV_PATH) else REPO_CSV_PATH

    if not os.path.exists(file_path):
        print(f"⚠️ {file_path} not found. Falling back to internal list.")
        return frozenset()

    symbols = set()
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]

        header_idx = -1
        for idx, line in enumerate(lines[:15]):
            line_up = line.upper()
            if "SYMBOL" in line_up or "UNDERLYING" in line_up:
                header_idx = idx
                break

        if header_idx != -1:
            csv_data = "\n".join(lines[header_idx:])
            df = pd.read_csv(io.StringIO(csv_data), skipinitialspace=True)
            df.columns = [str(c).strip().upper() for c in df.columns]

            target_col = None
            for candidate in ["SYMBOL", "UNDERLYING", "SECURITY"]:
                for c in df.columns:
                    if candidate in c:
                        target_col = c
                        break
                if target_col:
                    break

            if target_col:
                raw_syms = df[target_col].dropna().astype(str).str.strip().str.upper().unique()
                symbols = {
                    s for s in raw_syms 
                    if s and s.isalnum() and not any(idx in s for idx in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])
                }

        # Backup line scanner if standard CSV parsing fell short
        if len(symbols) < 100:
            for line in lines:
                parts = [p.strip().upper() for p in line.split(",") if p.strip()]
                for p in parts:
                    if re.match(r"^[A-Z0-9&-]{2,15}$", p):
                        if not any(idx in p for idx in ["NIFTY", "EXPIRY", "LOT", "SYMBOL", "UNDERLYING", "DERIVATIVES", "NAME"]):
                            symbols.add(p)

        print(f"✅ Successfully loaded {len(symbols)} pure F&O symbols from {file_path}")
        return frozenset(symbols)

    except Exception as e:
        print(f"⚠️ Error parsing fo_mktlots.csv: {e}")
        return frozenset()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates ATR(14), RVOL(20), 10D Local Liquidity, and 20D HTF structure levels."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    close_prev = close.shift(1)

    # 1. ATR(14)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()

    # 2. RVOL (Current Volume / 20-day Volume SMA)
    vol_sma20 = df['Volume'].rolling(window=RVOL_PERIOD, min_periods=5).mean()
    df['RVOL'] = (df['Volume'] / (vol_sma20 + 1e-5)).round(2)

    # 3. Local Liquidity (10 sessions, excluding current bar)
    df['prior_low'] = low.shift(1).rolling(window=LOCAL_LIQ_PERIOD, min_periods=LOCAL_LIQ_PERIOD).min()
    df['prior_high'] = high.shift(1).rolling(window=LOCAL_LIQ_PERIOD, min_periods=LOCAL_LIQ_PERIOD).max()

    # 4. HTF Structure Levels (20 sessions, excluding current bar)
    df['major_demand'] = low.shift(1).rolling(window=HTF_PERIOD, min_periods=HTF_PERIOD).min()
    df['major_supply'] = high.shift(1).rolling(window=HTF_PERIOD, min_periods=HTF_PERIOD).max()

    return df


def run_scanner():
    print("🚀 Initializing Institutional Daily Liquidity Sweep Engine...")
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
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
        print("⚠️ No candle data available in database.")
        return

    universe = load_fo_universe()
    df_raw["Symbol_Clean"] = (
        df_raw["Symbol"].astype(str).str.upper().str.strip()
        .str.replace(r"-EQ$", "", regex=True)
        .str.replace(r"\.EQ$", "", regex=True)
    )

    if universe:
        df_raw = df_raw[df_raw["Symbol_Clean"].isin(universe)].copy()

    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])
    latest_date_str = df_raw["Date_DT"].max().strftime("%d-%m-%Y")

    signals = []

    for symbol, df_sym in df_raw.groupby("Symbol_Clean"):
        if len(df_sym) < 22:
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

        # Volume confirmation
        vol_confirmed = (rvol >= 1.10)

        # ==========================================================
        # 1. BUY SETUP (10D Low Sweep + Bullish Reclaim + Runway)
        # ==========================================================
        sweep_buy = (l < prior_low) and (c > prior_low)
        lower_wick = min(o, c) - l
        rejection_buy = (c > o) and ((lower_wick >= (body * 0.35)) or (lower_wick >= (c_range * 0.20)))
        displacement_buy = (c >= (h - (c_range * 0.35))) and (c > prev['High'])
        supply_safe_buy = (c >= major_supply) or ((major_supply - c) >= (1.2 * atr))

        if sweep_buy and rejection_buy and displacement_buy and vol_confirmed and supply_safe_buy:
            sl = round(l - (1.0 * atr), 2)
            risk = max(c - sl, c * 0.005)
            target = round(c + (risk * RR_RATIO), 2)
            deliv_pct = round(float(curr.get('DeliveryPct', 0.0)), 1)

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
                "DeliveryPct": deliv_pct,
                "DelivSpikeRatio": round(rvol, 2),
                "ATR": round(atr, 2)
            })

        # ==========================================================
        # 2. SELL SETUP (10D High Grab + Bearish Reclaim + Runway)
        # ==========================================================
        grab_sell = (h > prior_high) and (c < prior_high)
        upper_wick = h - max(o, c)
        rejection_sell = (c < o) and ((upper_wick >= (body * 0.35)) or (upper_wick >= (c_range * 0.20)))
        displacement_sell = (c <= (l + (c_range * 0.35))) and (c < prev['Low'])
        demand_safe_sell = (c <= major_demand) or ((c - major_demand) >= (1.2 * atr))

        if grab_sell and rejection_sell and displacement_sell and vol_confirmed and demand_safe_sell:
            sl = round(h + (1.0 * atr), 2)
            risk = max(sl - c, c * 0.005)
            target = round(c - (risk * RR_RATIO), 2)
            deliv_pct = round(float(curr.get('DeliveryPct', 0.0)), 1)

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
                "DeliveryPct": deliv_pct,
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
    print(f"✅ Scan Complete: {len(today_df)} Institutional setups recorded for {latest_date_str}.")

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
    header = "🟢 <b>INSTITUTIONAL LIQUIDITY SWEEP (BUY)</b>" if is_buy else "🔴 <b>INSTITUTIONAL LIQUIDITY GRAB (SELL)</b>"
    action = "Execute BUY at CMP" if is_buy else "Execute SHORT at CMP"
    chart_url = f"https://in.tradingview.com/chart/?symbol=NSE:{sym}"

    msg = (
        f"{header}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Stock:</b> {sym} (NSE F&O EQ)\n"
        f"⏱ <b>Date:</b> {date} | <b>Timeframe:</b> Daily\n"
        f"⚡ <b>RVOL:</b> {rvol:.2f}x | <b>ATR(14):</b> ₹{atr:.2f}\n\n"
        f"📊 <b>TRADE PARAMETERS (1:2 R:R)</b>\n"
        f"• <b>Action Entry  :</b> ₹{entry:.2f} ({action})\n"
        f"• <b>Stop Loss (SL):</b> ₹{sl:.2f} (1.0x ATR buffer)\n"
        f"• <b>Target (TP)   :</b> ₹{target:.2f} (Strict 1:2)\n\n"
        f"🛡️ <b>CONFIRMATIONS</b>\n"
        f"• 10-Day Local Liquidity Sweep & Reclaim: ✅\n"
        f"• 1.2x ATR HTF Structural Clearance: ✅\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <a href='{chart_url}'>Open TradingView Chart</a>"
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
        f"🏛️ <b>Strategy:</b> 10D Liquidity Sweep + Reclaim + 1:2 R:R\n"
        f"🌐 <a href='{DASHBOARD_URL}'>Open Terminal Dashboard</a>\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )

    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    run_scanner()
