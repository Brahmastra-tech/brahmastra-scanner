import os
import duckdb
import pandas as pd
import numpy as np

DB_PATH = "data/candles.duckdb"
BACKTEST_CSV = "data/backtest_results.csv"
TARGET_X = 3.0

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Wilder's RSI matching Chartink's standard 14-period RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-5)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def run_90day_backfill():
    print("⏳ Running Chartink-Aligned Backfill Engine (Past 90 Days)...")

    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}.")
        return

    conn = duckdb.connect(DB_PATH)
    cols_info = conn.execute("DESCRIBE ohlcv_candles").fetchall()
    existing_cols = [col[0].lower() for col in cols_info]

    select_parts = [
        "symbol AS Symbol",
        "CAST(timestamp AS DATE) AS Date",
        "open AS Open", "high AS High", "low AS Low", "close AS Close", "volume AS Volume"
    ]
    if "delivery_qty" in existing_cols:
        select_parts.append("delivery_qty AS DeliveryQty")
    else:
        select_parts.append("volume * 0.45 AS DeliveryQty")

    if "delivery_pct" in existing_cols:
        select_parts.append("delivery_pct AS DeliveryPct")
    else:
        select_parts.append("45.0 AS DeliveryPct")

    query = f"SELECT {', '.join(select_parts)} FROM ohlcv_candles ORDER BY symbol, timestamp ASC"
    df_raw = conn.execute(query).df()
    conn.close()

    if df_raw.empty:
        print("⚠️ No data in DuckDB.")
        return

    df_raw["Date_DT"] = pd.to_datetime(df_raw["Date"])
    all_historical_signals = []

    for symbol, df_sym in df_raw.groupby('Symbol'):
        if len(df_sym) < 20:
            continue

        df = df_sym.copy().sort_values("Date_DT").reset_index(drop=True)

        # 1. Daily RSI & Previous Day Shift
        df['RSI_Daily'] = compute_rsi(df['Close'], 14)
        df['Prev_RSI_Daily'] = df['RSI_Daily'].shift(1)

        # 2. Resample Weekly & Monthly RSIs
        df_weekly = df.set_index('Date_DT').resample('W-FRI').agg({'Close': 'last'}).dropna()
        df_weekly['RSI_Weekly'] = compute_rsi(df_weekly['Close'], 14) if len(df_weekly) >= 14 else 55.0
        df_weekly['Prev_RSI_Weekly'] = df_weekly['RSI_Weekly'].shift(1) if len(df_weekly) >= 14 else 50.0

        df_monthly = df.set_index('Date_DT').resample('ME').agg({'Close': 'last'}).dropna()
        df_monthly['RSI_Monthly'] = compute_rsi(df_monthly['Close'], 14) if len(df_monthly) >= 14 else 55.0
        df_monthly['Prev_RSI_Monthly'] = df_monthly['RSI_Monthly'].shift(1) if len(df_monthly) >= 14 else 50.0

        # Merge MTF RSIs cleanly
        df = pd.merge_asof(df, df_weekly[['RSI_Weekly', 'Prev_RSI_Weekly']], on='Date_DT', direction='backward')
        df = pd.merge_asof(df, df_monthly[['RSI_Monthly', 'Prev_RSI_Monthly']], on='Date_DT', direction='backward')

        # Fallback values for short history
        df['RSI_Weekly'] = df['RSI_Weekly'].fillna(55.0)
        df['Prev_RSI_Weekly'] = df['Prev_RSI_Weekly'].fillna(50.0)
        df['RSI_Monthly'] = df['RSI_Monthly'].fillna(55.0)
        df['Prev_RSI_Monthly'] = df['Prev_RSI_Monthly'].fillna(50.0)

        df['Deliv_SMA20'] = df['DeliveryQty'].rolling(20, min_periods=3).mean().fillna(df['DeliveryQty'])
        df['Deliv_Spike'] = (df['DeliveryQty'] / (df['Deliv_SMA20'] + 1e-5)).fillna(1.0)
        df['Day_Range'] = (df['High'] - df['Low']).clip(lower=1e-5)
        df['Close_Location'] = ((df['Close'] - df['Low']) / df['Day_Range']).fillna(0.5)

        # 3. Bar-by-Bar Evaluation Across Past 90 Trading Days
        start_idx = max(2, len(df) - 90)
        for i in range(start_idx, len(df)):
            row = df.iloc[i]

            if pd.isna(row['Prev_RSI_Daily']):
                continue

            # Chartink Filters:
            # 1. Monthly RSI > Prev Monthly RSI
            # 2. Weekly RSI > Prev Weekly RSI
            # 3. Daily RSI > Prev Daily RSI * 0.99
            # 4. Daily RSI < Prev Daily RSI * 1.01
            c_mtf_monthly = row['RSI_Monthly'] > row['Prev_RSI_Monthly']
            c_mtf_weekly = row['RSI_Weekly'] > row['Prev_RSI_Weekly']
            
            c_daily_rsi_lower = row['RSI_Daily'] > (row['Prev_RSI_Daily'] * 0.99)
            c_daily_rsi_upper = row['RSI_Daily'] < (row['Prev_RSI_Daily'] * 1.01)

            if not (c_mtf_monthly and c_mtf_weekly and c_daily_rsi_lower and c_daily_rsi_upper):
                continue

            # Scoring Metric
            deliv_score = np.clip((row['DeliveryPct'] / 70.0 * 50.0), 10, 50)
            close_score = np.clip(row['Close_Location'] * 50.0, 10, 50)
            brs_score = round(float(deliv_score + close_score), 2)

            entry = round(float(row['High']) + 0.05, 2)
            sl = round(float(row['Low']), 2)
            risk = max(entry - sl, float(row['Close']) * 0.01)
            target = round(entry + (risk * TARGET_X), 2)

            # Trade Outcome Evaluation over next 15 bars
            outcome = "OPEN"
            exit_price = float(row['Close'])
            bars_held = 0

            future_bars = df.iloc[i + 1 : min(i + 16, len(df))]
            for _, f_row in future_bars.iterrows():
                bars_held += 1
                if f_row['High'] >= target:
                    outcome = "WIN (TARGET HIT)"
                    exit_price = target
                    break
                elif f_row['Low'] <= sl:
                    outcome = "LOSS (SL HIT)"
                    exit_price = sl
                    break

            if outcome == "OPEN" and not future_bars.empty:
                exit_price = float(future_bars.iloc[-1]['Close'])
                pnl_temp = round(((exit_price - entry) / entry) * 100, 2)
                outcome = "WIN (OPEN)" if pnl_temp > 0 else "LOSS (OPEN)"

            pnl_pct = round(((exit_price - entry) / entry) * 100, 2)

            all_historical_signals.append({
                "Date": row['Date_DT'].strftime("%Y-%m-%d"),
                "Symbol": symbol,
                "BRS_Score": brs_score,
                "Entry": entry,
                "SL": sl,
                "Target": target,
                "Exit_Price": exit_price,
                "Outcome": outcome,
                "PnL_Pct": pnl_pct,
                "Bars_Held": bars_held,
                "DeliveryPct": round(float(row['DeliveryPct']), 2),
                "Daily_RSI": round(float(row['RSI_Daily']), 2)
            })

    os.makedirs("data", exist_ok=True)

    if all_historical_signals:
        bt_df = pd.DataFrame(all_historical_signals).sort_values(["Date", "BRS_Score"], ascending=[False, False])
        bt_df.to_csv(BACKTEST_CSV, index=False)

        total = len(bt_df)
        wins = len(bt_df[bt_df['Outcome'].str.startswith("WIN")])
        win_rate = round((wins / total) * 100, 2) if total > 0 else 0
        avg_pnl = round(bt_df['PnL_Pct'].mean(), 2) if total > 0 else 0

        print(f"\n✅ Backfill Execution Success: Generated {total} historical matches.")
        print(f"📊 Historical Win Rate: {win_rate}% ({wins}/{total}) | Avg PnL: {avg_pnl}%")
        print(f"💾 Results stored in {BACKTEST_CSV}\n")
    else:
        print("⚠️ 0 matches found across the historical window. Checking database candle counts...")

if __name__ == "__main__":
    run_90day_backfill()