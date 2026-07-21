import duckdb

def inspect_database(db_path="data/candles.duckdb"):
    conn = duckdb.connect(db_path)
    
    # 1. Total row count
    total_rows = conn.execute("SELECT COUNT(*) FROM ohlcv_candles").fetchone()[0]
    
    # 2. Distinct symbols count
    symbol_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv_candles").fetchone()[0]
    
    # 3. Latest stored date
    latest_date = conn.execute("SELECT MAX(timestamp) FROM ohlcv_candles").fetchone()[0]
    
    print("=" * 50)
    print("📊 DUCKDB MARKET DATA SUMMARY")
    print("=" * 50)
    print(f"• Total Candle Records : {total_rows}")
    print(f"• Distinct F&O Symbols : {symbol_count}")
    print(f"• Latest Stored Date   : {latest_date}")
    print("=" * 50)
    
    # 4. Preview top 10 rows
    print("\n🔍 SAMPLE ROWS (Top 10 F&O Stocks):")
    sample_df = conn.execute("""
        SELECT symbol, timeframe, CAST(timestamp AS DATE) AS date, open, high, low, close, volume
        FROM ohlcv_candles
        ORDER BY symbol ASC
        LIMIT 10
    """).pl()
    
    print(sample_df)

if __name__ == "__main__":
    inspect_database()