import duckdb

def inspect_database(db_path="data/candles.duckdb"):
    conn = duckdb.connect(db_path)
    
    total_rows = conn.execute("SELECT COUNT(*) FROM ohlcv_candles").fetchone()[0]
    symbol_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv_candles").fetchone()[0]
    date_count = conn.execute("SELECT COUNT(DISTINCT timestamp) FROM ohlcv_candles").fetchone()[0]
    min_date = conn.execute("SELECT MIN(timestamp) FROM ohlcv_candles").fetchone()[0]
    max_date = conn.execute("SELECT MAX(timestamp) FROM ohlcv_candles").fetchone()[0]
    
    print("=" * 60)
    print("📊 DUCKDB MARKET DATA HEALTH CHECK")
    print("=" * 60)
    print(f"• Total Candle Records  : {total_rows:,}")
    print(f"• Distinct F&O Symbols  : {symbol_count}")
    print(f"• Total Trading Days    : {date_count}")
    print(f"• Date Range Stored     : {str(min_date)[:10]} to {str(max_date)[:10]}")
    print("=" * 60)
    
    print("\n📅 RECENT DAILY STOCKS COUNT:")
    daily_breakdown = conn.execute("""
        SELECT CAST(timestamp AS DATE) AS date, COUNT(*) AS total_stocks
        FROM ohlcv_candles
        GROUP BY date
        ORDER BY date DESC
        LIMIT 5
    """).pl()
    print(daily_breakdown)

if __name__ == "__main__":
    inspect_database()