from datetime import datetime, timedelta
from bhavcopy_ingest import BhavcopyIngestor

def run_90_day_backfill():
    ingestor = BhavcopyIngestor(db_path="data/candles.duckdb")
    today = datetime.today()
    
    # Generate past 90 days dates
    past_dates = [today - timedelta(days=i) for i in range(90)]
    
    print("🚀 Starting 90-Day Historical Data Backfill...")
    print("=" * 50)
    
    success_count = 0
    skipped_count = 0

    for dt in reversed(past_dates):  # Process oldest to newest
        # Skip Saturday (5) and Sunday (6)
        if dt.weekday() in (5, 6):
            continue
            
        date_str = dt.strftime("%Y-%m-%d")
        try:
            ingestor.fetch_and_store_daily(date_str)
            success_count += 1
        except Exception as e:
            print(f"⚠️ Error fetching {date_str}: {e}")
            skipped_count += 1

    print("=" * 50)
    print(f"✅ Backfill Complete!")
    print(f"• Days successfully ingested : {success_count}")
    print(f"• Days skipped/unavailable  : {skipped_count}")

if __name__ == "__main__":
    run_90_day_backfill()
