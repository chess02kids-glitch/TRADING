import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

TIMEFRAME_MS = {"1h": 3600_000, "4h": 14400_000, "1d": 86400_000}
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAMES = ["1h", "4h", "1d"]

def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()

def main():
    db_path = Path("data/db/kronos_trading.db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print("=" * 86)
    print("TIMESTAMP DIAGNOSTIC")
    
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            print("-" * 86)
            print(f"Series: {symbol} {tf}")
            cur.execute(
                "SELECT timestamp_ms, timestamp_utc FROM ohlcv_raw WHERE exchange='binance' AND symbol=? AND timeframe=? ORDER BY timestamp_ms ASC",
                (symbol, tf)
            )
            rows = cur.fetchall()
            if not rows:
                print("No rows.")
                continue
                
            tf_ms = TIMEFRAME_MS[tf]
            
            intervals = Counter()
            offsets = Counter()
            
            misaligned_samples = []
            
            prev_ms = None
            for i, row in enumerate(rows):
                ms = row["timestamp_ms"]
                utc_str = row["timestamp_utc"]
                
                offset = ms % tf_ms
                offsets[offset] += 1
                
                if offset != 0 and len(misaligned_samples) < 3:
                    prev_r = rows[i-1] if i > 0 else None
                    next_r = rows[i+1] if i < len(rows)-1 else None
                    misaligned_samples.append({
                        "row": row,
                        "prev": prev_r,
                        "next": next_r,
                        "offset": offset
                    })
                
                if prev_ms is not None:
                    intervals[ms - prev_ms] += 1
                prev_ms = ms
            
            print("Interval Distribution:")
            for interval, count in intervals.most_common(5):
                print(f"  {interval} ms ({interval/3600000:.2f}h): {count} occurrences")
            
            print("Offset from Boundary Distribution:")
            for offset, count in offsets.most_common(5):
                print(f"  {offset} ms: {count} occurrences")
                
            if misaligned_samples:
                print("Representative Misaligned Samples:")
                for sample in misaligned_samples:
                    ms = sample["row"]["timestamp_ms"]
                    offset = sample["offset"]
                    expected = ms - offset
                    
                    prev_str = iso(sample["prev"]["timestamp_ms"]) if sample["prev"] else "None"
                    next_str = iso(sample["next"]["timestamp_ms"]) if sample["next"] else "None"
                    
                    print(f"  Raw ms: {ms}")
                    print(f"  Stored UTC: {sample['row']['timestamp_utc']}")
                    print(f"  Converted UTC: {iso(ms)}")
                    print(f"  Expected Boundary: {iso(expected)} ({expected})")
                    print(f"  Offset: {offset} ms")
                    print(f"  Prev: {prev_str}")
                    print(f"  Current: {iso(ms)}")
                    print(f"  Next: {next_str}")
                    print()
                    
            if symbol == "BTC/USDT" and tf == "4h":
                print("Investigating Missing Candle around 2026-07-26T05:37:43.241000+00:00")
                target_iso = "2026-07-26T05:37:43.241000+00:00"
                target_ms = int(datetime.fromisoformat(target_iso).timestamp() * 1000)
                
                cur.execute(
                    "SELECT timestamp_ms, timestamp_utc FROM ohlcv_raw WHERE exchange='binance' AND symbol=? AND timeframe=? AND timestamp_ms BETWEEN ? AND ? ORDER BY timestamp_ms ASC",
                    (symbol, tf, target_ms - 2*tf_ms, target_ms + 2*tf_ms)
                )
                surrounding = cur.fetchall()
                for s in surrounding:
                    print(f"  {s['timestamp_ms']} -> {iso(s['timestamp_ms'])}")

if __name__ == '__main__':
    main()
