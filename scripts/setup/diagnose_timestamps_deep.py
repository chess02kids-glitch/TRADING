import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timezone

def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()

def main():
    db_path = Path("data/db/kronos_trading.db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    symbols = ["BTC/USDT", "ETH/USDT"]
    tfs = [("1h", 3600_000), ("4h", 14400_000), ("1d", 86400_000)]
    
    print("=== MALFORMED BLOCK ANALYSIS ===")
    for sym in symbols:
        for tf, tf_ms in tfs:
            cur.execute(
                "SELECT timestamp_ms, timestamp_utc, open, high, low, close, volume "
                "FROM ohlcv_raw WHERE exchange='binance' AND symbol=? AND timeframe=? "
                "ORDER BY timestamp_ms ASC", (sym, tf)
            )
            rows = cur.fetchall()
            if not rows: continue
            
            malformed = []
            valid_before = None
            
            for i, r in enumerate(rows):
                if r["timestamp_ms"] % tf_ms != 0:
                    malformed.append((i, r))
                elif not malformed:
                    valid_before = r
                    
            if not malformed:
                continue
            
            last_malformed_idx = malformed[-1][0]
            valid_after = rows[last_malformed_idx+1] if last_malformed_idx + 1 < len(rows) else None
            
            # Check continuity
            contiguous = True
            for k in range(1, len(malformed)):
                if malformed[k][0] != malformed[k-1][0] + 1:
                    contiguous = False
                    break
                    
            # Check offsets and validity
            offsets = set()
            all_valid_ohlc = True
            for idx, m in malformed:
                offsets.add(m["timestamp_ms"] % tf_ms)
                # validity
                o, h, l, c, v = m["open"], m["high"], m["low"], m["close"], m["volume"]
                if not (h >= l and h >= o and h >= c and l <= o and l <= c and o > 0 and v >= 0):
                    all_valid_ohlc = False
            
            print(f"\nSeries: {sym} {tf}")
            print(f"First misaligned: {iso(malformed[0][1]['timestamp_ms'])} ({malformed[0][1]['timestamp_ms']})")
            print(f"Last misaligned: {iso(malformed[-1][1]['timestamp_ms'])} ({malformed[-1][1]['timestamp_ms']})")
            print(f"Number affected: {len(malformed)}")
            print(f"Valid BEFORE block: {iso(valid_before['timestamp_ms']) if valid_before else 'None'}")
            print(f"Valid AFTER block: {iso(valid_after['timestamp_ms']) if valid_after else 'None'}")
            print(f"Contiguous block: {contiguous}")
            print(f"All OHLC/Volume valid: {all_valid_ohlc}")
            print(f"Offset distribution (ms): {offsets}")
            
    print("\n=== FETCH PROVENANCE ===")
    cur.execute(
        "SELECT id, symbol, timeframe, fetch_start_ms, fetch_end_ms, created_at, candles_inserted "
        "FROM fetch_metadata "
        "WHERE created_at >= '2026-07-11' AND created_at <= '2026-07-12' "
        "ORDER BY id ASC"
    )
    for fm in cur.fetchall():
        print(f"ID {fm['id']}: {fm['symbol']} {fm['timeframe']} - Created: {fm['created_at']} | Inserted: {fm['candles_inserted']}")
        
if __name__ == '__main__':
    main()
