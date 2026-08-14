import sqlite3
import argparse
from pathlib import Path

def main():
    db_path = Path("data/db/kronos_trading.db")
    conn = sqlite3.connect(f"file:{db_path}", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    symbols = ["BTC/USDT", "ETH/USDT"]
    tfs = [("1h", 3600_000), ("4h", 14400_000), ("1d", 86400_000)]
    
    print("=== SAFETY CHECK: ROWS TO DELETE ===")
    
    to_delete = {}
    total_to_delete = 0
    for sym in symbols:
        for tf, tf_ms in tfs:
            cur.execute(
                "SELECT count(*) as cnt FROM ohlcv_raw WHERE exchange='binance' AND symbol=? AND timeframe=? AND timestamp_ms % ? != 0",
                (sym, tf, tf_ms)
            )
            cnt = cur.fetchone()["cnt"]
            to_delete[(sym, tf)] = cnt
            total_to_delete += cnt
            print(f"{sym} {tf}: {cnt}")
            
    print("\n=== EXECUTING DELETION ===")
    
    deleted_counts = {}
    for sym in symbols:
        for tf, tf_ms in tfs:
            cur.execute(
                "DELETE FROM ohlcv_raw WHERE exchange='binance' AND symbol=? AND timeframe=? AND timestamp_ms % ? != 0",
                (sym, tf, tf_ms)
            )
            deleted = cur.rowcount
            deleted_counts[(sym, tf)] = deleted
            
    conn.commit()
    
    print("\n=== DELETED ROW COUNTS ===")
    for sym in symbols:
        for tf, tf_ms in tfs:
            print(f"{sym} {tf}: {deleted_counts[(sym, tf)]} rows deleted")

    conn.close()

if __name__ == '__main__':
    main()
