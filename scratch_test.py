import psycopg

url = "postgresql://postgres.xyqqitrydmjrcbasozej:7ZEEpxl5e43ovOWf@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"

try:
    with psycopg.connect(url, prepare_threshold=None) as conn:
        with conn:
            conn.execute("SELECT 1")
        print("Inside with conn: block succeeded")
        try:
            conn.execute("SELECT 2")
            print("After with conn: block succeeded")
        except Exception as e:
            print("After with conn failed:", e)
except Exception as e:
    print(f"Failed: {e}")
