import os
import psycopg
import sys
from dotenv import load_dotenv

def main():
    load_dotenv()
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        print("ERROR: SUPABASE_DB_URL not found.")
        sys.exit(1)

    schema_file = os.path.join(os.path.dirname(__file__), "004_research_schema.sql")
    with open(schema_file, "r") as f:
        sql = f.read()

    print(f"Applying schema to {db_url}...")
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            print("Successfully applied research schema.")
    except Exception as e:
        print(f"Error applying schema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
