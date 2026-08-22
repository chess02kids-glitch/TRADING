import os
import re

with open('kronos_trading/alerts/prediction_logger.py', 'r') as f:
    text = f.read()

text = text.replace('import psycopg', 'import psycopg\nimport sqlite3')

connect_new = '''def _connect(db_path: str):
    url = os.environ.get("SUPABASE_DB_URL")
    if url:
        return psycopg.connect(url, row_factory=dict_row, autocommit=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;").fetchall()
    conn.execute("PRAGMA synchronous=NORMAL;").fetchall()
    return conn

def _is_pg():
    return bool(os.environ.get("SUPABASE_DB_URL"))
'''
text = re.sub(r'def _connect.*?return psycopg\.connect.*?autocommit=True\)', connect_new, text, flags=re.DOTALL)

# Fix queries to be dynamic
def fix_queries(text):
    text = text.replace('INSERT INTO har_predictions', 'f\"\"\"INSERT {\"\" if _is_pg() else \"OR IGNORE \"}INTO har_predictions')
    text = text.replace('VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                    ON CONFLICT("timestamp", asset, timeframe) DO NOTHING', 'VALUES ({\", \".join([\"%s\" if _is_pg() else \"?\"] * 11)})\\n                    {\"ON CONFLICT(\\\"timestamp\\\", asset, timeframe) DO NOTHING\" if _is_pg() else \"\"}')
    text = text.replace('%s', '{ \"%s\" if _is_pg() else \"?\" }')
    
    # We need to make sure f-strings are used for execute commands.
    # Actually, simpler way: create a helper execute(conn, query, args) 
    return text

text = fix_queries(text)

with open('kronos_trading/alerts/prediction_logger.py', 'w') as f:
    f.write(text)
