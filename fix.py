import re

with open('kronos_trading/alerts/prediction_logger.py', 'r') as f:
    text = f.read()

# Fix imports and connect
text = re.sub(r'import sqlite3\s*', 'import psycopg\nfrom psycopg.rows import dict_row\nimport os\n', text)

connect_old = r'''def _connect\(db_path: str\) -> sqlite3\.Connection:.*?return conn'''
connect_new = '''def _connect(db_path: str):
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("SUPABASE_DB_URL missing")
    return psycopg.connect(url, row_factory=dict_row, autocommit=True)'''
text = re.sub(connect_old, connect_new, text, flags=re.DOTALL)

# Fix queries
text = text.replace('INSERT OR IGNORE INTO', 'INSERT INTO')
text = text.replace('VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                    ON CONFLICT("timestamp", asset, timeframe) DO NOTHING')
text = text.replace('WHERE "timestamp" = ?', 'WHERE "timestamp" = %s')
text = text.replace('AND asset = ? AND timeframe = ?', 'AND asset = %s AND timeframe = %s')
text = text.replace('SET actual_range = ?,', 'SET actual_range = %s,')
text = text.replace('prediction_error = ?,', 'prediction_error = %s,')
text = text.replace('abs_prediction_error = ?,', 'abs_prediction_error = %s,')
text = text.replace('breakout_flag = ?', 'breakout_flag = %s')

with open('kronos_trading/alerts/prediction_logger.py', 'w') as f:
    f.write(text)
