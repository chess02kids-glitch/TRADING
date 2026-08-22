import sys

with open('kronos_trading/alerts/prediction_logger.py', 'r') as f:
    text = f.read()

# 1. Imports
text = text.replace('import sqlite3', 'import sqlite3\nimport psycopg\nfrom psycopg.rows import dict_row\nimport os')

# 2. Connection logic
new_connect = '''class DBWrapper:
    def __init__(self, db_path):
        self.url = os.environ.get("SUPABASE_DB_URL")
        self.is_pg = bool(self.url)
        if self.is_pg:
            self.conn = psycopg.connect(self.url, autocommit=True, row_factory=dict_row)
        else:
            self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL;").fetchall()
            self.conn.execute("PRAGMA synchronous=NORMAL;").fetchall()
            
    def execute(self, sql, params=()):
        if self.is_pg:
            sql = sql.replace("?", "%s")
            if "INSERT OR IGNORE" in sql:
                sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO")
                sql += " ON CONFLICT(\\"timestamp\\", asset, timeframe) DO NOTHING"
        return self.conn.execute(sql, params)

    def close(self):
        self.conn.close()

def _connect(db_path: str):
    return DBWrapper(db_path)
'''
import re
text = re.sub(r'def _connect\(db_path: str\) -> sqlite3\.Connection:.*?return conn\n', new_connect, text, flags=re.DOTALL)

# 3. initialize_db should skip schema creation if pg
init_db_replacement = '''def initialize_db(db_path: str = DEFAULT_DB_PATH) -> None:
    if os.environ.get("SUPABASE_DB_URL"):
        return
    path = Path(str(db_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(path)) as conn:
        conn.execute(_SCHEMA)
'''
text = re.sub(r'def initialize_db.*?conn\.execute\(_SCHEMA\)\n', init_db_replacement, text, flags=re.DOTALL)

# 4. Remove with conn: because DBWrapper doesn't support context manager for transactions
# Actually, psycopg supports it, but our wrapper doesn't have __enter__ returning the connection.
# But wait, in the original code, with conn: is used to wrap transactions.
# In psycopg, autocommit=True avoids needing it for simple queries.
# Let's add __enter__ and __exit__ to DBWrapper!
dbwrapper_with_ctx = '''class DBWrapper:
    def __init__(self, db_path):
        self.url = os.environ.get("SUPABASE_DB_URL")
        self.is_pg = bool(self.url)
        if self.is_pg:
            self.conn = psycopg.connect(self.url, autocommit=True, row_factory=dict_row)
        else:
            self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL;").fetchall()
            self.conn.execute("PRAGMA synchronous=NORMAL;").fetchall()
            
    def execute(self, sql, params=()):
        if self.is_pg:
            sql = sql.replace("?", "%s")
            if "INSERT OR IGNORE" in sql:
                sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO")
                sql += "\\nON CONFLICT(\\"timestamp\\", asset, timeframe) DO NOTHING"
        return self.conn.execute(sql, params)

    def close(self):
        self.conn.close()

    def __enter__(self):
        self.ctx = self.conn.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.conn.__exit__(exc_type, exc_val, exc_tb)

def _connect(db_path: str):
    return DBWrapper(db_path)
'''
text = text.replace(new_connect, dbwrapper_with_ctx)

with open('kronos_trading/alerts/prediction_logger.py', 'w') as f:
    f.write(text)
