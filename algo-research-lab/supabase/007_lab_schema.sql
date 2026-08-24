-- 007: research_generations table for the Gen 1 v2 reset of the
-- algo-research-lab. Idempotent: creates the table if missing and adds
-- any missing columns to an existing table (never drops anything).

CREATE TABLE IF NOT EXISTS public.research_generations (
  id                 SERIAL PRIMARY KEY,
  generation         INTEGER,
  genome_id          TEXT,
  genome             TEXT,          -- JSONB in Postgres
  signal_type        TEXT,
  asset              TEXT,
  total_trades       INTEGER,
  profit_factor      REAL,
  sharpe_ratio       REAL,
  max_drawdown       REAL,
  total_return_pct   REAL,
  passed_all_gates   BOOLEAN,
  gate_failed        TEXT,
  failure_reason     TEXT,
  oos_sharpe         REAL,
  oos_positive_splits INTEGER,
  concentration_score REAL,
  robustness_score   REAL,
  stability_score    REAL,
  created_at         TIMESTAMPTZ DEFAULT NOW(),
  seed               INTEGER,
  parent_genome_ids  TEXT[]
);

-- Add-column guards (no-ops when columns already exist).
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS generation        INTEGER;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS genome_id         TEXT;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS genome            TEXT;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS signal_type       TEXT;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS asset             TEXT;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS total_trades      INTEGER;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS profit_factor     REAL;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS sharpe_ratio      REAL;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS max_drawdown      REAL;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS total_return_pct  REAL;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS passed_all_gates  BOOLEAN;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS gate_failed       TEXT;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS failure_reason    TEXT;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS oos_sharpe        REAL;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS oos_positive_splits INTEGER;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS concentration_score REAL;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS robustness_score  REAL;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS stability_score   REAL;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS created_at        TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS seed              INTEGER;
ALTER TABLE public.research_generations ADD COLUMN IF NOT EXISTS parent_genome_ids TEXT[];
