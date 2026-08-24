-- Autonomous Quantitative Research Laboratory Schema

CREATE TABLE IF NOT EXISTS public.research_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_version TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    start_time_ms BIGINT NOT NULL,
    end_time_ms BIGINT NOT NULL,
    row_count INTEGER,
    random_seed INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS public.strategy_hypotheses (
    hypothesis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES public.research_runs(run_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    family TEXT NOT NULL,
    name TEXT NOT NULL,
    logic_description TEXT,
    genome JSONB NOT NULL,
    is_variant BOOLEAN DEFAULT FALSE,
    parent_hypothesis_id UUID REFERENCES public.strategy_hypotheses(hypothesis_id),
    status TEXT NOT NULL DEFAULT 'PENDING'
);

CREATE TABLE IF NOT EXISTS public.backtests (
    backtest_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hypothesis_id UUID REFERENCES public.strategy_hypotheses(hypothesis_id),
    run_id UUID REFERENCES public.research_runs(run_id),
    stage TEXT NOT NULL, -- e.g., 'SCREENING', 'WALK_FORWARD', 'HOLDOUT'
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Core metrics
    return_pct REAL,
    sharpe REAL,
    sortino REAL,
    max_drawdown_pct REAL,
    calmar REAL,
    profit_factor REAL,
    win_rate REAL,
    
    -- Execution metrics
    trade_count INTEGER,
    turnover REAL,
    average_trade_pct REAL,
    exposure_pct REAL,
    
    -- Additional stats
    raw_results JSONB
);

CREATE TABLE IF NOT EXISTS public.strategy_rejections (
    rejection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hypothesis_id UUID REFERENCES public.strategy_hypotheses(hypothesis_id),
    stage TEXT NOT NULL,
    reason TEXT NOT NULL,
    details TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.strategy_insights (
    insight_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hypothesis_id UUID REFERENCES public.strategy_hypotheses(hypothesis_id),
    insight_text TEXT NOT NULL,
    suggested_followup TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
