-- 005_advanced_research_schema.sql

-- Generations Tracking
CREATE TABLE IF NOT EXISTS public.research_generations (
    generation_id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES public.research_runs(run_id) ON DELETE CASCADE,
    generation_number INTEGER NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- Parameter Stability Runs
CREATE TABLE IF NOT EXISTS public.parameter_stability (
    stability_id SERIAL PRIMARY KEY,
    hypothesis_id UUID REFERENCES public.strategy_hypotheses(hypothesis_id) ON DELETE CASCADE,
    parameter_name VARCHAR(100) NOT NULL,
    parameter_value JSONB NOT NULL,
    oos_return_pct REAL,
    oos_sharpe REAL,
    max_drawdown_pct REAL,
    profit_factor REAL,
    trade_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- Regime Analysis Results
CREATE TABLE IF NOT EXISTS public.regime_analysis (
    regime_analysis_id SERIAL PRIMARY KEY,
    hypothesis_id UUID REFERENCES public.strategy_hypotheses(hypothesis_id) ON DELETE CASCADE,
    regime_type VARCHAR(50) NOT NULL, -- e.g. "LOW_VOL", "TRENDING"
    return_pct REAL,
    sharpe REAL,
    sortino REAL,
    max_drawdown_pct REAL,
    win_rate REAL,
    profit_factor REAL,
    average_trade_pct REAL,
    trade_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- Trade Concentration
CREATE TABLE IF NOT EXISTS public.trade_concentration (
    concentration_id SERIAL PRIMARY KEY,
    hypothesis_id UUID REFERENCES public.strategy_hypotheses(hypothesis_id) ON DELETE CASCADE,
    best_trade_pct REAL,
    top_5_trades_contribution_pct REAL,
    top_10_trades_contribution_pct REAL,
    best_month_pct REAL,
    best_quarter_pct REAL,
    flag VARCHAR(50), -- "LOW", "MODERATE", "HIGH"
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- Cost Survival Curves
CREATE TABLE IF NOT EXISTS public.cost_survival_curves (
    cost_survival_id SERIAL PRIMARY KEY,
    hypothesis_id UUID REFERENCES public.strategy_hypotheses(hypothesis_id) ON DELETE CASCADE,
    multiplier REAL NOT NULL,
    return_pct REAL,
    sharpe REAL,
    sortino REAL,
    max_drawdown_pct REAL,
    profit_factor REAL,
    trade_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- Alter Hypotheses to track parent lineage and scores
ALTER TABLE public.strategy_hypotheses ADD COLUMN IF NOT EXISTS parent_hypothesis_id UUID REFERENCES public.strategy_hypotheses(hypothesis_id);
ALTER TABLE public.strategy_hypotheses ADD COLUMN IF NOT EXISTS generation_number INTEGER;
ALTER TABLE public.strategy_hypotheses ADD COLUMN IF NOT EXISTS robustness_score REAL;

-- Add holdout lock
ALTER TABLE public.strategy_hypotheses ADD COLUMN IF NOT EXISTS holdout_viewed BOOLEAN DEFAULT FALSE;
