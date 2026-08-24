-- Generation 2: Evidence-Driven Lineage and Complexity Tracking

-- 1. Add complexity tracking and lineage tracking to strategy_hypotheses
ALTER TABLE public.strategy_hypotheses
ADD COLUMN parent_hypothesis_id UUID REFERENCES public.strategy_hypotheses(hypothesis_id),
ADD COLUMN parent_failure_mode VARCHAR(255),
ADD COLUMN research_insight TEXT,
ADD COLUMN economic_mechanism TEXT,
ADD COLUMN complexity_score INTEGER DEFAULT 0;
