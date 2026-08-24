import os
import json
import logging
import uuid
import psycopg
from datetime import datetime, timezone
import sys
import pandas as pd
import hashlib

# Add the parent directory to PYTHONPATH to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import ResearchDataLoader
from strategy_genome.generator import StrategyGenerator
from vectorbt_engine.screener import run_fast_screen, filter_base_metrics
from backtesting.walk_forward import run_walk_forward
from backtesting.robustness import run_cost_stress_test, run_parameter_stability_test
from backtesting.regimes import run_regime_analysis
from backtesting.concentration import run_concentration_analysis
from strategy_genome.complexity import calculate_complexity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ResearchLoop:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.db_url = os.environ.get("SUPABASE_DB_URL")
        if not self.db_url:
            raise ValueError("SUPABASE_DB_URL not found")
            
        self.loader = ResearchDataLoader()
        self.run_id = None
        self.generation_id = None
        
    def start_generation(self, dataset_id: str = "default", gen_number: int = 1, desc: str = "Automated Batch"):
        with psycopg.connect(self.db_url, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                splits = self.loader.get_split_data(self.symbol, self.timeframe)
                train_df = splits["train"]
                start_ms = int(train_df["timestamp_ms"].min())
                end_ms = int(train_df["timestamp_ms"].max())
                row_count = len(train_df)
                
                cur.execute("""
                    INSERT INTO public.research_runs 
                    (agent_version, dataset_id, exchange, symbol, timeframe, start_time_ms, end_time_ms, row_count, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING run_id
                """, ("2.0", dataset_id, "binance", self.symbol, self.timeframe, start_ms, end_ms, row_count, "Generation " + str(gen_number)))
                self.run_id = cur.fetchone()[0]
                
                print(f"DEBUG: run_id is {self.run_id} type {type(self.run_id)}")
                cur.execute("""
                    INSERT INTO public.research_generations
                    (run_id, generation_number, description)
                    VALUES (%s, %s, %s)
                    RETURNING generation_id
                """, (self.run_id, gen_number, desc))
                self.generation_id = cur.fetchone()[0]
                conn.commit()
                
        logger.info(f"Started run_id: {self.run_id}, generation_id: {self.generation_id}")
        
    def _is_duplicate(self, genome: dict, cur) -> bool:
        """Checks if a structurally identical genome has already been evaluated."""
        # A simple hash of the sorted genome JSON
        genome_str = json.dumps(genome, sort_keys=True)
        g_hash = hashlib.md5(genome_str.encode()).hexdigest()
        # In a real rigorous system, we would hash this properly.
        # For now, we just query text match
        cur.execute("SELECT hypothesis_id FROM public.strategy_hypotheses WHERE genome::text = %s", (json.dumps(genome),))
        return cur.fetchone() is not None

    def test_hypothesis(self, hypothesis: dict, diagnostic_mode: bool = False):
        if not self.run_id: raise ValueError("Call start_generation() first")
        
        logger.info(f"Testing hypothesis: {hypothesis['name']} (Diagnostic: {diagnostic_mode})")
        
        with psycopg.connect(self.db_url, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                if self._is_duplicate(hypothesis, cur):
                    logger.warning(f"⏩ Skipping {hypothesis['name']} - DUPLICATE")
                    return False, "DUPLICATE"
                    
                cur.execute("""
                    INSERT INTO public.strategy_hypotheses 
                    (run_id, generation_number, family, name, logic_description, genome, status, parent_hypothesis_id, parent_failure_mode, research_insight, economic_mechanism, complexity_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING hypothesis_id
                """, (
                    self.run_id, hypothesis.get("generation_number", 2),
                    hypothesis["family"], hypothesis["name"], 
                    hypothesis.get("description", ""), json.dumps(hypothesis), "EVALUATING",
                    hypothesis.get("parent_hypothesis_id", None),
                    hypothesis.get("parent_failure", None),
                    hypothesis.get("research_insight", ""),
                    hypothesis.get("economic_mechanism", ""),
                    calculate_complexity(hypothesis)
                ))
                hypothesis_id = cur.fetchone()[0]
                conn.commit()

        splits = self.loader.get_split_data(self.symbol, self.timeframe)
        train_df = splits["train"]
        
        # We will collect failures instead of early returning if diagnostic_mode is True
        failed_gates = []
        def reject(stage, reason):
            with psycopg.connect(self.db_url, prepare_threshold=None) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO public.strategy_rejections (hypothesis_id, stage, reason) VALUES (%s, %s, %s)", (hypothesis_id, stage, reason))
                    if not diagnostic_mode:
                        cur.execute("UPDATE public.strategy_hypotheses SET status = 'REJECTED' WHERE hypothesis_id = %s", (hypothesis_id,))
                    conn.commit()
            logger.info(f"❌ {hypothesis['name']} FAILED GATE {stage}: {reason}")
            failed_gates.append(reason)
            if not diagnostic_mode:
                return False, reason
            return True, "CONTINUING_IN_DIAGNOSTIC"
            
        # 1. Fast Screen
        entries, exits, sizes = StrategyGenerator.compile_genome(train_df, hypothesis)
        metrics = run_fast_screen(entries, exits, sizes, train_df["close"])
        passed, r_reason = filter_base_metrics(metrics)
        if not passed: 
            ret = reject("SCREENING", r_reason)
            if not ret[0]: return ret

        # 2. Walk-Forward (WFO)
        wfo_passed, wfo_metrics = run_walk_forward(entries, exits, sizes, train_df["close"])
        if not wfo_passed: 
            ret = reject("WALK_FORWARD", "FAILED_OOS_CONSISTENCY")
            if not ret[0]: return ret
        
        # 3. Regime Analysis
        regime_results = run_regime_analysis(entries, exits, sizes, train_df)
        with psycopg.connect(self.db_url, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                for r in regime_results:
                    cur.execute("""
                        INSERT INTO public.regime_analysis 
                        (hypothesis_id, regime_type, return_pct, sharpe, sortino, max_drawdown_pct, win_rate, profit_factor, average_trade_pct, trade_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (hypothesis_id, r["regime_type"], r["return_pct"], r["sharpe"], r["sortino"], r["max_drawdown_pct"], r["win_rate"], r["profit_factor"], r["average_trade_pct"], r["trade_count"]))
                conn.commit()
                
        # 4. Trade Concentration
        conc_passed, conc_metrics, conc_flag = run_concentration_analysis(entries, exits, sizes, train_df)
        with psycopg.connect(self.db_url, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO public.trade_concentration (hypothesis_id, best_trade_pct, top_5_trades_contribution_pct, top_10_trades_contribution_pct, best_month_pct, best_quarter_pct, flag)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (hypothesis_id, conc_metrics["best_trade_pct"], conc_metrics["top_5_trades_contribution_pct"], conc_metrics["top_10_trades_contribution_pct"], conc_metrics["best_month_pct"], conc_metrics["best_quarter_pct"], conc_flag))
                conn.commit()
        if not conc_passed: 
            ret = reject("CONCENTRATION", conc_flag)
            if not ret[0]: return ret
        
        # 5. Cost Survival
        cost_passed, cost_limit, cost_curve = run_cost_stress_test(entries, exits, sizes, train_df["close"])
        with psycopg.connect(self.db_url, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                for pt in cost_curve:
                    cur.execute("""
                        INSERT INTO public.cost_survival_curves (hypothesis_id, multiplier, return_pct, sharpe, sortino, max_drawdown_pct, profit_factor, trade_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (hypothesis_id, pt["multiplier"], pt["return_pct"], pt["sharpe"], pt["sortino"], pt["max_drawdown_pct"], pt["profit_factor"], pt["trade_count"]))
                conn.commit()
        if not cost_passed: 
            ret = reject("ROBUSTNESS", "FAILED_COST_STRESS")
            if not ret[0]: return ret
        
        # 6. Parameter Stability
        stab_passed, stab_flag, stab_score, stab_runs = run_parameter_stability_test(hypothesis, train_df["close"], StrategyGenerator.compile_genome)
        with psycopg.connect(self.db_url, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                for run in stab_runs:
                    cur.execute("""
                        INSERT INTO public.parameter_stability (hypothesis_id, parameter_name, parameter_value, oos_return_pct, oos_sharpe, max_drawdown_pct, profit_factor, trade_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (hypothesis_id, run["parameter_name"], json.dumps(run["parameter_value"]), run["oos_return_pct"], run["oos_sharpe"], run["max_drawdown_pct"], run["profit_factor"], run["trade_count"]))
                conn.commit()
        if not stab_passed: 
            ret = reject("PARAMETER_STABILITY", stab_flag)
            if not ret[0]: return ret
        
        # Final Ranking Calculation
        # 25% OOS risk-adjusted (capped at 25), 20% DD (capped at 20), 15% WFO, 10% param, 10% cost, 5% regime, 5% concentration
        score_oos = min(25.0, max(0.0, wfo_metrics["oos_sharpe"] * 10))
        score_dd = min(20.0, max(0.0, (50.0 + wfo_metrics["worst_oos_drawdown"]) / 2.5)) # DD is negative
        score_wfo = min(15.0, (wfo_metrics["wfo_quality_score"] / 10.0) * 15.0)
        score_param = stab_score # out of 10
        score_cost = min(10.0, max(0.0, (cost_limit - 1.0) * 10.0))
        score_conc = 5.0 if conc_flag == "LOW_CONCENTRATION" else 2.5
        
        complexity = calculate_complexity(hypothesis)
        score_complexity = min(5.0, max(0.0, 5.0 - (complexity / 2.0)))
        
        robustness_score = score_oos + score_dd + score_wfo + score_param + score_cost + score_conc + score_complexity
        
        with psycopg.connect(self.db_url, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                final_status = 'REJECTED_DIAGNOSTIC' if diagnostic_mode and failed_gates else 'CANDIDATE'
                cur.execute("UPDATE public.strategy_hypotheses SET status = %s, robustness_score = %s WHERE hypothesis_id = %s", (final_status, robustness_score, hypothesis_id))
                
                # Update main backtest summary metrics
                cur.execute("""
                    INSERT INTO public.backtests (hypothesis_id, run_id, stage, return_pct, sharpe, sortino, max_drawdown_pct, calmar, profit_factor, win_rate, trade_count, average_trade_pct)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (hypothesis_id, self.run_id, "FINAL", metrics["return_pct"], wfo_metrics["oos_sharpe"], metrics["sortino"], wfo_metrics["worst_oos_drawdown"], metrics["calmar"], metrics["profit_factor"], metrics["win_rate"], metrics["trade_count"], metrics["average_trade_pct"]))
                conn.commit()
                
        if failed_gates:
            logger.info(f"📊 {hypothesis['name']} completed Diagnostic Mode. Failed gates: {failed_gates}")
            return False, "FAILED_GATES: " + ",".join(failed_gates)
            
        logger.info(f"🏆 {hypothesis['name']} SURVIVED all gates! Score: {robustness_score:.1f}")
        return True, "PASSED"
