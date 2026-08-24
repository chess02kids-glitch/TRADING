import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.research_loop import ResearchLoop

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_hypotheses():
    hypotheses = []
    h_id = 0
    
    # Base Directions
    directions = [
        {"family": "trend", "indicator": "sma_crossover", "params_list": [{"fast": 10, "slow": 50}, {"fast": 20, "slow": 100}, {"fast": 5, "slow": 20}]},
        {"family": "breakout", "indicator": "donchian", "params_list": [{"window": 20}, {"window": 48}, {"window": 96}]},
        {"family": "mean_reversion", "indicator": "rsi", "params_list": [{"window": 14, "lower": 30, "upper": 70}, {"window": 7, "lower": 20, "upper": 80}]},
        {"family": "momentum", "indicator": "roc", "params_list": [{"window": 14, "threshold": 2.0}, {"window": 24, "threshold": 5.0}]}
    ]
    
    regimes = [
        {}, # no regime filter
        {"allowed_trend_regimes": ["TRENDING"]},
        {"allowed_trend_regimes": ["RANGING"]},
        {"allowed_vol_regimes": ["LOW_VOL", "NORMAL_VOL"]},
        {"allowed_vol_regimes": ["HIGH_VOL", "EXTREME_VOL"]}
    ]
    
    confirmations = [
        {}, # no confirmation
        {"type": "trend"},
        {"type": "volatility"}
    ]
    
    for d in directions:
        for p in d["params_list"]:
            for r in regimes:
                for c in confirmations:
                    h_id += 1
                    name = f"GEN1_{d['family'][:3].upper()}_{h_id:03d}"
                    desc = f"{d['indicator']} with p={list(p.values())}, reg={list(r.values()) if r else 'None'}, conf={c.get('type','None')}"
                    
                    hypo = {
                        "generation_number": 1,
                        "family": d["family"],
                        "name": name,
                        "description": desc,
                        "direction": {
                            "family": d["family"],
                            "indicator": d["indicator"],
                            "params": p
                        },
                        "regime": r,
                        "confirmation": c
                    }
                    hypotheses.append(hypo)
                    
    return hypotheses

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    loop = ResearchLoop("BTC/USDT", "1h")
    loop.start_generation(dataset_id="btc_usd_1h_gen1", gen_number=1, desc="Generation 1: Automated Discovery")
    
    hypos = generate_hypotheses()
    logger.info(f"Generated {len(hypos)} hypotheses for Gen 1.")
    
    survivors = 0
    for h in hypos:
        passed, reason = loop.test_hypothesis(h)
        if passed:
            survivors += 1
            
    logger.info("="*50)
    logger.info("RESEARCH BATCH REPORT")
    logger.info("="*50)
    logger.info(f"Total evaluated: {len(hypos)}")
    logger.info(f"Total survivors: {survivors}")
    logger.info("="*50)
