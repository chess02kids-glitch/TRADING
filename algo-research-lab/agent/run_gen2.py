import os
import sys
import logging
import uuid
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.research_loop import ResearchLoop

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_hypotheses():
    hypotheses = []
    h_id = 0
    
    # 1. Dynamic Trend / Breakout (30%)
    trend_windows = [20, 48, 96]
    sizings = [{"type": "trend_strength"}, {"type": "breakout_strength"}, {"type": "volatility_inverse"}]
    
    for w in trend_windows:
        for s in sizings:
            h_id += 1
            hypotheses.append({
                "generation_number": 2,
                "parent_failure": "LOW_TRADE_COUNT",
                "research_insight": "Binary regime filters cull too many trades. Continuous position sizing adjusts exposure smoothly.",
                "economic_mechanism": f"Breakout strategy using dynamic {s['type']} for sizing.",
                "family": "breakout",
                "name": f"GEN2_DYN_{h_id:03d}",
                "direction": {
                    "family": "breakout",
                    "indicator": "donchian",
                    "params": {"window": w}
                },
                "sizing": s
            })
            
    # 2. Volatility Expansion Breakouts
    for w in trend_windows:
        for s in sizings:
            h_id += 1
            hypotheses.append({
                "generation_number": 2,
                "parent_failure": "FAILED_COST_STRESS",
                "research_insight": "Static breakouts decay in costs. Expansion filters improve survival.",
                "economic_mechanism": f"Volatility expansion confirmed breakout with {s['type']} sizing.",
                "family": "breakout",
                "name": f"GEN2_VOLEXP_{h_id:03d}",
                "direction": {
                    "family": "breakout",
                    "indicator": "volatility_expansion",
                    "params": {"window": w}
                },
                "sizing": s
            })

    # 3. Structural Mean Reversion (Z-score, ATR normalized)
    mr_windows = [24, 48]
    mr_thresholds = [-1.5, -2.0, -2.5]
    for w in mr_windows:
        for t in mr_thresholds:
            for s in sizings:
                h_id += 1
                hypotheses.append({
                    "generation_number": 2,
                    "parent_failure": "HIGH_CONCENTRATION",
                    "research_insight": "Simple RSI is extremely concentrated. Z-score MR is statistically normalized.",
                    "economic_mechanism": f"Z-score mean reversion (w={w}, t={t}) scaled by {s['type']}.",
                    "family": "mean_reversion",
                    "name": f"GEN2_ZSCORE_{h_id:03d}",
                    "direction": {
                        "family": "mean_reversion",
                        "indicator": "z_score",
                        "params": {"window": w, "threshold": t}
                    },
                    "sizing": s
                })
                h_id += 1
                hypotheses.append({
                    "generation_number": 2,
                    "parent_failure": "HIGH_CONCENTRATION",
                    "research_insight": "ATR normalized deviation captures short horizon mean-reversion without huge kurtosis.",
                    "economic_mechanism": f"ATR normalized mean reversion (w={w}, t={t}) scaled by {s['type']}.",
                    "family": "mean_reversion",
                    "name": f"GEN2_ATR_MR_{h_id:03d}",
                    "direction": {
                        "family": "mean_reversion",
                        "indicator": "atr_normalized",
                        "params": {"window": w, "threshold": t}
                    },
                    "sizing": s
                })

    # 4. Multi-timeframe strategies
    for w in trend_windows:
        for s in sizings:
            h_id += 1
            hypotheses.append({
                "generation_number": 2,
                "parent_failure": "FAILED_OOS_CONSISTENCY",
                "research_insight": "1h trend signals get chopped in ranging environments. 4h context protects.",
                "economic_mechanism": f"1h breakout confirmed by 4h trend using {s['type']} sizing.",
                "family": "multi_timeframe",
                "name": f"GEN2_MTF_{h_id:03d}",
                "direction": {
                    "family": "breakout",
                    "indicator": "donchian",
                    "params": {"window": w}
                },
                "multi_timeframe": {"type": "trend"},
                "sizing": s
            })

    # 5. Momentum Evolution
    mom_windows = [14, 24]
    for w in mom_windows:
        for s in sizings:
            h_id += 1
            hypotheses.append({
                "generation_number": 2,
                "parent_failure": "LOW_PROFIT_FACTOR",
                "research_insight": "Raw momentum decays. Acceleration is structurally distinct.",
                "economic_mechanism": f"Momentum acceleration (w={w}) scaled by {s['type']}.",
                "family": "momentum",
                "name": f"GEN2_MOMACC_{h_id:03d}",
                "direction": {
                    "family": "momentum",
                    "indicator": "acceleration",
                    "params": {"window": w}
                },
                "sizing": s
            })

    return hypotheses

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    loop = ResearchLoop("BTC/USDT", "1h")
    loop.start_generation(dataset_id="btc_usd_1h_gen2", gen_number=2, desc="Generation 2: Evidence-Driven Evolution")
    
    hypos = generate_hypotheses()
    logger.info(f"Generated {len(hypos)} hypotheses for Gen 2.")
    
    survivors = 0
    for h in hypos:
        passed, reason = loop.test_hypothesis(h)
        if passed:
            survivors += 1
            
    logger.info("="*50)
    logger.info("GENERATION 2 REPORT SUMMARY")
    logger.info("="*50)
    logger.info(f"Total evaluated: {len(hypos)}")
    logger.info(f"Total survivors: {survivors}")
    logger.info("="*50)
