import logging
import os
import sys
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath("algo-research-lab"))
from agent.research_loop import ResearchLoop

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

baselines = [
    {
        "name": "DIAG_EMA_10_50", "family": "trend",
        "direction": {"family": "trend", "indicator": "sma_crossover", "params": {"fast": 10, "slow": 50}},
        "sizing": {"type": "default"}
    },
    {
        "name": "DIAG_EMA_20_100", "family": "trend",
        "direction": {"family": "trend", "indicator": "sma_crossover", "params": {"fast": 20, "slow": 100}},
        "sizing": {"type": "default"}
    },
    {
        "name": "DIAG_EMA_50_200", "family": "trend",
        "direction": {"family": "trend", "indicator": "sma_crossover", "params": {"fast": 50, "slow": 200}},
        "sizing": {"type": "default"}
    },
    {
        "name": "DIAG_RSI_14_30_70", "family": "mean_reversion",
        "direction": {"family": "mean_reversion", "indicator": "rsi", "params": {"window": 14, "lower": 30, "upper": 70}},
        "sizing": {"type": "default"}
    },
    {
        "name": "DIAG_RSI_7_20_80", "family": "mean_reversion",
        "direction": {"family": "mean_reversion", "indicator": "rsi", "params": {"window": 7, "lower": 20, "upper": 80}},
        "sizing": {"type": "default"}
    },
    {
        "name": "DIAG_RSI_21_40_60", "family": "mean_reversion",
        "direction": {"family": "mean_reversion", "indicator": "rsi", "params": {"window": 21, "lower": 40, "upper": 60}},
        "sizing": {"type": "default"}
    },
    {
        "name": "DIAG_DONCHIAN_20", "family": "breakout",
        "direction": {"family": "breakout", "indicator": "donchian", "params": {"window": 20}},
        "sizing": {"type": "default"}
    },
    {
        "name": "DIAG_DONCHIAN_48", "family": "breakout",
        "direction": {"family": "breakout", "indicator": "donchian", "params": {"window": 48}},
        "sizing": {"type": "default"}
    },
    {
        "name": "DIAG_DONCHIAN_96", "family": "breakout",
        "direction": {"family": "breakout", "indicator": "donchian", "params": {"window": 96}},
        "sizing": {"type": "default"}
    },
    {
        "name": "DIAG_MOMENTUM_ROC_14", "family": "momentum",
        "direction": {"family": "momentum", "indicator": "roc", "params": {"window": 14, "threshold": 0}},
        "sizing": {"type": "default"}
    },
    {
        "name": "DIAG_MOMENTUM_ROC_48", "family": "momentum",
        "direction": {"family": "momentum", "indicator": "roc", "params": {"window": 48, "threshold": 2.0}},
        "sizing": {"type": "default"}
    }
]

def main():
    loop = ResearchLoop("BTC/USDT", "1h")
    logger.info("Starting Benchmark Suite...")
    loop.start_generation()
    
    for b in baselines:
        logger.info(f"Running Baseline {b['name']}...")
        loop.test_hypothesis(b, diagnostic_mode=True)
        
    logger.info("Diagnostic run complete.")

if __name__ == "__main__":
    main()
