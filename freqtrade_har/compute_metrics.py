import sys
import json
import numpy as np
from scipy import stats
import glob
import os

def analyze_trades(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Freqtrade export format usually has a top level key like "strategy_name"
    strategy_key = list(data['strategy'].keys())[0]
    trades = data['strategy'][strategy_key]['trades']
    
    profits = [t['profit_ratio'] for t in trades]
    if len(profits) < 2:
        return len(profits), np.nan, np.nan, np.nan
        
    n = len(profits)
    mean_profit = np.mean(profits)
    std_profit = np.std(profits, ddof=1)
    
    # 1-sample t-test against 0
    t_stat, p_val_2tailed = stats.ttest_1samp(profits, 0)
    
    # 1-tailed p-value for mean > 0
    if t_stat > 0:
        p_val_1tailed = p_val_2tailed / 2
    else:
        p_val_1tailed = 1 - (p_val_2tailed / 2)
        
    return n, mean_profit, std_profit, p_val_1tailed

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "freqtrade_har/user_data/backtest_results"
        
    # Find latest json file that ends with .json but not .meta.json
    files = [f for f in glob.glob(f"{directory}/*.json") if not f.endswith('.meta.json')]
    for file in sorted(files, key=os.path.getmtime, reverse=True):
        print(f"Analyzing {file}:")
        n, mean, std, pval = analyze_trades(file)
        print(f"Trades: {n}")
        print(f"Mean profit: {mean*100:.4f}%")
        print(f"Std dev: {std*100:.4f}%")
        print(f"p-value: {pval:.4f}")
        print("-" * 30)
