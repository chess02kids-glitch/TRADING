import os
import re
import json
import glob
import zipfile
import numpy as np
from scipy import stats

def parse_log(logfile):
    if not os.path.exists(logfile):
        return None, None, None, 0
    with open(logfile, 'r', encoding='latin-1') as f:
        content = f.read()
    
    total_profit = re.search(r'\|\s*Total profit %\s*\|\s*([^%]+)%', content)
    sharpe = re.search(r'\|\s*Sharpe \(closed trades\)\s*\|\s*([\-\.\d]+)', content)
    max_dd = re.search(r'\|\s*Max % of account underwater\s*\|\s*([^%]+)%', content)
    trades_m = re.search(r'\|\s*Total/Daily Avg Trades\s*\|\s*(\d+)', content)
    
    tp = float(total_profit.group(1)) if total_profit else None
    sh = float(sharpe.group(1)) if sharpe else None
    md = float(max_dd.group(1)) if max_dd else None
    tr = int(trades_m.group(1)) if trades_m else 0
    
    return tp, sh, md, tr

def get_p_value_from_zip(strategy_name):
    zips = glob.glob('freqtrade_har/user_data/backtest_results/*.zip')
    # Match the zip that contains the strategy name in its python file or we can just find it
    # We will search all zips for a JSON file that matches the strategy name
    best_p = np.nan
    
    for zf in sorted(zips, key=os.path.getmtime, reverse=True):
        with zipfile.ZipFile(zf, 'r') as z:
            json_files = [f for f in z.namelist() if f.endswith('.json') and not f.endswith('config.json')]
            for jf in json_files:
                data = json.loads(z.read(jf).decode('utf-8'))
                if 'strategy' in data and strategy_name in data['strategy']:
                    trades = data['strategy'][strategy_name]['trades']
                    profits = [t['profit_ratio'] for t in trades]
                    if len(profits) < 2:
                        continue
                    mean_p = np.mean(profits)
                    t_stat, p_val_2tailed = stats.ttest_1samp(profits, 0)
                    if t_stat > 0:
                        p_val = p_val_2tailed / 2
                    else:
                        p_val = 1 - (p_val_2tailed / 2)
                    return p_val
    return best_p

def main():
    log_files = {
        'HARStopBaseline': 'freqtrade_har/baseline.log',
        'HARStopDynamic': 'freqtrade_har/dynamic.log',
        'HARStopInverse': 'freqtrade_har/inverse.log',
        'HARStopDynamic (P1)': 'freqtrade_har/dynamic_p1.log',
        'HARStopDynamic (P2)': 'freqtrade_har/dynamic_p2.log',
        'HARStopDynamic (P3)': 'freqtrade_har/dynamic_p3.log'
    }
    
    for strategy_name, logf in log_files.items():
        tp, sh, md, tr = parse_log(logf)
        strat_key = strategy_name.split(" ")[0]
        pval = get_p_value_from_zip(strat_key)
        
        print(f"[{strategy_name}]")
        print(f"Trades: {tr}")
        print(f"Total Profit: {tp}%")
        print(f"Sharpe: {sh}")
        print(f"Max DD: {md}%")
        print(f"P-value: {pval:.4f}" if not np.isnan(pval) else "P-value: NaN")

if __name__ == "__main__":
    main()
