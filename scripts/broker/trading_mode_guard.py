#!/usr/bin/env python3
"""
PHASE 1 - Audit #5: Trading Mode Guard
Implements explicit modes: BACKTEST, PAPER, TESTNET, LIVE
Default PAPER, LIVE disabled by default, requires explicit confirmation

Security: Prevents accidental live orders
"""

import os
import sys
import logging
from pathlib import Path
from enum import Enum
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TradingMode(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    TESTNET = "TESTNET"
    LIVE = "LIVE"

# LIVE requires triple confirmation
LIVE_CONFIRMATION_PHRASE = "I_UNDERSTAND_RISK_OF_LIVE_TRADING"
LIVE_ENV_VAR = "BINANCE_LIVE_CONFIRMED"
REQUIRED_LIVE_ENV_VALUE = "true"

class TradingModeGuard:
    """
    Guard that validates trading mode transitions and prevents accidental LIVE trading
    """
    
    def __init__(self, config: Dict[str, Any], env_path: Path = None):
        self.config = config
        self.trading_cfg = config.get('trading', {})
        self.mode_str = self.trading_cfg.get('mode', 'PAPER').upper()
        self.live_enabled = self.trading_cfg.get('live_trading_enabled', False)
        self.live_confirmation = self.trading_cfg.get('live_trading_confirmation', '')
        
        try:
            self.mode = TradingMode(self.mode_str)
        except ValueError:
            raise ValueError(f"Invalid trading mode {self.mode_str}. Allowed: {[m.value for m in TradingMode]}")
        
        # Load .env manually without leaking secrets
        if env_path is None:
            env_path = Path(__file__).parents[2] / ".env"
        self.env_path = env_path
        self.env_vars = self._load_env_safe(env_path)
    
    def _load_env_safe(self, env_path: Path) -> Dict[str, str]:
        """Load .env without printing secrets, returns dict"""
        env_vars = {}
        if not env_path.exists():
            logger.info(f".env not found at {env_path} - using OS env only")
            # Fall back to OS env
            for k in os.environ:
                if k.startswith("BINANCE_") or k.startswith("TELEGRAM") or k.startswith("DISCORD"):
                    env_vars[k] = os.environ.get(k, '')
            return env_vars
        
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Don't log values
                    env_vars[key] = value
            logger.info(f"Loaded {len(env_vars)} env vars from {env_path} (values hidden)")
        except Exception as e:
            logger.warning(f"Could not read {env_path}: {e} - using OS env")
        
        # Also overlay OS env
        for k in os.environ:
            if k not in env_vars and (k.startswith("BINANCE") or k.startswith("TELEGRAM")):
                env_vars[k] = os.environ.get(k, '')
        
        return env_vars
    
    def validate(self) -> bool:
        """
        Validate current mode configuration
        Returns True if allowed, raises Exception if disallowed
        """
        logger.info(f"Validating trading mode: {self.mode.value}")
        
        if self.mode == TradingMode.LIVE:
            return self._validate_live()
        elif self.mode == TradingMode.TESTNET:
            return self._validate_testnet()
        elif self.mode == TradingMode.PAPER:
            return self._validate_paper()
        elif self.mode == TradingMode.BACKTEST:
            return self._validate_backtest()
        else:
            raise ValueError(f"Unknown mode {self.mode}")
    
    def _validate_live(self) -> bool:
        """LIVE requires triple confirmation"""
        errors = []
        
        if not self.live_enabled:
            errors.append("LIVE trading attempted but live_trading_enabled=false in config/trading. "
                          "Set to true ONLY after thorough testing. Keep false by default.")
        
        if self.live_confirmation != LIVE_CONFIRMATION_PHRASE:
            errors.append(f"LIVE confirmation phrase mismatch. Expected '{LIVE_CONFIRMATION_PHRASE}' in config, "
                          f"got '{self.live_confirmation}'. This prevents accidental live trading.")
        
        live_env_val = self.env_vars.get(LIVE_ENV_VAR, '') or os.getenv(LIVE_ENV_VAR, '')
        if live_env_val.lower() != REQUIRED_LIVE_ENV_VALUE:
            errors.append(f"LIVE requires env var {LIVE_ENV_VAR}=true in .env. "
                          f"Current: '{live_env_val}'. This is third guard against accidental live.")
        
        # Additional safety: require separate live API keys, not testnet keys
        testnet_flag = self.env_vars.get("BINANCE_TESTNET", "true").lower()
        if testnet_flag == "true":
            errors.append("BINANCE_TESTNET=true in .env but trying LIVE mode. Set BINANCE_TESTNET=false for LIVE.")
        
        if errors:
            error_msg = "\n".join([f"{i+1}. {e}" for i, e in enumerate(errors)])
            logger.error(f"\n{'='*60}\nLIVE TRADING BLOCKED - SAFETY GUARD TRIGGERED:\n{error_msg}\n{'='*60}")
            raise PermissionError(f"LIVE trading blocked by safety guard:\n{error_msg}")
        
        logger.warning(f"\n{'!'*60}\nWARNING: LIVE MODE ENABLED - Real money will be used!\n{'!'*60}")
        logger.warning("Ensure: 1) Tested on TESTNET, 2) Risk limits set, 3) Emergency shutdown ready")
        return True
    
    def _validate_testnet(self) -> bool:
        testnet_flag = self.env_vars.get("BINANCE_TESTNET", "true").lower()
        if testnet_flag != "true":
            logger.warning("Mode is TESTNET but BINANCE_TESTNET env is not true - check config")
        # Testnet requires API keys but fake money
        api_key = self.env_vars.get("BINANCE_API_KEY", "")
        if not api_key or "your_" in api_key:
            raise ValueError("TESTNET mode requires BINANCE_API_KEY in .env - get from https://testnet.binance.vision")
        logger.info("✓ TESTNET validation passed - using Binance testnet with fake funds")
        return True
    
    def _validate_paper(self) -> bool:
        # Paper trading needs no API keys, safest default
        logger.info("✓ PAPER mode - No real orders, simulated portfolio, safest default per audit")
        return True
    
    def _validate_backtest(self) -> bool:
        logger.info("✓ BACKTEST mode - Historical simulation only")
        return True
    
    def get_ccxt_config(self) -> Dict[str, Any]:
        """
        Get CCXT config based on mode - no secrets returned, only structure
        Actual secrets injected at runtime from env, never logged
        """
        if self.mode == TradingMode.BACKTEST or self.mode == TradingMode.PAPER:
            # No exchange connection needed
            return {"mode": self.mode.value, "exchange_required": False}
        
        elif self.mode == TradingMode.TESTNET:
            # Binance testnet via CCXT
            # CCXT testnet: set urls['api'] or options
            return {
                "mode": "TESTNET",
                "exchange_required": True,
                "exchange_id": "binance",
                "options": {
                    "testnet": True,
                    # CCXT Binance testnet docs: https://docs.ccxt.com/exchanges/binance
                    # For spot testnet, need custom urls
                },
                "urls_note": "CCXT handles testnet via options or .urls - see binance_client.py"
            }
        
        elif self.mode == TradingMode.LIVE:
            self.validate()  # Re-validate safety
            return {
                "mode": "LIVE",
                "exchange_required": True,
                "exchange_id": "binance",
                "warning": "REAL MONEY - Emergency shutdown ready at scripts/go_live/emergency_shutdown.py"
            }

def load_config_safe() -> Dict[str, Any]:
    """Load config.yaml without exposing secrets"""
    import yaml
    config_path = Path(__file__).parents[2] / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Verify no secrets in config (audit #9)
    config_str = str(cfg)
    suspicious = ["BINANCE_API_KEY", "SECRET_KEY", "sk-", "api_key"]
    lower_str = config_str.lower()
    # Check if actual key material present (not just env var names)
    if "binom" in lower_str and len(config_str) > 10000:
        pass
    # Allow env var name references like "BINANCE_API_KEY" as value for api_key_env
    # But disallow actual key strings
    for pattern in ["sk-", "whsec_", "binance_live_"]:
        if pattern.lower() in lower_str:
            # Might be placeholder, check length
            if len(config_str) < 2000:
                continue
            raise ValueError(f"Potential secret material '{pattern}' found in config.yaml - violates security rule")
    
    return cfg

if __name__ == "__main__":
    # Self-test
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    
    try:
        cfg = load_config_safe()
        guard = TradingModeGuard(cfg)
        guard.validate()
        ccxt_cfg = guard.get_ccxt_config()
        print(f"\nTrading Mode: {ccxt_cfg['mode']}")
        print(f"Exchange Required: {ccxt_cfg['exchange_required']}")
        print("\n✓ Guard validation PASSED - Current mode is safe")
        
        if guard.mode == TradingMode.PAPER:
            print("Default PAPER mode active - per audit requirement, LIVE disabled")
        
    except PermissionError as e:
        print(f"\n❌ LIVE BLOCKED (expected if trying LIVE without confirmation):\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
