#!/usr/bin/env python3
"""
PHASE 1 & 3 - Audit #6: Kronos Compatibility Layer (NO UPSTREAM MODIFICATION)
Keeps Kronos/ folder untouched.
If bug is detected, applies patch at runtime in memory via monkey-patching.

Documents exact file/line, upstream issue, root cause, and external patch.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KRONOS_DIR = PROJECT_ROOT / "Kronos"

def check_and_patch_topk_bug():
    """
    Runtime check for Bug #231 - top_k called as function
    
    Location: Kronos/model/kronos.py, sample_from_logits()
    Bug: top_k(probs, k=1, dim=-1) where top_k is int param -> TypeError
    Fix Commit: fde8f60a0d398395e848781ace62914d58315df6
    Issue: #231
    
    External Patch: Monkey-patch function at runtime if bug present, no file edit
    """
    # Ensure Kronos is importable
    sys.path.insert(0, str(KRONOS_DIR))
    sys.path.insert(0, str(KRONOS_DIR / "model"))
    
    try:
        import model.kronos as kronos_module
        
        # Inspect source of sample_from_logits
        import inspect
        source = inspect.getsource(kronos_module.sample_from_logits)
        
        # Check if buggy pattern present in source code runtime
        if "top_k(probs, k=1" in source and "torch.topk" not in source:
            logger.warning("⚠ Bug #231 detected in loaded module: top_k called as function")
            logger.info("Applying runtime monkey-patch (no file modification)")
            
            # Define fixed version
            import torch
            import torch.nn.functional as F
            
            def fixed_sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
                """Fixed version - uses torch.topk instead of top_k()"""
                logits = logits / temperature
                if top_k is not None or top_p is not None:
                    if top_k > 0 or top_p < 1.0:
                        logits = kronos_module.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                
                probs = F.softmax(logits, dim=-1)
                
                if not sample_logits:
                    # FIXED: was top_k(probs, k=1, dim=-1) -> TypeError
                    _, x = torch.topk(probs, k=1, dim=-1)
                else:
                    x = torch.multinomial(probs, num_samples=1)
                
                return x
            
            # Monkey-patch the function in module
            kronos_module.sample_from_logits = fixed_sample_from_logits
            logger.info("✅ Runtime patch applied - sample_from_logits now uses torch.topk")
            logger.info("   Upstream file Kronos/model/kronos.py NOT modified - patch is in memory only")
            return True, "patched_runtime"
        else:
            logger.info("✅ Bug #231 not present in loaded module - already fixed upstream")
            return True, "already_fixed"
            
    except Exception as e:
        logger.warning(f"Could not check/patch top_k bug: {e}")
        import traceback
        traceback.print_exc()
        return False, f"error: {e}"

def ensure_kronos_importable():
    """Ensure Kronos repo is importable without modifying it"""
    sys.path.insert(0, str(KRONOS_DIR))
    sys.path.insert(0, str(KRONOS_DIR / "model"))
    
    try:
        # Try import
        from model import Kronos, KronosTokenizer, KronosPredictor
        logger.info("✅ Kronos imports successful - upstream repo usable")
        return True
    except ImportError as e:
        logger.error(f"❌ Kronos import failed: {e}")
        logger.info(f"Check: Kronos dir exists at {KRONOS_DIR}? {KRONOS_DIR.exists()}")
        logger.info("Fix: git clone https://github.com/shiyu-coder/Kronos.git")
        return False

def get_model_info_without_loading():
    """Document model info without claiming unverified VRAM"""
    info = {
        "tokenizer_base": "NeoQuasar/Kronos-Tokenizer-base",
        "model_small": "NeoQuasar/Kronos-small",
        "params_small": "24.7M (documented by upstream, not measured yet - will be measured in verify_install)",
        "context_small": 512,
        "note": "Actual param count, VRAM, dtype measured at runtime in verify_install.py"
    }
    return info

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    
    print("="*70)
    print("KRONOS COMPATIBILITY LAYER - External Patches Only")
    print("="*70)
    print(f"Kronos dir: {KRONOS_DIR}")
    print(f"Exists: {KRONOS_DIR.exists()}")
    
    print("\n[1/2] Checking importability...")
    ok1 = ensure_kronos_importable()
    
    print("\n[2/2] Checking Bug #231 top_k runtime patch...")
    ok2, status = check_and_patch_topk_bug()
    print(f"Result: {status}")
    
    print("\n" + "="*70)
    print("No files in Kronos/ were modified - all patches are in-memory")
    print("="*70)
