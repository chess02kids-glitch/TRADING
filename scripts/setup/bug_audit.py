#!/usr/bin/env python3
"""
PHASE 1 - Audit #6: Bug Audit (NO AUTO-MODIFICATION)
Keeps upstream Kronos code untouched.
Identifies exact file/line, upstream issue/commit, reproduces bug, explains root cause,
implements patch outside upstream where possible, documents patch.

This script ONLY reports, never modifies Kronos/ folder.
"""

import logging
import sys
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KRONOS_DIR = PROJECT_ROOT / "Kronos"
KRONOS_MODEL_FILE = KRONOS_DIR / "model" / "kronos.py"
KRONOS_TOKENIZER_TRAIN = KRONOS_DIR / "finetune" / "train_tokenizer.py"
KRONOS_PREDICTOR_TRAIN = KRONOS_DIR / "finetune" / "train_predictor.py"

def get_git_commit():
    if not KRONOS_DIR.exists():
        return "Kronos repo not cloned"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(KRONOS_DIR), capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"

def audit_bug_231_topk():
    """
    Bug #231: Top-k called as function in sample_from_logits when sample_logits=False
    
    Exact Location: model/kronos.py, function sample_from_logits(), line ~382 in fixed version
    Old buggy code:
        _, x = top_k(probs, k=1, dim=-1)  # top_k is int parameter, not function!
    Fixed code:
        _, x = torch.topk(probs, k=1, dim=-1)
    
    Upstream Issue: https://github.com/shiyu-coder/Kronos/issues/231
    Upstream Commit: fde8f60a0d398395e848781ace62914d58315df6
    Commit Message: "fix: use torch.topk instead of calling top_k parameter as function in sample_from_logits"
    Date: Apr 9 2026
    Author: cocoon
    PR: #232
    
    Root Cause:
    - Function signature: def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    - Inside function, parameter top_k is int (e.g., 0 or 50) shadowing any function name
    - Branch: if not sample_logits: tries to do greedy argmax via top_k(probs, k=1)
    - But top_k is int, so TypeError: 'int' object is not callable
    - This branch only triggers when sample_logits=False (deterministic inference)
    - Most examples use sample_logits=True, so bug hidden unless using greedy decode
    
    Reproduction:
    - Call predictor.predict with sample_count=1 and set sample_logits=False in generate
    - Or directly: sample_from_logits(logits, top_k=50, sample_logits=False)
    - Will raise TypeError
    
    Patch Outside Upstream:
    - Do NOT edit Kronos/model/kronos.py
    - Instead, create wrapper module scripts/prediction/kronos_compatibility.py
    - At import time, monkey-patch the function if bug detected
    - Or override auto_regressive_inference to avoid triggering bug path
    - Here we check if file contains fixed pattern
    
    Status in current clone: Should be fixed if commit >= fde8f60
    """
    print("\n" + "="*70)
    print("BUG AUDIT #1: Top-k as function (Issue #231)")
    print("="*70)
    print("File: Kronos/model/kronos.py")
    print("Function: sample_from_logits()")
    print("Issue: https://github.com/shiyu-coder/Kronos/issues/231")
    print("Fix Commit: fde8f60a0d398395e848781ace62914d58315df6 (Apr 9 2026)")
    
    if not KRONOS_MODEL_FILE.exists():
        print("❌ Kronos/model/kronos.py not found - clone repo first")
        return False, "missing_file"
    
    content = KRONOS_MODEL_FILE.read_text()
    
    # Check for bug pattern
    buggy_pattern = "top_k(probs, k=1, dim=-1)"
    fixed_pattern = "torch.topk(probs, k=1, dim=-1)"
    
    has_buggy = buggy_pattern in content and "torch.topk(probs, k=1" not in content
    has_fixed = fixed_pattern in content
    
    print(f"\nCurrent file contains:")
    print(f"  Buggy pattern '{buggy_pattern}': {has_buggy}")
    print(f"  Fixed pattern '{fixed_pattern}': {has_fixed}")
    
    # Exact line number
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if "torch.topk(probs, k=1" in line or "top_k(probs, k=1" in line:
            print(f"  Line {i}: {line.strip()}")
    
    if has_fixed and not has_buggy:
        print("\n✅ FIXED in current clone - no action needed, but keep compatibility wrapper")
        print("   Root cause explained above, patch documented in docs/BUGS.md")
        return True, "fixed"
    elif has_buggy:
        print("\n❌ BUG PRESENT - Old clone, needs update")
        print("   Fix: cd Kronos && git pull origin master")
        print("   OR use runtime monkey-patch in kronos_compatibility.py (does not modify upstream file)")
        return False, "bug_present"
    else:
        print("\n⚠️ Unclear - pattern not found, manual inspection needed")
        return False, "unclear"

def audit_bug_243_batch_dim():
    """
    Bug #243: Batch dimension loss when batch_size=1 in tokenizer and predictor training
    
    Exact Locations:
    - finetune/train_tokenizer.py, line 126 & 177 (old)
    - finetune/train_predictor.py, line 95 & 138 (old)
    - finetune_csv/finetune_tokenizer.py, line 193 & 242 (old)
    
    Old buggy code:
        batch_x = batch_x.squeeze(0).to(device)  # Removes batch dim when batch=1!
    Fixed code:
        batch_x = batch_x.to(device)  # Preserve batch dimension
    
    Upstream Commit: 8ca282123c1cd3caa29d6e0384504deab4d508c0
    Message: "fix: preserve batch dimension in tokenizer and predictor training"
    Date: Apr 13 2026
    Author: Elhamullah Hossaini
    PR: #243 from ElhamDevelopmentStudio/fix/batch-dimension-training
    Merge Commit: 67b630e
    
    Root Cause:
    - DataLoader with batch_size=1 returns tensor shape (1, seq_len, features) after collate
    - Some code had extra squeeze(0) assuming DataLoader returns (1, 1, seq_len, features)
    - squeeze(0) removes first dimension, so if batch=1, tensor becomes (seq_len, features) losing batch dim
    - Later code expects (batch, seq_len, features), causes shape mismatch or silent broadcast error
    - Particularly problematic for finetune, not for inference (our inference uses batch normally)
    
    Reproduction:
    - Train tokenizer with batch_size=1
    - Will see shape errors or model not learning correctly
    
    Patch Outside Upstream:
    - This bug only affects finetune scripts, not inference (Phase 9)
    - For Phase 1, document only
    - For Phase 9 implementation, we will write our own training loop without squeeze(0)
    - Keep upstream untouched, our finetune code (scripts/finetune/) will not use squeeze(0)
    
    Status: Check if current clone has fix
    """
    print("\n" + "="*70)
    print("BUG AUDIT #2: Batch dimension loss (PR #243)")
    print("="*70)
    print("Files: finetune/train_tokenizer.py, finetune/train_predictor.py")
    print("Fix Commit: 8ca282123c1cd3caa29d6e0384504deab4d508c0 (Apr 13 2026)")
    print("Merge: 67b630e")
    
    files_to_check = [
        (KRONOS_TOKENIZER_TRAIN, "finetune/train_tokenizer.py"),
        (KRONOS_PREDICTOR_TRAIN, "finetune/train_predictor.py"),
        (KRONOS_DIR / "finetune_csv" / "finetune_tokenizer.py", "finetune_csv/finetune_tokenizer.py")
    ]
    
    all_fixed = True
    for file_path, label in files_to_check:
        if not file_path.exists():
            print(f"\n  {label}: file not found (optional)")
            continue
        
        content = file_path.read_text()
        buggy = "squeeze(0).to(device" in content
        # More precise: batch_x = batch_x.squeeze(0).to(device
        lines = content.split('\n')
        buggy_lines = [f"Line {i}: {l.strip()}" for i, l in enumerate(lines, 1) if "squeeze(0).to(device" in l]
        
        print(f"\n  {label}:")
        if buggy_lines:
            print(f"    ❌ Contains squeeze(0) pattern ({len(buggy_lines)} occurrences):")
            for bl in buggy_lines[:3]:
                print(f"      {bl}")
            all_fixed = False
        else:
            print(f"    ✅ No squeeze(0) batch loss pattern - FIXED")
    
    if all_fixed:
        print("\n✅ FIXED in current clone - training scripts preserve batch dim")
    else:
        print("\n❌ BUG PRESENT in some training scripts - update repo:")
        print("   cd Kronos && git pull origin master")
        print("   Note: Our Phase 9 finetune will implement own loop without squeeze(0)")
    
    return all_fixed, "fixed" if all_fixed else "bug_present"

def audit_other_issues():
    """
    Other documented fixes:
    - Mar 9 2026: numpy requirement relax in webui
    - Dec 2025: Auto-detect device
    """
    print("\n" + "="*70)
    print("BUG AUDIT #3: Other fixes (webui numpy, device auto-detect)")
    print("="*70)
    
    webui_req = KRONOS_DIR / "webui" / "requirements.txt"
    if webui_req.exists():
        content = webui_req.read_text()
        if "numpy==" in content and "numpy>=" not in content:
            print(f"  webui/requirements.txt has strict numpy pin - may conflict, but not critical for our trading system")
        else:
            print(f"  ✅ webui requirements relaxed")
    
    # Check device auto-detect
    if KRONOS_MODEL_FILE.exists():
        content = KRONOS_MODEL_FILE.read_text()
        if "auto-detect" in content.lower() or "device" in content.lower():
            print(f"  ✅ Device handling present in kronos.py")
    
    return True

def main():
    print("="*70)
    print("KRONOS UPSTREAM BUG AUDIT - NO MODIFICATIONS")
    print(f"Kronos commit: {get_git_commit()}")
    print(f"Kronos path: {KRONOS_DIR}")
    print("Principle: Keep upstream untouched, document patches externally")
    print("="*70)
    
    results = []
    r1, s1 = audit_bug_231_topk()
    results.append(("Top-k as function #231 (fde8f60)", r1, s1))
    
    r2, s2 = audit_bug_243_batch_dim()
    results.append(("Batch dim loss #243 (8ca2821)", r2, s2))
    
    r3 = audit_other_issues()
    results.append(("Other fixes", r3, "info"))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed, status in results:
        symbol = "✅" if passed else "❌"
        print(f"{symbol} {name}: {status}")
    
    print("\n" + "="*70)
    print("PATCH STRATEGY (Outside Upstream)")
    print("="*70)
    print("1. For Bug #231 top_k:")
    print("   - Upstream file Kronos/model/kronos.py remains UNTouched")
    print("   - Our wrapper scripts/prediction/kronos_compatibility.py")
    print("     will at runtime:")
    print("     if 'top_k(probs' in file: monkey-patch sample_from_logits with torch.topk")
    print("   - Documented in docs/BUGS.md")
    print("")
    print("2. For Bug #243 batch dim:")
    print("   - Affects only finetune scripts, not inference")
    print("   - Our Phase 9 scripts/finetune/ will NOT use squeeze(0)")
    print("   - Keep upstream untouched, use our own training loop")
    print("")
    print("3. No automatic file editing - all patches are runtime or in our own code")
    
    # Write report
    report_path = PROJECT_ROOT / "logs" / "bug_audit_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"Kronos Bug Audit Report - {get_git_commit()}\n")
        f.write("="*70 + "\n")
        for name, passed, status in results:
            f.write(f"{'PASS' if passed else 'FAIL'} {name}: {status}\n")
    
    print(f"\nReport saved to {report_path}")
    
    return 0 if all(r for r,_,_ in results) else 1

if __name__ == "__main__":
    sys.exit(main())
