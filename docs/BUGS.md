# Kronos Upstream Bugs - Audit Documentation

This document tracks known bugs in upstream Kronos repository https://github.com/shiyu-coder/Kronos
Principle: Keep upstream `Kronos/` folder untouched. Patches applied outside.

---

## Bug #1: Top-k called as function (Issue #231)

- **File:** `Kronos/model/kronos.py`
- **Function:** `sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True)`
- **Exact Location:** Line ~379-385 in version before fix, line 382 in fixed version
- **Old Buggy Code:**
  ```python
  if not sample_logits:
      _, x = top_k(probs, k=1, dim=-1)  # BUG: top_k is int param, not function
  ```
- **Error:** `TypeError: 'int' object is not callable`
- **Fixed Code (commit fde8f60):**
  ```python
  if not sample_logits:
      _, x = torch.topk(probs, k=1, dim=-1)  # FIXED: torch.topk is correct API
  ```
- **Upstream Issue:** https://github.com/shiyu-coder/Kronos/issues/231
- **Fix Commit:** `fde8f60a0d398395e848781ace62914d58315df6` - "fix: use torch.topk instead of calling top_k parameter as function"
- **Date:** Apr 9 2026
- **Author:** cocoon
- **PR:** #232
- **Merge:** #22140cd merges #232

- **Root Cause Analysis:**
  - Function signature has parameter `top_k: int = None` which shadows any global function name `top_k`
  - Inside function, when `sample_logits=False`, code intended greedy argmax (top-1)
  - Developer mistakenly called `top_k(probs, k=1, dim=-1)` as if `top_k` were torch.topk function
  - But `top_k` is integer (e.g., 50) from parameter, hence TypeError
  - Bug hidden because most usage has `sample_logits=True` (sampling path uses `torch.multinomial`), only deterministic path triggers bug

- **Reproduction:**
  ```python
  from model import Kronos, KronosTokenizer, KronosPredictor
  # ... load model ...
  predictor.predict(..., sample_count=1) # with internal call that sets sample_logits=False
  # Or directly:
  import torch
  from model.kronos import sample_from_logits
  logits = torch.randn(1, 10)
  sample_from_logits(logits, top_k=50, sample_logits=False) # Should raise TypeError in buggy version
  ```

- **External Patch (No upstream modification):**
  - File: `scripts/prediction/kronos_compatibility.py`
  - At runtime, inspect source of `sample_from_logits`
  - If buggy pattern detected, monkey-patch function in memory to use `torch.topk`
  - Upstream file `Kronos/model/kronos.py` remains untouched on disk
  - Verification: `python scripts/setup/bug_audit.py` checks pattern without editing

- **Status in current clone (commit 67b630e):** FIXED - file contains `torch.topk(probs, k=1, dim=-1)` at line 382

- **Impact for our trading system:**
  - Low for Phase 1 inference if using `sample_logits=True` (default)
  - Medium if we later want deterministic predictions for backtesting
  - Runtime wrapper ensures safety even if user clones old version

---

## Bug #2: Batch dimension loss when batch_size=1 (PR #243)

- **Files:**
  - `finetune/train_tokenizer.py` Lines 126, 177 (old)
  - `finetune/train_predictor.py` Lines 95, 138 (old)
  - `finetune_csv/finetune_tokenizer.py` Lines 193, 242 (old)

- **Old Buggy Code:**
  ```python
  for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
      batch_x = batch_x.squeeze(0).to(device, non_blocking=True)  # BUG: removes batch dim when batch=1
      batch_x_stamp = batch_x_stamp.squeeze(0).to(device, non_blocking=True)
  ```

- **Fixed Code (commit 8ca2821):**
  ```python
  for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
      batch_x = batch_x.to(device, non_blocking=True)  # FIXED: preserve batch dim
      batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
  ```

- **Fix Commit:** `8ca282123c1cd3caa29d6e0384504deab4d508c0` - "fix: preserve batch dimension in tokenizer and predictor training"
- **Date:** Apr 13 2026
- **Author:** Elhamullah Hossaini
- **PR:** #243 from `ElhamDevelopmentStudio/fix/batch-dimension-training`
- **Merge Commit:** `67b630e`

- **Root Cause Analysis:**
  - PyTorch DataLoader collate returns batch as (batch_size, seq_len, features) even when batch_size=1
  - Some code assumed DataLoader returned extra dim (1, batch_size, seq_len, features) needing squeeze(0)
  - Using `squeeze(0)` removes first dimension, so tensor shape changes:
    - Input: (1, 400, 6) with batch=1
    - After squeeze(0): (400, 6) - batch dimension lost!
  - Later code expects 3D tensor, causes shape mismatch or silent incorrect broadcasting during training
  - Training may appear to work but gradients wrong, or explicit error `Expected 3D but got 2D`

- **Reproduction:**
  - Train tokenizer with batch_size=1 config
  - Observe loss not decreasing or shape errors in model forward

- **External Patch:**
  - This bug only affects finetune scripts (Phase 9), not inference (Phase 3)
  - Our Phase 9 implementation in `scripts/finetune/` will NOT use `squeeze(0)` - we preserve batch dim
  - Keep upstream `Kronos/finetune/` untouched
  - Audit script `bug_audit.py` checks for pattern without editing

- **Status in current clone (67b630e):** FIXED - no `squeeze(0).to(device` pattern remains in those files

- **Impact for trading system:**
  - None for Phase 1-8 (inference, backtesting, paper trading)
  - Relevant only when fine-tuning Kronos on BTC/ETH data (Phase 9)
  - Our custom finetune loop will avoid bug by design

---

## Other Fixes Documented

### WebUI numpy pin (Mar 9 2026)

- **Commit:** `8c08af6` "fix: relax numpy requirement in webui dependencies"
- **File:** `webui/requirements.txt`
- **Old:** `numpy==1.26.4` strict pin conflicts with newer environments
- **Fixed:** Relaxed to `numpy>=1.19.0` or similar
- **Impact:** Not critical for our trading system (we don't use webui), but our pinned 1.26.4 is compatible

### Device auto-detect (Dec 20 2025)

- **Commit:** `369bc0a` "Auto-detect device for easier getting started"
- **Change:** Predictor auto-detects cuda/cpu instead of hardcoded "cuda:0"
- **Impact:** Good for RTX 3060 + CPU fallback - our compatibility layer also respects this

---

## Patching Policy (Per Audit #6)

1. **Never modify `Kronos/` folder on disk** - keep upstream pristine for reproducibility
2. **Identify exact file/line and upstream issue/commit** - documented above
3. **Reproduce bug** - provide reproduction snippet
4. **Explain root cause** - detailed analysis
5. **Implement external patch** - runtime monkey-patch or own implementation
6. **Document** - in this file + logs

## Verification

Run audit that does NOT modify files:

```bash
python scripts/setup/bug_audit.py
python scripts/prediction/kronos_compatibility.py
```

Expected output: Both bugs FIXED in current clone at commit 67b630e, but compatibility wrapper ready if old clone used.

## References

- Kronos repo: https://github.com/shiyu-coder/Kronos
- Current clone commit: `67b630e67f6a18c9e9be918d9b4337c960db1e9a` (latest as of Aug 10 2026)
- Issues: #231, #243
- Commits: fde8f60, 8ca2821, 67b630e
