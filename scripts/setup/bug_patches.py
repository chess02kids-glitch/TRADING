#!/usr/bin/env python3
"""
DEPRECATED - Per Audit #6: Do NOT automatically modify upstream Kronos repository
This file previously attempted to auto-patch Kronos/model/kronos.py on disk.
That violates audit requirement: keep upstream untouched.

New approach:
- Audit only (no modification): scripts/setup/bug_audit.py
- Runtime compatibility: scripts/prediction/kronos_compatibility.py
- Documentation: docs/BUGS.md

This file now only warns and redirects.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.warning("="*70)
    logger.warning("bug_patches.py is DEPRECATED per Audit #6")
    logger.warning("We do NOT modify upstream Kronos/ folder on disk")
    logger.warning("="*70)
    logger.info("")
    logger.info("Use instead:")
    logger.info("  1. Audit only (no modification): python scripts/setup/bug_audit.py")
    logger.info("  2. Runtime compatibility: python scripts/prediction/kronos_compatibility.py")
    logger.info("  3. Documentation: cat docs/BUGS.md")
    logger.info("")
    logger.info("If you see Bug #231 or #243, run:")
    logger.info("  cd Kronos && git pull origin master")
    logger.info("  (That updates upstream cleanly, without our patcher modifying files)")

if __name__ == "__main__":
    main()
