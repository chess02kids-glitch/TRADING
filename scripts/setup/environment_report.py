#!/usr/bin/env python3
"""
PHASE 1 - Audit #8: Reproducibility - Environment Report
Generates report containing:
- OS, Python version, PyTorch version, CUDA version, GPU, Kronos Git commit, model revision, dependency versions

Must be run after installation to verify exact versions
"""

import sys
import platform
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_os_info():
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform()
    }

def get_python_info():
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
        "path": sys.path[:5]
    }

def get_torch_info():
    try:
        import torch
        info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "hip_version": getattr(torch.version, 'hip', None),
        }
        
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count
                })
            info["gpus"] = gpu_info
            info["current_device"] = torch.cuda.current_device()
        else:
            info["gpus"] = []
        
        return info
    except ImportError:
        return {"installed": False, "error": "torch not installed"}
    except Exception as e:
        return {"installed": False, "error": str(e)}

def get_kronos_git_info():
    kronos_dir = PROJECT_ROOT / "Kronos"
    if not kronos_dir.exists():
        return {"exists": False, "error": "Kronos repo not cloned"}
    
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(kronos_dir),
            capture_output=True, text=True, timeout=10
        )
        commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=str(kronos_dir),
            capture_output=True, text=True, timeout=10
        )
        latest_commit_msg = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(kronos_dir),
            capture_output=True, text=True, timeout=10
        )
        branch = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(kronos_dir),
            capture_output=True, text=True, timeout=10
        )
        is_dirty = len(result.stdout.strip()) > 0 if result.returncode == 0 else None
        
        return {
            "exists": True,
            "commit_hash": commit_hash,
            "latest_commit_msg": latest_commit_msg,
            "branch": branch,
            "is_dirty": is_dirty,
            "path": str(kronos_dir)
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}

def get_model_revision():
    models_dir = PROJECT_ROOT / "models"
    selected_file = models_dir / "selected_model.yaml"
    
    info = {
        "models_dir_exists": models_dir.exists(),
        "selected_model_file_exists": selected_file.exists(),
        "hf_cache_exists": (Path.home() / ".cache" / "huggingface" / "hub").exists()
    }
    
    if selected_file.exists():
        try:
            import yaml
            with open(selected_file) as f:
                data = yaml.safe_load(f)
            info["selected_model"] = data
            
            # Try to get HF revision from local cache if available
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if hf_cache.exists():
                for model_name in [data.get("model", ""), data.get("tokenizer", "")]:
                    if not model_name:
                        continue
                    # HF cache dir format: models--<org>--<model>
                    cache_name = f"models--{model_name.replace('/', '--')}"
                    cache_path = hf_cache / cache_name
                    if cache_path.exists():
                        # Check refs/main
                        ref_file = cache_path / "refs" / "main"
                        if ref_file.exists():
                            info[f"{model_name}_hf_revision"] = ref_file.read_text().strip()
        except Exception as e:
            info["selected_model_error"] = str(e)
    
    # List cached models
    try:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            cached = [d.name for d in hf_cache.iterdir() if "kronos" in d.name.lower()]
            info["cached_kronos_models"] = cached[:10]
    except Exception:
        pass
    
    return info

def get_dependency_versions():
    deps = [
        "numpy", "pandas", "torch", "einops", "huggingface_hub", "tqdm",
        "safetensors", "matplotlib", "ccxt", "yaml", "dotenv", "sqlalchemy",
        "requests", "scipy", "sklearn", "vectorbt", "quantstats", "streamlit",
        "plotly", "pydantic", "loguru"
    ]
    
    versions = {}
    for dep in deps:
        try:
            # Special cases
            if dep == "yaml":
                import yaml
                versions[dep] = getattr(yaml, '__version__', 'unknown')
            elif dep == "dotenv":
                import dotenv
                versions[dep] = getattr(dotenv, '__version__', 'unknown')
            elif dep == "sklearn":
                import sklearn
                versions[dep] = sklearn.__version__
            else:
                mod = __import__(dep)
                versions[dep] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            versions[dep] = "NOT_INSTALLED"
        except Exception as e:
            versions[dep] = f"ERROR: {e}"
    
    return versions

def get_config_summary():
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        return {"exists": False}
    
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        # Only include non-sensitive summary
        return {
            "exists": True,
            "trading_mode": cfg.get("trading", {}).get("mode", "UNKNOWN"),
            "live_enabled": cfg.get("trading", {}).get("live_trading_enabled", False),
            "timeframes": cfg.get("data", {}).get("timeframes", []),
            "primary_timeframe": cfg.get("data", {}).get("primary_timeframe", ""),
            "confirmation_timeframe": cfg.get("data", {}).get("confirmation_timeframe", ""),
            "regime_timeframe": cfg.get("data", {}).get("regime_timeframe", ""),
            "assets": cfg.get("data", {}).get("assets", []),
            "model_size": cfg.get("hardware", {}).get("model_size", ""),
            "risk_per_trade": cfg.get("risk", {}).get("risk_per_trade", "")
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}

def generate_report():
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "project": "kronos_trading_system",
        "phase": "Phase 1 Audited",
        "os": get_os_info(),
        "python": get_python_info(),
        "pytorch": get_torch_info(),
        "kronos_git": get_kronos_git_info(),
        "model_revision": get_model_revision(),
        "dependencies": get_dependency_versions(),
        "config_summary": get_config_summary()
    }
    return report

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate environment reproducibility report")
    parser.add_argument("--output", type=str, default="logs/environment_report.json",
                        help="Output file for report")
    parser.add_argument("--yaml", action="store_true", help="Also output YAML")
    args = parser.parse_args()
    
    print("="*70)
    print("ENVIRONMENT REPORT - Reproducibility Audit #8")
    print("="*70)
    
    report = generate_report()
    
    # Save JSON
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to {output_path}")
    
    # Also YAML if requested
    if args.yaml:
        try:
            import yaml
            yaml_path = output_path.with_suffix('.yaml')
            with open(yaml_path, 'w') as f:
                yaml.dump(report, f, default_flow_style=False, sort_keys=False)
            print(f"✓ YAML report saved to {yaml_path}")
        except Exception as e:
            print(f"YAML export failed: {e}")
    
    # Print summary to console (no secrets)
    print("\n" + "="*70)
    print("SUMMARY (No secrets)")
    print("="*70)
    print(f"OS: {report['os']['system']} {report['os']['release']} ({report['os']['machine']})")
    print(f"Python: {report['python']['version']}")
    print(f"PyTorch: {report['pytorch'].get('version', 'NOT INSTALLED')}")
    print(f"CUDA Available: {report['pytorch'].get('cuda_available', False)}")
    if report['pytorch'].get('cuda_available'):
        print(f"CUDA Version: {report['pytorch'].get('cuda_version')}")
        for gpu in report['pytorch'].get('gpus', []):
            print(f"GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']}GB)")
    else:
        print("GPU: None (CPU mode)")
    
    print(f"Kronos Git Commit: {report['kronos_git'].get('commit_hash', 'NOT CLONED')[:12]}")
    print(f"Kronos Branch: {report['kronos_git'].get('branch', 'unknown')} | Dirty: {report['kronos_git'].get('is_dirty')}")
    print(f"Config Mode: {report['config_summary'].get('trading_mode', 'UNKNOWN')} | LIVE enabled: {report['config_summary'].get('live_enabled')}")
    print(f"Timeframes: {report['config_summary'].get('timeframes')}")
    
    print("\nKey Dependencies:")
    for dep in ["numpy", "pandas", "einops", "huggingface_hub", "ccxt", "safetensors"]:
        print(f"  {dep}: {report['dependencies'].get(dep)}")
    
    # Validation against expected pinned versions
    print("\n" + "="*70)
    print("PINNED VERSION CHECK")
    print("="*70)
    expected = {
        "numpy": "1.26.4",
        "pandas": "2.2.2",
        "einops": "0.8.1",
        "huggingface_hub": "0.33.1",
        "tqdm": "4.67.1",
        "safetensors": "0.6.2",
        "matplotlib": "3.9.3",
        "ccxt": "4.2.25"
    }
    
    all_match = True
    for pkg, exp_ver in expected.items():
        actual = report['dependencies'].get(pkg, "NOT_INSTALLED")
        match = exp_ver in actual if actual != "NOT_INSTALLED" else False
        symbol = "✅" if match else "❌"
        if not match:
            all_match = False
        print(f"{symbol} {pkg}: expected {exp_ver}, got {actual}")
    
    # Torch special check - should be 2.4.1 as per audited env
    torch_actual = report['dependencies'].get('torch', 'NOT_INSTALLED')
    torch_expected = "2.4.1"
    torch_match = torch_expected in torch_actual
    symbol = "✅" if torch_match else "⚠️ "
    if not torch_match and torch_actual != "NOT_INSTALLED":
        print(f"{symbol} torch: expected {torch_expected} (pinned), got {torch_actual} - check consistency")
    elif torch_actual == "NOT_INSTALLED":
        print(f"❌ torch: NOT INSTALLED - run conda env create -f environment.yml")
        all_match = False
    else:
        print(f"{symbol} torch: {torch_actual} (matches pinned {torch_expected})")
    
    if all_match:
        print("\n✅ All pinned versions match - reproducible")
    else:
        print("\n⚠️ Version mismatches - review environment.yml vs requirements_exact.txt")
    
    return 0 if all_match else 1

if __name__ == "__main__":
    sys.exit(main())
