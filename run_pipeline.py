"""
run_pipeline.py
===============
Master runner — executes the full OGDC analysis pipeline in order.
Run this once to regenerate every CSV, every image, and every report.

Usage:
    python run_pipeline.py              # run everything
    python run_pipeline.py --step 1    # run only step 1
    python run_pipeline.py --step 1 3  # run steps 1 and 3

Steps:
    1  ogdc_loader_cleaner.py    Load + clean raw CSV
    2  ogdc_analysis.py          Feature engineering + ML models
    3  ogdc_stat222.py           STAT-222 statistical tests
    4  ogdc_trend_analysis.py    Moving averages, BB, volatility
    5  ogdc_garch.py             GARCH volatility modelling
    6  ogdc_backtest.py          Bollinger Band backtesting
    7  ogdc_model_comparison.py  Cross-model comparison + markdown report
    8  ogdc_enhanced_ml.py       Enhanced ML with trend features
    9  ogdc_contradiction_analysis.py  Sentiment contradiction analysis
   10  export_frontend_data.py   Export JSON data for React frontend
"""

import subprocess, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ensure_dirs

ROOT    = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, "scripts")

STEPS = [
    (1,  "ogdc_loader_cleaner.py",         "Load & clean raw CSV"),
    (2,  "ogdc_analysis.py",               "Feature engineering + ML models"),
    (3,  "ogdc_stat222.py",                "STAT-222 statistical tests"),
    (4,  "ogdc_trend_analysis.py",         "Moving averages, BB, volatility"),
    (5,  "ogdc_garch.py",                  "GARCH volatility modelling"),
    (6,  "ogdc_backtest.py",               "Bollinger Band backtesting"),
    (7,  "ogdc_model_comparison.py",       "Model comparison + markdown report"),
    (8,  "ogdc_enhanced_ml.py",            "Enhanced ML with trend features"),
    (9,  "ogdc_contradiction_analysis.py", "Sentiment contradiction analysis"),
    (10, "export_frontend_data.py",        "Export JSON for React frontend"),
]

def run_step(n, script, label):
    path = os.path.join(SCRIPTS, script)
    if not os.path.exists(path):
        print(f"  [SKIP] Script not found: {script}")
        return True
    print(f"  Step {n}: {label}")
    print(f"  Script: scripts/{script}")
    t0  = time.time()
    res = subprocess.run([sys.executable, path], cwd=ROOT)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n   Step {n} failed (exit code {res.returncode})")
        return False
    print(f"\n   Step {n} complete ({elapsed:.1f}s)")
    return True

def main():
    # Create all output directories once before any step runs
    ensure_dirs()

    # Parse --step args
    requested = set()
    args = sys.argv[1:]
    if "--step" in args:
        idx = args.index("--step")
        for s in args[idx+1:]:
            if s.isdigit(): requested.add(int(s))

    steps_to_run = [(n,s,l) for n,s,l in STEPS if not requested or n in requested]

    print("  Ogdc Analysis Pipeline")
    print(f"  Running {len(steps_to_run)} of {len(STEPS)} steps")

    t_start  = time.time()
    failures = []

    for n, script, label in steps_to_run:
        ok = run_step(n, script, label)
        if not ok:
            failures.append(n)

    elapsed = time.time() - t_start
    if failures:
        print(f"   {len(failures)} step(s) failed: {failures}")
    else:
        print(f"   All steps complete in {elapsed:.1f}s")

if __name__ == "__main__":
    main()