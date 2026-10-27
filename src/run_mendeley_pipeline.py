"""
src/run_mendeley_pipeline.py - Master Pipeline Runner
Orchestrates Phase 4 (Mendeley Training), Phase 5 (Combined Training), and Phase 6 (Comparative Evaluation).
"""

import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = PROJECT_ROOT / ".venv/Scripts/python.exe"

def run_step(step_name, command_args):
    print("\n" + "=" * 75)
    print(f"PIPELINE STEP: {step_name}")
    print("=" * 75)
    t0 = time.time()
    res = subprocess.run([str(PYTHON_EXE)] + command_args, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with code {res.returncode}")
        sys.exit(res.returncode)
    print(f"\n[PASS] Step '{step_name}' completed in {elapsed:.1f}s ({elapsed/60:.2f} min).")

def main():
    print("=" * 75)
    print("STARTING AUTONOMOUS END-TO-END TRAINING & EVALUATION PIPELINE")
    print("=" * 75)

    # 1. Train Mendeley Standalone Model (36 classes)
    run_step("Train Standalone Mendeley Model (36 Classes)", ["src/train_mendeley.py", "--epochs", "15", "--batch-size", "64"])

    # 2. Train Combined Unified Model (Real Camera + Mendeley, 71,705 samples)
    run_step("Train Combined Unified Model (71,705 Samples, 36 Classes)", ["src/train_combined.py", "--epochs", "15", "--batch-size", "64"])

    # 3. Cross-Domain Comparative Benchmark
    run_step("Multi-Domain Cross-Evaluation Benchmark", ["src/evaluate_combined.py", "--batch-size", "128"])

    print("\n" + "=" * 75)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
