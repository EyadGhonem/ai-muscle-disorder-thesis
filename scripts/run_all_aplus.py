"""
Master runner — executes all 5 A+ analysis scripts in sequence.
Run: python scripts/run_all_aplus.py
"""
import os
import sys
import subprocess
import time

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = [
    ("run_shap_analysis.py",           os.path.join(BASE, "output", "aplus", "run_shap_analysis")),
    ("run_roc_analysis.py",            os.path.join(BASE, "output", "aplus", "run_roc_analysis")),
    ("run_gradcam.py",                 os.path.join(BASE, "output", "aplus", "run_gradcam")),
    ("run_tsne.py",                    os.path.join(BASE, "output", "aplus", "run_tsne")),
    ("run_bias_and_learning_curves.py",os.path.join(BASE, "output", "aplus", "run_bias_and_learning_curves")),
]

BORDER = "─" * 62
results = []

print(f"\n{'═'*62}")
print("  A+ Analysis Runner — AI-Powered Radiomics Thesis")
print(f"{'═'*62}\n")

for script_name, out_dir in SCRIPTS:
    script_path = os.path.join(BASE, "scripts", script_name)
    print(f"{BORDER}")
    print(f"  Running: {script_name}")
    print(f"{BORDER}")

    t0     = time.time()
    status = "PASSED"
    error  = ""

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            cwd    = BASE,
            timeout= 3600,          # 1-hour safety cap
        )
        if proc.returncode != 0:
            status = "FAILED"
            error  = f"exit code {proc.returncode}"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
        error  = "exceeded 3600 s"
    except Exception as exc:
        status = "ERROR"
        error  = str(exc)

    elapsed = time.time() - t0
    results.append((script_name, status, elapsed, out_dir, error))

    tag = "✓" if status == "PASSED" else "✗"
    print(f"\n  [{tag}] {status} in {elapsed:.0f}s\n")

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{'═'*62}")
print("  SUMMARY")
print(f"{'═'*62}")
print(f"  {'Script':<38} {'Status':<9} {'Time':>6}  Output folder")
print(f"  {'-'*38} {'-'*8} {'-'*6}  {'-'*30}")

for script_name, status, elapsed, out_dir, error in results:
    tag   = "✓" if status == "PASSED" else "✗"
    label = f"[{tag}] {status}"
    short = os.path.relpath(out_dir, BASE)
    note  = f"  ({error})" if error else ""
    print(f"  {script_name:<38} {label:<10} {elapsed:>5.0f}s  {short}{note}")

total_ok = sum(1 for _, s, *_ in results if s == "PASSED")
print(f"\n  {total_ok}/{len(results)} scripts completed successfully.")
print(f"  All outputs under: output/aplus/")
print(f"{'═'*62}\n")
