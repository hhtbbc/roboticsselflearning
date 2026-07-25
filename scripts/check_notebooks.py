#!/usr/bin/env python3
"""Notebook 自动验证脚本：批量执行并报告结果"""
import subprocess
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"

def check_notebook(ipynb_path: Path, timeout: int = 300) -> dict:
    """执行单个 Notebook 并返回结果"""
    start = time.time()
    try:
        result = subprocess.run(
            ["uv", "run", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--inplace", str(ipynb_path),
             "--ExecutePreprocessor.timeout", str(timeout),
             "--ExecutePreprocessor.kernel_name", "robotics-learning",
             "--output-dir", str(NOTEBOOKS)],
            capture_output=True, text=True, timeout=timeout + 60, cwd=ROOT
        )
        elapsed = time.time() - start
        success = result.returncode == 0
        return {
            "notebook": ipynb_path.name,
            "success": success,
            "elapsed_sec": round(elapsed, 1),
            "stderr": result.stderr[:500] if not success else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "notebook": ipynb_path.name,
            "success": False,
            "elapsed_sec": timeout,
            "stderr": "Timeout",
        }
    except Exception as e:
        return {
            "notebook": ipynb_path.name,
            "success": False,
            "elapsed_sec": time.time() - start,
            "stderr": str(e)[:500],
        }

def main():
    if len(sys.argv) > 1:
        files = [NOTEBOOKS / f for f in sys.argv[1:]]
    else:
        files = sorted(NOTEBOOKS.glob("*.ipynb"))

    results = []
    for f in files:
        print(f"Checking: {f.name} ... ", end="", flush=True)
        r = check_notebook(f)
        status = "✅" if r["success"] else "❌"
        print(f"{status} ({r['elapsed_sec']:.1f}s)")
        if r["stderr"]:
            print(f"  Error: {r['stderr'][:200]}")
        results.append(r)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")

    # Write JSON report
    report_path = ROOT / "outputs" / "notebook_check_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report: {report_path}")

    if failed > 0:
        print(f"\n❌ {failed} notebook(s) failed!")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
