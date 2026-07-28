#!/usr/bin/env python3
"""Notebook 自动验证脚本：批量执行并报告结果。

使用 nbclient 执行，结果输出至 outputs/executed_notebooks/，
报告写入 outputs/notebook_report.json。
原文件不会被覆盖。
"""
import subprocess
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
OUTPUT_DIR = ROOT / "outputs" / "executed_notebooks"
LOG_DIR = ROOT / "outputs" / "notebook_logs"


def check_notebook(ipynb_path: Path, timeout: int = 300) -> dict:
    """执行单个 Notebook 并返回结果。输出写入 output 目录，不覆盖原文件。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUTPUT_DIR / ipynb_path.name
    start = time.time()

    try:
        # 使用 nbclient 执行，输出到单独的 output 目录
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError

        nb = nbformat.read(str(ipynb_path), as_version=4)
        client = NotebookClient(
            nb, timeout=timeout,
            kernel_name="robotics-learning",
            record_timing=True,
        )

        client.execute()
        elapsed = time.time() - start

        # 检查输出中是否有 NaN/Inf
        nan_cells = []
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and cell.outputs:
                for out in cell.outputs:
                    if out.output_type == 'execute_result' and 'text/plain' in out.data:
                        text = out.data['text/plain']
                        if 'nan' in text.lower() or 'inf' in text.lower():
                            nan_cells.append(idx)
                            break

        # 保存执行结果
        nbformat.write(nb, str(out_path))

        success = True
        return {
            "notebook": ipynb_path.name,
            "success": success,
            "elapsed_sec": round(elapsed, 1),
            "nan_warnings": nan_cells,
            "output_path": str(out_path.relative_to(ROOT)),
        }
    except CellExecutionError as e:
        elapsed = time.time() - start
        # 保存执行到一半的 notebook
        try:
            import nbformat
            nbformat.write(e.nb if hasattr(e, 'nb') else nbformat.read(str(ipynb_path), as_version=4),
                          str(out_path))
        except Exception:
            pass
        return {
            "notebook": ipynb_path.name,
            "success": False,
            "elapsed_sec": round(elapsed, 1),
            "error": str(e)[:500],
            "cell_index": getattr(e, 'cell_index', None),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "notebook": ipynb_path.name,
            "success": False,
            "elapsed_sec": round(elapsed, 1),
            "error": str(e)[:500],
        }


def main():
    # 如果作为子进程回退到 nbconvert 模式（兼容性）
    if "--nbconvert" in sys.argv:
        check_nbconvert(sys.argv[1:] if len(sys.argv) > 1 else [])
        return

    # 尝试 nbclient mode
    try:
        import nbclient  # noqa: F401
    except ImportError:
        print("⚠ nbclient 未安装，回退至 nbconvert 模式")
        check_nbconvert(sys.argv[1:])
        return

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        files = [NOTEBOOKS / f for f in sys.argv[1:]]
    else:
        files = sorted(NOTEBOOKS.glob("*.ipynb"))

    results = []
    for f in files:
        print(f"Checking: {f.name} ... ", end="", flush=True)
        r = check_notebook(f)
        status = "✅" if r["success"] else "❌"
        print(f"{status} ({r['elapsed_sec']:.1f}s)")
        if not r["success"]:
            err = r.get("error", "")
            cell = r.get("cell_index")
            extra = f" [cell {cell}]" if cell is not None else ""
            print(f"  Error{extra}: {err[:300]}")
        if r.get("nan_warnings"):
            print(f"  ⚠ NaN/Inf in cells: {r['nan_warnings']}")
        results.append(r)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")

    # Write JSON report
    report_path = ROOT / "outputs" / "notebook_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report: {report_path}")
    print(f"Executed notebooks: {OUTPUT_DIR}")

    if failed > 0:
        print(f"\n❌ {failed} notebook(s) failed!")
        raise SystemExit(1)


def check_nbconvert(args):
    """兼容模式：使用 nbconvert --execute 输出至 outputs/ 目录 (不覆盖原文件)"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args:
        files = [NOTEBOOKS / f for f in args if not f.startswith("--")]
    else:
        files = sorted(NOTEBOOKS.glob("*.ipynb"))

    results = []
    for f in files:
        out_f = OUTPUT_DIR / f.name
        print(f"Checking: {f.name} ... ", end="", flush=True)
        try:
            result = subprocess.run(
                ["uv", "run", "jupyter", "nbconvert", "--to", "notebook",
                 "--execute", str(f),
                 "--ExecutePreprocessor.timeout", "300",
                 "--ExecutePreprocessor.kernel_name", "robotics-learning",
                 "--output", str(out_f)],
                capture_output=True, text=True, timeout=360, cwd=ROOT
            )
            success = result.returncode == 0
            r = {"notebook": f.name, "success": success,
                 "elapsed_sec": 0.0, "stderr": result.stderr[:500] if not success else ""}
            status = "✅" if success else "❌"
            print(status)
            if not success:
                print(f"  Error: {r['stderr'][:200]}")
            results.append(r)
        except subprocess.TimeoutExpired:
            print("❌ (timeout)")
            results.append({"notebook": f.name, "success": False,
                           "elapsed_sec": 300.0, "stderr": "Timeout"})

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")

    if failed > 0:
        print(f"\n❌ {failed} notebook(s) failed!")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
