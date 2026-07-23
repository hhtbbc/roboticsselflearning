#!/bin/bash
# 批量执行所有 Notebook 并报告结果
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOTEBOOKS="$ROOT/notebooks"
LOG_DIR="$ROOT/outputs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"
TOTAL=0; PASSED=0; FAILED=0
echo "=== 批量执行 $(date) ===" | tee "$LOG_FILE"
for nb in "$NOTEBOOKS"/*.ipynb; do
    name=$(basename "$nb"); TOTAL=$((TOTAL+1))
    printf "[%2d/26] %-45s " "$TOTAL" "$name" | tee -a "$LOG_FILE"
    t0=$(date +%s)
    if uv run jupyter nbconvert --to notebook --execute --inplace "$nb" \
        --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=robotics-learning \
        --output-dir "$NOTEBOOKS" >> "$LOG_FILE" 2>&1; then
        echo "✅ (${elapsed}s)" | tee -a "$LOG_FILE"; PASSED=$((PASSED+1))
    else
        echo "❌ FAILED" | tee -a "$LOG_FILE"; FAILED=$((FAILED+1))
    fi
done
echo "=== $PASSED/$TOTAL passed ===" | tee -a "$LOG_FILE"
