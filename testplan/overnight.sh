#!/bin/bash
# overnight.sh — runs independently of Claude session
# Waits for run11, consolidates, runs Gemma-4-31B-IT-NVFP4, consolidates again.

set -euo pipefail
cd $HOME/southbyte/southbyte-vllm/testplan
source .venv/bin/activate

LOG=logs/overnight.log
exec > >(tee -a "$LOG") 2>&1

echo "=== overnight.sh started $(date) ==="

# ── Step 1: Wait for run11 (PID 1844073) ────────────────────────────────────
RUN11_PID=1844073
if kill -0 "$RUN11_PID" 2>/dev/null; then
    echo "$(date) Waiting for run11 (PID $RUN11_PID) to finish..."
    while kill -0 "$RUN11_PID" 2>/dev/null; do
        sleep 30
    done
    echo "$(date) run11 finished."
else
    echo "$(date) run11 PID $RUN11_PID already gone — proceeding."
fi

# ── Step 2: Consolidate after run11 ─────────────────────────────────────────
echo "$(date) Running consolidate_reports.py (post-run11)..."
python3 consolidate_reports.py && echo "$(date) consolidate_reports.py done." \
    || echo "$(date) WARNING: consolidate_reports.py exited non-zero — continuing anyway."

# ── Step 3: Run Gemma-4-31B-IT-NVFP4 (run12) ────────────────────────────────
echo "$(date) Starting run12: Gemma-4-31B-IT-NVFP4..."
python3 orchestrator.py --models "Gemma-4-31B-IT-NVFP4" --continue-after-ko \
    > logs/cohort_retry_run12.log 2>&1 \
    && echo "$(date) run12 finished OK." \
    || echo "$(date) run12 finished with non-zero exit — check cohort_retry_run12.log."

# ── Step 4: Consolidate after run12 ─────────────────────────────────────────
echo "$(date) Running consolidate_reports.py (post-run12)..."
python3 consolidate_reports.py && echo "$(date) consolidate_reports.py done." \
    || echo "$(date) WARNING: consolidate_reports.py exited non-zero."

echo "=== overnight.sh complete $(date) ==="
