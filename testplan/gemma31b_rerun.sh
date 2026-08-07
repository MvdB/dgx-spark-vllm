#!/bin/bash
# gemma31b_rerun.sh — gezielter Rerun für Gemma-4-31B nach Threshold-Senkung
# (min_quality_pass_rate 0.75 → 0.70). Detached / fail-safe.
#
# Usage (dry-run):  ./gemma31b_rerun.sh --dry-run
# Usage (real):     setsid nohup ./gemma31b_rerun.sh > /dev/null 2>&1 < /dev/null & disown

set -u
cd $HOME/dgx-spark/dgx-spark-vllm/testplan
source .venv/bin/activate

# Wir laufen auf gb10-worker2 (10.0.0.8). gb10-desktop löst hier nicht auf —
# alle vLLM-Container laufen lokal → TARGET_HOST=localhost.
export TARGET_HOST=localhost

TS="$(date +%Y%m%d_%H%M)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/gemma31b_rerun_${TS}.log"

EXTRA_ARGS="$*"

{
  echo "=== gemma31b_rerun start $(date -Is) ==="
  echo "PID: $$"
  echo "Log: $LOG"
  echo "Extra args: ${EXTRA_ARGS:-<none>}"
  echo

  echo "--- Phase 1: orchestrator (--continue-after-ko, --models Gemma-4-31B) ---"
  python orchestrator.py --continue-after-ko --models Gemma-4-31B ${EXTRA_ARGS}
  ORC_RC=$?
  echo "[$(date -Is)] orchestrator exit=${ORC_RC}"

  if [ -z "${EXTRA_ARGS}" ] || ! echo "${EXTRA_ARGS}" | grep -q -- '--dry-run'; then
    echo
    echo "--- Phase 2: consolidate_reports ---"
    python consolidate_reports.py || true
    echo "[$(date -Is)] consolidate done"
  else
    echo "Dry-Run: consolidate_reports übersprungen"
  fi

  echo
  echo "=== gemma31b_rerun end $(date -Is) ==="
} >> "$LOG" 2>&1
