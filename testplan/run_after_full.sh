#!/bin/bash
# run_after_full.sh — waits for the active full_run (PID passed as $1) to
# finish, then orchestrates a single follow-up model and consolidates.
#
# Usage:  setsid nohup ./run_after_full.sh <waitpid> <model-name> \
#           > logs/run_after_full_<ts>.log 2>&1 < /dev/null & disown

set -u
WAIT_PID="${1:?usage: run_after_full.sh <waitpid> <model-name>}"
MODEL="${2:?usage: run_after_full.sh <waitpid> <model-name>}"

cd $HOME/southbyte/southbyte-vllm/testplan
source .venv/bin/activate

TS="$(date +%Y%m%d_%H%M)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/run_after_full_${TS}.log"

{
  echo "=== run_after_full start $(date -Is) ==="
  echo "Watcher PID: $$"
  echo "Waiting on PID: ${WAIT_PID}"
  echo "Follow-up model: ${MODEL}"
  echo

  if kill -0 "${WAIT_PID}" 2>/dev/null; then
    echo "[$(date -Is)] full_run (PID ${WAIT_PID}) still running — waiting..."
    while kill -0 "${WAIT_PID}" 2>/dev/null; do
      sleep 60
    done
    echo "[$(date -Is)] full_run finished."
  else
    echo "[$(date -Is)] full_run PID ${WAIT_PID} already gone — proceeding."
  fi

  echo
  echo "--- Phase 1: orchestrator --models ${MODEL} ---"
  python orchestrator.py --models "${MODEL}" --continue-after-ko
  ORC_RC=$?
  echo "[$(date -Is)] orchestrator exit=${ORC_RC}"

  echo
  echo "--- Phase 2: consolidate_reports ---"
  python consolidate_reports.py
  CONS_RC=$?
  echo "[$(date -Is)] consolidate exit=${CONS_RC}"

  echo
  echo "=== run_after_full done $(date -Is) — orchestrator=${ORC_RC} consolidate=${CONS_RC} ==="
} >> "$LOG" 2>&1
