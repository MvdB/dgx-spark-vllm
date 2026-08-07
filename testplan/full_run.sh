#!/bin/bash
# full_run.sh — runs orchestrator over ALL active models in testplan.yaml,
# then consolidates the cross-model dashboard. Detached / fail-safe.
#
# Usage:  setsid nohup ./full_run.sh > logs/full_run_<ts>.log 2>&1 < /dev/null & disown

set -u
cd $HOME/southbyte/southbyte-vllm/testplan
source .venv/bin/activate

TS="$(date +%Y%m%d_%H%M)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/full_run_${TS}.log"

{
  echo "=== full_run start $(date -Is) ==="
  echo "PID: $$"
  echo "Log: $LOG"
  echo

  # Phase 1: orchestrator over all active models, fail-safe
  echo "--- Phase 1: orchestrator (all active, --continue-after-ko) ---"
  python orchestrator.py --continue-after-ko
  ORC_RC=$?
  echo "[$(date -Is)] orchestrator exit=${ORC_RC}"

  # Phase 2: consolidate cross-model dashboard regardless of orchestrator exit
  echo
  echo "--- Phase 2: consolidate_reports ---"
  python consolidate_reports.py
  CONS_RC=$?
  echo "[$(date -Is)] consolidate exit=${CONS_RC}"

  echo
  echo "=== full_run done $(date -Is) — orchestrator=${ORC_RC} consolidate=${CONS_RC} ==="
} >> "$LOG" 2>&1
