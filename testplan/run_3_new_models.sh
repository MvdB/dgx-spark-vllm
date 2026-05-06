#!/bin/bash
# run_3_new_models.sh — Single-Run nur über die echten Neuzugänge,
# danach consolidate. Detached / fail-safe.
#
# Usage: setsid nohup ./run_3_new_models.sh > logs/run_3_new_<ts>.log 2>&1 < /dev/null & disown

set -u
cd /home/mvdb/dgx-spark-vllm/testplan
source .venv/bin/activate

TS="$(date +%Y%m%d_%H%M)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/run_3_new_${TS}.log"
MODELS="Mistral-Small-3.2-24B-NVFP4,Granite-4.1-30B,Mistral-Medium-3.5-128B-NVFP4"

{
  echo "=== run_3_new start $(date -Is) ==="
  echo "PID: $$"
  echo "Models: $MODELS"
  echo "Log: $LOG"
  echo

  echo "--- Phase 1: orchestrator --models ${MODELS} --continue-after-ko ---"
  python orchestrator.py --models "${MODELS}" --continue-after-ko
  ORC_RC=$?
  echo "[$(date -Is)] orchestrator exit=${ORC_RC}"

  echo
  echo "--- Phase 2: consolidate_reports ---"
  python consolidate_reports.py
  CONS_RC=$?
  echo "[$(date -Is)] consolidate exit=${CONS_RC}"

  echo
  echo "=== run_3_new done $(date -Is) — orchestrator=${ORC_RC} consolidate=${CONS_RC} ==="
} >> "$LOG" 2>&1
