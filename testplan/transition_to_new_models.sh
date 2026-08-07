#!/bin/bash
# transition_to_new_models.sh — wartet bis Qwen3.6-27B-FP8 (aktuell laufendes
# Modell im full_run) sein cooldown erreicht, killt dann full_run + orchestrator,
# und startet einen sauberen Lauf nur über die 3 echten Neuzugänge.
#
# Usage: setsid nohup ./transition_to_new_models.sh <full_run_pid> <full_run_log> \
#          > logs/transition_<ts>.log 2>&1 < /dev/null & disown

set -u
FULL_RUN_PID="${1:?usage: transition_to_new_models.sh <full_run_pid> <full_run_log>}"
WATCH_LOG="${2:?usage: transition_to_new_models.sh <full_run_pid> <full_run_log>}"

cd $HOME/southbyte/southbyte-vllm/testplan
source .venv/bin/activate

TS="$(date +%Y%m%d_%H%M)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/transition_${TS}.log"
NEW_MODELS="Mistral-Small-3.2-24B-NVFP4,Granite-4.1-30B,Mistral-Medium-3.5-128B-NVFP4"

{
  echo "=== transition start $(date -Is) ==="
  echo "Self PID: $$"
  echo "Watching log: $WATCH_LOG"
  echo "Waiting for next 'Cooldown: 30s' line (= Qwen3.6-27B-FP8 done)"
  echo "Then killing full_run PID: $FULL_RUN_PID"
  echo "Then running orchestrator over: $NEW_MODELS"
  echo

  # --- Wait for next Cooldown line in the log ---
  echo "[$(date -Is)] tailing log..."
  tail -F -n 0 "$WATCH_LOG" 2>/dev/null | grep -m1 -F "Cooldown: 30s"
  echo "[$(date -Is)] cooldown line seen — killing full_run."

  # --- Kill full_run + its python orchestrator child ---
  if kill -0 "$FULL_RUN_PID" 2>/dev/null; then
    pkill -TERM -P "$FULL_RUN_PID" 2>/dev/null || true
    sleep 2
    kill -TERM "$FULL_RUN_PID" 2>/dev/null || true

    # Wait up to 60s for clean exit, then SIGKILL
    for i in $(seq 1 30); do
      kill -0 "$FULL_RUN_PID" 2>/dev/null || break
      sleep 2
    done
    if kill -0 "$FULL_RUN_PID" 2>/dev/null; then
      echo "[$(date -Is)] still alive — SIGKILL"
      pkill -KILL -P "$FULL_RUN_PID" 2>/dev/null || true
      kill -KILL "$FULL_RUN_PID" 2>/dev/null || true
    fi
    echo "[$(date -Is)] full_run terminated."
  else
    echo "[$(date -Is)] full_run already gone — proceeding."
  fi

  # Belt-and-suspenders: stop any leftover vllm container so the next orchestrator
  # has a clean machine_b
  echo "[$(date -Is)] stopping any leftover vllm-* container on local docker..."
  docker ps --format '{{.Names}}' | grep -E '^vllm-' | xargs -r -I {} docker stop {} 2>&1 | sed 's/^/  /'

  echo
  echo "--- Phase 1: orchestrator --models ${NEW_MODELS} ---"
  python orchestrator.py --models "${NEW_MODELS}" --continue-after-ko
  ORC_RC=$?
  echo "[$(date -Is)] orchestrator exit=${ORC_RC}"

  echo
  echo "--- Phase 2: consolidate_reports ---"
  python consolidate_reports.py
  CONS_RC=$?
  echo "[$(date -Is)] consolidate exit=${CONS_RC}"

  echo
  echo "=== transition done $(date -Is) — orchestrator=${ORC_RC} consolidate=${CONS_RC} ==="
} >> "$LOG" 2>&1
