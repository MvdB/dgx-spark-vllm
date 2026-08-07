#!/bin/bash
# apertus_run.sh — Testlauf über die 3 neuen Swiss-AI Apertus v1.1 Modelle
# (0.5B / 1.5B / 4B Instruct) auf vLLM v0.23.0. Danach Cross-Model-Dashboard.
# Detached / fail-safe.
#
# Voraussetzung: Port 8000 frei (ggf. fremde vLLM-Prozesse vorher stoppen).
#
# Usage (dry-run):  ./apertus_run.sh --dry-run
# Usage (real):     setsid nohup ./apertus_run.sh > /dev/null 2>&1 < /dev/null & disown

set -u
cd $HOME/southbyte/southbyte-vllm/testplan
source .venv/bin/activate

# Lauf auf gb10-worker2 (10.0.0.8). gb10-desktop löst hier nicht auf —
# alle vLLM-Container laufen lokal → TARGET_HOST=localhost.
export TARGET_HOST=localhost

TS="$(date +%Y%m%d_%H%M)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/apertus_run_${TS}.log"

# 0.5B → 1.5B → 4B (klein → groß)
MODELS="Apertus-v1.1-0.5B,Apertus-v1.1-1.5B,Apertus-v1.1-4B"

EXTRA_ARGS="$*"

{
  echo "=== apertus_run start $(date -Is) ==="
  echo "PID: $$"
  echo "Log: $LOG"
  echo "Scope: 3 Swiss-AI Apertus v1.1 Modelle (vLLM v0.23.0)"
  echo "Models: $MODELS"
  echo "Extra args: ${EXTRA_ARGS:-<none>}"
  echo

  echo "--- Phase 1: orchestrator (--continue-after-ko) ---"
  python orchestrator.py --continue-after-ko --models "$MODELS" ${EXTRA_ARGS}
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
  echo "=== apertus_run end $(date -Is) ==="
} >> "$LOG" 2>&1
