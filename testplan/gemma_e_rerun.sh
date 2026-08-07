#!/bin/bash
# gemma_e_rerun.sh — Re-Run der kleinen Gemma-4-Modelle (E2B / E4B) auf vLLM
# v0.23.0 mit dem aktuellen Testsatz (inkl. gefixtem max_tokens-Clamp), für
# einen sauberen Apples-to-Apples-Vergleich mit der Apertus-v1.1-Kohorte.
# Danach Cross-Model-Dashboard.
#
# Detached / fail-safe.
# Usage (real): setsid nohup ./gemma_e_rerun.sh > /dev/null 2>&1 < /dev/null & disown

set -u
cd $HOME/dgx-spark/dgx-spark-vllm/testplan
source .venv/bin/activate
export TARGET_HOST=localhost

TS="$(date +%Y%m%d_%H%M%S)"          # Sekunden-Auflösung → keine Log-Kollision
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/gemma_e_rerun_${TS}.log"

MODELS="Gemma-4-E2B,Gemma-4-E4B"     # klein → größer

{
  echo "=== gemma_e_rerun start $(date -Is) ==="
  echo "PID: $$ | Log: $LOG"
  echo "Models: $MODELS | vLLM v0.23.0 | alle Playbooks"
  echo

  echo "--- Phase 1: orchestrator (--continue-after-ko) ---"
  python orchestrator.py --continue-after-ko --models "$MODELS"
  echo "[$(date -Is)] orchestrator exit=$?"

  echo
  echo "--- Phase 2: consolidate_reports ---"
  python consolidate_reports.py || true
  echo "[$(date -Is)] consolidate done"

  echo
  echo "=== gemma_e_rerun end $(date -Is) ==="
} >> "$LOG" 2>&1
