#!/bin/bash
# v21_migrated_run.sh — testplan über die 11 aktiven v0.21.0-Migrationen,
# danach Cross-Model-Dashboard. Detached / fail-safe.
#
# Usage: setsid nohup ./v21_migrated_run.sh > /dev/null 2>&1 < /dev/null & disown

set -u
cd $HOME/southbyte/southbyte-vllm/testplan
source .venv/bin/activate

# Wir laufen auf gb10-worker2 (10.0.0.8). gb10-desktop löst hier nicht auf —
# SSH zu localhost geht, alle vLLM-Container laufen lokal. Daher TARGET_HOST
# auf localhost setzen (überschreibt .env via os.environ.setdefault).
export TARGET_HOST=localhost

TS="$(date +%Y%m%d_%H%M)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/v21_migrated_${TS}.log"

MODELS="Granite-4.1-30B,Gemma-4-E2B,Gemma-4-E4B,Gemma-4-31B-IT-NVFP4,Nemotron-3-Nano-Omni-30B,Qwen3.5-27B-FP8,Nemotron-3-Nano-30B,Qwen3.6-27B-FP8,Qwen3.6-35B-A3B-FP8,Gemma-4-26B-A4B,Gemma-4-31B"

{
  echo "=== v21_migrated_run start $(date -Is) ==="
  echo "PID: $$"
  echo "Log: $LOG"
  echo "Scope: 11 freshly-migrated v0.21.0 models"
  echo "Models: $MODELS"
  echo

  echo "--- Phase 1: orchestrator (--continue-after-ko) ---"
  python orchestrator.py --continue-after-ko --models "$MODELS"
  ORC_RC=$?
  echo "[$(date -Is)] orchestrator exit=${ORC_RC}"

  echo
  echo "--- Phase 2: consolidate_reports ---"
  python consolidate_reports.py || true
  echo "[$(date -Is)] consolidate done"

  echo
  echo "=== v21_migrated_run end $(date -Is) ==="
} >> "$LOG" 2>&1
