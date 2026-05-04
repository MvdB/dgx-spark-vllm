#!/bin/bash
# v020_full_run.sh — full testplan run for the 4 v0.20.0 candidate models.
# Detached (nohup/setsid) — survives SSH disconnect.

set -u
cd /home/mvdb/dgx-spark-vllm/testplan
source .venv/bin/activate

TS="$(date +%Y%m%d_%H%M)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/v020_full_${TS}.log"

# Reihenfolge: Omni zuerst (höchste Priorität), Qwen3.5-9B-GPTQ skipped (Registry-Bug auch in v0.20.0).
MODELS="Nemotron-3-Nano-Omni-30B,Gemma-4-31B-IT-NVFP4,Mistral-Small-3.2-24B-NVFP4"

echo "=== v020_full_run start $(date) ===" | tee -a "$LOG"
echo "Models: $MODELS"                        | tee -a "$LOG"
echo "Log: $LOG"                              | tee -a "$LOG"

python orchestrator.py --models "$MODELS" --continue-after-ko >> "$LOG" 2>&1
RC=$?

echo "=== v020_full_run done $(date) — orchestrator exit=$RC ===" | tee -a "$LOG"
exit $RC
