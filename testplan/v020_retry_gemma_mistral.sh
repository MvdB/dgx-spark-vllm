#!/bin/bash
# Re-Run für Gemma-4-31B-IT-NVFP4 + Mistral-Small-3.2-24B-NVFP4 nach Profil-Fixes:
#  - Gemma:   PROFILE_MAX_NUM_BATCHED_TOKENS=4096  (Default 2048 < max_tokens_per_mm_item=2496)
#  - Mistral: tokenizer_config.json (PreTrainedTokenizerFast) erzwungen, TOKENIZER_MODE=hf
# Detached (nohup/setsid) — survives SSH disconnect.

set -u
cd $HOME/southbyte/southbyte-vllm/testplan
source .venv/bin/activate

TS="$(date +%Y%m%d_%H%M)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/v020_retry_${TS}.log"

MODELS="Gemma-4-31B-IT-NVFP4,Mistral-Small-3.2-24B-NVFP4"

echo "=== v020_retry start $(date) ===" | tee -a "$LOG"
echo "Models: $MODELS"                  | tee -a "$LOG"
echo "Log: $LOG"                        | tee -a "$LOG"

python orchestrator.py --models "$MODELS" --continue-after-ko >> "$LOG" 2>&1
RC=$?

echo "=== v020_retry done $(date) — orchestrator exit=$RC ===" | tee -a "$LOG"
exit $RC
