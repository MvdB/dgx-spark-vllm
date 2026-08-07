#!/bin/bash
# headsup_rerun.sh — Re-Run der vom Leere-Antworten-Bug betroffenen Playbooks
# für beide Nemotrons, nach Fix max_tokens 2048→8192 / timeout 300→900 in
# evaluators/base.py. Danach consolidate + STT-Neustart.
#
# Usage (real): setsid nohup ./headsup_rerun.sh > /dev/null 2>&1 < /dev/null & disown

set -u
cd $HOME/dgx-spark/dgx-spark-vllm/testplan
source .venv/bin/activate
export TARGET_HOST=localhost

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/headsup_rerun_${TS}.log"

MODELS="Nemotron-Puzzle-75B,Nemotron-3-Super"
PLAYBOOKS="01_quality,02_german_language,03_bias,04_security"

{
  echo "=== headsup_rerun start $(date -Is) ==="
  echo "PID: $$ | Log: $LOG"
  echo "Models: $MODELS | Playbooks: $PLAYBOOKS | max_tokens=8192, timeout=900"
  echo

  echo "--- Phase 1: orchestrator (--continue-after-ko) ---"
  python orchestrator.py --continue-after-ko --models "$MODELS" --playbooks "$PLAYBOOKS"
  echo "[$(date -Is)] orchestrator exit=$?"

  echo
  echo "--- Phase 2: consolidate_reports ---"
  python consolidate_reports.py || true
  echo "[$(date -Is)] consolidate done"

  echo
  echo "--- Phase 3: STT-Server wieder hochfahren ---"
  docker ps -aq --filter name=^/vllm-server$ | xargs -r docker rm -f 2>/dev/null || true
  HOST_PORT=8000 HF_MODELS_DIR="$HOME/hf_models" \
    bash $HOME/dgx-spark/dgx-spark-vllm/runner/vllm_spark.sh \
    --model granite-speech-4.1-2b-plus --skip-pull || \
    echo "[$(date -Is)] WARNUNG: STT-Neustart fehlgeschlagen — manuell starten"
  echo "[$(date -Is)] STT-Restart angestoßen"

  echo
  echo "=== headsup_rerun end $(date -Is) ==="
} >> "$LOG" 2>&1
