#!/bin/bash
# headsup_nemotron_run.sh — Heads-up Nemotron-Puzzle-75B vs Nemotron-3-Super-120B
# auf vLLM v0.24.0, alle Playbooks (+ lokale *.local.jsonl-Testfälle, falls
# vorhanden). Detached / fail-safe. Nach dem Lauf wird der granite-speech STT-Server
# wieder gestartet (lief vor dem Lauf auf Port 8000).
#
# WICHTIG: Erst NACH bestandenem Smoke-Test beider Modelle auf v0.24.0 starten.
# Fallback Super-120B: v0.18.0 (validiert). Puzzle-75B hat keinen Fallback.
#
# Usage (real): setsid nohup ./headsup_nemotron_run.sh > /dev/null 2>&1 < /dev/null & disown

set -u
cd $HOME/southbyte/southbyte-vllm/testplan
source .venv/bin/activate
export TARGET_HOST=localhost

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/headsup_nemotron_${TS}.log"

# Klein → groß für frühe Ergebnisse.
MODELS="Nemotron-Puzzle-75B,Nemotron-3-Super"

{
  echo "=== headsup_nemotron start $(date -Is) ==="
  echo "PID: $$ | Log: $LOG"
  echo "Models: $MODELS | vLLM v0.24.0 | alle Playbooks"
  echo

  echo "--- Phase 1: orchestrator (--continue-after-ko) ---"
  python orchestrator.py --continue-after-ko --models "$MODELS"
  echo "[$(date -Is)] orchestrator exit=$?"

  echo
  echo "--- Phase 2: consolidate_reports ---"
  python consolidate_reports.py || true
  echo "[$(date -Is)] consolidate done"

  echo
  echo "--- Phase 3: STT-Server wieder hochfahren ---"
  docker ps -aq --filter name=^/vllm-server$ | xargs -r docker rm -f 2>/dev/null || true
  HOST_PORT=8000 HF_MODELS_DIR="$HOME/hf_models" \
    bash $HOME/southbyte/southbyte-vllm/runner/vllm_spark.sh \
    --model granite-speech-4.1-2b-plus --skip-pull || \
    echo "[$(date -Is)] WARNUNG: STT-Neustart fehlgeschlagen — manuell starten"
  echo "[$(date -Is)] STT-Restart angestoßen"

  echo
  echo "=== headsup_nemotron end $(date -Is) ==="
} >> "$LOG" 2>&1
