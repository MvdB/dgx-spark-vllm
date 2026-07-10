#!/bin/bash
# unified_v024_run.sh — Vergleichslauf: 7 Modelle, einheitlich vLLM v0.24.0,
# optimierte Runtime-Params:
#   * MTP/speculative decoding aktiv wo der Checkpoint es hergibt
#     (beide Nemotrons, beide Qwen3.6 — Gemma/Granite haben keine MTP-Gewichte)
#   * Nemotrons: generation_config-Sampling (temp=1.0/top_p=0.95) + Thinking
#     medium (enable_thinking+low_effort), andere temp=0.1 wie bisher
#   * Degenerations-Guard aktiv (Token-Limit ohne Content → FAIL statt PASS)
#   * Default-System-Prompt inkl. neuer Prämissenprüfung
# Danach consolidate + Nemotron-Heads-up-Report + STT-Neustart.
#
# Usage (real): setsid nohup ./unified_v024_run.sh > /dev/null 2>&1 < /dev/null & disown

set -u
cd /home/mvdb/dgx-spark-vllm/testplan
source .venv/bin/activate
export TARGET_HOST=localhost

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/unified_v024_${TS}.log"

MODELS="Nemotron-3-Super,Nemotron-Puzzle-75B,Gemma-4-26B-A4B,Gemma-4-31B,Granite-4.1-30B,Qwen3.6-27B-FP8,Qwen3.6-35B-A3B-FP8"
PLAYBOOKS="01_quality,02_german_language,03_bias,04_security,05_code,06_performance"

{
  echo "=== unified_v024_run start $(date -Is) ==="
  echo "PID: $$ | Log: $LOG"
  echo "Models: $MODELS"
  echo "Playbooks: $PLAYBOOKS"
  echo "Config: v0.24.0 einheitlich, MTP (Nemotrons+Qwens), Praemissenpruefung im System-Prompt, Degenerations-Guard"
  echo

  echo "--- Phase 1: orchestrator (--continue-after-ko) ---"
  python orchestrator.py --continue-after-ko --models "$MODELS" --playbooks "$PLAYBOOKS"
  echo "[$(date -Is)] orchestrator exit=$?"

  echo
  echo "--- Phase 2: consolidate_reports ---"
  python consolidate_reports.py || true
  echo "[$(date -Is)] consolidate done"

  echo
  echo "--- Phase 3: Nemotron-Heads-up-Report neu generieren ---"
  RUN_DIR="$(ls -1dt reports/2026-* 2>/dev/null | head -1 | xargs -r basename)"
  if [ -n "$RUN_DIR" ]; then
    python nemotron_headsup_report.py --post-run "$RUN_DIR" || \
      echo "[$(date -Is)] WARNUNG: Heads-up-Report fehlgeschlagen (Rohdaten in reports/$RUN_DIR)"
  fi
  echo "[$(date -Is)] report done (RUN_DIR=$RUN_DIR)"

  echo
  echo "--- Phase 4: STT-Server wieder hochfahren ---"
  docker ps -aq --filter name=^/vllm-server$ | xargs -r docker rm -f 2>/dev/null || true
  HOST_PORT=8000 HF_MODELS_DIR="$HOME/hf_models" \
    bash /home/mvdb/dgx-spark-vllm/runner/vllm_spark.sh \
    --model granite-speech-4.1-2b-plus --skip-pull || \
    echo "[$(date -Is)] WARNUNG: STT-Neustart fehlgeschlagen — manuell starten"
  echo "[$(date -Is)] STT-Restart angestoßen"

  echo
  echo "=== unified_v024_run end $(date -Is) ==="
} >> "$LOG" 2>&1
