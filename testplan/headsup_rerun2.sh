#!/bin/bash
# headsup_rerun2.sh — Re-Run des Nemotron-Heads-ups mit:
#   * NVIDIA-Sampling (temperature=1.0, top_p=0.95 aus generation_config)
#   * Thinking "medium" (enable_thinking + low_effort via chat_template_kwargs)
#   * Degenerations-Guard (Token-Limit ohne Content → Retry, dann FAIL statt
#     geschenktem Refusal-PASS; fing jail-004 / loc-bay-001 im Lauf _1813)
# Playbooks 01–05 (Code mit, da Sampling sich ändert; 06_performance bleibt
# aus dem Originallauf 2026-07-08_0935). Danach consolidate + Heads-up-Report
# + STT-Neustart.
#
# Usage (real): setsid nohup ./headsup_rerun2.sh > /dev/null 2>&1 < /dev/null & disown

set -u
cd /home/mvdb/dgx-spark-vllm/testplan
source .venv/bin/activate
export TARGET_HOST=localhost

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/headsup_rerun2_${TS}.log"

MODELS="Nemotron-Puzzle-75B,Nemotron-3-Super"
PLAYBOOKS="01_quality,02_german_language,03_bias,04_security,05_code"

{
  echo "=== headsup_rerun2 start $(date -Is) ==="
  echo "PID: $$ | Log: $LOG"
  echo "Models: $MODELS | Playbooks: $PLAYBOOKS"
  echo "Config: temp=1.0 top_p=0.95, enable_thinking+low_effort, Degenerations-Guard"
  echo

  echo "--- Phase 1: orchestrator (--continue-after-ko) ---"
  python orchestrator.py --continue-after-ko --models "$MODELS" --playbooks "$PLAYBOOKS"
  echo "[$(date -Is)] orchestrator exit=$?"

  echo
  echo "--- Phase 2: consolidate_reports ---"
  python consolidate_reports.py || true
  echo "[$(date -Is)] consolidate done"

  echo
  echo "--- Phase 3: Heads-up-Report neu generieren ---"
  RUN_DIR="$(ls -1dt reports/2026-* 2>/dev/null | head -1 | xargs -r basename)"
  if [ -n "$RUN_DIR" ]; then
    python nemotron_headsup_report.py --post-run "$RUN_DIR" || \
      echo "[$(date -Is)] WARNUNG: Heads-up-Report fehlgeschlagen (Rohdaten liegen in reports/$RUN_DIR)"
  else
    echo "[$(date -Is)] WARNUNG: kein Run-Verzeichnis gefunden"
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
  echo "=== headsup_rerun2 end $(date -Is) ==="
} >> "$LOG" 2>&1
