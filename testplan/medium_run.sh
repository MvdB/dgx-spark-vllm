#!/bin/bash
# medium_run.sh — Mittelklasse-Kohorte (~8–35B) auf vLLM v0.23.0, aktueller
# Testsatz (gefixter Code-Evaluator). Danach Cross-Model-Dashboard.
# Detached / fail-safe.
#
# WICHTIG: Erst NACH bestandenem Smoke-Test je Modell starten. Alle hier
# gelisteten Modelle waren zuvor BEWUSST auf v0.21.0 gepinnt (cluster-A/B-Fixes,
# Qwen3.6 = VL mit conv3d-Patch-Bedarf auf GB10-cuDNN). Fällt ein Modell auf
# v0.23.0 durch den Smoke-Test, zurück auf seinen v0.21.0-Pin und aus MODELS
# entfernen.
#
# Usage (real): setsid nohup ./medium_run.sh > /dev/null 2>&1 < /dev/null & disown

set -u
cd /home/mvdb/dgx-spark-vllm/testplan
source .venv/bin/activate
export TARGET_HOST=localhost

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/medium_run_${TS}.log"

# Alle 9 nach Smoke-Validierung. Reihenfolge klein → groß für frühe Ergebnisse.
MODELS="Qwen3.6-27B-FP8,Gemma-4-26B-A4B,Granite-4.1-30B,Nemotron-3-Nano-30B,Nemotron-3-Nano-Omni-30B,GLM-4.7-Flash,Olmo-3.1-32B-Instruct,Gemma-4-31B,Qwen3.6-35B-A3B-FP8"

{
  echo "=== medium_run start $(date -Is) ==="
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
  echo "=== medium_run end $(date -Is) ==="
} >> "$LOG" 2>&1
