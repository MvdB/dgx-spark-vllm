#!/bin/bash
# apertus_code_rerun.sh — Nur das 05_code-Playbook für die 3 Apertus-Modelle
# erneut laufen lassen (mit gefixtem base.py / max_tokens-Clamp), die Ergebnisse
# in den bestehenden Voll-Report mergen und das Dashboard neu konsolidieren.
#
# Detached / fail-safe.
# Usage (real): setsid nohup ./apertus_code_rerun.sh > /dev/null 2>&1 < /dev/null & disown

set -u
cd $HOME/southbyte/southbyte-vllm/testplan
source .venv/bin/activate
export TARGET_HOST=localhost

TS="$(date +%Y%m%d_%H%M%S)"          # Sekunden-Auflösung → keine Log-Kollision
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/apertus_code_rerun_${TS}.log"

MODELS="Apertus-v1.1-0.5B,Apertus-v1.1-1.5B,Apertus-v1.1-4B"
TARGET_RUN="reports/2026-06-19_1530"   # Voll-Report, in den gemergt wird

{
  echo "=== apertus_code_rerun start $(date -Is) ==="
  echo "PID: $$ | Log: $LOG"
  echo "Models: $MODELS | Playbook: 05_code | Ziel: $TARGET_RUN"
  echo

  # Report-Verzeichnisse vor dem Lauf merken, um das neue zu identifizieren
  BEFORE="$(ls -d reports/*/ 2>/dev/null | sort)"

  echo "--- Phase 1: orchestrator (nur 05_code) ---"
  python orchestrator.py --continue-after-ko --models "$MODELS" --playbooks 05_code
  echo "[$(date -Is)] orchestrator exit=$?"

  AFTER="$(ls -d reports/*/ 2>/dev/null | sort)"
  NEW_DIR="$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | tail -1 | sed 's:/*$::')"
  echo "Neues Code-Report-Verzeichnis: ${NEW_DIR:-<keins>}"

  if [ -z "$NEW_DIR" ]; then
    echo "FEHLER: kein neues Report-Verzeichnis gefunden — Merge übersprungen."
    echo "=== apertus_code_rerun end $(date -Is) ==="
    exit 1
  fi

  echo
  echo "--- Phase 2: merge 05_code → $TARGET_RUN ---"
  python merge_code_results.py "$TARGET_RUN" "$NEW_DIR" --models "$MODELS" --min-quality-pass-rate 0.70

  echo
  echo "--- Phase 3: consolidate_reports ---"
  python consolidate_reports.py || true
  echo "[$(date -Is)] consolidate done"

  echo
  echo "=== apertus_code_rerun end $(date -Is) ==="
} >> "$LOG" 2>&1
