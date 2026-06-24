#!/bin/bash
# smoke_medium.sh — Smoke-Test der Mittelklasse-Kandidaten auf vLLM v0.23.0.
# Je Modell: Container starten, /v1/models pollen, 1 Completion, Teardown.
# Schreibt ein Ergebnisprotokoll (PASS/FAIL + Grund). KEINE Judge-Tokens.
# Detached / fail-safe.

set -u
cd /home/mvdb/dgx-spark-vllm
VLLM=runner/vllm_spark.sh

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="testplan/logs"; mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/smoke_medium_${TS}.log"
RES="${LOG_DIR}/smoke_medium_${TS}.result"

# Profil-Verzeichnisnamen (vllm_spark.sh --model <dir>)
MODELS=(
  Qwen--Qwen3.6-27B-FP8
  google--gemma-4-26B-A4B-it
  ibm-granite--granite-4.1-30b
  nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
  allenai--Olmo-3.1-32B-Instruct
  google--gemma-4-31B-it
  Qwen--Qwen3.6-35B-A3B-FP8
  nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
)

TIMEOUT=480   # s pro Modell bis /v1/models

teardown() { docker ps -aq --filter publish=8000 | xargs -r docker rm -f >/dev/null 2>&1; }

{
  echo "=== smoke_medium start $(date -Is) ==="
  echo "PID: $$ | Log: $LOG | Result: $RES"
  echo "Modelle: ${#MODELS[@]} | Image: v0.23.0 | Timeout: ${TIMEOUT}s"
  : > "$RES"

  for m in "${MODELS[@]}"; do
    echo; echo "######## $m ########  $(date -Is)"
    teardown; sleep 2

    if ! HOST_PORT=8000 bash "$VLLM" --model "$m" --skip-pull >/dev/null 2>&1; then
      echo "  Start-Kommando fehlgeschlagen"
      echo "${m}|FAIL|start-cmd" >> "$RES"; continue
    fi

    ready=0; dead=0
    for i in $(seq 1 $((TIMEOUT/5))); do
      if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then ready=1; break; fi
      if ! docker ps -q --filter publish=8000 | grep -q .; then dead=1; break; fi
      sleep 5
    done

    if [ "$ready" != 1 ]; then
      reason=$([ "$dead" = 1 ] && echo "container-tot" || echo "timeout-${TIMEOUT}s")
      echo "  NICHT bereit ($reason) — letzte Logzeilen:"
      docker logs --tail 25 $(docker ps -aq --filter publish=8000 | head -1) 2>&1 | sed 's/^/    /' | tail -25
      echo "${m}|FAIL|${reason}" >> "$RES"; teardown; continue
    fi

    MID=$(curl -s http://127.0.0.1:8000/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    ANS=$(curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"Antworte in genau einem Wort: Hauptstadt von Frankreich?\"}],\"max_tokens\":20}" \
      | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'].strip().replace(chr(10),' '))" 2>/dev/null)

    if [ -n "$ANS" ]; then
      echo "  PASS — id=$MID — Antwort: $ANS"
      echo "${m}|PASS|${ANS:0:40}" >> "$RES"
    else
      echo "  Completion fehlgeschlagen (Modell lädt, aber /chat/completions liefert nichts)"
      echo "${m}|WARN|loads-no-completion" >> "$RES"
    fi
    teardown
  done

  echo; echo "=== ERGEBNIS ==="
  cat "$RES"
  echo "=== smoke_medium end $(date -Is) ==="
} >> "$LOG" 2>&1
