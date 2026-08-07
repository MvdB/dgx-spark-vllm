#!/bin/bash
# diffusiongemma_run.sh — Nachzügler-Lauf: google/diffusiongemma-26B-A4B-it
# in die 7er-Kohorte (v0.24.0) einfließen lassen.
# Kette: Download abwarten -> STT stoppen -> Smoke (Boot+Bench) ->
#        orchestrator (nur dieses Modell) -> consolidate -> STT-Neustart.
# Kein MTP (Diffusions-Decoder), TRITON_ATTN, max_num_seqs=4.
#
# Usage (real): setsid nohup ./diffusiongemma_run.sh > /dev/null 2>&1 < /dev/null & disown

set -u
cd $HOME/southbyte/southbyte-vllm/testplan
source .venv/bin/activate
export TARGET_HOST=localhost

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/diffusiongemma_${TS}.log"

RUNNER=$HOME/southbyte/southbyte-vllm/runner/vllm_spark.sh
DIR="$HOME/hf_models/google--diffusiongemma-26B-A4B-it"

bench() {
  $HOME/southbyte/southbyte-vllm/testplan/.venv/bin/python - <<'PY'
import time, sys
from openai import OpenAI
c = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='x', timeout=300)
mid = c.models.list().data[0].id
t0=time.monotonic(); first=None; usage=None
s = c.chat.completions.create(model=mid, messages=[{"role":"user","content":"Erkläre in ~200 Wörtern den Unterschied zwischen Prozess und Thread."}],
    max_tokens=400, temperature=0.1, stream=True, stream_options={"include_usage": True})
txt=[]
for ch in s:
    if ch.usage: usage=ch.usage
    if ch.choices and ch.choices[0].delta.content:
        if first is None: first=time.monotonic()
        txt.append(ch.choices[0].delta.content)
dt=time.monotonic()-(first or t0)
tok=usage.completion_tokens if usage else 0
if tok < 50: print("BENCH_FAIL zu wenig Tokens:", tok, "| content:", repr("".join(txt))[:200]); sys.exit(1)
print(f"BENCH_OK {tok/dt:.1f}")
print("SAMPLE:", "".join(txt)[:300].replace(chr(10)," "))
PY
}

restart_stt() {
  docker ps -aq --filter name=^/vllm-server$ | xargs -r docker rm -f 2>/dev/null || true
  HOST_PORT=8000 HF_MODELS_DIR="$HOME/hf_models" \
    bash "$RUNNER" --model granite-speech-4.1-2b-plus --skip-pull || \
    echo "[$(date -Is)] WARNUNG: STT-Neustart fehlgeschlagen — manuell starten"
  echo "[$(date -Is)] STT-Restart angestoßen"
}

{
  echo "=== diffusiongemma_run start $(date -Is) ==="
  echo "PID: $$ | Log: $LOG"

  echo "--- Phase 0: Download abwarten ---"
  done_dl=0
  for i in $(seq 1 480); do
    shards=$(ls "$DIR"/model-*-of-00011.safetensors 2>/dev/null | wc -l)
    incomplete=$(find "$DIR" -name "*.incomplete" -o -name "*.tmp" 2>/dev/null | wc -l)
    if [ "$shards" -eq 11 ] && [ "$incomplete" -eq 0 ]; then
      s1=$(du -sb "$DIR" | cut -f1); sleep 45; s2=$(du -sb "$DIR" | cut -f1)
      if [ "$s1" = "$s2" ]; then done_dl=1; echo "DOWNLOAD_DONE $(date -Is) ($((s2/1024/1024/1024)) GiB)"; break; fi
    fi
    sleep 30
  done
  if [ "$done_dl" != 1 ]; then
    echo "RESULT diffusiongemma FAIL download-timeout"
    echo "=== diffusiongemma_run end $(date -Is) ==="
    exit 1
  fi

  echo "--- Phase 1: STT stoppen, Smoke-Test ---"
  docker ps -aq --filter name=^/vllm-server$ | xargs -r docker rm -f 2>/dev/null || true
  docker rm -f vllm-dgemma-test >/dev/null 2>&1 || true
  CONTAINER_NAME=vllm-dgemma-test HOST_PORT=8000 HF_MODELS_DIR="$HOME/hf_models" \
    bash "$RUNNER" --model diffusiongemma-26B-A4B-it --skip-pull >/dev/null 2>&1 || {
      echo "RESULT diffusiongemma FAIL start-script"; restart_stt; exit 1; }
  ok=0
  for i in $(seq 1 80); do
    if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then ok=1; break; fi
    if ! docker ps -q --filter name=vllm-dgemma-test | grep -q .; then break; fi
    sleep 15
  done
  if [ "$ok" != 1 ]; then
    echo "RESULT diffusiongemma FAIL boot"
    docker logs vllm-dgemma-test 2>&1 | grep -iE "error|raise|unrecognized|not supported|Traceback" | tail -10
    docker rm -f vllm-dgemma-test >/dev/null 2>&1 || true
    restart_stt
    echo "=== diffusiongemma_run end $(date -Is) ==="
    exit 1
  fi
  out="$(bench 2>&1)"
  echo "$out"
  case "$out" in
    BENCH_OK*) echo "SMOKE diffusiongemma OK $(echo "$out" | head -1 | cut -d' ' -f2) tok/s";;
    *) echo "RESULT diffusiongemma FAIL bench"
       docker rm -f vllm-dgemma-test >/dev/null 2>&1 || true
       restart_stt
       echo "=== diffusiongemma_run end $(date -Is) ==="
       exit 1;;
  esac
  docker rm -f vllm-dgemma-test >/dev/null 2>&1 || true

  echo
  echo "--- Phase 2: orchestrator (nur DiffusionGemma-26B-A4B) ---"
  python orchestrator.py --continue-after-ko --models "DiffusionGemma-26B-A4B" \
    --playbooks "01_quality,02_german_language,03_bias,04_security,05_code,06_performance"
  echo "[$(date -Is)] orchestrator exit=$?"

  echo
  echo "--- Phase 3: consolidate_reports ---"
  python consolidate_reports.py || true
  echo "[$(date -Is)] consolidate done"

  echo
  echo "--- Phase 4: STT-Server wieder hochfahren ---"
  restart_stt

  echo
  echo "=== diffusiongemma_run end $(date -Is) ==="
} >> "$LOG" 2>&1
