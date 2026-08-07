#!/bin/bash
# v020_smoke.sh — Smoke-Test: 4 Problem-Modelle gegen vllm/vllm-openai:v0.20.0
# Detached-fähig (per nohup). Jeder Modell-Slot ist failsafe (set +e), Reihenfolge bleibt erhalten.

set -u

IMAGE='vllm/vllm-openai:v0.20.0'
HF_DIR='$HOME/hf_models'
PORT=8000
CONTAINER='v020-smoke'
LOG_DIR='$HOME/dgx-spark/dgx-spark-vllm/testplan/logs'
TS="$(date +%Y%m%d_%H%M)"
MASTER_LOG="${LOG_DIR}/v020_smoke_${TS}.log"
RESULT_TABLE="${LOG_DIR}/v020_smoke_${TS}_results.txt"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=== v020_smoke started $(date) ==="
echo "Image: $IMAGE"
echo "Master log: $MASTER_LOG"
echo "Result table: $RESULT_TABLE"
echo

# ── Hilfsfunktionen ─────────────────────────────────────────────────────────
cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_ready() {
  local timeout="$1"; local label="$2"
  local start=$(date +%s)
  while true; do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" -o /dev/null 2>/dev/null; then
      echo "  [$(date +%T)] $label ready after $(( $(date +%s) - start ))s"
      return 0
    fi
    if (( $(date +%s) - start >= timeout )); then
      echo "  [$(date +%T)] $label TIMEOUT after ${timeout}s"
      return 1
    fi
    # Fail fast wenn Container weg
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
      echo "  [$(date +%T)] $label CONTAINER EXITED"
      return 2
    fi
    sleep 10
  done
}

smoke_chat() {
  local prompt="$1"
  local resp
  resp=$(curl -sf -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys; print(json.dumps({'model':'/model','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':32,'temperature':0.0}))" "$prompt")" \
    2>&1)
  if [[ -z "$resp" ]]; then
    echo "  Smoke: NO RESPONSE"
    return 1
  fi
  local content
  content=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:120])" 2>/dev/null || echo "")
  if [[ -n "$content" ]]; then
    echo "  Smoke: PASS — \"$content\""
    return 0
  else
    echo "  Smoke: PARSE FAIL — $(echo "$resp" | head -c 200)"
    return 1
  fi
}

run_test() {
  local name="$1"; shift
  local model_dir="$1"; shift
  local timeout="$1"; shift
  local docker_args=("$@")

  echo
  echo "======================================================================"
  echo "[$(date)] Test: $name"
  echo "======================================================================"
  cleanup

  local model_log="${LOG_DIR}/v020_smoke_${TS}_${name}.log"
  echo "  Model dir: $model_dir"
  echo "  Container log: $model_log"

  if [[ ! -d "$model_dir" ]]; then
    echo "  SKIP: model dir nicht vorhanden"
    echo "$name | SKIP | model dir missing" >> "$RESULT_TABLE"
    return
  fi

  echo "  Starte Container ..."
  docker run -d \
    --name "$CONTAINER" \
    --gpus all \
    --ipc=host \
    -p "${PORT}:8000" \
    -v "${model_dir}:/model:ro" \
    "${docker_args[@]}" >/dev/null
  if (( $? != 0 )); then
    echo "  docker run FAIL"
    echo "$name | FAIL | docker run failed" >> "$RESULT_TABLE"
    return
  fi

  # Logs im Hintergrund mitschneiden
  ( docker logs -f "$CONTAINER" > "$model_log" 2>&1 ) &
  local logpid=$!

  echo "  Warte auf Readiness (max ${timeout}s) ..."
  if wait_ready "$timeout" "$name"; then
    if smoke_chat "Sag bitte exakt: HALLO_${name}"; then
      echo "$name | PASS | started+chat ok" >> "$RESULT_TABLE"
    else
      echo "$name | PARTIAL | started but chat failed" >> "$RESULT_TABLE"
    fi
  else
    echo "  Letzte 40 Logzeilen:"
    tail -40 "$model_log" | sed 's/^/    /'
    echo "$name | FAIL | not ready after ${timeout}s" >> "$RESULT_TABLE"
  fi

  echo "  Stoppe Container ..."
  cleanup
  kill "$logpid" 2>/dev/null || true
  echo "  GPU-Cooldown 30s ..."
  sleep 30
}

# ── Test-Slots ──────────────────────────────────────────────────────────────

# Slot 1: Qwen3.5-9B GPTQ-INT4 — Registry-Bug-Test (text-only)
run_test "qwen35-9b-gptq" \
  "${HF_DIR}/mssfj--Qwen3.5-9B-GPTQ-INT4" \
  1800 \
  "$IMAGE" \
  /model \
  --quantization gptq_marlin \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching

# Slot 2: Gemma-4-31B-IT-NVFP4 — NVFP4 native (kein Emulations-Backend mehr)
run_test "gemma4-31b-nvfp4" \
  "${HF_DIR}/nvidia--Gemma-4-31B-IT-NVFP4" \
  1800 \
  "$IMAGE" \
  /model \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching

# Slot 3: Mistral-Small-3.2-24B-NVFP4 — NVFP4 native + Mistral tokenizer
run_test "mistral-24b-nvfp4" \
  "${HF_DIR}/RedHatAI--Mistral-Small-3.2-24B-Instruct-2506-NVFP4" \
  1800 \
  "$IMAGE" \
  /model \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --tokenizer-mode mistral \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --enable-prefix-caching

# Slot 4: Nemotron-3-Nano-Omni-30B — Multimodal MoE NVFP4 + audio install
# Spezialfall: bash-wrapper + pip install vllm[audio]
echo
echo "======================================================================"
echo "[$(date)] Test: nemotron-omni-30b (special: pip install vllm[audio])"
echo "======================================================================"
cleanup
NEMO_LOG="${LOG_DIR}/v020_smoke_${TS}_nemotron-omni.log"
NEMO_DIR="${HF_DIR}/nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
echo "  Model dir: $NEMO_DIR"
echo "  Container log: $NEMO_LOG"

if [[ ! -d "$NEMO_DIR" ]]; then
  echo "  SKIP: model dir nicht vorhanden"
  echo "nemotron-omni-30b | SKIP | model dir missing" >> "$RESULT_TABLE"
else
  docker run -d \
    --name "$CONTAINER" \
    --gpus all \
    --ipc=host \
    -p "${PORT}:8000" \
    -v "${NEMO_DIR}:/model:ro" \
    --entrypoint /bin/bash \
    "$IMAGE" -c "pip install vllm[audio] && vllm serve /model \
      --served-model-name=nemotron_3_nano_omni \
      --max-num-seqs 8 \
      --max-model-len 131072 \
      --port 8000 \
      --trust-remote-code \
      --gpu-memory-utilization 0.8 \
      --limit-mm-per-prompt '{\"video\": 1, \"image\": 1, \"audio\": 1}' \
      --media-io-kwargs '{\"video\": {\"fps\": 2,  \"num_frames\": 256}}' \
      --allowed-local-media-path=/ \
      --enable-prefix-caching \
      --max-num-batched-tokens 32768 \
      --reasoning-parser nemotron_v3 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder" >/dev/null

  ( docker logs -f "$CONTAINER" > "$NEMO_LOG" 2>&1 ) &
  NEMO_LOGPID=$!

  # +600s zusätzlich für pip install vllm[audio]
  echo "  Warte auf Readiness (max 2400s — inkl. pip install) ..."
  if wait_ready 2400 "nemotron-omni-30b"; then
    # Smoke-Chat mit served-model-name
    resp=$(curl -sf -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"nemotron_3_nano_omni","messages":[{"role":"user","content":"Sag bitte exakt: HALLO_nemotron"}],"max_tokens":32,"temperature":0.0}' 2>&1)
    content=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:120])" 2>/dev/null || echo "")
    if [[ -n "$content" ]]; then
      echo "  Smoke: PASS — \"$content\""
      echo "nemotron-omni-30b | PASS | started+chat ok" >> "$RESULT_TABLE"
    else
      echo "  Smoke: PARSE FAIL — $(echo "$resp" | head -c 200)"
      echo "nemotron-omni-30b | PARTIAL | started but chat failed" >> "$RESULT_TABLE"
    fi
  else
    echo "  Letzte 40 Logzeilen:"
    tail -40 "$NEMO_LOG" | sed 's/^/    /'
    echo "nemotron-omni-30b | FAIL | not ready" >> "$RESULT_TABLE"
  fi

  cleanup
  kill "$NEMO_LOGPID" 2>/dev/null || true
  sleep 30
fi

# ── Zusammenfassung ─────────────────────────────────────────────────────────
echo
echo "======================================================================"
echo "Ergebnisse:"
echo "======================================================================"
if [[ -f "$RESULT_TABLE" ]]; then
  cat "$RESULT_TABLE"
else
  echo "(keine Ergebnisse)"
fi
echo
echo "=== v020_smoke fertig $(date) ==="
