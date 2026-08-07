#!/bin/bash
# smoke_and_retest_eager_off.sh — Verify the 4 enforce_eager=0 patches boot,
# then run the testplan over the survivors, then consolidate.
#
# Detached / fail-safe.
# Usage: setsid nohup ./smoke_and_retest_eager_off.sh \
#          > logs/smoke_retest_eager_off_<ts>.log 2>&1 < /dev/null & disown

set -u
RUNNER=$HOME/dgx-spark/dgx-spark-vllm/runner
TESTPLAN=$HOME/dgx-spark/dgx-spark-vllm/testplan
cd "$TESTPLAN"
source .venv/bin/activate

TS="$(date +%Y%m%d_%H%M)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/smoke_retest_eager_off_${TS}.log"

# --- profile dir → testplan model name mapping ---
declare -a CANDIDATES=(
  "ibm-granite--granite-4.1-30b|Granite-4.1-30B"
  "Qwen--Qwen3.6-27B-FP8|Qwen3.6-27B-FP8"
  "nvidia--Gemma-4-31B-IT-NVFP4|Gemma-4-31B-IT-NVFP4"
  "zdy1995love--Mistral-Medium-3.5-128B-NVFP4|Mistral-Medium-3.5-128B-NVFP4"
)

stop_all_vllm() {
  docker ps --format '{{.Names}}' | grep -E '^vllm' | xargs -r -I {} docker stop {} 2>&1 | sed 's/^/  /'
  docker ps -a --format '{{.Names}}' | grep -E '^vllm' | xargs -r -I {} docker rm -f {} 2>&1 | sed 's/^/  /'
}

{
  echo "=== smoke_retest_eager_off start $(date -Is) ==="
  echo "PID: $$"
  echo "Log: $LOG"
  echo

  PASSED=()
  FAILED=()

  for entry in "${CANDIDATES[@]}"; do
    PROFILE_DIR="${entry%%|*}"
    MODEL_NAME="${entry##*|}"
    PATTERN="$(basename "$PROFILE_DIR")"

    echo "===== smoke: $MODEL_NAME ($PROFILE_DIR) ====="
    echo "[$(date -Is)] cleaning any leftover container..."
    stop_all_vllm
    sleep 3

    echo "[$(date -Is)] starting vllm_spark.sh --model '$PATTERN'..."
    (cd "$RUNNER" && CONTAINER_NAME=vllm-server ./vllm_spark.sh --skip-pull --model "$PATTERN") 2>&1 | tail -40

    # Poll readiness for up to 12 minutes (graph capture can take a while on 128B)
    READY=0
    for i in $(seq 1 144); do
      sleep 5
      if curl -sf -o /dev/null -m 3 http://localhost:8000/v1/models 2>/dev/null; then
        READY=1
        break
      fi
      # Detect dead container
      if ! docker ps --format '{{.Names}}' | grep -q '^vllm-server$' 2>/dev/null; then
        if [ $i -gt 6 ]; then  # give it 30s to appear
          echo "[$(date -Is)] container vllm-server vanished"
          break
        fi
      fi
    done

    if [ "$READY" = "1" ]; then
      echo "[$(date -Is)] ✓ READY: $MODEL_NAME"
      curl -s http://localhost:8000/v1/models | head -c 400
      echo
      PASSED+=("$MODEL_NAME")
    else
      echo "[$(date -Is)] ✗ FAILED: $MODEL_NAME — last 80 lines of container log:"
      docker logs vllm-server 2>&1 | tail -80
      FAILED+=("$MODEL_NAME")
    fi

    echo "[$(date -Is)] stopping container..."
    stop_all_vllm
    echo "[$(date -Is)] cooldown 30s..."
    sleep 30
    echo
  done

  echo "===== smoke summary ====="
  echo "PASSED: ${PASSED[*]:-(none)}"
  echo "FAILED: ${FAILED[*]:-(none)}"
  echo

  if [ ${#PASSED[@]} -eq 0 ]; then
    echo "[$(date -Is)] no survivors — skipping retest."
    echo "=== smoke_retest_eager_off done $(date -Is) ==="
    exit 1
  fi

  # Build comma-separated list for orchestrator
  IFS=','
  MODELS_CSV="${PASSED[*]}"
  unset IFS

  echo "===== retest phase: orchestrator --models '$MODELS_CSV' --continue-after-ko ====="
  python orchestrator.py --models "$MODELS_CSV" --continue-after-ko
  ORC_RC=$?
  echo "[$(date -Is)] orchestrator exit=$ORC_RC"

  echo
  echo "===== consolidate ====="
  python consolidate_reports.py
  CONS_RC=$?
  echo "[$(date -Is)] consolidate exit=$CONS_RC"

  echo
  echo "=== smoke_retest_eager_off done $(date -Is) — orchestrator=$ORC_RC consolidate=$CONS_RC ==="
} >> "$LOG" 2>&1
