#!/bin/bash
# retest2_medium.sh — Verifikation der Profil-Fixes (Nemotron nemotron_v3,
# Omni Mamba/max_num_batched_tokens) + GLM-Diagnose (chat vs raw).
set -u
cd /home/mvdb/dgx-spark-vllm
VLLM=runner/vllm_spark.sh
TS="$(date +%Y%m%d_%H%M%S)"
LOG="testplan/logs/retest2_medium_${TS}.log"
RES="testplan/logs/retest2_medium_${TS}.result"
TIMEOUT=600
teardown() { docker ps -aq --filter publish=8000 | xargs -r docker rm -f >/dev/null 2>&1; }

probe() {
  local m="$1"
  echo; echo "######## $m ######## $(date -Is)"
  teardown; sleep 2
  HOST_PORT=8000 bash "$VLLM" --model "$m" --skip-pull >/dev/null 2>&1
  sleep 3
  local cid; cid=$(docker ps -aq --filter publish=8000 | head -1)
  local ready=0 dead=0
  for i in $(seq 1 $((TIMEOUT/5))); do
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { ready=1; break; }
    docker ps -q --filter publish=8000 | grep -q . || { dead=1; break; }
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    echo "  NICHT bereit ($([ "$dead" = 1 ] && echo container-tot || echo timeout))"
    [ -n "$cid" ] && docker logs "$cid" 2>&1 | grep -iE "error|assert|raise|not support|unsupport|invalid|valueerror|choice" | tail -12 | sed 's/^/    /'
    echo "${m}|FAIL|not-ready" >> "$RES"; teardown; return
  fi
  local MID; MID=$(curl -s http://127.0.0.1:8000/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  echo "  CHAT:"
  curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"Antworte kurz: Hauptstadt von Frankreich?\"}],\"max_tokens\":120}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);c=d['choices'][0];m=c['message']
print('    finish=',c.get('finish_reason'),'content=',repr((m.get('content') or '')[:60]),'reasoning=',repr((m.get('reasoning_content') or '')[:40]))" 2>/dev/null || echo "    (chat-fehler)"
  echo "  RAW:"
  curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MID\",\"prompt\":\"Die Hauptstadt von Frankreich ist\",\"max_tokens\":12}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);c=d['choices'][0];print('    finish=',c.get('finish_reason'),'text=',repr((c.get('text') or '')[:40]))" 2>/dev/null || echo "    (raw-fehler)"
  # PASS wenn chat-content nicht leer
  CC=$(curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"Hauptstadt von Frankreich? Ein Wort.\"}],\"max_tokens\":120}" \
    | python3 -c "import sys,json;print((json.load(sys.stdin)['choices'][0]['message'].get('content') or '').strip()[:40])" 2>/dev/null)
  if [ -n "$CC" ]; then echo "${m}|PASS|${CC}" >> "$RES"; else echo "${m}|WARN|chat-content-leer" >> "$RES"; fi
  teardown
}

{
  echo "=== retest2 start $(date -Is) ==="; : > "$RES"
  probe nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
  probe nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
  probe zai-org--GLM-4.7-Flash
  echo; echo "=== ERGEBNIS ==="; cat "$RES"
  echo "=== retest2 end $(date -Is) ==="
} >> "$LOG" 2>&1
