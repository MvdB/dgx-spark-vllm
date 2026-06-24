#!/bin/bash
# retest_medium.sh — gezielter Nachtest: Nemotron-30B (mehr Tokens, Reasoner?),
# Omni-FP8 (Todesursache), GLM-4.7-Flash (neu fertig). Fail-safe/detached.
set -u
cd /home/mvdb/dgx-spark-vllm
VLLM=runner/vllm_spark.sh
TS="$(date +%Y%m%d_%H%M%S)"
LOG="testplan/logs/retest_medium_${TS}.log"
RES="testplan/logs/retest_medium_${TS}.result"
TIMEOUT=600

teardown() { docker ps -aq --filter publish=8000 | xargs -r docker rm -f >/dev/null 2>&1; }

# $1=profil-dir  $2=max_tokens
start_and_test() {
  local m="$1" mt="$2"
  echo; echo "######## $m (max_tokens=$mt) ######## $(date -Is)"
  teardown; sleep 2
  HOST_PORT=8000 bash "$VLLM" --model "$m" --skip-pull >/dev/null 2>&1
  sleep 3
  local cid; cid=$(docker ps -aq --filter publish=8000 | head -1)
  echo "  container=$cid"
  local ready=0 dead=0
  for i in $(seq 1 $((TIMEOUT/5))); do
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { ready=1; break; }
    docker ps -q --filter publish=8000 | grep -q . || { dead=1; break; }
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    echo "  NICHT bereit ($([ "$dead" = 1 ] && echo container-tot || echo timeout))"
    echo "  --- volle Logs (Fehlersuche) ---"
    [ -n "$cid" ] && docker logs "$cid" 2>&1 | grep -iE "error|trace|raise|not support|unsupport|assert|fail|exception|valueerror|keyerror" | tail -20 | sed 's/^/    /'
    echo "${m}|FAIL|not-ready" >> "$RES"; teardown; return
  fi
  local MID ANS REAS
  MID=$(curl -s http://127.0.0.1:8000/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  read -r ANS REAS < <(curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"Antworte kurz: Hauptstadt von Frankreich?\"}],\"max_tokens\":$mt}" \
    | python3 -c "import sys,json
d=json.load(sys.stdin);m=d['choices'][0]['message']
c=(m.get('content') or '').strip().replace(chr(10),' ')
r=(m.get('reasoning_content') or '').strip().replace(chr(10),' ')
print((c[:50] or 'LEER'), '||', ('reasoning:'+r[:40]) if r else 'kein-reasoning')" 2>/dev/null)
  echo "  content='$ANS'  $REAS"
  if [ "$ANS" != "LEER" ] && [ -n "$ANS" ]; then echo "${m}|PASS|${ANS:0:40}" >> "$RES"
  else echo "${m}|WARN|content-leer ${REAS}" >> "$RES"; fi
  teardown
}

{
  echo "=== retest_medium start $(date -Is) ==="; : > "$RES"
  start_and_test nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 250
  start_and_test nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 50
  start_and_test zai-org--GLM-4.7-Flash 50
  echo; echo "=== ERGEBNIS ==="; cat "$RES"
  echo "=== retest_medium end $(date -Is) ==="
} >> "$LOG" 2>&1
