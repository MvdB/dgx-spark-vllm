#!/bin/bash
# 2026-08-11 Fix-Kette v2: Ornith (max_num_batched_tokens 4096) → Medium (0.80/8k) →
# Apertus-70B. Diesmal ZUVERLÄSSIGER Live-Log-Capture: alte exited-Container weg, dann
# `docker logs -f` auf den LAUFENDEN Container streamen (kein stale-Grab mehr).
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
VLLM=/home/mvdb/southbyte/southbyte-vllm
TP=$VLLM/testplan
RESULTS=/home/mvdb/southbyte/southbyte-results
PY=$TP/.venv/bin/python
COH=$TP/reports/2026-08-08_1130
TODAY=2026-08-11
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017ZReBk3tV93kPjrGwHWc6i'
ML=$TP/fix_chain_0811b.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$ML"; }
valid(){ $PY - "$1" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); pbs=d.get("playbooks",{})
it=pbs.values() if isinstance(pbs,dict) else pbs
t=sum(len(p.get("results",[])) for p in it)
it=pbs.values() if isinstance(pbs,dict) else pbs
e=sum(1 for p in it for r in p.get("results",[]) if r.get("verdict")=="error")
print(f"{t} Faelle, {e} error -> {'VALIDE' if t and e/t<=0.3 else 'KAPUTT'}")
PY
}
run_model(){
  local M="$1" CSUB="$2" TB="$3"
  log "=== $M starten (Live-Log → $TB) ==="
  # alte exited/dead Container mit diesem Substring wegräumen (verhindert stale-Grab & Namenskonflikt)
  for c in $(docker ps -a --format '{{.Names}}|{{.Status}}' 2>/dev/null | grep -i "$CSUB" | grep -iE 'Exited|Dead|Created' | cut -d'|' -f1); do
    docker rm -f "$c" >/dev/null 2>&1 && log "alten Container $c entfernt"; done
  : > "$TP/$TB"
  # Live-Capture: warte auf LAUFENDEN Container, dann logs -f streamen bis er stirbt
  ( for i in $(seq 1 240); do
      [ -f "$TP/.chain_done_$M" ] && break
      rn=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -i "$CSUB" | head -1)
      [ -n "$rn" ] && { docker logs -f "$rn" >> "$TP/$TB" 2>&1; break; }
      sleep 5
    done ) &
  local CAP=$!
  $PY -u orchestrator.py --models "$M" --continue-after-ko > "$TP/${TB%.log}_run.log" 2>&1
  log "$M orchestrator rc=$?"
  touch "$TP/.chain_done_$M"; kill "$CAP" 2>/dev/null || true
  local ND
  ND=$(ls -1dt "$TP"/reports/${TODAY}_*/ 2>/dev/null | while read d; do
         [ "$d" = "$COH/" ] && continue; [ -f "$d$M.json" ] && { echo "$d"; break; }; done)
  if [ -n "$ND" ]; then
    cp "$ND$M.json" "$COH/$M.json"; local V; V=$(valid "$COH/$M.json"); log "$M <- ${ND}: $V"
    if echo "$V" | grep -q VALIDE; then
      sed -i "/name: \"$M\"/,/params_b:/ { s/active: false/active: true/; /notes: \"N\/A/d }" config/testplan.yaml
      log "$M VALIDE → in testplan.yaml reaktiviert"
    fi
  else
    log "$M: KEIN frischer Report — gescheitert. Echter Grund → $TP/$TB (Live-Log)"
    log "$M kritische Zeilen:"; grep -iE 'AssertionError|out of memory|OOM|No available|not enough|CUDA error|RuntimeError|ValueError|Error:|EngineCore failed|GiB' "$TP/$TB" 2>/dev/null | grep -viE 'observability|metrics|jit_monitor' | tail -6 | tee -a "$ML"
  fi
  rm -f "$TP/.chain_done_$M"
}
push(){ local repo="$1" msg="$2"; shift 2; cd "$repo" || return
  git add "$@" 2>/dev/null
  git commit -q -m "$msg

$CO" 2>/dev/null && log "commit: $msg" || log "nichts zu committen ($repo)"
  git pull --rebase origin main >/dev/null 2>&1 || true
  git push origin main 2>&1 | tail -1 | tee -a "$ML"; }

cd "$TP"
for i in $(seq 1 480); do docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^vllm-' || break; sleep 30; done
log "########## FIX-KETTE 2026-08-11 v2 START ##########"

run_model "Ornith-1.0-35B-FP8"            "Ornith-1.0-35B"   ornith3_traceback.log
run_model "Mistral-Medium-3.5-128B-NVFP4" "Mistral-Medium"   medium4_traceback.log
run_model "Apertus-v1.5-70B"              "Apertus-v1.5-70B" apertus70b2_traceback.log

log "Sites bauen + pushen"
python3 "$TP/make_public_site.py" >>"$ML" 2>&1
push "$VLLM" "Fix-Kette v2 2026-08-11: Ornith (max_num_batched_tokens 4096), Medium (0.80/8k), Apertus-70B" docs testplan/config/testplan.yaml
python3 "$RESULTS/build_site.py" >>"$ML" 2>&1
push "$RESULTS" "Hub: Fix-Kette v2 2026-08-11" docs
log "########## FIX_CHAIN_0811b_DONE ##########"
