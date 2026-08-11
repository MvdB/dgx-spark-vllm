#!/bin/bash
# 2026-08-11 Restkette v3: Medium (load_format=auto gegen Lade-OOM) → Apertus-70B.
# WICHTIG: publish NACH JEDEM Modell (host-OOM-resilient — Ornith ist schon live, Medium/Apertus
# sollen ihr Ergebnis sofort sichern). Live-Log-Capture, Fail-Fast, frischer-Report-Schutz.
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
ML=$TP/fix_chain_0811c.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$ML"; }
valid(){ $PY - "$1" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); pbs=d.get("playbooks",{})
it=list(pbs.values()) if isinstance(pbs,dict) else pbs
t=sum(len(p.get("results",[])) for p in it)
e=sum(1 for p in it for r in p.get("results",[]) if r.get("verdict")=="error")
print(f"{t} Faelle, {e} error -> {'VALIDE' if t and e/t<=0.3 else 'KAPUTT'}")
PY
}
push_sites(){ local why="$1"
  log "Publish ($why): Sites bauen + pushen"
  python3 "$TP/make_public_site.py" >>"$ML" 2>&1
  python3 "$RESULTS/build_site.py"  >>"$ML" 2>&1
  cd "$VLLM"; git add docs testplan/config/testplan.yaml 2>/dev/null
  git commit -q -m "$why

$CO" 2>/dev/null && log "vllm commit: $why" || log "vllm nichts zu committen"
  git pull --rebase origin main >/dev/null 2>&1 || true; git push origin main 2>&1 | tail -1 | tee -a "$ML"
  cd "$RESULTS"; git add docs 2>/dev/null
  git commit -q -m "Hub: $why

$CO" 2>/dev/null && log "results commit" || log "results nichts zu committen"
  git pull --rebase origin main >/dev/null 2>&1 || true; git push origin main 2>&1 | tail -1 | tee -a "$ML"
  cd "$TP"
}
run_model(){
  local M="$1" CSUB="$2" TB="$3"
  log "=== $M starten (Live-Log → $TB) ==="
  for c in $(docker ps -a --format '{{.Names}}|{{.Status}}' 2>/dev/null | grep -i "$CSUB" | grep -iE 'Exited|Dead|Created' | cut -d'|' -f1); do
    docker rm -f "$c" >/dev/null 2>&1 && log "alten Container $c entfernt"; done
  : > "$TP/$TB"
  ( for i in $(seq 1 360); do
      [ -f "$TP/.chainc_done_$M" ] && break
      rn=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -i "$CSUB" | head -1)
      [ -n "$rn" ] && { docker logs -f "$rn" >> "$TP/$TB" 2>&1; break; }
      sleep 5
    done ) &
  local CAP=$!
  $PY -u orchestrator.py --models "$M" --continue-after-ko > "$TP/${TB%.log}_run.log" 2>&1
  log "$M orchestrator rc=$?"
  touch "$TP/.chainc_done_$M"; kill "$CAP" 2>/dev/null || true
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
    log "$M kritische Zeilen:"; grep -iE 'Loading model|AssertionError|out of memory|OOM|No available|not enough|CUDA error|RuntimeError|ValueError|Error:|EngineCore failed|GiB|Killed' "$TP/$TB" 2>/dev/null | grep -viE 'observability|metrics|jit_monitor' | tail -8 | tee -a "$ML"
  fi
  rm -f "$TP/.chainc_done_$M"
}

cd "$TP"
for i in $(seq 1 480); do docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^vllm-' || break; sleep 30; done
log "########## RESTKETTE 2026-08-11 v3 START ##########"

run_model "Mistral-Medium-3.5-128B-NVFP4" "Mistral-Medium"   medium5_traceback.log
push_sites "Mistral-Medium-3.5: load_format=auto Versuch (Lade-OOM-Fix)"

run_model "Apertus-v1.5-70B"              "Apertus-v1.5-70B" apertus70b3_traceback.log
push_sites "Apertus-v1.5-70B: finaler Versuch"

log "########## FIX_CHAIN_0811c_DONE ##########"
