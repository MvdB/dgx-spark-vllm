#!/bin/bash
# Ornith-1.0-35B-FP8 mit HF-Card-Recipe (trust-remote-code + qwen3_xml/qwen3-Parser).
# Gegated auf medium_retry2.sh-Ende + freie GPU. Fail-Fast + Container-Log-Capture,
# damit ein evtl. Arch-Fail den ECHTEN Grund liefert (nicht raten).
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
VLLM=/home/mvdb/southbyte/southbyte-vllm
TP=$VLLM/testplan
RESULTS=/home/mvdb/southbyte/southbyte-results
PY=$TP/.venv/bin/python
COH=$TP/reports/2026-08-08_1130
M=Ornith-1.0-35B-FP8
C=vllm-ornith-ai--Ornith-1.0-35B-FP8
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013gePcuZs3qnocpMR9U6rLL'
ML=$TP/ornith_retry.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$ML"; }
push(){ local repo="$1" msg="$2"; shift 2; cd "$repo" || return
  git add "$@" 2>/dev/null
  git commit -q -m "$msg

$CO" 2>/dev/null && log "commit: $msg" || log "nichts zu committen ($repo)"
  git pull --rebase origin main >/dev/null 2>&1 || true
  git push origin main 2>&1 | tail -1 | tee -a "$ML"; }
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
cd "$TP"
log "=== ORNITH-RETRY armiert — warte auf medium_retry2.sh-Ende + freie GPU ==="
for i in $(seq 1 5760); do pgrep -f '[m]edium_retry2\.sh' >/dev/null || break; sleep 30; done
for i in $(seq 1 480); do docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^vllm-' || break; sleep 30; done
log "GPU frei — starte $M (HF-Card-Recipe, Fail-Fast)"

# Hintergrund: Container-Log sichern sobald er stirbt (bevor Fail-Fast ihn entfernt)
( for i in $(seq 1 360); do
    [ -f "$TP/reports_ornith_ok" ] && break
    stx=$(docker ps -a --format '{{.Names}}|{{.Status}}' 2>/dev/null | grep -i 'Ornith-1.0-35B' | head -1)
    case "${stx#*|}" in
      Exited*|Dead*) docker logs "${stx%%|*}" > "$TP/ornith_traceback.log" 2>&1; break;;
    esac
    sleep 10
  done ) &
CAP=$!

$PY -u orchestrator.py --models "$M" --continue-after-ko > "$TP/ornith_run.log" 2>&1
log "orchestrator rc=$?"
touch "$TP/reports_ornith_ok"; kill "$CAP" 2>/dev/null || true

ND=$(ls -1dt "$TP"/reports/2026-08-10_*/ 2>/dev/null | while read d; do
       [ "$d" = "$COH/" ] && continue; [ -f "$d$M.json" ] && { echo "$d"; break; }; done)
if [ -n "$ND" ]; then
  cp "$ND$M.json" "$COH/$M.json"; V=$(valid "$COH/$M.json"); log "$M <- ${ND}: $V"
  # Bei VALIDE: N/A-Markierung in yaml zurücknehmen (active:true, Note ohne N/A)
  if echo "$V" | grep -q VALIDE; then
    sed -i '/name: "Ornith-1.0-35B-FP8"/,/params_b:/ { s/active: false/active: true/; /notes: "N\/A 2026-08-10: Arch qwen3_5_moe/d }' config/testplan.yaml
    log "$M VALIDE → in testplan.yaml reaktiviert (N/A entfernt)"
  fi
else
  log "$M: KEIN frischer Report — erneut gescheitert. Grund siehe ornith_traceback.log + ornith_run.log."
fi
rm -f "$TP/reports_ornith_ok"
log "Sites bauen + pushen"
python3 "$TP/make_public_site.py" >>"$ML" 2>&1
push "$VLLM" "Ornith-1.0-35B-FP8 mit HF-Card-Recipe (trust-remote-code) nachgezogen" docs testplan/config/testplan.yaml
python3 "$RESULTS/build_site.py" >>"$ML" 2>&1
push "$RESULTS" "Hub: Ornith-1.0-35B-FP8 Retry" docs
log "ORNITH_RETRY_DONE"
