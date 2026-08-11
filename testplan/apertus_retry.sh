#!/bin/bash
# Apertus-v1.5-8B + -70B erneut, mit BASH_WRAPPER=1 (Image-Entrypoint-Fix, exit 126).
# Gegated auf medium_retry.sh-Ende + freie GPU. Neuer Orchestrator = Fail-Fast.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
VLLM=/home/mvdb/southbyte/southbyte-vllm
TP=$VLLM/testplan
RESULTS=/home/mvdb/southbyte/southbyte-results
PY=$TP/.venv/bin/python
COH=$TP/reports/2026-08-08_1130
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013gePcuZs3qnocpMR9U6rLL'
ML=$TP/apertus_retry.log
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
log "=== APERTUS-RETRY armiert — warte auf medium_retry.sh-Ende + freie GPU ==="
for i in $(seq 1 5760); do pgrep -f '[m]edium_retry\.sh' >/dev/null || break; sleep 30; done
for i in $(seq 1 480); do docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^vllm-' || break; sleep 30; done
log "GPU frei — Apertus 8B + 70B (BASH_WRAPPER-Fix, Fail-Fast)"
$PY -u orchestrator.py --models "Apertus-v1.5-8B,Apertus-v1.5-70B" --continue-after-ko > "$TP/apertus_run2.log" 2>&1
log "orchestrator rc=$?"
for M in Apertus-v1.5-8B Apertus-v1.5-70B; do
  ND=$(ls -1dt "$TP"/reports/2026-*/ 2>/dev/null | while read d; do
         [ "$d" = "$COH/" ] && continue; [ -f "$d$M.json" ] && { echo "$d"; break; }; done)
  # nur Reports aus HEUTIGEM Run akzeptieren (kein stale-Fallback wie beim Medium-Vorfall)
  if [ -n "$ND" ] && [[ "$ND" == *"/reports/2026-08-10_"* ]]; then
    cp "$ND$M.json" "$COH/$M.json"; log "$M <- ${ND}: $(valid "$COH/$M.json")"
  else log "$M: KEIN frischer Report (Crash? siehe apertus_run2.log) — N/A-Kandidat"; fi
done
log "Sites bauen + pushen"
python3 "$TP/make_public_site.py" >>"$ML" 2>&1
push "$VLLM" "Apertus-v1.5 mit Entrypoint-Fix nachgezogen" docs
python3 "$RESULTS/build_site.py" >>"$ML" 2>&1
push "$RESULTS" "Hub: Apertus-v1.5 Entrypoint-Fix" docs
log "APERTUS_RETRY_DONE"
