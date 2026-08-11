#!/bin/bash
# Mistral-Medium-3.5 — 2. Fix: tokenizer_mode=mistral + config/load-format mistral
# (exakt wie valider Zwilling Mistral-Small-4). Behebt Pixtral-MM-Crash UND chat_template-400.
# Gegated auf apertus_retry.sh-Ende + freie GPU. Fail-Fast aktiv. Stale-Report-Schutz.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
VLLM=/home/mvdb/southbyte/southbyte-vllm
TP=$VLLM/testplan
RESULTS=/home/mvdb/southbyte/southbyte-results
PY=$TP/.venv/bin/python
COH=$TP/reports/2026-08-08_1130
M=Mistral-Medium-3.5-128B-NVFP4
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013gePcuZs3qnocpMR9U6rLL'
ML=$TP/medium_retry2.log
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
log "=== MEDIUM-RETRY-2 armiert — warte auf apertus_retry.sh-Ende + freie GPU ==="
for i in $(seq 1 5760); do pgrep -f '[a]pertus_retry\.sh' >/dev/null || break; sleep 30; done
for i in $(seq 1 480); do docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^vllm-' || break; sleep 30; done
log "GPU frei — starte $M mit mistral-Mode (Fail-Fast aktiv)"
$PY -u orchestrator.py --models "$M" --continue-after-ko > "$TP/medium_run2.log" 2>&1
log "orchestrator rc=$?"
# nur Report aus HEUTIGEM Run-Dir akzeptieren (kein stale-Fallback)
ND=$(ls -1dt "$TP"/reports/2026-08-10_*/ 2>/dev/null | while read d; do
       [ "$d" = "$COH/" ] && continue; [ -f "$d$M.json" ] && { echo "$d"; break; }; done)
if [ -n "$ND" ]; then cp "$ND$M.json" "$COH/$M.json"; log "$M <- ${ND}: $(valid "$COH/$M.json")"
else log "$M: KEIN frischer Report (erneut gescheitert — siehe medium_run2.log). Falls wieder KAPUTT → N/A für heute."; fi
log "Sites bauen + pushen"
python3 "$TP/make_public_site.py" >>"$ML" 2>&1
push "$VLLM" "Mistral-Medium-3.5 mit mistral-Mode nachgezogen (2. Fix)" docs testplan/config/testplan.yaml
python3 "$RESULTS/build_site.py" >>"$ML" 2>&1
push "$RESULTS" "Hub: Mistral-Medium-3.5 (mistral-Mode)" docs
log "MEDIUM_RETRY2_DONE"
