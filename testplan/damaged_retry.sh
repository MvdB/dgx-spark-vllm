#!/bin/bash
# Holt die durch den v2-Build-Vorfall beschädigten/übersprungenen Modelle sauber nach,
# sobald der Hauptlauf die GPU freigibt: Mistral-Small-4 (Image-Fix v0.26.0),
# Nemotron-3-Nano-30B (38% error im Judge-Fenster), Qwen3.6-35B-A3B-FP8 (Judge tot).
# Reports werden in die 1130-Kohorte kopiert, Validität geprüft, Sites gepusht.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
VLLM=/home/mvdb/southbyte/southbyte-vllm
TP=$VLLM/testplan
RESULTS=/home/mvdb/southbyte/southbyte-results
PY=$TP/.venv/bin/python
COH=$TP/reports/2026-08-08_1130
MODELS="Mistral-Small-4,Nemotron-3-Nano-30B,Qwen3.6-35B-A3B-FP8,Mistral-Medium-3.5-128B-NVFP4"
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019YpS7jJGxYt9cD5nRMS7fH'
ML=$TP/damaged_retry.log
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
t=sum(len(p.get("results",[])) for p in (pbs.values() if isinstance(pbs,dict) else pbs))
e=sum(1 for p in (pbs.values() if isinstance(pbs,dict) else pbs) for r in p.get("results",[]) if r.get("verdict")=="error")
print(f"{t} Fälle, {e} error → {'VALIDE' if t and e/t<=0.3 else 'KAPUTT'}")
PY
}

cd "$TP"
log "=== DAMAGED-RETRY: warte auf Hauptlauf-Ende (GPU-Fenster) ==="
for i in $(seq 1 5760); do   # max 48h
  pgrep -f 'orchestrator.py --continue' >/dev/null || break
  sleep 30
done
log "GPU frei — Nachzügler: $MODELS"
$PY -u orchestrator.py --models "$MODELS" --continue-after-ko > "$TP/damaged_retry_run.log" 2>&1
log "orchestrator rc=$?"

# frische Reports in die Kohorte kopieren + Validität prüfen
for M in Mistral-Small-4 Nemotron-3-Nano-30B Qwen3.6-35B-A3B-FP8 Mistral-Medium-3.5-128B-NVFP4; do
  ND=$(ls -1dt "$TP"/reports/2026-*/ 2>/dev/null | while read d; do
         [ "$d" = "$COH/" ] && continue; [ -f "$d$M.json" ] && { echo "$d"; break; }; done)
  if [ -n "$ND" ]; then
    cp "$ND$M.json" "$COH/$M.json"; log "$M ← ${ND}: $(valid "$COH/$M.json")"
  else
    log "$M: KEIN neuer Report (erneut gescheitert? siehe damaged_retry_run.log)"
  fi
done

log "Sites bauen + pushen"
python3 "$TP/make_public_site.py" >>"$ML" 2>&1
push "$VLLM" "Beschädigte Modelle nachgezogen (Mistral-Small-4, Nemotron-Nano-30B, Qwen3.6-35B)" docs testplan/damaged_retry.sh
python3 "$RESULTS/build_site.py" >>"$ML" 2>&1
push "$RESULTS" "Hub: Nachzügler-Modelle ergänzt" build_site.py docs
log "DAMAGED_RETRY_DONE"
log "=== DAMAGED-RETRY ENDE ==="
