#!/bin/bash
# Recovery nach Overnight-Stall (damaged_retry wedged, GPU 11h idle). Hauptlauf ist
# NACHWEISLICH fertig (Testplan abgeschlossen 05:01) → KEINE Wait-Loop mehr, direkt die
# 4 reparierbaren Nachzügler fahren, in Kohorte kopieren, validieren, publizieren.
# Arch-Fails (Apertus-v1.5-8B/70B=apertus1p5, Ornith-35B=qwen3_5_moe) sind NICHT dabei —
# die brauchen ein neueres vLLM-Image (User-Entscheid). GLM-4.7-Flash separat.
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
ML=$TP/recover_retry.log
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
# Sicherheitscheck: kein anderer Orchestrator aktiv (GPU frei)
if pgrep -f '[o]rchestrator\.py --' >/dev/null; then log "ABBRUCH: orchestrator.py läuft noch — GPU nicht frei."; exit 1; fi
log "=== RECOVERY-RETRY START (4 Modelle, GPU frei) ==="
$PY -u orchestrator.py --models "$MODELS" --continue-after-ko > "$TP/recover_retry_run.log" 2>&1
log "orchestrator rc=$?"

for M in Mistral-Small-4 Nemotron-3-Nano-30B Qwen3.6-35B-A3B-FP8 Mistral-Medium-3.5-128B-NVFP4; do
  ND=$(ls -1dt "$TP"/reports/2026-*/ 2>/dev/null | while read d; do
         [ "$d" = "$COH/" ] && continue; [ -f "$d$M.json" ] && { echo "$d"; break; }; done)
  if [ -n "$ND" ]; then cp "$ND$M.json" "$COH/$M.json"; log "$M ← ${ND}: $(valid "$COH/$M.json")"
  else log "$M: KEIN neuer Report (erneut gescheitert? siehe recover_retry_run.log)"; fi
done

log "Sites bauen + pushen"
python3 "$TP/make_public_site.py" >>"$ML" 2>&1
push "$VLLM" "Recovery: 4 Nachzügler nachgezogen (Mistral-Small-4, Nemotron-Nano-30B, Qwen3.6-35B, Mistral-Medium-3.5)" docs testplan/license_cache.json
python3 "$RESULTS/build_site.py" >>"$ML" 2>&1
push "$RESULTS" "Hub: Recovery-Nachzügler ergänzt" build_site.py docs
log "RECOVERY_RETRY_DONE"
