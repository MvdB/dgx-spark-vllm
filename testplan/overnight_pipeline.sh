#!/bin/bash
# Losgelöste Rest-Pipeline (setsid nohup) — läuft unabhängig von der Claude-Session.
# 1) 6 rote SaaS fertigstellen · 2) none-Patch · 3) Sites push · 4) Judge→sonnet-5
# 5) vLLM-Orchestrator (21 aktive) · 6) Sites push mit lokalen Ergebnissen.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
VLLM=/home/mvdb/southbyte/southbyte-vllm
TP=$VLLM/testplan
RESULTS=/home/mvdb/southbyte/southbyte-results
PY=$TP/.venv/bin/python
RUN=2026-08-07_saas
RED="Grok-4.5,Grok-4.1-Fast,Mistral-Large,Mistral-Medium,Magistral-Medium,Ministral-8B"
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019YpS7jJGxYt9cD5nRMS7fH'

log(){ echo "[$(date '+%F %T')] $*"; }
push(){ # $1=repo $2=msg $3..=paths
  local repo="$1" msg="$2"; shift 2
  cd "$repo" || return
  git add "$@" 2>/dev/null
  git commit -q -m "$msg

$CO" 2>/dev/null && log "commit: $msg" || log "nichts zu committen ($repo)"
  git pull --rebase origin main >/dev/null 2>&1 || true
  git push origin main 2>&1 | tail -1
}

cd "$TP" || exit 1
log "=== OVERNIGHT-PIPELINE START ==="

# ── Phase 1: 6 rote SaaS fertigstellen (idempotent) ─────────────────────────
log "Phase 1: SaaS red-Run vervollständigen"
for a in 1 2 3 4; do
  INC=$($PY - <<PYEOF
import sys; sys.path.insert(0,"$TP")
from complete_cohort import err_rate
print(",".join(n for n in "$RED".split(",") if err_rate(n)>0.3))
PYEOF
)
  if [ -z "$INC" ]; then log "alle 6 rot vollständig"; break; fi
  log "Versuch $a — offen: $INC"
  $PY -u "$TP/saas_cohort_run.py" --run-id "$RUN" --models "$INC" --max-run-cost 0
done

# ── Phase 2: none-Patch der 6 roten ─────────────────────────────────────────
log "Phase 2: none-Patch (Budget 16384)"
$PY -u "$TP/saas_patch_cases.py" --none --max-tokens 16384 --models "$RED"

# ── Phase 3: Sites rebuild + push (alle 16 SaaS) ────────────────────────────
log "Phase 3: Sites (SaaS komplett)"
python3 "$TP/make_public_site.py"
push "$VLLM" "SaaS-Kohorte komplett (16) + Fall-Patches" docs testplan/make_public_site.py testplan/license_cache.json testplan/overnight_pipeline.sh
python3 "$RESULTS/build_site.py"
push "$RESULTS" "Hub: SaaS-Kohorte komplett" build_site.py docs

# ── Phase 4: Judge auf sonnet-5 ─────────────────────────────────────────────
log "Phase 4: JUDGE_MODEL=claude-sonnet-5"
sed -i 's|^JUDGE_MODEL=.*|JUDGE_MODEL=claude-sonnet-5|' "$TP/.env"
grep -E '^JUDGE_MODEL=' "$TP/.env"

# ── Phase 5: vLLM-Orchestrator (21 aktive) ──────────────────────────────────
log "Phase 5: vLLM-Orchestrator START (das dauert Stunden)"
cd "$TP"
$PY -u orchestrator.py 2>&1
log "Phase 5: Orchestrator FERTIG (rc=$?)"

# ── Phase 6: Sites rebuild + push mit lokalen Ergebnissen ───────────────────
log "Phase 6: Sites mit lokalen vLLM-Ergebnissen"
python3 "$TP/make_public_site.py"
push "$VLLM" "vLLM-Lauf lokal: Ergebnisse publiziert" docs testplan/license_cache.json
python3 "$RESULTS/build_site.py"
push "$RESULTS" "Hub: lokale vLLM-Ergebnisse" build_site.py docs

log "=== OVERNIGHT-PIPELINE FERTIG ==="
