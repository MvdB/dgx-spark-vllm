#!/bin/bash
# Die 8 restlichen Frontier-SaaS-Modelle mit weichem $20-Deckel, dann Perf + publish.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
TP=/home/mvdb/southbyte/southbyte-vllm/testplan
VLLM=/home/mvdb/southbyte/southbyte-vllm
RES=/home/mvdb/southbyte/southbyte-results
PY=$TP/.venv/bin/python
NEW8="MiniMax-M3,Qwen3.8-Max,Qwen3.7-Plus,Step-3.7-Flash,Hunyuan-3,MiMo-v2.5-Pro,MiMo-v2.5,Nemotron-3-Ultra-550B"
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019YpS7jJGxYt9cD5nRMS7fH'
L=$TP/saas_add8.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
push(){ local repo="$1" msg="$2"; shift 2; cd "$repo" || return
  git add "$@" 2>/dev/null
  git commit -q -m "$msg

$CO" 2>/dev/null && log "commit: $msg" || log "nichts zu committen ($repo)"
  git pull --rebase origin main >/dev/null 2>&1 || true
  git push origin main 2>&1 | tail -1 | tee -a "$L"; }

cd "$TP"
log "=== SAAS-ADD8 START (Deckel \$30) ==="
$PY -u saas_cohort_run.py --run-id 2026-08-07_saas --models "$NEW8" --max-run-cost 30 >> "$L" 2>&1
log "cohort rc=$?"
grep -E 'Weicher Deckel|key-spend' "$L" | tail -3

log "Perf für alle (inkl. neue, Option B)"
$PY -u saas_perf.py >> "$L" 2>&1

log "Sites bauen + pushen"
python3 "$TP/make_public_site.py" >> "$L" 2>&1
push "$VLLM" "SaaS-Kohorte erweitert (+8 Frontier-Modelle, Deckel 30 USD)" docs testplan/saas_cohort_run.py testplan/saas_perf.py testplan/saas_add8_chain.sh
python3 "$RES/build_site.py" >> "$L" 2>&1
push "$RES" "Hub: SaaS-Kohorte erweitert" build_site.py docs
log "SAAS_ADD8_DONE"
