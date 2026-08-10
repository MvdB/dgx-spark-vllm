#!/bin/bash
# Smoke war grün (verifiziert). Direkt: 21 aktive Modelle lokal → Sites → push.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
VLLM=/home/mvdb/southbyte/southbyte-vllm
TP=$VLLM/testplan
RESULTS=/home/mvdb/southbyte/southbyte-results
PY=$TP/.venv/bin/python
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019YpS7jJGxYt9cD5nRMS7fH'
NL=$TP/night2.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$NL"; }
push(){ local repo="$1" msg="$2"; shift 2; cd "$repo" || return
  git add "$@" 2>/dev/null
  git commit -q -m "$msg

$CO" 2>/dev/null && log "commit: $msg" || log "nichts zu committen ($repo)"
  git pull --rebase origin main >/dev/null 2>&1 || true
  git push origin main 2>&1 | tail -1 | tee -a "$NL"
}

cd "$TP" || exit 1
log "=== RUN21-CHAIN START (Smoke grün verifiziert) ==="
log "RUN_21_START — orchestrator über alle aktiven Modelle (Judge=sonnet-5)"
$PY -u "$TP/orchestrator.py" --continue-after-ko > "$TP/night_full.log" 2>&1
RC=$?
log "RUN_21_DONE rc=$RC"

# Ergebnis-Kurzfassung ins night2.log
grep -E 'MODELL:|übersprungen — Fehler|Testplan abgeschlossen' "$TP/night_full.log" 2>/dev/null | tail -30 >> "$NL"

log "Sites bauen"
python3 "$TP/make_public_site.py"  >>"$NL" 2>&1
push "$VLLM" "vLLM lokal (gb10-worker2): 21er-Kohorte publiziert" docs testplan/license_cache.json testplan/run21_chain.sh
python3 "$RESULTS/build_site.py"   >>"$NL" 2>&1
push "$RESULTS" "Hub: lokale vLLM-Ergebnisse (21er)" build_site.py docs

log "CHAIN_DONE=green rc=$RC"
log "=== RUN21-CHAIN ENDE ==="
