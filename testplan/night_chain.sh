#!/bin/bash
# Losgelöste Autonom-Kette (gb10-worker2): Smoke abwarten → grün/rot prüfen →
# bei grün die 21 aktiven Modelle lokal → Sites bauen + pushen. Läuft ohne User.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
VLLM=/home/mvdb/southbyte/southbyte-vllm
TP=$VLLM/testplan
RESULTS=/home/mvdb/southbyte/southbyte-results
PY=$TP/.venv/bin/python
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019YpS7jJGxYt9cD5nRMS7fH'
NL=$TP/night.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$NL"; }
push(){ local repo="$1" msg="$2"; shift 2; cd "$repo" || return
  git add "$@" 2>/dev/null
  git commit -q -m "$msg

$CO" 2>/dev/null && log "commit: $msg" || log "nichts zu committen ($repo)"
  git pull --rebase origin main >/dev/null 2>&1 || true
  git push origin main 2>&1 | tail -1 | tee -a "$NL"
}

cd "$TP" || exit 1
log "=== NIGHT-CHAIN START ==="

# ── 1) Smoke abwarten ───────────────────────────────────────────────────────
log "warte auf Smoke-Abschluss (Apertus-v1.1-4B)…"
for i in $(seq 1 720); do   # max 60 min
  pgrep -f 'orchestrator.py --models Apertus-v1.1-4B' >/dev/null || break
  sleep 5
done

# ── 2) Green-Check ──────────────────────────────────────────────────────────
COMPLETED=$(grep -c 'Testplan abgeschlossen' "$TP/smoke.log" 2>/dev/null || echo 0)
SKIPPED=$(grep -c 'Apertus-v1.1-4B übersprungen' "$TP/smoke.log" 2>/dev/null || echo 0)
REPDIR=$(ls -1dt "$TP"/reports/*/ 2>/dev/null | head -1)
REPJSON=""; [ -n "$REPDIR" ] && [ -f "${REPDIR}report.json" ] && REPJSON="${REPDIR}report.json"
# Verdikte vorhanden? (Judge lieferte etwas)
VERD=0; [ -n "$REPJSON" ] && VERD=$(grep -Eoc '"verdict"|knockout|passed|"score"' "$REPJSON" 2>/dev/null || echo 0)
EXITLINE=$(grep 'Testplan abgeschlossen. Exit-Code' "$TP/smoke.log" 2>/dev/null | tail -1)
log "Smoke: completed=$COMPLETED skipped=$SKIPPED report=${REPJSON:-none} verdikte=$VERD | $EXITLINE"

if [ "$COMPLETED" -ge 1 ] && [ "$SKIPPED" -eq 0 ] && [ -n "$REPJSON" ] && [ "$VERD" -ge 1 ]; then
  log "SMOKE_VERDICT=green"
else
  log "SMOKE_VERDICT=red"
  log "CHAIN_DONE=red_no_launch — 21er-Lauf NICHT gestartet (Smoke nicht sauber)."
  log "=== NIGHT-CHAIN ENDE ==="
  exit 0
fi

# ── 3) Voller 21er-Lauf (alle aktiven) ──────────────────────────────────────
log "RUN_21_START — orchestrator über alle aktiven Modelle (Judge=sonnet-5)"
$PY -u "$TP/orchestrator.py" --continue-after-ko > "$TP/night_full.log" 2>&1
RC=$?
log "RUN_21_DONE rc=$RC"

# ── 4) Sites bauen + pushen ─────────────────────────────────────────────────
log "Sites bauen"
python3 "$TP/make_public_site.py"  >>"$NL" 2>&1
push "$VLLM" "vLLM lokal (gb10-worker2): 21er-Kohorte publiziert" docs testplan/license_cache.json testplan/night_chain.sh
python3 "$RESULTS/build_site.py"   >>"$NL" 2>&1
push "$RESULTS" "Hub: lokale vLLM-Ergebnisse (21er)" build_site.py docs

log "CHAIN_DONE=green rc=$RC"
log "=== NIGHT-CHAIN ENDE ==="
