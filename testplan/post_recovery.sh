#!/bin/bash
# Post-Recovery-Arbeit, startet automatisch sobald recover_retry.sh KOMPLETT fertig
# ist und die GPU frei (kein vllm-Container). Zwei Aufgaben (User 2026-08-10):
#   Phase 1: GLM-4.7-Flash Engine-Traceback einfangen (degraded, Inferenz-Crash,
#            Report trägt keinen Fehlertext) → glm_traceback.log zum Diagnostizieren.
#   Phase 2: Apertus-v1.5-8B + -70B mit swiss-ai Fork-Image (arch apertus1p5),
#            über den NEUEN Orchestrator (Fail-Fast bei Container-Crash).
# Dann Reports in Kohorte, validieren, Sites publizieren.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
VLLM=/home/mvdb/southbyte/southbyte-vllm
TP=$VLLM/testplan
RESULTS=/home/mvdb/southbyte/southbyte-results
PY=$TP/.venv/bin/python
COH=$TP/reports/2026-08-08_1130
IMG='ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64'
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013gePcuZs3qnocpMR9U6rLL'
ML=$TP/post_recovery.log
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

# ── Gate: recover_retry.sh komplett fertig + GPU frei ────────────────────────
log "=== POST-RECOVERY armiert — warte auf Recovery-Ende + freie GPU ==="
for i in $(seq 1 5760); do   # max 48h
  pgrep -f '[r]ecover_retry\.sh' >/dev/null || break
  sleep 30
done
log "recover_retry.sh beendet — warte auf GPU-Freigabe (kein vllm-Container)"
for i in $(seq 1 480); do    # max 4h Puffer
  docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^vllm-' || break
  sleep 30
done
log "GPU frei."

# ── Phase 1: GLM-4.7-Flash Engine-Traceback ─────────────────────────────────
GLM_C=vllm-zai-org--GLM-4.7-Flash
log "PHASE 1: GLM-4.7-Flash starten für Traceback-Capture ..."
docker rm -f "$GLM_C" 2>/dev/null || true
( cd "$VLLM" && CONTAINER_NAME="$GLM_C" HOST_PORT=8000 \
    bash runner/vllm_spark.sh --model zai-org--GLM-4.7-Flash --skip-pull \
    > "$TP/glm_diag_start.log" 2>&1 )
# auf Readiness warten (max 20 min)
ready=0
for i in $(seq 1 240); do
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)
  [ "$code" = "200" ] && { ready=1; break; }
  docker ps -a --format '{{.Names}}|{{.Status}}' | grep -q "$GLM_C|Exited" && { log "GLM-Container schon beim Start gestorben"; break; }
  sleep 5
done
if [ "$ready" = "1" ]; then
  log "GLM ready — sende Inferenz-Trigger (Crash provozieren) ..."
  MID=$(curl -s -m 10 http://localhost:8000/v1/models | $PY -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "zai-org--GLM-4.7-Flash")
  for q in "Erkläre kurz Photosynthese." "Was ist 17*23? Nur die Zahl." "Schreibe einen Satz über den Mond."; do
    curl -s -m 60 -X POST http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":64}" \
      >> "$TP/glm_probe.log" 2>&1
    echo "---" >> "$TP/glm_probe.log"
    sleep 2
  done
else
  log "GLM nie ready — Startup-Log siehe glm_diag_start.log"
fi
log "GLM: volle Container-Logs sichern → glm_traceback.log"
docker logs "$GLM_C" > "$TP/glm_traceback.log" 2>&1 || true
docker rm -f "$GLM_C" 2>/dev/null || true
log "PHASE 1 fertig. Traceback: $TP/glm_traceback.log (Diagnose durch Claude)"

# ── Phase 2: Apertus-v1.5-8B + -70B (swiss-ai Image, Fail-Fast-Orchestrator) ─
log "PHASE 2: warte auf Apertus-Image ($IMG) ..."
for i in $(seq 1 120); do
  docker image inspect "$IMG" >/dev/null 2>&1 && { log "Image vorhanden."; break; }
  sleep 30
done
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  log "ABBRUCH Phase 2: Apertus-Image fehlt (Pull nicht fertig?) — siehe apertus_pull.log"
else
  log "Apertus 8B + 70B über Orchestrator (Fail-Fast aktiv) ..."
  $PY -u orchestrator.py --models "Apertus-v1.5-8B,Apertus-v1.5-70B" --continue-after-ko \
    > "$TP/apertus_run.log" 2>&1
  log "orchestrator rc=$?"
  for M in Apertus-v1.5-8B Apertus-v1.5-70B; do
    ND=$(ls -1dt "$TP"/reports/2026-*/ 2>/dev/null | while read d; do
           [ "$d" = "$COH/" ] && continue; [ -f "$d$M.json" ] && { echo "$d"; break; }; done)
    if [ -n "$ND" ]; then cp "$ND$M.json" "$COH/$M.json"; log "$M <- ${ND}: $(valid "$COH/$M.json")"
    else log "$M: KEIN neuer Report (Crash/OOM? siehe apertus_run.log) — N/A-Kandidat"; fi
  done
fi

# ── Phase 3: Publizieren ────────────────────────────────────────────────────
log "PHASE 3: Sites bauen + pushen"
python3 "$TP/make_public_site.py" >>"$ML" 2>&1
push "$VLLM" "Post-Recovery: Apertus-v1.5 (swiss-ai Image) nachgezogen + GLM-Traceback erfasst" docs
python3 "$RESULTS/build_site.py" >>"$ML" 2>&1
push "$RESULTS" "Hub: Apertus-v1.5 Post-Recovery" docs
log "POST_RECOVERY_DONE"
