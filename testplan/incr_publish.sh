#!/bin/bash
# Inkrementeller Publisher: bei jedem neuen 5er-Meilenstein fertiger LOKALER
# Modelle (5/10/15/20) beide Sites bauen + pushen. Endet, wenn orchestrator weg
# ist — den finalen 21er-Push macht run21_chain.sh.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
VLLM=/home/mvdb/southbyte/southbyte-vllm
TP=$VLLM/testplan
RESULTS=/home/mvdb/southbyte/southbyte-results
RD=$TP/reports/2026-08-08_1130
CO='Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019YpS7jJGxYt9cD5nRMS7fH'
IL=$TP/incr_publish.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$IL"; }
push(){ local repo="$1" msg="$2"; shift 2; cd "$repo" || return
  git add "$@" 2>/dev/null
  git commit -q -m "$msg

$CO" 2>/dev/null && log "commit: $msg" || log "nichts zu committen ($repo)"
  git pull --rebase origin main >/dev/null 2>&1 || true
  git push origin main 2>&1 | tail -1 | tee -a "$IL"
}
# Zählt nur VALIDE Modelle (error-rate ≤30 %) — analog zum make_public_site-Filter,
# damit kaputte Läufe (z.B. DiffusionGemma 99% error) die ≥5-Publish-Schwelle nicht blockieren.
count_done(){ python3 - "$RD" <<'PY'
import json, glob, os, re, sys
d = sys.argv[1]; n = 0
for f in glob.glob(d + "/*.json"):
    if re.search(r'dashboard|index|/_', f): continue
    try: j = json.load(open(f))
    except Exception: continue
    pbs = j.get("playbooks", {})
    tot = err = 0
    for pd in (pbs.values() if isinstance(pbs, dict) else pbs):
        rs = pd.get("results", []) if isinstance(pd, dict) else []
        tot += len(rs); err += sum(1 for r in rs if r.get("verdict") == "error")
    if tot and err / tot <= 0.3: n += 1
print(n)
PY
}

cd "$TP"
log "=== INCR-PUBLISHER START (Meilenstein je 5 lokale Modelle) ==="
last=0
while true; do
  running=0; pgrep -f 'orchestrator.py --continue' >/dev/null && running=1
  n=$(count_done)
  ms=$n
  if [ "$ms" -ge 3 ] && [ "$ms" -gt "$last" ]; then
    log "Neues valides Modell: $n lokal → Sites bauen + pushen"
    python3 "$TP/make_public_site.py" >>"$IL" 2>&1
    push "$VLLM" "vLLM lokal: $n/21 Modelle publiziert (inkrementell)" docs testplan/license_cache.json
    python3 "$RESULTS/build_site.py" >>"$IL" 2>&1
    push "$RESULTS" "Hub: vLLM lokal $n/21 (inkrementell)" build_site.py docs
    last=$ms
  fi
  if [ "$running" = "0" ]; then
    log "orchestrator beendet bei $n fertigen — Publisher endet, finaler Push via run21_chain."
    break
  fi
  sleep 60
done
log "=== INCR-PUBLISHER ENDE ==="
