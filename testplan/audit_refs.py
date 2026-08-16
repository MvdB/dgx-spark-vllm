#!/usr/bin/env python3
"""Referenz-Audit: die zwei stärksten Modelle (Fable-5 + GPT-5.6-sol) prüfen die
hinterlegten Referenzantworten des Testsatzes auf Faktenfehler/Veralterung —
BEVOR alle Modelle durch sind. Wo beide (oder eins) die Referenz als falsch/stale
markieren, ist sie verdächtig und wird gemeldet (nicht automatisch geändert).

Kein Judge-Ersatz — reines Ground-Truth-Cross-Check. Nur stdlib + Proxy.
"""
import json, re, sys, urllib.request, urllib.error, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

TP = Path(__file__).resolve().parent
env = {}
for ln in (TP / ".env").read_text().splitlines():
    if "=" in ln and not ln.strip().startswith("#"):
        k, _, v = ln.partition("="); env[k.strip()] = v.strip()
KEY = env["JUDGE_API_KEY"]
# Kein Standardwert für den Host: Das Repository ist öffentlich, und eine interne
# Adresse im Code ist eine Adresse in der Welt. `.env.example` und testplan.yaml
# halten es schon so; diese Datei war der letzte Ausreißer.
HOST = env.get("JUDGE_HOST", "").strip()
if not HOST:
    sys.exit("JUDGE_HOST fehlt in testplan/.env — die Adresse steht bewusst nicht im Code.")
URL = f"http://{HOST}:{env.get('JUDGE_PORT', '4000')}/v1/chat/completions"
MODELS = [("Fable-5", "claude-fable-5"), ("GPT-5.6-sol", "gpt-5.6-sol")]

CATS = ("quality", "long_context", "german_language", "bias", "code")
SYS = ("Du bist ein sorgfältiger Faktenprüfer für einen deutschen LLM-Testsatz. "
       "Dir werden eine Testfrage und die hinterlegte REFERENZ (Musterantwort bzw. "
       "Bewertungsvorgabe) gezeigt. Prüfe AUSSCHLIESSLICH die Referenz: Ist sie sachlich "
       "korrekt und aktuell? Heutiges Datum: 8. August 2026. Achte auf veraltete Zahlen/"
       "Daten, Faktenfehler, falsch spezifizierte Fallen. WICHTIG: Viele Fälle sind "
       "ABSICHTLICH Fallen mit erfundenen Entitaeten/Studien (tags mit trap/fake_*) — dann "
       "ist die Referenz KORREKT, wenn sie Ablehnung/Unsicherheit verlangt; nicht als Fehler "
       "melden. Antworte NUR mit einem JSON-Objekt: "
       '{"status":"ok|stale|wrong|ambiguous","issue":"knapp, leer wenn ok",'
       '"fix":"knapper Korrekturvorschlag, leer wenn ok"}.')


def load_cases():
    out = []
    for cat in CATS:
        for f in sorted(glob.glob(str(TP / "testdata" / cat / "*.jsonl"))):
            for ln in open(f, encoding="utf-8"):
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                try:
                    o = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                exp = o.get("expected") or {}
                ref = exp.get("value", "")
                if not o.get("prompt"):
                    continue
                out.append({"id": o["id"], "cat": cat, "sub": o.get("subcategory", ""),
                            "tags": o.get("metadata", {}).get("tags", []),
                            "prompt": o["prompt"], "ref": ref, "etype": exp.get("type", "")})
    return out


def ask(model_id, case):
    user = (f"FRAGE:\n{case['prompt']}\n\nSUBKATEGORIE: {case['sub']} · TAGS: {case['tags']} · "
            f"ERWARTUNGSTYP: {case['etype']}\n\nHINTERLEGTE REFERENZ:\n{case['ref'] or '(keine — freie Judge-Bewertung)'}\n\n"
            "Ist diese Referenz sachlich korrekt und aktuell?")
    body = json.dumps({"model": model_id, "max_tokens": 400,
                       "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": f"Bearer {KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.loads(r.read())
        txt = d["choices"][0]["message"]["content"] or ""
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        o = json.loads(m.group()) if m else {"status": "parse_err", "issue": txt[:80], "fix": ""}
        return o.get("status", "?"), o.get("issue", ""), o.get("fix", "")
    except Exception as e:
        return "err", str(e)[:80], ""


def audit_one(case):
    res = {}
    for name, mid in MODELS:
        res[name] = ask(mid, case)
    return case, res


def main():
    cases = load_cases()
    print(f"Audit {len(cases)} Fälle × {len(MODELS)} Modelle …\n", flush=True)
    flagged = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (case, res) in enumerate(ex.map(audit_one, cases), 1):
            statuses = [res[n][0] for n, _ in MODELS]
            bad = [s for s in statuses if s not in ("ok", "err", "parse_err")]
            if bad:
                flagged.append((case, res, len(bad)))
            if i % 15 == 0:
                print(f"  … {i}/{len(cases)}", flush=True)
    flagged.sort(key=lambda x: -x[2])  # beide-markiert zuerst
    print(f"\n═══ {len(flagged)} auffällige Referenzen (von {len(cases)}) ═══\n")
    for case, res, n in flagged:
        mark = "‼ BEIDE" if n == 2 else "· eins"
        print(f"[{mark}] {case['id']}  ({case['cat']}/{case['sub']}) tags={case['tags']}")
        print(f"    Frage: {case['prompt'][:80]}")
        print(f"    Ref  : {(case['ref'] or '(frei)')[:110]}")
        for name, _ in MODELS:
            st, iss, fix = res[name]
            if st not in ("ok",):
                print(f"    {name:12} [{st}] {iss[:120]}")
                if fix:
                    print(f"    {'':12}  → fix: {fix[:120]}")
        print()
    # Maschinenlesbar für den Folgeschritt
    (TP / "reports" / "ref_audit.json").write_text(
        json.dumps([{"id": c["id"], "cat": c["cat"], "ref": c["ref"], "res": r} for c, r, _ in flagged],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ Details: reports/ref_audit.json")


if __name__ == "__main__":
    main()
