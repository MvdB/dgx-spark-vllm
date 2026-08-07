#!/usr/bin/env python3
"""southbyte-vllm — baut die publizierte LLM- & Guardrail-Vergleichsseite docs/.

Eigenständige Pages-Seite dieses Repos (wie southbyte-tts / southbyte-image ihre
eigene haben). Liest die lokalen, gitignorierten Läufe unter testplan/reports/,
rendert kuratierte Übersicht + Detailseiten je Modell/Guard und schreibt nach
<repo>/docs/. Nur stdlib; kein GPU.

Sicherheit: 04_security wird NIE publiziert; 06_performance hat keine Transkripte.
Antworten sind auf 500 Zeichen gekürzt (so im Report gespeichert). Rohdaten unter
testplan/reports/ bleiben lokal (gitignored) — nur docs/ wird committet.
"""
from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

TESTPLAN = Path(__file__).resolve().parent
REPO = TESTPLAN.parent
DOCS = Path(os.environ.get("DOCS_DIR", REPO / "docs"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", TESTPLAN / "reports"))
TESTDATA_DIR = Path(os.environ.get("TESTDATA_DIR", TESTPLAN / "testdata"))
GUARDS_DIR = Path(os.environ.get("GUARDS_DIR", TESTPLAN / "reports" / "guardrails"))

HUB_URL = "https://results.southbyte.de/"

EXCLUDE_PLAYBOOKS = {"04_security"}
PLAYBOOK_LABELS = {
    "01_quality": "Qualität", "02_german_language": "Deutsch", "03_bias": "Bias",
    "05_code": "Code", "06_performance": "Performance",
}
_JUDGE_PBS = ("01_quality", "02_german_language", "03_bias", "05_code")
_OV_CLS = {"PASS": "pass", "WARN": "warn", "FAIL": "fail", "K.O.": "knockout"}

CI_STYLE = """
 :root{--bg:#060C0A;--bg-raised:#0A1410;--bg-card:#0E1A14;--border:#162A1E;--border-hi:#1A5C38;
   --green:#00E676;--green-dim:#00994A;--amber:#F59E0B;--text:#D4EDE0;--text-muted:#5E8A72;--text-dim:#2E5040;
   --ko:#FF5A5A;--mono:'Courier New',Consolas,'Cascadia Code','SF Mono',Menlo,monospace;
   --sans:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.7}
 .grid-bg{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.5;
   background-image:linear-gradient(rgba(0,230,118,.15) 1px,transparent 1px),
     linear-gradient(90deg,rgba(0,230,118,.15) 1px,transparent 1px);background-size:80px 80px}
 .wrap{position:relative;z-index:1;max-width:960px;margin:0 auto;padding:2.5rem 1.25rem}
 .wordmark{font-family:var(--mono);font-weight:700;font-size:1.5rem;letter-spacing:1.4px;color:var(--text);text-decoration:none}
 .wordmark .dot{color:var(--green)}
 .tagline{font-family:var(--mono);font-size:.7rem;letter-spacing:.25em;text-transform:uppercase;color:var(--text-muted);margin-top:.3rem}
 .back{font-family:var(--mono);font-size:.8rem;display:inline-block;margin:1rem 0 .3rem}
 h1{font-family:var(--mono);font-size:1.9rem;margin:1.2rem 0 .3rem;color:var(--text)}
 .lede{color:var(--text-muted);margin:0 0 1.5rem;max-width:62ch}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin:1.5rem 0}
 .card{border:1px solid var(--border);border-radius:10px;padding:1rem;background:var(--bg-card)}
 .card h3{margin:0 0 .5rem;font-family:var(--mono);font-size:.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.1em}
 .card .big{font-size:1.8rem;font-weight:700;color:var(--text)} .card .sub{color:var(--text-muted);font-size:.85rem}
 .card a{text-decoration:none;color:inherit} .card a:hover .big{color:var(--green)}
 h2{font-family:var(--mono);text-transform:uppercase;letter-spacing:.15em;color:var(--green);font-size:1.05rem;
   margin-top:2.4rem;padding-top:.8rem;border-top:1px solid var(--border-hi)}
 table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}
 th,td{border:1px solid var(--border);padding:.45rem .6rem;text-align:center}
 th{font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted);background:var(--bg-raised)}
 th:first-child,td:first-child{text-align:left} tbody tr:hover{background:var(--bg-raised)}
 code{font-family:var(--mono);color:var(--green);background:var(--bg-card);padding:.05em .35em;border-radius:4px}
 a{color:var(--green)} a:hover{color:var(--green-dim)} strong{color:var(--text)}
 .ko{color:var(--ko);font-weight:600} .empty,.note{color:var(--text-muted);font-size:.9rem}
 footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border);color:var(--text-muted);font-size:.82rem}
 footer .wm{font-family:var(--mono);font-weight:700;letter-spacing:1px;color:var(--text)} footer .wm .dot{color:var(--green)}
 .case{border:1px solid var(--border);border-left-width:5px;border-radius:8px;padding:.7rem 1rem;margin:.7rem 0;background:var(--bg-card)}
 .case.pass{border-left-color:var(--green)} .case.warn{border-left-color:var(--amber)}
 .case.fail,.case.knockout{border-left-color:var(--ko)} .case.error{border-left-color:var(--text-dim)}
 .case .hd{display:flex;justify-content:space-between;gap:.6rem;align-items:baseline;flex-wrap:wrap}
 .case .cid{font-family:var(--mono);font-size:.8rem;color:var(--text-muted)}
 .badge{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;padding:.1em .5em;border-radius:4px;border:1px solid var(--border-hi)}
 .badge.pass{color:var(--green)} .badge.warn{color:var(--amber)} .badge.fail,.badge.knockout{color:var(--ko)} .badge.error{color:var(--text-dim)}
 .qa{margin:.45rem 0} .qa .lbl{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted);display:block;margin-bottom:.15rem}
 .qa .txt{white-space:pre-wrap;word-break:break-word}
 .resp{background:var(--bg-raised);border-left:3px solid var(--border-hi);padding:.45rem .65rem;border-radius:3px}
 .judge{background:var(--bg-raised);border-left:3px solid var(--green);padding:.45rem .65rem;border-radius:3px}
 details{margin:.3rem 0} summary{cursor:pointer;color:var(--text-muted);font-family:var(--mono);font-size:.76rem}
 .outcome-TP,.outcome-TN{color:var(--green);font-weight:600} .outcome-FP,.outcome-FN{color:var(--ko);font-weight:600}
 .teaser{border:1px solid var(--border-hi);border-radius:8px;padding:.85rem 1.1rem;margin:.3rem 0 1.6rem;background:var(--bg-card)}
 .teaser .facts{display:flex;flex-wrap:wrap;gap:.35rem 1.4rem;font-size:.86rem;margin-bottom:.6rem}
 .teaser .facts b{font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted);margin-right:.4rem}
 .teaser .perf{display:flex;flex-wrap:wrap;gap:.5rem 1rem;align-items:baseline}
 .teaser .big{font-family:var(--mono);font-size:1.1rem} .teaser .pbmini{color:var(--text-muted);font-size:.85rem;font-family:var(--mono)}
 .modellist{columns:2;gap:1.5rem} .modellist a{display:block;padding:.15rem 0} @media(max-width:640px){.modellist{columns:1}}
"""

_FOOTER = ('<footer><span class="wm">SOUTH<span class="dot">.</span>BYTE</span> — Michael van den Berg · '
           f'Cross-Modality-Hub: <a href="{HUB_URL}">results.southbyte.de</a> · '
           '<a href="https://southbyte.de">southbyte.de</a></footer>')


def esc(x) -> str:
    return html.escape(str(x))


def num(x) -> str:
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def table(headers, rows) -> str:
    if not rows:
        return '<p class="empty">Noch keine Daten.</p>'
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def card(title, big, sub, href=None) -> str:
    inner = f'<div class="big">{esc(big)}</div><div class="sub">{esc(sub)}</div>'
    body = f'<a href="{esc(href)}">{inner}</a>' if href else inner
    return f'<div class="card"><h3>{esc(title)}</h3>{body}</div>'


def slugify(s) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def page_shell(title, inner, subtitle="", back="index.html") -> str:
    backlink = f'<a class="back" href="{esc(back)}">← zurück</a>' if back else ""
    sub = f'<p class="lede">{subtitle}</p>' if subtitle else ""
    home = "../index.html" if back and back != "index.html" else "index.html"
    return (f'<!doctype html>\n<html lang="de"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f'<title>SOUTH.BYTE — {esc(title)}</title>\n<style>{CI_STYLE}</style></head>'
            f'<body><div class="grid-bg"></div><div class="wrap">\n'
            f'<header><a class="wordmark" href="{home}">SOUTH<span class="dot">.</span>BYTE</a>'
            f'<div class="tagline">AI Governance &amp; IT-Beratung</div></header>\n'
            f'{backlink}\n<h1>{esc(title)}</h1>\n{sub}\n{inner}\n{_FOOTER}\n</div></body></html>')


# ── Daten laden ──────────────────────────────────────────────────────────────
def _load_run_rows(files):
    rows, saas = [], 0
    for j in files:
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta, summ, pbs = d.get("meta", {}), d.get("summary", {}), d.get("playbooks", {})
        total = err = 0
        for k, v in pbs.items():
            if k in EXCLUDE_PLAYBOOKS or not isinstance(v, dict):
                continue
            for res in v.get("results", []):
                total += 1
                if res.get("verdict") == "error":
                    err += 1
        if total == 0 or err / total > 0.3:
            continue
        if meta.get("source") == "saas_proxy":
            saas += 1
        name = str(meta.get("model") or j.stem).rsplit("/", 1)[-1]
        pr = {k: v.get("pass_rate") for k, v in pbs.items()
              if k not in EXCLUDE_PLAYBOOKS and isinstance(v, dict)}
        rows.append({"model": name, "overall": summ.get("overall"), "pass_rate": summ.get("pass_rate"),
                     "ko": summ.get("knockouts", 0), "pb": pr, "file": j, "stem": j.stem})
    rows.sort(key=lambda r: float(r["pass_rate"] or 0), reverse=True)
    return rows, saas


def load_llm_runs():
    local = saas = None
    for d in sorted(REPORTS_DIR.glob("2026-*"), reverse=True):
        models = [j for j in d.glob("*.json") if not re.search(r"dashboard|index", j.name, re.I)]
        if len(models) < 5:
            continue
        rows, nsaas = _load_run_rows(sorted(models))
        if len(rows) < 5:
            continue
        kind = "saas" if nsaas * 2 >= len(rows) else "local"
        if kind == "saas" and saas is None:
            saas = {"run": d.name, "rows": rows}
        elif kind == "local" and local is None:
            local = {"run": d.name, "rows": rows}
        if local and saas:
            break
    return {"local": local, "saas": saas}


def load_guards():
    out = []
    for j in sorted(GUARDS_DIR.glob("*.json")):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"label": d.get("label", j.stem), "metrics": d.get("metrics", {}),
                    "knockouts": d.get("knockouts", []), "per_case": d.get("per_case", []),
                    "protocol": d.get("protocol", ""), "threshold": d.get("threshold"),
                    "reasoning_effort": d.get("reasoning_effort", ""),
                    "served_model": d.get("served_model", ""), "slug": slugify(d.get("label", j.stem))})
    return out


def load_guard_prompts():
    """{id: {prompt, truth, harm}} aus testdata/guardrails/ für die Detailseiten."""
    out = {}
    for f in sorted((TESTDATA_DIR / "guardrails").glob("*.jsonl")):
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if o.get("id"):
                out[o["id"]] = {"prompt": o.get("prompt", ""),
                                "truth": (o.get("expected") or {}).get("value", ""),
                                "harm": (o.get("metadata") or {}).get("harm_category", "")}
    return out


def load_testdata_prompts():
    out = {}
    for cat in ("quality", "german_language", "bias", "code", "long_context"):
        for f in sorted((TESTDATA_DIR / cat).glob("*.jsonl")):
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for ln in lines:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                try:
                    o = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if o.get("id"):
                    out[o["id"]] = {"prompt": o.get("prompt", ""), "subcategory": o.get("subcategory", "")}
    return out


# ── Übersichts-Abschnitte ────────────────────────────────────────────────────
def llm_chapter(data, cid, title, lead, card_title):
    if not data or not data["rows"]:
        return "", card(card_title, "—", "keine Berichte")
    cols = list(PLAYBOOK_LABELS)
    header = ["Modell", "Gesamt", "K.O."] + [PLAYBOOK_LABELS[c] for c in cols]
    rows = []
    for r in data["rows"]:
        ov = r["overall"] or "—"
        ov_html = f'<span class="ko">{esc(ov)}</span>' if ov == "K.O." else esc(ov)
        link = f'<a href="m/{esc(r["stem"])}.html">{esc(r["model"])}</a>'
        cells = [link, f'{ov_html} {esc(r["pass_rate"])}%', str(r["ko"] or 0)]
        for c in cols:
            v = r["pb"].get(c)
            cells.append("—" if v is None else f"{round(float(v) * 100)}%")
        rows.append(cells)
    best = data["rows"][0]
    sec = (f'<h2 id="{cid}">{esc(title)}</h2>\n'
           f'<p>{lead} Lauf <code>{esc(data["run"])}</code> · {len(rows)} Modelle · Pass-Rate je Playbook. '
           f'<strong>Modellname anklicken</strong> → Detail (Prompt, Antwort, Judge je Fall). '
           f'<strong>Sicherheit (04) ausgeschlossen</strong>.</p>\n'
           f'<div style="overflow-x:auto">{table(header, rows)}</div>')
    c = card(card_title, str(len(rows)), f'Modelle · Top {esc(best["pass_rate"])}%', f"#{cid}")
    return sec, c


def guards_section(guards):
    if not guards:
        return "", card("Guards", "—", "kein Feldlauf")
    keys = []
    for g in guards:
        for k in g["metrics"]:
            if isinstance(g["metrics"][k], (int, float)) and k not in keys:
                keys.append(k)

    def glabel(g):
        return (f'<a href="g/{esc(g["slug"])}.html">{esc(g["label"])}</a>'
                if g.get("per_case") else esc(g["label"]))
    rows = [[glabel(g)] + [num(g["metrics"].get(k)) for k in keys]
            + ["✓" if not g["knockouts"] else f'<span class="ko">K.O. {len(g["knockouts"])}</span>'] for g in guards]
    best = max(guards, key=lambda g: g["metrics"].get("f1", 0) or 0)
    sec = (f'<h2 id="guards">Guardrails (Playbook 08)</h2>\n'
           f'<p class="note">Guard-Name anklicken → Fall für Fall (Wahrheit vs. Vorhersage). '
           f'Kein Judge — das Label ist die Wahrheit.</p>\n{table(["Guard"] + keys + ["K.O."], rows)}')
    c = card("Guards", f'{(best["metrics"].get("f1", 0) or 0):.3f}', f'bestes F1 · {best["label"]}', "#guards")
    return sec, c


# ── Detailseiten ─────────────────────────────────────────────────────────────
def _llm_teaser(meta, summ, pbs):
    is_saas = meta.get("source") == "saas_proxy"
    src = "SaaS · LiteLLM-Proxy" if is_saas else "lokal · DGX Spark / vLLM"
    prof_label = "Proxy-ID" if is_saas else "Profil / Quant"
    facts = [f'<span><b>Quelle</b>{esc(src)}</span>',
             f'<span><b>{prof_label}</b><code>{esc(meta.get("profile", "—"))}</code></span>',
             f'<span><b>Judge</b><code>{esc(meta.get("judge", "—"))}</code></span>']
    vllm = meta.get("vllm") or meta.get("vllm_tag")
    if vllm and not is_saas:
        facts.insert(1, f'<span><b>vLLM</b><code>{esc(vllm)}</code></span>')
    overall = summ.get("overall", "—")
    ov_cls = _OV_CLS.get(overall, "error")
    mini = " · ".join(f'{PLAYBOOK_LABELS[pb]} {round(float(pbs[pb].get("pass_rate", 0) or 0) * 100)}%'
                      for pb in _JUDGE_PBS if pb in pbs)
    return (f'<div class="teaser"><div class="facts">{"".join(facts)}</div>'
            f'<div class="perf"><span class="badge {ov_cls} big">{esc(overall)} · {esc(summ.get("pass_rate", "—"))}%</span>'
            f'<span>{summ.get("passed", 0)}/{summ.get("total_tests", 0)} Tests bestanden</span>'
            f'<span class="pbmini">{mini}</span></div></div>')


def llm_detail_html(row, prompts):
    d = json.loads(row["file"].read_text(encoding="utf-8"))
    meta, pbs, summ = d.get("meta", {}), d.get("playbooks", {}), d.get("summary", {})
    parts = [_llm_teaser(meta, summ, pbs)]
    for pb in _JUDGE_PBS:
        if pb not in pbs:
            continue
        v = pbs[pb]
        results = sorted(v.get("results", []), key=lambda r: str(r.get("test_id", "")).startswith("loc-bay"))
        if not results:
            continue
        parts.append(f'<h2 id="{esc(pb)}">{esc(PLAYBOOK_LABELS[pb])} · '
                     f'{v.get("passed", 0)}/{v.get("total", 0)} · ø {float(v.get("mean_score", 0) or 0):.2f}</h2>')
        for r in results:
            verdict = (r.get("verdict") or "").lower()
            tid = r.get("test_id", "")
            info = prompts.get(tid, {})
            sub = info.get("subcategory") or r.get("evaluator", "")
            score = r.get("score")
            score_s = f' · {score:.2f}' if isinstance(score, (int, float)) else ""
            blk = [f'<div class="case {esc(verdict)}">',
                   f'<div class="hd"><span class="cid">{esc(tid)} · {esc(sub)}</span>'
                   f'<span class="badge {esc(verdict)}">{esc(verdict or "—")}{score_s}</span></div>']
            if info.get("prompt"):
                blk.append(f'<div class="qa"><span class="lbl">Prompt</span><div class="txt">{esc(info["prompt"])}</div></div>')
            blk.append(f'<div class="qa"><span class="lbl">Antwort</span>'
                       f'<div class="txt resp">{esc(r.get("response", "")) or "—"}</div></div>')
            if r.get("thinking"):
                blk.append(f'<details><summary>Thinking</summary><div class="txt">{esc(r["thinking"])}</div></details>')
            jr = r.get("reasoning") or (r.get("metadata") or {}).get("judge_raw", "")
            if jr:
                blk.append(f'<div class="qa"><span class="lbl">Judge · {esc(meta.get("judge", ""))}</span>'
                           f'<div class="txt judge">{esc(jr)}</div></div>')
            blk.append("</div>")
            parts.append("".join(blk))
    src = "SaaS (LiteLLM-Proxy)" if meta.get("source") == "saas_proxy" else "lokal (DGX Spark / vLLM)"
    subtitle = (f'{src} · Antworten auf 500 Zeichen gekürzt · Sicherheits-Playbook (04) nicht enthalten.')
    return page_shell(f'{meta.get("model", row["stem"])} — LLM-Detail', "\n".join(parts), subtitle=subtitle, back="index.html")


def _guard_teaser(g):
    """Gleicher Header wie LLM-Detail: Guard-Daten + Gesamt-Performance."""
    m = g.get("metrics", {})
    conf = m.get("confusion", {})
    served = str(g.get("served_model", "")).rstrip("/").rsplit("/", 1)[-1]
    facts = [f'<span><b>Quelle</b>Guard · vLLM (DGX Spark)</span>',
             f'<span><b>Served</b><code>{esc(served)}</code></span>',
             f'<span><b>Protokoll</b><code>{esc(g.get("protocol", "—"))}</code></span>']
    if g.get("threshold") is not None:
        facts.append(f'<span><b>Threshold</b>{esc(g["threshold"])}</span>')
    if g.get("reasoning_effort"):
        facts.append(f'<span><b>Reasoning</b>{esc(g["reasoning_effort"])}</span>')
    ov_cls, ov = ("fail", "K.O.") if g.get("knockouts") else ("pass", "OK")
    confs = (f'TP {conf.get("tp", 0)} · TN {conf.get("tn", 0)} · '
             f'<span class="ko">FP {conf.get("fp", 0)} · FN {conf.get("fn", 0)}</span>')
    return (f'<div class="teaser"><div class="facts">{"".join(facts)}</div>'
            f'<div class="perf"><span class="badge {ov_cls} big">{ov} · F1 {float(m.get("f1", 0) or 0):.3f}</span>'
            f'<span class="pbmini">Acc {float(m.get("accuracy", 0) or 0):.3f} · Recall {float(m.get("recall", 0) or 0):.3f} · '
            f'FPR {float(m.get("fpr", 0) or 0):.3f} · Trap-FPR {float(m.get("trap_fpr", 0) or 0):.3f}</span>'
            f'<span class="pbmini">{confs}</span></div></div>')


def _guard_text(c, gprompts):
    """Eingabetext; unsafe maskiert (Harm-Kategorie + ~55 Zeichen), Rest Volltext."""
    info = gprompts.get(c.get("id"), {})
    text = info.get("prompt", "")
    if (c.get("truth") or info.get("truth")) == "unsafe":
        harm = c.get("harm_category") or info.get("harm") or "unsafe"
        snip = text[:55].rstrip()
        return f'[unsafe · {harm}] {snip}…' if snip else f'[unsafe · {harm}]'
    return text


def guard_detail_html(g, gprompts):
    # Case-Overview (kompakt, alle Fälle)
    ohdr = ["Fall", "Subkat.", "Trap", "Wahrheit", "Vorhersage", "Outcome", "Score", "ms"]
    orows = []
    for c in g.get("per_case", []):
        oc = c.get("outcome", "")
        sc, lat = c.get("score"), c.get("latency_ms")
        orows.append([esc(c.get("id", "")), esc(c.get("subcategory", "")), "⚠" if c.get("trap") else "",
                      esc(c.get("truth", "")), esc(c.get("prediction", "")),
                      f'<span class="outcome-{esc(oc)}">{esc(oc)}</span>',
                      f"{sc:.3f}" if isinstance(sc, (int, float)) else "—",
                      f"{lat:.0f}" if isinstance(lat, (int, float)) else "—"])
    # Fälle: Fehlklassifikationen (FP/FN) zuerst, dann korrekte
    cases = sorted(g.get("per_case", []), key=lambda c: c.get("outcome", "") in ("TP", "TN"))
    blocks = []
    for c in cases:
        oc = c.get("outcome", "")
        cls = "pass" if oc in ("TP", "TN") else "fail"
        badges = f'<span class="badge {cls}">{esc(oc)}</span>'
        if c.get("trap"):
            badges += ' <span class="badge warn">TRAP</span>'
        sc = c.get("score")
        sc_s = f' · {sc:.3f}' if isinstance(sc, (int, float)) else ""
        blocks.append(
            f'<div class="case {cls}"><div class="hd">'
            f'<span class="cid">{esc(c.get("id", ""))} · {esc(c.get("subcategory", ""))}</span>{badges}</div>'
            f'<div class="qa"><span class="lbl">Eingabe</span><div class="txt">{esc(_guard_text(c, gprompts))}</div></div>'
            f'<div class="qa"><span class="lbl">Wahrheit → Vorhersage</span>'
            f'<div class="txt">{esc(c.get("truth", ""))} → <strong>{esc(c.get("prediction", ""))}</strong>{sc_s}</div></div></div>')
    inner = (f'{_guard_teaser(g)}\n<h2>Überblick</h2>\n<div style="overflow-x:auto">{table(ohdr, orows)}</div>\n'
             f'<h2>Fälle</h2>\n<p class="note">Fehlklassifikationen (FP/FN) zuerst · unsafe-Eingaben maskiert.</p>\n'
             + "\n".join(blocks))
    subtitle = f'Kein Judge — das Label ist die Wahrheit · {len(cases)} Fälle.'
    return page_shell(f'{g.get("label", "")} — Guard-Detail', inner, subtitle=subtitle, back="index.html")


def _landing(title, groups, subdir):
    """Verzeichnisseite für docs/<subdir>/ — verhindert 404 auf /m/ bzw. /g/."""
    body = []
    for gtitle, items in groups:
        if not items:
            continue
        body.append(f'<h2>{esc(gtitle)}</h2><div class="modellist">'
                    + "".join(f'<a href="{esc(href)}">{esc(label)}</a>' for label, href in items) + "</div>")
    return page_shell(title, "\n".join(body), back="../index.html")


def generate_details(runs, guards, prompts, gprompts):
    (DOCS / "m").mkdir(parents=True, exist_ok=True)
    (DOCS / "g").mkdir(parents=True, exist_ok=True)
    m_groups = []
    for kind, gtitle in (("local", "Lokale Modelle (DGX Spark)"), ("saas", "SaaS-Referenzkohorte")):
        data = runs.get(kind)
        if not data:
            continue
        items = []
        for row in data["rows"]:
            (DOCS / "m" / f"{row['stem']}.html").write_text(llm_detail_html(row, prompts), encoding="utf-8")
            items.append((row["model"], f"{row['stem']}.html"))
        m_groups.append((gtitle, items))
    g_items = []
    for g in guards:
        if g.get("per_case"):
            (DOCS / "g" / f"{g['slug']}.html").write_text(guard_detail_html(g, gprompts), encoding="utf-8")
            g_items.append((g["label"], f"{g['slug']}.html"))
    (DOCS / "m" / "index.html").write_text(_landing("LLM-Detailseiten", m_groups, "m"), encoding="utf-8")
    (DOCS / "g" / "index.html").write_text(_landing("Guard-Detailseiten", [("Guards", g_items)], "g"), encoding="utf-8")
    return sum(len(i) for _, i in m_groups), len(g_items)


def build_index(runs, guards):
    local_sec, local_card = llm_chapter(runs["local"], "llm-local", "LLM — Lokale Modelle (DGX Spark)",
                                        "Auf dem GB10 selbst serviert (vLLM), Judge-bewertet.", "LLM lokal")
    saas_sec, saas_card = llm_chapter(runs["saas"], "llm-saas", "LLM — SaaS-Referenzkohorte",
                                      "Frontier-Modelle über eine OpenAI-kompatible API als Referenzrahmen, "
                                      "gleicher Testsatz, Judge <code>claude-sonnet-5</code>.", "LLM SaaS")
    g_sec, g_card = guards_section(guards)
    cards = "\n".join([local_card, saas_card, g_card])
    inner = (f'<div class="cards">{cards}</div>\n{local_sec}\n{saas_sec}\n{g_sec}')
    lede = ("LLM- und Guardrail-Evaluationen auf dem NVIDIA DGX Spark (GB10). Modell- und Guard-Namen "
            "sind anklickbar — Fall für Fall mit Prompt, Antwort und Judge. Rohdaten bleiben lokal; "
            "Sicherheits-Playbook (04) wird nicht publiziert.")
    return page_shell("LLM- & Guardrail-Evaluationen", inner, subtitle=lede, back="")


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    runs = load_llm_runs()
    guards = load_guards()
    prompts = load_testdata_prompts()
    gprompts = load_guard_prompts()
    (DOCS / "index.html").write_text(build_index(runs, guards), encoding="utf-8")
    n_llm, n_guard = generate_details(runs, guards, prompts, gprompts)
    print(f"✓ docs/index.html + {n_llm} LLM-Detail + {n_guard} Guard-Detail (+ Landing m/ g/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
