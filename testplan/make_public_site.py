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
import urllib.request
from pathlib import Path

TESTPLAN = Path(__file__).resolve().parent
REPO = TESTPLAN.parent
DOCS = Path(os.environ.get("DOCS_DIR", REPO / "docs"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", TESTPLAN / "reports"))
TESTDATA_DIR = Path(os.environ.get("TESTDATA_DIR", TESTPLAN / "testdata"))
MODELS_YAML = Path(os.environ.get("MODELS_YAML", TESTPLAN / "config" / "models.yaml"))


def _load_models() -> dict:
    """Zentrale Modell-Metadaten (name → release_date/license/hf_repo) aus models.yaml.
    Flach über alle Sektionen; Namen eindeutig (TTS-Keys = Engine-Repo)."""
    out: dict = {}
    try:
        lines = MODELS_YAML.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    cur = None
    for ln in lines:
        m = re.match(r'\s*-\s*name:\s*"?([^"#\n]+?)"?\s*$', ln)
        if m:
            cur = {}
            out[m.group(1).strip()] = cur
            continue
        f = re.match(r'\s+(hf_repo|release_date|license|provider):\s*"?([^"#\n]+?)"?\s*$', ln)
        if f and cur is not None:
            cur[f.group(1)] = f.group(2).strip()
    return out


_MODELS = _load_models()


def release_date(name: str) -> str:
    """Release-Zelle mit numerischem Sortier-Key (YYYYMM) — sonst liest der
    Sortierer nur '2026' und ignoriert den Monat."""
    d = str(_MODELS.get(name or "", {}).get("release_date", "") or "")
    m = re.match(r"(\d{4})-(\d{2})", d)
    return f'<span data-sort="{m.group(1)}{m.group(2)}">{esc(d)}</span>' if m else "—"
GUARDS_DIR = Path(os.environ.get("GUARDS_DIR", TESTPLAN / "reports" / "guardrails"))

HUB_URL = "https://results.southbyte.de/"
HF_MODELS_DIR = Path(os.environ.get("HF_MODELS_DIR", Path.home() / "hf_models"))

# SaaS-Modelle sind proprietär (API); Link auf den Anbieter statt HF.
_SAAS_PROVIDER = [
    ("claude", "Anthropic · proprietär (API)", "https://www.anthropic.com/claude"),
    ("gpt", "OpenAI · proprietär (API)", "https://platform.openai.com/docs/models"),
    ("gemini", "Google · proprietär (API)", "https://ai.google.dev/gemini-api/docs/models"),
    ("grok", "xAI · proprietär (API)", "https://docs.x.ai/docs/models"),
    ("xai", "xAI · proprietär (API)", "https://docs.x.ai/docs/models"),
    ("magistral", "Mistral · proprietär (API)", "https://docs.mistral.ai/getting-started/models/"),
    ("ministral", "Mistral · proprietär (API)", "https://docs.mistral.ai/getting-started/models/"),
    ("mistral", "Mistral · proprietär (API)", "https://docs.mistral.ai/getting-started/models/"),
]

# Persistenter Lizenz-Cache (viele Modelle sind lokal retired → HF-API + Cache).
_LIC_CACHE_FILE = TESTPLAN / "license_cache.json"
try:
    _lic_disk: dict[str, str] = json.loads(_LIC_CACHE_FILE.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    _lic_disk = {}
_lic_dirty = [False]
# Wenn weder README noch HF etwas liefern: bekannte Owner-Lizenzen.
_OWNER_FALLBACK = [("nvidia--", "NVIDIA Open Model License"), ("LiquidAI--", "LFM Open License")]


def _license_from_readme(profile: str) -> str:
    try:
        txt = (HF_MODELS_DIR / profile / "README.md").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    m = re.match(r"^---\s*\n(.*?)\n---", txt, re.DOTALL)
    fm = m.group(1) if m else txt[:2000]
    lic = re.search(r"^license:\s*(.+)$", fm, re.MULTILINE)
    name = re.search(r"^license_name:\s*(.+)$", fm, re.MULTILINE)
    val = (lic.group(1).strip().strip('"\'') if lic else "")
    if val.lower() in ("other", "") and name:
        val = name.group(1).strip().strip('"\'')
    return val


def _license_from_hf(profile: str) -> str:
    if "--" not in profile:
        return ""
    repo = profile.replace("--", "/", 1)
    try:
        req = urllib.request.Request("https://huggingface.co/api/models/" + repo,
                                     headers={"User-Agent": "southbyte-site"})
        with urllib.request.urlopen(req, timeout=12) as r:
            d = json.loads(r.read())
    except Exception:
        return ""
    cd = d.get("cardData") or {}
    val = cd.get("license") or ""
    if str(val).lower() in ("", "other"):
        val = cd.get("license_name") or val
    if not val:
        for t in d.get("tags", []):
            if str(t).startswith("license:"):
                val = t.split(":", 1)[1]
                break
    return str(val or "")


def model_license(profile: str, is_saas: bool) -> str:
    """SaaS → proprietär; lokal → README, sonst HF-API (gecacht), sonst Owner-Fallback."""
    if is_saas:
        for needle, label, _ in _SAAS_PROVIDER:
            if needle in profile.lower():
                return label
        return "proprietär (API)"
    if not profile:
        return "—"
    if profile in _lic_disk:
        return _lic_disk[profile]
    val = _license_from_readme(profile) or _license_from_hf(profile)
    if not val:
        for pre, lab in _OWNER_FALLBACK:
            if profile.startswith(pre):
                val = lab
                break
    _lic_disk[profile] = val or "—"
    _lic_dirty[0] = True
    return _lic_disk[profile]


def model_repo(profile: str, is_saas: bool) -> tuple[str, str]:
    """(url, label) für den Repo-/Anbieter-Link."""
    if is_saas:
        for needle, _, url in _SAAS_PROVIDER:
            if needle in profile.lower():
                return url, "Anbieter ↗"
        return "", ""
    if "--" in profile:
        return "https://huggingface.co/" + profile.replace("--", "/", 1), "HF ↗"
    return "", ""

EXCLUDE_PLAYBOOKS = {"04_security"}
# Modelle, die trotz evtl. vorhandenem Report NIE publiziert werden (z.B. gehören in
# eine andere Collection). Der laufende Orchestrator hat seine Liste beim Start
# gesnapshottet, testet Qwen-AgentWorld also noch — der Report wird hier gefiltert.
_EXCLUDE_MODELS = {"Qwen-AgentWorld-35B-A3B"}
PLAYBOOK_LABELS = {
    "01_quality": "Qualität", "02_german_language": "Deutsch", "03_bias": "Bias",
    "05_code": "Code", "06_performance": "Performance",
}
_JUDGE_PBS = ("01_quality", "02_german_language", "03_bias", "05_code")
_OV_CLS = {"PASS": "pass", "WARN": "warn", "FAIL": "fail", "K.O.": "knockout"}


def _perf(pbs):
    """Perf-Metriken (TTFT, Tok/s) aus 06_performance/perf_benchmark ziehen.
    response ist auf 500 Zeichen gekürzt (JSON evtl. mittendrin abgeschnitten) →
    tolerant per Regex; die Kernwerte (TTFT p50/95/99, Durchsatz) liegen davor."""
    pb = pbs.get("06_performance")
    if not isinstance(pb, dict):
        return None
    r = next((x for x in pb.get("results", []) if x.get("test_id") == "perf_benchmark"), None)
    if not r:
        return None
    s = r.get("response", "") or ""

    def numv(key):
        m = re.search(rf'"{key}"\s*:\s*([\d.]+)', s)
        return float(m.group(1)) if m else None

    by = {}
    mt = re.search(r'"throughput_by_type"\s*:\s*\{([^}]*)', s)
    if mt:
        for t in ("short", "medium", "long"):
            mm = re.search(rf'"{t}"\s*:\s*([\d.]+)', mt.group(1))
            if mm:
                by[t] = float(mm.group(1))
    ms = re.search(r'"source"\s*:\s*"([^"]*)"', s)
    p = {"ttft_p50": numv("ttft_p50_ms"), "ttft_p95": numv("ttft_p95_ms"),
         "ttft_p99": numv("ttft_p99_ms"), "tok_mean": numv("throughput_mean_tok_s"),
         "tok_median": numv("throughput_median_tok_s"), "by": by,
         "cloud": bool(ms and "proxy" in ms.group(1).lower())}
    return p if (p["tok_median"] is not None or p["ttft_p50"] is not None) else None


def _perf_section(p):
    tk = lambda v: f"{v:.1f} tok/s" if v is not None else "—"
    ms = lambda v: f"{v:.0f} ms" if v is not None else "—"
    by = p.get("by", {})
    rows = [["Durchsatz (Median / Ø)", f'{tk(p["tok_median"])} / {tk(p["tok_mean"])}'],
            ["Durchsatz kurz / mittel / lang",
             f'{tk(by.get("short"))} / {tk(by.get("medium"))} / {tk(by.get("long"))}'],
            ["TTFT p50 / p95 / p99", f'{ms(p["ttft_p50"])} / {ms(p["ttft_p95"])} / {ms(p["ttft_p99"])}']]
    note = ("gemessen über <b>LiteLLM</b> — Cloud-Modell + Netz, nicht lokale Hardware. "
            "Dieser LiteLLM-Aufbau (ein Endpoint für Cloud- und lokale Modelle) wird von SouthByte empfohlen"
            if p.get("cloud") else "lokale Messung auf DGX Spark (GB10)")
    return (f'<h2 id="performance">Performance</h2>'
            f'<p class="note">Durchsatz = Generierungs-Tokens/s · TTFT = Time-to-First-Token · {note}.</p>'
            f'{table(["Metrik", "Wert"], rows)}')

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
 @keyframes scanline{0%{transform:translateY(-100vh)}100%{transform:translateY(100vh)}}
 .scanline{position:fixed;left:0;top:0;width:100%;height:80px;background:linear-gradient(to bottom,transparent,rgba(0,230,118,.03) 40%,rgba(0,230,118,.07) 50%,rgba(0,230,118,.03) 60%,transparent);pointer-events:none;z-index:0;animation:scanline 8s linear infinite;will-change:transform}
 @media(prefers-reduced-motion:reduce){.scanline{display:none}}
 .wrap{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:2.5rem 1.25rem}
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
 .badge.valid{color:var(--green)} .badge.running{color:var(--amber)} .badge.degraded{color:#F59E0B;border-color:#7A4A0A}
 .badge.pending{color:var(--text-muted)} .badge.na{color:var(--text-dim)}
 tr.st-degraded td:first-child,tr.st-pending td:first-child,tr.st-na td:first-child{color:var(--text-muted)}
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


# Bestwert-Richtung je Spaltentitel (grün hervorgehoben): min = niedriger besser,
# max = höher besser. Unbekannte Titel bleiben unmarkiert.
_BEST_DIR = {
    "Gesamt": "max", "Tok/s": "max", "Durchsatz": "max", "F1": "max", "Recall": "max",
    "Qualität": "max", "Deutsch": "max", "Bias": "max", "Code": "max", "Performance": "max",
    "HSF": "max", "Guardrails": "max", "Sicherheit": "max",
    "Präz.": "max", "Acc": "max",
    "TTFT": "min", "FPR": "min", "Trap-FPR": "min",
    "FN-Rate": "min", "Lat ø": "min", "Lat p95": "min",
}

# Kurz-Header für die (breite) Guard-Metrik-Tabelle, damit sie in .wrap passt.
_GUARD_KEY_LABEL = {
    "n_unsafe": "Unsafe", "n_safe": "Safe", "recall": "Recall", "fn_rate": "FN-Rate",
    "fpr": "FPR", "trap_fpr": "Trap-FPR", "precision": "Präz.", "f1": "F1",
    "accuracy": "Acc", "latency_ms_mean": "Lat ø", "latency_ms_p95": "Lat p95",
}


def best_attr(h) -> str:
    d = _BEST_DIR.get(str(h).strip())
    return f' data-best="{d}"' if d else ""


def table(headers, rows) -> str:
    if not rows:
        return '<p class="empty">Noch keine Daten.</p>'
    th = "".join(f"<th{best_attr(h)}>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def card(title, big, sub, href=None) -> str:
    inner = f'<div class="big">{esc(big)}</div><div class="sub">{esc(sub)}</div>'
    body = f'<a href="{esc(href)}">{inner}</a>' if href else inner
    return f'<div class="card"><h3>{esc(title)}</h3>{body}</div>'


def slugify(s) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


# SouthByte-Favicon (Brand-SVG als Data-URI): dunkles Rounded-Square, helles S + grüner Dot.
FAVICON = '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMiAzMiIgcm9sZT0iaW1nIiBhcmlhLWxhYmVsPSJTb3V0aEJ5dGUiPgogIDx0aXRsZT5Tb3V0aEJ5dGU8L3RpdGxlPgogIDxyZWN0IHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgZmlsbD0iIzA2MEMwQSIvPgogIDx0ZXh0IHg9IjIiIHk9IjIzIgogICAgICAgIGZvbnQtZmFtaWx5PSInQ291cmllciBOZXcnLCBDb25zb2xhcywgJ1NGIE1vbm8nLCBtb25vc3BhY2UiCiAgICAgICAgZm9udC1zaXplPSIxNiIKICAgICAgICBmb250LXdlaWdodD0iNzAwIgogICAgICAgIGxldHRlci1zcGFjaW5nPSIwLjUiPgogICAgPHRzcGFuIGZpbGw9IiNENEVERTAiPlM8L3RzcGFuPjx0c3BhbiBmaWxsPSIjMDBFNjc2Ij4uPC90c3Bhbj48dHNwYW4gZmlsbD0iI0Q0RURFMCI+QjwvdHNwYW4+CiAgPC90ZXh0PgogIDxyZWN0IHg9IjIiIHk9IjI2IiB3aWR0aD0iMjgiIGhlaWdodD0iMS41IiBmaWxsPSIjMDBFNjc2IiBvcGFjaXR5PSIwLjQiLz4KPC9zdmc+Cg==">'

# Klick-Sortierung für alle Tabellen (vanilla JS, keine Abhängigkeiten).
SORT_CSS = (
    " table th{cursor:pointer;user-select:none}"
    " table th::after{content:' ';opacity:.35;font-size:.75em}"
    " table th[aria-sort=ascending]::after{content:' \\25B2';opacity:.9}"
    " table th[aria-sort=descending]::after{content:' \\25BC';opacity:.9}"
    " table td.best{font-weight:700;color:var(--green);background:var(--bg-raised)}"
    " .gtbl table{font-size:.82rem}"
    " .gtbl th,.gtbl td{padding:.35rem .45rem}"
)
SORT_SCRIPT = """
<script>
(function(){
  function val(td){var s=td.getAttribute('data-sort');if(s===null){var el=td.querySelector('[data-sort]');if(el)s=el.getAttribute('data-sort');}return (s!==null?s:(td.textContent||'')).trim();}
  function num(t){var m=t.replace(/\\u00a0/g,'').replace(/\\s+/g,'').replace(',','.').match(/-?\\d+(?:\\.\\d+)?/);return m?parseFloat(m[0]):null;}
  function isEmpty(t){return t===''||t==='—'||t==='-';}
  function sortTable(table,idx,asc){
    var tb=table.tBodies[0]; if(!tb) return;
    var rows=Array.prototype.slice.call(tb.rows);
    var allNum=rows.every(function(r){var c=r.cells[idx];if(!c)return true;var v=val(c);return isEmpty(v)||num(v)!==null;});
    rows.sort(function(a,b){
      var av=a.cells[idx]?val(a.cells[idx]):'',bv=b.cells[idx]?val(b.cells[idx]):'';
      var e1=isEmpty(av),e2=isEmpty(bv);
      if(e1&&e2)return 0; if(e1)return 1; if(e2)return -1;
      var r=allNum?((num(av)||0)-(num(bv)||0)):av.localeCompare(bv,'de',{numeric:true});
      return asc?r:-r;
    });
    rows.forEach(function(r){tb.appendChild(r);});
  }
  document.querySelectorAll('table').forEach(function(table){
    var head=table.tHead; if(!head||!head.rows.length) return;
    Array.prototype.forEach.call(head.rows[0].cells,function(th,idx){
      th.setAttribute('title','Klick: sortieren');
      th.addEventListener('click',function(){
        var asc=th.getAttribute('aria-sort')!=='ascending';
        Array.prototype.forEach.call(head.rows[0].cells,function(o){o.removeAttribute('aria-sort');});
        th.setAttribute('aria-sort',asc?'ascending':'descending');
        sortTable(table,idx,asc);
      });
    });
  });
  // Bestwert je Spalte grün markieren (data-best=min|max am th); überlebt Sortierung.
  document.querySelectorAll('table').forEach(function(table){
    var head=table.tHead, tb=table.tBodies[0]; if(!head||!head.rows.length||!tb) return;
    Array.prototype.forEach.call(head.rows[0].cells,function(th,idx){
      var dir=th.getAttribute('data-best'); if(dir!=='min'&&dir!=='max') return;
      var best=null;
      Array.prototype.forEach.call(tb.rows,function(r){var c=r.cells[idx];if(!c)return;var v=num(val(c));if(v===null)return;if(best===null||(dir==='min'?v<best:v>best))best=v;});
      if(best===null)return;
      Array.prototype.forEach.call(tb.rows,function(r){var c=r.cells[idx];if(!c)return;var v=num(val(c));if(v!==null&&v===best)c.classList.add('best');});
    });
  });
})();
</script>
"""


def page_shell(title, inner, subtitle="", back="index.html") -> str:
    backlink = f'<a class="back" href="{esc(back)}">← zurück</a>' if back else ""
    sub = f'<p class="lede">{subtitle}</p>' if subtitle else ""
    home = "../index.html" if back and back != "index.html" else "index.html"
    return (f'<!doctype html>\n<html lang="de"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f'<title>SOUTH.BYTE — {esc(title)}</title>\n{FAVICON}\n<style>{CI_STYLE}{SORT_CSS}</style></head>'
            f'<body><div class="grid-bg"></div><div class="scanline"></div><div class="wrap">\n'
            f'<header><a class="wordmark" href="{home}">SOUTH<span class="dot">.</span>BYTE</a>'
            f'<div class="tagline">AI Governance &amp; IT-Beratung</div></header>\n'
            f'{backlink}\n<h1>{esc(title)}</h1>\n{sub}\n{inner}\n{_FOOTER}\n{SORT_SCRIPT}</div></body></html>')


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
        name = str(meta.get("model") or j.stem).rsplit("/", 1)[-1]
        if name in _EXCLUDE_MODELS:
            continue
        if meta.get("source") == "saas_proxy":
            saas += 1
        pr = {k: v.get("pass_rate") for k, v in pbs.items()
              if k not in EXCLUDE_PLAYBOOKS and isinstance(v, dict)}
        rows.append({"model": name, "overall": summ.get("overall"), "pass_rate": summ.get("pass_rate"),
                     "ko": summ.get("knockouts", 0), "pb": pr, "file": j, "stem": j.stem, "perf": _perf(pbs),
                     "profile": meta.get("profile", ""), "is_saas": meta.get("source") == "saas_proxy"})
    rows.sort(key=lambda r: float(r["pass_rate"] or 0), reverse=True)
    return rows, saas


# Kanonische lokale Kohorte: alle Nachzügler/Retries kopieren ihre frischen Reports
# HIERHER zurück (COH in *_retry.sh). Maßgeblich ist diese Dir — nicht das neueste
# (= partielles Retry-Dir → nur 3 valide) und nicht das größte (= alte Explorationsläufe).
LOCAL_COHORT_RUN = "2026-08-08_1130"


def load_llm_runs():
    locals_, saas = [], None
    for d in sorted(REPORTS_DIR.glob("2026-*"), reverse=True):
        models = [j for j in d.glob("*.json") if not re.search(r"dashboard|index", j.name, re.I)]
        if len(models) < 3:
            continue
        rows, nsaas = _load_run_rows(sorted(models))
        if len(rows) < 3:
            continue
        if nsaas * 2 >= len(rows):
            if saas is None:  # neuester SaaS-Lauf
                saas = {"run": d.name, "rows": rows}
        else:
            locals_.append({"run": d.name, "rows": rows})
    # lokale Kohorte: gepinnte 1130 bevorzugen, sonst neuester lokaler Lauf
    local = next((r for r in locals_ if r["run"] == LOCAL_COHORT_RUN), None) \
        or (locals_[0] if locals_ else None)
    return {"local": local, "saas": saas}


def load_roster():
    """Kohorten-Plan aus config/testplan.yaml (stdlib-Regex, kein pyyaml — die
    Seite ist bewusst dependency-frei). Roster = Modelle, die zum Lauf gehören:
    active ODER explizit als N/A markiert. Liefert (name, profile, active, na)."""
    cfg = TESTPLAN / "config" / "testplan.yaml"
    try:
        txt = cfg.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for b in re.split(r"\n\s*-\s+name:\s*", txt)[1:]:
        name = b.splitlines()[0].strip().strip("\"'")
        mp = re.search(r'\n\s*profile:\s*"?([^"\n]+)"?', b)
        if not mp:  # nur echte Modell-Einträge (haben ein profile:) — keine Playbooks
            continue
        profile = mp.group(1).strip().strip("\"'")
        ma = re.search(r"\n\s*active:\s*(true|false)", b)
        active = (ma.group(1) == "true") if ma else True
        mn = re.search(r'\n\s*notes:\s*"?(.*)', b)
        note = mn.group(1) if mn else ""
        na = (not active) and bool(re.search(r"\bN/?A\b", name + " " + note))
        out.append({"name": name, "profile": profile, "active": active, "na": na})
    return out


def _scan_local_reports(run_dir):
    """ALLE lokalen Reports eines Laufs (auch fehlerhafte) → {name: info} mit
    Validitätsflag. Basis für die Roster-Statusspalte (gültig vs. degraded)."""
    out = {}
    if not run_dir:
        return out
    for j in sorted(Path(run_dir).glob("*.json")):
        if re.search(r"dashboard|index|summary", j.name, re.I):
            continue
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta, summ, pbs = d.get("meta", {}), d.get("summary", {}), d.get("playbooks", {})
        if meta.get("source") == "saas_proxy":
            continue
        total = err = 0
        for k, v in pbs.items():
            if k in EXCLUDE_PLAYBOOKS or not isinstance(v, dict):
                continue
            for res in v.get("results", []):
                total += 1
                if res.get("verdict") == "error":
                    err += 1
        rate = (err / total) if total else 1.0
        name = str(meta.get("model") or j.stem).rsplit("/", 1)[-1]
        if name in _EXCLUDE_MODELS:
            continue
        pr = {k: v.get("pass_rate") for k, v in pbs.items()
              if k not in EXCLUDE_PLAYBOOKS and isinstance(v, dict)}
        out[name] = {"stem": j.stem, "err_rate": rate, "valid": total > 0 and rate <= 0.3,
                     "pass_rate": summ.get("pass_rate"), "overall": summ.get("overall"),
                     "ko": summ.get("knockouts", 0), "pb": pr, "perf": _perf(pbs),
                     "profile": meta.get("profile", ""), "total": total}
    return out


def _running_profile():
    """Profil-Verzeichnis des aktuell servierten vLLM-Containers (für Status
    „läuft"). Best effort — fehlt Docker/Host, dann eben „ausstehend"."""
    try:
        import subprocess
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.startswith("vllm-") and "--" in ln:
            return ln[len("vllm-"):]
    return ""


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
    cols = [c for c in PLAYBOOK_LABELS if c != "06_performance"]
    header = (["Modell", "Release", "Gesamt", "K.O."] + [PLAYBOOK_LABELS[c] for c in cols]
              + ["Tok/s", "TTFT", "Lizenz"])
    rows = []
    for r in data["rows"]:
        ov = r["overall"] or "—"
        ov_html = f'<span class="ko">{esc(ov)}</span>' if ov == "K.O." else esc(ov)
        url, lbl = model_repo(r["profile"], r["is_saas"])
        if not url:  # SaaS ohne Anbieter-Link → HF-Card aus models.yaml (Kimi/MiniMax/DeepSeek/…)
            hr = _MODELS.get(r["model"], {}).get("hf_repo")
            url = ("https://huggingface.co/" + hr) if hr else url
        hf = f' <a href="{esc(url)}" title="Repo/Anbieter" target="_blank" rel="noopener">↗</a>' if url else ""
        link = f'<a href="m/{esc(r["stem"])}.html">{esc(r["model"])}</a>{hf}'
        cells = [link, release_date(r["model"]), f'{ov_html} {esc(r["pass_rate"])}%', str(r["ko"] or 0)]
        for c in cols:
            v = r["pb"].get(c)
            cells.append("—" if v is None else f"{round(float(v) * 100)}%")
        p = r.get("perf") or {}
        cells.append(f'{p["tok_median"]:.1f}' if p.get("tok_median") is not None else "—")
        cells.append(f'{p["ttft_p50"]:.0f} ms' if p.get("ttft_p50") is not None else "—")
        cells.append(esc(model_license(r["profile"], r["is_saas"])))
        rows.append(cells)
    best = data["rows"][0]
    sec = (f'<h2 id="{cid}">{esc(title)}</h2>\n'
           f'<p>{lead} {len(rows)} Modelle · Pass-Rate je Playbook. '
           f'<strong>Modellname anklicken</strong> → Detail (Prompt, Antwort, Judge je Fall). '
           f'<strong>Sicherheit (04) ausgeschlossen</strong>.</p>\n'
           f'<div style="overflow-x:auto">{table(header, rows)}</div>')
    c = card(card_title, str(len(rows)), f'Modelle · Top {esc(best["pass_rate"])}%', f"#{cid}")
    return sec, c


_STATUS_RANK = {"valid": 0, "running": 1, "degraded": 2, "pending": 3, "na": 4}
_STATUS_BADGE = {"valid": "✅ gültig", "running": "🔄 läuft", "degraded": "⚠ degraded",
                 "pending": "⏳ ausstehend", "na": "⛔ N/A"}


def llm_local_chapter(local, roster, reports, running_prof):
    """Volles Roster der lokalen Kohorte mit Statusspalte: jedes geplante Modell
    erscheint (gültig / läuft / degraded / ausstehend / N/A) — nicht nur die
    bestandenen. Gültige Zeilen sind verlinkt und voll bewertet."""
    cols = [c for c in PLAYBOOK_LABELS if c != "06_performance"]
    header = (["Modell", "Release", "Gesamt", "K.O."] + [PLAYBOOK_LABELS[c] for c in cols]
              + ["Tok/s", "TTFT", "Lizenz"])
    valid_by_name = {r["model"]: r for r in (local["rows"] if local else [])}
    entries, seen = [], set()
    for m in roster:
        name = m["name"]
        # Kohorte = geplant (active) ODER explizit N/A ODER hat einen Report.
        # Dormante Modelle (inactive, nie gelaufen) gehören nicht ins Roster.
        if not m["active"] and not m["na"] and name not in reports:
            continue
        seen.add(name); rep = reports.get(name)
        if name in valid_by_name:
            status = "valid"
        elif m["na"]:  # explizites N/A überstimmt einen degradierten Report
            status = "na"
        elif rep and rep["total"] and not rep["valid"]:
            status = "degraded"
        elif running_prof and m["profile"] and m["profile"] == running_prof:
            status = "running"
        else:
            status = "pending"
        entries.append((status, name, m, rep))
    for name, rep in reports.items():  # Reports ohne Roster-Eintrag (Sicherheit)
        if name in seen:
            continue
        entries.append(("valid" if rep["valid"] else "degraded", name,
                        {"name": name, "profile": rep["profile"], "na": False}, rep))

    def sk(e):
        status, name, _m, _r = e
        pr = float(valid_by_name[name]["pass_rate"] or 0) if status == "valid" and name in valid_by_name else 0.0
        return (_STATUS_RANK[status], -pr, name.lower())
    entries.sort(key=sk)

    dash = ["—"] * (len(cols) + 2)  # Playbooks + Tok/s + TTFT
    rows = []
    n_valid = 0
    for status, name, m, rep in entries:
        badge = f'<span class="badge {status}" data-sort="{_STATUS_RANK[status]}">{_STATUS_BADGE[status]}</span>'
        prof = m.get("profile", "") or (rep or {}).get("profile", "")
        lic = esc(model_license(prof, False))
        rel = release_date(name)
        if status == "valid":
            n_valid += 1
            r = valid_by_name[name]
            ov = r["overall"] or "—"
            ov_html = f'<span class="ko">{esc(ov)}</span>' if ov == "K.O." else esc(ov)
            url, _l = model_repo(r["profile"], False)
            hf = f' <a href="{esc(url)}" title="Repo" target="_blank" rel="noopener">↗</a>' if url else ""
            link = f'<a href="m/{esc(r["stem"])}.html">{esc(name)}</a>{hf}'
            cells = [link, rel, f'{ov_html} {esc(r["pass_rate"])}%', str(r["ko"] or 0)]
            for c in cols:
                v = r["pb"].get(c)
                cells.append("—" if v is None else f"{round(float(v) * 100)}%")
            p = r.get("perf") or {}
            cells.append(f'{p["tok_median"]:.1f}' if p.get("tok_median") is not None else "—")
            cells.append(f'{p["ttft_p50"]:.0f} ms' if p.get("ttft_p50") is not None else "—")
            cells.append(lic)
        elif status == "degraded":
            gesamt = f'<span class="ko">{round(rep["err_rate"] * 100)}% Fehler</span>'
            cells = [esc(name), rel, gesamt, str(rep.get("ko") or 0)]
            for c in cols:
                v = rep["pb"].get(c)
                cells.append("—" if v is None else f'<span class="note">{round(float(v) * 100)}%</span>')
            p = rep.get("perf") or {}
            cells.append(f'{p["tok_median"]:.1f}' if p.get("tok_median") is not None else "—")
            cells.append(f'{p["ttft_p50"]:.0f} ms' if p.get("ttft_p50") is not None else "—")
            cells.append(lic)
        else:  # running / pending / na
            cells = [esc(name), rel, "—", "—"] + dash + [lic]
        rows.append((status, cells))

    trs = "".join(f'<tr class="st-{st}">' + "".join(f"<td>{c}</td>" for c in cs) + "</tr>"
                  for st, cs in rows)
    th = "".join(f"<th{best_attr(h)}>{esc(h)}</th>" for h in header)
    tbl = f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"
    counts = {}
    for st, _cs in rows:
        counts[st] = counts.get(st, 0) + 1
    legend = " · ".join(f'{_STATUS_BADGE[s]} {counts[s]}' for s in _STATUS_RANK if counts.get(s))
    sec = (f'<h2 id="llm-local">LLM — Lokale Modelle (DGX Spark)</h2>\n'
           f'<p>Auf dem GB10 selbst serviert (vLLM), Judge-bewertet. <strong>Vollständiges '
           f'Kohorten-Roster</strong> — jedes geplante Modell mit Status, nicht nur die bestandenen. '
           f'{len(rows)} Modelle ({legend}). '
           f'<strong>Gültige Modelle anklicken</strong> → Detail (Prompt, Antwort, Judge je Fall). '
           f'<strong>Sicherheit (04) ausgeschlossen</strong>.</p>\n'
           f'<div style="overflow-x:auto">{tbl}</div>')
    c = card("LLM lokal", f'{n_valid}/{len(rows)}', "gültig · volles Roster", "#llm-local")
    return sec, c


def guards_section(guards):
    if not guards:
        return "", card("Guards", "—", "kein Feldlauf")
    # K.O. (bei Guards ohne Judge kaum unterscheidend) und Lat p95 (zu spezifisch,
    # Lat ø reicht) fliegen zugunsten der Lizenz-Spalte raus.
    hide = {"n_unsafe", "n_safe", "fn_rate", "latency_ms_p95"}
    keys = []
    for g in guards:
        for k in g["metrics"]:
            if k not in hide and isinstance(g["metrics"][k], (int, float)) and k not in keys:
                keys.append(k)

    _GN = {"granite-guardian": "Granite-Guardian-4.1-8B", "gpt-oss-safeguard": "gpt-oss-safeguard-20b",
           "nemotron-3-5": "Nemotron-3.5-Content-Safety", "nemotron-3": "Nemotron-3-Content-Safety",
           "shieldstral": "Shieldstral-1.0-3B"}

    def glabel(g):
        return (f'<a href="g/{esc(g["slug"])}.html">{esc(g["label"])}</a>'
                if g.get("per_case") else esc(g["label"]))

    def glic(g):
        return esc(_MODELS.get(_GN.get(g["slug"], ""), {}).get("license", "—") or "—")
    rows = [[glabel(g), release_date(_GN.get(g["slug"], ""))] + [num(g["metrics"].get(k)) for k in keys]
            + [glic(g)] for g in guards]
    best = max(guards, key=lambda g: g["metrics"].get("f1", 0) or 0)
    sec = (f'<h2 id="guards">Guardrails (Playbook 08)</h2>\n'
           f'<p class="note">Guard-Name anklicken → Fall für Fall (Wahrheit vs. Vorhersage). '
           f'Kein Judge — das Label ist die Wahrheit.</p>\n'
           f'<div class="gtbl" style="overflow-x:auto">'
           f'{table(["Guard", "Release"] + [_GUARD_KEY_LABEL.get(k, k) for k in keys] + ["Lizenz"], rows)}</div>')
    c = card("Guards", f'{(best["metrics"].get("f1", 0) or 0):.3f}', f'bestes F1 · {best["label"]}', "#guards")
    return sec, c


# ── Detailseiten ─────────────────────────────────────────────────────────────
def _llm_teaser(meta, summ, pbs):
    is_saas = meta.get("source") == "saas_proxy"
    src = "SaaS · LiteLLM-Proxy" if is_saas else "lokal · DGX Spark / vLLM"
    prof_label = "Proxy-ID" if is_saas else "Profil / Quant"
    profile = meta.get("profile", "")
    facts = [f'<span><b>Quelle</b>{esc(src)}</span>',
             f'<span><b>{prof_label}</b><code>{esc(profile or "—")}</code></span>',
             f'<span><b>Lizenz</b>{esc(model_license(profile, is_saas))}</span>',
             f'<span><b>Judge</b><code>{esc(meta.get("judge", "—"))}</code></span>']
    url, lbl = model_repo(profile, is_saas)
    if url:
        facts.insert(3, f'<span><b>Repo</b><a href="{esc(url)}" target="_blank" rel="noopener">{esc(lbl)}</a></span>')
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
    perf = _perf(pbs)
    if perf:
        parts.append(_perf_section(perf))
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
             f'<span><b>Lizenz</b>{esc(model_license(served, False))}</span>',
             f'<span><b>Protokoll</b><code>{esc(g.get("protocol", "—"))}</code></span>']
    url, lbl = model_repo(served, False)
    if url:
        facts.insert(3, f'<span><b>Repo</b><a href="{esc(url)}" target="_blank" rel="noopener">{esc(lbl)}</a></span>')
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


def _landing_page(title, sections):
    """Verzeichnisseite (Tabelle je Gruppe) für docs/<subdir>/ — verhindert 404."""
    body = []
    for heading, hdr, rows in sections:
        if not rows:
            continue
        body.append(f'<h2>{esc(heading)}</h2>\n<div style="overflow-x:auto">{table(hdr, rows)}</div>')
    return page_shell(title, "\n".join(body), back="../index.html")


def generate_details(runs, guards, prompts, gprompts):
    (DOCS / "m").mkdir(parents=True, exist_ok=True)
    (DOCS / "g").mkdir(parents=True, exist_ok=True)
    m_sections = []
    for kind, gtitle in (("local", "Lokale Modelle (DGX Spark)"), ("saas", "SaaS-Referenzkohorte")):
        data = runs.get(kind)
        if not data:
            continue
        rows = []
        for row in data["rows"]:
            (DOCS / "m" / f"{row['stem']}.html").write_text(llm_detail_html(row, prompts), encoding="utf-8")
            ov = row["overall"] or "—"
            ov_html = f'<span class="ko">{esc(ov)}</span>' if ov == "K.O." else esc(ov)
            url, _lbl = model_repo(row["profile"], row["is_saas"])
            hf = f' <a href="{esc(url)}" title="Repo/Anbieter" target="_blank" rel="noopener">↗</a>' if url else ""
            rows.append([f'<a href="{esc(row["stem"])}.html">{esc(row["model"])}</a>{hf}',
                         f'{ov_html} {esc(row["pass_rate"])}%',
                         esc(model_license(row["profile"], row["is_saas"]))])
        m_sections.append((gtitle, ["Modell", "Gesamt", "Lizenz"], rows))
    g_rows, n_g = [], 0
    for g in guards:
        if not g.get("per_case"):
            continue
        (DOCS / "g" / f"{g['slug']}.html").write_text(guard_detail_html(g, gprompts), encoding="utf-8")
        n_g += 1
        m = g["metrics"]
        g_rows.append([f'<a href="{esc(g["slug"])}.html">{esc(g["label"])}</a>',
                       num(m.get("f1")), num(m.get("recall")), num(m.get("fpr")), num(m.get("trap_fpr")),
                       "✓" if not g["knockouts"] else '<span class="ko">K.O.</span>'])
    (DOCS / "m" / "index.html").write_text(
        _landing_page("LLM-Detailseiten", m_sections), encoding="utf-8")
    (DOCS / "g" / "index.html").write_text(
        _landing_page("Guard-Detailseiten",
                      [("Guards", ["Guard", "F1", "Recall", "FPR", "Trap-FPR", "K.O."], g_rows)]),
        encoding="utf-8")
    return sum(len(s[2]) for s in m_sections), n_g


def build_index(runs, guards, roster, reports, running_prof):
    local_sec, local_card = llm_local_chapter(runs["local"], roster, reports, running_prof)
    saas_sec, saas_card = llm_chapter(runs["saas"], "llm-saas", "LLM — SaaS-Referenzkohorte",
                                      "Frontier-Modelle über <b>LiteLLM</b> als Referenzrahmen — ein Endpoint für "
                                      "Cloud- und lokale Modelle, von SouthByte empfohlen. Gleicher Testsatz, "
                                      "Judge <code>claude-sonnet-5</code>. Tok/s &amp; TTFT messen hier Cloud+Netz "
                                      "(nicht lokale Hardware).", "LLM SaaS")
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
    roster = load_roster()
    run_dir = (REPORTS_DIR / runs["local"]["run"]) if runs.get("local") else None
    reports = _scan_local_reports(run_dir)
    running_prof = _running_profile()
    (DOCS / "index.html").write_text(build_index(runs, guards, roster, reports, running_prof), encoding="utf-8")
    n_llm, n_guard = generate_details(runs, guards, prompts, gprompts)
    if _lic_dirty[0]:
        _LIC_CACHE_FILE.write_text(json.dumps(_lic_disk, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ docs/index.html + {n_llm} LLM-Detail + {n_guard} Guard-Detail (+ Landing m/ g/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
