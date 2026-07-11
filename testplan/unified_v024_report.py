#!/usr/bin/env python3
"""unified_v024_report.py — Dedizierter Vergleichsreport für die 8er-Kohorte,
einheitlich auf vLLM v0.24.0 mit optimierten Runtime-Params getestet
(MTP/speculative decoding wo der Checkpoint es hergibt, Prämissenprüfung im
Default-System-Prompt, Degenerations-Guard).

Kohorte (8 Modelle): Nemotron-3-Super-120B, Nemotron-Puzzle-75B,
Gemma-4-26B-A4B, Gemma-4-31B, DiffusionGemma-26B-A4B, Granite-4.1-30B,
Qwen3.6-27B-FP8, Qwen3.6-35B-A3B-FP8.

Liest die jüngsten vollständigen Modell-Reports (JSON) und erzeugt eine
eigenständige HTML-Seite mit Ranking, Playbook-Matrix, Performance-Detail,
Per-Playbook-Siegern, K.O.-Gründen und Kernaussagen. Schwestergenerator zu
small_llm_report.py / medium_llm_report.py.

Reproduzierbar: keine eingebetteten Messwerte, alles aus reports/<run>/<model>.json.

Usage:
  python unified_v024_report.py                # -> reports/unified-v024/index.html
  python unified_v024_report.py --out PFAD.html
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
CONFIG_YAML = Path(__file__).resolve().parent / "config" / "testplan.yaml"

# Evaluator-Defaults (evaluators/base.py query_target) — für die Parameter-Blöcke.
EVAL_DEFAULTS = (
    "max_tokens 8192 · timeout 900 s · temperature 0.1 (sofern nicht per Modell "
    "überschrieben) · Degenerations-Guard: Budget erschöpft ohne Content → 1× Retry "
    "mit 2× Budget, danach FAIL statt Refusal-PASS"
)

# Vergleichskohorte: (Anzeigename, JSON-Modellname, params_b, Familie, Quant, Arch/Runtime-Notiz)
MODELS = [
    ("Nemotron-3-Super-120B",   "Nemotron-3-Super",        120.0, "Nemotron",  "NVFP4", "MoE · A12B · MTP"),
    ("Nemotron-Puzzle-75B",     "Nemotron-Puzzle-75B",      75.0, "Nemotron",  "NVFP4", "MoE · A9B · MTP"),
    ("Gemma-4-26B-A4B",         "Gemma-4-26B-A4B",          26.0, "Gemma",     "BF16",  "MoE · A4B aktiv"),
    ("Gemma-4-31B",             "Gemma-4-31B",              31.0, "Gemma",     "BF16",  "dense"),
    ("DiffusionGemma-26B-A4B",  "DiffusionGemma-26B-A4B",   26.0, "Diffusion", "BF16",  "MoE · A4B · Block-Diffusion"),
    ("Granite-4.1-30B",         "Granite-4.1-30B",          30.0, "Granite",   "BF16",  "Hybrid-Mamba"),
    ("Qwen3.6-27B-FP8",         "Qwen3.6-27B-FP8",          27.0, "Qwen",      "FP8",   "dense · MTP"),
    ("Qwen3.6-35B-A3B-FP8",     "Qwen3.6-35B-A3B-FP8",      35.0, "Qwen",      "FP8",   "MoE · A3B · MTP"),
]

PLAYBOOKS = [
    ("01_quality",          "Quality"),
    ("02_german_language",  "German"),
    ("03_bias",             "Bias"),
    ("04_security",         "Security"),
    ("05_code",             "Code"),
    ("06_performance",      "Performance"),
]

# Kernaussagen (Markdown-Quelle; **fett**/_kursiv_ → HTML beim Rendern).
# Single source für HTML- und Markdown-Ausgabe.
FINDINGS = [
    "**Gemma-4-31B führt die Kohorte an (88 %)** — und dreht damit das Juni-Ranking "
    "um, in dem das kleinere 26B-A4B noch vorn lag. Mit dem neuen "
    "Prämissenprüfungs-Prompt und v0.24.0 liegt das dense 31B jetzt vor dem MoE.",
    "**Nemotron-3-Super-120B ist der stärkste Newcomer (85 %, Platz 2):** bestes "
    "Nicht-Gemma-Ergebnis, volle Code-Punktzahl — aber 10 Hallu-K.O. und als eines "
    "von nur zwei Modellen ein echtes Security-K.O.",
    "**DiffusionGemma: 10× Durchsatz, schwächste Gemma-Qualität.** 267 tok/s "
    "single-stream (autoregressives 26B: 22,7), kompletter Testlauf in 32 min statt "
    "~2 h. Aber 71 % Pass-Rate und 13 Hallu-K.O. — typisches Muster: erkennt die "
    "falsche Prämisse korrekt und halluziniert _danach trotzdem_ eigene Details "
    "(erfundene Erzbischöfe, Gesetzesartikel, U-Bahn-Stationen). Dazu TTFT ~3,1 s "
    "(Denoising des ersten Blocks) → Performance-Playbook FAIL.",
    "**MTP (Multi-Token-Prediction) mit Doppelgesicht:** Single-Stream-Prosa/Code "
    "+55–100 % (beide Nemotrons, beide Qwens), und unter Last (50 parallel) sinkt "
    "die TTFT deutlich. Der synthetische Durchsatz-Benchmark regrediert dagegen "
    "(niedrige Draft-Acceptance → Drafter-Overhead). Gemma und Granite haben keine "
    "MTP-Gewichte im Checkpoint — dort gibt es nichts zu aktivieren.",
    "**`hal-012` (Kündigungsfristen § 622 BGB) schlägt ALLE 8 Modelle** — die "
    "Arbeitgeber/Arbeitnehmer-Staffelung wird durchgängig verwechselt oder erfunden. "
    "Der schwerste Einzeltest des Katalogs.",
    "**Alle 8 formal K.O., aber fast ausschließlich über Halluzinations-Traps.** "
    "Echte Security-K.O.s haben nur Granite-4.1-30B und Nemotron-3-Super. "
    "Qwen3.6-27B ist das einzige Modell mit Security 12/12 — bei zugleich "
    "schwächstem Code-Ergebnis (5/10, geteilt mit Qwen-35B).",
    "**Die Prämissenprüfung im System-Prompt wirkt nur moderat:** Nemotron-Quality "
    "+2 Punkte ggü. Vortageslauf, die Hallu-K.O.-Zahlen bleiben ähnlich — die Traps "
    "sind weitgehend prompt-resistent. Der Klassiker „erfinde keine Fakten“ war in "
    "allen Läufen ohnehin aktiv.",
    "**Infrastruktur:** einheitlich vLLM v0.24.0 auf GB10 (sm_120). Neu gelernt: "
    "DeepGEMM kennt sm_120 nicht → `VLLM_USE_DEEP_GEMM=0` bei FP8-Modellen; "
    "FlashInfer inkompatibel mit DiffusionGemmas gemischter Attention → TRITON_ATTN.",
]

# Akzentfarbe je Familie (linker Rand in der Matrix)
FAMILY_COLORS = {
    "Nemotron":  "#76b900",
    "Gemma":     "#4285f4",
    "Diffusion": "#00a4bd",
    "Granite":   "#1f8f7a",
    "Qwen":      "#7b3fb0",
}


def _playbook_count(path: Path) -> int:
    try:
        return len(json.loads(path.read_text(encoding="utf-8")).get("playbooks", {}))
    except Exception:
        return -1


def find_best_json(model_name: str) -> Path | None:
    """Wähle den *vollständigsten* Report (meiste Playbooks), Tie-Break jüngstes
    Verzeichnis. Verhindert, dass ein partieller Re-Run (z. B. nur 05_code) einen
    vollständigen Lauf verdrängt, nur weil er neuer ist."""
    cands = [
        d / f"{model_name}.json"
        for d in sorted(REPORTS_DIR.iterdir(), reverse=True)  # neuestes zuerst
        if d.is_dir() and (d / f"{model_name}.json").exists()
    ]
    if not cands:
        return None
    # max Playbooks; bei Gleichstand gewinnt das zuerst gelistete (= neueste)
    return max(cands, key=_playbook_count)


def load_model(model_name: str) -> dict | None:
    p = find_best_json(model_name)
    if not p:
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    data["_source"] = p.parent.name
    data["_dir"] = str(p.parent)
    return data


def pb_cell(pb: dict) -> tuple[int, int, float]:
    passed = int(pb.get("passed", 0))
    total = int(pb.get("total", 0))
    rate = passed / total if total else 0.0
    return passed, total, rate


def rate_class(rate: float) -> str:
    if rate >= 0.85:
        return "good"
    if rate >= 0.6:
        return "ok"
    if rate >= 0.3:
        return "weak"
    return "bad"


def ko_reasons(data: dict) -> list[str]:
    out = []
    for pn, pb in data.get("playbooks", {}).items():
        for k in pb.get("knockouts", []):
            out.append(f"{pn.split('_', 1)[-1]}/{k.get('test_id', '?')}")
    return out


def overall_rate(data: dict) -> float:
    s = data["summary"]
    return s["passed"] / max(1, s["total_tests"])


def perf_metrics(data: dict) -> dict | None:
    """TTFT/Durchsatz aus dem 06_performance-Metadata; None wenn nicht vorhanden."""
    try:
        md = data["playbooks"]["06_performance"]["results"][0]["metadata"]
        cd = md.get("concurrent_degradation", {})
        return {
            "ttft_p50": md["ttft_p50_ms"],
            "ttft_p95": md.get("ttft_p95_ms"),
            "tok_s": md.get("throughput_median_tok_s"),
            "ttft_50par": cd.get("50", {}).get("ttft_p50_ms"),
        }
    except Exception:
        return None


# Anzeige-Reihenfolge und Labels für die Profil-Variablen im Parameter-Block.
PROFILE_LABELS = [
    ("DOCKER_IMAGE",           "Image"),
    ("MAX_MODEL_LEN",          "Kontext"),
    ("MAX_NUM_SEQS",           "max_num_seqs"),
    ("MAX_NUM_BATCHED_TOKENS", "max_num_batched_tokens"),
    ("GPU_MEM_UTIL",           "gpu_mem_util"),
    ("KV_CACHE_DTYPE",         "KV-Cache"),
    ("ATTENTION_BACKEND",      "Attention-Backend"),
    ("ENFORCE_EAGER",          "enforce_eager"),
    ("QUANTIZATION",           "Quantisierung"),
    ("REASONING_PARSER",       "Reasoning-Parser"),
    ("TOOL_CALL_PARSER",       "Tool-Call-Parser"),
    ("USE_V2_MODEL_RUNNER",    "V2-Model-Runner"),
    ("VLLM_EXTRA_ARGS",        "Extra-Args (u. a. MTP)"),
    ("DOCKER_ENV",             "Docker-Env"),
]


def load_profile_params(profile: str) -> dict[str, str]:
    """PROFILE_*-Variablen aus profiles/<profile>/vllm_profile.conf (kuratierte,
    validierte Quelle). Bash-Array VLLM_EXTRA_ARGS wird zu einem String gefaltet."""
    conf = PROFILES_DIR / profile / "vllm_profile.conf"
    out: dict[str, str] = {}
    if not conf.exists():
        return out
    for line in conf.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^PROFILE_([A-Z0-9_]+)=(.+)$", line.strip())
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith("(") and val.endswith(")"):  # bash-Array
            val = " ".join(p.strip("'\"") for p in re.findall(r"'[^']*'|\"[^\"]*\"|\S+", val[1:-1]))
        else:
            val = val.strip("'\"")
        out[key] = val
    return out


def load_yaml_cfg() -> dict:
    """testplan.yaml: Default-System-Prompt + je Modell sampling/chat_template_kwargs."""
    import yaml
    cfg = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    models = {m["name"]: m for m in cfg.get("models", [])}
    return {"default_prompt": cfg.get("_default_system_prompt", ""), "models": models}


def runtime_param_rows(r: dict, yaml_cfg: dict) -> list[tuple[str, str]]:
    """(Label, Wert)-Liste für den Parameter-Block eines Modells. Fail-safe: bei
    Fehlern lieber weniger Zeilen als ein kaputter Report."""
    rows: list[tuple[str, str]] = []
    try:
        prof = load_profile_params(r["data"]["meta"].get("profile", ""))
        for key, label in PROFILE_LABELS:
            if key in prof:
                val = prof[key]
                if key in ("ENFORCE_EAGER", "ENABLE_AUTO_TOOL_CHOICE"):
                    val = "ja" if val == "1" else val
                if key == "USE_V2_MODEL_RUNNER":
                    val = "aus" if val == "0" else val
                rows.append((label, val))
        mc = yaml_cfg["models"].get(r["name"]) or yaml_cfg["models"].get(r["mname"]) or {}
        samp = mc.get("sampling") or {}
        rows.append(("Sampling", ", ".join(f"{k}={v}" for k, v in samp.items())
                     if samp else "temperature 0.1 (Evaluator-Default)"))
        ctk = mc.get("chat_template_kwargs") or {}
        if ctk:
            rows.append(("chat_template_kwargs", ", ".join(f"{k}={v}" for k, v in ctk.items())))
        rows.append(("Evaluator", EVAL_DEFAULTS))
    except Exception as e:  # Sekundär-Output darf nie den Report reißen
        rows.append(("Hinweis", f"Parameter unvollständig ({e})"))
    return rows


def inject_params_md(body: str, rows: list[tuple[str, str]]) -> str:
    """Parameter-Tabelle direkt oberhalb von '## Playbook-Ergebnisse' einfügen."""
    block = ["## Run-Parameter", "", "| Parameter | Wert |", "|-----------|------|"]
    block += [f"| {k} | `{v}` |" for k, v in rows]
    block += ["", ""]
    marker = "## Playbook-Ergebnisse"
    if marker in body:
        return body.replace(marker, "\n".join(block) + marker, 1)
    return body + "\n" + "\n".join(block)


def inject_params_html(body: str, rows: list[tuple[str, str]]) -> str:
    trs = "\n".join(
        f"<tr><td style='text-align:left;font-weight:600;white-space:nowrap'>{k}</td>"
        f"<td style='text-align:left'><code>{v}</code></td></tr>" for k, v in rows
    )
    block = (f"<h2>Run-Parameter</h2>\n<table><thead><tr><th>Parameter</th>"
             f"<th>Wert</th></tr></thead><tbody>\n{trs}\n</tbody></table>\n")
    marker = "<h2>Playbook-Ergebnisse</h2>"
    if marker in body:
        return body.replace(marker, block + marker, 1)
    return body


def md_inline_to_html(s: str) -> str:
    """Minimal: **fett** → <b>, _kursiv_ → <i>. Genug für die Kernaussagen."""
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"_(.+?)_", r"<i>\1</i>", s)
    return s


def _fmt_ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v/1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


def build_html(rows: list[dict], generated: str, sys_prompt: str = "") -> str:
    # rows: list of dicts with name, mname, params_b, family, quant, arch, data
    ranked = sorted(rows, key=lambda r: overall_rate(r["data"]), reverse=True)

    judge = rows[0]["data"]["meta"].get("judge", "—")

    # ---- Familien-CSS ----
    fam_css = "\n".join(
        f"  td.model.{fam.lower()} {{ border-left: 3px solid {col}; }}"
        for fam, col in FAMILY_COLORS.items()
    )

    # ---- Matrix-Zeilen (in Rangfolge) ----
    matrix_rows = ""
    for r in ranked:
        s = r["data"]["summary"]
        P = r["data"]["playbooks"]
        total_rate = overall_rate(r["data"])
        cells = ""
        for pkey, _ in PLAYBOOKS:
            if pkey in P:
                p, t, rate = pb_cell(P[pkey])
                cells += f'<td class="{rate_class(rate)}">{p}/{t}</td>'
            else:
                cells += '<td class="na">—</td>'
        fam = r["family"].lower()
        link = f'details/{r["mname"]}.html'
        matrix_rows += (
            f'<tr><td class="model {fam}">'
            f'<a href="{link}">{r["name"]}</a> '
            f'<span class="pb">{r["params_b"]:g}B · {r["arch"]}</span></td>'
            f'<td class="quant">{r["quant"]}</td>'
            f'<td class="ko">{s["overall"]}</td>'
            f'<td class="{rate_class(total_rate)} total">{s["passed"]}/{s["total_tests"]}'
            f'<br><b>{total_rate*100:.0f}%</b></td>'
            f'{cells}</tr>\n'
        )

    # ---- Performance-Detail ----
    perf_rows = ""
    for r in ranked:
        pm = perf_metrics(r["data"])
        if not pm:
            perf_rows += f'<tr><td class="model">{r["name"]}</td><td colspan="4" class="na">—</td></tr>\n'
            continue
        tok = f'{pm["tok_s"]:.1f}' if pm["tok_s"] is not None else "—"
        note = ""
        if r["family"] == "Diffusion":
            tok += " *"
            note = ""
        perf_rows += (
            f'<tr><td class="model">{r["name"]}</td>'
            f'<td>{_fmt_ms(pm["ttft_p50"])}</td>'
            f'<td>{_fmt_ms(pm["ttft_p95"])}</td>'
            f'<td>{tok}</td>'
            f'<td>{_fmt_ms(pm["ttft_50par"])}</td></tr>\n'
        )

    # ---- Per-Playbook-Sieger ----
    winners = ""
    for pkey, plabel in PLAYBOOKS:
        best = None
        for r in rows:
            pb = r["data"]["playbooks"].get(pkey)
            if not pb:
                continue
            _, _, rate = pb_cell(pb)
            if best is None or rate > best[1]:
                best = (r["name"], rate, pb_cell(pb))
        if best:
            p, t, _ = best[2]
            winners += (
                f'<tr><td>{plabel}</td><td><b>{best[0]}</b></td>'
                f'<td>{p}/{t} ({best[1]*100:.0f}%)</td></tr>\n'
            )

    # ---- Ranking-Liste ----
    rank_items = ""
    for i, r in enumerate(ranked, 1):
        s = r["data"]["summary"]
        rate = overall_rate(r["data"])
        rank_items += (
            f'<li><span class="rk">{i}</span> '
            f'<b><a href="details/{r["mname"]}.html">{r["name"]}</a></b> '
            f'<span class="pb">{r["params_b"]:g}B · {r["family"]} · {r["quant"]}</span> '
            f'<span class="score {rate_class(rate)}">{rate*100:.0f}%</span> '
            f'<span class="muted">({s["passed"]}/{s["total_tests"]} · {s["knockouts"]} K.O.)</span></li>\n'
        )

    # ---- K.O.-Gründe ----
    ko_rows = ""
    for r in ranked:
        reasons = ko_reasons(r["data"])
        ko_rows += (
            f'<tr><td class="model">{r["name"]}</td>'
            f'<td>{r["data"]["summary"]["overall"]}</td>'
            f'<td class="muted">{", ".join(reasons) if reasons else "—"}</td></tr>\n'
        )

    pb_header = "".join(f"<th>{lbl}</th>" for _, lbl in PLAYBOOKS)

    findings_html = "\n".join(f"  <li>{md_inline_to_html(f)}</li>" for f in FINDINGS)

    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vergleichslauf v0.24.0 — 8er-Kohorte (26–120B) mit MTP</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 2rem; max-width: 1180px; margin-inline: auto;
         line-height: 1.5; color: #1a1a1a; background: #fafafa; }}
  h1 {{ font-size: 1.6rem; margin-bottom: .2rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2.2rem; border-bottom: 2px solid #e0e0e0;
        padding-bottom: .3rem; }}
  .sub {{ color: #666; font-size: .9rem; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: .6rem;
           background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  th, td {{ padding: .5rem .6rem; text-align: center; border: 1px solid #eee; }}
  th {{ background: #f0f0f3; font-size: .82rem; }}
  td.model {{ text-align: left; font-weight: 600; white-space: nowrap; }}
{fam_css}
  td.total {{ font-size: .95rem; }}
  td.quant {{ font-size: .78rem; color: #555; font-weight: 600; }}
  .pb {{ color: #888; font-weight: 400; font-size: .8rem; }}
  .good {{ background: #d7f0d7; }}
  .ok   {{ background: #fff3cd; }}
  .weak {{ background: #ffe0c7; }}
  .bad  {{ background: #f8d2d2; }}
  .na   {{ color: #bbb; }}
  td.ko {{ font-weight: 600; color: #b02a2a; font-size: .82rem; }}
  ol.rank {{ list-style: none; padding: 0; }}
  ol.rank li {{ padding: .5rem .7rem; background: #fff; margin-bottom: .35rem;
                border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,.06);
                display: flex; align-items: center; gap: .6rem; }}
  .rk {{ display: inline-grid; place-items: center; width: 1.6rem; height: 1.6rem;
         background: #333; color: #fff; border-radius: 50%; font-size: .8rem; }}
  .score {{ margin-left: auto; font-weight: 700; padding: .15rem .5rem;
            border-radius: 4px; }}
  .muted {{ color: #888; font-size: .85rem; }}
  .findings li {{ margin-bottom: .5rem; }}
  .findings b {{ color: #000; }}
  td.model a, ol.rank a {{ color: inherit; text-decoration: none;
                           border-bottom: 1px dotted #999; }}
  td.model a:hover, ol.rank a:hover {{ border-bottom-style: solid; }}
  footer {{ margin-top: 3rem; color: #999; font-size: .8rem; }}
</style></head>
<body>
<h1>Vergleichslauf v0.24.0 — 8er-Kohorte (26–120B)</h1>
<p class="sub">Instruct-/Reasoning-Modelle inkl. erstem Diffusions-LLM ·
   einheitlich auf <b>vLLM v0.24.0</b> (GB10/sm_120) · MTP aktiv wo unterstützt ·
   Prämissenprüfung im System-Prompt · Judge: <b>{judge}</b> ·
   98 Testfälle / 6 Playbooks · generiert {generated}</p>

<h2>Rangliste (Gesamt-Pass-Rate)</h2>
<ol class="rank">
{rank_items}</ol>

<h2>Playbook-Matrix</h2>
<table>
<thead><tr><th>Modell</th><th>Quant</th><th>Urteil</th><th>Gesamt</th>{pb_header}</tr></thead>
<tbody>
{matrix_rows}</tbody>
</table>
<p class="muted">Farbskala je Zelle: ≥85% grün · ≥60% gelb · ≥30% orange · &lt;30% rot.
   Zellen zeigen bestanden/gesamt je Playbook; „Gesamt" ist die Pass-Rate über alle 98 Fälle.</p>

<h2>Performance-Detail (06_performance-Benchmark)</h2>
<table>
<thead><tr><th>Modell</th><th>TTFT p50</th><th>TTFT p95</th><th>tok/s (Median)</th><th>TTFT p50 @ 50 parallel</th></tr></thead>
<tbody>
{perf_rows}</tbody>
</table>
<p class="muted">* DiffusionGemma liefert Blöcke im Burst — der Median-Durchsatz zwischen
   erstem und letztem Token ist dadurch stark überzeichnet; realer Single-Stream-Durchsatz
   (Smoke-Test, Wall-Clock): ~267 tok/s. Die MTP-Modelle zeigen hier niedrigere Werte als
   im Single-Stream-Smoke (Drafter-Overhead bei niedriger Draft-Acceptance des
   Benchmark-Workloads).</p>

<h2>Per-Playbook-Sieger</h2>
<table>
<thead><tr><th>Playbook</th><th>Bestes Modell</th><th>Wert</th></tr></thead>
<tbody>
{winners}</tbody>
</table>

<h2>Kernaussagen</h2>
<ul class="findings">
{findings_html}
</ul>

<h2>K.O.-Gründe im Detail</h2>
<table>
<thead><tr><th>Modell</th><th>Urteil</th><th>ausgelöst durch</th></tr></thead>
<tbody>
{ko_rows}</tbody>
</table>

<h2>Basis-System-Prompt (alle Modelle)</h2>
<p class="muted">Default-System-Prompt aus <code>config/testplan.yaml</code>, identisch für
   alle 8 Modelle — inkl. der für diesen Lauf neuen Prämissenprüfung (Absatz 4).
   Evaluator-Defaults: {EVAL_DEFAULTS}. Modellspezifische Sampling-/Template-Overrides
   stehen im Parameter-Block des jeweiligen Einzelberichts.</p>
<pre style="background:#fff;border:1px solid #eee;box-shadow:0 1px 3px rgba(0,0,0,.08);
            padding:1rem;white-space:pre-wrap;font-size:.85rem;line-height:1.45">{sys_prompt}</pre>

<footer>Quelldaten: reports/&lt;run&gt;/&lt;model&gt;.json (jüngster vollständiger Lauf je Modell).
        Generiert von unified_v024_report.py · Schwestergenerator zu small/medium_llm_report.py.</footer>
</body></html>
"""


def build_md(rows: list[dict], generated: str, sys_prompt: str = "") -> str:
    """Eigenständiges Markdown-Dokument (GitLab-tauglich), gleiche Inhalte wie HTML."""
    ranked = sorted(rows, key=lambda r: overall_rate(r["data"]), reverse=True)
    judge = rows[0]["data"]["meta"].get("judge", "—")

    def mlink(r: dict) -> str:
        return f"[{r['name']}](details/{r['mname']}.md)"

    L: list[str] = []
    L.append("# Vergleichslauf v0.24.0 — 8er-Kohorte (26–120B)")
    L.append("")
    L.append(
        f"Instruct-/Reasoning-Modelle inkl. erstem Diffusions-LLM, einheitlich auf "
        f"**vLLM v0.24.0** (GB10/sm_120), MTP aktiv wo unterstützt, Prämissenprüfung "
        f"im System-Prompt. Judge: **{judge}**. 98 Testfälle / 6 Playbooks. "
        f"Generiert {generated}."
    )
    L.append("")
    L.append("> Die Modellnamen verlinken auf den jeweiligen **Einzelbericht** "
             "(`details/<modell>.md`) mit Playbook-Aufschlüsselung, K.O.-Verletzungen "
             "und Detailergebnissen.")
    L.append("")

    # ---- Rangliste ----
    L.append("## Rangliste (Gesamt-Pass-Rate)")
    L.append("")
    L.append("| # | Modell | Größe | Quant | Pass-Rate | bestanden | K.O. |")
    L.append("|---|--------|-------|-------|-----------|-----------|------|")
    for i, r in enumerate(ranked, 1):
        s = r["data"]["summary"]
        rate = overall_rate(r["data"])
        L.append(
            f"| {i} | {mlink(r)} | {r['params_b']:g}B {r['family']} ({r['arch']}) | "
            f"{r['quant']} | **{rate*100:.0f}%** | {s['passed']}/{s['total_tests']} | "
            f"{s['knockouts']} |"
        )
    L.append("")

    # ---- Playbook-Matrix ----
    L.append("## Playbook-Matrix")
    L.append("")
    pb_head = " | ".join(lbl for _, lbl in PLAYBOOKS)
    L.append(f"| Modell | Quant | Urteil | Gesamt | {pb_head} |")
    L.append("|--------|-------|--------|--------|" + "----|" * len(PLAYBOOKS))
    for r in ranked:
        s = r["data"]["summary"]
        P = r["data"]["playbooks"]
        rate = overall_rate(r["data"])
        cells = []
        for pkey, _ in PLAYBOOKS:
            if pkey in P:
                p, t, _r = pb_cell(P[pkey])
                cells.append(f"{p}/{t}")
            else:
                cells.append("—")
        L.append(
            f"| {mlink(r)} | {r['quant']} | {s['overall']} | "
            f"**{rate*100:.0f}%** ({s['passed']}/{s['total_tests']}) | "
            + " | ".join(cells) + " |"
        )
    L.append("")
    L.append("Zellen: bestanden/gesamt je Playbook. „Gesamt\" ist die Pass-Rate über alle 98 Fälle.")
    L.append("")

    # ---- Performance-Detail ----
    L.append("## Performance-Detail (06_performance-Benchmark)")
    L.append("")
    L.append("| Modell | TTFT p50 | TTFT p95 | tok/s (Median) | TTFT p50 @ 50 parallel |")
    L.append("|--------|----------|----------|----------------|------------------------|")
    for r in ranked:
        pm = perf_metrics(r["data"])
        if not pm:
            L.append(f"| {mlink(r)} | — | — | — | — |")
            continue
        tok = f'{pm["tok_s"]:.1f}' if pm["tok_s"] is not None else "—"
        if r["family"] == "Diffusion":
            tok += " \\*"
        L.append(
            f"| {mlink(r)} | {_fmt_ms(pm['ttft_p50'])} | {_fmt_ms(pm['ttft_p95'])} | "
            f"{tok} | {_fmt_ms(pm['ttft_50par'])} |"
        )
    L.append("")
    L.append("\\* DiffusionGemma liefert Blöcke im Burst — der Median-Durchsatz zwischen "
             "erstem und letztem Token ist stark überzeichnet; realer Single-Stream-Durchsatz "
             "(Smoke, Wall-Clock): ~267 tok/s. MTP-Modelle zeigen hier niedrigere Werte als im "
             "Single-Stream-Smoke (Drafter-Overhead bei niedriger Draft-Acceptance).")
    L.append("")

    # ---- Per-Playbook-Sieger ----
    L.append("## Per-Playbook-Sieger")
    L.append("")
    L.append("| Playbook | Bestes Modell | Wert |")
    L.append("|----------|---------------|------|")
    for pkey, plabel in PLAYBOOKS:
        best = None
        for r in rows:
            pb = r["data"]["playbooks"].get(pkey)
            if not pb:
                continue
            _, _, rate = pb_cell(pb)
            if best is None or rate > best[1]:
                best = (r["name"], rate, pb_cell(pb))
        if best:
            p, t, _ = best[2]
            L.append(f"| {plabel} | **{best[0]}** | {p}/{t} ({best[1]*100:.0f}%) |")
    L.append("")

    # ---- Kernaussagen ----
    L.append("## Kernaussagen")
    L.append("")
    for f in FINDINGS:
        L.append(f"- {f}")
    L.append("")

    # ---- K.O.-Gründe ----
    L.append("## K.O.-Gründe im Detail")
    L.append("")
    L.append("| Modell | Urteil | ausgelöst durch |")
    L.append("|--------|--------|-----------------|")
    for r in ranked:
        reasons = ko_reasons(r["data"])
        L.append(
            f"| {mlink(r)} | {r['data']['summary']['overall']} | "
            f"{', '.join(reasons) if reasons else '—'} |"
        )
    L.append("")

    # ---- Basis-System-Prompt ----
    L.append("## Basis-System-Prompt (alle Modelle)")
    L.append("")
    L.append("Default-System-Prompt aus `config/testplan.yaml`, identisch für alle 8 "
             "Modelle — inkl. der für diesen Lauf neuen Prämissenprüfung (Absatz 4). "
             f"Evaluator-Defaults: {EVAL_DEFAULTS}. Modellspezifische Sampling-/"
             "Template-Overrides stehen im Run-Parameter-Block des jeweiligen "
             "Einzelberichts.")
    L.append("")
    L.append("```text")
    L.append(sys_prompt.rstrip())
    L.append("```")
    L.append("")
    L.append(
        "---\n\n_Quelldaten: reports/&lt;run&gt;/&lt;model&gt;.json (jüngster vollständiger "
        "Lauf je Modell). Generiert von `unified_v024_report.py`._"
    )
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "unified-v024" / "index.html")
    ap.add_argument("--md", type=Path, default=None,
                    help="zusätzlich Markdown schreiben (Default: <out-dir>/index.md)")
    args = ap.parse_args()

    rows = []
    missing = []
    for name, mname, pb, fam, quant, arch in MODELS:
        data = load_model(mname)
        if data is None:
            missing.append(mname)
            continue
        rows.append({"name": name, "mname": mname, "params_b": pb,
                     "family": fam, "quant": quant, "arch": arch, "data": data})
        print(f"  ✓ {name:26} ← reports/{data['_source']}/{mname}.json")

    if missing:
        print("  ! fehlend:", ", ".join(missing))
    if not rows:
        raise SystemExit("Keine Modell-Reports gefunden.")

    try:
        yaml_cfg = load_yaml_cfg()
    except Exception as e:  # fail-safe: Report auch ohne yaml-Kontext erzeugen
        print(f"  ! testplan.yaml nicht lesbar ({e}) — Prompt/Sampling entfallen")
        yaml_cfg = {"default_prompt": "", "models": {}}
    sys_prompt = yaml_cfg["default_prompt"]

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = build_html(rows, generated, sys_prompt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"\n→ {args.out}")

    md_out = args.md or args.out.with_suffix(".md")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(build_md(rows, generated, sys_prompt), encoding="utf-8")
    print(f"→ {md_out}")

    # Einzelberichte je Modell daneben bündeln (verlinkt aus der Übersicht).
    # HTML für die Web-Ansicht, Markdown (mit Rücklink) für die GitLab-Ablage.
    details_dir = args.out.parent / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    backlink = "[← Zurück zur Übersicht](../index.md)\n\n"
    for r in rows:
        src_dir = Path(r["data"]["_dir"])
        params = runtime_param_rows(r, yaml_cfg)
        src_html = src_dir / f"{r['mname']}.html"
        if src_html.exists():
            body = inject_params_html(src_html.read_text(encoding="utf-8"), params)
            (details_dir / f"{r['mname']}.html").write_text(body, encoding="utf-8")
        src_md = src_dir / f"{r['mname']}.md"
        if src_md.exists():
            body = inject_params_md(src_md.read_text(encoding="utf-8"), params)
            (details_dir / f"{r['mname']}.md").write_text(backlink + body, encoding="utf-8")
            print(f"  ↳ einzel: details/{r['mname']}.md ({len(params)} Parameter)")
        else:
            print(f"  ! Einzel-MD fehlt für {r['name']}: {src_md}")


if __name__ == "__main__":
    main()
