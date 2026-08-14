#!/usr/bin/env python3
"""Einzelne Testfaelle eines Reports gezielt nachbewerten.

Gedacht fuer Faelle, die nicht am Modell gescheitert sind, sondern am Harness
oder am Judge — etwa ein Judge-Timeout oder ein Neustart des LiteLLM-Proxys.
Wiederholt werden ausschliesslich die betroffenen Faelle, nicht das Playbook:
ein voller 01_quality-Durchlauf kostet Stunden, ein Einzelfall Minuten.

Betroffen ist ein Fall, wenn er
  * verdict == "error" hat, oder
  * eine Begruendung traegt, die mit "Parse-Fehler" beginnt (Altbestand vor
    2026-08-14 — damals lieferte der Parser bei nicht parsebarer Judge-Antwort
    still einen Ersatzwert von 0.6 statt eines Fehlers).

WICHTIG: Das Modell des Reports muss unter --endpoint laufen. Das Skript
prueft das und bricht ab, wenn dort ein anderes Modell antwortet — sonst
wuerden Zahlen eines fremden Modells in den Report geschrieben.

Beispiel:
    python repair_cases.py reports/<lauf>/<Modell>.json --endpoint http://localhost:8000
    python repair_cases.py reports/<lauf>/<Modell>.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI

from evaluators.base import EvalResult, PlaybookResult, Verdict
from evaluators.bias import BiasEvaluator
from evaluators.code_eval import CodeEvaluator
from evaluators.quality import QualityEvaluator
from evaluators.security import SecurityEvaluator
from lib.config import ModelConfig, TestplanConfig
from lib.testdata import TestDataLoader
from reporter import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("repair")

# Playbook → Evaluator. 06_performance fehlt bewusst: es nutzt keinen Judge und
# misst Durchsatz unter Last, das laesst sich nicht sinnvoll je Fall wiederholen.
EVALUATOREN = {
    "01_quality": QualityEvaluator,
    "02_german_language": QualityEvaluator,
    "03_bias": BiasEvaluator,
    "04_security": SecurityEvaluator,
    "05_code": CodeEvaluator,
}

ALLE_KATEGORIEN = [
    "quality", "long_context", "german_language", "bias", "security", "code",
]


# Fehler, die eine Wiederholung NICHT behebt: der Anbieter blockt den Prompt.
# Das ist kein Ausfall, sondern ein Befund ueber das Modell im Einsatz und
# gehoert unveraendert im Report stehen.
_ENDGUELTIG = ("content-filter", "content_filter")


def betroffen(r: dict) -> bool:
    grund = str(r.get("reasoning", ""))
    if any(m in grund.lower() for m in _ENDGUELTIG):
        return False
    return r.get("verdict") == "error" or grund.startswith("Parse-Fehler")


def aggregate_neu(pb: dict) -> None:
    """Playbook-Kennzahlen aus den Einzelergebnissen neu rechnen."""
    res = pb["results"]
    pb["passed"] = sum(1 for r in res if r["verdict"] == "pass")
    pb["failed"] = sum(1 for r in res if r["verdict"] in ("fail", "knockout"))
    pb["total"] = len(res)
    pb["pass_rate"] = pb["passed"] / pb["total"] if pb["total"] else 0.0
    scores = [r["score"] for r in res if r["verdict"] != "error"]
    pb["mean_score"] = sum(scores) / len(scores) if scores else 0.0
    pb["knockouts"] = [r for r in res if r["verdict"] == "knockout"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--config", default="config/testplan.yaml")
    ap.add_argument("--dry-run", action="store_true", help="nur auflisten, nichts aendern")
    args = ap.parse_args()

    d = json.loads(args.report.read_text(encoding="utf-8"))
    modellname = d.get("meta", {}).get("model") or args.report.stem

    offen = {pb: [r["test_id"] for r in p["results"] if betroffen(r)]
             for pb, p in d["playbooks"].items()}
    offen = {pb: ids for pb, ids in offen.items() if ids}
    gesamt = sum(len(v) for v in offen.values())
    if not gesamt:
        logger.info("%s: nichts zu reparieren", modellname)
        return 0
    for pb, ids in offen.items():
        logger.info("%s / %s: %d Fall/Faelle -> %s", modellname, pb, len(ids), ", ".join(ids))
    if args.dry_run:
        logger.info("Dry-Run: %d Fall/Faelle waeren zu wiederholen", gesamt)
        return 0

    cfg = TestplanConfig.load(args.config)
    ist_saas = d.get("meta", {}).get("source") == "saas_proxy"

    if ist_saas:
        # SaaS-Modelle stehen nicht in testplan.yaml; der Report fuehrt sie ueber
        # meta.profile (die Modell-ID im LiteLLM-Proxy). Ziel ist derselbe Proxy
        # wie der Judge — keine lokale GPU noetig, deshalb unabhaengig davon,
        # welches Modell gerade auf der Maschine laeuft.
        served = d["meta"]["profile"]
        target = OpenAI(base_url=cfg.judge.api_url, api_key=cfg.judge.api_key or "not-needed")
        verfuegbar = {m.id for m in target.models.list().data}
        if served not in verfuegbar:
            logger.error("Proxy kennt '%s' nicht (%d Modelle verfuegbar). Abbruch.",
                         served, len(verfuegbar))
            return 2
        vorlage = next((m for m in cfg.models if m.system_prompt), None)
        model = ModelConfig(name=modellname, profile=served, machine="saas",
                            system_prompt=vorlage.system_prompt if vorlage else "")
        logger.info("SaaS-Modus: %s ueber %s", served, cfg.judge.base_url)
    else:
        modelle = [m for m in cfg.models if m.name == modellname]
        if not modelle:
            logger.error("Modell '%s' steht nicht in %s", modellname, args.config)
            return 2
        model = modelle[0]
        target = OpenAI(base_url=f"{args.endpoint}/v1", api_key="not-needed")
        served = target.models.list().data[0].id
        # Sicherung: laeuft dort wirklich das Modell des Reports?
        if model.profile.split("--")[-1].lower() not in served.lower():
            logger.error("Endpoint bedient '%s', der Report gehoert zu '%s' (Profil %s). Abbruch.",
                         served, modellname, model.profile)
            return 2
        logger.info("Endpoint bestaetigt: %s", served)

    judge = OpenAI(base_url=cfg.judge.api_url, api_key=cfg.judge.api_key or "not-needed")
    loader = TestDataLoader(cfg.testdata_dir)
    faelle = {}
    for kat in ALLE_KATEGORIEN:
        try:
            for tc in loader.load_category(kat):
                faelle.setdefault(tc.id, tc)
        except Exception as e:      # Kategorie fehlt → ueberspringen
            logger.debug("Kategorie %s nicht ladbar: %s", kat, e)

    repariert = fehlgeschlagen = 0
    for pb, ids in offen.items():
        klasse = EVALUATOREN.get(pb)
        if klasse is None:
            logger.warning("%s: kein Evaluator hinterlegt, uebersprungen", pb)
            continue
        ev = klasse(
            target_client=target, target_model=served,
            judge_client=judge, judge_model=cfg.judge.model,
            default_system_prompt=model.system_prompt,
            sampling=model.sampling, chat_template_kwargs=model.chat_template_kwargs,
            extra_body=model.extra_body, omit_sampling=model.omit_sampling,
        )
        for tid in ids:
            tc = faelle.get(tid)
            if tc is None:
                logger.warning("%s: Testfall nicht in den Testdaten gefunden", tid)
                continue
            try:
                neu = ev.evaluate(tc)
            except Exception as e:
                logger.error("%s: Wiederholung fehlgeschlagen: %s", tid, e)
                fehlgeschlagen += 1
                continue
            eintrag = neu.to_dict()
            eintrag.setdefault("metadata", {})
            eintrag["metadata"]["nachbewertet"] = (
                "Einzeln wiederholt mit repair_cases.py — der urspruengliche Fall "
                "scheiterte am Harness oder am Judge, nicht am Modell."
            )
            res = d["playbooks"][pb]["results"]
            idx = [i for i, r in enumerate(res) if r["test_id"] == tid]
            if len(idx) != 1:
                logger.error("%s: %d Treffer im Report, uebersprungen", tid, len(idx))
                fehlgeschlagen += 1
                continue
            alt = res[idx[0]]
            res[idx[0]] = eintrag
            logger.info("%s: %s (%.2f) -> %s (%.2f)", tid, alt["verdict"], alt["score"],
                        eintrag["verdict"], eintrag["score"])
            repariert += 1
        aggregate_neu(d["playbooks"][pb])

    # Aggregate ALLER Playbooks neu rechnen, nicht nur der reparierten. Die
    # gespeicherten Zaehler weichen in Altbestaenden von den Einzelergebnissen
    # ab: bei Nemotron-3-Super standen 67 in playbooks[*].passed, waehrend die
    # results 70 pass-Verdicts enthielten. Wer nur die angefassten Playbooks
    # nachrechnet, laesst die Datei in sich widerspruechlich zurueck.
    for pb in d["playbooks"]:
        aggregate_neu(d["playbooks"][pb])

    # Summary NICHT selbst rechnen: der Reporter zaehlt zusaetzlich einen
    # synthetischen K.O., wenn 01_quality unter thresholds.min_quality_pass_rate
    # liegt. Eine eigene Formel hier hat genau diesen Fall verloren und die
    # K.O.-Zahl des Llama-3.3-70B am 2026-08-14 von 16 auf 15 verfaelscht.
    # Deshalb dieselbe Quelle benutzen, die auch die Reports schreibt.
    try:
        pb_results = [
            PlaybookResult(playbook=name, model=modellname, duration_seconds=p.get("duration_seconds", 0.0),
                           results=[EvalResult(
                               test_id=r["test_id"], model=r["model"], evaluator=r["evaluator"],
                               verdict=Verdict(r["verdict"]), score=r["score"],
                               response=r.get("response", ""), reasoning=r.get("reasoning", ""),
                               latency_ms=r.get("latency_ms", 0.0),
                               tokens_generated=r.get("tokens_generated", 0),
                               thinking=r.get("thinking", ""),
                               response_type=r.get("response_type", "answer"),
                               metadata=r.get("metadata") or {},
                           ) for r in p["results"]])
            for name, p in d["playbooks"].items()
        ]
        gen = ReportGenerator(cfg, args.report.parent.name)
        d["summary"] = gen._model_summary(pb_results)
        args.report.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        safe = modellname.replace("/", "_").replace(" ", "_")
        gen._write_html(safe, model, pb_results)
        gen._write_markdown(safe, model, pb_results)
        logger.info("%s: %d repariert, %d fehlgeschlagen | summary %s",
                    modellname, repariert, fehlgeschlagen, d["summary"])
        logger.info("HTML/MD neu erzeugt")
    except Exception as e:
        logger.error("Summary/HTML/MD fehlgeschlagen — JSON NICHT geschrieben: %s", e)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
