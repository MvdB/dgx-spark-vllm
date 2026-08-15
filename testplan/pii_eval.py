#!/usr/bin/env python3
"""Bewertet einen PII-Filter gegen testdata/pii/*.jsonl.

Eigenes Werkzeug statt eines Playbooks im Orchestrator, und zwar aus einem
handfesten Grund: dieses Modell spricht keine OpenAI-Schnittstelle. Es laeuft
weder unter vLLM noch ueber HTTP, sondern nur ueber die Bibliothek opf im
Prozess. Der Orchestrator ist um Endpunkte herum gebaut — ihn dafuer zu
verbiegen haette mehr kaputt gemacht als gebracht.

Die Ausgabe folgt trotzdem der Form der Guardrail-Berichte
(reports/guardrails/*.json): label, metrics, knockouts, per_case. Damit kann
die Seitenerzeugung sie spaeter genauso einsammeln.

  ./pii_eval.py                                    # deutsches Fine-Tune, CPU
  ./pii_eval.py --model openai--privacy-filter     # Basismodell zum Vergleich
  ./pii_eval.py --device cuda --label de-ft-gpu

Der Ladeweg wird aus der config.json bestimmt (siehe evaluators/pii_backends.py):
opf-Checkpoints ueber die Bibliothek opf, transformers-Checkpoints ueber
AutoModelForTokenClassification. opf liegt nicht auf PyPI:
  pip install "git+https://github.com/openai/privacy-filter"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from evaluators.pii import Span, bewerte_fall, kennzahlen, knockouts  # noqa: E402
from evaluators.pii_backends import waehle_backend  # noqa: E402

logger = logging.getLogger("pii_eval")

HF_MODELS_DIR = pathlib.Path(os.environ.get("HF_MODELS_DIR", pathlib.Path.home() / "hf_models"))
HIER = pathlib.Path(__file__).resolve().parent


def lade_faelle(pfad: pathlib.Path) -> list[dict]:
    faelle = []
    for zeile in pfad.read_text(encoding="utf-8").splitlines():
        zeile = zeile.strip()
        if zeile:
            faelle.append(json.loads(zeile))
    return faelle


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="digitflow--privacy-filter-de-ft",
                   help="Verzeichnisname unter $HF_MODELS_DIR")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="cpu genuegt: 2,8 GB Modell, die Laufzeit dominiert nicht")
    p.add_argument("--daten", default="testdata/pii/de_spans.jsonl")
    p.add_argument("--label", default="", help="Name im Bericht (Vorgabe: Modellverzeichnis)")
    p.add_argument("--out", default="reports/pii", help="Zielverzeichnis fuer den Bericht")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    modell_pfad = HF_MODELS_DIR / args.model
    if not modell_pfad.is_dir():
        logger.error("Modellverzeichnis fehlt: %s", modell_pfad)
        return 2

    daten = pathlib.Path(args.daten)
    if not daten.is_absolute():
        daten = HIER / daten
    if not daten.is_file():
        logger.error("Testdaten fehlen: %s", daten)
        return 2

    faelle = lade_faelle(daten)
    logger.info("%d Faelle aus %s", len(faelle), daten.name)

    t0 = time.monotonic()
    try:
        backend = waehle_backend(modell_pfad, args.device)
    except ImportError as e:
        logger.error("Bibliothek fehlt: %s", e)
        logger.error('  opf:          pip install "git+https://github.com/openai/privacy-filter"')
        logger.error("  transformers: pip install transformers torch")
        return 2
    except ValueError as e:
        logger.error("%s", e)
        return 2
    logger.info("Modell geladen in %.1f s (%s, Ladeweg %s)",
                time.monotonic() - t0, args.device, backend.name)

    ergebnisse = []
    for fall in faelle:
        erwartet = [Span(**{k: s[k] for k in ("label", "start", "end", "text")})
                    for s in fall["expected"]["value"]]
        t = time.monotonic()
        fehler = ""
        erkannt: list[Span] = []
        try:
            erkannt = [Span(label=lab, start=a, end=b, text=txt)
                       for lab, a, b, txt in backend.spans(fall["text"])]
        except Exception as e:  # noqa: BLE001 — ein Fall darf den Lauf nicht beenden
            fehler = f"{type(e).__name__}: {e}"
            logger.warning("%s: %s", fall["id"], fehler)
        ergebnisse.append(bewerte_fall(fall["id"], erwartet, erkannt,
                                       (time.monotonic() - t) * 1000, fehler))

    metriken = kennzahlen(ergebnisse)

    # Kein Bericht aus lauter Fehlern. Beim ersten Lauf gegen das Basismodell
    # scheiterte JEDER Fall am Laden ("Checkpoint config field encoding must be
    # a non-empty string") — und heraus kam ein Bericht mit Recall 0.000 und
    # einem K.O., der aussah wie ein Messergebnis. Ein Modell, das nicht laedt,
    # ist nicht schlecht, es ist ungemessen. Der Unterschied muss sichtbar
    # bleiben, sonst landet eine Null in einer Tabelle und wird gelesen wie eine.
    anteil_fehler = metriken["n_fehler"] / max(1, metriken["n_faelle"])
    if anteil_fehler > 0.2:
        logger.error("%d von %d Faellen fehlgeschlagen (%.0f%%) — kein Bericht.",
                     metriken["n_fehler"], metriken["n_faelle"], anteil_fehler * 100)
        erster = next((e.fehler for e in ergebnisse if e.fehler), "")
        logger.error("Erster Fehler: %s", erster)
        logger.error("Das ist kein Messergebnis, sondern ein Ladeproblem.")
        return 3

    ko = knockouts(metriken)
    label = args.label or args.model

    # Rohdaten zuerst, Zusammenfassung danach — dieselbe Regel wie in den
    # anderen Harnessen der Familie: ein Lauf, der beim Zusammenfassen
    # scheitert, soll die Einzelergebnisse trotzdem hinterlassen haben.
    ziel = HIER / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    ziel.mkdir(parents=True, exist_ok=True)
    datei = ziel / f"{label.replace('/', '--')}.json"
    datei.write_text(json.dumps({
        "label": label,
        "model_dir": str(modell_pfad),
        "device": args.device,
        "backend": backend.name,
        "testdata": daten.name,
        "date": date.today().isoformat(),
        "metrics": metriken,
        "knockouts": ko,
        "per_case": [{
            "id": e.kennung,
            "erwartet": [vars(s) for s in e.erwartet],
            "erkannt": [vars(s) for s in e.erkannt],
            "streng": e.streng.als_dict(),
            "tolerant": e.tolerant.als_dict(),
            "dauer_ms": round(e.dauer_ms, 2),
            "fehler": e.fehler,
        } for e in ergebnisse],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    st, to = metriken["streng"], metriken["tolerant"]
    print()
    print(f"  {label} ({args.device}, {backend.name})")
    print(f"  {'':22s} {'P':>7} {'R':>7} {'F1':>7}   TP/FP/FN")
    print(f"  {'streng (Grenzen exakt)':22s} {st['precision']:7.3f} {st['recall']:7.3f} "
          f"{st['f1']:7.3f}   {st['tp']}/{st['fp']}/{st['fn']}")
    print(f"  {'tolerant (Ueberlappung)':22s} {to['precision']:7.3f} {to['recall']:7.3f} "
          f"{to['f1']:7.3f}   {to['tp']}/{to['fp']}/{to['fn']}")
    print()
    print(f"  Negativfaelle: {metriken['negativfaelle']['mit_fehlfund']}"
          f"/{metriken['negativfaelle']['n']} mit Fehlfund "
          f"({metriken['negativfaelle']['fehlfund_rate']:.1%})")
    print(f"  Latenz Median: {metriken['latenz_ms_median']} ms")
    print()
    print("  Je Klasse (tolerant):")
    for klasse, z in metriken["je_klasse"].items():
        print(f"    {klasse:18s} F1 {z['f1']:.3f}   {z['tp']}/{z['fp']}/{z['fn']}")
    if ko:
        print()
        for k in ko:
            print(f"  K.O.  {k}")
    print()
    print(f"  Bericht: {datei}")
    return 2 if ko else 0


if __name__ == "__main__":
    raise SystemExit(main())
