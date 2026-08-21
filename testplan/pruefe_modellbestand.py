#!/usr/bin/env python3
"""Meldet gesyncte Modellverzeichnisse, die in keinem Testplaneintrag stehen.

Der Sync zieht, was in der LocalCache-Collection steht. Die Modellliste in
config/testplan.yaml ist handgepflegt. Zwischen beidem gibt es keine
Verbindung — poolside/Laguna-S-2.1-NVFP4 lag deshalb ab dem 12.08.2026 neun
Tage unbemerkt da, bis es jemandem auffiel.

Diese Pruefung schliesst die Luecke. Sie meldet nur, was ein Sprachmodell fuer
den Testplan sein koennte; Bild, Sprache, Einbettungen und Drafter erkennt sie
an der Architektur und laesst sie weg, denn die gehoeren nach southbyte-image
und southbyte-tts oder sind Beiwerk.

Aufruf:
    pruefe_modellbestand.py              # Bericht, Exitcode 1 bei Fundstellen
    pruefe_modellbestand.py --alle       # auch zeigen, was warum aussortiert wurde
    pruefe_modellbestand.py --leise      # nur Exitcode, fuer Skripte

Bewusste Ausnahmen gehoeren nach config/testplan.yaml unter `inventar:` —
mit Begruendung, damit spaeter nachvollziehbar ist, warum etwas nicht laeuft.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# Architektur- und Namensmerkmale, an denen ein Verzeichnis als "gehoert nicht
# in den LLM-Testplan" erkannt wird. Klein geschrieben, geprueft wird auf
# Teilstring in architectures + model_type + Verzeichnisname.
NICHT_LLM = {
    "Sprache/Audio": ("asr", "tts", "speech", "whisper", "voxtral", "vibevoice",
                      "music", "audex", "higgs", "voicechat", "dramabox",
                      "magpie", "voxcpm", "chatterbox"),
    "Bild/OCR":      ("ocr", "docling", "image", "flux", "ideogram", "krea",
                      "mage-flow", "hidream"),
    "Einbettung":    ("embed", "modernbert", "tokenclassification", "privacy-filter",
                      "privacy_filter"),
    "Drafter":       ("assistant", "dspark"),
    "Tokenizer":     ("tokenizer",),
}


def einstufen(verzeichnis: Path) -> tuple[str, str]:
    """(Klasse, Begruendung). Klasse 'LLM' heisst: gehoert in den Testplan."""
    name = verzeichnis.name.lower()
    conf = verzeichnis / "config.json"

    if (verzeichnis / "model_index.json").exists():
        return "Bild/OCR", "model_index.json — Diffusers-Pipeline"
    if not conf.exists():
        return "unklar", "keine config.json — Aufbau nicht lesbar"

    try:
        j = json.loads(conf.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return "unklar", f"config.json nicht lesbar ({e.__class__.__name__})"

    arch = " ".join(j.get("architectures") or [])
    heu = f"{arch} {j.get('model_type', '')} {name}".lower()

    for klasse, muster in NICHT_LLM.items():
        for m in muster:
            if m in heu:
                return klasse, f"'{m}' in {arch or j.get('model_type', '?')}"

    if not arch:
        return "unklar", f"keine architectures, model_type={j.get('model_type', '?')}"
    return "LLM", arch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alle", action="store_true", help="auch Aussortiertes zeigen")
    ap.add_argument("--leise", action="store_true", help="nur Exitcode")
    ap.add_argument("--modelle", default=os.environ.get("HF_MODELS_DIR", ""),
                    help="Modellspeicher (Standard: $HF_MODELS_DIR oder ~/hf_models)")
    ap.add_argument("--config", default=None, help="Pfad zur testplan.yaml")
    a = ap.parse_args()

    speicher = Path(a.modelle).expanduser() if a.modelle else Path.home() / "hf_models"
    conf_pfad = Path(a.config) if a.config else Path(__file__).resolve().parent / "config" / "testplan.yaml"

    if not speicher.is_dir():
        print(f"Modellspeicher nicht gefunden: {speicher}", file=sys.stderr)
        return 2

    plan = yaml.safe_load(conf_pfad.read_text(encoding="utf-8"))
    eingetragen = {m.get("profile") for m in plan.get("models", []) if m.get("profile")}
    ausnahmen = {k: v for k, v in (plan.get("inventar") or {}).get("ignorieren", {}).items()}

    # Gegenrichtung: Eintraege, deren Verzeichnis fehlt. Wer ein altes Modell
    # von der Platte nimmt, um Platz zu schaffen, laesst sonst einen aktiven
    # Eintrag zurueck — und der naechste Sammellauf scheitert daran.
    verwaist = []
    for m in plan.get("models", []):
        prof = m.get("profile")
        if not prof or not m.get("active"):
            continue
        # SaaS-Zeilen haben kein Verzeichnis: ihr "Profil" ist eine
        # API-Kennung (xai/grok-4.6), und sie laufen auf machine: saas.
        # Erkannt an beidem, damit eine der zwei Angaben reichen wuerde.
        if m.get("machine") == "saas" or "saas" in (m.get("tags") or []):
            continue
        if not (speicher / prof).is_dir():
            verwaist.append((m.get("name", "?"), prof))

    luecken, aussortiert, bekannt = [], [], 0
    for eintrag in sorted(os.listdir(speicher)):
        pfad = speicher / eintrag
        # Nur Modellverzeichnisse: der Sync legt sie als <org>--<name> an.
        if not pfad.is_dir() or "--" not in eintrag:
            continue
        if eintrag in eingetragen:
            bekannt += 1
            continue
        if eintrag in ausnahmen:
            aussortiert.append((eintrag, "Ausnahme", ausnahmen[eintrag]))
            continue
        klasse, grund = einstufen(pfad)
        if klasse == "LLM":
            luecken.append((eintrag, grund))
        else:
            aussortiert.append((eintrag, klasse, grund))

    if a.leise:
        return 1 if (luecken or verwaist) else 0

    print(f"Modellspeicher : {speicher}")
    print(f"Im Testplan    : {bekannt}")
    print(f"Aussortiert    : {len(aussortiert)}  (Bild, Sprache, Einbettungen, Drafter, Ausnahmen)")
    print(f"OHNE EINTRAG   : {len(luecken)}")
    print(f"AKTIV OHNE MODELL: {len(verwaist)}")

    if verwaist:
        print("\nDiese Eintraege stehen auf active: true, aber ihr Verzeichnis fehlt.")
        print("Der naechste Sammellauf scheitert daran:\n")
        for name, prof in verwaist:
            print(f"  {name:36} -> {prof}")
        print("\nAuf active: false setzen. Die Ergebnisse bleiben davon unberuehrt —")
        print("sie liegen in reports/ und in docs/, nicht im Modellspeicher.")

    if luecken:
        print("\nDiese Verzeichnisse sehen nach einem Sprachmodell aus und stehen in")
        print("keinem Testplaneintrag:\n")
        for name, arch in luecken:
            groesse = sum(f.stat().st_size for f in (speicher / name).rglob("*") if f.is_file())
            print(f"  {name:52} {groesse / 1e9:6.1f} GB  {arch}")
        print("\nEntweder einen Eintrag in config/testplan.yaml anlegen — oder unter")
        print("`inventar.ignorieren` eintragen, mit Begruendung.")

    if a.alle and aussortiert:
        print("\nAussortiert:\n")
        for name, klasse, grund in aussortiert:
            print(f"  {name:52} {klasse:14} {grund}")

    return 1 if (luecken or verwaist) else 0


if __name__ == "__main__":
    sys.exit(main())
