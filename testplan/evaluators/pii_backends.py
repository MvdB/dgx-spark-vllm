"""Zwei Ladewege fuer PII-Filter, weil die Modelle unterschiedlich verpackt sind.

Die beiden Modelle derselben Familie kommen in verschiedenen Formaten:

  openai/privacy-filter            transformers-Checkpoint mit tokenizer.json,
                                   architectures und id2label. Laedt mit
                                   AutoModelForTokenClassification.
  digitflow/privacy-filter-de-ft   opf-Checkpoint. Keine architectures, kein
                                   Tokenizer im Verzeichnis, stattdessen
                                   "encoding": "o200k_base" und span_class_names.

Der Versuch, beide mit opf zu laden, scheitert am Basismodell:
    ValueError: Checkpoint config field encoding must be a non-empty string
Umgekehrt fehlt dem deutschen Fine-Tune alles, was transformers braucht.

Statt eines der Modelle wegzulassen gibt es hier beide Wege. Welcher gilt,
entscheidet die config.json — geraten wird nichts.
"""

from __future__ import annotations

import json
import pathlib


class Backend:
    """Gemeinsame Schnittstelle: Text rein, Spannen raus."""

    name = "?"

    def spans(self, text: str) -> list[tuple[str, int, int, str]]:
        raise NotImplementedError


class OpfBackend(Backend):
    name = "opf"

    def __init__(self, pfad: str, device: str = "cpu"):
        import opf

        self._m = opf.OPF(model=pfad, device=device,
                          output_mode="typed", decode_mode="viterbi")

    def spans(self, text: str) -> list[tuple[str, int, int, str]]:
        r = self._m.redact(text)
        return [(s.label, s.start, s.end, s.text) for s in r.detected_spans]


class TransformersBackend(Backend):
    """Token-Klassifikation mit BIOES-Dekodierung ueber die Offset-Abbildung.

    Die Offsets kommen vom Tokenizer, nicht aus einer eigenen Rechnung. Ein
    selbst gebautes Mapping von Token zu Zeichen geht bei Umlauten und
    Mehrbyte-Zeichen schief, und zwar still — die Spanne verschiebt sich um ein
    paar Zeichen und sieht trotzdem plausibel aus.
    """

    name = "transformers"

    def __init__(self, pfad: str, device: str = "cpu"):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(pfad)
        self._m = AutoModelForTokenClassification.from_pretrained(pfad).to(device).eval()
        self._device = device
        self._id2label = self._m.config.id2label

    def spans(self, text: str) -> list[tuple[str, int, int, str]]:
        enc = self._tok(text, return_tensors="pt", return_offsets_mapping=True,
                        truncation=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with self._torch.no_grad():
            logits = self._m(**enc).logits
        tags = [self._id2label[i] for i in logits.argmax(-1)[0].tolist()]
        return _bioes_zu_spans(tags, offsets, text)


def _bioes_zu_spans(tags: list[str], offsets: list[list[int]],
                    text: str) -> list[tuple[str, int, int, str]]:
    """BIOES-Folge in Zeichenspannen uebersetzen.

    B beginnt, I setzt fort, E schliesst ab, S steht allein. Unsaubere Folgen
    kommen vor — ein I ohne vorangehendes B etwa —, und sie werden hier
    aufgefangen statt verworfen: was das Modell markiert hat, soll auch in die
    Auswertung, sonst misst man die Dekodierung statt des Modells.
    """
    spans: list[tuple[str, int, int, str]] = []
    offen_label: str | None = None
    offen_start = 0
    offen_ende = 0

    def schliessen() -> None:
        """Spanne abschliessen und dabei Leerraum an den Raendern abschneiden.

        Der Tokenizer schlaegt das fuehrende Leerzeichen dem Wort zu — aus
        "ist Juergen Mueller" wird eine Spanne, die bei dem Leerzeichen vor
        "Juergen" beginnt. Beim ersten Lauf hatten 29 von 43 Spannen des
        Basismodells so einen Rand, und die strenge Wertung fiel dadurch von
        brauchbar auf F1 0.198 — gemessen wurde die Zeichenkonvention des
        Tokenizers, nicht das Modell. Eine Spanne soll die Sache bezeichnen,
        nicht das Leerzeichen davor.
        """
        nonlocal offen_label
        if offen_label is None:
            return
        a, b = offen_start, offen_ende
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            spans.append((offen_label, a, b, text[a:b]))
        offen_label = None

    for tag, (a, b) in zip(tags, offsets, strict=False):
        if a == b == 0:          # Sondertoken haben eine leere Spanne
            continue
        if tag == "O":
            schliessen()
            continue
        praefix, _, label = tag.partition("-")
        if not label:
            schliessen()
            continue
        if praefix == "S":
            schliessen()
            offen_label, offen_start, offen_ende = label, a, b
            schliessen()
        elif praefix == "B":
            schliessen()
            offen_label, offen_start, offen_ende = label, a, b
        elif praefix in ("I", "E"):
            if offen_label == label:
                offen_ende = b
            else:
                # Fortsetzung ohne Anfang: als eigene Spanne behandeln, statt
                # sie fallen zu lassen.
                schliessen()
                offen_label, offen_start, offen_ende = label, a, b
            if praefix == "E":
                schliessen()
    schliessen()
    return spans


def waehle_backend(pfad: pathlib.Path, device: str = "cpu") -> Backend:
    """Entscheidet anhand der config.json, nicht anhand des Verzeichnisnamens."""
    konfig = json.loads((pfad / "config.json").read_text(encoding="utf-8"))
    if konfig.get("encoding"):
        return OpfBackend(str(pfad), device)
    if konfig.get("architectures"):
        return TransformersBackend(str(pfad), device)
    raise ValueError(
        f"{pfad.name}: weder 'encoding' (opf) noch 'architectures' (transformers) "
        "in der config.json — Ladeweg nicht bestimmbar."
    )
