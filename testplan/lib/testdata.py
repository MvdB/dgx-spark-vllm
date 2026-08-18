"""Testdaten-Management: Laden, Filtern und Validieren von JSONL-Testfällen.

JSONL-Schema pro Zeile:
{
    "id": "qual-001",
    "category": "quality|bias|security|code|performance|guardrails",
    "subcategory": "hallucination|factual|coherence|...",
    "language": "de|en|fr|...",
    "difficulty": "easy|medium|hard",
    "prompt": "Die eigentliche Frage / Anweisung",
    "system_prompt": "(optional) System-Prompt Override",
    "context": "(optional) Kontext-Dokument für RAG-Szenarien",
    "expected": {
        "type": "exact|contains|regex|semantic|code_exec|judge",
        "value": "Erwarteter Wert oder Beschreibung",
        "alternatives": ["Akzeptable Alternativen"],
        "forbidden": ["Darf NICHT enthalten sein"]
    },
    "metadata": {
        "source": "Herkunft des Testfalls",
        "reviewer": "Name des Zweitprüfers",
        "created": "2026-04-06",
        "tags": ["rag", "medical", "legal", ...]
    }
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("testplan.testdata")


@dataclass
class Expected:
    type: str  # exact, contains, regex, semantic, code_exec, judge
    value: str
    alternatives: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)


@dataclass
class TestCase:
    id: str
    category: str
    subcategory: str
    language: str
    difficulty: str
    prompt: str
    expected: Expected | None
    system_prompt: str = ""
    context: str = ""
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestCase:
        exp_data = data.get("expected")
        expected = (
            Expected(
                type=exp_data["type"],
                value=exp_data.get("value", ""),
                alternatives=exp_data.get("alternatives", []),
                forbidden=exp_data.get("forbidden", []),
            )
            if exp_data is not None
            else None
        )
        return cls(
            id=data["id"],
            category=data["category"],
            subcategory=data.get("subcategory", ""),
            language=data.get("language", "de"),
            difficulty=data.get("difficulty", "medium"),
            prompt=data["prompt"],
            system_prompt=data.get("system_prompt", ""),
            context=data.get("context", ""),
            max_tokens=data.get("max_tokens"),
            expected=expected,
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "subcategory": self.subcategory,
            "language": self.language,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "context": self.context,
            "metadata": self.metadata,
        }
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        if self.expected is not None:
            d["expected"] = {
                "type": self.expected.type,
                "value": self.expected.value,
                "alternatives": self.expected.alternatives,
                "forbidden": self.expected.forbidden,
            }
        return d


def pruefe_schema(base_dir: Path, max_meldungen: int = 20) -> list[str]:
    """Alle JSONL-Faelle gegen testdata/schema.json pruefen.

    Laeuft ueber JEDE Datei unter base_dir/*/*.jsonl, nicht nur ueber die
    Kategorien, die der Orchestrator laedt — sonst bliebe pii/ ungeprueft, das
    von pii_eval.py gelesen wird. Das Schema ist am 17.08.2026 an den Bestand
    angeglichen worden; ohne diesen Aufruf faellt eine erneute Abweichung erst
    auf, wenn ein Auswerter ueber ein fehlendes Feld stolpert.
    """
    schema_datei = base_dir / "schema.json"
    if not schema_datei.exists():
        return [f"{schema_datei} fehlt — Testdaten ungeprueft"]
    try:
        from jsonschema import Draft7Validator
    except ImportError:
        return ["jsonschema nicht installiert — Schemapruefung uebersprungen "
                "(.venv/bin/python -m pip install jsonschema)"]

    schema = json.loads(schema_datei.read_text(encoding="utf-8"))
    Draft7Validator.check_schema(schema)          # Fehler IM Schema, nicht in den Daten
    validator = Draft7Validator(schema)

    fehler: list[str] = []
    gezaehlt = 0
    for datei in sorted(base_dir.glob("*/*.jsonl")):
        for zeilennr, zeile in enumerate(datei.read_text(encoding="utf-8").splitlines(), 1):
            zeile = zeile.strip()
            if not zeile or zeile.startswith("#"):
                continue
            try:
                fall = json.loads(zeile)
            except json.JSONDecodeError as e:
                gezaehlt += 1
                if len(fehler) < max_meldungen:
                    fehler.append(f"{datei.parent.name}/{datei.name}:{zeilennr}: kein JSON ({e})")
                continue
            for v in sorted(validator.iter_errors(fall), key=lambda e: list(e.path)):
                gezaehlt += 1
                if len(fehler) < max_meldungen:
                    pfad = ".".join(str(x) for x in v.path) or "(Wurzel)"
                    fehler.append(
                        f"{datei.parent.name}/{datei.name}:{zeilennr} "
                        f"{fall.get('id', '?')}: {pfad}: {v.message}")
    if gezaehlt > len(fehler):
        fehler.append(f"... und {gezaehlt - len(fehler)} weitere Schemaverstoesse")
    return fehler


class TestDataLoader:
    """Lädt und filtert Testfälle aus JSONL-Dateien."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def load_category(self, category: str) -> list[TestCase]:
        """Lade alle Testfälle einer Kategorie."""
        category_dir = self.base_dir / category
        cases: list[TestCase] = []

        if not category_dir.exists():
            logger.warning("Testdaten-Verzeichnis nicht gefunden: %s", category_dir)
            return cases

        for jsonl_file in sorted(category_dir.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        data = json.loads(line)
                        cases.append(TestCase.from_dict(data))
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(
                            "Fehler in %s, Zeile %d: %s", jsonl_file.name, line_num, e
                        )

        logger.info("Kategorie '%s': %d Testfälle geladen", category, len(cases))
        return cases

    def load_all(self) -> dict[str, list[TestCase]]:
        """Lade alle Kategorien."""
        categories = [
            "quality", "bias", "security", "code",
            "german_language", "long_context", "performance", "guardrails",
        ]
        return {cat: self.load_category(cat) for cat in categories if (self.base_dir / cat).exists()}

    def filter_cases(
        self,
        cases: list[TestCase],
        *,
        language: str | None = None,
        subcategory: str | None = None,
        difficulty: str | None = None,
        tags: list[str] | None = None,
    ) -> list[TestCase]:
        """Filtere Testfälle nach Kriterien."""
        filtered = cases
        if language:
            filtered = [c for c in filtered if c.language == language]
        if subcategory:
            filtered = [c for c in filtered if c.subcategory == subcategory]
        if difficulty:
            filtered = [c for c in filtered if c.difficulty == difficulty]
        if tags:
            filtered = [
                c for c in filtered
                if any(t in c.metadata.get("tags", []) for t in tags)
            ]
        return filtered

    def validate(self, schema: bool = True) -> list[str]:
        """Validiere alle Testdaten und gib Fehler zurück.

        schema=False, wenn der Aufrufer pruefe_schema() schon selbst gefahren
        hat — der Orchestrator tut das, weil ein Schemaverstoss den Lauf
        abbricht, die weichen Befunde hier aber nur eine Warnung sind.
        """
        errors: list[str] = list(pruefe_schema(self.base_dir)) if schema else []
        all_ids: set[str] = set()
        all_cases = self.load_all()

        for category, cases in all_cases.items():
            for case in cases:
                # Duplikat-Check
                if case.id in all_ids:
                    errors.append(f"Doppelte ID: {case.id}")
                all_ids.add(case.id)

                # Pflichtfelder
                if not case.prompt.strip():
                    errors.append(f"{case.id}: Leerer Prompt")
                if case.expected is not None:
                    if not case.expected.value and case.expected.type != "judge":
                        errors.append(f"{case.id}: Kein expected.value (außer bei type=judge)")

        # Sprachverteilung prüfen
        total = sum(len(c) for c in all_cases.values())
        if total > 0:
            lang_counts: dict[str, int] = {}
            for cases in all_cases.values():
                for case in cases:
                    lang_counts[case.language] = lang_counts.get(case.language, 0) + 1

            de_ratio = lang_counts.get("de", 0) / total
            if de_ratio < 0.65:
                errors.append(
                    f"Deutsch-Anteil nur {de_ratio:.0%} (Ziel: ≥70%)"
                )

        if errors:
            logger.warning("%d Validierungsfehler gefunden", len(errors))
        else:
            logger.info("✓ Alle Testdaten valide (%d Fälle)", total)

        return errors
