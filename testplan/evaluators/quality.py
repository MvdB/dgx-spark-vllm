"""Qualitäts-Evaluator: Halluzination, Faktentreue, Kohärenz, Instruktionsbefolgung.

Nutzt LLM-as-Judge (Mistral-Small-24B auf Spark A) für semantische Bewertungen.
Strukturierte Judge-Prompts mit erzwungenem JSON-Output für Parsing-Sicherheit.
"""

from __future__ import annotations

import json
import logging
import re

from lib.testdata import TestCase

from .base import BaseEvaluator, EvalResult, Verdict, parse_json_response

logger = logging.getLogger("testplan.evaluators.quality")

# ---------------------------------------------------------------------------
# Judge-Prompts (Deutsch, da 80%+ der Testfälle deutsch sind)
# ---------------------------------------------------------------------------

HALLUCINATION_JUDGE_SYSTEM = """\
Du bist ein strenger Qualitätsprüfer für KI-generierte Antworten.
Deine Aufgabe: Prüfe ob die Antwort Halluzinationen enthält — also Fakten, \
Behauptungen oder Details, die NICHT aus dem gegebenen Kontext oder der Frage ableitbar \
sind und die sachlich falsch oder erfunden wirken.

Bewerte auf einer Skala von 1-5:
5 = Keine Halluzinationen, alles belegbar oder korrekt eingeräumt wenn unsicher
4 = Minimale Ungenauigkeiten, keine erfundenen Fakten
3 = Einzelne fragwürdige Behauptungen, aber Kernaussage korrekt
2 = Mehrere erfundene oder falsche Fakten
1 = Schwerwiegende Halluzinationen, komplett erfundene Informationen

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:
{"score": <1-5>, "hallucinations": ["<Liste konkreter Halluzinationen>"], \
"reasoning": "<Begründung in 2-3 Sätzen>"}"""

FACTUAL_JUDGE_SYSTEM = """\
Du bist ein Faktenchecker. Vergleiche die KI-Antwort mit der erwarteten \
Referenzantwort und bewerte die faktische Korrektheit.

Bewerte auf einer Skala von 1-5:
5 = Faktisch vollständig korrekt, alle Kernfakten enthalten
4 = Überwiegend korrekt, kleinere Auslassungen
3 = Teilweise korrekt, wichtige Fakten fehlen oder sind ungenau
2 = Überwiegend falsch oder unvollständig
1 = Komplett falsch oder am Thema vorbei

Antworte AUSSCHLIESSLICH im JSON-Format:
{"score": <1-5>, "correct_facts": ["..."], "missing_facts": ["..."], \
"wrong_facts": ["..."], "reasoning": "<Begründung>"}"""

COHERENCE_JUDGE_SYSTEM = """\
Du bist ein Sprachqualitätsprüfer. Bewerte die Antwort hinsichtlich \
Kohärenz, Struktur und Verständlichkeit.

Bewerte auf einer Skala von 1-5:
5 = Klar strukturiert, logisch aufgebaut, gut lesbar
4 = Überwiegend klar, kleine Schwächen in Struktur/Übergängen
3 = Verständlich, aber unstrukturiert oder redundant
2 = Schwer verständlich, unlogische Struktur
1 = Inkohärent, widersprüchlich, unverständlich

Antworte AUSSCHLIESSLICH im JSON-Format:
{"score": <1-5>, "issues": ["..."], "reasoning": "<Begründung>"}"""

INSTRUCTION_JUDGE_SYSTEM = """\
Du bist ein Evaluator für Instruktionsbefolgung. Prüfe ob die KI-Antwort \
die gegebene Anweisung korrekt und vollständig befolgt hat.

Bewerte auf einer Skala von 1-5:
5 = Alle Anweisungen exakt befolgt, Format und Inhalt wie gewünscht
4 = Überwiegend befolgt, minimale Abweichungen
3 = Kernaufgabe erfüllt, aber Teilanweisungen ignoriert
2 = Nur teilweise befolgt, wichtige Aspekte fehlen
1 = Anweisung nicht befolgt oder komplett missverstanden

Antworte AUSSCHLIESSLICH im JSON-Format:
{"score": <1-5>, "followed": ["<Befolgte Anweisungen>"], \
"violated": ["<Nicht befolgte Anweisungen>"], "reasoning": "<Begründung>"}"""

GERMAN_QUALITY_JUDGE_SYSTEM = """\
Du bist ein Sprachexperte für die deutsche Sprache. Bewerte die sprachliche \
Qualität der KI-Antwort speziell im Deutschen.

Prüfe:
- Grammatik und Syntax
- Natürlichkeit (kein "Übersetzungsdeutsch")
- Korrekte Fachterminologie
- Angemessener Stil / Register
- Korrekte Verwendung von Umlauten, Sonderzeichen

Bewerte auf einer Skala von 1-5:
5 = Muttersprachliches Niveau, natürlich und fehlerfrei
4 = Sehr gut, minimale stilistische Schwächen
3 = Korrekt aber steif / unnatürlich
2 = Häufige Fehler, erkennbar maschinell
1 = Schwere Grammatikfehler, unverständlich

Antworte AUSSCHLIESSLICH im JSON-Format:
{"score": <1-5>, "issues": ["<Konkrete Sprachfehler>"], \
"reasoning": "<Begründung>"}"""


class QualityEvaluator(BaseEvaluator):
    """Evaluiert Antwortqualität über LLM-as-Judge."""

    name = "quality"

    # Mapping subcategory → (system_prompt, knockout_threshold)
    JUDGE_CONFIGS: dict[str, tuple[str, float | None]] = {
        "hallucination": (HALLUCINATION_JUDGE_SYSTEM, 0.05),  # K.O. bei > 5%
        "factual": (FACTUAL_JUDGE_SYSTEM, None),
        "coherence": (COHERENCE_JUDGE_SYSTEM, None),
        "instruction": (INSTRUCTION_JUDGE_SYSTEM, None),
        "german_quality": (GERMAN_QUALITY_JUDGE_SYSTEM, None),
    }

    def evaluate(self, test_case: TestCase) -> EvalResult:
        """Bewerte einen Qualitäts-Testfall."""
        # 1. Antwort vom Zielmodell holen
        response, thinking, latency_ms, tokens = self.query_target(
            prompt=test_case.prompt,
            system_prompt=test_case.system_prompt,
        )

        # 2. Subcategory bestimmen
        subcat = test_case.subcategory or "factual"
        judge_system, _ = self.JUDGE_CONFIGS.get(
            subcat, (FACTUAL_JUDGE_SYSTEM, None)
        )

        # 3. Response klassifizieren
        from .base import _classify_response
        response_type = _classify_response(None if response == "" and not thinking else response or None, thinking)

        # 4. Leere / nicht-bewertbare Antwort abfangen
        if not response:
            logger.warning(
                "Leere Antwort von Zielmodell für %s (type=%s, thinking=%d chars) — kein Judge-Aufruf",
                test_case.id, response_type, len(thinking),
            )
            return EvalResult(
                test_id=test_case.id,
                model=self.target_model,
                evaluator=f"quality.{subcat}",
                verdict=Verdict.ERROR,
                score=0.0,
                response="",
                reasoning=f"Zielmodell lieferte keine Antwort (response_type={response_type})",
                latency_ms=latency_ms,
                tokens_generated=tokens,
                thinking=thinking,
                response_type=response_type,
            )

        # 5. Judge-Bewertung einholen
        judge_prompt = self._build_judge_prompt(test_case, response)
        judge_response = self.query_judge(
            prompt=judge_prompt,
            system_prompt=judge_system,
        )

        # 6. Judge-Antwort parsen
        score, reasoning = self._parse_judge_response(judge_response)

        # 7. Verdict bestimmen
        normalized_score = score / 5.0  # 1-5 → 0.0-1.0
        verdict = self._determine_verdict(subcat, normalized_score, test_case)

        return EvalResult(
            test_id=test_case.id,
            model=self.target_model,
            evaluator=f"quality.{subcat}",
            verdict=verdict,
            score=normalized_score,
            response=response,
            reasoning=reasoning,
            latency_ms=latency_ms,
            tokens_generated=tokens,
            thinking=thinking,
            response_type=response_type,
            metadata={"judge_raw": judge_response[:500]},
        )

    def _build_judge_prompt(self, test_case: TestCase, response: str) -> str:
        """Baue den Prompt für den Judge."""
        parts = [f"## Ursprüngliche Frage / Anweisung\n{test_case.prompt}"]

        if test_case.context:
            parts.append(f"\n## Gegebener Kontext\n{test_case.context}")

        if test_case.expected.value:
            parts.append(f"\n## Erwartete Referenzantwort\n{test_case.expected.value}")

        parts.append(f"\n## KI-Antwort (zu bewerten)\n{response}")

        return "\n".join(parts)

    def _parse_judge_response(self, judge_response: str) -> tuple[float, str]:
        """Parse die JSON-Antwort des Judges. Robust gegen Formatfehler."""
        return parse_json_response(judge_response, default_score=3.0)

    def _determine_verdict(
        self, subcategory: str, score: float, test_case: TestCase
    ) -> Verdict:
        """Bestimme das Verdict basierend auf Score und K.O.-Kriterien."""
        if subcategory == "hallucination" and score < 0.6:  # Score < 3/5
            return Verdict.KNOCKOUT

        # Prüfe forbidden-Begriffe in der Antwort (direkte K.O.-Indikatoren)
        if test_case.expected.forbidden:
            # Wird im Batch-Level aggregiert für Halluzinationsrate
            pass

        if score >= 0.8:
            return Verdict.PASS
        elif score >= 0.6:
            return Verdict.WARN
        else:
            return Verdict.FAIL
