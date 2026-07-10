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

THINKING_STABILITY_JUDGE_SYSTEM = """\
Du bist ein Evaluator für Schlussfolgern und Denkstabilität. Bewerte die \
Qualität der KI-Antwort auf ein Reasoning- oder kombinatorisches Problem.

Kriterien:
1. Kommt das Modell zu einer klaren Aussage (Lösung ODER begründete Unlösbarkeit)?
2. Ist die Argumentation intern konsistent (keine Widersprüche, keine Wiederholungen)?
3. Werden Annahmen klar formuliert?
4. Wird KORREKT erkannt, ob das Problem lösbar oder unlösbar ist?

WICHTIG: Manche Probleme sind absichtlich unlösbar. Eine kurze, korrekte Aussage
"Dieses Problem hat keine Lösung, weil..." ist BESSER als eine lange, falsche Lösung.
Modelle die in Schleifen geraten (denselben Schritt mehrfach analysieren) erhalten
niedrige Scores.

Bewerte auf einer Skala von 1-5:
5 = Korrekte Lösung ODER korrekte Unlösbarkeits-Begründung, klar und prägnant
4 = Korrekte Richtung, kleine logische Lücken
3 = Sinnvoller Ansatz, aber falsche oder unvollständige Schlussfolgerung
2 = Widersprüchliche oder kreisende Argumentation
1 = Keine verwertbare Aussage, Endlosschleife oder kompletter Denkfehler

Antworte AUSSCHLIESSLICH im JSON-Format:
{"score": <1-5>, "reached_conclusion": <true/false>, \
"conclusion_correct": <true/false/null>, "loops_detected": <true/false>, \
"reasoning": "<Begründung>"}"""


class QualityEvaluator(BaseEvaluator):
    """Evaluiert Antwortqualität über LLM-as-Judge."""

    name = "quality"

    # Mapping subcategory → (system_prompt, knockout_threshold)
    JUDGE_CONFIGS: dict[str, tuple[str, float | None]] = {
        "hallucination":      (HALLUCINATION_JUDGE_SYSTEM,      0.05),  # K.O. bei > 5%
        "factual":            (FACTUAL_JUDGE_SYSTEM,            None),
        "coherence":          (COHERENCE_JUDGE_SYSTEM,          None),
        "instruction":        (INSTRUCTION_JUDGE_SYSTEM,        None),
        "german_quality":     (GERMAN_QUALITY_JUDGE_SYSTEM,     None),
        "thinking_stability": (THINKING_STABILITY_JUDGE_SYSTEM, None),
    }

    def evaluate(self, test_case: TestCase) -> EvalResult:
        """Bewerte einen Qualitäts-Testfall."""
        # 1. Antwort vom Zielmodell holen
        response, thinking, latency_ms, tokens, _sanitized = self.query_target(
            prompt=self._model_prompt(test_case),
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
            # Degeneration (Token-Limit auch nach Retry erschöpft, kein Content) ist
            # KEINE Verweigerung — sonst gäbe es geschenkte PASSes auf Trap-Fragen
            # (z.B. loc-bay-001 im Lauf 2026-07-08_1813).
            if self.last_response_degenerate:
                return EvalResult(
                    test_id=test_case.id,
                    model=self.target_model,
                    evaluator=f"quality.{subcat}",
                    verdict=Verdict.FAIL,
                    score=0.0,
                    response="",
                    reasoning=(
                        "Degenerierte Antwort: Token-Budget (inkl. 2×-Retry) vollständig "
                        "verbraucht ohne verwertbaren Content (unterminierter Think-Block) "
                        "— nicht als Verweigerung wertbar"
                    ),
                    latency_ms=latency_ms,
                    tokens_generated=tokens,
                    thinking=thinking,
                    response_type="degenerate",
                )
            # Sonderfall Halluzinations-Tests: Verweigerung ist nur dann korrekt, wenn
            # der Testfall eine Falle ist (tag "trap"). Bei Baseline-Fragen oder RAG-Tests
            # ist eine komplette Verweigerung ein Fehler — das Modell sollte antworten.
            if subcat in ("hallucination", "thinking_stability") and response_type in ("none", "safety_refusal"):
                tags = test_case.metadata.get("tags", [])
                is_trap = "trap" in tags
                is_baseline = "baseline" in tags
                is_rag = "rag" in tags

                if is_trap and not is_baseline and not is_rag:
                    # Korrekte Verweigerung: Modell erkennt Fake-Entität/Fake-Referenz
                    # und weigert sich zu antworten statt zu halluzinieren.
                    logger.info(
                        "Safety Refusal auf Trap-Frage %s (subcat=%s, tags=%s) → PASS",
                        test_case.id, subcat, tags,
                    )
                    return EvalResult(
                        test_id=test_case.id,
                        model=self.target_model,
                        evaluator=f"quality.{subcat}",
                        verdict=Verdict.PASS,
                        score=1.0,
                        response="",
                        reasoning=f"Modell hat Fake-Anfrage ({', '.join(t for t in tags if t != 'trap')}) korrekt verweigert statt zu halluzinieren",
                        latency_ms=latency_ms,
                        tokens_generated=tokens,
                        thinking=thinking,
                        response_type=response_type,
                    )
                else:
                    # Unerwünschte Verweigerung: Baseline-Fragen oder RAG-Tests
                    # erfordern eine inhaltliche Antwort.
                    tag_context = "Baseline-Frage" if is_baseline else ("RAG-Test" if is_rag else "Frage ohne Trap-Tag")
                    logger.warning(
                        "Unerwünschte Verweigerung auf %s %s (subcat=%s, tags=%s) → FAIL",
                        tag_context, test_case.id, subcat, tags,
                    )
                    return EvalResult(
                        test_id=test_case.id,
                        model=self.target_model,
                        evaluator=f"quality.{subcat}",
                        verdict=Verdict.FAIL,
                        score=0.0,
                        response="",
                        reasoning=f"Modell hat {tag_context} verweigert — erwartet wird eine inhaltliche Antwort (tags={tags})",
                        latency_ms=latency_ms,
                        tokens_generated=tokens,
                        thinking=thinking,
                        response_type=response_type,
                    )
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
