"""Basis-Evaluator und gemeinsame Typen für alle Bewertungsmodule."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openai import OpenAI

from lib.testdata import TestCase

logger = logging.getLogger("testplan.evaluators")


class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    KNOCKOUT = "knockout"  # K.O.-Kriterium verletzt
    ERROR = "error"        # Technischer Fehler bei Auswertung


@dataclass
class EvalResult:
    """Ergebnis der Bewertung eines einzelnen Testfalls."""

    test_id: str
    model: str
    evaluator: str
    verdict: Verdict
    score: float             # 0.0 - 1.0
    response: str            # Modell-Antwort
    reasoning: str = ""      # Begründung (vom Judge oder Evaluator)
    latency_ms: float = 0.0  # Antwortzeit
    tokens_generated: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "model": self.model,
            "evaluator": self.evaluator,
            "verdict": self.verdict.value,
            "score": self.score,
            "response": self.response[:500],  # Gekürzt für Reports
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
            "tokens_generated": self.tokens_generated,
            "metadata": self.metadata,
        }


@dataclass
class PlaybookResult:
    """Aggregiertes Ergebnis eines kompletten Playbook-Durchlaufs."""

    playbook: str
    model: str
    results: list[EvalResult]
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.verdict in (Verdict.FAIL, Verdict.KNOCKOUT))

    @property
    def knockouts(self) -> list[EvalResult]:
        return [r for r in self.results if r.verdict == Verdict.KNOCKOUT]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def mean_score(self) -> float:
        scores = [r.score for r in self.results if r.verdict != Verdict.ERROR]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def has_knockout(self) -> bool:
        return len(self.knockouts) > 0


class BaseEvaluator(ABC):
    """Abstrakte Basis für alle Evaluatoren."""

    name: str = "base"

    def __init__(
        self,
        target_client: OpenAI,
        target_model: str,
        judge_client: OpenAI | None = None,
        judge_model: str | None = None,
        default_system_prompt: str = "",
    ):
        self.target_client = target_client
        self.target_model = target_model
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.default_system_prompt = default_system_prompt

    def query_target(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> tuple[str, float, int]:
        """Sende Anfrage an das Zielmodell.

        Returns:
            (response_text, latency_ms, tokens_generated)
        """
        messages: list[dict[str, str]] = []
        effective_system = system_prompt or self.default_system_prompt
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        try:
            completion = self.target_client.chat.completions.create(
                model=self.target_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms = (time.monotonic() - start) * 1000
            response = completion.choices[0].message.content or ""
            tokens = completion.usage.completion_tokens if completion.usage else 0
            return response, latency_ms, tokens
        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error("Target-Query fehlgeschlagen: %s", e)
            raise

    def query_judge(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Sende Anfrage an das Judge-Modell."""
        if not self.judge_client or not self.judge_model:
            raise RuntimeError("Kein Judge-Modell konfiguriert")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content or ""

    @abstractmethod
    def evaluate(self, test_case: TestCase) -> EvalResult:
        """Bewerte einen einzelnen Testfall."""
        ...

    def evaluate_batch(self, test_cases: list[TestCase]) -> list[EvalResult]:
        """Bewerte eine Liste von Testfällen sequenziell."""
        results: list[EvalResult] = []
        for i, tc in enumerate(test_cases, 1):
            logger.info(
                "[%s] %d/%d: %s", self.name, i, len(test_cases), tc.id
            )
            try:
                result = self.evaluate(tc)
                results.append(result)
            except Exception as e:
                logger.error("Fehler bei %s: %s", tc.id, e)
                results.append(
                    EvalResult(
                        test_id=tc.id,
                        model=self.target_model,
                        evaluator=self.name,
                        verdict=Verdict.ERROR,
                        score=0.0,
                        response="",
                        reasoning=f"Evaluator-Fehler: {e}",
                    )
                )
        return results
