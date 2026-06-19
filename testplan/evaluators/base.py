"""Basis-Evaluator und gemeinsame Typen für alle Bewertungsmodule."""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import unicodedata

from openai import OpenAI, APIStatusError

from lib.testdata import TestCase

logger = logging.getLogger("testplan.evaluators")

# Sentinel: max_model_len wurde noch nie abgefragt (vs. None = abgefragt, unbekannt).
_MAX_LEN_UNFETCHED = -1
# Sicherheitsabstand (Tokens) zwischen Prompt+Output und Kontextfenster.
_CONTEXT_SAFETY_MARGIN = 32

# Heuristik: Safety-Refusal-Phrases (DE + EN)
_SAFETY_PATTERNS = re.compile(
    r"\b(i (can'?t|cannot|won'?t|will not|am unable to|refuse to)|"
    r"i'?m (not able|unable)|"
    r"ich (kann|werde|darf) (das |es |nicht|leider)|"
    r"als (ki|ki-modell|sprachmodell)|"
    r"as an (ai|language model)|"
    r"i'?m sorry,? (but )?i|"
    r"sorry,? (but )?i (can'?t|cannot)|"
    r"tut mir leid|"
    r"entschuldigung,? aber ich)\b",
    re.IGNORECASE,
)


_TOKENIZER_ERROR_PATTERN = re.compile(
    r"unexpected tokens remaining in message header|could not decode header",
    re.IGNORECASE,
)

# Sonderzeichen die tiktoken-basierte Tokenizer aus dem Takt bringen können
_NBSP_VARIANTS = re.compile(r"[\u00a0\u2007\u202f\u2060\ufeff]")
# <|...|>-Muster die als Special Tokens interpretiert werden könnten
_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^|>]{1,30}\|>")


def _sanitize_prompt(text: str) -> str:
    """Normalisiere Prompt für Tokenizer-kompatible Darstellung.

    Angewendet als Fallback nach HTTP-500-Tokenizer-Fehler.
    - NFKC-Normalisierung (z.B. non-breaking spaces → standard)
    - Verbleibende NBSP-Varianten → Leerzeichen
    - <|special|>-Token-Artefakte entfernen
    """
    text = unicodedata.normalize("NFKC", text)
    text = _NBSP_VARIANTS.sub(" ", text)
    text = _SPECIAL_TOKEN_PATTERN.sub("", text)
    return text


def _classify_response(content: str | None, thinking: str) -> str:
    """Klassifiziere den Response-Typ für Debugging und Reporting."""
    if content is None:
        return "none"
    if content == "":
        return "thinking_only" if thinking else "empty"
    if len(content) < 300 and _SAFETY_PATTERNS.search(content):
        return "safety_refusal"
    return "answer"


def parse_json_response(text: str, default_score: float = 3.0) -> tuple[float, str]:
    """Parse eine JSON-Antwort des Judges — robust gegen Markdown-Code-Blöcke.

    Gibt (score, reasoning) zurück. Score ist roh (nicht normalisiert).
    """
    import json
    import re

    # Markdown-Code-Block entfernen (```json ... ``` oder ``` ... ```)
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        stripped = stripped.strip()

    # Direkt parsen
    try:
        data = json.loads(stripped)
        return float(data.get("score", default_score)), str(data.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex: erstes JSON-Objekt extrahieren (auch mit verschachtelten Arrays)
    json_match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return float(data.get("score", default_score)), str(data.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError):
            pass

    # Score aus Text extrahieren
    score_match = re.search(r'"?score"?\s*[:=]\s*(\d)', text)
    if score_match:
        return float(score_match.group(1)), text[:200]

    logger.warning("Judge-Antwort nicht parsebar: %s", text[:200])
    return default_score, f"Parse-Fehler: {text[:200]}"


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
    response: str            # Modell-Antwort (ohne Thinking-Teil)
    reasoning: str = ""      # Begründung (vom Judge oder Evaluator)
    latency_ms: float = 0.0  # Antwortzeit
    tokens_generated: int = 0
    thinking: str = ""       # Chain-of-Thought / reasoning_content des Modells
    response_type: str = "answer"  # "answer" | "none" | "empty" | "thinking_only" | "safety_refusal"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "model": self.model,
            "evaluator": self.evaluator,
            "verdict": self.verdict.value,
            "score": self.score,
            "response_type": self.response_type,
            "response": self.response[:500],  # Gekürzt für Reports
            "thinking": self.thinking[:1000] if self.thinking else "",  # Thinking-Excerpt
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
        # Kontextfenster des Zielmodells (max_model_len) — lazy aus /v1/models.
        # _MAX_LEN_UNFETCHED = noch nicht abgefragt; None = unbekannt/nicht verfügbar.
        self._target_max_model_len: int | None = _MAX_LEN_UNFETCHED

    def _model_prompt(self, test_case: TestCase) -> str:
        """Kombiniert context und prompt für den Modell-Query.

        Wenn der Testfall ein context-Feld hat (z.B. Vertragsdokument, RAG-Kontext),
        wird es dem Modell vorangestellt — ohne dieses würde das Modell blind antworten.
        """
        if test_case.context:
            return f"{test_case.context}\n\n{test_case.prompt}"
        return test_case.prompt

    def _target_context_window(self) -> int | None:
        """max_model_len des Zielmodells (aus /v1/models), gecacht.

        Liefert None, wenn das Feld nicht ermittelbar ist (dann kein Clamping).
        """
        if self._target_max_model_len is not _MAX_LEN_UNFETCHED:
            return self._target_max_model_len
        max_len: int | None = None
        try:
            for model in self.target_client.models.list().data:
                data = model.model_dump()
                if model.id == self.target_model or max_len is None:
                    val = data.get("max_model_len")
                    if isinstance(val, int) and val > 0:
                        max_len = val
                if model.id == self.target_model:
                    break
        except Exception as e:  # /v1/models nicht erreichbar o. Feld fehlt → kein Clamp
            logger.debug("max_model_len nicht ermittelbar: %s", e)
        self._target_max_model_len = max_len
        return max_len

    def _clamp_max_tokens(self, max_tokens: int, messages: list[dict[str, str]]) -> int:
        """Begrenzt max_tokens, sodass Prompt + Output ins Kontextfenster passen.

        Ohne dieses Clamping schlägt z.B. der Code-Evaluator (max_tokens=4096) bei
        Modellen mit kleinem Kontext (Apertus: max_model_len=4096) mit HTTP 400 fehl,
        weil Prompt + 4096 Output-Tokens das Fenster überschreiten.
        """
        ctx = self._target_context_window()
        if not ctx:
            return max_tokens
        # Konservative Token-Schätzung des Prompts (~3 Zeichen/Token + Rollen-Overhead).
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_est = prompt_chars // 3 + 8 * len(messages)
        budget = ctx - prompt_est - _CONTEXT_SAFETY_MARGIN
        if budget <= 0:
            # Prompt allein sprengt den Kontext — Clamp kann nichts retten.
            return max_tokens
        if max_tokens > budget:
            logger.info(
                "max_tokens %d → %d gekürzt (Kontext %d, Prompt~%d) für '%s'",
                max_tokens, budget, ctx, prompt_est, self.target_model,
            )
            return budget
        return max_tokens

    def query_target(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        timeout: int = 300,
    ) -> tuple[str, str, float, int]:
        """Sende Anfrage an das Zielmodell.

        Returns:
            (response_text, thinking, latency_ms, tokens_generated)

            thinking: Chain-of-Thought aus reasoning_content (vLLM Reasoning-API)
                      oder aus inline <think>...</think>-Tags im Content.
                      Leer wenn Modell kein Thinking unterstützt.
        """
        messages: list[dict[str, str]] = []
        effective_system = system_prompt or self.default_system_prompt
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        # max_tokens an das Kontextfenster des Modells anpassen (verhindert HTTP 400
        # bei Modellen mit kleinem Kontext, z.B. Apertus max_model_len=4096).
        max_tokens = self._clamp_max_tokens(max_tokens, messages)

        start = time.monotonic()
        sanitized = False
        try:
            completion = self.target_client.chat.completions.create(
                model=self.target_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        except APIStatusError as e:
            if e.status_code == 500 and _TOKENIZER_ERROR_PATTERN.search(str(e)):
                logger.warning(
                    "Tokenizer-Fehler (500) für '%s' — sanitisiere Prompt und retry",
                    prompt[:80],
                )
                sanitized = True
                clean_messages = [
                    {**msg, "content": _sanitize_prompt(msg["content"])}
                    for msg in messages
                ]
                try:
                    completion = self.target_client.chat.completions.create(
                        model=self.target_model,
                        messages=clean_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout,
                    )
                except Exception as retry_e:
                    latency_ms = (time.monotonic() - start) * 1000
                    logger.error("Retry nach Sanitisierung fehlgeschlagen: %s", retry_e)
                    raise
            else:
                latency_ms = (time.monotonic() - start) * 1000
                logger.error("Target-Query fehlgeschlagen: %s", e)
                raise
        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error("Target-Query fehlgeschlagen: %s", e)
            raise

        try:
            latency_ms = (time.monotonic() - start) * 1000
            message = completion.choices[0].message
            raw_content = message.content  # kann None sein

            # Thinking aus reasoning_content (vLLM Reasoning-API, z.B. Qwen3.5, DeepSeek-R1)
            thinking: str = getattr(message, "reasoning_content", None) or ""

            # Fallback: inline <think>...</think> aus Content extrahieren
            content = raw_content or ""
            if not thinking and "<think>" in content:
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Fallback: Modell hat nur thinking geliefert, kein Content (Reasoning-Modell)
            if not content and thinking:
                logger.warning(
                    "Modell lieferte nur thinking_content, kein Content — verwende thinking als Antwort: %s",
                    prompt[:60],
                )
                content = thinking

            response_type = _classify_response(raw_content, thinking)
            if response_type != "answer":
                logger.warning(
                    "Unerwarteter Response-Typ für %s: %s (thinking=%d chars)",
                    prompt[:60],
                    response_type,
                    len(thinking),
                )

            tokens = completion.usage.completion_tokens if completion.usage else 0
            if sanitized:
                logger.info("Prompt-Sanitisierung erfolgreich für '%s'", prompt[:80])
            return content, thinking, latency_ms, tokens, sanitized
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
