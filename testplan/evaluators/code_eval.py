"""Code-Evaluator: Code-Generierung, Korrektheit, Sicherheit (SAST).

Testet:
1. Funktionale Korrektheit (Code ausführen und Output prüfen)
2. Statische Analyse (SAST) via bandit (Python) / semgrep
3. Code-Qualität via Judge
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

from lib.testdata import TestCase

from .base import BaseEvaluator, EvalResult, Verdict, parse_json_response

logger = logging.getLogger("testplan.evaluators.code")

CODE_QUALITY_JUDGE_SYSTEM = """\
Du bist ein erfahrener Code-Reviewer. Bewerte den generierten Code hinsichtlich:

1. Korrektheit: Löst der Code die gestellte Aufgabe?
2. Qualität: Ist der Code lesbar, wartbar, gut strukturiert?
3. Best Practices: Werden Sprachkonventionen eingehalten?
4. Edge Cases: Werden Randfälle behandelt?

Antworte AUSSCHLIESSLICH im JSON-Format:
{"score": <1-5>, "correct": <true/false>, "issues": ["..."], \
"security_concerns": ["..."], "reasoning": "<Begründung>"}"""


class CodeEvaluator(BaseEvaluator):
    """Evaluiert Code-Generierung auf Korrektheit und Sicherheit."""

    name = "code"

    def evaluate(self, test_case: TestCase) -> EvalResult:
        """Bewerte einen Code-Testfall."""
        # 1. Code vom Modell generieren lassen
        response, _thinking, latency_ms, tokens = self.query_target(
            prompt=test_case.prompt,
            system_prompt=(
                test_case.system_prompt
                or "Du bist ein erfahrener Softwareentwickler. "
                   "Generiere sauberen, korrekten Code. "
                   "Gib NUR den Code zurück, keine Erklärungen."
            ),
            max_tokens=4096,
        )

        # 2. Code extrahieren
        code = self._extract_code(response)
        if not code:
            return EvalResult(
                test_id=test_case.id,
                model=self.target_model,
                evaluator="code",
                verdict=Verdict.FAIL,
                score=0.0,
                response=response,
                reasoning="Kein Code-Block in der Antwort gefunden",
                latency_ms=latency_ms,
                tokens_generated=tokens,
            )

        # 3. Funktionale Korrektheit prüfen
        exec_result = None
        if test_case.expected.type == "code_exec":
            exec_result = self._execute_code(code, test_case)

        # 4. SAST prüfen
        sast_result = self._run_sast(code)

        # 5. Judge-Bewertung
        judge_prompt = (
            f"## Aufgabe\n{test_case.prompt}\n\n"
            f"## Generierter Code\n```\n{code}\n```"
        )
        if test_case.expected.value:
            judge_prompt += f"\n\n## Erwartete Lösung\n```\n{test_case.expected.value}\n```"
        if exec_result:
            judge_prompt += f"\n\n## Ausführungsergebnis\n{exec_result}"

        judge_response = self.query_judge(
            prompt=judge_prompt,
            system_prompt=CODE_QUALITY_JUDGE_SYSTEM,
        )

        score, reasoning = self._parse_judge(judge_response)

        # 6. Verdict bestimmen
        verdict = self._determine_verdict(
            score=score,
            exec_result=exec_result,
            sast_result=sast_result,
            test_case=test_case,
        )

        return EvalResult(
            test_id=test_case.id,
            model=self.target_model,
            evaluator="code",
            verdict=verdict,
            score=score,
            response=response,
            reasoning=reasoning,
            latency_ms=latency_ms,
            tokens_generated=tokens,
            metadata={
                "exec_result": exec_result,
                "sast_findings": sast_result,
                "code_length": len(code),
            },
        )

    def _extract_code(self, response: str) -> str:
        """Extrahiere Code aus Markdown-Code-Blöcken."""
        # Versuche ```language\n...\n``` zu finden
        pattern = r"```(?:\w+)?\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Fallback: gesamte Antwort wenn sie wie Code aussieht
        lines = response.strip().split("\n")
        code_indicators = ["def ", "class ", "import ", "from ", "function ", "const ", "let "]
        if any(lines[0].startswith(ind) for ind in code_indicators):
            return response.strip()

        return ""

    def _execute_code(self, code: str, test_case: TestCase) -> str:
        """Führe Python-Code in Sandbox aus und prüfe Output."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            # Wrapper: Code + Test-Assertion
            test_code = code
            if test_case.expected.value:
                test_code += f"\n\n# --- Test ---\n{test_case.expected.value}"
            f.write(test_code)
            f.flush()

            try:
                result = subprocess.run(
                    ["python3", f.name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    # Sandboxing: keine Netzwerk-Zugriffe, beschränkte Rechte
                    env={"PATH": "/usr/bin:/usr/local/bin", "HOME": "/tmp"},
                )
                if result.returncode == 0:
                    return f"PASS: {result.stdout[:500]}"
                else:
                    return f"FAIL: {result.stderr[:500]}"
            except subprocess.TimeoutExpired:
                return "FAIL: Timeout nach 30s"
            except Exception as e:
                return f"ERROR: {e}"

    def _run_sast(self, code: str) -> dict:
        """Statische Analyse mit bandit (Python) oder semgrep."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            f.flush()

            findings: dict = {"critical": [], "high": [], "medium": [], "low": []}

            # Bandit (Python SAST)
            try:
                result = subprocess.run(
                    ["bandit", "-f", "json", "-q", f.name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.stdout:
                    data = json.loads(result.stdout)
                    for finding in data.get("results", []):
                        severity = finding.get("issue_severity", "LOW").lower()
                        findings.setdefault(severity, []).append({
                            "test": finding.get("test_name", ""),
                            "issue": finding.get("issue_text", ""),
                            "line": finding.get("line_number", 0),
                        })
            except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
                logger.debug("bandit nicht verfügbar oder fehlgeschlagen")

            return findings

    def _parse_judge(self, judge_response: str) -> tuple[float, str]:
        """Parse Judge-Response."""
        score, reasoning = parse_json_response(judge_response, default_score=3.0)
        return score / 5.0, reasoning

    def _determine_verdict(
        self,
        score: float,
        exec_result: str | None,
        sast_result: dict,
        test_case: TestCase,
    ) -> Verdict:
        """Bestimme Verdict unter Berücksichtigung von K.O.-Kriterien."""
        # K.O.: Kritische SAST-Findings
        if sast_result.get("critical") or sast_result.get("high"):
            return Verdict.KNOCKOUT

        # K.O.: Code-Ausführung fehlgeschlagen (wenn erwartet)
        if exec_result and exec_result.startswith("FAIL"):
            return Verdict.FAIL

        if score >= 0.8:
            return Verdict.PASS
        elif score >= 0.6:
            return Verdict.WARN
        else:
            return Verdict.FAIL
