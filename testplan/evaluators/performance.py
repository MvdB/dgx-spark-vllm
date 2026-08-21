"""Performance-Evaluator: TTFT, Throughput, Latenz, Skalierungstests.

Misst:
1. Time to First Token (TTFT) — Streaming-basiert
2. Tokens pro Sekunde (Throughput) — Einzelanfrage
3. Latenz-Verteilung (P50, P95, P99)
4. Concurrent-Request-Degradation (Skalierungstest)
5. HSF-Kalibrierung (DGX Spark vs. Produktionsmaschine)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from statistics import mean, median, quantiles

import aiohttp
import numpy as np

from lib.config import Thresholds

logger = logging.getLogger("testplan.evaluators.performance")


@dataclass
class LatencyMeasurement:
    """Einzelne Latenz-Messung."""

    ttft_ms: float          # Time to First Token
    total_ms: float         # Gesamte Antwortzeit
    tokens_generated: int
    tok_per_sec: float
    error: str = ""


@dataclass
class PerformanceReport:
    """Aggregiertes Performance-Ergebnis für ein Modell."""

    model: str
    measurements: list[LatencyMeasurement] = field(default_factory=list)
    measurements_by_type: dict[str, list[LatencyMeasurement]] = field(default_factory=dict)
    concurrent_results: dict[int, list[LatencyMeasurement]] = field(default_factory=dict)

    @property
    def ttft_values(self) -> list[float]:
        return [m.ttft_ms for m in self.measurements if not m.error]

    @property
    def throughput_values(self) -> list[float]:
        return [m.tok_per_sec for m in self.measurements if not m.error]

    def ttft_percentile(self, p: int) -> float:
        vals = sorted(self.ttft_values)
        if not vals:
            return 0.0
        if p == 50:
            return median(vals)
        idx = int(len(vals) * p / 100)
        return vals[min(idx, len(vals) - 1)]

    def summary(self) -> dict:
        ttft = self.ttft_values
        tps = self.throughput_values

        # Aufschlüsselung nach Prompt-Typ (short/medium/long)
        throughput_by_type: dict[str, float] = {}
        for prompt_type, measurements in self.measurements_by_type.items():
            tps_type = [m.tok_per_sec for m in measurements if not m.error]
            if tps_type:
                throughput_by_type[prompt_type] = round(median(tps_type), 1)

        return {
            "model": self.model,
            "n_measurements": len(self.measurements),
            "n_errors": sum(1 for m in self.measurements if m.error),
            "ttft_p50_ms": median(ttft) if ttft else 0,
            "ttft_p95_ms": self.ttft_percentile(95),
            "ttft_p99_ms": self.ttft_percentile(99),
            "throughput_mean_tok_s": mean(tps) if tps else 0,
            "throughput_median_tok_s": median(tps) if tps else 0,
            "throughput_by_type": throughput_by_type,
            "concurrent_degradation": self._concurrent_summary(),
        }

    def _concurrent_summary(self) -> dict:
        result = {}
        for concurrency, measurements in sorted(self.concurrent_results.items()):
            ttfts = [m.ttft_ms for m in measurements if not m.error]
            errors = sum(1 for m in measurements if m.error)
            result[concurrency] = {
                "ttft_p50_ms": median(ttfts) if ttfts else 0,
                "error_rate": errors / len(measurements) if measurements else 0,
            }
        return result


# ---------------------------------------------------------------------------
# Standard-Prompts für Performance-Messungen (verschiedene Längen)
# ---------------------------------------------------------------------------

PERF_PROMPTS = {
    "short": "Was ist die Hauptstadt von Deutschland? Antworte in einem Satz.",
    "medium": (
        "Erkläre die Grundprinzipien der Objektorientierung in Python. "
        "Gib für jedes Prinzip ein kurzes Code-Beispiel."
    ),
    "long": (
        "Schreibe eine detaillierte Analyse der Vor- und Nachteile von "
        "Microservices-Architektur gegenüber Monolithen. Berücksichtige "
        "Aspekte wie Skalierbarkeit, Wartbarkeit, Deployment-Komplexität, "
        "Datenkonsistenz und Team-Organisation. Mindestens 500 Wörter."
    ),
}


class PerformanceEvaluator:
    """Misst Performance-Metriken via Streaming-API."""

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/v1/chat/completions"
        # Ein lokaler vLLM braucht keinen Schluessel, ein LiteLLM-Proxy schon.
        # Ohne ihn beantwortet der Proxy jede Anfrage mit 401, und der Bericht
        # zeigt saubere Nullen statt eines Fehlers — genau die Sorte Messwert,
        # die man fuer bare Muenze nimmt.
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def measure_single(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> LatencyMeasurement:
        """Einzelne Streaming-Messung: TTFT und Throughput."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            # Ohne include_usage bleibt nur die Zahl der Chunks, und die ist
            # nicht die Zahl der Token: bei spekulativer Dekodierung schickt
            # vLLM alle angenommenen Entwurfstoken eines Schritts in EINEM
            # Chunk. Der Benchmark hat Qwen3.8-27B-NVFP4-MTP dadurch mit 8,2
            # statt 24,2 tok/s gemessen — Faktor 2,94, also knapp drei Token
            # je Chunk bei drei Spekulationstoken. Aufgefallen 21.08.2026.
            "stream_options": {"include_usage": True},
        }

        tokens = 0
        chunks = 0
        ttft_ms = 0.0
        start = time.monotonic()
        first_token_time = None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return LatencyMeasurement(
                            ttft_ms=0, total_ms=0, tokens_generated=0,
                            tok_per_sec=0, error=f"HTTP {resp.status}: {body[:200]}",
                        )

                    async for line in resp.content:
                        decoded = line.decode("utf-8").strip()
                        if not decoded.startswith("data: "):
                            continue
                        data = decoded[6:]
                        if data == "[DONE]":
                            break

                        if first_token_time is None:
                            first_token_time = time.monotonic()
                            ttft_ms = (first_token_time - start) * 1000

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        # Der Nutzungs-Chunk kommt zuletzt und hat ein leeres
                        # choices. Er zaehlt nicht als Ausgabe-Chunk, sonst
                        # waere der Rueckfallwert um eins zu hoch.
                        if chunk.get("choices"):
                            chunks += 1
                        usage = chunk.get("usage") or {}
                        if usage.get("completion_tokens"):
                            tokens = int(usage["completion_tokens"])

        except asyncio.TimeoutError:
            return LatencyMeasurement(
                ttft_ms=ttft_ms, total_ms=(time.monotonic() - start) * 1000,
                tokens_generated=tokens, tok_per_sec=0, error="Timeout",
            )
        except Exception as e:
            return LatencyMeasurement(
                ttft_ms=0, total_ms=0, tokens_generated=0,
                tok_per_sec=0, error=str(e),
            )

        if not tokens and chunks:
            # Aelteres oder fremdes Backend ohne stream_options. Dann ist die
            # Chunkzahl die einzige Naeherung — fuer nichtspekulatives Serving
            # stimmt sie, fuer spekulatives misst sie zu niedrig. Deshalb laut.
            logger.warning(
                "Keine Nutzungsdaten im Stream — Durchsatz aus %d Chunks "
                "genaehert. Bei spekulativer Dekodierung ist dieser Wert zu "
                "niedrig und gehoert nicht in einen Vergleich.", chunks,
            )
            tokens = chunks

        total_ms = (time.monotonic() - start) * 1000
        gen_time = total_ms - ttft_ms
        tok_per_sec = (tokens / gen_time * 1000) if gen_time > 0 else 0

        return LatencyMeasurement(
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_generated=tokens,
            tok_per_sec=tok_per_sec,
        )

    async def run_benchmark(
        self,
        n_iterations: int = 20,
        warmup: int = 3,
    ) -> PerformanceReport:
        """Vollständiger Performance-Benchmark.

        1. Warmup-Runden (werden verworfen)
        2. Messungen mit verschiedenen Prompt-Längen
        3. Concurrent-Request-Tests
        """
        report = PerformanceReport(model=self.model)

        # Warmup
        logger.info("Warmup: %d Runden...", warmup)
        for _ in range(warmup):
            await self.measure_single(PERF_PROMPTS["short"], max_tokens=50)

        # Einzelmessungen
        iters_by_type = {"short": n_iterations, "medium": n_iterations, "long": 10}
        for prompt_type, prompt in PERF_PROMPTS.items():
            max_tok = {"short": 64, "medium": 256, "long": 512}[prompt_type]
            n_iters = iters_by_type[prompt_type]
            logger.info("Benchmark '%s': %d Iterationen...", prompt_type, n_iters)
            report.measurements_by_type[prompt_type] = []

            for i in range(n_iters):
                m = await self.measure_single(prompt, max_tokens=max_tok)
                m.error = m.error or ""
                report.measurements.append(m)
                report.measurements_by_type[prompt_type].append(m)

                if m.error:
                    logger.warning("Iteration %d/%d Fehler: %s", i + 1, n_iterations, m.error)

        # Concurrent-Request-Tests
        for concurrency in [1, 5, 10, 25, 50]:
            logger.info("Concurrent-Test: %d parallele Requests...", concurrency)
            results = await self._concurrent_test(concurrency, n_requests=concurrency * 2)
            report.concurrent_results[concurrency] = results

        return report

    async def _concurrent_test(
        self,
        concurrency: int,
        n_requests: int,
    ) -> list[LatencyMeasurement]:
        """Parallele Requests zur Messung von Degradation unter Last."""
        sem = asyncio.Semaphore(concurrency)

        async def bounded_request() -> LatencyMeasurement:
            async with sem:
                return await self.measure_single(
                    PERF_PROMPTS["medium"], max_tokens=128,
                )

        tasks = [bounded_request() for _ in range(n_requests)]
        return list(await asyncio.gather(*tasks))

    def check_thresholds(
        self,
        report: PerformanceReport,
        thresholds: Thresholds,
        model_params_b: int = 0,
        model_tags: list[str] | None = None,
    ) -> list[str]:
        """Prüfe Performance gegen konfigurierte Schwellenwerte."""
        violations: list[str] = []
        summary = report.summary()

        if summary["ttft_p50_ms"] > thresholds.ttft_p50_ms:
            violations.append(
                f"TTFT P50 ({summary['ttft_p50_ms']:.0f}ms) "
                f"überschreitet Schwellenwert ({thresholds.ttft_p50_ms}ms)"
            )
        if summary["ttft_p95_ms"] > thresholds.ttft_p95_ms:
            violations.append(
                f"TTFT P95 ({summary['ttft_p95_ms']:.0f}ms) "
                f"überschreitet Schwellenwert ({thresholds.ttft_p95_ms}ms)"
            )
        tput_threshold = thresholds.throughput_for_model(model_params_b, model_tags)
        if summary["throughput_median_tok_s"] < tput_threshold:
            violations.append(
                f"Throughput ({summary['throughput_median_tok_s']:.1f} tok/s) "
                f"unter Schwellenwert ({tput_threshold} tok/s)"
            )

        return violations


class HSFCalibrator:
    """Hardware-Skalierungsfaktor-Kalibrierung.

    Fährt identische Workloads auf DGX Spark und Produktionsmaschine,
    berechnet Skalierungsfaktoren mit Unsicherheitskorridor.
    """

    def __init__(
        self,
        spark_evaluator: PerformanceEvaluator,
        prod_evaluator: PerformanceEvaluator,
    ):
        self.spark = spark_evaluator
        self.prod = prod_evaluator

    async def calibrate(
        self,
        n_samples: int = 50,
    ) -> dict:
        """Führe Kalibrierungslauf durch.

        Returns:
            HSF-Tabelle mit Skalierungsfaktoren und Konfidenzintervall
        """
        logger.info("HSF-Kalibrierung: %d Samples pro Maschine...", n_samples)

        # Identische Workloads auf beiden Maschinen
        spark_results: list[LatencyMeasurement] = []
        prod_results: list[LatencyMeasurement] = []

        for prompt_type, prompt in PERF_PROMPTS.items():
            max_tok = {"short": 64, "medium": 256, "long": 512}[prompt_type]

            for _ in range(n_samples // len(PERF_PROMPTS)):
                spark_m = await self.spark.measure_single(prompt, max_tokens=max_tok)
                prod_m = await self.prod.measure_single(prompt, max_tokens=max_tok)
                spark_results.append(spark_m)
                prod_results.append(prod_m)

        # Skalierungsfaktoren berechnen
        spark_ttfts = np.array([m.ttft_ms for m in spark_results if not m.error])
        prod_ttfts = np.array([m.ttft_ms for m in prod_results if not m.error])
        spark_tps = np.array([m.tok_per_sec for m in spark_results if not m.error])
        prod_tps = np.array([m.tok_per_sec for m in prod_results if not m.error])

        hsf_ttft = float(np.median(spark_ttfts) / np.median(prod_ttfts)) if len(prod_ttfts) > 0 else 1.0
        hsf_throughput = float(np.median(prod_tps) / np.median(spark_tps)) if len(spark_tps) > 0 else 1.0

        # Bootstrap-Konfidenzintervall
        n_bootstrap = 1000
        bootstrap_ttft = []
        bootstrap_tps = []
        for _ in range(n_bootstrap):
            s_idx = np.random.choice(len(spark_ttfts), size=len(spark_ttfts), replace=True)
            p_idx = np.random.choice(len(prod_ttfts), size=len(prod_ttfts), replace=True)
            bootstrap_ttft.append(
                float(np.median(spark_ttfts[s_idx]) / np.median(prod_ttfts[p_idx]))
            )
            bootstrap_tps.append(
                float(np.median(prod_tps[p_idx]) / np.median(spark_tps[s_idx]))
            )

        return {
            "model": self.spark.model,
            "n_samples": len(spark_results),
            "hsf_ttft": {
                "factor": hsf_ttft,
                "ci_lower": float(np.percentile(bootstrap_ttft, 2.5)),
                "ci_upper": float(np.percentile(bootstrap_ttft, 97.5)),
                "interpretation": (
                    f"Prod-TTFT ≈ Spark-TTFT × {1/hsf_ttft:.2f} "
                    f"(Produktion ist {hsf_ttft:.1f}× schneller)"
                    if hsf_ttft > 1 else
                    f"Prod-TTFT ≈ Spark-TTFT × {1/hsf_ttft:.2f}"
                ),
            },
            "hsf_throughput": {
                "factor": hsf_throughput,
                "ci_lower": float(np.percentile(bootstrap_tps, 2.5)),
                "ci_upper": float(np.percentile(bootstrap_tps, 97.5)),
                "interpretation": (
                    f"Prod-Throughput ≈ Spark-Throughput × {hsf_throughput:.2f}"
                ),
            },
        }
