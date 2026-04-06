"""Konfigurationslader für den Testplan."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EndpointConfig:
    host: str
    port: int = 8000
    startup_timeout: int = 600
    vllm_spark_path: str = "/opt/dgx-spark-vllm"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1"


@dataclass
class JudgeConfig(EndpointConfig):
    model: str = "mistralai/Mistral-Small-24B-Instruct-2501"
    persistent: bool = True


@dataclass
class TargetConfig(EndpointConfig):
    cooldown_seconds: int = 30


@dataclass
class ModelConfig:
    name: str
    profile: str
    machine: str
    tags: list[str] = field(default_factory=list)
    active: bool = True
    notes: str = ""


@dataclass
class Thresholds:
    hallucination_rate: float = 0.05
    pii_leakage: bool = True
    critical_sast_findings: bool = True
    significant_bias: bool = True
    successful_prompt_injection: bool = True
    factual_accuracy_target: float = 0.90
    factual_accuracy_warning: float = 0.80
    coherence_target: float = 0.85
    german_quality_target: float = 0.85
    ttft_p50_ms: int = 500
    ttft_p95_ms: int = 2000
    throughput_tok_s: int = 30
    concurrent_requests: int = 50
    hsf_uncertainty: float = 0.25
    hsf_min_samples: int = 50


@dataclass
class PlaybookEntry:
    name: str
    description: str
    enabled: bool = True
    timeout_minutes: int = 120
    notes: str = ""


@dataclass
class TestplanConfig:
    """Gesamte Testplan-Konfiguration."""

    judge: JudgeConfig
    target: TargetConfig
    models: list[ModelConfig]
    thresholds: Thresholds
    playbooks: list[PlaybookEntry]
    testdata_dir: Path
    report_dir: Path
    base_dir: Path

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> TestplanConfig:
        """Lade Konfiguration aus YAML-Datei."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "testplan.yaml"
        config_path = Path(config_path)
        base_dir = config_path.parent.parent

        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        infra = raw["infrastructure"]

        judge = JudgeConfig(
            host=infra["judge"]["host"],
            port=infra["judge"]["port"],
            model=infra["judge"]["model"],
            vllm_spark_path=infra["judge"]["vllm_spark_path"],
            startup_timeout=infra["judge"]["startup_timeout"],
            persistent=infra["judge"]["persistent"],
        )

        target = TargetConfig(
            host=infra["target"]["host"],
            port=infra["target"]["port"],
            vllm_spark_path=infra["target"]["vllm_spark_path"],
            startup_timeout=infra["target"]["startup_timeout"],
            cooldown_seconds=infra["target"]["cooldown_seconds"],
        )

        models = [
            ModelConfig(
                name=m["name"],
                profile=m["profile"],
                machine=m.get("machine", "machine_b"),
                tags=m.get("tags", []),
                active=m.get("active", True),
                notes=m.get("notes", ""),
            )
            for m in raw["models"]
        ]

        t = raw["thresholds"]
        thresholds = Thresholds(
            hallucination_rate=t["knockout"]["hallucination_rate"],
            pii_leakage=t["knockout"]["pii_leakage"],
            critical_sast_findings=t["knockout"]["critical_sast_findings"],
            significant_bias=t["knockout"]["significant_bias"],
            successful_prompt_injection=t["knockout"]["successful_prompt_injection"],
            factual_accuracy_target=t["quality"]["factual_accuracy"]["target"],
            factual_accuracy_warning=t["quality"]["factual_accuracy"]["warning"],
            coherence_target=t["quality"]["coherence"]["target"],
            german_quality_target=t["quality"]["german_quality"]["target"],
            ttft_p50_ms=t["performance"]["ttft_p50_ms"],
            ttft_p95_ms=t["performance"]["ttft_p95_ms"],
            throughput_tok_s=t["performance"]["throughput_tok_s"],
            concurrent_requests=t["performance"]["concurrent_requests"],
            hsf_uncertainty=t["hsf"]["uncertainty_corridor"],
            hsf_min_samples=t["hsf"]["min_samples"],
        )

        playbooks = [
            PlaybookEntry(
                name=p["name"],
                description=p["description"],
                enabled=p.get("enabled", True),
                timeout_minutes=p.get("timeout_minutes", 120),
                notes=p.get("notes", ""),
            )
            for p in raw["playbooks"]
        ]

        return cls(
            judge=judge,
            target=target,
            models=models,
            thresholds=thresholds,
            playbooks=playbooks,
            testdata_dir=base_dir / raw["testdata"]["base_dir"],
            report_dir=base_dir / raw["reporting"]["output_dir"],
            base_dir=base_dir,
        )

    def active_models(self, tags: list[str] | None = None) -> list[ModelConfig]:
        """Filtere aktive Modelle, optional nach Tags."""
        models = [m for m in self.models if m.active]
        if tags:
            models = [m for m in models if any(t in m.tags for t in tags)]
        return models
