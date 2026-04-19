"""Remote vLLM-Lifecycle-Management via SSH.

Steuert vllm_spark.sh auf den DGX Spark Maschinen:
- Modell starten (mit Profil aus dem Repo)
- Health-Check / Readiness warten
- Modell stoppen und GPU-Speicher freigeben
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import paramiko
from openai import OpenAI

from .config import EndpointConfig, JudgeConfig, ModelConfig, TargetConfig

logger = logging.getLogger("testplan.vllm_control")


@dataclass
class VllmInstance:
    """Repräsentiert eine laufende vLLM-Instanz."""

    endpoint: EndpointConfig
    model: ModelConfig | None
    container_name: str
    client: OpenAI | None = None
    _model_id: str | None = None

    @property
    def api_url(self) -> str:
        return self.endpoint.api_url

    def get_client(self) -> OpenAI:
        if self.client is None:
            api_key = getattr(self.endpoint, "api_key", "") or "not-needed"
            self.client = OpenAI(
                base_url=self.api_url,
                api_key=api_key,
            )
        return self.client

    def resolve_model_id(self) -> str:
        """Frage die echte Modell-ID vom laufenden vLLM-Endpoint ab."""
        if self._model_id is None:
            try:
                models = self.get_client().models.list()
                self._model_id = models.data[0].id
                logger.info("Modell-ID vom Endpoint: %s", self._model_id)
            except Exception as e:
                logger.warning("Konnte Modell-ID nicht abfragen: %s", e)
                self._model_id = f"/hf_models/{self.model.profile}" if self.model else ""
        return self._model_id


class VllmController:
    """Steuert vLLM-Instanzen auf Remote-DGX-Spark-Maschinen."""

    def __init__(self, ssh_key_path: str | None = None, ssh_user: str = "mvdb"):
        self.ssh_key_path = ssh_key_path
        self.ssh_user = ssh_user
        self._connections: dict[str, paramiko.SSHClient] = {}

    def _get_ssh(self, host: str) -> paramiko.SSHClient:
        """SSH-Verbindung zu einem Host herstellen oder wiederverwenden."""
        if host not in self._connections or not self._connections[host].get_transport():
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            connect_kwargs: dict = {"hostname": host, "username": self.ssh_user}
            if self.ssh_key_path:
                connect_kwargs["key_filename"] = self.ssh_key_path
            client.connect(**connect_kwargs)
            self._connections[host] = client
            logger.info("SSH-Verbindung zu %s hergestellt", host)
        return self._connections[host]

    def _exec(self, host: str, cmd: str, timeout: int = 60) -> tuple[str, str, int]:
        """Befehl via SSH ausführen. Gibt (stdout, stderr, exit_code) zurück."""
        ssh = self._get_ssh(host)
        logger.debug("SSH %s: %s", host, cmd)
        _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        return stdout.read().decode(), stderr.read().decode(), exit_code

    def start_model(
        self,
        endpoint: TargetConfig | JudgeConfig,
        model: ModelConfig,
    ) -> VllmInstance:
        """Starte ein Modell via vllm_spark.sh auf dem Remote-Host.

        Ablauf:
        1. Eventuell laufenden Container stoppen
        2. vllm_spark.sh mit Modellname aufrufen (nutzt Profil aus profiles/)
        3. Auf Readiness warten (Health-Endpoint)
        """
        container_name = f"vllm-{model.profile.replace('/', '--')}"
        host = endpoint.host

        logger.info("Starte %s auf %s ...", model.name, host)

        # Alle Container mit diesem Namen oder auf diesem Port entfernen (inkl. gestoppte)
        port = endpoint.port
        self._exec(host, (
            f"docker ps -aq --filter name=^/{container_name}$ | xargs -r docker rm -f 2>/dev/null || true && "
            f"docker ps -aq --filter publish={port} | xargs -r docker rm -f 2>/dev/null || true && "
            f"sleep 2"
        ))

        # vllm_spark.sh im Non-Interactive-Modus starten.
        # docker run -d ist bereits im Script → Script endet nach dem Start selbst.
        spark_path = endpoint.vllm_spark_path
        hf_env = f"HF_MODELS_DIR={endpoint.hf_models_dir} " if endpoint.hf_models_dir else ""
        start_cmd = (
            f"cd {spark_path} && "
            f"{hf_env}"
            f"CONTAINER_NAME={container_name} "
            f"HOST_PORT={endpoint.port} "
            f"bash runner/vllm_spark.sh --model {model.profile} --skip-pull"
        )
        stdout, stderr, exit_code = self._exec(host, start_cmd, timeout=120)

        if exit_code != 0:
            logger.error("Start fehlgeschlagen: %s\n%s", stdout, stderr)
            raise RuntimeError(
                f"vllm_spark.sh für {model.name} auf {host} fehlgeschlagen "
                f"(exit={exit_code}): {stderr[:500]}"
            )

        # Auf Readiness warten
        instance = VllmInstance(
            endpoint=endpoint,
            model=model,
            container_name=container_name,
        )
        try:
            self._wait_for_ready(instance, timeout=endpoint.startup_timeout)
        except TimeoutError:
            logger.warning("Startup-Timeout für %s — räume Container auf ...", model.name)
            self._exec(host, f"docker rm -f {container_name} 2>/dev/null || true")
            raise

        logger.info("✓ %s bereit auf %s", model.name, endpoint.base_url)
        return instance

    def _wait_for_ready(self, instance: VllmInstance, timeout: int = 600) -> None:
        """Warte bis der vLLM Health-Endpoint antwortet."""
        import urllib.request
        import urllib.error

        health_url = f"{instance.endpoint.base_url}/health"
        start = time.monotonic()
        last_log = 0.0

        while time.monotonic() - start < timeout:
            try:
                req = urllib.request.Request(health_url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass

            elapsed = time.monotonic() - start
            if elapsed - last_log >= 30:
                logger.info(
                    "Warte auf %s ... (%.0fs / %ds)",
                    instance.model.name if instance.model else "Judge",
                    elapsed,
                    timeout,
                )
                last_log = elapsed
            time.sleep(5)

        raise TimeoutError(
            f"{instance.model.name if instance.model else 'Judge'} nicht bereit "
            f"nach {timeout}s auf {instance.endpoint.base_url}"
        )

    def stop_model(self, instance: VllmInstance) -> None:
        """Stoppe Container und gib GPU-Speicher frei."""
        host = instance.endpoint.host
        logger.info("Stoppe %s auf %s ...", instance.container_name, host)
        self._exec(host, f"docker rm -f {instance.container_name}")
        logger.info("✓ %s gestoppt", instance.container_name)

    def ensure_judge_running(self, judge_config: JudgeConfig) -> VllmInstance:
        """Stelle sicher, dass der Judge läuft. Starte bei Bedarf.

        Externer Judge (api_key gesetzt): kein SSH-Start, direkt verbinden.
        """
        instance = VllmInstance(
            endpoint=judge_config,
            model=ModelConfig(
                name="Judge",
                profile=judge_config.model,
                machine="judge",
            ),
            container_name="vllm-judge",
        )

        if judge_config.api_key:
            # Externer Judge — kein SSH-Start, nur Erreichbarkeit prüfen
            logger.info("Externer Judge: %s (Modell: %s)", judge_config.base_url, judge_config.model)
            return instance

        try:
            self._wait_for_ready(instance, timeout=10)
            logger.info("Judge bereits aktiv auf %s", judge_config.base_url)
            return instance
        except TimeoutError:
            pass

        # Lokalen Judge starten
        judge_pattern = judge_config.profile or judge_config.model
        return self.start_model(
            judge_config,
            ModelConfig(
                name="Judge",
                profile=judge_pattern,
                machine="judge",
            ),
        )

    def close(self) -> None:
        """Alle SSH-Verbindungen schließen."""
        for host, client in self._connections.items():
            try:
                client.close()
                logger.debug("SSH-Verbindung zu %s geschlossen", host)
            except Exception:
                pass
        self._connections.clear()
