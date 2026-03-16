#!/usr/bin/env python3
"""
HuggingFace 'LocalCache' Collection Sync
-----------------------------------------
Lädt alle Modelle aus der HF-Collection "LocalCache" herunter und prüft
auf Aktualisierungen. Bereits heruntergeladene, aktuelle Modelle werden
übersprungen.

Verzeichnisstruktur:
    ./mistralai--Mistral-7B-v0.1/
    ./meta-llama--Llama-3-8B/
    ...

Konfiguration:
    .env Datei im selben Verzeichnis mit HF_TOKEN=hf_...
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

COLLECTION_NAME = os.environ.get("HF_COLLECTION", "LocalCache")  # überschreibbar per .env

# Dateimuster, die NICHT heruntergeladen werden (None = alles herunterladen)
# Entkommentieren um z.B. TF/Flax-Gewichte zu überspringen:
# IGNORE_PATTERNS = ["*.msgpack", "flax_model*", "tf_model*", "rust_model*"]
IGNORE_PATTERNS: list[str] | None = None

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
ENV_FILE    = SCRIPT_DIR / ".env"
STATE_FILE  = SCRIPT_DIR / ".sync_state.json"
LOG_FILE    = SCRIPT_DIR / "hf_sync.log"

# .env früh laden, damit HF_COLLECTION bereits beim Modulstart verfügbar ist
load_dotenv(ENV_FILE, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def load_token() -> str:
    """Liest HF_TOKEN aus .env-Datei."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        log.error(f"HF_TOKEN nicht gefunden. Bitte '{ENV_FILE}' anlegen:")
        log.error("  HF_TOKEN=hf_dein_token_hier")
        sys.exit(1)
    return token


def load_state() -> dict:
    """Liest den lokalen Status (welcher SHA wurde zuletzt heruntergeladen)."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Beschädigte State-Datei – wird zurückgesetzt.")
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def local_dir_for(model_id: str) -> Path:
    """'mistralai/Mistral-7B-v0.1' → './mistralai--Mistral-7B-v0.1'"""
    return SCRIPT_DIR / model_id.replace("/", "--")


def find_collection(api: HfApi, username: str, token: str):
    """Sucht die 'LocalCache'-Collection des Nutzers und lädt sie vollständig."""
    collections = list(api.list_collections(owner=username, token=token))
    if not collections:
        log.error(f"Keine Collections für '{username}' gefunden.")
        sys.exit(1)

    # Exakter Treffer zuerst
    match = None
    for col in collections:
        if col.title.strip().lower() == COLLECTION_NAME.lower():
            match = col
            break

    # Teilübereinstimmung
    if match is None:
        matches = [c for c in collections if COLLECTION_NAME.lower() in c.title.strip().lower()]
        if matches:
            log.warning(
                f"Kein exakter Treffer für '{COLLECTION_NAME}', "
                f"verwende '{matches[0].title}'."
            )
            match = matches[0]

    if match is None:
        available = [f"  • {c.title}" for c in collections]
        log.error(f"Collection '{COLLECTION_NAME}' nicht gefunden.")
        log.error("Verfügbare Collections:\n" + "\n".join(available))
        sys.exit(1)

    # Vollständige Collection laden (list_collections kürzt Items ab)
    return api.get_collection(match.slug, token=token)


def get_remote_sha(api: HfApi, model_id: str, token: str) -> str | None:
    """Holt den aktuellen Commit-SHA des Modells von HuggingFace."""
    try:
        info = api.model_info(model_id, token=token)
        return info.sha
    except RepositoryNotFoundError:
        log.error(f"[{model_id}] Repository nicht gefunden (privat oder gelöscht?).")
        return None
    except GatedRepoError:
        log.error(f"[{model_id}] Zugriff verweigert (gated model – Lizenz akzeptieren?).")
        return None
    except Exception as exc:
        log.warning(f"[{model_id}] SHA konnte nicht abgerufen werden: {exc}")
        return None


def sync_model(
    model_id: str,
    token: str,
    remote_sha: str | None,
    state: dict,
) -> str:
    """
    Lädt ein Modell herunter oder überspringt es wenn aktuell.
    Gibt zurück: 'skipped' | 'downloaded' | 'updated' | 'failed'
    """
    local_dir   = local_dir_for(model_id)
    stored      = state.get(model_id, {})
    stored_sha  = stored.get("sha")
    is_present  = local_dir.exists() and any(local_dir.iterdir())

    # Aktuell prüfen
    if is_present and stored_sha and remote_sha and stored_sha == remote_sha:
        log.info(f"[OK]       {model_id}  (sha {remote_sha[:8]}…)")
        return "skipped"

    # Status-Ausgabe
    if not is_present:
        log.info(f"[DOWNLOAD] {model_id}")
    else:
        old = stored_sha[:8] + "…" if stored_sha else "?"
        new = remote_sha[:8] + "…" if remote_sha else "?"
        log.info(f"[UPDATE]   {model_id}  ({old} → {new})")

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        kwargs: dict = dict(
            repo_id=model_id,
            local_dir=str(local_dir),
            token=token,
        )
        if IGNORE_PATTERNS:
            kwargs["ignore_patterns"] = IGNORE_PATTERNS

        snapshot_download(**kwargs)

    except GatedRepoError:
        log.error(f"[FAIL]     {model_id}  → gated, bitte Lizenz auf HF akzeptieren.")
        return "failed"
    except RepositoryNotFoundError:
        log.error(f"[FAIL]     {model_id}  → nicht gefunden.")
        return "failed"
    except KeyboardInterrupt:
        log.warning("Abgebrochen (Strg+C). Bisher gespeicherte Fortschritte bleiben erhalten.")
        save_state(state)
        sys.exit(0)
    except Exception as exc:
        log.error(f"[FAIL]     {model_id}  → {exc}")
        return "failed"

    state[model_id] = {"sha": remote_sha, "local_dir": str(local_dir)}
    save_state(state)

    was_update = is_present
    return "updated" if was_update else "downloaded"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info(f"HF LocalCache Sync  —  Collection: '{COLLECTION_NAME}'")
    log.info("=" * 60)

    token = load_token()
    api   = HfApi()
    state = load_state()

    # Nutzer identifizieren
    try:
        user_info = api.whoami(token=token)
        username  = user_info["name"]
    except Exception as exc:
        log.error(f"Authentifizierung fehlgeschlagen: {exc}")
        sys.exit(1)
    log.info(f"Angemeldet als: {username}")

    # Collection laden
    collection = find_collection(api, username, token)
    model_items = [item for item in collection.items if item.item_type == "model"]
    skipped_types = [item for item in collection.items if item.item_type != "model"]

    log.info(f"Collection '{collection.title}': {len(model_items)} Modelle")
    if skipped_types:
        log.info(
            f"Übersprungen (kein Modell): "
            + ", ".join(f"{i.item_id} ({i.item_type})" for i in skipped_types)
        )

    # Modelle synchronisieren
    counts = {"skipped": 0, "downloaded": 0, "updated": 0, "failed": 0}

    for item in model_items:
        model_id   = item.item_id
        remote_sha = get_remote_sha(api, model_id, token)
        result     = sync_model(model_id, token, remote_sha, state)
        counts[result] += 1

    # Zusammenfassung
    log.info("")
    log.info("=" * 60)
    log.info("Zusammenfassung:")
    log.info(f"  Aktuell (übersprungen): {counts['skipped']}")
    log.info(f"  Neu heruntergeladen:    {counts['downloaded']}")
    log.info(f"  Aktualisiert:           {counts['updated']}")
    log.info(f"  Fehlgeschlagen:         {counts['failed']}")
    log.info("=" * 60)

    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
