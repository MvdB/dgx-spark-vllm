"""
title: Granite Speech STT — Sprecher-ID
author: MvdB
version: 0.1.0
license: MIT
description: Sprecher-attribuierte Transkription (SAA) via granite-speech-4.1-2b-plus auf vLLM. Audio-Datei an die Nachricht anhängen und absenden.
"""

# Open-WebUI-Pipe für ibm-granite/granite-speech-4.1-2b-plus auf vLLM (DGX Spark).
#
# Modi (Valve MODE):
#   saa        – Sprecher-Diarization: "[Speaker N]:" vor jedem Sprecherwechsel
#   asr        – reine Transkription (kleingeschrieben, ohne Interpunktion)
#   timestamps – Wort-Timestamps "[T:N]" (Centisekunden mod 1000)
#
# Lange Audios werden per ffmpeg in CHUNK_SECONDS-Segmente geteilt (16 kHz
# mono WAV) und mit prefix_text (Chat-Template-Kwarg) inkrementell dekodiert,
# damit die Sprecher-Nummerierung über Segmente hinweg möglichst stabil
# bleibt. Achtung: Sprecher-Re-Identifikation über Chunk-Grenzen ist
# best-effort — das Modell hört den vorherigen Chunk nicht mehr.
#
# WICHTIG (vLLM-Eigenheit, Stand v0.23.0): model_type granite_speech_plus
# fehlt in vLLMs Placeholder-Map, daher muss "<|audio|>" manuell am Anfang
# des Text-Contents stehen — die Prompts unten tun das bereits.

import asyncio
import base64
import json
import math
import mimetypes
import os
import re
import shutil
import tempfile

import aiohttp
from pydantic import BaseModel, Field

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".webm", ".mp4"}

PROMPTS = {
    "asr": "<|audio|> can you transcribe the speech into a written format?",
    "saa": "<|audio|> Speaker attribution: Transcribe and denote who is speaking "
           "by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.",
    "timestamps": "<|audio|> Timestamps: Transcribe the speech. After each word, add a "
                  "timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]",
}

CTX_LIMIT = 4096          # max_model_len des Servers (Backbone-Limit)
AUDIO_TOK_PER_SEC = 12    # empirisch ~11,7 → aufgerundet
PREFIX_TAIL_CHARS = 1200  # prefix_text-Budget bei langen Aufnahmen
# Enges Output-Budget: echte Sprache ~4,3 tok/s (SAA/ASR), ~12 tok/s mit
# [T:N]-Tags. Loops (Gesang/Musik) laufen so schnell ins "length"-Limit
# und sind sicher als degeneriert erkennbar.
OUT_TOK_PER_SEC = {"saa": 6, "asr": 6, "timestamps": 16}


def _norm_word(w: str) -> str:
    return re.sub(r"[.,!?;:]+$", "", w.lower())


def strip_loops(text: str, min_rep: int = 5) -> str:
    """n-Gramm-Wiederholungs-Läufe (1–4 Wörter, ≥min_rep) auf "…" kürzen.

    Greedy-Decoding kippt bei Gesang/Musik in Endlos-Loops ("la la la …").
    """
    w = text.split()
    out, i = [], 0
    while i < len(w):
        hit = False
        for n in range(8, 0, -1):
            if i + 2 * n > len(w):
                continue
            gram = [_norm_word(x) for x in w[i:i + n]]
            reps = 1
            while [_norm_word(x) for x in w[i + reps * n:i + (reps + 1) * n]] == gram:
                reps += 1
            if reps >= min_rep:
                out.extend(w[i:i + n])
                out.append("…")
                i += reps * n
                hit = True
                break
        if not hit:
            out.append(w[i])
            i += 1
    return " ".join(out)


class Pipe:
    class Valves(BaseModel):
        VLLM_BASE_URL: str = Field(
            default="http://localhost:8000/v1",
            description="Basis-URL des vLLM-Servers (Spark), inkl. /v1",
        )
        MODEL_ID: str = Field(
            default="granite-speech-4.1-2b-plus",
            description="Served model name auf dem vLLM-Server",
        )
        MODE: str = Field(
            default="saa",
            description="Transkriptionsmodus: saa | asr | timestamps",
        )
        CHUNK_SECONDS: int = Field(
            default=180,
            description="Segmentlänge in Sekunden für lange Audios (ffmpeg nötig)",
        )
        TIMEOUT_SECONDS: int = Field(
            default=300,
            description="HTTP-Timeout pro Segment",
        )
        PRETTY_OUTPUT: bool = Field(
            default=True,
            description="Sprecherwechsel als Markdown formatieren (fett + Absätze)",
        )
        PREFIX_CHAINING: bool = Field(
            default=False,
            description="EXPERIMENTELL: Segment-Verkettung via prefix_text. Stabilisiert "
                        "theoretisch die Sprecher-Nummern über Segmente, kann aber "
                        "Sprachdrift auslösen (validiert: bairisches Hörspiel kippte "
                        "damit ins Niederländische). Ohne Verkettung zählt Speaker N "
                        "pro Segment neu.",
        )
        API_KEY: str = Field(
            default="",
            description="Optionaler Bearer-Token für den vLLM-Server",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ── Kern (Open-WebUI-unabhängig, lokal testbar) ────────────────────────

    async def _probe_duration(self, path: str) -> float | None:
        """Audiodauer in Sekunden via ffprobe; None wenn nicht ermittelbar."""
        if not shutil.which("ffprobe"):
            return None
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        try:
            return float(out.decode().strip())
        except ValueError:
            return None

    async def _split_audio(self, path: str, tmpdir: str) -> list[str]:
        """Per ffmpeg in 16-kHz-Mono-WAV-Segmente von CHUNK_SECONDS teilen."""
        pattern = os.path.join(tmpdir, "chunk%04d.wav")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-v", "error", "-i", path,
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-f", "segment", "-segment_time", str(self.valves.CHUNK_SECONDS),
            pattern,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg-Split fehlgeschlagen: {err.decode()[:300]}")
        return sorted(
            os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.startswith("chunk")
        )

    async def _transcribe_chunk(
        self, session: aiohttp.ClientSession, path: str,
        duration: float | None, prefix: str | None,
    ) -> str:
        mime = mimetypes.guess_type(path)[0] or "audio/wav"
        b64 = base64.b64encode(open(path, "rb").read()).decode()

        # Output-Budget gegen das 4096er-Kontextlimit rechnen
        dur = duration or self.valves.CHUNK_SECONDS
        est_prompt = 80 + len(prefix or "") // 3 + math.ceil(dur * AUDIO_TOK_PER_SEC)
        budget = math.ceil(dur * OUT_TOK_PER_SEC[self.valves.MODE]) + 64
        max_tokens = max(160, min(CTX_LIMIT - est_prompt - 16, budget))

        payload = {
            "model": self.valves.MODEL_ID,
            "messages": [{"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": PROMPTS[self.valves.MODE]},
            ]}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if prefix:
            payload["chat_template_kwargs"] = {"prefix_text": prefix}

        headers = {"Content-Type": "application/json"}
        if self.valves.API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.API_KEY}"

        async with session.post(
            f"{self.valves.VLLM_BASE_URL.rstrip('/')}/chat/completions",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.valves.TIMEOUT_SECONDS),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"vLLM HTTP {resp.status}: {(await resp.text())[:300]}")
            data = await resp.json()
        choice = data["choices"][0]
        return choice["message"]["content"].strip(), choice["finish_reason"] == "length"

    @staticmethod
    def _trim_prefix(text: str) -> str:
        """prefix_text auf Budget kürzen, möglichst an einer [Speaker-Grenze."""
        if len(text) <= PREFIX_TAIL_CHARS:
            return text
        tail = text[-PREFIX_TAIL_CHARS:]
        cut = tail.find("[Speaker")
        return tail[cut:] if cut > 0 else tail

    async def transcribe_file(self, path: str, emit=None) -> str:
        """Datei transkribieren; teilt lange Audios in Chunks (mit prefix_text)."""
        if self.valves.MODE not in PROMPTS:
            raise ValueError(f"Ungültiger MODE '{self.valves.MODE}' (saa|asr|timestamps)")

        async def status(msg, done=False):
            if emit:
                await emit({"type": "status",
                            "data": {"description": msg, "done": done}})

        duration = await self._probe_duration(path)
        degenerate_ranges = []
        async with aiohttp.ClientSession() as session:
            if duration is not None and duration > self.valves.CHUNK_SECONDS:
                if not shutil.which("ffmpeg"):
                    raise RuntimeError(
                        f"Audio ist {duration:.0f}s lang (> {self.valves.CHUNK_SECONDS}s), "
                        "aber ffmpeg fehlt für das Chunking."
                    )
                with tempfile.TemporaryDirectory() as tmpdir:
                    chunks = await self._split_audio(path, tmpdir)
                    parts = []
                    clean_tail = ""   # Prefix-Hygiene: nur saubere Chunks verketten
                    for i, chunk in enumerate(chunks):
                        await status(f"Transkribiere Segment {i + 1}/{len(chunks)} …")
                        # Verkettung default AUS: prefix_text kann Sprachdrift
                        # auslösen — Segmente laufen unabhängig, Speaker N zählt
                        # pro Segment neu (Zeitmarken markieren die Grenzen).
                        prefix = (
                            self._trim_prefix(clean_tail) + " "
                            if self.valves.PREFIX_CHAINING and clean_tail else None
                        )
                        raw, capped = await self._transcribe_chunk(
                            session, chunk, min(self.valves.CHUNK_SECONDS, duration),
                            prefix,
                        )
                        cleaned = strip_loops(raw)
                        sec = i * self.valves.CHUNK_SECONDS
                        stamp = f"{sec // 60}:{sec % 60:02d}"
                        # "length"-Abbruch oder massives Loop-Stripping = degeneriert
                        # (typisch: Gesang/Musik) → nicht in den Prefix übernehmen
                        if capped or len(cleaned) < 0.7 * len(raw):
                            degenerate_ranges.append(
                                f"{stamp}–{(sec + self.valves.CHUNK_SECONDS) // 60}:"
                                f"{(sec + self.valves.CHUNK_SECONDS) % 60:02d}"
                            )
                        else:
                            clean_tail = (clean_tail + " " + cleaned).strip()
                        parts.append(f"\n\n---\n*{stamp}*\n\n{cleaned}" if len(chunks) > 1
                                     else cleaned)
                    transcript = "".join(parts).strip()
            else:
                await status("Transkribiere …")
                raw, capped = await self._transcribe_chunk(session, path, duration, None)
                transcript = strip_loops(raw)
                if capped or len(transcript) < 0.7 * len(raw):
                    degenerate_ranges.append("gesamte Aufnahme")

        await status("Transkription abgeschlossen", done=True)
        if degenerate_ranges:
            transcript += (
                "\n\n> ⚠️ Wiederholungs-Loops erkannt und gekürzt "
                "(vermutlich Gesang/Musik): " + ", ".join(degenerate_ranges)
            )
        return transcript

    @staticmethod
    def _pretty(text: str) -> str:
        """"[Speaker N]:"-Tags als Markdown-Absätze mit Fettdruck formatieren."""
        import re
        out = re.sub(r"\s*\[Speaker (\d+)\]:\s*", r"\n\n**Speaker \1:** ", text).strip()
        return out or text

    # ── Open-WebUI-Integration ──────────────────────────────────────────────

    @staticmethod
    def _resolve_file(item) -> tuple[str, str] | None:
        """(lokaler Pfad, Dateiname) für einen Upload-Eintrag ermitteln."""
        f = item.get("file", item) if isinstance(item, dict) else {}
        file_id, filename = f.get("id"), f.get("filename") or f.get("name") or ""

        candidates = []
        if f.get("path"):
            candidates.append(f["path"])
        if file_id:
            try:
                from open_webui.models.files import Files
                rec = Files.get_file_by_id(file_id)
                if rec and rec.path:
                    filename = filename or rec.filename
                    try:
                        from open_webui.storage.provider import Storage
                        candidates.append(Storage.get_file(rec.path))
                    except Exception:
                        candidates.append(rec.path)
            except Exception:
                pass
            try:
                from open_webui.config import UPLOAD_DIR
                candidates.append(os.path.join(UPLOAD_DIR, f"{file_id}_{filename}"))
            except Exception:
                pass

        for path in candidates:
            if path and os.path.isfile(path):
                return path, filename
        return None

    @classmethod
    def _find_audio_file(cls, body: dict, __files__) -> tuple[str, str] | None:
        items = (
            (__files__ or [])
            + (body.get("files") or [])
            + ((body.get("metadata") or {}).get("files") or [])
        )
        for item in reversed(items):  # neueste Datei zuerst
            f = item.get("file", item) if isinstance(item, dict) else {}
            name = (f.get("filename") or f.get("name") or "").lower()
            ctype = ((f.get("meta") or {}).get("content_type") or "").lower()
            is_audio = (
                ctype.startswith("audio/")
                or ctype in ("video/webm", "video/mp4")
                or os.path.splitext(name)[1] in AUDIO_EXTS
            )
            if is_audio:
                resolved = cls._resolve_file(item)
                if resolved:
                    return resolved
        return None

    async def pipe(self, body: dict, __files__=None, __event_emitter__=None):
        found = self._find_audio_file(body, __files__)
        if not found:
            return (
                "Bitte eine Audio-Datei anhängen (WAV/MP3/FLAC/OGG/M4A/WebM) "
                "und die Nachricht erneut absenden. Modus: "
                f"`{self.valves.MODE}` (änderbar in den Valves)."
            )
        path, filename = found
        try:
            transcript = await self.transcribe_file(path, emit=__event_emitter__)
        except Exception as e:
            return f"❌ Transkription von `{filename}` fehlgeschlagen: {e}"

        if self.valves.PRETTY_OUTPUT and self.valves.MODE == "saa":
            transcript = self._pretty(transcript)
        return transcript
