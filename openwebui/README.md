# openwebui

Open-WebUI-Integrationen für Modelle, die auf dem DGX Spark via vLLM laufen.

## granite_stt_speaker_id.py — STT mit Sprecher-Diarization

Pipe-Function für `ibm-granite/granite-speech-4.1-2b-plus` (Profil siehe
`profiles/ibm-granite--granite-speech-4.1-2b-plus/`). Erscheint in Open WebUI
als eigenes „Modell": Audio-Datei an die Chatnachricht anhängen, absenden,
sprecher-attribuiertes Transkript zurückbekommen (`**Speaker 1:** … **Speaker 2:** …`).

### Warum eine Function?

vLLM bedient das Modell auf zwei Wegen:

| Weg | Sprecher-Labels |
|---|---|
| `/v1/audio/transcriptions` (Whisper-kompatibel) | ❌ nur reines ASR — `prompt`-Param wird ignoriert |
| `/v1/chat/completions` (base64 `audio_url` + SAA-Prompt) | ✅ |

Für Sprecher-ID braucht es also den Chat-Weg — genau das kapselt diese Pipe.
Für reine Diktat-Transkription (Mikrofon-Button) reicht dagegen Open WebUIs
eingebautes STT-Setting, ganz ohne Function (siehe unten).

### Installation

1. Admin-Panel → **Functions** → **+ Neu** → Inhalt von
   `granite_stt_speaker_id.py` einfügen → Speichern → Function **aktivieren**.
2. Zahnrad an der Function → **Valves**:
   - `VLLM_BASE_URL`: `http://<spark-ip>:8000/v1`
   - `MODEL_ID`: `granite-speech-4.1-2b-plus` (served model name, siehe Profil)
   - `MODE`: `saa` (Sprecher-ID) | `asr` | `timestamps`
3. Im Chat das Modell **„Granite Speech STT — Sprecher-ID"** wählen,
   Audio-Datei anhängen (WAV/MP3/FLAC/OGG/M4A/WebM), Nachricht absenden.

### Lange Aufnahmen

Bei ctx 4096 passt max. ~5 min Audio in einen Request. Längere Dateien teilt
die Function per ffmpeg (im Open-WebUI-Docker-Image enthalten) in
`CHUNK_SECONDS`-Segmente (Default 180 s). Die Segmente laufen **unabhängig**:
`Speaker N` zählt pro Segment neu, Zeitmarken (`*3:00*`) markieren die
Grenzen. IBM-Angaben: bis 9 min SAA, 3,5 min Timestamps (mit Chunking).

**Loop-Schutz:** Greedy-Decoding kippt bei Gesang/Musik in Endlos-Loops
(„la la la …"). Enges Token-Budget + n-Gramm-Stripping kürzen die Loops;
betroffene Zeitbereiche werden unter dem Transkript als Warnung ausgewiesen.

**Valve `PREFIX_CHAINING` (default aus):** Segment-Verkettung via
`prefix_text` (IBM „incremental decoding") stabilisiert theoretisch die
Sprecher-Nummern, löste im Test mit einem bairischen Hörspiel aber
Sprachdrift aus (Transkript kippte ins Niederländische). Nur für klar
artikuliertes Standard-Material aktivieren.

### Plain-STT ohne Function (Mikrofon/Diktat)

Admin-Panel → Settings → **Audio** → Speech-to-Text:

- Engine: `OpenAI`
- API Base URL: `http://<spark-ip>:8000/v1`
- API Key: beliebig (vLLM ohne `--api-key` ignoriert ihn)
- **STT Model: `granite-speech-4.1-2b-plus`** — nicht `whisper-1` stehen
  lassen, sonst antwortet vLLM mit 404 `The model 'whisper-1' does not exist`.

Liefert kleingeschriebenes Transkript ohne Interpunktion und ohne
Sprecher-Labels (Plus-Variante macht generell keine Interpunktion).
