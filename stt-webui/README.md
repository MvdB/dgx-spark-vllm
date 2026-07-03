# stt-webui

Eigenständige Single-File-WebUI für Transkription mit
`ibm-granite/granite-speech-4.1-2b-plus` auf vLLM (DGX Spark) — kein Backend,
keine Dependencies, eine `index.html`. Der Browser spricht direkt mit dem
vLLM-Server (CORS ist bei `vllm serve` per Default offen).

## Features

- **Datei-Upload + Drag & Drop** (WAV/MP3/M4A/OGG/FLAC/WebM — alles, was der
  Browser dekodieren kann)
- **Mikrofon-Aufnahme** direkt im Browser (secure context nötig, s. u.)
- **3 Modi**: Sprecher-ID (SAA, farbige Speaker-Chips), reines ASR,
  Wort-Timestamps
- **Clientseitiges Chunking**: Audio wird im Browser auf 16 kHz mono
  resampled und in Segmente geteilt (Default 180 s, Timestamps max. 210 s) —
  Kontextlimit 4096 wird nie gerissen
- **Loop-Schutz**: Greedy-Decoding kippt bei Gesang/Musik in Endlos-Loops
  („la la la …"). Enges Token-Budget macht das erkennbar
  (`finish_reason: length`), n-Gramm-Stripping (1–8 Wörter) kürzt die Loops,
  betroffene Zeitbereiche werden in der UI ausgewiesen
- **Export**: Kopieren, `.txt`, `.md` (Speaker fett), `.srt`-Untertitel aus dem
  Timestamps-Modus (inkl. `[T:N]`-Rollover-Umrechnung alle 10 s und
  Chunk-Offsets)
- Serveradresse/Modell/Modus werden in `localStorage` gemerkt

## Nutzung

vLLM-Server mit dem Modell starten (Profil siehe
`profiles/ibm-granite--granite-speech-4.1-2b-plus/`):

```bash
cd runner && ./vllm_spark.sh --model granite-speech-4.1-2b-plus
```

Dann eine der drei Varianten:

```bash
# 1) Auf dem Spark hosten (von jedem Gerät im LAN erreichbar)
python3 -m http.server 8081 --directory stt-webui
# → http://<spark-ip>:8081

# 2) Direkt öffnen (Doppelklick / file://) — Server-URL in der UI eintragen

# 3) Mit Mikrofon von einem anderen Rechner: Port-Forward statt LAN-HTTP
ssh -L 8081:localhost:8081 -L 8000:localhost:8000 <spark>
# → http://localhost:8081 (secure context → Mikro erlaubt)
```

### Mikrofon & secure context

Browser geben `getUserMedia` nur in secure contexts frei: `https://`,
`http://localhost` oder `file://`. Über `http://<ip>:8081` von einem anderen
Rechner ist der Mikrofon-Button daher blockiert — Datei-Upload funktioniert
immer. Abhilfe: Variante 2 oder 3 oben, oder einen Reverse-Proxy mit TLS
davorsetzen.

## Grenzen

- **Segmente laufen unabhängig** — `Speaker N` zählt pro Segment neu, die UI
  zeigt Zeitmarken an den Grenzen. Die Segment-Verkettung via `prefix_text`
  (IBM-Doku „incremental decoding") ist als experimentelle Option vorhanden,
  aber default AUS: Im Test mit einem bairischen Hörspiel löste sie
  Sprachdrift aus (Transkript kippte ab Minute 9 ins Niederländische und
  erholte sich nicht mehr). Zuverlässigste Diarization bei Aufnahmen ≤ 3 min
  (ein Segment).
- Gesang/Musik wird nicht sinnvoll transkribiert — Loops werden gekürzt und
  als Warnung ausgewiesen, der Inhalt dieser Passagen fehlt aber.
- IBM-Angaben: bis 9 min SAA, 3,5 min Timestamps (mit Chunking).
- Plus-Variante liefert keine Interpunktion/Großschreibung (by design).
- `--kv-cache-dtype fp8` verschlechtert die Decodierung messbar (Loop-Neigung)
  — das kuratierte Profil erzwingt deshalb volle KV-Präzision.

## Tests

Die reine Logik (SRT-Builder inkl. Rollover/Chunk-Offsets, prefix-Trimming,
WAV-Encoder-Bytelayout, Markdown-Export) ist mit `gjs` (SpiderMonkey)
unit-getestet; die HTTP-Payload-Form ist identisch mit der live validierten
aus `openwebui/granite_stt_speaker_id.py`. Browser-APIs (decodeAudioData,
MediaRecorder) sind Standard und ungemockt getestet nur im echten Browser.
