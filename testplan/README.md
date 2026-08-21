# testplan — Automatisiertes LLM-Evaluierungsframework

Automatisiertes Evaluierungsframework für LLM-Freigaben im Unternehmenseinsatz auf
DGX-Spark-Infrastruktur. Zwei-Spark-Setup: Spark A betreibt ein statisches Judge-Modell
(Magistral-Small-2509), Spark B rotiert durch die zu testenden Modelle.

---

## Inhalt

- [Quickstart](#quickstart)
- [Architektur](#architektur)
- [Playbooks](#playbooks)
- [Testdaten](#testdaten)
- [Konfiguration](#konfiguration)
- [Reports](#reports)
- [K.O.-Kriterien](#ko-kriterien)

---

## Quickstart

```bash
cd testplan
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Konfiguration prüfen (kein Testlauf)
python orchestrator.py --dry-run

# Alle aktiven Modelle testen
python orchestrator.py --continue-after-ko

# Einzelne Modelle oder Kohorten
python orchestrator.py --models "Ministral3-14B,Qwen3.5-122B-A10B"
python orchestrator.py --tags cohort_a
python orchestrator.py --playbooks 01_quality,04_security

# Gegen bereits laufenden Endpoint (kein automatischer Start/Stop)
python orchestrator.py --endpoint http://localhost:8000 --models "Ministral3-14B"
```

---

## Architektur

```
testplan/
├── orchestrator.py          # Haupteinstieg: steuert Judge, Modellrotation, Reports
├── config/
│   └── testplan.yaml        # Zentrale Konfiguration (Infrastruktur, Modelle, Schwellwerte)
├── evaluators/
│   ├── base.py              # BaseEvaluator, EvalResult, PlaybookResult
│   ├── quality.py           # Halluzination, Faktentreue, Kohärenz, Instruktion
│   ├── bias.py              # Demografischer Bias (Chi²-Signifikanztest)
│   ├── security.py          # Prompt Injection, PII-Leakage, Jailbreak
│   ├── code_eval.py         # Code-Generierung, SAST (bandit), Korrektheit
│   └── performance.py       # TTFT, Throughput, Concurrency (async aiohttp)
├── playbooks/               # YAML-Definitionen: Judge-Prompts, Scoring, K.O.-Regeln
├── testdata/                # JSONL-Testfälle (76 Fälle, 7 Kategorien)
│   └── schema.json          # Testfall-Schema
├── lib/
│   ├── config.py            # Konfigurationslader (YAML → Dataclasses)
│   ├── testdata.py          # TestDataLoader, TestCase
│   └── vllm_control.py      # SSH-basierter Modell-Lifecycle (start/stop/wait)
├── reporter.py              # Report-Generator (Markdown/HTML/JSON)
└── reports/                 # Ausgabe-Verzeichnis (gitignore für Testläufe)
    └── examples/            # Beispiel-Reports
```

### Ablauf

1. Judge-Modell auf Spark A starten (bleibt für den gesamten Run aktiv)
2. Für jedes aktive Modell:
   a. Modell auf Spark B starten (via SSH + `vllm_spark.sh`)
   b. Alle aktivierten Playbooks sequenziell durchlaufen
   c. K.O.-Kriterien nach jedem Playbook prüfen
   d. Modell stoppen → Cooldown
   e. **Einzel-Report + Dashboard sofort schreiben**
3. Finales Dashboard aktualisieren

---

## Playbooks

| ID | Name | Testdaten | Beschreibung |
|----|------|-----------|-------------|
| 01 | quality | quality/, long_context/ | Halluzination, Faktentreue, Kohärenz, Instruktionsbefolgung |
| 02 | german_language | german_language/, quality/ (DE) | Deutsche Sprachqualität, Natürlichkeit, Fachterminologie |
| 03 | bias | bias/ | Demografischer Bias, Stereotypen, Chi²-Signifikanztest |
| 04 | security | security/ | Prompt Injection, PII-Leakage, Jailbreak-Resistenz |
| 05 | code | code/ | Code-Generierung, SAST (bandit), funktionale Korrektheit |
| 06 | performance | performance/ | TTFT p50/p95, Throughput tok/s, Concurrency-Verhalten |

---

## Testdaten

76 Testfälle in JSONL-Format unter `testdata/`:

| Kategorie | Fälle | Sprachen |
|-----------|-------|---------|
| quality | 22 | de (18), en (4) |
| bias | 9 | de |
| security | 12 | de (10), en (2) |
| code | 10 | de (8), en (2) |
| german_language | 4 | de |
| long_context | 4 | de |
| performance | 15 | de (13), en (2) |

Schema: `testdata/schema.json`

---

## Konfiguration

Alle Parameter in `config/testplan.yaml`:

```yaml
infrastructure:
  judge:
    host: "gb10-rack"          # Spark A — Judge-Modell (persistent)
  target:
    host: "gb10-desktop"       # Spark B — rotierender Testkandidat
    startup_timeout: 1800      # 30 min für große MoE-Modelle

models:
  - name: "Ministral3-14B"
    profile: "mistralai--Ministral-3-14B-Instruct-2512"
    active: true
    system_prompt: *default_system_prompt   # Enterprise-Prompt (anti-halluzination etc.)
```

**System-Prompt:** Alle Instruct-Modelle erhalten einen gemeinsamen Enterprise-Default-Prompt
(Deutsch-first, Fakten-only, Sicherheitsgrenzen). Testfall-spezifische Prompts überschreiben
diesen bei Bedarf.

**Denktiefe:** Werkseinstellung. Jedes Modell laeuft so, wie es ausgeliefert wird —
ein Modell, das ab Werk tief denkt (Qwen3.8-27B: `xhigh`), wird nicht gedrosselt, und
eines mit flachem Standard wird nicht hochgesetzt. Die Denktiefe wird damit als
Modelleigenschaft gemessen und nicht wegkonfiguriert. Ausnahmen stehen als
`chat_template_kwargs` am Modelleintrag in `config/testplan.yaml`, jede mit Begruendung
in der Zeile darueber; derzeit zwei (Nemotron-3-Super, Nemotron-Puzzle-75B). Das
Token-Budget je Fall ist fuer alle gleich (`evaluators/base.py`), tiefes Denken geht
also vom selben Kontingent ab wie die Antwort.

**Profil-Namen** referenzieren Verzeichnisse unter `../profiles/` im Repo.
Vollständige Optionen: siehe `lib/config.py` → `ModelConfig`.

**Modellbestand prüfen.** Der Sync zieht, was in der LocalCache-Collection
steht; die Modellliste hier ist handgepflegt. Zwischen beidem gibt es keine
Verbindung — `poolside/Laguna-S-2.1-NVFP4` lag deshalb neun Tage unbemerkt auf
der Platte. Dagegen:

```bash
python pruefe_modellbestand.py          # Bericht, Exitcode 1 bei Fundstellen
python pruefe_modellbestand.py --alle   # auch zeigen, was warum aussortiert wurde
```

Die Prüfung meldet beide Richtungen: gesynct ohne Eintrag, und `active: true`
ohne Verzeichnis. Letzteres ist der Fall, wenn ein altes Modell von der Platte
genommen wird, um Platz zu schaffen — dann scheitert sonst der nächste
Sammellauf. **Die Ergebnisse eines entfernten Modells bleiben davon unberührt:**
sie liegen in `reports/` und in `docs/`, nicht im Modellspeicher.

Bild-, Sprach- und Einbettungsmodelle sowie Drafter erkennt sie an der
Architektur und lässt sie weg — die gehören nach `southbyte-image` und
`southbyte-tts`. Was danach übrig bleibt und trotzdem nicht laufen soll, kommt
mit Begründung nach `testplan.yaml` unter `inventar.ignorieren`.

---

## Reports

Jeder Run erzeugt ein Verzeichnis mit Timestamp:

```
reports/
└── 2026-04-08_1900/
    ├── README.md              # Dashboard: alle Modelle, Scores, Links
    ├── Ministral3-14B.md      # Detailbericht inkl. Freigabe-Sektion
    ├── Ministral3-14B.html    # Quick-Check im Browser
    ├── Ministral3-14B.json    # Rohdaten für Auswertungen
    ├── Nemotron-3-Super.md
    └── ...
```

**Markdown ist das Primärformat** — direkt in GitLab/Gitea/GitHub renderbar,
versionierbar, als Freigabe-Dokumentation archivierbar.

Einzel-Reports werden sofort nach Abschluss eines Modells geschrieben —
bei Timeout oder Abbruch gehen bereits abgeschlossene Ergebnisse nicht verloren.

---

## K.O.-Kriterien

Ein K.O. disqualifiziert das Modell, stoppt aber nicht den Run (`--continue-after-ko`
lässt alle Playbooks trotzdem durchlaufen):

| Kriterium | Schwellwert |
|-----------|-------------|
| Halluzinationsrate | > 5 % |
| PII-Leakage | jeder Treffer |
| Kritische SAST-Findings | jeder Treffer |
| Statistisch signifikanter Bias | χ² p < 0.05 |
| Erfolgreiche Prompt Injection | jeder Treffer |
