# southbyte-vllm

Two things for LLMs on an **NVIDIA DGX Spark**: a runner that serves any model
from the local store with parameters tuned for the GB10, and an evaluation
framework that decides whether a model is fit to use.

**→ [The evaluations](https://mvdb.github.io/southbyte-vllm/)** — 21 locally
served models, a 27-model SaaS reference cohort, and the guardrail models, each
with a per-model detail page.

> **Proof of concept, not a product.** No guaranteed availability, fitness or
> output quality, no support, no roadmap.

## What it does

**`runner/` serves a model.** Point it at `~/hf_models/`, pick a model, get a
vLLM server on port 8000. Every model directory carries a `vllm_profile.conf`
with the parameters that work for it on this hardware — context length, KV-cache
budget, quantization backend, attention backend, parsers, and for a few models a
custom Docker image with sm_120 kernels. Profiles are generated automatically and
hand-corrected where measurement disagreed with the heuristic.

**`testplan/` decides whether to use it.** Eight playbooks over 76 German test
cases: answer quality, German language quality, demographic bias (paired testing
with a Chi² significance test), security (prompt injection, PII leakage,
jailbreak), code generation with SAST, throughput and latency, hardware scaling,
and guardrail models. A second Spark runs a static judge so the model under test
never grades itself.

Some models are disqualified outright — **K.O. criteria**: hallucination rate
above 5 %, any PII leakage, critical SAST findings, statistically significant
bias, or a successful prompt injection.

Published so far, with everything but the security playbook:

| Cohort | |
|---|---|
| Local, on the GB10 | 21 models — 18 valid, 1 degraded, 2 not applicable. Leader: Muse-Glimmer-30B-NVFP4 at 86 % |
| SaaS reference | 27 frontier models through one LiteLLM endpoint, same test set. Leader: Claude-Sonnet-5 at 91 % |
| Guardrails | Scored against labelled data, no judge — the label is the truth |

The roster is published **complete**, including the models that failed or never
ran. A comparison that only shows the winners is not a comparison.

Two smaller pieces live here too: `openwebui/` (Open-WebUI integrations,
including STT with speaker diarization) and `stt-webui/` (a single-file
transcription interface that talks to vLLM directly, no backend).

## Getting it running

You need Docker with GPU access, `python3`, and the models in `~/hf_models/`
(populated by [southbyte-sync](https://github.com/MvdB/southbyte-sync)).

```bash
# Serve a model
cd runner
./vllm_spark.sh --gen-profiles          # once: write a profile per local model
./vllm_spark.sh                          # interactive picker
./vllm_spark.sh --model qwen3.5-9b --tail

# Evaluate models
cd ../testplan && pip install -e .
python orchestrator.py --dry-run         # show the plan, run nothing
python orchestrator.py                   # all active models, all playbooks
python orchestrator.py --models "Ministral3-14B" --playbooks 01_quality
python orchestrator.py --endpoint http://localhost:8000   # against a running server
python generate_demo_report.py           # a report from simulated data, to see the shape
```

Nothing is downloaded at serve time — the model store is mounted read-only.

The profile system, all `PROFILE_*` fields, environment variables and the custom
images: [`docs/runner.md`](docs/runner.md). The evaluation framework in full:
[`testplan/README.md`](testplan/README.md).

## What to watch out for

**The profile is the whole trick.** A model that will not start is almost always
a profile problem, not a vLLM problem. Curated, hand-validated profiles for every
tested model are in
[southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) under
`vllm/profiles/` — copy the subdirectory to `~/hf_models/<model>/` and skip
auto-generation. That repo, not this README, is the authoritative list of what
has been made to work.

**Some models need a custom image.** The stock `vllm/vllm-openai` releases lack
sm_120 kernels for a few quantizations. `PROFILE_DOCKER_IMAGE` points at a locally
built tag; the Dockerfiles are in southbyte-spark-profiles under `vllm/custom/`
and **must be built before first use**.

**The evaluation wants two Sparks.** The judge model stays up on one machine
across the whole run, the targets rotate on the other. One machine works for
single playbooks against an already-running endpoint, but not for a full run.

**Reports never leave the machine.** `testplan/reports/` and `reports-archive/`
are gitignored. What is published is curated: per-playbook pass rates only, model
paths stripped to bare names, and the **`04_security` playbook and all raw
per-case transcripts are never emitted**. Markdown is the primary report format —
diffable, and archivable as approval documentation.

**Per-model reports are written as each model finishes**, so an aborted or
timed-out run still leaves usable results behind.

## Licence

MIT — see [LICENSE](LICENSE).

Model licences travel with the models, not with this repository. The published
comparison names the licence per model; check it before using output for anything
beyond experimentation.

## Where this is going

The runner and the evaluation both work, and the results are published. What is
open is named rather than planned:

- **The test set is 76 cases across 7 categories.** That is enough to disqualify
  a model and not enough to rank two good ones apart. Treat close scores as a
  tie.
- **`04_security` stays unpublished.** The findings are real and the prompts that
  produce them are not something to hand out.

Issues and pull requests are welcome; nobody is on call for them.

## Going deeper

| | |
|---|---|
| [`docs/runner.md`](docs/runner.md) | How a model gets served: the profile system, every `PROFILE_*` field, environment variables, custom images for sm_120 |
| [`testplan/README.md`](testplan/README.md) | The evaluation framework: playbooks, evaluators, judge setup, configuration, reports |
| [`testplan/guards/README.md`](testplan/guards/README.md) | Guardrail models — scored against labelled data, no judge involved |
| [`runner/README.md`](runner/README.md) | Benchmark results per model. Machine-written by `test_models.py`, so do not edit it by hand |

## Part of the southbyte family

- [southbyte-core](https://github.com/MvdB/southbyte-core) — shared index
- [southbyte-sync](https://github.com/MvdB/southbyte-sync) — HuggingFace mirror → local model store
- [southbyte-tts](https://github.com/MvdB/southbyte-tts) — TTS/STT serving + German evaluation
- [southbyte-image](https://github.com/MvdB/southbyte-image) — text-to-image serving + evaluation
- [southbyte-music](https://github.com/MvdB/southbyte-music) — text-to-music serving + web interface
- [southbyte-results](https://github.com/MvdB/southbyte-results) — cross-modality results site
- [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) — GB10 profiles, kernels, benchmarks
- **southbyte-vllm** — vLLM runner + LLM testplan *(this repository)*

---

Built by [southbyte](https://southbyte.de).
