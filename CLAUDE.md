# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

Tooling for running vLLM on the **NVIDIA DGX Spark** (GB10 SoC, sm_120, 128 GB unified memory). Two subsystems:

1. **`runner/`** – Bash + Python scripts that start a vLLM Docker container for any local model, generating and loading per-model parameter profiles.
2. **`testplan/`** – Automated LLM evaluation framework. Two-Spark setup: Spark A runs a static judge (Magistral-Small-2509), Spark B rotates target models. 8 playbooks (quality, German, bias, security, code, performance, HSF, guardrails). Generates per-model reports + cross-model dashboard. The guardrails playbook (08) is different: the guard model *is* the classifier scored against labeled data, so there is **no judge** — see `testplan/guards/README.md`.

The HuggingFace collection mirror (`hf-sync`, formerly `repo-sync/` here) is the standalone repo [southbyte-sync](https://github.com/MvdB/southbyte-sync); it populates `~/hf_models/` (which is itself a clone of that repo) and defines the `<owner>--<model-name>` directory naming. Spark-tuned per-model profiles + custom kernel Dockerfiles live in [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles).

## Common commands

```bash
# Start vLLM server (interactive model picker)
cd runner && ./vllm_spark.sh

# Start a specific model directly
./vllm_spark.sh --model qwen3.5-9b --tail

# Generate profiles for all local models (run once after downloading models)
./vllm_spark.sh --gen-profiles

# Regenerate profile for one model
./vllm_spark.sh --model ministral-8b --regen-profile

# Run profile generator directly
python3 runner/vllm_spark_profiler.py ~/hf_models/mistralai--Ministral-3-8B-Instruct-2512 --force

# Run smoke-test + benchmark for all models (writes runner/README.md)
python3 runner/test_models.py

# Run benchmark for a single model
python3 runner/test_models.py qwen3.5-9b

# Build custom Docker image for Mistral-Small-4
(cd ~/southbyte/southbyte-spark-profiles/vllm/custom && docker build -t spark-mistral-small4:v1 -f Dockerfile.mistral-small4 .)

# --- testplan ---
# Dry run (show config, don't execute)
cd testplan && python orchestrator.py --dry-run

# Full test run (all active models)
python orchestrator.py

# Test specific models or cohorts
python orchestrator.py --models "Magistral-Small-2509,Ministral3-14B"
python orchestrator.py --tags cohort_a
python orchestrator.py --playbooks 01_quality,04_security

# Test against already-running endpoint
python orchestrator.py --endpoint http://localhost:8000

# Generate demo report with simulated data
python generate_demo_report.py
```

## Architecture

### Profile system (central concept)

Each model directory under `~/hf_models/<owner>--<model-name>/` gets a `vllm_profile.conf` — a bash-sourceable file with `PROFILE_*` variables. `vllm_spark.sh` sources this file and translates each variable to vLLM CLI flags, guarded by `vllm_supports()` which checks the target image's `--help=all` output.

`vllm_spark_profiler.py` generates profiles automatically:
1. Check `KNOWN_GOOD` dict first (empirically validated, takes precedence)
2. Fall back to heuristics: read `config.json`, detect architecture/quantization, estimate model size from directory name, calculate KV-cache budget against the 128 GB memory target (85% util, 5 GB overhead)
3. `ARCH_HINTS` dict applies parser/tool settings per architecture class

Key profile fields: `PROFILE_DOCKER_IMAGE` (per-model custom image override), `PROFILE_BASH_WRAPPER` (wraps `vllm serve` in `bash -lc` for non-standard entrypoints), `PROFILE_IPC_HOST` (`--ipc=host` — mutually exclusive with `--shm-size`), `PROFILE_DOCKER_ENV` (space-separated `KEY=VALUE` injected via `--env`).

### `vllm_spark.sh` execution stages

1. **Prereqs** – verify docker, python3, `HF_MODELS_DIR`, profiler script
2. **Model pick** – scan `~/hf_models/` for dirs with `config.json` or `params.json`, filter by `PROFILE_VLLM_COMPATIBLE`, interactive menu or `--model` pattern
3. **Pull** – `docker pull` (skip with `--skip-pull`)
4. **Verify** – check weight files exist
5. **Run** – apply profile, build docker run command, start detached container

### `test_models.py` flow

Iterates compatible models, calls `vllm_spark.sh --skip-pull` per model, polls `/v1/models`, runs warmup + correctness queries + streaming benchmark (TTFT, tok/s), then writes `runner/README.md` and git-pushes after each model.

`VLLM_SCRIPT` defaults to the repo-local `runner/vllm_spark.sh`; override with the `VLLM_SCRIPT` env var to point at a different copy.

### Model directory naming

HuggingFace model IDs use `--` instead of `/`: `mistralai/Mistral-7B-v0.1` → `mistralai--Mistral-7B-v0.1`. The profiler and southbyte-sync both follow this convention.

### Per-model custom Docker images

Some models need kernels not in `vllm/vllm-openai` (especially sm_120 NVFP4). Custom Dockerfiles live in [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) under `vllm/custom/`. The profile's `PROFILE_DOCKER_IMAGE` field points to the locally-built tag. Must be built before first use.

### Curated profiles (southbyte-spark-profiles)

`vllm/profiles/<model-dir>/vllm_profile.conf` in [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) contains hand-validated settings. Copy to `~/hf_models/<model>/` to bypass auto-generation. The auto-generator's `KNOWN_GOOD` dict contains the same data as Python source.

### `testplan/` architecture

Central config: `testplan/config/testplan.yaml` — defines infrastructure (judge/target hosts), all model definitions with `profile` names matching `vllm/profiles/` directories in southbyte-spark-profiles, K.O. thresholds, quality targets, and playbook definitions.

**Orchestrator flow** (`orchestrator.py`):
1. Start judge model on Spark A (persistent across all targets)
2. For each active model: start on Spark B → run enabled playbooks → check K.O. criteria → stop model → cooldown
3. Generate consolidated reports (JSON/HTML/CSV) + cross-model dashboard

**Evaluators** (`evaluators/`): Each evaluator queries the target model, then uses the judge model (LLM-as-Judge pattern) to score the response. Exception: `performance.py` uses async streaming (aiohttp) for TTFT measurement, no judge needed.

**Playbooks** (`playbooks/*.yaml`): Contain German judge system prompts, Jinja2 user templates, scoring rules, K.O. criteria, and testdata subcategory mappings.

**Testdata** (`testdata/*.jsonl`): JSONL format with schema in `testdata/schema.json`. 76 test cases across 7 categories. Expected answers support types: exact, contains, regex, semantic, code_exec, judge.

**Dashboard** (`dashboard.py`): Cross-model comparison HTML with executive summary cards, pass-rate/score matrices, performance comparison, runtime estimation, drill-down per model, and compliance sections. Also provides `estimate_full_runtime()` for predicting test duration.

**Key dependencies**: openai (vLLM API), paramiko (SSH model lifecycle), aiohttp (async streaming), scipy (Chi² bias test), numpy (bootstrap CI), jinja2 (templates), bandit (SAST).

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `HF_MODELS_DIR` | `~/hf_models` | Local model store |
| `IMAGE_REPO` | `vllm/vllm-openai` | Docker image repo |
| `DEFAULT_VLLM_TAG` | `v0.26.0` | vLLM image tag (fallback; curated profiles pin their own image via `PROFILE_DOCKER_IMAGE`) |
| `CONTAINER_NAME` | `vllm-server` | Docker container name |
| `HOST_PORT` | `8000` | Host port for API |
| `HF_TOKEN` | — | HuggingFace token (or `HUGGING_FACE_HUB_TOKEN`) |
| `VLLM_EXTRA_ARGS` | — | Extra flags appended to `vllm serve` |

