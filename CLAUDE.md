# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

Tooling for running vLLM on the **NVIDIA DGX Spark** (GB10 SoC, sm_120, 128 GB unified memory). Two independent subsystems:

1. **`runner/`** – Bash + Python scripts that start a vLLM Docker container for any local model, generating and loading per-model parameter profiles.
2. **`repo-sync/`** – Python script that mirrors a named HuggingFace collection to `~/hf_models/` using commit-SHA-based update detection.

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
docker build -t spark-mistral-small4:v1 -f custom/Dockerfile.mistral-small4 .

# Sync HuggingFace collection to ~/hf_models/
cd repo-sync && source .venv/bin/activate && python hf_sync.py
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

`VLLM_SCRIPT` is read from `~/vllm_spark.sh` (not the repo path) — deploy the script to home if running tests from outside the repo.

### Model directory naming

HuggingFace model IDs use `--` instead of `/`: `mistralai/Mistral-7B-v0.1` → `mistralai--Mistral-7B-v0.1`. The profiler and sync script both follow this convention.

### Per-model custom Docker images

Some models need kernels not in `vllm/vllm-openai` (especially sm_120 NVFP4). Custom Dockerfiles live in `custom/`. The profile's `PROFILE_DOCKER_IMAGE` field points to the locally-built tag. Must be built before first use.

### Curated profiles in `profiles/`

`profiles/<model-dir>/vllm_profile.conf` contains hand-validated settings. Copy to `~/hf_models/<model>/` to bypass auto-generation. The auto-generator's `KNOWN_GOOD` dict contains the same data as Python source.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `HF_MODELS_DIR` | `~/hf_models` | Local model store |
| `IMAGE_REPO` | `vllm/vllm-openai` | Docker image repo |
| `DEFAULT_VLLM_TAG` | `v0.17.1` | vLLM image tag |
| `CONTAINER_NAME` | `vllm-server` | Docker container name |
| `HOST_PORT` | `8000` | Host port for API |
| `HF_TOKEN` | — | HuggingFace token (or `HUGGING_FACE_HUB_TOKEN`) |
| `VLLM_EXTRA_ARGS` | — | Extra flags appended to `vllm serve` |

## repo-sync setup

```bash
cd repo-sync
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add HF_TOKEN
```

Set `HF_COLLECTION` in `.env` to match the collection name (default: `LocalCache`).
