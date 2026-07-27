# dgx-spark-vllm

Tooling for running [vLLM](https://github.com/vllm-project/vllm) on the
**NVIDIA DGX Spark** (GB10 SoC, 128 GB unified memory), plus an automated
LLM evaluation framework.

Shared infrastructure (HuggingFace collection mirror, common tooling) lives in
[dgx-spark-core](https://github.com/MvdB/dgx-spark-core).

## Repository structure

```
dgx-spark-vllm/
├── runner/                    # vLLM container runner
│   ├── vllm_spark.sh          #   interactive model picker + server start
│   └── vllm_spark_profiler.py #   auto-generates per-model vLLM profiles
├── profiles/                  # Curated vllm_profile.conf for known-good models
│   ├── mistralai--Mistral-Small-4-119B-2603-NVFP4/
│   ├── nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/
│   └── ...                    #   one subdirectory per model
├── testplan/                  # Automated LLM evaluation framework
│   ├── orchestrator.py        #   end-to-end test runner
│   ├── dashboard.py           #   cross-model comparison dashboard
│   ├── config/testplan.yaml   #   central config (models, thresholds, playbooks)
│   ├── evaluators/            #   quality, bias, security, code, performance
│   ├── playbooks/             #   7 test playbooks with judge prompts
│   └── testdata/              #   JSONL test cases (76 across 7 categories)
└── custom/                    # Custom Docker images for models needing special kernels
    └── Dockerfile.mistral-small4  # avarok/dgx-vllm-nvfp4-kernel base + mistral_common
```

---

## runner – vLLM on DGX Spark

### Requirements

- Docker with GPU support (`--gpus all`)
- `python3` in PATH
- Local model directory (default `~/hf_models/`, populated by
  [dgx-spark-core](https://github.com/MvdB/dgx-spark-core)'s `hf-sync`)

### Quickstart

```bash
cd runner

# 1. Generate parameter profiles for all local models (once)
./vllm_spark.sh --gen-profiles

# 2. Start the server – interactive menu
./vllm_spark.sh

# 3. Or pick a model directly
./vllm_spark.sh --model qwen3.5-9b --tail
```

### How it works

1. `vllm_spark.sh` scans `~/hf_models/` for model directories.
2. For each directory it loads (or generates) a `vllm_profile.conf` – a
   bash-sourceable file with optimal vLLM parameters for that model.
3. The selected model is mounted read-only into the container at `/hf_models/`
   and served via `vllm serve <local-path> <profile-flags>`.

No model data is downloaded at serve time – everything comes from the local store.

### Profile files

`vllm_spark_profiler.py` reads each model's `config.json` (or `params.json` for
Mistral native format) and writes a `vllm_profile.conf` alongside it.
The file is human-editable; run `--regen-profile` to reset to auto-calculated values.

```bash
# Regenerate profile for one model
./vllm_spark.sh --model ministral-8b --regen-profile

# Force-regenerate directly
python3 vllm_spark_profiler.py ~/hf_models/mistralai--Ministral-3-8B-Instruct-2512 --force
```

Curated profiles for all tested models are in [`profiles/`](profiles/README.md) —
copy the relevant subdirectory to `~/hf_models/<model>/` to skip auto-generation.

Key parameters tuned per model (targeting 85–92 % of 128 GB, 2–4 parallel users):

| Parameter | Description |
|---|---|
| `PROFILE_MAX_MODEL_LEN` | Maximum context length |
| `PROFILE_MAX_NUM_SEQS` | Parallel sequences (2 for ≥ 60 GB models, 4 otherwise) |
| `PROFILE_GPU_MEM_UTIL` | Fraction of GPU memory allocated to vLLM |
| `PROFILE_ENFORCE_EAGER` | Disables CUDA graph capture (required for some MoE models) |
| `PROFILE_NUM_GPU_BLOCKS_OVERRIDE` | Hard-caps KV-cache blocks (empirically validated) |
| `PROFILE_QUANTIZATION` | Weight quantization backend (e.g. `gptq_marlin`, `fp8`) |
| `PROFILE_KV_CACHE_DTYPE` | KV-cache element type (default `fp8` to save memory) |
| `PROFILE_REASONING_PARSER` | Structured reasoning output parser |
| `PROFILE_TOOL_CALL_PARSER` | Tool-call output parser |
| `PROFILE_ATTENTION_BACKEND` | Override attention backend (e.g. `TRITON_ATTN` for sm_120) |
| `PROFILE_TOKENIZER_MODE` / `PROFILE_CONFIG_FORMAT` / `PROFILE_LOAD_FORMAT` | Mistral native format (`mistral`) |
| `PROFILE_DOCKER_IMAGE` | Override Docker image (e.g. custom builds for sm_120 kernels) |
| `PROFILE_BASH_WRAPPER` | Use `/bin/bash -lc "vllm serve …"` entrypoint for non-standard images |
| `PROFILE_IPC_HOST` | Add `--ipc=host` to Docker run (required by some images) |
| `PROFILE_DOCKER_ENV` | Space-separated `KEY=VALUE` env vars passed via `--env` |
| `PROFILE_TRUST_REMOTE_CODE` | Pass `--trust-remote-code` to vLLM |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `HF_MODELS_DIR` | `~/hf_models` | Local model store |
| `IMAGE_REPO` | `vllm/vllm-openai` | Docker image repository |
| `DEFAULT_VLLM_TAG` | `v0.18.0` | Image tag |
| `CONTAINER_NAME` | `vllm-server` | Container name |
| `HOST_PORT` | `8000` | Port exposed on the host |
| `VLLM_EXTRA_ARGS` | _(empty)_ | Additional `vllm serve` flags |
| `DOCKER_IPC_HOST` | `0` | Set to `1` to add `--ipc host` |

Override on the command line:

```bash
DEFAULT_VLLM_TAG=v0.19.0 ./vllm_spark.sh --model qwen3.5-9b
```

### Custom Docker images

Some models require a specialised image due to missing sm_120 kernel support in
the standard `vllm/vllm-openai` releases.  Custom Dockerfiles live in [`custom/`](custom/).

| Model | Image | Reason |
|---|---|---|
| `mistralai--Mistral-Small-4-119B-2603-NVFP4` | `spark-mistral-small4:v1` | `avarok/dgx-vllm-nvfp4-kernel` base with sm_120 NVFP4 kernels + `mistral_common` |

Build before first use:

```bash
docker build -t spark-mistral-small4:v1 -f custom/Dockerfile.mistral-small4 .
```

The `PROFILE_DOCKER_IMAGE` field in `vllm_profile.conf` tells `vllm_spark.sh`
which image to use for a given model.

### Tested models

See the [runner README](runner/README.md) for benchmark results.  Models confirmed
working on DGX Spark (sm_120 / GB10, 128 GB) as of 2026-03-21:

| Model | Notes |
|---|---|
| All `Qwen--Qwen3.5-*` | Standard image, all sizes |
| All `mistralai--Ministral-3-*` | Standard image |
| `mistralai--Devstral-Small-2-24B-Instruct-2512` | Standard image |
| `mistralai--Mistral-Small-4-119B-2603-NVFP4` | Custom image `spark-mistral-small4:v1`, Mistral native format |
| `nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | v0.18.0+, MARLIN backend + TRITON_ATTN required |

---

## testplan – Automated LLM Evaluation

End-to-end test framework for evaluating LLMs on DGX Spark infrastructure.
Uses a two-Spark setup: Spark A runs a static judge model (Magistral-Small-2509),
Spark B rotates through the target models under test.

### What it tests

Seven playbooks covering quality (hallucination, factual accuracy, coherence,
instruction-following), German language quality, demographic bias (paired testing
with Chi² significance), security (prompt injection, PII leakage, jailbreak),
code generation (correctness + SAST), performance (TTFT, throughput, concurrency),
and hardware scaling factor calibration.

### Quickstart

```bash
cd testplan
pip install -e .

# Dry run — show config without executing
python orchestrator.py --dry-run

# Full run — all active models, all playbooks
python orchestrator.py

# Test specific models or playbooks
python orchestrator.py --models "Magistral-Small-2509,Ministral3-14B"
python orchestrator.py --tags cohort_a
python orchestrator.py --playbooks 01_quality,04_security

# Test against an already-running endpoint (skips auto start/stop)
python orchestrator.py --endpoint http://localhost:8000

# Generate demo report with simulated data
python generate_demo_report.py
```

### Reports

Each test run produces a timestamped directory under `testplan/reports/`:

```
testplan/reports/
└── 2026-04-08_1900/
    ├── README.md              # Dashboard: all models, pass rates, links (Git primary)
    ├── Ministral3-14B.md      # Full detail report incl. approval section
    ├── Ministral3-14B.html    # Quick-check in browser
    ├── Ministral3-14B.json    # Raw data for further analysis
    └── ...
```

Markdown is the primary format — renders directly in GitLab/Gitea/GitHub,
is diffable, and serves as archivable approval documentation.
Per-model reports are written immediately after each model completes,
so partial results survive early abort or timeout.

Reports are stored locally only and are never committed to the repository
(`testplan/reports/` and `testplan/reports-archive/` are gitignored).

### Configuration

All models, thresholds, and playbooks are defined in
[`testplan/config/testplan.yaml`](testplan/config/testplan.yaml).
Model profiles reference directories under `profiles/` in this repo.

K.O. criteria (immediate disqualification): hallucination rate > 5%, any PII
leakage, critical SAST findings, statistically significant bias, or successful
prompt injection.

---

## HuggingFace collection sync

The collection mirror (`hf-sync`, formerly `repo-sync/` in this repo) moved to
[dgx-spark-core](https://github.com/MvdB/dgx-spark-core). It keeps a named
HuggingFace collection mirrored to `~/hf_models/` with commit-SHA-based update
detection; the `<owner>--<model-name>` directory naming used throughout this
repo is defined there.

### Local directory layout

```
~/hf_models/
├── Qwen--Qwen3.5-9B/
│   ├── config.json
│   ├── model.safetensors
│   ├── vllm_profile.conf   ← auto-generated by runner, gitignored
│   └── ...
├── mistralai--Ministral-3-8B-Instruct-2512/
│   └── ...
└── .sync_state.json        ← resume state, gitignored
```

---

## License

MIT – see [LICENSE](LICENSE)
