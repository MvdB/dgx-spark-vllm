# How a model gets served

`runner/vllm_spark.sh` starts a vLLM container for any model in the local store.
The interesting part is not the container — it is the profile.

## The flow

1. `vllm_spark.sh` scans `~/hf_models/` for model directories.
2. For each it loads, or generates, a `vllm_profile.conf` — a bash-sourceable
   file with the vLLM parameters that work for that model on this hardware.
3. The selected model is mounted **read-only** into the container at
   `/hf_models/` and served with `vllm serve <local-path> <profile-flags>`.

Nothing is downloaded at serve time; everything comes from the local store.

```bash
cd runner
./vllm_spark.sh --gen-profiles          # once, after downloading models
./vllm_spark.sh                          # interactive picker
./vllm_spark.sh --model qwen3.5-9b --tail
```

## Profile files

`vllm_spark_profiler.py` reads each model's `config.json` — or `params.json` for
Mistral's native format — and writes `vllm_profile.conf` alongside it. The file
is meant to be edited by hand; `--regen-profile` resets it to auto-calculated
values.

```bash
./vllm_spark.sh --model ministral-8b --regen-profile
python3 vllm_spark_profiler.py ~/hf_models/mistralai--Ministral-3-8B-Instruct-2512 --force
```

The generator checks a `KNOWN_GOOD` table first — empirically validated settings
take precedence over any heuristic. Only then does it fall back to reading the
config, detecting architecture and quantization, estimating model size and
budgeting KV cache against the 128 GB target (85 % utilisation, 5 GB overhead).

**Curated profiles for every tested model** are in
[southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) under
`vllm/profiles/`. Copy the relevant subdirectory to `~/hf_models/<model>/` to skip
auto-generation entirely. That repo is the authoritative list of what has been
made to work — this page only explains the mechanism.

## The fields

Tuned per model, targeting 85–92 % of 128 GB with 2–4 parallel users:

| Field | |
|---|---|
| `PROFILE_MAX_MODEL_LEN` | Maximum context length |
| `PROFILE_MAX_NUM_SEQS` | Parallel sequences — 2 for models ≥ 60 GB, 4 otherwise |
| `PROFILE_GPU_MEM_UTIL` | Fraction of GPU memory given to vLLM |
| `PROFILE_ENFORCE_EAGER` | Disables CUDA graph capture; required for some MoE models |
| `PROFILE_NUM_GPU_BLOCKS_OVERRIDE` | Hard cap on KV-cache blocks, empirically validated |
| `PROFILE_QUANTIZATION` | Weight quantization backend, e.g. `gptq_marlin`, `fp8` |
| `PROFILE_KV_CACHE_DTYPE` | KV-cache element type, default `fp8` to save memory |
| `PROFILE_REASONING_PARSER` | Structured reasoning output parser |
| `PROFILE_TOOL_CALL_PARSER` | Tool-call output parser |
| `PROFILE_ATTENTION_BACKEND` | Override the attention backend, e.g. `TRITON_ATTN` for sm_120 |
| `PROFILE_TOKENIZER_MODE` · `PROFILE_CONFIG_FORMAT` · `PROFILE_LOAD_FORMAT` | Mistral native format (`mistral`) |
| `PROFILE_DOCKER_IMAGE` | Override the Docker image — custom sm_120 builds |
| `PROFILE_BASH_WRAPPER` | Use a `/bin/bash -lc "vllm serve …"` entrypoint for non-standard images |
| `PROFILE_IPC_HOST` | Add `--ipc=host`; mutually exclusive with `--shm-size` |
| `PROFILE_DOCKER_ENV` | Space-separated `KEY=VALUE` passed through as `--env` |
| `PROFILE_TRUST_REMOTE_CODE` | Pass `--trust-remote-code` |

Each field is translated to a CLI flag only if the target image supports it —
`vllm_supports()` checks the image's own `--help=all` output first, so an older
image does not fail on a flag it has never heard of.

## Environment

| Variable | Default | |
|---|---|---|
| `HF_MODELS_DIR` | `~/hf_models` | Local model store |
| `IMAGE_REPO` | `vllm/vllm-openai` | Docker image repository |
| `DEFAULT_VLLM_TAG` | `v0.26.0` | Image tag — a fallback; curated profiles pin their own |
| `CONTAINER_NAME` | `vllm-server` | Container name |
| `HOST_PORT` | `8000` | Port on the host |
| `VLLM_EXTRA_ARGS` | — | Extra flags appended to `vllm serve` |
| `DOCKER_IPC_HOST` | `0` | Set to `1` to add `--ipc host` |

```bash
DEFAULT_VLLM_TAG=v0.19.0 ./vllm_spark.sh --model qwen3.5-9b
```

## Custom images for sm_120

Some models need kernels the stock `vllm/vllm-openai` releases do not ship. The
Dockerfiles live in
[southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) under
`vllm/custom/`, and **must be built before first use**:

```bash
cd ~/southbyte/southbyte-spark-profiles/vllm/custom
docker build -t spark-mistral-small4:v1 -f Dockerfile.mistral-small4 .
```

`PROFILE_DOCKER_IMAGE` in the model's profile then selects it. Example:
`mistralai--Mistral-Small-4-119B-2603-NVFP4` needs `spark-mistral-small4:v1`, an
`avarok/dgx-vllm-nvfp4-kernel` base with sm_120 NVFP4 kernels plus
`mistral_common`.

## Model directory naming

HuggingFace IDs use `--` in place of `/`: `mistralai/Mistral-7B-v0.1` becomes
`mistralai--Mistral-7B-v0.1`. The profiler and
[southbyte-sync](https://github.com/MvdB/southbyte-sync) both follow it; the
convention is defined there.

```
~/hf_models/
├── Qwen--Qwen3.5-9B/
│   ├── config.json
│   ├── model.safetensors
│   ├── vllm_profile.conf   ← generated by the runner, gitignored
│   └── …
└── .sync_state.json        ← resume state, gitignored
```

## Benchmarks

`runner/test_models.py` iterates the compatible models, starts each one, polls
`/v1/models`, runs warmup plus correctness queries plus a streaming benchmark
(TTFT, tokens per second), and writes the result into `runner/README.md`.

That file is therefore **machine-written** — do not edit it by hand, the next run
overwrites it.
