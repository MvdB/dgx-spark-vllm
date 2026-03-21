# profiles

Curated `vllm_profile.conf` files for models tested on DGX Spark (GB10, sm_120, 128 GB).

Each subdirectory mirrors the model directory name used by `vllm_spark.sh`
(`<owner>--<model-name>`, slashes replaced by `--`).

## Usage

Copy the profile for your model into its local checkout directory:

```bash
cp profiles/mistralai--Mistral-Small-4-119B-2603-NVFP4/vllm_profile.conf \
   ~/hf_models/mistralai--Mistral-Small-4-119B-2603-NVFP4/
```

Or copy all at once (skips models you don't have locally):

```bash
for dir in profiles/*/; do
  model=$(basename "$dir")
  dest=~/hf_models/$model
  [[ -d "$dest" ]] && cp "$dir/vllm_profile.conf" "$dest/" && echo "copied $model"
done
```

`vllm_spark.sh` loads the profile automatically when starting a model.
If no profile exists it generates one via `vllm_spark_profiler.py` — the
curated profiles here take precedence over auto-generation for known-good models.

## Models

| Model | Compatible | Notes |
|---|:---:|---|
| `mistralai--Devstral-Small-2-24B-Instruct-2512` | ✅ | Standard image |
| `mistralai--Ministral-3-14B-Instruct-2512` | ✅ | Standard image |
| `mistralai--Ministral-3-14B-Reasoning-2512` | ✅ | Standard image |
| `mistralai--Ministral-3-3B-Instruct-2512` | ✅ | Standard image |
| `mistralai--Ministral-3-3B-Reasoning-2512` | ✅ | Standard image |
| `mistralai--Ministral-3-8B-Instruct-2512` | ✅ | Standard image |
| `mistralai--Ministral-3-8B-Reasoning-2512` | ✅ | Standard image |
| `mistralai--Mistral-Small-4-119B-2603-NVFP4` | ✅ | Custom image `spark-mistral-small4:v1` — build first: `docker build -t spark-mistral-small4:v1 -f custom/Dockerfile.mistral-small4 .` |
| `nvidia--Nemotron-3-Content-Safety` | ✅ | Standard image |
| `nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | ✅ | v0.18.0+, MARLIN + TRITON_ATTN; FLASHINFER_CUTLASS crashes on sm_120 |
| `openai--gpt-oss-20b` | ✅ | Standard image |
| `openai--gpt-oss-safeguard-20b` | ✅ | Standard image |
| `openai--gpt-oss-120b` | ❌ | `AttributeError: 'NoneType' has no attribute 'endswith'` |
| `Qwen--Qwen3.5-0.8B` | ✅ | Standard image |
| `Qwen--Qwen3.5-2B` | ✅ | Standard image |
| `Qwen--Qwen3.5-4B` | ✅ | Standard image |
| `Qwen--Qwen3.5-9B` | ✅ | Standard image |
| `Qwen--Qwen3.5-27B-GPTQ-Int4` | ✅ | Standard image |
| `Qwen--Qwen3.5-35B-A3B-GPTQ-Int4` | ✅ | Standard image |
| `Qwen--Qwen3.5-122B-A10B-GPTQ-Int4` | ✅ | Standard image |
