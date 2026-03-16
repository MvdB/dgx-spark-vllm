# Contributing

Contributions are welcome. Please follow these guidelines.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
pip install -r requirements.txt
pip install ruff mypy
```

## Code style

Python code is linted and formatted with [ruff](https://docs.astral.sh/ruff/)
and type-checked with [mypy](https://mypy.readthedocs.io/).

```bash
ruff check hf_sync.py vllm_spark_profiler.py
ruff format hf_sync.py vllm_spark_profiler.py
mypy hf_sync.py vllm_spark_profiler.py
```

Shell scripts are checked with [shellcheck](https://www.shellcheck.net/):

```bash
shellcheck vllm_spark.sh
```

CI runs all checks on every push and pull request.

## Pull requests

- Keep changes focused – one concern per PR.
- Do not commit `.env`, `.sync_state.json`, log files, model weights, or
  `vllm_profile.conf` files.
- Update `README.md` if behaviour or configuration changes.
- Add a note to the PR description if you validated a new model on DGX Spark.

## Reporting bugs

Open a GitHub issue and include:
- The model name and profile (`vllm_profile.conf` contents, redacted if needed)
- Relevant lines from `docker logs vllm-server`
- vLLM image tag used
