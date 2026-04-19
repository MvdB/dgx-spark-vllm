#!/usr/bin/env python3
"""
vLLM Profile Generator für DGX Spark (GB10, 128 GB unified memory).

Verwendung:
  python3 vllm_spark_profiler.py <model_dir> [--force]

Schreibt vllm_profile.conf in das Modell-Verzeichnis.
Das Profil ist bash-sourceable und wird von vllm_spark.sh geladen.
"""

import json
import os
import re
import sys

# ── Hardware-Konstanten ────────────────────────────────────────────────────
TOTAL_MEM_GB   = 128.0
TARGET_UTIL    = 0.85
AVAILABLE_GB   = TOTAL_MEM_GB * TARGET_UTIL   # 108.8 GB
OVERHEAD_GB    = 5.0                           # CUDA-Kontext, Libs, Puffer
BLOCK_SIZE     = 16                            # vLLM Standard-Block-Tokens
TARGET_SEQS    = 4                             # Ziel: 2–4 parallele User

# ── Nicht von vllm serve unterstützte Architekturen ───────────────────────
UNSUPPORTED_ARCHS = {
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSTokenizerV2Model",
    # Audio-Modelle: vLLM unterstützt kein Audio (nur vLLM-Omni)
    "VoxtralForConditionalGeneration",
    "VoxtralRealtimeForConditionalGeneration",
}

# ── Empirisch validierte Profile (aus Crash-Analyse und Logs) ─────────────
# Überschreiben die automatisch berechneten Werte vollständig.
KNOWN_GOOD = {
    # Gemischte NVFP4/FP8-Quantisierung (modelopt). Alle FP4-GEMM-Backends
    # schlagen auf sm_120 (GB10) fehl mit vllm/vllm-openai:v0.17.1:
    #   FLASHINFER_CUTLASS → TVM hat keine sm_120-Tactic
    #   VLLM_CUTLASS       → cutlass_scaled_fp4_mm: Error Internal
    #   MARLIN             → NaN-Logits (modelopt mixed NVFP4/FP8 nicht korrekt interpretiert)
    # TODO: avarok/dgx-vllm-nvfp4-kernel:v23 + VLLM_TEST_FORCE_FP8_MARLIN=1 noch nicht getestet
    #       (hat für Mistral-Small-4 funktioniert – könnte auch hier helfen)
    "nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": {
        "PROFILE_VLLM_COMPATIBLE":          1,
        "PROFILE_GPU_MEM_UTIL":             "0.92",
        "PROFILE_MAX_MODEL_LEN":            32768,
        "PROFILE_MAX_NUM_SEQS":             4,
        "PROFILE_MAX_NUM_BATCHED_TOKENS":   16384,
        "PROFILE_KV_CACHE_DTYPE":           "fp8",
        "PROFILE_TOOL_CALL_PARSER":         "qwen3_coder",
        "PROFILE_ENABLE_AUTO_TOOL_CHOICE":  1,
        "PROFILE_REASONING_PARSER":         "nemotron_v3",
        "PROFILE_TRUST_REMOTE_CODE":        1,
        "PROFILE_ATTENTION_BACKEND":        "TRITON_ATTN",
        "PROFILE_DOCKER_IMAGE":             "vllm/vllm-openai:v0.18.0",
        "PROFILE_DOCKER_ENV": (
            "VLLM_NVFP4_GEMM_BACKEND=marlin"
            " VLLM_USE_FLASHINFER_MOE_FP4=0"
            " VLLM_TEST_FORCE_FP8_MARLIN=1"
            " VLLM_ENGINE_CORE_STARTUP_TIMEOUT=300"
        ),
        "PROFILE_NOTES": (
            "Bestätigt funktionsfähig auf DGX Spark (sm_120) mit vllm/vllm-openai:v0.18.0. "
            "Benötigt MARLIN-Backend (FLASHINFER_CUTLASS crasht auf sm_120). "
            "v0.17.1 war inkompatibel (NaN-Logits mit MARLIN, crash mit anderen Backends)."
        ),
    },
    "Qwen--Qwen3.5-122B-A10B-GPTQ-Int4": {
        "PROFILE_VLLM_COMPATIBLE":          1,
        "PROFILE_DTYPE":                    "float16",
        "PROFILE_QUANTIZATION":             "gptq_marlin",
        "PROFILE_GPU_MEM_UTIL":             "0.85",
        "PROFILE_MAX_MODEL_LEN":            32768,
        "PROFILE_MAX_NUM_SEQS":             4,
        "PROFILE_MAX_NUM_BATCHED_TOKENS":   4096,
        "PROFILE_ENFORCE_EAGER":            1,
        "PROFILE_NUM_GPU_BLOCKS_OVERRIDE":  512,
        "PROFILE_TRUST_REMOTE_CODE":        0,
        "PROFILE_KV_CACHE_DTYPE":           "fp8",
        "PROFILE_HF_OVERRIDES":             '{"num_nextn_predict_layers": 0}',
        "PROFILE_REASONING_PARSER":         "qwen3",
        "PROFILE_TOOL_CALL_PARSER":         "qwen3_coder",
        "PROFILE_ENABLE_AUTO_TOOL_CHOICE":  1,
        "PROFILE_NOTES": (
            "Empirisch validiert 2026-03-09. "
            "MoE 122B GPTQ-Int4; Encoder-Cache-Overhead reduziert KV-Pool auf ~27 GiB (571 Blöcke). "
            "512 Blöcke Sicherheitspuffer. enforce_eager pflicht (CUDA-Graph-OOM). "
            "max_model_len=32768 mit evtl. KV-Eviction für lange Kontexte."
        ),
    },
    # ── Phi-4-reasoning-plus ─────────────────────────────────────────────────
    # INKOMPATIBEL: Modell bleibt in endlosem Reasoning-Loop hängen, trifft
    # max_tokens ohne finales Answer zu produzieren.
    # vLLM-Issue: https://github.com/vllm-project/vllm/issues/18141
    # Status: closed "not planned" — kein Fix verfügbar.
    "microsoft--Phi-4-reasoning-plus": {
        "PROFILE_VLLM_COMPATIBLE":          0,
        "PROFILE_NOTES": (
            "INKOMPATIBEL mit vLLM. Modell hängt in endlosem Thinking-Loop "
            "(hits max_tokens ohne finales Answer). "
            "vLLM-Issue #18141 — closed 'not planned', kein Fix verfügbar."
        ),
    },
    # ── Nemotron-3-Nano-30B-A3B FP8 ─────────────────────────────────────────
    # NemotronH MoE-Mamba-Hybrid. Gleiche Architektur wie Nemotron-3-Super,
    # benötigt identische Setup: vllm/vllm-openai:v0.18.0 + TRITON_ATTN + trust_remote_code.
    # FP8-Quantisierung: quant_method nicht in config.json (vLLM autoerkennung).
    # 30B FP8 ~30 GB, 52 Layer, 2 KV-Heads, head_dim=128.
    # KV-Bytes/Token: 52 × 2 × 2 × 128 × 1 = 26.624 B → sehr kleiner KV-Bedarf.
    "nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": {
        "PROFILE_VLLM_COMPATIBLE":          1,
        "PROFILE_GPU_MEM_UTIL":             "0.88",
        "PROFILE_MAX_MODEL_LEN":            131072,
        "PROFILE_MAX_NUM_SEQS":             4,
        "PROFILE_KV_CACHE_DTYPE":           "fp8",
        "PROFILE_ENFORCE_EAGER":            1,
        "PROFILE_TRUST_REMOTE_CODE":        1,
        "PROFILE_ATTENTION_BACKEND":        "TRITON_ATTN",
        "PROFILE_REASONING_PARSER":         "deepseek_r1",
        "PROFILE_DOCKER_IMAGE":             "vllm/vllm-openai:v0.18.0",
        "PROFILE_DOCKER_ENV": (
            "VLLM_ENGINE_CORE_STARTUP_TIMEOUT=300"
        ),
        "PROFILE_NOTES": (
            "Nemotron-3-Nano 30B A3B FP8. NemotronH MoE-Mamba-Hybrid (auto_map → custom code). "
            "Gleiche Anforderungen wie Nemotron-3-Super: vllm/vllm-openai:v0.18.0 + TRITON_ATTN. "
            "52 Layer, 2 KV-Heads, head_dim=128 → sehr kleiner KV-Bedarf (26 KB/Token). "
            "trust_remote_code=1 für auto_map (NemotronHForCausalLM)."
        ),
    },
    # ── Gemma 4 (google) ──────────────────────────────────────────────────────
    # Alle Varianten: vllm/vllm-openai:v0.19.1+ erforderlich (Gemma4ForConditionalGeneration).
    # Architektur: Interleaved Sliding/Global Attention (layer_types: 5 sliding + 1 full).
    # KV-Budget-Berechnung: nur Full-Attention-Layer skalieren linear mit Kontext.
    # Sliding-Layer sind durch die Fenstergröße begrenzt und haben keinen
    # kontextlängenabhängigen KV-Bedarf.
    #
    # 26B A4B MoE · 30 Layer (5 full, 25 sliding) · BF16 ~52 GB
    # Global-KV pro Token: 5 × 2 heads × 512 dim × 2 (K+V) × 1 Byte = 10.240 Byte
    # Bei max_model_len=131072, max_num_seqs=2: ~2,6 GB KV → sehr großzügig.
    "google--gemma-4-26B-A4B-it": {
        "PROFILE_VLLM_COMPATIBLE":         1,
        "PROFILE_GPU_MEM_UTIL":            "0.88",
        "PROFILE_MAX_MODEL_LEN":           131072,
        "PROFILE_MAX_NUM_SEQS":            2,
        "PROFILE_KV_CACHE_DTYPE":          "fp8",
        "PROFILE_ENFORCE_EAGER":           1,
        "PROFILE_REASONING_PARSER":        "gemma4",
        "PROFILE_NOTES": (
            "Gemma 4 26B A4B MoE. 30 Layer (5 full, 25 sliding), sliding_window=1024. "
            "BF16 ~52 GB. MoE (128 Experten, top-8) + Interleaved Attention → enforce_eager. "
            "Nur Full-Attention-Layer skalieren mit Kontext: 5 × 2 × 512 × 2 = 10.240 B/Token. "
            "Bei 2 Seq. und 131072 Token nur ~2,6 GB KV-Cache. "
            "Benötigt vllm/vllm-openai:v0.19.1+."
        ),
    },
    # 31B Dense · 60 Layer (10 full, 50 sliding) · BF16 ~62 GB
    # Global-KV pro Token: 10 × 4 heads × 512 dim × 2 (K+V) × 1 Byte = 40.960 Byte
    # Bei max_model_len=131072, max_num_seqs=2: ~10,2 GB KV → sicher bei ~48 GB Budget.
    "google--gemma-4-31B-it": {
        "PROFILE_VLLM_COMPATIBLE":         1,
        "PROFILE_GPU_MEM_UTIL":            "0.88",
        "PROFILE_MAX_MODEL_LEN":           131072,
        "PROFILE_MAX_NUM_SEQS":            2,
        "PROFILE_KV_CACHE_DTYPE":          "fp8",
        "PROFILE_ENFORCE_EAGER":           1,
        "PROFILE_REASONING_PARSER":        "gemma4",
        "PROFILE_NOTES": (
            "Gemma 4 31B Dense. 60 Layer (10 full, 50 sliding), sliding_window=1024. "
            "BF16 ~62 GB. Interleaved Attention → enforce_eager. "
            "Nur Full-Attention-Layer skalieren mit Kontext: 10 × 4 × 512 × 2 = 40.960 B/Token. "
            "Bei 2 Seq. und 131072 Token ~10,2 GB KV-Cache (Budget ~48 GB). "
            "Auf 262144 erhöhbar (~20 GB KV), dann max_num_seqs=1 empfohlen. "
            "Benötigt vllm/vllm-openai:v0.19.1+."
        ),
    },
    # E2B · 35 Layer (7 full, 28 sliding) · BF16 ~4 GB (+ Audio-Encoder)
    # Global-KV pro Token: 7 × 1 head × 512 dim × 2 (K+V) × 1 Byte = 7.168 Byte
    # Bei max_model_len=131072, max_num_seqs=4: ~3,5 GB KV → sehr großzügig.
    # Audio-Input: encoder vorhanden, vLLM-Audio-Support für Gemma 4 ungetestet.
    "google--gemma-4-E4B-it": {
        "PROFILE_VLLM_COMPATIBLE":         1,
        "PROFILE_GPU_MEM_UTIL":            "0.80",
        "PROFILE_MAX_MODEL_LEN":           131072,
        "PROFILE_MAX_NUM_SEQS":            4,
        "PROFILE_ENFORCE_EAGER":           1,
        "PROFILE_KV_CACHE_DTYPE":          "fp8",
        "PROFILE_REASONING_PARSER":        "gemma4",
        "PROFILE_DOCKER_IMAGE":            "spark-gemma4:v1",
        "PROFILE_NOTES": (
            "Gemma 4 E4B. 42 Layer (7 full, 35 sliding), sliding_window=512. "
            "BF16 ~8 GB + Audio/Vision-Encoder. "
            "Nur Full-Attention-Layer skalieren mit Kontext: 7 × 2 × 512 × 2 = 14.336 B/Token. "
            "Bei 4 Seq. und 131072 Token ~7 GB KV-Cache. "
            "Audio-Input: encoder vorhanden, vLLM-Support ungetestet – Text/Vision empfohlen. "
            "Benötigt spark-gemma4:v1 (vllm/vllm-openai:v0.19.1+ Basis)."
        ),
    },
    "google--gemma-4-E2B-it": {
        "PROFILE_VLLM_COMPATIBLE":         1,
        "PROFILE_GPU_MEM_UTIL":            "0.80",
        "PROFILE_MAX_MODEL_LEN":           131072,
        "PROFILE_MAX_NUM_SEQS":            4,
        "PROFILE_ENFORCE_EAGER":           1,
        "PROFILE_KV_CACHE_DTYPE":          "fp8",
        "PROFILE_REASONING_PARSER":        "gemma4",
        "PROFILE_DOCKER_IMAGE":            "spark-gemma4:v1",
        "PROFILE_NOTES": (
            "Gemma 4 E2B. 35 Layer (7 full, 28 sliding), sliding_window=512. "
            "BF16 ~4 GB + Audio/Vision-Encoder. "
            "Nur Full-Attention-Layer skalieren mit Kontext: 7 × 1 × 512 × 2 = 7.168 B/Token. "
            "Bei 4 Seq. und 131072 Token ~3,5 GB KV-Cache. "
            "Audio-Input: encoder vorhanden, vLLM-Support ungetestet – Text/Vision empfohlen. "
            "Benötigt spark-gemma4:v1 (vllm/vllm-openai:v0.19.1+ Basis)."
        ),
    },
    # ── Mistral-Small-4 119B NVFP4 ───────────────────────────────────────────
    # Mistral-Small-4 119B NVFP4: benötigt avarok/dgx-vllm-nvfp4-kernel:v23
    # (Community-Image mit sm_120-kompatiblen NVFP4-Kerneln + mistral_common).
    # Bau: docker build -t spark-mistral-small4:v1 -f custom/Dockerfile.mistral-small4 .
    # Mistral-Format-Loading: tokenizer-mode/config-format/load-format = "mistral"
    # VLLM_MLA_DISABLE=1: verhindert Abstürze durch MLA-Attention auf diesem Modell.
    # VLLM_TEST_FORCE_FP8_MARLIN=1: erzwingt FP8-MARLIN-Pfad für gemischte Quantisierung.
    "mistralai--Mistral-Small-4-119B-2603-NVFP4": {
        "PROFILE_VLLM_COMPATIBLE":         1,
        "PROFILE_GPU_MEM_UTIL":            "0.75",
        "PROFILE_MAX_MODEL_LEN":           32768,
        "PROFILE_MAX_NUM_SEQS":            4,
        "PROFILE_MAX_NUM_BATCHED_TOKENS":  16384,
        "PROFILE_KV_CACHE_DTYPE":          "fp8",
        "PROFILE_TOKENIZER_MODE":          "mistral",
        "PROFILE_CONFIG_FORMAT":           "mistral",
        "PROFILE_LOAD_FORMAT":             "mistral",
        "PROFILE_TOOL_CALL_PARSER":        "mistral",
        "PROFILE_ENABLE_AUTO_TOOL_CHOICE": 1,
        "PROFILE_DOCKER_IMAGE":            "spark-mistral-small4:v1",
        "PROFILE_BASH_WRAPPER":            1,
        "PROFILE_IPC_HOST":                1,
        "PROFILE_DOCKER_ENV": (
            "VLLM_MLA_DISABLE=1"
            " VLLM_NVFP4_GEMM_BACKEND=marlin"
            " VLLM_USE_FLASHINFER_MOE_FP4=0"
            " VLLM_TEST_FORCE_FP8_MARLIN=1"
            " VLLM_ENGINE_CORE_STARTUP_TIMEOUT=300"
        ),
        "PROFILE_NOTES": (
            "Mistral Small 4 119B NVFP4. Benötigt custom Image spark-mistral-small4:v1 "
            "(avarok/dgx-vllm-nvfp4-kernel:v23 + mistral_common). "
            "Bau: docker build -t spark-mistral-small4:v1 -f custom/Dockerfile.mistral-small4 . "
            "Erst bauen, dann Modell starten."
        ),
    },
}

# ── Architektur-Hints ──────────────────────────────────────────────────────
# Werden auf auto-berechnete Profile angewendet.
ARCH_HINTS = {
    # Interleaved linear_attention / full_attention → enforce_eager pflicht.
    # Kein reasoning_parser: Qwen3.5 unterstützt Thinking-Mode optional, aber
    # kleine Modelle (≤9B) produzieren nur <think>-Tokens ohne Inhalt → content=None.
    # Thinking nur explizit in KNOWN_GOOD aktivieren (z.B. 122B).
    "Qwen3_5ForConditionalGeneration": {
        "enforce_eager":           1,
        "tool_call_parser":        "qwen3_coder",
        "enable_auto_tool_choice": 1,
        "check_mtp":               True,
    },
    "Qwen3_5MoeForConditionalGeneration": {
        "enforce_eager":           1,
        "tool_call_parser":        "qwen3_coder",
        "enable_auto_tool_choice": 1,
        "check_mtp":               True,
        "gpu_blocks_override_large_moe": True,
    },
    "Mistral3ForConditionalGeneration": {
        "enable_auto_tool_choice": 1,
        "tool_call_parser":        "mistral",
        # Mistral-Reasoning-Modelle: kein eigener reasoning_parser nötig;
        # das Modell gibt CoT nativ aus.
    },
    "NemotronHForCausalLM": {
        "trust_remote_code": 1,    # auto_map → custom modeling code
        "enforce_eager":     1,
        "check_mtp":         True,
    },
    "GptOssForCausalLM": {
        "enforce_eager":     1,
    },
    # Gemma 4: multimodale Familie (Text + Vision, E2B/E4B zusätzlich Audio).
    # Interleaved Sliding/Global Attention → enforce_eager, reasoning_parser.
    # MoE-Varianten (26B A4B) werden zusätzlich über enable_moe_block erkannt.
    "Gemma4ForConditionalGeneration": {
        "enforce_eager":           1,
        "reasoning_parser":        "gemma4",
    },
}

# Reihenfolge der Felder in der Ausgabe
KEY_ORDER = [
    "PROFILE_VLLM_COMPATIBLE",
    "PROFILE_DTYPE",
    "PROFILE_QUANTIZATION",
    "PROFILE_GPU_MEM_UTIL",
    "PROFILE_MAX_MODEL_LEN",
    "PROFILE_MAX_NUM_SEQS",
    "PROFILE_MAX_NUM_BATCHED_TOKENS",
    "PROFILE_ENFORCE_EAGER",
    "PROFILE_NUM_GPU_BLOCKS_OVERRIDE",
    "PROFILE_TRUST_REMOTE_CODE",
    "PROFILE_KV_CACHE_DTYPE",
    "PROFILE_HF_OVERRIDES",
    "PROFILE_ATTENTION_BACKEND",
    "PROFILE_CHAT_TEMPLATE",
    "PROFILE_TOKENIZER_MODE",
    "PROFILE_CONFIG_FORMAT",
    "PROFILE_LOAD_FORMAT",
    "PROFILE_REASONING_PARSER",
    "PROFILE_REASONING_PARSER_PLUGIN",
    "PROFILE_TOOL_CALL_PARSER",
    "PROFILE_ENABLE_AUTO_TOOL_CHOICE",
    "PROFILE_DOCKER_IMAGE",
    "PROFILE_BASH_WRAPPER",
    "PROFILE_IPC_HOST",
    "PROFILE_DOCKER_ENV",
    "PROFILE_NOTES",
]


# ── Hilfsfunktionen ───────────────────────────────────────────────────────

def load_config(model_dir: str) -> dict | None:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path) as f:
        return json.load(f)


def get_text_cfg(cfg: dict) -> dict:
    """Gibt text_config zurück (falls vorhanden), sonst das Top-Level-Dict."""
    return cfg.get("text_config") or cfg


def get_quant_info(cfg: dict) -> tuple[str, int]:
    qcfg = cfg.get("quantization_config") or {}
    return qcfg.get("quant_method", ""), int(qcfg.get("bits", 16))


def bytes_per_param(method: str, bits: int) -> float:
    """Bytes pro Parameter für Gewichts-Speicherabschätzung."""
    if method in ("gptq", "awq") and bits == 4:
        return 0.5
    if method in ("gptq", "awq") and bits == 8:
        return 1.0
    if method == "fp8":
        return 1.0
    if method in ("nvfp4", "mxfp4", "fp4", "modelopt"):
        return 0.5
    return 2.0   # BF16 / FP16


def estimate_model_gb(dir_name: str, bpp: float) -> float | None:
    """Schätzt Modellgröße in GiB aus dem Verzeichnisnamen."""
    # Muster: ...122B... oder ...120b...
    m = re.search(r"[-_](\d+(?:\.\d+)?)[Bb](?:$|[-_GPTQ])", dir_name)
    if m:
        return float(m.group(1)) * bpp
    return None


def kv_bytes_per_token_fp8(text_cfg: dict) -> int:
    """
    KV-Cache-Bytes pro Token bei FP8-KV (1 Byte/Element).

    Für Modelle mit Interleaved Sliding/Global Attention (z.B. Gemma 4):
    Nur Full-Attention-Layer skalieren linear mit der Kontextlänge.
    Sliding-Window-Layer sind durch die Fenstergröße begrenzt – ihr
    KV-Bedarf wächst NICHT mit max_model_len und wird hier ignoriert
    (der fixe Overhead ist im Vergleich vernachlässigbar).
    """
    layers   = text_cfg.get("num_hidden_layers", 32)
    kv_heads = text_cfg.get("num_key_value_heads",
                            text_cfg.get("num_attention_heads", 8))
    head_dim = text_cfg.get("head_dim", 128)

    # Interleaved Attention (Gemma 4): layer_types listet "full_attention" /
    # "sliding_attention" pro Layer. Nur Full-Attention-Layer skalieren
    # linear mit dem Kontext. Globale Layer haben ggf. eigene KV-Head/Dim-Werte.
    layer_types = text_cfg.get("layer_types")
    if layer_types:
        n_full          = sum(1 for t in layer_types if t == "full_attention")
        global_kv_heads = text_cfg.get("num_global_key_value_heads") or kv_heads
        global_head_dim = text_cfg.get("global_head_dim") or head_dim
        return int(n_full * 2 * global_kv_heads * global_head_dim * 1)

    return int(layers * 2 * kv_heads * head_dim * 1)  # FP8 = 1 Byte


def compute_max_model_len(kv_budget_gb: float,
                          kv_bpt: int,
                          native_max: int,
                          n_seqs: int,
                          hard_cap: int = 131072) -> int:
    if kv_bpt > 0 and kv_budget_gb > 0:
        total_tokens  = int(kv_budget_gb * 1024**3 / kv_bpt)
        per_seq       = total_tokens // n_seqs
    else:
        per_seq = hard_cap
    return max(4096, min(native_max, hard_cap, per_seq))


def round_to_power2(n: int) -> int:
    """Rundet auf die nächste Potenz von 2 ab (mindestens 4096)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return max(4096, p)


# ── Haupt-Profil-Berechnung ───────────────────────────────────────────────

def compute_profile(model_dir: str) -> dict:
    dir_name = os.path.basename(model_dir.rstrip("/"))

    # Empirisch validiertes Profil geht vor
    if dir_name in KNOWN_GOOD:
        return dict(KNOWN_GOOD[dir_name])

    cfg = load_config(model_dir)
    if cfg is None:
        return {
            "PROFILE_VLLM_COMPATIBLE": 0,
            "PROFILE_NOTES": "Kein config.json gefunden.",
        }

    arch = (cfg.get("architectures") or ["Unknown"])[0]

    if arch in UNSUPPORTED_ARCHS:
        return {
            "PROFILE_VLLM_COMPATIBLE": 0,
            "PROFILE_NOTES": f"Nicht von 'vllm serve' unterstützt (Arch: {arch}).",
        }

    text_cfg = get_text_cfg(cfg)
    method, bits = get_quant_info(cfg)
    bpp          = bytes_per_param(method, bits)
    hints        = ARCH_HINTS.get(arch, {})

    # ── Quantisierung / dtype ─────────────────────────────────────────────
    dtype        = "auto"
    quantization = ""
    kv_dtype     = "fp8"   # Standard: FP8-KV spart Speicher

    if method == "gptq" and bits == 4:
        dtype, quantization = "float16", "gptq_marlin"
    elif method == "gptq":
        dtype, quantization = "float16", "gptq"
    elif method == "fp8":
        dtype, quantization = "bfloat16", "fp8"
    elif method in ("nvfp4", "mxfp4", "fp4"):
        dtype = "bfloat16"
        # vLLM erkennt quantization aus der config.json selbst bei diesen Typen
    elif method == "modelopt":
        dtype = "bfloat16"
    else:
        dtype = "bfloat16"

    # ── Speicherabschätzung ───────────────────────────────────────────────
    model_gb    = estimate_model_gb(dir_name, bpp)
    effective_model_gb = model_gb if model_gb is not None else 20.0
    kv_budget   = max(4.0, AVAILABLE_GB - effective_model_gb - OVERHEAD_GB)

    # ── KV-Cache-Bytes pro Token ──────────────────────────────────────────
    kv_bpt      = kv_bytes_per_token_fp8(text_cfg)
    native_max  = text_cfg.get("max_position_embeddings", 32768)

    is_large    = effective_model_gb >= 55
    n_seqs      = 2 if is_large else TARGET_SEQS

    raw_len     = compute_max_model_len(kv_budget, kv_bpt, native_max, n_seqs)
    max_model_len = round_to_power2(raw_len)

    # ── GPU-Auslastung ────────────────────────────────────────────────────
    gpu_mem_util = "0.85"

    # ── enforce_eager ─────────────────────────────────────────────────────
    # is_moe: klassische Namenskonvention ("Moe"/"moe") ODER Gemma-4-Style
    # (text_config.enable_moe_block = true).
    is_moe        = ("Moe" in arch or "moe" in cfg.get("model_type", "")
                     or bool(text_cfg.get("enable_moe_block")))
    enforce_eager = 1 if (is_moe or is_large or hints.get("enforce_eager", 0)) else 0

    # ── trust_remote_code ─────────────────────────────────────────────────
    has_auto_map     = bool(cfg.get("auto_map"))
    trust_remote     = 1 if (has_auto_map or hints.get("trust_remote_code", 0)) else 0

    # ── tokenizer-mode: MistralCommon (Tekken) ────────────────────────────
    # Modelle mit tekken.json haben kein tokenizer_config.json. Ohne
    # --tokenizer-mode mistral scheitert vLLM mit AttributeError in
    # MistralCommonTokenizer (all_special_ids vs. _all_special_ids).
    tokenizer_mode = "mistral" if os.path.exists(
        os.path.join(model_dir, "tekken.json")
    ) else ""

    # ── HF-Overrides ──────────────────────────────────────────────────────
    hf_overrides = ""
    if hints.get("check_mtp"):
        mtp = cfg.get("num_nextn_predict_layers",
                      text_cfg.get("num_nextn_predict_layers", 0))
        if mtp and int(mtp) > 0:
            hf_overrides = '{"num_nextn_predict_layers": 0}'

    # ── Parser ────────────────────────────────────────────────────────────
    reasoning_parser        = hints.get("reasoning_parser", "")
    tool_call_parser        = hints.get("tool_call_parser", "")
    enable_auto_tool_choice = 1 if (hints.get("enable_auto_tool_choice")
                                    or tool_call_parser) else 0

    # ── num_gpu_blocks_override ───────────────────────────────────────────
    # Nur bei sehr großen MoE-Modellen, wo Profiling überschätzt (Encoder-Cache-Overhead)
    num_gpu_blocks_override = ""
    if hints.get("gpu_blocks_override_large_moe") and is_large:
        num_gpu_blocks_override = "512"

    # ── Notizen ───────────────────────────────────────────────────────────
    notes = (
        f"Auto-generiert. Arch={arch}. "
        f"Modellgröße~{effective_model_gb:.0f} GB (geschätzt). "
        f"KV-Budget~{kv_budget:.0f} GB → max_model_len={max_model_len} "
        f"bei {n_seqs} parallelen Seqs. "
        "Parameter nach erstem Start anhand von 'docker logs vllm-server' validieren."
    )

    # ── Profil zusammenbauen ──────────────────────────────────────────────
    p: dict = {
        "PROFILE_VLLM_COMPATIBLE": 1,
        "PROFILE_DTYPE":           dtype,
        "PROFILE_GPU_MEM_UTIL":    gpu_mem_util,
        "PROFILE_MAX_MODEL_LEN":   max_model_len,
        "PROFILE_MAX_NUM_SEQS":    n_seqs,
        "PROFILE_ENFORCE_EAGER":   enforce_eager,
        "PROFILE_TRUST_REMOTE_CODE": trust_remote,
        "PROFILE_KV_CACHE_DTYPE":  kv_dtype,
        "PROFILE_NOTES":           notes,
    }

    if quantization:
        p["PROFILE_QUANTIZATION"] = quantization
    if hf_overrides:
        p["PROFILE_HF_OVERRIDES"] = hf_overrides
    if reasoning_parser:
        p["PROFILE_REASONING_PARSER"] = reasoning_parser
    if tokenizer_mode:
        p["PROFILE_TOKENIZER_MODE"] = tokenizer_mode
    if tool_call_parser:
        p["PROFILE_TOOL_CALL_PARSER"] = tool_call_parser
    if enable_auto_tool_choice:
        p["PROFILE_ENABLE_AUTO_TOOL_CHOICE"] = enable_auto_tool_choice
    if num_gpu_blocks_override:
        p["PROFILE_NUM_GPU_BLOCKS_OVERRIDE"] = num_gpu_blocks_override

    return p


def write_profile(model_dir: str, profile: dict) -> str:
    profile_path = os.path.join(model_dir, "vllm_profile.conf")
    dir_name     = os.path.basename(model_dir.rstrip("/"))
    hf_handle    = dir_name.replace("--", "/", 1)

    lines = [
        "# vLLM Profil – auto-generiert von vllm_spark_profiler.py",
        f"# Modell  : {dir_name}",
        f"# HF-ID   : {hf_handle}",
        "# Manuell anpassen wenn nötig. Wird von vllm_spark.sh geladen.",
        "",
    ]

    # Felder in definierter Reihenfolge, Rest alphabetisch
    written = set()
    for key in KEY_ORDER:
        if key in profile:
            _append_field(lines, key, profile[key])
            written.add(key)
    for key in sorted(profile.keys()):
        if key not in written:
            _append_field(lines, key, profile[key])

    lines.append("")
    with open(profile_path, "w") as f:
        f.write("\n".join(lines))
    return profile_path


def _append_field(lines: list, key: str, val) -> None:
    if isinstance(val, str):
        escaped = val.replace("'", "'\\''")
        lines.append(f"{key}='{escaped}'")
    elif isinstance(val, bool):
        lines.append(f"{key}={1 if val else 0}")
    else:
        lines.append(f"{key}={val}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    model_dir = args[0]
    force     = "--force" in args

    if not os.path.isdir(model_dir):
        print(f"[ERR] Kein Verzeichnis: {model_dir}", file=sys.stderr)
        sys.exit(1)

    profile_path = os.path.join(model_dir, "vllm_profile.conf")

    if os.path.isfile(profile_path) and not force:
        print(f"[OK] Profil bereits vorhanden: {profile_path}  (--force zum Überschreiben)",
              file=sys.stderr)
        sys.exit(0)

    profile = compute_profile(model_dir)
    out     = write_profile(model_dir, profile)
    compat  = profile.get("PROFILE_VLLM_COMPATIBLE", 1)
    status  = "OK" if compat else "INKOMPATIBEL"
    print(f"[{status}] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
