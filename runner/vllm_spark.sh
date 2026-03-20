#!/usr/bin/env bash
set -Eeuo pipefail

########################################
# vLLM (DGX Spark) – Dynamischer Model Picker + Runner
#
# Modelle werden dynamisch aus HF_MODELS_DIR gescannt.
# Pro Modell wird ein vllm_profile.conf generiert und gecacht.
# Container lädt Modelle direkt aus dem lokalen Verzeichnis (kein Hub-Download).
########################################

# ---------- Farben / Logging ----------
C_RESET="\033[0m"
C_GREEN="\033[0;32m"
C_RED="\033[0;31m"
C_YELLOW="\033[0;33m"

log()  { echo -e "$*"; }
info() { echo -e "${C_YELLOW}[..]${C_RESET} $*"; }
ok()   { echo -e "${C_GREEN}[OK]${C_RESET} $*"; }
warn() { echo -e "${C_YELLOW}[!!]${C_RESET} $*"; }
err()  { echo -e "${C_RED}[ERR]${C_RESET} $*"; }

on_err() {
  local exit_code=$?
  local line_no=${1:-"?"}
  err "Fehler in Zeile ${line_no} (Exit-Code: ${exit_code})."
  exit "${exit_code}"
}
trap 'on_err $LINENO' ERR

usage() {
  cat <<'USAGE'
Usage: vllm_spark.sh [OPTIONEN]

Optionen:
  --list                 Zeigt alle verfügbaren Modelle aus HF_MODELS_DIR.
  --model <pattern>      Wählt Modell per Pattern (case-insensitive, gegen Verz.-Name).
  --gen-profiles         Generiert vllm_profile.conf für alle Modelle ohne Profil, dann Exit.
  --regen-profile        Erzwingt Neuberechnung des Profils für das gewählte Modell.
  --skip-pull            Überspringt docker pull des vLLM-Images.
  --tail-logs, --tail    Tailed docker logs -f nach dem Start.
  -h, --help             Hilfe.

Umgebungsvariablen (Defaults in Klammern):
  HF_MODELS_DIR              ($HOME/hf_models)   Lokales HF-Modell-Verzeichnis.
  IMAGE_REPO                 (vllm/vllm-openai)
  DEFAULT_VLLM_TAG           (v0.17.1)
  LATEST_VLLM_VERSION        (leer -> DEFAULT_VLLM_TAG wird genutzt)
  CONTAINER_NAME             (vllm-server)
  HOST_PORT                  (8000)
  SHM_SIZE                   (10g)
  HF_TOKEN / HUGGING_FACE_HUB_TOKEN  (leer)
  VLLM_EXTRA_ARGS            (leer)  Zusätzliche vLLM Args (ans Ende gehängt)
  DOCKER_IPC_HOST            (0)     Wenn 1: docker run --ipc host

Beispiele:
  ./vllm_spark.sh                            # Interaktives Menü
  ./vllm_spark.sh --model qwen3.5-9b         # Pattern-Auswahl
  ./vllm_spark.sh --gen-profiles             # Alle Profile generieren
  ./vllm_spark.sh --model ministral-14b --regen-profile --tail
USAGE
}

# ---------- Defaults / Konfiguration ----------
HF_MODELS_DIR="${HF_MODELS_DIR:-${HOME}/hf_models}"
PROFILER_SCRIPT="$(dirname "$(realpath "$0")")/vllm_spark_profiler.py"

# Offizielles vLLM-Image (Docker Hub). Tag = feste Version; kein Auto-Update.
IMAGE_REPO="${IMAGE_REPO:-vllm/vllm-openai}"
DEFAULT_VLLM_TAG="${DEFAULT_VLLM_TAG:-v0.17.1}"

CONTAINER_NAME="${CONTAINER_NAME:-vllm-server}"
HOST_PORT="${HOST_PORT:-8000}"
SHM_SIZE="${SHM_SIZE:-10g}"

HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

SKIP_PULL="${SKIP_PULL:-0}"
TAIL_LOGS="${TAIL_LOGS:-0}"
DOCKER_IPC_HOST="${DOCKER_IPC_HOST:-0}"

# ---------- Globale Variablen (werden in den Stages gesetzt) ----------
MODELS=()
MODEL_LABEL=""
MODEL_HANDLE=""   # Container-interner Pfad: /hf_models/<dir_name>
MODEL_DIR=""      # Host-seitiger Pfad: HF_MODELS_DIR/<dir_name>
VLLM_TAG=""

# ---------- Args parsing ----------
SELECT_PATTERN=""
LIST_ONLY=0
GEN_PROFILES_ONLY=0
REGEN_PROFILE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)          LIST_ONLY=1; shift ;;
    --gen-profiles)  GEN_PROFILES_ONLY=1; shift ;;
    --regen-profile) REGEN_PROFILE=1; shift ;;
    --model)
      SELECT_PATTERN="${2:-}"
      [[ -n "$SELECT_PATTERN" ]] || { err "--model benötigt ein Pattern."; exit 2; }
      shift 2
      ;;
    --skip-pull) SKIP_PULL=1; shift ;;
    --tail-logs|--tail) TAIL_LOGS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      err "Unbekannte Option: $1"
      usage
      exit 2
      ;;
  esac
done

# ---------- Helpers ----------
trim() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' <<<"$1"; }

split_model_entry() {
  # Eingabe: "label|handle"
  MODEL_LABEL="$(cut -d'|' -f1 <<<"$1" | sed 's/[[:space:]]*$//')"
  MODEL_HANDLE="$(cut -d'|' -f2 <<<"$1" | sed 's/^[[:space:]]*//')"
}

# ---------- Modell-Verzeichnis aus Label ableiten ----------
model_host_dir() {
  # Erwartet MODEL_LABEL gesetzt
  echo "${HF_MODELS_DIR}/${MODEL_LABEL}"
}

# ---------- Profile generieren ----------
ensure_profile() {
  local dir="$1"
  local force="${2:-0}"
  local profile="${dir}/vllm_profile.conf"

  if [[ ! -f "${PROFILER_SCRIPT}" ]]; then
    warn "Profiler nicht gefunden: ${PROFILER_SCRIPT}"
    warn "Bitte sicherstellen, dass vllm_spark_profiler.py neben vllm_spark.sh liegt."
    return 1
  fi

  if [[ -f "${profile}" && "${force}" == "0" ]]; then
    return 0
  fi

  local force_flag=""
  [[ "${force}" == "1" ]] && force_flag="--force"

  # shellcheck disable=SC2086
  if python3 "${PROFILER_SCRIPT}" "${dir}" ${force_flag}; then
    return 0
  else
    warn "Profil-Generierung fehlgeschlagen für: $(basename "${dir}")"
    return 1
  fi
}

# ---------- Stage 0: Prereqs ----------
stage_prereqs() {
  info "Stage 0: Prüfe Voraussetzungen ..."
  command -v docker   >/dev/null 2>&1 || { err "docker fehlt im PATH."; exit 1; }
  command -v python3  >/dev/null 2>&1 || { err "python3 fehlt im PATH."; exit 1; }

  [[ -d "${HF_MODELS_DIR}" ]] || {
    err "HF_MODELS_DIR nicht gefunden: ${HF_MODELS_DIR}"
    exit 1
  }
  [[ -f "${PROFILER_SCRIPT}" ]] || {
    err "Profiler nicht gefunden: ${PROFILER_SCRIPT}"
    err "Stelle sicher, dass vllm_spark_profiler.py neben vllm_spark.sh liegt."
    exit 1
  }
  ok "Voraussetzungen OK. HF_MODELS_DIR=${HF_MODELS_DIR}"
}

# ---------- Stage 0b: Modelle scannen ----------
scan_hf_models() {
  MODELS=()
  local dir dir_name profile compat

  for dir in "${HF_MODELS_DIR}"/*/; do
    [[ -d "${dir}" ]] || continue
    dir_name="$(basename "${dir}")"

    # Nur Einträge mit config.json (echte Modell-Verzeichnisse)
    [[ -f "${dir}/config.json" ]] || continue

    # Profil generieren falls fehlend (still, nur stderr)
    ensure_profile "${dir}" 0 || true

    # Kompatibilität aus Profil prüfen
    profile="${dir}/vllm_profile.conf"
    if [[ -f "${profile}" ]]; then
      compat="$(grep '^PROFILE_VLLM_COMPATIBLE=' "${profile}" 2>/dev/null \
                | cut -d= -f2 | tr -d "'" || echo "1")"
      [[ "${compat}" == "0" ]] && continue
    fi

    # Label = Verzeichnisname, Handle = Container-Pfad
    MODELS+=("${dir_name}|/hf_models/${dir_name}")
  done

  if (( ${#MODELS[@]} == 0 )); then
    err "Keine vLLM-kompatiblen Modelle in ${HF_MODELS_DIR} gefunden."
    err "Führe '--gen-profiles' aus und prüfe vllm_profile.conf-Dateien."
    exit 1
  fi
}

# ---------- Stage 0c: Alle Profile generieren ----------
stage_gen_profiles() {
  info "Generiere Profile für alle Modelle in ${HF_MODELS_DIR} ..."
  local dir dir_name count=0 skipped=0

  for dir in "${HF_MODELS_DIR}"/*/; do
    [[ -d "${dir}" ]] || continue
    dir_name="$(basename "${dir}")"
    [[ -f "${dir}/config.json" ]] || continue

    local profile="${dir}/vllm_profile.conf"
    if [[ -f "${profile}" ]]; then
      info "  Überspringe (vorhanden): ${dir_name}"
      skipped=$((skipped+1))
      continue
    fi

    info "  Generiere: ${dir_name}"
    if python3 "${PROFILER_SCRIPT}" "${dir}"; then
      count=$((count+1))
    else
      warn "  Fehlgeschlagen: ${dir_name}"
    fi
  done

  ok "Profile generiert: ${count} neu, ${skipped} bereits vorhanden."
  log ""
  log "Tipp: Profile manuell anpassen in ${HF_MODELS_DIR}/<Modell>/vllm_profile.conf"
  log "      '--force' zum Überschreiben: python3 vllm_spark_profiler.py <dir> --force"
}

print_models() {
  log "Verfügbare Modelle (aus ${HF_MODELS_DIR}):"
  local i=1
  for e in "${MODELS[@]}"; do
    split_model_entry "${e}"
    local hf_id="${MODEL_LABEL/--//}"
    local profile="${HF_MODELS_DIR}/${MODEL_LABEL}/vllm_profile.conf"
    local ctx_info=""
    if [[ -f "${profile}" ]]; then
      local len
      len="$(grep '^PROFILE_MAX_MODEL_LEN=' "${profile}" 2>/dev/null \
             | cut -d= -f2 | tr -d "'" || echo "")"
      [[ -n "${len}" ]] && ctx_info=" [ctx ${len}]"
    fi
    printf "  %2d) %-52s %s%s\n" "$i" "${MODEL_LABEL}" "${hf_id}" "${ctx_info}"
    i=$((i+1))
  done
}

# ---------- Stage 1: Modell wählen ----------
stage_model_pick() {
  info "Stage 1: Modell-Auswahl ..."
  log ""

  # Modelle scannen (generiert ggf. fehlende Profile)
  scan_hf_models

  if (( LIST_ONLY == 1 )); then
    print_models
    exit 0
  fi

  local count="${#MODELS[@]}"

  # Nicht-interaktive Auswahl per Pattern
  if [[ -n "${SELECT_PATTERN}" ]]; then
    local pat_lc matches=()
    pat_lc="$(echo "${SELECT_PATTERN}" | tr '[:upper:]' '[:lower:]')"

    for e in "${MODELS[@]}"; do
      split_model_entry "${e}"
      local hay_lc
      hay_lc="$(echo "${MODEL_LABEL}" | tr '[:upper:]' '[:lower:]')"
      [[ "${hay_lc}" == *"${pat_lc}"* ]] && matches+=("${e}")
    done

    if (( ${#matches[@]} == 0 )); then
      err "Kein Modell matcht Pattern: ${SELECT_PATTERN}"
      print_models
      exit 1
    fi

    if (( ${#matches[@]} == 1 )); then
      split_model_entry "${matches[0]}"
      ok "Gewählt (Pattern): ${MODEL_LABEL}"
      MODEL_DIR="$(model_host_dir)"
      return 0
    fi

    warn "Pattern matcht mehrere Modelle – interaktive Auswahl:"
    MODELS=("${matches[@]}")
    count="${#MODELS[@]}"
  fi

  # Interaktives Menü
  print_models
  log ""
  PS3="Bitte Nummer wählen (1-${count}): "
  local options=()
  for e in "${MODELS[@]}"; do
    split_model_entry "${e}"
    options+=("${MODEL_LABEL}")
  done

  select choice in "${options[@]}"; do
    if [[ -n "${choice:-}" ]]; then
      local idx=$((REPLY-1))
      split_model_entry "${MODELS[$idx]}"
      break
    fi
    warn "Ungültige Auswahl."
  done

  MODEL_DIR="$(model_host_dir)"
  ok "Gewählt: ${MODEL_LABEL}  →  ${MODEL_HANDLE}"
}

# ---------- Stage 2: vLLM Container-Tag ----------
resolve_vllm_tag() {
  # Priorität: Env-Override → Default-Tag (feste Version)
  echo "${LATEST_VLLM_VERSION:-${DEFAULT_VLLM_TAG}}"
}

stage_pull_vllm() {
  info "Stage 2: Ermittle vLLM Container-Tag ..."
  VLLM_TAG="$(resolve_vllm_tag)"
  ok "Verwende vLLM Tag: ${VLLM_TAG}"

  if (( SKIP_PULL == 1 )); then
    warn "SKIP_PULL=1 → Überspringe docker pull."
    return 0
  fi

  info "docker pull ${IMAGE_REPO}:${VLLM_TAG}"
  if docker pull "${IMAGE_REPO}:${VLLM_TAG}"; then
    ok "Container gezogen."
  else
    err "docker pull fehlgeschlagen."
    log "  → NGC Login: docker login nvcr.io  (Username: \$oauthtoken)"
    exit 1
  fi
}

# ---------- Stage 3: Modell-Dateien prüfen ----------
stage_verify_model() {
  info "Stage 3: Prüfe lokales Modell-Verzeichnis ..."
  MODEL_DIR="${HF_MODELS_DIR}/${MODEL_LABEL}"

  [[ -d "${MODEL_DIR}" ]] || {
    err "Modell-Verzeichnis nicht gefunden: ${MODEL_DIR}"
    exit 1
  }
  [[ -f "${MODEL_DIR}/config.json" ]] || {
    err "config.json fehlt in: ${MODEL_DIR}"
    exit 1
  }

  # Gewichte prüfen (mindestens eine .safetensors oder .bin Datei)
  local weight_count
  weight_count="$(find "${MODEL_DIR}" -maxdepth 1 \
    \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | wc -l)"
  if (( weight_count == 0 )); then
    warn "Keine Gewichtsdateien (.safetensors / .bin) in ${MODEL_DIR}."
    warn "Das Modell wurde möglicherweise noch nicht vollständig heruntergeladen."
  else
    ok "Modell-Verzeichnis OK: ${weight_count} Gewichtsdatei(en) gefunden."
  fi
}

# ---------- Stage 4: Profil laden + vLLM-Args aufbauen ----------
VLLM_HELP_CACHE=""
VLLM_HELP_CACHE_IMAGE=""

fetch_vllm_help_once() {
  # Re-fetch if a different image is now active (e.g. per-model custom image)
  local image="${EFFECTIVE_IMAGE:-${IMAGE_REPO}:${VLLM_TAG}}"
  [[ -n "${VLLM_HELP_CACHE}" && "${image}" == "${VLLM_HELP_CACHE_IMAGE}" ]] && return 0
  VLLM_HELP_CACHE_IMAGE="${image}"
  if (( BASH_WRAPPER == 1 )); then
    # Image uses plain bash entrypoint — query help via bash wrapper
    VLLM_HELP_CACHE="$(docker run --rm --gpus all --entrypoint "" "${image}" \
      /bin/bash -lc "vllm serve --help=all" 2>&1 || true)"
  else
    # Standard vllm/vllm-openai image: ENTRYPOINT is already "vllm serve"
    VLLM_HELP_CACHE="$(docker run --rm --gpus all "${image}" --help=all 2>&1 || true)"
  fi
}

vllm_supports() {
  fetch_vllm_help_once
  grep -qF -- "$1" <<<"${VLLM_HELP_CACHE}"
}

apply_model_profile() {
  # Profil ggf. (neu) generieren
  if (( REGEN_PROFILE == 1 )); then
    info "Regeneriere Profil für ${MODEL_LABEL} ..."
    python3 "${PROFILER_SCRIPT}" "${MODEL_DIR}" --force
  fi
  ensure_profile "${MODEL_DIR}" 0

  local profile="${MODEL_DIR}/vllm_profile.conf"
  [[ -f "${profile}" ]] || {
    err "Kein Profil vorhanden: ${profile}"
    exit 1
  }

  # Profil-Variablen in lokalen Scope laden
  # (Präfix PROFILE_ verhindert Kollisionen mit globalen Vars)
  local PROFILE_VLLM_COMPATIBLE=1
  local PROFILE_DTYPE=""
  local PROFILE_QUANTIZATION=""
  local PROFILE_GPU_MEM_UTIL="0.85"
  local PROFILE_MAX_MODEL_LEN=32768
  local PROFILE_MAX_NUM_SEQS=4
  local PROFILE_MAX_NUM_BATCHED_TOKENS=""
  local PROFILE_ENFORCE_EAGER=0
  local PROFILE_NUM_GPU_BLOCKS_OVERRIDE=""
  local PROFILE_TRUST_REMOTE_CODE=0
  local PROFILE_KV_CACHE_DTYPE="fp8"
  local PROFILE_HF_OVERRIDES=""
  local PROFILE_REASONING_PARSER=""
  local PROFILE_TOOL_CALL_PARSER=""
  local PROFILE_ENABLE_AUTO_TOOL_CHOICE=0
  local PROFILE_ATTENTION_BACKEND=""
  local PROFILE_REASONING_PARSER_PLUGIN=""
  local PROFILE_CHAT_TEMPLATE=""
  local PROFILE_TOKENIZER_MODE=""   # e.g. "mistral"
  local PROFILE_CONFIG_FORMAT=""    # e.g. "mistral"
  local PROFILE_LOAD_FORMAT=""      # e.g. "mistral"
  local PROFILE_DOCKER_IMAGE=""     # override image (e.g. custom build with sm_120 kernels)
  local PROFILE_BASH_WRAPPER=0     # 1 = wrap "vllm serve" in bash -lc (non-standard entrypoint)
  local PROFILE_IPC_HOST=0         # 1 = use --ipc=host instead of --shm-size
  local PROFILE_DOCKER_ENV=""      # space-separated KEY=VALUE pairs to inject into container
  local PROFILE_NOTES=""
  # shellcheck source=/dev/null
  source "${profile}"

  if [[ "${PROFILE_VLLM_COMPATIBLE}" == "0" ]]; then
    err "Modell ist laut Profil nicht vLLM-kompatibel."
    err "  Hinweis: ${PROFILE_NOTES}"
    exit 1
  fi

  [[ -n "${PROFILE_NOTES}" ]] && info "Profil-Notiz: ${PROFILE_NOTES}"

  # Effektives Docker-Image bestimmen (Profil kann Standard überschreiben)
  EFFECTIVE_IMAGE="${PROFILE_DOCKER_IMAGE:-${IMAGE_REPO}:${VLLM_TAG}}"
  BASH_WRAPPER="${PROFILE_BASH_WRAPPER}"
  [[ "${EFFECTIVE_IMAGE}" != "${IMAGE_REPO}:${VLLM_TAG}" ]] && \
    info "  Image   : ${EFFECTIVE_IMAGE} (custom)"

  # ── Argumente aufbauen ──────────────────────────────────────────────────
  MODEL_VLLM_ARGS=()
  TRUST_REMOTE_CODE=""
  DOCKER_EXTRA_ENV=()
  DOCKER_EXTRA_ARGS=()

  # dtype
  if [[ -n "${PROFILE_DTYPE}" ]] && vllm_supports "--dtype"; then
    MODEL_VLLM_ARGS+=(--dtype "${PROFILE_DTYPE}")
  fi

  # quantization
  if [[ -n "${PROFILE_QUANTIZATION}" ]] && vllm_supports "--quantization"; then
    MODEL_VLLM_ARGS+=(--quantization "${PROFILE_QUANTIZATION}")
  fi

  # gpu-memory-utilization
  if vllm_supports "--gpu-memory-utilization"; then
    MODEL_VLLM_ARGS+=(--gpu-memory-utilization "${PROFILE_GPU_MEM_UTIL}")
  fi

  # max-model-len
  if vllm_supports "--max-model-len"; then
    MODEL_VLLM_ARGS+=(--max-model-len "${PROFILE_MAX_MODEL_LEN}")
  fi

  # max-num-seqs
  if vllm_supports "--max-num-seqs"; then
    MODEL_VLLM_ARGS+=(--max-num-seqs "${PROFILE_MAX_NUM_SEQS}")
  fi

  # max-num-batched-tokens
  if [[ -n "${PROFILE_MAX_NUM_BATCHED_TOKENS}" ]] \
       && vllm_supports "--max-num-batched-tokens"; then
    MODEL_VLLM_ARGS+=(--max-num-batched-tokens "${PROFILE_MAX_NUM_BATCHED_TOKENS}")
  fi

  # enforce-eager
  if (( PROFILE_ENFORCE_EAGER == 1 )) && vllm_supports "--enforce-eager"; then
    MODEL_VLLM_ARGS+=(--enforce-eager)
  fi

  # num-gpu-blocks-override
  if [[ -n "${PROFILE_NUM_GPU_BLOCKS_OVERRIDE}" ]] \
       && vllm_supports "--num-gpu-blocks-override"; then
    MODEL_VLLM_ARGS+=(--num-gpu-blocks-override "${PROFILE_NUM_GPU_BLOCKS_OVERRIDE}")
  fi

  # kv-cache-dtype
  if [[ -n "${PROFILE_KV_CACHE_DTYPE}" ]] && vllm_supports "--kv-cache-dtype"; then
    MODEL_VLLM_ARGS+=(--kv-cache-dtype "${PROFILE_KV_CACHE_DTYPE}")
  fi

  # prefix caching + chunked prefill (allgemein sinnvoll)
  if vllm_supports "--enable-prefix-caching"; then
    MODEL_VLLM_ARGS+=(--enable-prefix-caching)
  fi
  if vllm_supports "--enable-chunked-prefill"; then
    MODEL_VLLM_ARGS+=(--enable-chunked-prefill)
  fi

  # hf-overrides
  if [[ -n "${PROFILE_HF_OVERRIDES}" ]] && vllm_supports "--hf-overrides"; then
    MODEL_VLLM_ARGS+=(--hf-overrides "${PROFILE_HF_OVERRIDES}")
  fi

  # attention-backend
  if [[ -n "${PROFILE_ATTENTION_BACKEND}" ]] && vllm_supports "--attention-backend"; then
    MODEL_VLLM_ARGS+=(--attention-backend "${PROFILE_ATTENTION_BACKEND}")
  fi

  # reasoning-parser + optional plugin
  if [[ -n "${PROFILE_REASONING_PARSER}" ]] && vllm_supports "--reasoning-parser"; then
    MODEL_VLLM_ARGS+=(--reasoning-parser "${PROFILE_REASONING_PARSER}")
  fi
  if [[ -n "${PROFILE_REASONING_PARSER_PLUGIN}" ]] && vllm_supports "--reasoning-parser-plugin"; then
    MODEL_VLLM_ARGS+=(--reasoning-parser-plugin "${PROFILE_REASONING_PARSER_PLUGIN}")
  fi
  if [[ -n "${PROFILE_CHAT_TEMPLATE}" ]] && vllm_supports "--chat-template"; then
    MODEL_VLLM_ARGS+=(--chat-template "${PROFILE_CHAT_TEMPLATE}")
  fi

  # tool-call-parser + auto-tool-choice
  if (( PROFILE_ENABLE_AUTO_TOOL_CHOICE == 1 )) \
       && vllm_supports "--enable-auto-tool-choice"; then
    MODEL_VLLM_ARGS+=(--enable-auto-tool-choice)
  fi
  if [[ -n "${PROFILE_TOOL_CALL_PARSER}" ]] && vllm_supports "--tool-call-parser"; then
    MODEL_VLLM_ARGS+=(--tool-call-parser "${PROFILE_TOOL_CALL_PARSER}")
  fi

  # trust-remote-code
  if (( PROFILE_TRUST_REMOTE_CODE == 1 )); then
    TRUST_REMOTE_CODE="--trust-remote-code"
    warn "Setze automatisch: --trust-remote-code (Modell benötigt custom code)"
  fi

  # tokenizer-mode / config-format / load-format (e.g. "mistral")
  if [[ -n "${PROFILE_TOKENIZER_MODE}" ]] && vllm_supports "--tokenizer-mode"; then
    MODEL_VLLM_ARGS+=(--tokenizer-mode "${PROFILE_TOKENIZER_MODE}")
  fi
  if [[ -n "${PROFILE_CONFIG_FORMAT}" ]] && vllm_supports "--config-format"; then
    MODEL_VLLM_ARGS+=(--config-format "${PROFILE_CONFIG_FORMAT}")
  fi
  if [[ -n "${PROFILE_LOAD_FORMAT}" ]] && vllm_supports "--load-format"; then
    MODEL_VLLM_ARGS+=(--load-format "${PROFILE_LOAD_FORMAT}")
  fi

  # IPC host: either from profile or global DOCKER_IPC_HOST env
  # --ipc=host and --shm-size are mutually exclusive in Docker
  if (( PROFILE_IPC_HOST == 1 || DOCKER_IPC_HOST == 1 )); then
    DOCKER_EXTRA_ARGS+=(--ipc host)
  fi

  # Extra Docker-Env-Variablen aus Profil (z.B. VLLM_NVFP4_GEMM_BACKEND=cutlass)
  if [[ -n "${PROFILE_DOCKER_ENV}" ]]; then
    for kv in ${PROFILE_DOCKER_ENV}; do
      DOCKER_EXTRA_ENV+=(--env "${kv}")
    done
  fi
}

# ---------- Stage 4: vLLM starten ----------
stage_run_vllm() {
  info "Stage 4: Starte vLLM ..."

  # Bestehenden Container entfernen
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    warn "Container '${CONTAINER_NAME}' existiert → wird entfernt."
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi

  # Profil anwenden (sets EFFECTIVE_IMAGE + BASH_WRAPPER), then cache help for that image
  apply_model_profile
  fetch_vllm_help_once

  # Runtime-Cache für vLLM-interne Daten (Tokenizer-Cache etc.)
  local runtime_cache="${HOME}/.cache/vllm-runtime/${MODEL_LABEL}"
  mkdir -p "${runtime_cache}"

  # Docker-Envs
  local DOCKER_ENV=()
  if [[ -n "${HF_TOKEN}" ]]; then
    DOCKER_ENV+=(--env "HF_TOKEN=${HF_TOKEN}")
    DOCKER_ENV+=(--env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
  fi
  # Transformers offline → keine HF-Netzwerkanfragen (Modell ist lokal)
  DOCKER_ENV+=(--env "TRANSFORMERS_OFFLINE=1")
  DOCKER_ENV+=(--env "HF_DATASETS_OFFLINE=1")

  # Proxy-Envs durchreichen
  for p in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    if [[ -n "${!p:-}" ]]; then
      DOCKER_ENV+=(--env "${p}=${!p}")
    fi
  done

  info "Starte Container '${CONTAINER_NAME}' auf Host-Port ${HOST_PORT} ..."
  info "  Modell  : ${MODEL_DIR}  →  /hf_models/${MODEL_LABEL}"
  info "  Profile : ${MODEL_DIR}/vllm_profile.conf"

  # CMD-Args für vllm serve
  local vllm_args=()
  vllm_args+=("${MODEL_HANDLE}")
  [[ -n "${TRUST_REMOTE_CODE}" ]] && vllm_args+=("${TRUST_REMOTE_CODE}")
  vllm_args+=("${MODEL_VLLM_ARGS[@]}")
  if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    read -ra _extra <<< "${VLLM_EXTRA_ARGS}"
    vllm_args+=("${_extra[@]}")
  fi

  # Bash-wrapper: für Images ohne "vllm serve"-Entrypoint (z.B. avarok/dgx-vllm-nvfp4-kernel)
  local entrypoint_args=()
  local vllm_cmd=()
  if (( BASH_WRAPPER == 1 )); then
    entrypoint_args=(--entrypoint "")
    # shellcheck disable=SC2145
    vllm_cmd=(/bin/bash -lc "vllm serve ${vllm_args[*]@Q}" )
  else
    vllm_cmd=("${vllm_args[@]}")
  fi

  # --shm-size und --ipc=host schließen sich gegenseitig aus
  local shm_arg=()
  local ipc_already=0
  for a in "${DOCKER_EXTRA_ARGS[@]+"${DOCKER_EXTRA_ARGS[@]}"}"; do
    [[ "${a}" == "host" || "${a}" == "--ipc" ]] && ipc_already=1
  done
  (( ipc_already == 0 )) && shm_arg=(--shm-size "${SHM_SIZE}")

  docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -p "${HOST_PORT}:8000" \
    "${shm_arg[@]+"${shm_arg[@]}"}" \
    -v "${HF_MODELS_DIR}:/hf_models:ro" \
    -v "${runtime_cache}:/root/.cache/huggingface" \
    "${entrypoint_args[@]+"${entrypoint_args[@]}"}" \
    "${DOCKER_EXTRA_ARGS[@]+"${DOCKER_EXTRA_ARGS[@]}"}" \
    "${DOCKER_ENV[@]}" \
    "${DOCKER_EXTRA_ENV[@]+"${DOCKER_EXTRA_ENV[@]}"}" \
    "${EFFECTIVE_IMAGE}" \
    "${vllm_cmd[@]}"

  ok "vLLM gestartet."
  log ""
  log "Nützliche Befehle:"
  log "  Logs  : docker logs -f ${CONTAINER_NAME}"
  log "  Stop  : docker rm -f ${CONTAINER_NAME}"
  log "  Test  : curl http://127.0.0.1:${HOST_PORT}/v1/models"
  log ""

  if (( TAIL_LOGS == 1 )); then
    info "Tailing Logs (Strg+C beendet Tail, Container läuft weiter) ..."
    docker logs -f "${CONTAINER_NAME}"
  fi
}

# ---------- main ----------
main() {
  stage_prereqs

  # Modus: Nur Profile generieren
  if (( GEN_PROFILES_ONLY == 1 )); then
    stage_gen_profiles
    exit 0
  fi

  stage_model_pick
  stage_pull_vllm
  stage_verify_model
  stage_run_vllm
  ok "Fertig."
}

main "$@"
