#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-/data3/Qwen2.5-14B-Instruct-1M}"
MODEL_TAG="${MODEL_TAG:-qwen2.5-14b-1m}"
DATASET_STEM="${DATASET_STEM:-Qwen2.5-7B-Instruct-1M}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
RULER_TASK="${RULER_TASK:-auto}"
MAX_LENGTH_DEVIATION_RATIO="${MAX_LENGTH_DEVIATION_RATIO:--1}"
DATA_SAMPLE_SCAN_LIMIT="${DATA_SAMPLE_SCAN_LIMIT:-1}"
D_TYPE="${DTYPE:-bf16}"
GEN_LEN="${GEN_LEN:-50}"
PREFILL_BSZ="${PREFILL_BSZ:-1}"
PREFILL_METHOD="${PREFILL_METHOD:-full}"
SEQ_LIST="${SEQ_LIST:-4 8 16 32 64 128 256}" #4 8 16 32 64 128 256
BSZ_LIST="${BSZ_LIST:-1}"
TOPK="${TOPK:-${RETRIEVAL_BUDGET:-0.10}}"
RETRIEVAL_BUDGET="${RETRIEVAL_BUDGET:-${TOPK}}"
ESTIMATION_BUDGET="${ESTIMATION_BUDGET:-0.232}"
CACHE="${CACHE:-${CACHE_RATIO:-1}}"
CACHE_RATIO="${CACHE_RATIO:-${CACHE}}"
CACHE_POLICY="${CACHE_POLICY:-LRU}" # FIFO | LRU
USE_CUDA_GRAPH="${USE_CUDA_GRAPH:-1}"
GPU_ONLY="${GPU_ONLY:-0}"
RETRO_CACHE_STATS="${RETRO_CACHE_STATS:-1}"
OMP_THREADS="${OMP_THREADS:-64}"
RETROINFER_CORE="${RETROINFER_CORE:-22}"
GPU_MEM_MONITOR_INTERVAL_SEC="${GPU_MEM_MONITOR_INTERVAL_SEC:-0.5}"
USE_NUMACTL="${USE_NUMACTL:-1}"
NUMA_NODE="${NUMA_NODE:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

NSYS_PROFILE="${NSYS_PROFILE:-0}"
NSYS_BIN="${NSYS_BIN:-}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn}"
NSYS_EXPORT="${NSYS_EXPORT:-sqlite}"
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"
NSYS_CPUCTXSW="${NSYS_CPUCTXSW:-none}"
NSYS_CUDA_TRACE_SCOPE="${NSYS_CUDA_TRACE_SCOPE:-process-tree}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-node}"
NSYS_FORCE_OVERWRITE="${NSYS_FORCE_OVERWRITE:-true}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:-}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/Qwen2.5-14B-Instruct-1M-retroinfer-seqlen-logs-nsys${NSYS_PROFILE}-cache${CACHE}-policy${CACHE_POLICY}-cudagraph${USE_CUDA_GRAPH}}"
NSYS_HAS_SAMPLE=0
NSYS_HAS_CPUCTXSW=0
NSYS_HAS_CUDA_TRACE_SCOPE=0
NSYS_HAS_CUDA_GRAPH_TRACE=0
NSYS_HAS_FORCE_OVERWRITE=0
NSYS_HAS_EXPORT=0

if [[ -z "${NSYS_BIN}" && -x "/jhe/nsys-2025.6.1/pkg/bin/nsys" ]]; then
  NSYS_BIN="/jhe/nsys-2025.6.1/pkg/bin/nsys"
fi

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_nsys_bin() {
  if [[ -n "${NSYS_BIN}" ]]; then
    if [[ -x "${NSYS_BIN}" ]]; then
      echo "${NSYS_BIN}"
      return 0
    fi
    echo "[ERROR] NSYS_BIN is set but not executable: ${NSYS_BIN}" >&2
    return 1
  fi

  if command -v nsys >/dev/null 2>&1; then
    command -v nsys
    return 0
  fi

  local candidate
  for candidate in \
    /jhe/nsys-2025.6.1/pkg/bin/nsys \
    /jhe/nsys-2025.6.1/pkg/target-linux-x64/nsys \
    /opt/nvidia/nsight-systems/*/bin/nsys \
    /opt/nvidia/nsight-systems/*/target-linux-x64/nsys \
    /usr/local/cuda/bin/nsys; do
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  echo "[ERROR] nsys not found. Set NSYS_BIN=/path/to/nsys or add nsys to PATH." >&2
  return 1
}

detect_nsys_profile_features() {
  local help_text
  help_text="$("${RESOLVED_NSYS_BIN}" profile --help 2>&1 || true)"

  if [[ "${help_text}" == *"--sample"* ]]; then
    NSYS_HAS_SAMPLE=1
  fi
  if [[ "${help_text}" == *"--cpuctxsw"* ]]; then
    NSYS_HAS_CPUCTXSW=1
  fi
  if [[ "${help_text}" == *"--cuda-trace-scope"* ]]; then
    NSYS_HAS_CUDA_TRACE_SCOPE=1
  fi
  if [[ "${help_text}" == *"--cuda-graph-trace"* ]]; then
    NSYS_HAS_CUDA_GRAPH_TRACE=1
  fi
  if [[ "${help_text}" == *"--force-overwrite"* ]]; then
    NSYS_HAS_FORCE_OVERWRITE=1
  fi
  if [[ "${help_text}" == *"--export"* ]]; then
    NSYS_HAS_EXPORT=1
  fi
}

start_gpu_memory_monitor() {
  local gpu_index="$1"
  local output_csv="$2"
  local interval_sec="$3"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[GPU_MEM][WARN] nvidia-smi not found, skip GPU memory monitoring" >&2
    return 1
  fi

  (
    echo "timestamp_utc,memory_used_mib" > "${output_csv}"
    while true; do
      local ts
      local mem
      ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
      mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_index}" 2>/dev/null | head -n 1 | tr -d " " || true)"
      if [[ -n "${mem}" ]]; then
        echo "${ts},${mem}" >> "${output_csv}"
      fi
      sleep "${interval_sec}" || break
    done
  ) >/dev/null 2>&1 &
  gpu_mem_monitor_pid="$!"
}

stop_gpu_memory_monitor() {
  local monitor_pid="${1:-}"
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" >/dev/null 2>&1; then
    kill "${monitor_pid}" >/dev/null 2>&1 || true
    wait "${monitor_pid}" >/dev/null 2>&1 || true
  fi
}

gpu_mem_monitor_pid=""
trap 'stop_gpu_memory_monitor "${gpu_mem_monitor_pid:-}"' EXIT
trap 'stop_gpu_memory_monitor "${gpu_mem_monitor_pid:-}"; exit 130' INT TERM

append_decode_summary_outputs() {
  local log_file="$1"
  local summary_csv="$2"
  local summary_json_dir="$3"
  local run_name="$4"
  local seq_tokens="$5"
  local batch_size="$6"
  local cache_policy="$7"
  local gpu_mem_log_file="$8"
  "${PYTHON_BIN}" - "${log_file}" "${summary_csv}" "${summary_json_dir}" "${run_name}" "${seq_tokens}" "${batch_size}" "${cache_policy}" "${gpu_mem_log_file}" <<'PY'
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
summary_json_dir = Path(sys.argv[3])
run_name = sys.argv[4]
seq_tokens = int(sys.argv[5])
batch_size = int(sys.argv[6])
cache_policy = sys.argv[7]
gpu_mem_log_path = Path(sys.argv[8])

if not log_path.exists():
    print(f"[DECODE] missing log file: {log_path}")
    raise SystemExit(0)

ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
metric_re = re.compile(
    r"Decoding latency:\s*(?:(?:[0-9]*\.?[0-9]+)\s*s\s*\()?([0-9]*\.?[0-9]+)\s*ms/step\)?"
    r",\s*Throughput:\s*([0-9]*\.?[0-9]+)\s*tokens/s"
)
overall_cache_re = re.compile(
    r"Overall\s*\|\s*[^|]*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9]*\.?[0-9]+)%"
)

decode_ms = None
throughput = None
cache_hit_blocks = None
cache_miss_blocks = None
cache_update_blocks = None
cache_hit_ratio = None
peak_gpu_mem_mib = None
if gpu_mem_log_path.exists():
    with gpu_mem_log_path.open(encoding="utf-8", errors="ignore") as f:
        next(f, None)
        for raw_line in f:
            parts = raw_line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                mem_mib = int(float(parts[1]))
            except ValueError:
                continue
            peak_gpu_mem_mib = mem_mib if peak_gpu_mem_mib is None else max(peak_gpu_mem_mib, mem_mib)
for raw_line in reversed(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
    line = ansi_re.sub("", raw_line)
    if cache_hit_ratio is None:
        cm = overall_cache_re.search(line)
        if cm:
            cache_hit_blocks = int(cm.group(1))
            cache_miss_blocks = int(cm.group(2))
            cache_update_blocks = int(cm.group(3))
            cache_hit_ratio = float(cm.group(4)) / 100.0
    m = metric_re.search(line)
    if m:
        decode_ms = float(m.group(1))
        throughput = float(m.group(2))
        if cache_hit_ratio is not None:
            break

if decode_ms is None:
    print(f"[DECODE] run={run_name} decode_ms_per_step=NA (pattern not found)")
    raise SystemExit(0)

header = [
    "run_name",
    "seq_tokens",
    "batch_size",
    "cache_policy",
    "decode_ms_per_step",
    "decode_tokens_per_s",
    "cache_hit_blocks",
    "cache_miss_blocks",
    "cache_update_blocks",
    "cache_hit_ratio",
    "peak_gpu_mem_mib",
    "gpu_mem_log_file",
    "log_file",
]
row = [
    run_name,
    seq_tokens,
    batch_size,
    cache_policy,
    decode_ms,
    throughput,
    cache_hit_blocks if cache_hit_blocks is not None else "",
    cache_miss_blocks if cache_miss_blocks is not None else "",
    cache_update_blocks if cache_update_blocks is not None else "",
    cache_hit_ratio if cache_hit_ratio is not None else "",
    peak_gpu_mem_mib if peak_gpu_mem_mib is not None else "",
    str(gpu_mem_log_path) if gpu_mem_log_path.exists() else "",
    str(log_path),
]

summary_path.parent.mkdir(parents=True, exist_ok=True)
if summary_path.exists():
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        existing_rows = list(csv.reader(f))
    if existing_rows and existing_rows[0] != header:
        old_header = existing_rows[0]
        old_rows = existing_rows[1:]
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for old_row in old_rows:
                mapped = dict(zip(old_header, old_row))
                writer.writerow([mapped.get(col, "") for col in header])

write_header = not summary_path.exists() or summary_path.stat().st_size == 0
with summary_path.open("a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    writer.writerow(row)

summary_json_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "run_name": run_name,
    "seq_tokens": seq_tokens,
    "batch_size": batch_size,
    "cache_policy": cache_policy,
    "decode_ms_per_step": decode_ms,
    "decode_tokens_per_s": throughput,
    "cache_hit_blocks": cache_hit_blocks,
    "cache_miss_blocks": cache_miss_blocks,
    "cache_update_blocks": cache_update_blocks,
    "cache_hit_ratio": cache_hit_ratio,
    "peak_gpu_mem_mib": peak_gpu_mem_mib,
    "gpu_mem_log_file": str(gpu_mem_log_path) if gpu_mem_log_path.exists() else None,
    "log_file": str(log_path),
    "parsed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
run_json_path = summary_json_dir / f"{run_name}.json"
run_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

index_path = summary_json_dir / "index.json"
index_payload = {"runs": []}
if index_path.exists():
    try:
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
            index_payload = loaded
    except Exception:
        pass

runs = [r for r in index_payload.get("runs", []) if isinstance(r, dict) and r.get("run_name") != run_name]
runs.append(payload)
runs.sort(key=lambda x: (x.get("seq_tokens", 0), x.get("batch_size", 0), x.get("run_name", "")))
index_payload["runs"] = runs
index_payload["total_runs"] = len(runs)
index_payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

print(
    f"[DECODE] run={run_name} seq_tokens={seq_tokens} batch_size={batch_size} cache_policy={cache_policy} "
    f"decode_ms_per_step={decode_ms:.4f} decode_tokens_per_s={throughput:.4f} "
    f"cache_hit_ratio={(cache_hit_ratio if cache_hit_ratio is not None else float('nan')):.4f} "
    f"peak_gpu_mem_mib={(peak_gpu_mem_mib if peak_gpu_mem_mib is not None else 'NA')} "
    f"summary_csv={summary_path}"
)
print(
    f"[DECODE_JSON] run={run_name} json={run_json_path} index={index_path}"
)
PY
}

print_output_layout() {
  local base_dir="$1"
  echo "[OUT_DIR] base=${base_dir}"
  if [[ -d "${base_dir}" ]]; then
    find "${base_dir}" -maxdepth 3 -type f | sort
  fi
}

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

SEQ_LIST="${SEQ_LIST//,/ }"
BSZ_LIST="${BSZ_LIST//,/ }"
RESOLVED_NSYS_BIN=""
if is_truthy "${NSYS_PROFILE}"; then
  RESOLVED_NSYS_BIN="$(resolve_nsys_bin)"
  detect_nsys_profile_features
  echo "[NSYS] enabled bin=${RESOLVED_NSYS_BIN} output_root=${LOG_DIR}/seq*/nsys trace=${NSYS_TRACE} export=${NSYS_EXPORT}"
  if [[ "${NSYS_HAS_CUDA_TRACE_SCOPE}" != "1" ]]; then
    echo "[NSYS][WARN] this nsys version does not support --cuda-trace-scope, skip it"
  fi
  if [[ "${NSYS_HAS_CUDA_GRAPH_TRACE}" != "1" ]]; then
    echo "[NSYS][WARN] this nsys version does not support --cuda-graph-trace, CUDA graph kernels may appear folded"
  fi
fi

for seq in ${SEQ_LIST}; do
  if (( seq >= 1000 )); then
    seq_tokens="${seq}"
    seq_in_k=$((seq / 1000))
  else
    seq_in_k="${seq}"
    seq_tokens=$((seq * 1000))
  fi

  actual_ruler_task="${RULER_TASK}"
  actual_data_length=""
  legacy_data_file="${DATA_DIR}/RULER-${DATASET_STEM}-${seq_in_k}K.jsonl"
  if [[ -f "${legacy_data_file}" ]]; then
    data_file="${legacy_data_file}"
    actual_ruler_task="legacy"
    actual_data_length="$("${PYTHON_BIN}" - "${data_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        if line.strip():
            length = json.loads(line).get("length")
            print(length if length is not None else "")
            break
PY
)"
  elif [[ "${RULER_TASK}" == "auto" ]]; then
    read -r data_file actual_ruler_task actual_data_length < <("${PYTHON_BIN}" - "${DATA_DIR}" "${seq_in_k}" "${seq_tokens}" "${DATA_SAMPLE_SCAN_LIMIT}" <<'PY'
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
seq_in_k = sys.argv[2]
target = int(sys.argv[3])
scan_limit = int(sys.argv[4])
best = None
for path in sorted((data_dir / f"{seq_in_k}K").glob("*/validation.jsonl")):
    with path.open(encoding="utf-8") as f:
        scanned = 0
        for line in f:
            if not line.strip():
                continue
            scanned += 1
            length = json.loads(line).get("length")
            if not isinstance(length, (int, float)):
                continue
            score = abs(int(length) - target)
            if best is None or score < best[0]:
                best = (score, path, path.parent.name, int(length))
            if scan_limit > 0 and scanned >= scan_limit:
                break
if best is not None:
    print(best[1], best[2], best[3])
PY
    )
  else
    data_file="${DATA_DIR}/${seq_in_k}K/${actual_ruler_task}/validation.jsonl"
  fi
  if [[ ! -f "${data_file}" ]]; then
    echo "[WARN] data file not found: ${data_file}, skip"
    continue
  fi

  seq_output_dir="${LOG_DIR}/seq${seq_tokens}"
  seq_nsys_output_dir="${seq_output_dir}/nsys"
  seq_decode_summary_csv="${seq_output_dir}/decode_summary.csv"
  seq_decode_summary_json_dir="${seq_output_dir}/decode_summary_json"
  mkdir -p "${seq_output_dir}" "${seq_decode_summary_json_dir}"
  if is_truthy "${NSYS_PROFILE}"; then
    mkdir -p "${seq_nsys_output_dir}"
  fi

  for bsz in ${BSZ_LIST}; do
    run_name="${MODEL_TAG}-retroinfer-bsz${bsz}-seq${seq_tokens}-top${TOPK}-cache${CACHE}-policy${CACHE_POLICY}"
    if [[ -n "${RUN_SUFFIX}" ]]; then
      run_name="${run_name}-${RUN_SUFFIX}"
    fi
    log_file="${seq_output_dir}/${run_name}.log"
    gpu_mem_log_file="${seq_output_dir}/${run_name}.gpu_mem.csv"

    cmd=(
      "${PYTHON_BIN}" -u "${SCRIPT_DIR}/test.py"
      --model_name "${MODEL_NAME}"
      --attn_type RetroInfer
      --context_len "${seq_tokens}"
      --gen_len "${GEN_LEN}"
      --data "${data_file}"
      --batch_size "${bsz}"
      --prefill_bsz "${PREFILL_BSZ}"
      --prefill_method "${PREFILL_METHOD}"
      --dtype "${D_TYPE}"
      --retrieval_budget "${RETRIEVAL_BUDGET}"
      --estimation_budget "${ESTIMATION_BUDGET}"
      --cache_ratio "${CACHE_RATIO}"
      --cache_policy "${CACHE_POLICY}"
      --max_length_deviation_ratio "${MAX_LENGTH_DEVIATION_RATIO}"
      --data_sample_scan_limit "${DATA_SAMPLE_SCAN_LIMIT}"
    )
    if is_truthy "${USE_CUDA_GRAPH}"; then
      cmd+=(--use_cuda_graph)
    fi
    if is_truthy "${GPU_ONLY}"; then
      cmd+=(--gpu_only)
    fi

    launch_cmd=("${cmd[@]}")
    if [[ "${USE_NUMACTL}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
      launch_cmd=(
        numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}"
        "${cmd[@]}"
      )
    fi

    run_cmd=("${launch_cmd[@]}")
    if is_truthy "${NSYS_PROFILE}"; then
      nsys_output="${seq_nsys_output_dir}/${run_name}"
      nsys_cmd=(
        "${RESOLVED_NSYS_BIN}" profile
        "--trace=${NSYS_TRACE}"
        "-o" "${nsys_output}"
      )
      if [[ "${NSYS_HAS_SAMPLE}" == "1" ]]; then
        nsys_cmd+=("--sample=${NSYS_SAMPLE}")
      fi
      if [[ "${NSYS_HAS_CPUCTXSW}" == "1" ]]; then
        nsys_cmd+=("--cpuctxsw=${NSYS_CPUCTXSW}")
      fi
      if [[ "${NSYS_HAS_CUDA_TRACE_SCOPE}" == "1" ]]; then
        nsys_cmd+=("--cuda-trace-scope=${NSYS_CUDA_TRACE_SCOPE}")
      fi
      if [[ "${NSYS_HAS_CUDA_GRAPH_TRACE}" == "1" ]]; then
        nsys_cmd+=("--cuda-graph-trace=${NSYS_CUDA_GRAPH_TRACE}")
      fi
      if [[ "${NSYS_HAS_FORCE_OVERWRITE}" == "1" ]]; then
        nsys_cmd+=("--force-overwrite=${NSYS_FORCE_OVERWRITE}")
      fi
      if [[ -n "${NSYS_EXPORT}" && "${NSYS_HAS_EXPORT}" == "1" ]]; then
        nsys_cmd+=("--export=${NSYS_EXPORT}")
      fi
      if [[ -n "${NSYS_EXTRA_ARGS}" ]]; then
        read -r -a nsys_extra_args <<< "${NSYS_EXTRA_ARGS}"
        nsys_cmd+=("${nsys_extra_args[@]}")
      fi
      nsys_cmd+=("${launch_cmd[@]}")
      run_cmd=("${nsys_cmd[@]}")
      echo "[NSYS] ${run_name} -> ${nsys_output}.nsys-rep"
    fi

    echo "[GPU_MEM] ${run_name} sampling gpu=${CUDA_DEVICE} interval=${GPU_MEM_MONITOR_INTERVAL_SEC}s -> ${gpu_mem_log_file}"
    gpu_mem_monitor_pid=""
    if start_gpu_memory_monitor "${CUDA_DEVICE}" "${gpu_mem_log_file}" "${GPU_MEM_MONITOR_INTERVAL_SEC}"; then
      :
    else
      gpu_mem_monitor_pid=""
    fi

    echo "[RUN] ${run_name} data=${data_file} ruler_task=${actual_ruler_task} data_length=${actual_data_length:-unknown} sample_scan_limit=${DATA_SAMPLE_SCAN_LIMIT} max_len_dev=${MAX_LENGTH_DEVIATION_RATIO} dtype=${D_TYPE} gen_len=${GEN_LEN} prefill_bsz=${PREFILL_BSZ} prefill_method=${PREFILL_METHOD} retrieval_budget=${RETRIEVAL_BUDGET} estimation_budget=${ESTIMATION_BUDGET} cache_multiplier=${CACHE_RATIO} cache=${CACHE} cache_policy=${CACHE_POLICY} cuda_graph=${USE_CUDA_GRAPH} gpu_only=${GPU_ONLY} cache_stats=${RETRO_CACHE_STATS}"
    set +e
    RATIO="${TOPK}" CACHE="${CACHE}" RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_POLICY="${CACHE_POLICY}" RETRO_CACHE_STATS="${RETRO_CACHE_STATS}" OMP_NUM_THREADS="${OMP_THREADS}" \
      "${run_cmd[@]}" 2>&1 | tee "${log_file}"
    run_status=${PIPESTATUS[0]}
    stop_gpu_memory_monitor "${gpu_mem_monitor_pid}"
    set -e

    if [[ "${run_status}" -ne 0 ]]; then
      echo "[ERROR] run failed: ${run_name}" >&2
      exit "${run_status}"
    fi

    append_decode_summary_outputs "${log_file}" "${seq_decode_summary_csv}" "${seq_decode_summary_json_dir}" "${run_name}" "${seq_tokens}" "${bsz}" "${CACHE_POLICY}" "${gpu_mem_log_file}" | tee -a "${log_file}"
  done
done

print_output_layout "${LOG_DIR}"
