#!/usr/bin/env bash
# set -euo pipefail

# Match the old RetrievalAttention_eval throughput semantics:
# retrieve 10% clusters plus the fixed 23.2% estimation zone.
# Override these from the environment if you want a different sparsity setting.
export TOPK="${TOPK:-0.10}"
export RETRIEVAL_BUDGET="${RETRIEVAL_BUDGET:-${TOPK}}"
export ESTIMATION_BUDGET="${ESTIMATION_BUDGET:-0.232}"
export RETROINFER_CORE="${RETROINFER_CORE:-22}"

run_experiment() {
    local name="$1"
    shift

    echo "[RUN] ${name}"
    "$@" || echo "[WARN] experiment failed, continue: ${name}"
}

# # llama Cache=1 用作bg 
# run_experiment "llama seqlen nsys0 cache1 LRU" env NSYS_PROFILE=0 CACHE=1 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
# run_experiment "llama seqlen nsys1 cache1 LRU" env NSYS_PROFILE=1 CACHE=1 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh

# run_experiment "llama seqlen nsys0 cache1 FIFO" env NSYS_PROFILE=0 CACHE=1 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh
# run_experiment "llama seqlen nsys1 cache1 FIFO" env NSYS_PROFILE=1 CACHE=1 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh

# llama Cache=3 用作bg
run_experiment "llama seqlen nsys0 cache3 LRU" env NSYS_PROFILE=0 CACHE=3 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
run_experiment "llama seqlen nsys1 cache3 LRU" env NSYS_PROFILE=1 CACHE=3 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh

run_experiment "llama seqlen nsys0 cache3 FIFO" env NSYS_PROFILE=0 CACHE=3 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh
run_experiment "llama seqlen nsys1 cache3 FIFO" env NSYS_PROFILE=1 CACHE=3 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh




# #llama用作eval seq 需要nsys LRU
# run_experiment "llama seqlen nsys0 cache3 LRU" env NSYS_PROFILE=0 CACHE=3 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
# run_experiment "llama seqlen nsys0 cache3 LRU" env NSYS_PROFILE=1 CACHE=3 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh


# #qwen用作eval seq 需要nsys LRU
# run_experiment "qwen seqlen nsys0 cache3 LRU" env NSYS_PROFILE=0 CACHE=3 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU  bash ./run_n2n_qwen_retroinfer_seqlen.sh
# run_experiment "qwen seqlen nsys1 cache3 LRU" env NSYS_PROFILE=1 CACHE=3 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU  bash ./run_n2n_qwen_retroinfer_seqlen.sh




# #llama用作eval batchsize 不需要nsys
# run_experiment "llama batchsize nsys0 cache3 LRU" env NSYS_PROFILE=0 CACHE=3 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU  bash ./run_n2n_llama_retroinfer_batchsize.sh
# #qwen用作eval batchsize 不需要nsys
# run_experiment "qwen batchsize nsys0 cache3 LRU" env NSYS_PROFILE=0 CACHE=3 RETROINFER_CORE="${RETROINFER_CORE}" RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU  bash ./run_n2n_qwen_retroinfer_batchsize.sh
