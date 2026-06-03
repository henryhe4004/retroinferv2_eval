#!/usr/bin/env bash
# set -euo pipefail

# Match the old RetrievalAttention_eval throughput semantics:
# retrieve 10% clusters plus the fixed 23.2% estimation zone.
# Override these from the environment if you want a different sparsity setting.
export TOPK="${TOPK:-0.10}"
export RETRIEVAL_BUDGET="${RETRIEVAL_BUDGET:-${TOPK}}"
export ESTIMATION_BUDGET="${ESTIMATION_BUDGET:-0.232}"

run_experiment() {
    local name="$1"
    shift

    echo "[RUN] ${name}"
    "$@" || echo "[WARN] experiment failed, continue: ${name}"
}

# llama Cache=1 用作bg
run_experiment "llama seqlen nsys1 cache3 LRU" env NSYS_PROFILE=0 CACHE=1 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
run_experiment "llama seqlen nsys1 cache3 LRU" env NSYS_PROFILE=1 CACHE=1 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
run_experiment "llama seqlen nsys1 cache3 LRU" env NSYS_PROFILE=0 CACHE=1 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
run_experiment "llama seqlen nsys1 cache3 LRU" env NSYS_PROFILE=1 CACHE=1 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh

run_experiment "llama seqlen nsys1 cache3 FIFO" env NSYS_PROFILE=0 CACHE=1 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh
run_experiment "llama seqlen nsys1 cache3 FIFO" env NSYS_PROFILE=1 CACHE=1 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh
run_experiment "llama seqlen nsys1 cache3 FIFO" env NSYS_PROFILE=0 CACHE=1 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh
run_experiment "llama seqlen nsys1 cache3 FIFO" env NSYS_PROFILE=1 CACHE=1 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh


#用作eval seq
run_experiment "llama seqlen nsys1 cache3 LRU" env NSYS_PROFILE=0 CACHE=3 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
run_experiment "llama seqlen nsys1 cache3 LRU" env NSYS_PROFILE=1 CACHE=3 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
run_experiment "llama seqlen nsys1 cache3 LRU" env NSYS_PROFILE=0 CACHE=3 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
run_experiment "llama seqlen nsys1 cache3 LRU" env NSYS_PROFILE=1 CACHE=3 RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh



run_experiment "qwen seqlen nsys1 cache3 LRU" env NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU  RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 bash ./run_n2n_qwen_retroinfer_seqlen.sh
run_experiment "qwen seqlen nsys1 cache3 LRU" env NSYS_PROFILE=1 CACHE=3 CACHE_POLICY=LRU  RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 bash ./run_n2n_qwen_retroinfer_seqlen.sh
run_experiment "qwen seqlen nsys1 cache3 LRU" env NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU  RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 bash ./run_n2n_qwen_retroinfer_seqlen.sh
run_experiment "qwen seqlen nsys1 cache3 LRU" env NSYS_PROFILE=1 CACHE=3 CACHE_POLICY=LRU  RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 bash ./run_n2n_qwen_retroinfer_seqlen.sh


#用作eval batchsize
run_experiment "llama batchsize nsys0 cache3 LRU" env NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 bash ./run_n2n_llama_retroinfer_batchsize.sh
run_experiment "qwen batchsize nsys0 cache3 LRU" env NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU RETRO_CACHE_STATS=0 GPU_MEM_MONITOR_INTERVAL_SEC=999999 bash ./run_n2n_qwen_retroinfer_batchsize.sh
