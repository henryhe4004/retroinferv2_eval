#!/usr/bin/env bash
set -euo pipefail

# Target ~10% total cluster compute: retrieve 10%, no extra estimation zone.
# Override these from the environment if you want a different sparsity setting.
export TOPK="${TOPK:-0.10}"
export RETRIEVAL_BUDGET="${RETRIEVAL_BUDGET:-${TOPK}}"
export ESTIMATION_BUDGET="${ESTIMATION_BUDGET:-0}"

# llama
NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=1 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=3 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh

NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
#qwen
# NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=FIFO bash ./run_n2n_qwen_retroinfer_seqlen.sh
# NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=FIFO bash ./run_n2n_qwen_retroinfer_seqlen.sh
# NSYS_PROFILE=1 CACHE=1 CACHE_POLICY=FIFO bash ./run_n2n_qwen_retroinfer_seqlen.sh
# NSYS_PROFILE=1 CACHE=3 CACHE_POLICY=FIFO bash ./run_n2n_qwen_retroinfer_seqlen.sh

NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_seqlen.sh
NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_seqlen.sh


NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_batchsize.sh
NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_batchsize.sh
NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_batchsize.sh
NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_batchsize.sh
