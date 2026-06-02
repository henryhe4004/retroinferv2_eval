# llama
NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=1 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=3 CACHE_POLICY=FIFO bash ./run_n2n_llama_retroinfer_seqlen.sh

NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_seqlen.sh
# qwen
NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=FIFO bash ./run_n2n_qwen_retroinfer_seqlen.sh
NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=FIFO bash ./run_n2n_qwen_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=1 CACHE_POLICY=FIFO bash ./run_n2n_qwen_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=3 CACHE_POLICY=FIFO bash ./run_n2n_qwen_retroinfer_seqlen.sh

NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_seqlen.sh
NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_seqlen.sh
NSYS_PROFILE=1 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_seqlen.sh


NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_batchsize.sh
NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_llama_retroinfer_batchsize.sh
NSYS_PROFILE=0 CACHE=1 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_batchsize.sh
NSYS_PROFILE=0 CACHE=3 CACHE_POLICY=LRU bash ./run_n2n_qwen_retroinfer_batchsize.sh