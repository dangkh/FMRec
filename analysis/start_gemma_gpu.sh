#!/usr/bin/env bash
# start_gemma_gpu.sh <gpu_index> <port>
set -uo pipefail
G="${1:?gpu}"; P="${2:?port}"
unset CUDA_VISIBLE_DEVICES
free=$(/home/ubuntu/miniconda3/bin/python -c "
import torch; f,_=torch.cuda.mem_get_info($G); print(int(f/1e9))" 2>/dev/null)
echo "gpu$G free=${free}GB"
[ "${free:-0}" -ge 20 ] || { echo "ABORT: gpu$G has only ${free}GB free"; exit 1; }
export CUDA_VISIBLE_DEVICES="$G" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=/home/ubuntu/vllm_gemma_gpu${G}.log
: > "$LOG"
setsid nohup /home/ubuntu/miniconda3/bin/vllm serve /home/ubuntu/24nam.nh/models/gemma-3-4b-it \
  --port "$P" --served-model-name gpt-3.5-turbo-16k-0613 \
  --gpu-memory-utilization 0.85 --max-num-seqs 32 --max-model-len 8192 \
  >> "$LOG" 2>&1 < /dev/null & disown
echo "gemma starting on gpu$G port $P, log $LOG"
