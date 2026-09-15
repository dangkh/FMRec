#!/usr/bin/env bash
# Server-side so it survives the client losing the lab network.
# Kills vLLM once noself1 finishes, so no GPU server is ever left idle.
D=/home/ubuntu/24nam.nh/video_games_data
R=$D/evaluation_results_noself1_gemma
LOG=$D/monitor_ns1.log
: > "$LOG"
for i in $(seq 1 720); do   # up to 12h
  n=$(ls $R/*/*.summary.json 2>/dev/null | wc -l)
  j=$(pgrep -fc "run_name_suffix ns1" 2>/dev/null || echo 0)
  echo "$(date +%H:%M) done=$n/8 jobs=$j" >> "$LOG"
  if [ "$n" -ge 8 ] || { [ "$j" -eq 0 ] && [ "$i" -gt 3 ]; }; then
    echo "$(date +%H:%M) FINISHED done=$n jobs=$j -> killing vLLM" >> "$LOG"
    pkill -f "vllm serve"
    sleep 5
    echo "$(date +%H:%M) vllm_remaining=$(pgrep -fc 'vllm serve' || echo 0)" >> "$LOG"
    exit 0
  fi
  sleep 60
done
echo "$(date +%H:%M) TIMEOUT -> killing vLLM anyway" >> "$LOG"
pkill -f "vllm serve"
