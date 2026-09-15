#!/usr/bin/env bash
# The symmetric counterpart to same1. Config is byte-for-byte same1's
# (gk3 nk10 mf1 mw55 tb420, n=1000) -- ONLY the retrieval scope differs, so the
# contrast isolates *whose* lesson fills the single slot that C2 shows matters.
#   same1       (already run): scope same_user                -> 1 personal fact
#   noself1     (here):        scope ..._noself               -> 1 cross-user fact
#   noself1rand (here):        scope ..._noself_random        -> random neighbour
# n=1000, not 300: at 300 the CI is ~+-0.03, far too wide for effects of ~0.005.
set -uo pipefail
PORT="${1:?port}"; N="${2:-1000}"
REPO=/home/ubuntu/24nam.nh/video_games_data/MEMCF
DATA=/home/ubuntu/24nam.nh/video_games_data
EVAL_ROOT=$DATA/evaluation_results_noself1_gemma
MEMROOT=$DATA/agent_memory_memcf_i_temporal_1k_rebuild_v2
DATASETS="Software_1000u Prime_Pantry_1000u Video_Game Industrial_and_Scientific_1000u"
MARK=ns1

cd "$REPO" || exit 1
a=$(pgrep -fc "run_name_suffix ${MARK}" || true)
[ "${a:-0}" -eq 0 ] || { echo "LAUNCH_FAIL: $a already running"; exit 1; }
python3 - "$PORT" <<'PY' || exit 1
import sys, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:%s/v1/models" % sys.argv[1], timeout=8); print("endpoint OK")
except Exception as e:
    print("LAUNCH_FAIL:", e); sys.exit(1)
PY
export PYTHONPATH="$(pwd)/src"
export MEMCF_DATA_ROOT=$DATA/runtime_data_rebuild_v2_seed42
export MEMCF_EVAL_ROOT=$EVAL_ROOT
export chat_model_name=gpt-3.5-turbo-16k-0613
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export chat_api_base="http://127.0.0.1:${PORT}/v1"; export api_base="$chat_api_base"

# identical to same1's command line
com="--number_of_users $N --max_positive_interactions 5 --max_negative_candidates 9 \
--max_iterations 1 --candidate_negative_mode candidate_hard --ranking_prompt_style compact_score \
--phase eval_only --eval_split test --use_memory --min_evidence_terms 1 \
--max_memory_fact_words 55 --memory_token_budget 420"

run () { # ds, mem, scope, tag
  nohup python3 -m memcf --data_name "$1" $com \
    --graph_retrieval_scope "$3" --fmrec_top_k_neighbors 3 \
    --graph_memory_k 3 --max_memory_facts 1 \
    --memory_file "$2" --run_name_suffix "${MARK}_$4" --eval_workers 1 \
    >> "$EVAL_ROOT/$1/logs/$4.log" 2>&1 & disown
}
for DS in $DATASETS; do
  mkdir -p "$EVAL_ROOT/$DS/logs"
  MEM="$MEMROOT/$DS/i_temporal_nuser1000_8shards.memory.json"
  [ -f "$MEM" ] || { echo "LAUNCH_FAIL: missing $MEM"; exit 1; }
  run "$DS" "$MEM" dense_lgcn_fmrec_topk_noself        noself1
  run "$DS" "$MEM" dense_lgcn_fmrec_topk_noself_random noself1rand
  echo "launched 2 arms for $DS"
done
sleep 50
n=$(pgrep -fc "run_name_suffix ${MARK}" || echo 0)
echo "EVAL_ROOT=$EVAL_ROOT"
echo "running=$n (expect 8)"
[ "$n" -eq 8 ] && echo LAUNCH_OK || echo "LAUNCH_WARN saw $n"
