# Running the FMRec experiment (pool-LLM version)

This folder holds the code behind `experiment_with_fmrec_topk.py`, the evaluation
harness used for the paper's FMRec numbers. The FMRec configuration is the
**memory pool + LLM selector** variant. For each test user it works as follows:

1. Find the user's top-N most similar users by LightGCN embedding.
2. Pool every failure lesson from those users plus the user's own lessons.
3. Have an LLM selector keep the L lessons that apply to the current history and candidates.
4. Have the LLM rank the 20 candidates using the selected lessons.

## Layout

```text
experiment_with_fmrec_topk.py   entry point (python experiment_with_fmrec_topk.py ...)
fmrec/                          the pool-LLM path
  cli.py            command-line arguments + main driver (main_v2)
  memory_build.py   builds the failure-lesson memory (temporal_factual protocol), save/load
  graph_index.py    MemoryGraphIndex: lesson store, LightGCN neighbours, pool retrieval
  selector.py       LLM memory selector, lesson -> prompt facts
  ranking.py        LLM ranking step
  evaluation.py     per-user evaluation
  llm_client.py     OpenAI-compatible LLM client, token/call accounting, traces
  prompts.py, records.py, common.py, data_metrics.py
fmrec/variants/                 experimental options the pool-LLM path does not use
                                (other retrieval scopes, selectors, prompt styles, legacy protocols)
```

When copying the code to another machine, copy **both** `experiment_with_fmrec_topk.py`
and the whole `fmrec/` folder.

## Requirements

- Python 3.9+ with `numpy` and `tqdm`. `torch` and `transformers` are optional and
  only used when no API endpoint is configured.
- An OpenAI-compatible chat endpoint. The paper uses vLLM serving Gemma-3-4B-it:

```bash
vllm serve /path/to/gemma-3-4b-it --port 8001 \
  --served-model-name gemma-3-4b-it --max-model-len 8192 --max-num-seqs 32
```

## Inputs

| What | Where |
|---|---|
| Dataset | `$MEMCF_DATA_ROOT/<data_name>/items.json`, `user_sequences_10.json`, `user_negatives_10.json` |
| LightGCN user embeddings | JSON passed with `--lightgcn_embeddings_json` |
| Failure memory (eval only) | JSON passed with `--memory_file` |

The 20 candidates per user are the test item plus the 19 negatives in
`user_negatives_10.json`, so every method sees the same candidate set.
`fmrec_memory_export/<dataset>/` already contains the data, the LightGCN embeddings
and the trained memory for Video_Game, Software_1000u and Prime_Pantry_1000u.

## Environment variables

```bash
export chat_api_base=http://127.0.0.1:8001/v1     # LLM endpoint
export chat_model_name=gemma-3-4b-it              # must match --served-model-name
export MEMCF_DATA_ROOT=/path/to/runtime_data      # parent of <data_name>/
export MEMCF_EVAL_ROOT=/path/to/evaluation_results
export MEMCF_MEMORY_ROOT=/path/to/agent_memory
# optional: MEMCF_TEMPERATURE (default 0.0)
```

Without the three `MEMCF_*_ROOT` variables, the defaults are `data/`,
`evaluation_results/` and `agent_memory/` next to `experiment_with_fmrec_topk.py`.

## 1. Evaluate FMRec with an existing memory (Table 3 setting: N=4, L=3)

```bash
D=fmrec_memory_export/Video_Game
export MEMCF_DATA_ROOT=$D/runtime_data_rebuild_v2_seed42

python experiment_with_fmrec_topk.py \
  --data_name Video_Game --number_of_users 1000 --phase eval_only \
  --memory_file $D/i_temporal_nuser1000_8shards.memory.json \
  --lightgcn_embeddings_json $D/Video_Game.lgcn_embeddings.json \
  --graph_retrieval_scope dense_lgcn_fmrec_pool --fmrec_top_k_neighbors 4 \
  --memory_selector llm --memory_selector_top_k 3 --memory_selector_top_m 20 \
  --memory_selector_min_relevance 0.60 --memory_selector_neutral_cross \
  --max_memory_facts 3
```

- `--fmrec_top_k_neighbors` is **N**, the number of similar users whose lessons are pooled.
- `--memory_selector_top_k` is **L**, the number of lessons the selector may keep.
- `--max_memory_facts` should equal L.
- Add `--eval_workers 8` to evaluate users in parallel.
- For the other datasets, change `D` and `--data_name` (e.g. `Software_1000u`, `Prime_Pantry_1000u`).

## 2. Build the failure memory yourself (optional)

```bash
python experiment_with_fmrec_topk.py \
  --data_name Video_Game --number_of_users 1000 --phase train_only \
  --failure_memory_protocol temporal_factual --graph_retrieval_scope temporal_full \
  --ranking_prompt_style compact_safe_residual_score \
  --memory_file agent_memory/Video_Game/i_temporal_nuser1000.memory.json
```

Then run step 1 with `--memory_file` pointing at the new file. The memory in
`fmrec_memory_export/` was built in 8 user shards (`--num_user_shards 8 --user_shard_id k`)
and merged afterwards. The merge script is not part of this folder, so for a fresh
build run a single process as shown above.

## Outputs

Everything goes to `$MEMCF_EVAL_ROOT/<data_name>/`:

- `<run_name>.summary.json`: Hit/Recall/NDCG at 1, 3, 5, 10 and 20, plus `llm_usage`
  (calls and prompt/completion tokens, split into `memory_selector` and `ranking`)
  and memory statistics.
- `<run_name>.json`: per-user candidates, rankings and metrics.
- `traces/<run_name>_<timestamp>/`: JSONL traces of every LLM call and selector decision.
  Disable them with `--disable_trace`.

The run name encodes the configuration, so runs with different settings do not overwrite each other.
