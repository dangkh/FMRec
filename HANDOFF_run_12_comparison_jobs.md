# Handoff: run the MEMCF vs FMRec (dangkh) comparison — 12 jobs

You are picking this up on a machine that only has the `FMRec` repo (
github.com/dangkh/FMRec) cloned, with no local copy of `MEMCF`. That's fine
-- everything needed lives on the remote GPU server, which this machine can
reach directly over SSH (no VPN hurdle here, unlike the session that wrote
this handoff). You do the whole task by SSHing in and operating on the
remote copy of the repo; you don't need MEMCF cloned locally at all.

A companion file, `experiment_with_fmrec_topk.py`, was sent alongside this
one -- it is the complete, ready-to-deploy replacement for the remote
`experiment.py`. Read this whole file before touching anything.

## 1. What MEMCF / BINRec / FMRec is

A research project (target: ECIR 2027, paper title "FMRec: Failure Memory
for LLM-based Recommendation") studying whether an LLM reranker can improve
by learning from records of its own past ranking mistakes ("failure
lessons"), including whether a lesson learned from one user can transfer to
a *different*, collaboratively-similar user (the paper's RQ2).

The user (your principal) is a researcher building this with an advisor and
a collaborator, "dangkh" / Kieu Hai Dang, who independently built the
`FMRec` repo you already have locally -- a **much simpler**, ungated version
of the same core idea, on a different dataset (books) and a different local
LLM (Gemma-3-4B via Unsloth). MEMCF is the primary/more developed codebase;
its retrieval mechanisms have gone through several redesigns this session
after diagnosing a real bug (see §2). Full narrative in
`docs/AGENT_HANDOFF_SERVER_OPS.md` and `docs/reference_implementations/RQ2_PROGRESS.md`
inside the MEMCF repo on the remote server -- read those too once you're in
(`cat` them over SSH), they have more depth than this file repeats.

## 2. The immediate task: 12 comparison runs

Goal: compare MEMCF's own cross-user retrieval mechanism against a faithful
port of `dangkh/FMRec`'s mechanism, on **identical** data/LLM/lessons, to
isolate the effect of mechanism design alone (their repo uses a totally
different dataset+LLM, so a direct cross-repo comparison would not be
controlled).

Three configs, each run on 4 datasets, 100 users each = 12 runs:

| Config | What it is | `--use_memory`? | `--graph_retrieval_scope` |
|---|---|---|---|
| **A0** (no memory) | Shared "without memory" baseline for both versions | omit the flag entirely | n/a |
| **Version A** (MEMCF's own mechanism) | Requires >=2 independently-similar users to agree on the same item before surfacing it; ranks by mean voter similarity, not raw vote count (a bug-fix landed this session -- see §2a) | pass `--use_memory` | `dense_lgcn_userscore_consensus` |
| **Version B** (ported FMRec-style) | Faithful port of `dangkh/FMRec`'s `retrieve_fmrec_lessons.py`: top-K similar users by LightGCN cosine, **no threshold, no consensus, no item-overlap gate at all** -- each neighbor's single highest-confidence lesson is surfaced unconditionally | pass `--use_memory` | `dense_lgcn_fmrec_topk` |

Datasets (all 4, every config): `Software_1000u`, `Prime_Pantry_1000u`,
`Video_Game`, `Industrial_and_Scientific_1000u`.

### 2a. Why this comparison matters (one paragraph)

MEMCF's mechanism used to work like FMRec's still does -- naive top-K by
embedding similarity, no gate -- and an audit found that design's retrieval
score saturates at ~1.0 for nearly every surfaced cross-user fact (the
"similarity" signal carried no real information), so it never showed a
causal cross-user effect. MEMCF's current mechanism was redesigned
specifically to fix that (embedding-only ranking + required multi-user
agreement + rank by mean similarity, not raw vote count) but coverage is
very low as a result (only ~1-14% of users ever get a cross-user fact at
all) and a confirmed effect has not yet replicated at N=1,000 (see
`RQ2_PROGRESS.md`). FMRec's design is the opposite bet: no gate at all, much
higher coverage, but personal risk factors MEMCF's own experiments flagged
elsewhere (abstract, ungrounded lessons hurting accuracy vs. direct
evidence -- see the `mm_rule_overrides_direct_evidence` memory note if you
have access to it). This 12-run comparison is the first apples-to-apples
test of that tradeoff.

## 3. Server access

```bash
ssh 24nam.nh@10.140.24.17
```
You said this machine can reach it directly -- confirm with a plain
connectivity check first. Land in a container; `whoami` there is `ubuntu`,
repo is at:
```
/home/ubuntu/24nam.nh/video_games_data/MEMCF/
```
**Always pipe commands through `ssh -tt` with a heredoc**, don't rely on a
single quoted remote command string -- this channel is flaky about plain
non-tty invocations. Pattern:
```bash
printf '%s\n' 'command one' 'command two' 'exit' | ssh -tt 24nam.nh@10.140.24.17
```

## 4. Deploy the updated experiment.py

The remote file already has an earlier fix (the mean-similarity consensus
ranking) but **does not yet have** the new `dense_lgcn_fmrec_topk` mechanism
-- that's what `experiment_with_fmrec_topk.py` (sent alongside this file)
adds. Deploy it:

1. Get `experiment_with_fmrec_topk.py` onto the remote host at
   `/tmp/experiment_new.py` by any transfer method that works from this
   machine (scp is blocked on the *other* path to this server in past
   sessions -- test it fresh here; if blocked, heredoc/base64 through the
   `ssh -tt` pipe like everything else in this doc).
2. On remote:
   ```bash
   cd /home/ubuntu/24nam.nh/video_games_data/MEMCF
   cp src/memcf/experiment.py "src/memcf/experiment.py.bak_$(date +%Y%m%d_%H%M%S)"
   cp /tmp/experiment_new.py src/memcf/experiment.py
   python3 -m py_compile src/memcf/experiment.py && echo COMPILE_OK
   PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' 2>&1 | tail -6
   ```
   Expect `COMPILE_OK` and `Ran 26 tests ... OK`. **Do not proceed to launching
   runs if either fails** -- fix or ask before running anything.
3. Smoke-test on 5 users before the real batch (catches CLI/argparse typos
   for free in ~30s instead of after a wasted 100-user run):
   ```bash
   cd /home/ubuntu/24nam.nh/video_games_data/MEMCF
   export PYTHONPATH="$(pwd)/src"
   export MEMCF_DATA_ROOT=/home/ubuntu/24nam.nh/video_games_data/runtime_data_rebuild_v2_seed42
   export MEMCF_EVAL_ROOT=/tmp/smoketest_fmrec_compare
   export chat_model_name=gpt-3.5-turbo-16k-0613
   export chat_api_base="http://127.0.0.1:8000/v1"
   export api_base="$chat_api_base"
   python3 -m memcf --data_name Software_1000u --number_of_users 5 \
     --max_positive_interactions 5 --max_negative_candidates 19 --max_iterations 1 \
     --candidate_negative_mode candidate_hard --ranking_prompt_style compact_score \
     --phase eval_only --eval_split test --use_memory \
     --graph_retrieval_scope dense_lgcn_fmrec_topk --graph_memory_k 3 \
     --fmrec_top_k_neighbors 3 --min_evidence_terms 1 --max_memory_facts 3 \
     --max_memory_fact_words 55 --memory_token_budget 420 \
     --memory_file /home/ubuntu/24nam.nh/video_games_data/agent_memory_memcf_i_temporal_1k_rebuild_v2/Software_1000u/i_temporal_nuser1000_8shards.memory.json \
     --lightgcn_embeddings_json /home/ubuntu/24nam.nh/video_games_data/lgcn_embeddings_1k_rebuild_v2/Software_1000u.lgcn_embeddings.json \
     --run_name_suffix smoketest --eval_workers 2
   ```
   Confirm it completes and prints a metrics table before moving on.

## 5. Check the GPU/vLLM endpoints are alive

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
ps aux | grep vllm | grep -v grep
```
Ports `8000` and `8003` are the two established endpoints this project uses
(a ~24GB-used-but-0%-utilization GPU is normal -- it means the model is
loaded and idle, not stuck). If neither is up, stop and ask the user rather
than starting one yourself.

## 6. Launch all 12 runs

**Capacity note: the user says each GPU can hold up to ~32 concurrent runs
here** -- do not queue these sequentially like earlier sessions did. Fire
all 12 as independent background processes right away, split across the two
ports (6 each is a reasonable split, exact balance doesn't matter given the
stated headroom).

Common settings for every run (only `--data_name`, `--use_memory` /
`--graph_retrieval_scope`, `--memory_file`, `--lightgcn_embeddings_json`,
`--run_name_suffix`, and the port differ):

```bash
cd /home/ubuntu/24nam.nh/video_games_data/MEMCF
export PYTHONPATH="$(pwd)/src"
export MEMCF_DATA_ROOT=/home/ubuntu/24nam.nh/video_games_data/runtime_data_rebuild_v2_seed42
export MEMCF_EVAL_ROOT=/home/ubuntu/24nam.nh/video_games_data/evaluation_results_memcf_fmrec_compare_100u
export chat_model_name=gpt-3.5-turbo-16k-0613
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
MEMROOT=/home/ubuntu/24nam.nh/video_games_data/agent_memory_memcf_i_temporal_1k_rebuild_v2
EMBROOT=/home/ubuntu/24nam.nh/video_games_data/lgcn_embeddings_1k_rebuild_v2
```

For each of `Software_1000u`, `Prime_Pantry_1000u`, `Video_Game`,
`Industrial_and_Scientific_1000u` (call the dataset `$DS` below), launch
THREE background jobs (A0, Version A, Version B), each `>>`-logging to its
own file so you can tell them apart:

```bash
mkdir -p "$MEMCF_EVAL_ROOT/$DS/logs"

# A0 -- no memory at all
export chat_api_base="http://127.0.0.1:8000/v1"; export api_base="$chat_api_base"
nohup python3 -m memcf --data_name "$DS" --number_of_users 100 \
  --max_positive_interactions 5 --max_negative_candidates 19 --max_iterations 1 \
  --candidate_negative_mode candidate_hard --ranking_prompt_style compact_score \
  --phase eval_only --eval_split test \
  --run_name_suffix "fmrec_compare_A0" --eval_workers 4 \
  >> "$MEMCF_EVAL_ROOT/$DS/logs/A0.log" 2>&1 & disown

# Version A -- MEMCF's own consensus mechanism
export chat_api_base="http://127.0.0.1:8000/v1"; export api_base="$chat_api_base"
nohup python3 -m memcf --data_name "$DS" --number_of_users 100 \
  --max_positive_interactions 5 --max_negative_candidates 19 --max_iterations 1 \
  --candidate_negative_mode candidate_hard --ranking_prompt_style compact_score \
  --phase eval_only --eval_split test --use_memory \
  --graph_retrieval_scope dense_lgcn_userscore_consensus --graph_memory_k 3 \
  --consensus_top_n_users 15 --consensus_min_users 2 \
  --min_evidence_terms 1 --max_memory_facts 3 --max_memory_fact_words 55 --memory_token_budget 420 \
  --memory_file "$MEMROOT/$DS/i_temporal_nuser1000_8shards.memory.json" \
  --lightgcn_embeddings_json "$EMBROOT/$DS.lgcn_embeddings.json" \
  --run_name_suffix "fmrec_compare_versionA" --eval_workers 4 \
  >> "$MEMCF_EVAL_ROOT/$DS/logs/versionA.log" 2>&1 & disown

# Version B -- ported FMRec-style (no gate)
export chat_api_base="http://127.0.0.1:8003/v1"; export api_base="$chat_api_base"
nohup python3 -m memcf --data_name "$DS" --number_of_users 100 \
  --max_positive_interactions 5 --max_negative_candidates 19 --max_iterations 1 \
  --candidate_negative_mode candidate_hard --ranking_prompt_style compact_score \
  --phase eval_only --eval_split test --use_memory \
  --graph_retrieval_scope dense_lgcn_fmrec_topk --graph_memory_k 3 \
  --fmrec_top_k_neighbors 3 \
  --min_evidence_terms 1 --max_memory_facts 3 --max_memory_fact_words 55 --memory_token_budget 420 \
  --memory_file "$MEMROOT/$DS/i_temporal_nuser1000_8shards.memory.json" \
  --lightgcn_embeddings_json "$EMBROOT/$DS.lgcn_embeddings.json" \
  --run_name_suffix "fmrec_compare_versionB" --eval_workers 4 \
  >> "$MEMCF_EVAL_ROOT/$DS/logs/versionB.log" 2>&1 & disown
```

Repeat for all 4 datasets (12 `nohup ... & disown` launches total). Split
the `chat_api_base` port assignment roughly evenly across 8000/8003 given
the stated per-GPU headroom (e.g. alternate which port each dataset's three
jobs land on, or just put half the 12 on each port) -- exact balance isn't
critical.

After launching, confirm:
```bash
ps aux | grep "python3 -m memcf" | grep -v grep | wc -l   # expect 12
```

## 7. Timing expectations

At 100 users/job with all 12 running concurrently (not queued), expect each
job to finish in roughly **7-9 minutes** (linear extrapolation from this
session's data: 200 users ~14 min, 1,000 users ~65 min, i.e. ~3.9s/user
marginal + ~1 min fixed startup overhead). Since they're not serialized
this time, the whole batch should complete in about that same ~10-minute
window, not 12x longer -- confirm this assumption holds by checking after
~10-15 minutes; if it's clearly running much slower than single-job pace,
the two vLLM endpoints may be more contended than expected and you should
report that rather than assuming.

## 8. Verify completion and pull results

```bash
ls /home/ubuntu/24nam.nh/video_games_data/evaluation_results_memcf_fmrec_compare_100u/*/*.summary.json | wc -l   # expect 12
```

Each `.summary.json` has a top-level `metrics` dict with `ndcg@10` etc., and
each `.json` (non-summary) is the per-user array with `user_id` and
`metrics.ndcg@10`. Use the paired-bootstrap pattern used throughout this
project (5,000 resamples, seed=2027) to compare, per dataset:
- Version A vs A0
- Version B vs A0
- Version A vs Version B (the headline comparison)

```python
import json, random, statistics, glob

def boot_ci(diffs, seed=2027, n=5000):
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    lo, hi = means[int(0.025 * n)], means[int(0.975 * n) - 1]
    return lo, hi, (lo > 0 or hi < 0)

def load(pattern):
    matches = [m for m in glob.glob(pattern) if not m.endswith(".summary.json")]
    rows = json.load(open(matches[0]))
    return {r["user_id"]: r["metrics"]["ndcg@10"] for r in rows}
```
Join two runs' dicts on `user_id`, diff, feed to `boot_ci`. Report `n`
(common users), both means, the diff, the CI, and whether it excludes zero
-- not just point estimates, matching how every other result in this
project has been reported.

## 9. Safety rules to follow throughout (non-negotiable)

- Never overwrite an existing `MEMCF_EVAL_ROOT` from a prior run -- this one
  (`evaluation_results_memcf_fmrec_compare_100u`) is new/unused, keep it that
  way for future comparisons.
- `chat_api_base` must be an **exported env var**, never a `--chat_api_base`
  CLI flag -- passing it as a flag causes a silent instant argparse failure
  that looks like a fast "success" (~4s) but produced nothing. This is why
  the smoke test in §4 step 3 matters.
- Back up before any further code edit, `py_compile` + the 26-test suite
  after every edit, before running anything at scale.
- If anything is ambiguous or a command fails in a way this doc didn't
  anticipate, stop and report back rather than guessing around it --
  several past mistakes this session came from silently working around a
  confusing result instead of flagging it.

## 10. What to report back when done

Per dataset and per comparison (Version A vs A0, Version B vs A0, Version A
vs Version B): mean NDCG@10 for each side, the diff, the bootstrap 95% CI,
and whether it's significant. Plus: did Version B (no gate, like the
collaborator's FMRec) get meaningfully higher lesson-injection coverage
than Version A (heavily gated), and did that translate into better or worse
NDCG? That coverage-vs-quality tradeoff is the actual research question
this comparison is meant to answer.
