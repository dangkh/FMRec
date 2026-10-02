# Plan: memory pool + LLM selector (dangkh's proposal)

Written 2026-09-17. Companion to PAPER_PLAN.md.

## 1. What is being tested

dangkh's expectation (Messenger, 13:47):

> random < personal ~ full-no-filter < **with filter**

Mechanism: take the top-N users most similar by LightGCN (as now), but instead of
one best lesson per user, **pool every lesson** of those N users plus the target
user's own, then let an LLM selector pick the few that actually apply to the
current history + candidate set.

This is the one RQ2 mechanism not yet ruled out. C6/C13 killed *who to borrow
from*; this tests *which lesson to keep*, which is a different question.

The user's message says "top 3" once and "5 user" once. N is made a flag and
both are run; N=5 is the primary arm.

## 2. Baseline to match — Gemma-3-4B, 20 candidates

Root `ladder_memcfproto_gemma_1000u`, config `gk3 nk10 mf3 mw55 tb420`,
`--max_negative_candidates 19`, n=1,000, 4 datasets. Mean N@10:

| arm | N@10 |
|---|---|
| no memory | 0.3452 |
| random neighbours | 0.4024 |
| personal only | 0.4068 |
| **FMRec top-K (the bar to beat)** | **0.4143** |

Everything new must be paired against these exact files (same users, same
candidate lists — verify with `paircheck.py`, 0 mismatches required).

## 3. Code changes — IMPLEMENTED 2026-09-17 (local, compiled, unit-tested; not yet on server)

`git diff --stat`: +187 / −8 lines in `experiment_with_fmrec_topk.py`.

| # | where | change |
|---|---|---|
| 3a | `retrieve_dense_lgcn_fmrec_topk` | new `pool_all` mode: ALL lessons of self + ALL lessons of each of the top-N donors, sorted by `(-confidence, memory_id)`; neighbour rows keep `consensus_verified` so the gate passes the pool through; returns `retrieved[:top_k]` where top_k is widened by 3d |
| 3b | scope whitelist ×3 + dispatch | `dense_lgcn_fmrec_pool`, `dense_lgcn_fmrec_pool_random` |
| 3c | `read_graph_lessons_as_facets_v2` L~9186 | **bug fix**: overflow cap uses `memory_selector_top_m*2` (was `top_k*2` = 6, truncating every pool to 6 rows before selection) |
| 3d | L~10871 | `retrieval_top_k = max(gk, top_m)` for pool scopes even with selector off, and for `heuristic` |
| 3e | new `select_memory_facts_heuristic()` | zero-LLM selector: `score = 3·direct_candidate_match + 2·|candidate_matches| + 1·|strong_history_matches| + confidence`; rows with no overlap dropped regardless of source; keep top_k |
| 3f | `select_memory_facts_with_llm_v2` | `neutral_cross` param: rule 5 becomes "judge same-user and cross-user by the same standard" instead of "cross-user needs stronger evidence" |
| 3g | CLI | `--memory_selector {none,llm,heuristic}`, `--memory_selector_neutral_cross`; run-name tag `_selheuristic_sm18_sk3_sr0p60`, `_nc1`, `_selnone_sm18` |
| 3h | audit | `pool_size_in / pool_own_in / pool_cross_in / neutral_cross` written into `selector_audit` for every user |

Local test (`python3 - <<PY`, standalone exec of the heuristic): ranks by overlap not confidence, drops a 0.95-confidence own lesson with zero overlap, drops a 0.99-confidence cross lesson with zero overlap, audit counts correct. Rule-5 templating verified present in the LLM selector source.

## 4. Arms — 20 jobs, all Gemma-3-4B, 20-cand, n=1,000, 4 datasets

| tag | scope | N | selector | isolates |
|---|---|---|---|---|
| `pool5_none` | `dense_lgcn_fmrec_pool` | 5 | none → top-3 by confidence | bigger pool, no filter |
| **`pool5_llm`** | `dense_lgcn_fmrec_pool` | 5 | **llm** top_k=3 top_m=18, neutral_cross | **dangkh's proposal** |
| **`pool5_heur`** | `dense_lgcn_fmrec_pool` | 5 | **heuristic** top_k=3 top_m=18 | same pool, rule-based, 0 extra calls |
| `pool5_rand_llm` | `dense_lgcn_fmrec_pool_random` | 5 | llm, neutral_cross | does the LLM filter need good donors? |
| `pool3_llm` | `dense_lgcn_fmrec_pool` | 3 | llm, top_m=12, neutral_cross | the "top 3" reading |

Baseline `topk` (0.4143), `same_user` (0.4068), `random` (0.4024), `nomem` (0.3452) already exist in `ladder_memcfproto_gemma_1000u` — same users, same candidates, same memory file.

## 5. Cost estimate

| | now (topk) | pool5_filt |
|---|---|---|
| LLM calls / user | 1.02 | ~2.0 (+1 selector call) |
| selector prompt | — | ~18 lessons × ~70 tok + history/candidates ≈ 2,000 tok |
| tokens / user | 1,962 | ~4,000 |

Doubles the per-query cost and puts us level with MemRec `read` (2.03 calls).
If this arm wins, the "1 call vs 2" efficiency claim in the cost table is gone
for this variant — report both variants.

16 jobs ≈ 16 × 1,000 × 2 calls = 32k LLM calls; the baseline batch of 28 jobs
took ~2.5 h wall on one GPU, so expect ~2 h. One GPU, one vLLM, within the
32-run limit. Start vLLM and launch the jobs in the same SSH session (lesson
from the 09-13 outage).

## 6. Integrity checks before reading a single number

1. `paircheck.py` against `ladder_memcfproto` topk: 0 candidate / 0 GT mismatches.
2. `mr_fallback.py`-style check: identity rate must stay in the 3–15% band; a
   jump means the selector broke the prompt and the reranker fell back.
3. `selector_audit.input_memory_count` mean must be ≈ 15–18 for N=5. If it
   reads ≤ 6, edit 3b did not take.
4. `memory_selector_fallbacks` ≈ 0. Each fallback is a JSON parse failure that
   silently returns the unfiltered pool.
5. `llm_usage.errors` = 0 in every summary.

## 7. Decision rule

Pooled paired bootstrap (5,000 resamples, seed 2027) + permutation, N@1/N@3/N@5/N@10.

| outcome | verdict | consequence for the paper |
|---|---|---|
| `pool5_filt` > `topk` SIG **and** `pool5_filt` > `pool5_rand_filt` SIG | **positive** | first collaborative win; Table 3 fills; RQ2b rewritten |
| `pool5_filt` > `topk` but ≈ `pool5_rand_filt` | filter helps, donors don't | selector is a *content* mechanism, not collaborative; C13 stands |
| `pool5_filt` ≈ `topk` | null | drop Table 3; keep the C2+C4 axis |
| `pool5_nofilt` < `topk` | bigger pool hurts unfiltered | consistent with C2 (only slot 1 matters) — worth one sentence |

Given C2 (94–95% of the effect is the first lesson), the honest prior is
"null". The experiment is worth running because it is the last mechanism
standing and dangkh asked for it explicitly, not because the odds are good.

## 8. Order of work

1. Patch `experiment.py` on the server (3a–3d), `py_compile`, run the 26 unit
   tests, take a timestamped backup first — same routine as the four earlier patches.
2. Smoke test: 1 dataset, 20 users, `pool5_filt`, confirm `input_memory_count`
   ≈ 15–18 and fallbacks = 0 **before** launching the batch.
3. Launch 16 jobs via a parameterized `launch_pool.sh` that prints its own
   `EVAL_ROOT` (trap #4: never sed-chain a launcher).
4. Server-side monitor kills vLLM on completion.
5. Analyse with a `pool_analysis.py` built from `screen_analysis.py`, which
   already carries the pairing guard and the config-confound warning.
6. Pull the patched `experiment.py` back to the repo branch with md5 check, so
   the committed code stays the code that ran.
