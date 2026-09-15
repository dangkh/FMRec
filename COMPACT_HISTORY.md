# Compact history — FMRec vs MEMCF

Written 2026-09-09. Purpose: if the chat context is compacted or a new session
starts cold, this file alone should be enough to understand where the research
stands, what has been measured, and what is still open. Update it as results land.

---

## 1. The two systems

Two independent implementations of the same idea — memory built from an LLM
ranker's own past ranking mistakes ("failure lessons"), retrieved across
collaboratively similar users. Built by the user and a colleague (dangkh / Kieu
Hai Dang) with similar scope.

**Final goal: ONE paper, using ONE of the two versions, that beats every baseline
and answers all three RQs.** Everything below serves choosing that version and
closing its evidence gaps.

| | **FMRec** | **MEMCF** |
|---|---|---|
| Repo | local `/Users/nambincoi/FM-REC/FMRec` (github.com/dangkh/FMREC) | server `/home/ubuntu/24nam.nh/video_games_data/MEMCF` |
| Benchmark | InstructRec: Books, GoodReads, MovieTV (+Yelp named) | Amazon: Software, Prime_Pantry, Video_Game, Industrial, CDs, Digital_Music |
| Candidates | **10** (1 pos + 9 neg) | **20** (1 pos + 19 neg) |
| LLM | Gemma-3-4B-Instruct 4-bit (Unsloth), temp 0 | Qwen2.5-7B-Instruct via vLLM (served as `gpt-3.5-turbo-16k-0613`) |
| Eval N | 1,000 users (from `books_candidates_1k_memrec_test.json`) | 1,000 users/dataset |
| Retrieval | LightGCN top-Kn=3 similar users, no gate, <=4 lessons | many scopes; main = `full`; also consensus-gated variants |
| Lesson coverage | **59.6%** (4,398 of 7,377 Books users) | ~98-100% |
| Paper RQs | RQ1 effectiveness vs baselines · RQ2 cross-user transfer + does similarity help · RQ3 robustness to retrieval config and LLM backbone | same three-RQ structure |

**Naming trap that caused confusion for several turns:** the arm called
"Version B" / `dense_lgcn_fmrec_topk` in MEMCF is **a port of FMRec's retrieval
rule running inside MEMCF**, on Amazon data with Qwen. It is *not* FMRec. Only
code run from the local FMRec repo (Books + Gemma) is FMRec.

---

## 2. What is proven, per system

### MEMCF — strong evidence base

Two independent 1,000-user ablation sets, paired bootstrap 5,000 resamples seed
2027 + permutation test, integrity-checked (candidate sets / ground truth /
baseline metrics identical across arms; 0 mismatches).

Primary metric NDCG@10 unless stated.

| Contrast | `mainpaper_rebuild_v2` (4 ds) | `memcf_1k_paper` (6 ds) | Verdict |
|---|---|---|---|
| memory vs no memory | **+0.0332** [+0.0234,+0.0433] 4/4 | **+0.0361** [+0.0282,+0.0441] 6/6 | ✅ significant, replicates |
| memory vs profile-only | +0.0128 [+0.0061,+0.0196] 4/4 | +0.0012 [-0.0045,+0.0068] | ⚠️ the two sets disagree |
| shuffled lessons vs no memory | **+0.0272** [+0.0172,+0.0371] 4/4 | — | ✅ significant |
| **real vs shuffled lessons** | **+0.0060** [-0.0007,+0.0129] 3/4 | — | ❌ **not significant** |
| **cross-user vs same-user only** | +0.0013 [-0.0042,+0.0067] 1/4 | -0.0043 [-0.0090,+0.0007] 1/6 | ❌ **precise null** |
| no-harm arbitration | +0.0035 [+0.0002,+0.0068] 4/4 | -0.0003 | marginal |

**Decomposition of MEMCF's +0.0332 (NDCG@5 lift over its own no-memory arm):**

| Ingredient | Lift | Share of total |
|---|---|---|
| Profile text only, **zero lessons** | +9.1% | **66%** |
| **Shuffled** lessons (wrong user, wrong context) | +11.9% | **82%** |
| Same-user (real, personal) lessons | +13.5% | 98% |
| Full (+ cross-user) | +13.8% | 100% |
| Full + arbitration | +15.3% | 111% |

**The problem this creates:** 82% of the gain survives replacing every lesson
with a wrong one. What is left for correct, failure-derived content is +0.0060
and not significant. So "memory helps" (RQ1) is solid; "*failure* memory is why"
is not yet defensible. Cross-user transfer (RQ2) is a well-powered null.

### FMRec — thin evidence base

From the paper (Table 3, Books only; Goodreads and MovieTV columns are all `0.`):

| Model | H@1 | H@3 | N@3 | H@5 | N@5 |
|---|---|---|---|---|---|
| Vanilla LLM (no memory) | 0.3138 | 0.5617 | 0.4533 | 0.7270 | 0.5226 |
| MemRec (best baseline) | 0.4633 | 0.6667 | 0.5794 | 0.7967 | 0.6320 |
| **FMRec** | 0.4436 | 0.6537 | 0.5624 | 0.7833 | 0.6162 |
| Improv. vs best baseline | **-4.4%** | **-1.9%** | **-3.0%** | **-1.7%** | **-2.5%** |

- **FMRec currently LOSES to MemRec on all five metrics** — contradicts the
  abstract's "consistently improves over competitive baselines".
- Lift over its **own** no-memory arm is healthy: N@5 0.5226 -> 0.6162 = **+17.9%**.
- §4.3 (RQ2/RQ3 results) is **empty**. No ablation, no statistical testing, no CIs.
- Paper never states the number of evaluated users (Table 2's 7.4K is dataset
  size, not eval size). Actual eval N = **1,000**.
- Local repo has Phase 1 only: `results/fmrec_lessons_books_r2/` (4,398 lessons).
  The ranking notebook `scripts/fmrec_retrieve_and_rank.ipynb` has **0/20 cells
  with output** — never run, or outputs cleared.

---

## 3. Cross-system comparison

Absolute scores are **not comparable** (different benchmarks, 10 vs 20
candidates — random H@1 alone differs 2x — and different backbones). Two things
are comparable:

**(a) Lift over each system's own no-memory arm:** FMRec +17.9% vs MEMCF +13.8%
(N@5). FMRec's larger figure is *not* evidence it is better: a 10-candidate task
with a weaker 4-bit model leaves far more headroom.

**(b) Both mechanisms inside the SAME harness** (identical data, LLM, lesson pool)
— the only controlled comparison:

| Contrast | Result |
|---|---|
| FMRec design vs MEMCF `full` (200u) | +0.0019 [-0.0115,+0.0152] n.s., 2/4 |
| consensus-gated vs FMRec top-K (100u) | -0.0041 [-0.0254,+0.0178] n.s., 1/4 |

→ **As mechanisms, the two are indistinguishable. As evidence bases, MEMCF is far
ahead.** FMRec's one advantage is a promising unverified signal (see §4).

---

## 4. RQ2 is a well-powered null on BOTH halves  (resolved 2026-09-10)

FMRec's RQ2 asks two things. Each now has a direct experiment at n=1,000 on 4
datasets, and both come back null with tight intervals.

| RQ2 half | Experiment | 200u | **1,000u (final)** |
|---|---|---|---|
| Does collaborative similarity help pick whose lesson to reuse? | `fmrec_topk` vs `fmrec_topk_random` (K similar vs K **random** users; same mechanism, same budget, same fact count) | +0.0110, 3/4 | **-0.0012** [-0.0072,+0.0050] n.s., 2/4 |
| Does failure knowledge transfer across users at all? | `fmrec_topk` vs `same_user` | +0.0008 | **+0.0012** [-0.0050,+0.0074] n.s., 3/4 |

**The 200u +0.0110 was noise.** It was projected to land near [+0.005,+0.017] at
n=1,000; it collapsed to -0.0012 instead. Picking the 3 most similar users by
LightGCN cosine performs the same as picking 3 users at random. Treat this as
the settled answer unless a different similarity signal is tried.

### The rest of the n=1,000 ladder (`fmrec_ablation_1000u`, NDCG@10)

| Contrast | Diff | 95% CI | Direction | |
|---|---|---|---|---|
| memory vs no memory | **+0.0334** | [+0.0233,+0.0431] | 4/4 | ✅ significant |
| lessons vs profile-only | **+0.0127** | [+0.0061,+0.0196] | 3/4 | ✅ significant |
| real vs shuffled lessons | +0.0060 | [-0.0009,+0.0126] | **4/4** | ⚠️ just misses |
| FMRec design vs MEMCF `full` | **-0.0000** | [-0.0067,+0.0065] | 2/4 | tied, well-powered |

Notes:
- "lessons beat a bare profile" is now **significant in two independent sets**
  (+0.0128 in mainpaper_rebuild_v2, +0.0127 here) — the earlier disagreement with
  `1k_paper` is outweighed.
- real-vs-shuffled is +0.0060 in *both* sets with 4/4 direction and a CI that only
  just touches zero. Likely a small real effect that more datasets would resolve.
- the two mechanisms are now a **well-powered tie**, not an unresolved question.
- integrity: 0 baseline mismatches on all 4 datasets; within-batch vs cross-batch
  offset 0.0002, so cross-batch contrasts are sound.

## 5. Methodology facts that must not be re-learned the hard way

1. **`--use_memory` defaults to `True`.** `action="store_true", default=True`.
   Omitting it does NOT disable memory; a no-memory arm needs `--no_use_memory`.
   Confirm from the output filename: the no-memory path emits a `memcf_nomemory_`
   prefix. (The 2026-09-04 handoff got this wrong and would have produced a
   counterfeit baseline had the jobs not crashed.)
2. **N=100 and N=200 are too small.** Taking the 1,000-user data where memory
   beats no-memory by a significant +0.0332 and subsampling to 100 users gives
   **+0.0002, n.s.** CI width 0.062 at N=100 vs 0.020 at N=1,000. Never conclude
   "no effect" from a 100/200-user run. **1,000 is the working minimum.**
3. **Cross-batch offset is small.** Re-running an identical no-memory config in a
   later batch differed by +0.0051 (100u) and +0.0010 (200u), both n.s. — so
   cross-batch contrasts are usable, but an in-batch baseline is preferred.
4. **`--candidate_negative_mode` does not change the eval candidate set.** In
   `eval_only` it only affects memory construction and the run name; eval
   negatives come from precomputed `test_neg`, truncated by
   `--max_negative_candidates`. So `--max_negative_candidates 9` gives a
   10-candidate task, but "random vs hard negatives" cannot be switched at eval
   time without rebuilding runtime data.
5. **Always verify pairing** before a paired test: candidate sets, ground truth
   and baseline metrics must match across arms (0 mismatches).
6. **`selected_memory_facts_total` is NOT the number of injected facts.** It counts
   the *candidate pool*, which is built with a 2x overflow buffer for token packing:
   `overflow_limit = max_memory_facts * 2` (experiment_with_fmrec_topk.py ~L9123).
   The real cap is enforced later by `pack_memory_facts(max_facts=...)` (L74:
   `if max_facts > 0 and len(facts) >= max_facts: break`). So `--max_memory_facts 1`
   reads **1.87** in the diagnostic but injects exactly **1** fact. The `same1` arm
   was therefore valid. Only trust `pack_memory_facts` output / `packing_audit`
   when reporting facts-per-user in the paper; re-verify the other arms' counts
   (same_user 2.44, topk 3.00, cross_user_only 2.79) the same way before publishing.
7. **Two incompatible candidate-set families exist. NEVER cross them.**
   `evaluation_results_fmrec_ablation_1000u` is a **20-candidate** task
   (no-memory N@10 = 0.3956). Every other root — `proto7_gemma_1000u`,
   `same1_gemma_1000u`, `p1_gemma_1000u`, `screen300_gemma` — is **10-candidate**
   (no-memory N@10 = 0.6035). Pairing across the two produced a spurious
   **+0.2411 N@10 "effect"** that was pure task difficulty; `paircheck.py` showed
   300/300 candidate-list mismatches with 0 ground-truth mismatches.
   **The 10-candidate no-memory baseline is `proto7_gemma_nomem`.**
   A second, subtler axis: `screen300` used `tb700`, all 1,000-user roots use
   `tb420`. Match token budget too, or the contrast is confounded.
   Run `audit_cands.py` before any new cross-root contrast.
8. **`memrec_full` is an INVALID run — the vLLM engine crashed mid-run.**
   Its logs contain **36,457 `Error in LLM Reranker ...: Connection error.`**
   plus 4 `EngineCore encountered an issue` (vLLM 500s). `reranker_llm.py:336`
   catches every exception and returns the **original candidate order** with
   synthetic descending scores ("Fallback: original candidate order"), printing
   to stdout but never raising. Consequence: `predictions == candidates` for
   **100.0% of users on all 6 datasets**, `facets == []`, and metrics that land
   exactly on the random-ranking reference (H@1 .039-.059 vs .050; H@3 .134 vs
   .150; H@5 .246 vs .250; H@10 .489 vs .500).
   **Do NOT interpret `full - read` as the effect of Stage-W.** The full arm
   measures a dead LLM backend. It must be re-run before any Stage-W claim.
   `memrec_read` and `no_memory` logs contain **no** such errors (identity rate
   3-13%, facets 1000/1000), so those two arms are sound.
   Detector: `mr_fallback.py` — flags any arm whose predictions equal its
   candidate order at a high rate. Run it on every new MemRec batch.

---

## 6. Server operations

```
ssh 24nam.nh@10.140.24.17          # lands in a container; whoami = ubuntu
```

- **Network:** 10.140.24.17 is lab-network-only. Probe port 22 before planning work.
- **TTY forced:** scp/sftp and any non-tty `ssh host 'cmd'` fail
  ("the input device is not a TTY"). Use `printf '%s\n' 'cmd' 'exit' | ssh -tt host`.
  File transfer = base64 chunked through the tty (`fold -w 3000`, batches of 30);
  `fold` omits the final newline, so append one or the last chunk is dropped.
  Pull files back with a `^B64LINE:` prefix anchored at line start — without the
  anchor, grep also matches the echoed command and corrupts the stream.
- **No `curl`, no `nvidia-smi`** in the container (NVML init error), but **CUDA
  works** — probe endpoints with `python3 urllib` instead.
- **GPU policy (hard rules from the user):**
  - **at most 2 GPUs at a time**
  - **shut vLLM down whenever nothing is running — an idle vLLM risks a server ban**
  - vLLM is configured `--max-num-seqs 32`; ~32 concurrent sequences is the cap.
- **Start vLLM** (`/tmp/start_vllm.sh`):
  ```bash
  export CUDA_VISIBLE_DEVICES=4; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
  /home/ubuntu/miniconda3/bin/vllm serve \
    /home/ubuntu/shared/hieu.tm2/models/Qwen2.5-7B-Instruct \
    --port 8000 --served-model-name gpt-3.5-turbo-16k-0613 \
    --gpu-memory-utilization 0.95 --max-num-seqs 32 --max-model-len 16384
  ```
  Stop with `pkill -f "vllm serve"`.
- **`chat_api_base` must be an exported env var**, never a `--chat_api_base` CLI
  flag — as a flag it causes a silent instant argparse failure that looks like a
  fast success.
- Hugging Face **is** reachable from the server (so Gemma can be downloaded).

### Key paths

```
DATA=/home/ubuntu/24nam.nh/video_games_data
$DATA/MEMCF                                          # code
$DATA/runtime_data_rebuild_v2_seed42                 # MEMCF_DATA_ROOT
$DATA/agent_memory_memcf_i_temporal_1k_rebuild_v2/<DS>/i_temporal_nuser1000_8shards.memory.json
$DATA/lgcn_embeddings_1k_rebuild_v2/<DS>.lgcn_embeddings.json
$DATA/evaluation_results_memcf_a_mainpaper_rebuild_v2   # 1k ladder: nomem/profile/same/shuffled/full(+arb)
$DATA/evaluation_results_memcf_1k_paper                 # 6-dataset ladder
$DATA/evaluation_results_memcf_fmrec_compare_100u       # 12-run A0/consensus/fmrec_topk, 100u
$DATA/evaluation_results_fmrec_ablation_200u            # FMRec-design ladder, 200u
$DATA/evaluation_results_fmrec_ablation_1000u           # FMRec-design ladder, 1000u  <- in flight
```

### Blocked: running the real FMRec on the server

Missing all of: `instructrec-books.inter` (`memrec/data/processed/` holds only
`.gitkeep`), LightGCN embeddings for books, Unsloth, and a Gemma-3-4B checkout.
The local `failure_memory.jsonl` is 10.1 MB — impractical to push through the
tty. `eval_user_sample_1k_instructrec-books.json` **is** present on the server.

---

## 6b. STATE AS OF 2026-09-15

**DONE 2026-09-15 (8/8):** `noself1` — 8 jobs (noself1 + noself1rand x 4 datasets),
n=1,000, Gemma-3-4B on **gpu5 port 8001**, eval root
`evaluation_results_noself1_gemma`. A server-side monitor
(`monitor_ns1.sh`, log `monitor_ns1.log`) kills vLLM the moment the 8th summary
lands, so no idle GPU server is left behind if the client loses the lab network.

**RESULT** — C12: `noself1` - `nomemory` = **+0.0258 N@10, SIG 4/4** (transfer is
real); `noself1` - `same1` = **-0.0104 N@10, SIG 0/4** (strictly inferior).
C13: `noself1` - `noself1rand` = **-0.0011 N@10, n.s., 1/4** (neighbour identity
worthless). RQ2b is unsupported; the old +0.0133 topk-vs-random was a packing
interaction, not collaborative retrieval.

**Config**: byte-for-byte `same1`'s (gk3 nk10 mf1 mw55 tb420 negcandidate_hard),
only `--graph_retrieval_scope` differs. That makes `noself1 - same1` a clean
ownership contrast: one fact, slot 1, differing solely in who wrote it.

**Completed since the last entry**
- `screen300` 24/24 → C10 (neighbour choice is worthless without the personal
  anchor) and C11 (no dose-response over K=1..5). Both negative for RQ2.
- `p1` 8/8 → C6: LightGCN == raw co-interaction; the graph encoder is removable.

**Network**: the lab network was unreachable from roughly 2026-09-13 18:00 to
2026-09-15 02:20. During that window a vLLM server sat idle on gpu5 because it
was started minutes before the outage. Lesson: **start vLLM and launch the jobs
in the same SSH session**, never in two steps.

---

## 7. Status of runs

| Batch | N | Arms | State |
|---|---|---|---|
| `memcf_1k_paper` | 1,000 x6 ds | nomem, profile, same_user, full(+arb) | ✅ analysed |
| `mainpaper_rebuild_v2` | 1,000 x4 ds | + shuffled_memory | ✅ analysed |
| `fmrec_compare_100u` | 100 x4 ds | A0, consensus, fmrec_topk | ✅ analysed (underpowered) |
| `fmrec_ablation_200u` | 200 x4 ds | topk, topk_random, nomem | ✅ analysed (underpowered) |
| `fmrec_ablation_1000u` | 1,000 x4 ds | topk, topk_random, nomem | ✅ analysed — §4 |
| `proto7_qwen_1000u` | 1,000 x4 ds | 7 arms, FMRec protocol, Qwen | ✅ **analysed — §12** |
| `proto7_gemma_1000u` | 1,000 x4 ds | 7 arms, FMRec protocol, Gemma | ✅ **analysed — §12** |

---

## 8. What is still missing for the paper

| Gap | Which system | Status |
|---|---|---|
| Beat the strongest baseline (MemRec) | FMRec | ❌ currently -2.5% on Books |
| Datasets 2 and 3 | FMRec | ❌ Goodreads/MovieTV empty |
| Any ablation at all | FMRec | ❌ §4.3 empty |
| Statistical testing / CIs | FMRec | ❌ point estimates only |
| RQ2 positive result | both | ❌ **settled null on both halves** (§4) — reframe RQ2 as a negative result |
| Defend the *failure*-memory mechanism | both | ⚠️ shuffled control recovers 82% |
| RQ3 (retrieval config + backbone) | both | ⚠️ **backbone now varied — see §12**; k-sweep still 2/12 |

---

## 9. Analysis scripts

Kept in the session scratchpad; re-transfer to `/tmp` on the server as needed.

| Script | Does |
|---|---|
| `analyze_fmrec_compare.py` | 3-arm A0/A/B paired comparison + bootstrap |
| `analyze_rq_ablation.py` | full ladder by arm, decomposed per RQ |
| `analyze_fmrec_ablation.py` / `_1000u.py` | FMRec-design ladder vs reference arms |
| `compare_100u_vs_1k.py` | reproducibility + the power/subsampling analysis |
| `all_arms_same_users.py` | every arm on one common user set and baseline |
| `memcf_paper_metrics.py` | MEMCF arms in the FMRec paper's metric set (H/N@1,3,5) |
| `power_curve.py` | empirical detection rate vs N |

Standing convention: paired bootstrap **5,000 resamples, seed 2027**, plus a
5,000-flip permutation test; report n, both means, the diff, the CI, and whether
it excludes zero — never point estimates alone.

## 10. Published report

Artifact (live, same URL across updates):
https://claude.ai/code/artifact/5f350296-87ce-421a-b7dc-ef413dd3edce

---

## 11. Session log

- **2026-09-09** ran `fmrec_ablation_200u` (12 runs) then `fmrec_ablation_1000u`
  (12 runs). Lab network dropped mid-run; jobs survived (nohup+disown) but the
  shutdown monitor could not reach the server. On reconnect at 00:31 the batch
  was complete (12/12) and Qwen was shut down automatically.
- **Phase 2 (Gemma) attempt failed**: another user took gpu2 during the 8GB
  download, so vLLM aborted at init with
  `Free memory on device (2.28/23.68 GiB) ... less than desired (0.9, 21.31 GiB)`.
  No GPU memory was held. **The model is already downloaded** at
  `/home/ubuntu/24nam.nh/models/gemma-3-4b-it` (8.1G) — no need to re-fetch.
  Re-check free memory immediately before starting, and lower
  `--gpu-memory-utilization` if the card is shared.
- **Verified the port is faithful**: FMRec's `best_lesson_for_user` returns
  `rows[0]`, but `load_failure_memory` pre-sorts by `(-confidence, memory_id)`,
  so it equals MEMCF's `min(key=(-confidence, memory_id))`. The only deliberate
  difference is the fact cap (FMRec allows 4 = 1 self + 3 collab; the port used 3
  to keep every arm on one budget).
- **Still not run**: Phase 1 / Phase 2 (MEMCF ladder under FMRec's protocol:
  `--max_negative_candidates 9`, `--max_memory_facts 4`). Launcher is on the
  server at `/tmp/launch_fmrecproto.sh <port> <tag>`.

---

## 12. THE BACKBONE RESULT  (2026-09-10, the most important finding so far)

56 runs, 7 arms x 4 Amazon datasets x 1,000 users, **all under FMRec's protocol**
(10 candidates via `--max_negative_candidates 9`, 4 lessons via
`--max_memory_facts 4`), on two backbones in parallel on two GPUs.
Every arm is in the same batch, so every contrast is within-batch.

### The CF proof splits by backbone

`topk` vs `topk_random` = identical mechanism, identical budget, identical fact
count; the ONLY difference is whether the 3 neighbours are the most similar by
LightGCN cosine or 3 random users. This IS FMRec's RQ2 second half.

| Settings | Backbone | topk - topk_random | |
|---|---|---|---|
| MEMCF (20 cand, 3 lessons) | Qwen2.5-7B | -0.0012 [-0.0072,+0.0050] | ❌ null, 2/4 |
| FMRec (10 cand, 4 lessons) | Qwen2.5-7B | -0.0013 [-0.0062,+0.0035] | ❌ null, 1/4 |
| FMRec (10 cand, 4 lessons) | **Gemma-3-4B** | **+0.0101 [+0.0049,+0.0154]** | ✅ **SIGNIFICANT, 4/4** |

**The protocol is not what matters — the backbone is.** Two independent Qwen
measurements under two different protocols are both flat; Gemma is significant on
all four datasets. This is the first significant positive CF result in the project.

**FMRec's paper uses Gemma-3-4B** — i.e. the backbone on which its central claim
actually holds.

### The whole ladder, both backbones (NDCG@10, n=1,000 x 4 datasets)

| Contrast | Qwen2.5-7B | Gemma-3-4B |
|---|---|---|
| memory vs no memory | +0.0137 [+0.0058,+0.0218] ✅ 3/4 | **+0.0360** [+0.0276,+0.0444] ✅ 4/4 |
| memory facts over profile | +0.0109 [+0.0054,+0.0167] ✅ 4/4 | +0.0085 [+0.0028,+0.0141] ✅ 3/4 |
| **real vs shuffled lessons** | +0.0035 [-0.0019,+0.0087] ❌ 3/4 | **+0.0096 [+0.0040,+0.0152]** ✅ **4/4** |
| **CF: similar vs random** | -0.0013 ❌ 1/4 | **+0.0101 [+0.0049,+0.0154]** ✅ **4/4** |
| cross-user over personal-only | +0.0002 ❌ 2/4 | -0.0025 ❌ **0/4** |
| FMRec design vs MEMCF design | -0.0002 ❌ | +0.0002 ❌ |
| profile text alone | +0.0028 ❌ 3/4 | +0.0275 ✅ 4/4 |
| shuffled lessons alone | +0.0102 ✅ 3/4 | +0.0264 ✅ 4/4 |

### How to read it

1. **On Gemma the memory machinery is alive; on Qwen it is muted.** Every
   ingredient is larger on the weaker model. Two readings, both worth stating:
   genuine collaborative signal that a stronger model already has internally, OR
   a weaker model simply being more suggestible to injected text. The data here
   does not separate them.
2. **`real vs shuffled` flips to significant on Gemma (+0.0096, 4/4).** This is
   the mechanism claim that was borderline everywhere else. On its own backbone,
   failure-lesson *content* does matter.
3. **Cross-user over personal-only stays null on both** (+0.0002 Qwen; -0.0025
   Gemma, 0/4). This half of RQ2 has now failed in every configuration tested:
   2 protocols x 2 backbones x 4-6 datasets. Treat it as settled.
4. **The two designs remain tied on both backbones** (-0.0002 / +0.0002), with
   tight intervals. Mechanism choice is not what decides the paper.
5. **Protocol effect, isolated:** on Qwen, memory-vs-no-memory drops from +0.0334
   (20 candidates) to +0.0137 (10 candidates). The easier 10-candidate task
   leaves less headroom, so all memory effects shrink.

### Consequence for choosing a version

FMRec's claims look defensible **on Gemma-3-4B**, which is what its paper uses.
Any RQ2/RQ3 write-up must state the backbone dependence explicitly, and ideally
report both backbones -- that dependence is itself the RQ3 answer nobody has
published.

Result dirs: `evaluation_results_proto7_qwen_1000u`,
`evaluation_results_proto7_gemma_1000u`. Analyzer: `analyze_proto7.py <root> <tag>`.

---

## 13. Data & network facts (learned the hard way, 2026-09-10)

- **The InstructRec datasets exist** at the Google Drive folder linked from the
  MemRec README (published by github.com/agiresearch/iAgent):
  `https://drive.google.com/drive/folders/1-3kHU9D4IH210kSYL-m2cCgWbcY5ilBI`
  Needed per dataset: `<name>All_recagent.pkl` + `combined_<name>_asin_mapping.csv`,
  converted by `scripts/convert_all_instructrec.sh`. File ids are recorded in
  `fetch_iagent.sh` in the session scratchpad.
- **The server cannot download them.** Its firewall is an allowlist:
  - reachable: `huggingface.co`, `cdn-lfs-us-1.hf.co`, `github.com`,
    `objects.githubusercontent.com`, `pypi.org`, `storage.googleapis.com`
  - blocked (TLS EOF): **`googleusercontent.com`** (where Drive serves content),
    dropbox, zenodo, 0x0.st, transfer.sh, temp.sh, catbox
  So gdown lists the folder and downloads nothing. Any transfer in must go via
  HuggingFace or GitHub.
- Downloaded to the **local Mac** at `/Users/nambincoi/FM-REC/iagent_data`
  (~3.7 GB): Books (327M pkl + 567M csv), MovieTV (143M + 315M), GoodReads
  (659M + 1.5G), Yelp (93M pkl + csv). Transfer to the server is unresolved --
  tty base64 would take ~38 h for a single 567 MB file.
- **The server already has iAgent data for the Amazon domains** at
  `video_games_data/iAgent/data/` (`Software_1000uAll_recagent.pkl`,
  `Prime_Pantry_...`, `Video_Game...`, `Industrial_and_Scientific_...`, plus CDs
  and Digital_Music, each with its `combined_*_asin_mapping.csv`, all 1.6-7.3 MB).
  **These are exactly the 4 datasets everything else runs on**, so MemRec can be
  converted and run on identical data without any InstructRec transfer.
- The full MemRec repo IS on the server at `video_games_data/memrec` (src/ present),
  cloned locally at `/Users/nambincoi/FM-REC/memrec`.

### MemRec's unfair step (confirmed by reading the code)

`trainer_memrec.py::_evaluate_single_user` calls `rerank()` (line 327) and then
`write()` (line 353) **during evaluation on the test split**. The guard is
`if self.enable_stage_w and self.eval_feedback != 'none'`, and every shipped
config sets `eval_feedback: gt`. `_generate_feedback(mode='gt')` takes
`target_item` -- the ground truth -- and if it is in the ranked top-5, Stage-W
writes a memory recording that click into a **shared** graph that later test
users read from. That is test-label leakage during evaluation. FMRec and MEMCF
both run `--phase eval_only` against a frozen, read-only memory.

Fix needs no code change -- set `eval_feedback: none`. Recommended 3-arm ladder:
`gt` (reproduce published) / `random` (writes memory but no labels -- separates
leakage from extra memory updates) / `none` (fair, matches FMRec+MEMCF).
`warmup.enabled: true, rounds: 1` is legitimate (train data) -- leave it.
