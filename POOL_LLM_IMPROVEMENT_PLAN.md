# Improving `pool5_llm` — evidence review and proposed changes

Written 2026-09-18. `pool5_llm` (top-5 LightGCN donors, pool every lesson, LLM
applicability selector, neutral to source, L=3) is now the main FMRec method.
This file lists what the runs so far say about it and what to change next.

## 1. What the runs say

### 1a. It clears the bar on Gemma, not on Qwen
Gemma 20-cand, the 2 uncontaminated datasets (Software, Video_Game), paired:

| vs | N@1 | N@3 | N@5 | N@10 | |
|---|---|---|---|---|---|
| vanilla | +0.0975 | +0.0962 | +0.0866 | +0.0787 | SIG 2/2 |
| shuffled | +0.0220 | +0.0257 | +0.0187 | +0.0199 | SIG 2/2 |
| random donors | +0.0210 | +0.0190 | +0.0141 | +0.0133 | SIG 2/2 |
| personal only | +0.0120 | +0.0110 | +0.0057 | +0.0067 | 2/2, n.s. |

Qwen 20-cand, 4 datasets: every memory arm sits within 0.0024 N@10 of every
other; `random` is the best (0.3684). Nothing beats `random` or `personal`.
Same verdict as C3/C4/C6/C13: Qwen ignores which lessons it is given.
**Optimise on Gemma. Report Qwen as RQ3 robustness (negative).**

### 1b. The collaborative signal is real, and only appears with a filter
`pool5_llm - pool5_rand_llm` = +0.0195 N@3, +0.0139 N@5, +0.0111 N@10, SIG
2/2. Same selector, same pool size, only the donors differ. Earlier designs
(1 best lesson per donor, C13) found no donor effect; pooling everything and
filtering by decision relevance is what makes similarity pay.

### 1c. The LLM filter does not beat a term-coverage rule
`pool5_llm - pool5_heur`: mixed sign, n.s. on the clean datasets; N@5 -0.0077
SIG on the (contaminated) 4-dataset set. LLM wins at N@1 (+66.3% vs +62.9%
over vanilla) and loses from N@5 down. Two calls/user vs one.

### 1d. The selector under-fills — the clearest leak found (`sel_diag.py`)
| Gemma | kept 0 | kept 1 | kept 2 | kept 3 | mean |
|---|---|---|---|---|---|
| Software | **117 (11.7%)** | 35 | 202 | 646 | 2.38 |
| Video_Game | 51 (5.1%) | 72 | **562** | 315 | 2.14 |

Kept relevance p50 = 0.85, p10 = 0.75 — the LLM is confident about what it
keeps; it simply keeps too few. 5-12% of users receive **no memory at all**,
and C2 says the first lesson carries ~95% of the whole effect. `pool5_heur`
always fills 3; that alone can explain its N@5/N@10 edge.
Kept lists are already ordered by relevance (100%), so slot-1 ordering is fine.

### 1e. Own lessons are over-selected relative to their pool share
Software: own = 24% of the pool but 42% of kept; Video_Game: 25% -> 60%.
The source-blind selector finds own lessons more applicable on its own — which
matches C12 (own lesson > stranger's at slot 1). But it is not guaranteed:
`topk` (own lesson always in slot 1) still edges `pool5_llm` on N@10.

### 1f. Prompt length is dominated by the memory block (`why_fail.py`)
66-76% of the selector prompt is the memory rows; each row repeats the item
titles three times (in the lesson text, in correct/wrong_item_title, in the
match lists). Industrial titles average 87 chars vs 46 for Software, so the
same pool overflows 8k context on Industrial (57% fallback) and not on
Software (0%). Root cause is the item-specific lesson format (C7).

### 1g. pool size 3 vs 5 is a tie; unfiltered pool is worse than topk
`pool3_llm ~ pool5_llm` (n.s.). `pool5_none - topk` = -0.0068 N@10 SIG 0/4.
Bigger pool helps only through the filter; extra donors beyond 3 add little.

## 2. Proposed changes, ranked by expected gain x confidence / cost

### P1. Backfill to L — never leave slots empty  (tiny code, no extra cost)
When the selector returns fewer than L rows above `min_relevance`, fill the
remaining slots from the selector's own relevance ranking below threshold, and
if that is still short, from term coverage. Targets 1d directly: the 5-12% of
users with zero memory get their slot-1 lesson back.
Expected: recover most of `pool5_heur`'s N@5/N@10 edge while keeping the LLM's
N@1 edge. Flag: `--memory_selector_backfill {none,relevance,coverage}`.
Test: `pool5_llm_bf` vs `pool5_llm`, 4 datasets Gemma. 4 jobs.

### P2. Personal anchor + LLM-selected cross  (small code, no extra cost)
Guarantee the user's best own lesson in slot 1; let the LLM pick the other
L-1 from the cross-user part of the pool (source-blind among cross rows).
Backed by C12 (own > stranger at slot 1) and 1e (the selector already prefers
own lessons but not always). Combines `topk`'s anchor with 1b's collaborative
gain. Flag: `--memory_selector_anchor_self`.
Risk: if own lesson is irrelevant to this decision it now occupies a slot
unconditionally — the very thing the selector was meant to avoid. Mitigate:
anchor only if own lesson coverage > 0.
Test: `pool5_llm_anchor` vs `pool5_llm`. 4 jobs.

### P3. Slim the selector prompt  (no quality change expected; -40% tokens; kills the overflow for good)
Per memory row keep: `memory_id`, `memory_fact` capped at 220 chars,
`candidate_matches[:4]`, `history_profile_matches[:4]`, `confidence`. Drop
`correct_item_title`/`wrong_item_title` (already inside the fact text),
`sources`, `overgeneralization_risk`, the direct_* booleans. Drop the
`rejected` list from the required JSON output (halves completion tokens).
Then `max-model-len 8192` is enough again and the selector call falls from
~3.7k to ~2.2k tokens.
Changes the selector input, so it must be re-run on all 4 datasets and
compared to the 16k-context `pool5_llm` to show no loss. 4 jobs.

### P4. Hybrid: coverage pre-filter, then LLM  (cheaper, may be better)
Rank the pool by term coverage, hand only the top-8 to the LLM, LLM picks 3.
Halves the selector prompt independently of P3, removes obviously irrelevant
rows the LLM would reject anyway (1c shows coverage is a strong signal), and
lets the LLM spend its judgement where it matters. If P4 ~ P1 in quality it
is the cheaper production choice.
Flag: `--memory_selector_prefilter_k 8`. 4 jobs.

### P5. Re-distil lessons at attribute level  (largest upside, largest cost)
C7: 99.9% of lessons name a specific product; only 1.17% of cross-user lessons
mention an item in the recipient's candidate set. 1f: the same property blows
the prompt budget. dangkh's own distiller prompt (`build_fmrec_lessons.py`
L339-354: "ONE transferable lesson", <=45 words) is the reference. Re-distil the
4 datasets (~2.8k tok/user, ~4k LLM calls per dataset), rebuild the memory
files, re-run `pool5_llm` + controls. This is the only change that can move
the cross-user applicability ceiling; everything above optimises within it.
~16 build jobs + 20 eval jobs.

### Not proposed
- More donors (N=8+): 1g says extra donors add little; costs prompt length.
- Tuning `min_relevance`: P1 subsumes it without a new hyper-parameter.
- Qwen-specific work: 1a — no memory arm separates from random on Qwen.

## 3. Order
1. Finish the 8-job rerun (in flight, 16k context) -> clean 4-dataset baseline
   for `pool5_llm` and `pool5_rand_llm`.
2. P1 + P2 + P4 as three 4-job batches on Gemma (12 jobs, one GPU, ~3 h).
3. P3 once P1/P2/P4 have picked the winner, so the slim prompt is applied to
   the final design only.
4. P5 if time allows before the deadline; it is the paper's G3 anyway.

Every new arm: `paircheck` 0 mismatches, `sel_diag` kept-distribution, fallback
= 0, before any number is read.
