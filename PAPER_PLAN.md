# Paper plan — conclusions so far and the experiments that remain

Updated 2026-09-15. Companion to `COMPACT_HISTORY.md` (raw numbers) and
`RQ2_PLAN.md` (RQ2 diagnosis). This file is the decision document.

---

# PART 1 — What is established

All figures: Gemma-3-4B unless stated, n=1,000 per dataset, 4 Amazon datasets,
paired bootstrap 5,000 resamples seed 2027.

### C1. Failure memory works ✅ (RQ1, first half)
Memory vs no memory: **+0.0362 N@10 / +0.0473 N@5 / +0.0540 H@1**, significant,
**4/4 datasets**. On Qwen2.5-7B: +0.0135 N@10, significant, 4/4. Replicated
across 2 protocols x 2 backbones x up to 6 datasets. **This is the solid result.**

### C2. The benefit saturates after the FIRST fact ⭐ *the key mechanistic finding*
| step | H@1 | N@5 | N@10 | |
|---|---|---|---|---|
| 0 -> 1 fact | +0.0558 | +0.0458 | +0.0361 | ✅ SIG |
| 1 -> 2.4 facts | +0.0032 | +0.0030 | +0.0023 | n.s. |
| 2.4 personal -> 1 personal + 2 cross | -0.0050 | -0.0015 | -0.0023 | n.s. |

**94-95% of the entire memory effect lands in the FIRST SINGLE fact.**
(`--max_memory_facts 1` injects exactly one: `pack_memory_facts` hard-caps it.
The `selected_memory_facts_total` diagnostic reads 1.87 because it counts the
candidate pool, which carries a 2x overflow buffer for token packing.) Every later
slot is flat, whoever owns it. This explains every cross-user null at once: the
capacity is spent before cross-user content is ever reached.

### C3. Lesson *content* matters — but only on Gemma
real vs shuffled lessons: **+0.0098 to +0.0154, SIG, 4/4** on Gemma;
**n.s. on all metrics** on Qwen.

### C4. The backbone gap is NOT a headroom artifact ⭐
No-memory scores are nearly identical (N@5: Qwen **0.4523** vs Gemma **0.4509**),
yet Gemma gains ~3x more from memory. So the difference is not that Gemma is
weaker — it is that Gemma *uses* injected context and Qwen largely ignores it.

### C5. Cross-user transfer does not add value ❌ (RQ2, first half)
- on top of personal memory: n.s. everywhere, 0-2/4 favourable
- pure + owner-filtered + guaranteed relevant (`oracle`): **-0.0249 vs personal,
  SIG, 0/4** — worse than shuffled lessons (-0.0128, 0/4)
- monotone dose-response: the purer the cross-user content, the worse
  `personal 0.0000 > FMRec -0.0023 > cross(leaky) -0.0051 > random-nbr -0.0124 > pure+relevant -0.0249`

### C6. Neighbour selection matters, but LightGCN does not ⭐ (RQ2, second half)
| contrast | N@10 | verdict |
|---|---|---|
| LightGCN vs random | +0.0101 | ✅ SIG 4/4 |
| LightGCN vs most-prolific | +0.0082 | ✅ SIG 3/4 |
| **LightGCN vs raw co-interaction count** | **+0.0011** | ❌ **n.s. 2/4** |

Choosing neighbours who overlap with *this* user is real and significant. The
pretrained LightGCN embedding contributes nothing over counting shared items.

### C7. Only 1.17% of cross-user lessons are even applicable
A cross-user lesson mentions an item in the recipient's candidate set **1.17%**
of the time; MEMCF lessons name exact product titles in **99.9%** of cases.
Attribute-level phrasing would reach **35-55%** applicability (>=2 shared title
tokens); category level is useless (one category covers 91-100% of items).

### C8. The two designs are equivalent
FMRec top-K vs MEMCF `full`: tied in 2 of 3 configurations
(-0.0002 / +0.0002, CI +/-0.005); +0.0085 SIG in the third, driven by one dataset.

### C9. MemRec's Stage-W leakage is real in code but ~0 in effect
`gt` vs `none`: **+0.0029** mean, 3/4, and `random` performs the same as `gt`.
Verified fair: both eval paths honour `eval_feedback: none`, and warmup targets
the validation item (T-1), never the test item.

---

# PART 2 — What is missing

| Gap | Blocks | Status |
|---|---|---|
| **G1** A positive RQ2 result | RQ2 | **CLOSED**. C12: transfer is real (+0.0258 vs no-memory, 4/4) but strictly inferior to personal (-0.0104, 0/4). C13: neighbour identity n.s. RQ2b is unsupported |
| **G2** Beating the strongest baseline | RQ1 | MemRec numbers not comparable — different candidate sets, no no-memory arm, different backbone |
| **G3** Lesson abstraction (E6) | RQ2 mechanism | 1.17% applicability caps everything; untested |
| **G4** Real FMRec on InstructRec | external validity | blocked: server firewall denies googleusercontent.com |
| **G5** Dose-response on K | RQ2/RQ3 | **done → negative** (C11): K=1→5 gives +0.0008 N@10, n.s. |

---

# PART 3 — Experiment plan

## screen300 results (2026-09-13, 24/24 complete, all 10-candidate, tb700)
Only within-batch contrasts are valid here; the cross-root ones were discarded
after `paircheck.py` exposed a 10-vs-20 candidate mismatch (COMPACT_HISTORY trap #7).

**C10. Removing the personal anchor does not make neighbour choice matter.**
`noself3` (3 cross facts, nearest neighbours) vs `noself3rand` (random neighbours):
H@1 +0.0033, N@5 +0.0027, N@10 +0.0010 — **all n.s.**, 2-3/4 datasets.
So even with cross-user content occupying every slot, *which* neighbour supplied
it is worth nothing. This is the strongest evidence yet against the CF premise:
it holds where the personal anchor can no longer mask a cross-user effect.

**C11. No collaborative dose-response (G5 answered, negatively).**
K sweep, budget K+1, pooled N@10: k2-k1 -0.0032 n.s. | k3-k2 +0.0047 SIG |
k5-k3 -0.0007 n.s. | **k5-k1 +0.0008 n.s.** The one significant step is
non-monotone and cancels: adding four more neighbours' lessons changes nothing.
A real collaborative signal would accumulate; this does not.

## RESOLVED 2026-09-15 — `noself1`, the decisive RQ2 test (n=1,000, 8/8)
Config identical to `same1` (gk3 nk10 mf1 mw55 tb420); only the scope differs.
One fact, slot 1, differing solely in **who wrote it**. Pairing verified:
0 candidate-count and 0 ground-truth mismatches.

**C12. A stranger's lesson helps — but strictly less than your own.**
| contrast | H@1 | N@5 | N@10 | verdict |
|---|---|---|---|---|
| `noself1` - `nomemory` | +0.0365 | +0.0331 | **+0.0258** | SIG, 4/4, p=0.0002 |
| `noself1` - `same1` | -0.0192 | -0.0126 | **-0.0104** | SIG, **0/4**, p=0.0004 |

Cross-user failure knowledge **does** transfer: one stranger's lesson beats no
memory on every metric, all 4 datasets. But at matched budget and matched slot
it is significantly *worse* than the user's own lesson, on 0/4 datasets
favourable. Transfer is real and strictly inferior — not a null, a **deficit**.

**C13. Neighbour identity is worth exactly nothing (the OPEN TENSION, resolved).**
`noself1` - `noself1rand`: H@1 -0.0018, N@5 -0.0019, N@10 -0.0011 — **n.s. on
every metric**, 1/4 datasets, with the tightest CIs in the whole study (+-0.002).
This is the cleanest possible design: anchor-free, budget-1, config-matched.
A randomly chosen stranger is as useful as the LightGCN-nearest one.

The earlier `topk - random` = +0.0133 (SIG) therefore was **not** a
neighbour-selection effect. Explanation (1) stands: it is an interaction among
the 4 packed facts (diversity / duplicate avoidance), not collaborative
retrieval. **The Table B RQ2b row must be overridden to "unsupported".**

*Caveat to keep:* at budget 1 both arms pack the highest-confidence lesson from
their 3-neighbour pool, so selection has less room to act than at budget 4. The
claim is therefore "neighbour identity does not matter in the only slot that
moves the metric", which is what C2 says is the slot that counts.

## Tier 1 — status 2026-09-15
| Exp | Arms | n | Answers |
|---|---|---|---|
| ~~noself3~~ **done** | 3 cross facts, no anchor, + random control | 300 | → **C10, negative**: neighbour choice worth nothing without the anchor |
| ~~noself1~~ ⭐ **DONE 2026-09-15** | **exactly 1** cross-user fact in slot 1, + random control | 300 | G1, *the decisive form*. C2 says only slot 1 moves the metric, so the clean test holds budget at 1 and varies only **whose** lesson fills it: `noself1` vs `same1`. `noself3` cannot do this — it injects 3 facts against `same1`'s 1, confounding budget with ownership. Launcher: `launch_noself1.sh` |
| ~~K sweep~~ **done** | K = 1,2,3,5, budget K+1 | 300 | → **C11, negative**: no accumulation, K=1→5 is +0.0008 n.s. |

## Tier 2 — next, ordered by value
**T2.1 Confirm whatever Tier 1 finds, at n=1,000.** 300 users cannot resolve a
+0.010 effect (threshold ~+/-0.019); 300u is screening only.

**T2.2 MemRec made comparable (G2).** Add `enable_stage_r: false` — MemRec's own
no-memory baseline — and re-run on **Gemma**. Absolute scores stay incomparable
(different candidate sets) but *lift over its own baseline* is directly
comparable to ours. 8 runs.

**T2.3 Attribute-level re-distillation (G3).** Rewrite lessons over transferable
attributes — product type, brand, feature — instead of exact titles, and fill the
schema fields that already exist but are empty (`applies_if` is blank,
`failure_type` is a constant). Then re-run the 7-arm ladder unchanged, so the
contrast isolates abstraction exactly as `topk` vs `topkrand` isolated selection.

**T2.4 Drop LightGCN (C6).** Since raw co-interaction ties with the learned
embedding, report the simplification and keep the cheaper mechanism, or show a
setting where the embedding does earn its place.

## Tier 3 — only if Tier 2 opens a door
- failure-space retrieval (embed lessons, not users)
- prompt-format ablation: explicit source labels, personal-first ordering
- third backbone to separate "uses context" from "suggestible"

---

# The paper this currently supports

> Failure memory improves LLM reranking (+0.036 NDCG@10, 4/4 datasets), but
> **the benefit saturates after roughly two facts**, and essentially all of it
> comes from the user's *own* failures. Cross-user transfer adds nothing on top
> and actively hurts when it is made pure and relevant. Collaborative neighbour
> selection is nonetheless real — picking users who overlap with the target beats
> random or popular selection — though a pretrained LightGCN embedding performs
> no better than counting shared items. All of these effects appear on
> Gemma-3-4B and vanish on Qwen2.5-7B, despite the two backbones scoring
> identically without memory.

Honest, fully controlled, and novel in three places: the saturation curve, the
backbone dissociation at equal baseline, and the LightGCN-is-unnecessary result.
What it is *not* is the paper the current abstract promises.
