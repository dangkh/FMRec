# §4.3 draft — RQ2 (cross-user transfer) and RQ3 (robustness)

Status 2026-09-15. Written against measured numbers only. `noself1` has now
landed, so **§4.3.2 is settled and rewritten below** — the verdict is that RQ2b
is unsupported and the old Table B row must be overridden.

Setup for every number below: Gemma-3-4B, n=1,000 users per dataset, 4 Amazon
datasets, 10 candidates, paired bootstrap 5,000 resamples seed 2027, stratified
pooling within dataset. "SIG" = 95% CI excludes zero. "dir k/4" = datasets
agreeing in sign.

---

## 4.3.1 Does failure knowledge transfer across users? (RQ2a)

Read naively, the answer is yes: cross-user lessons alone beat the memory-free
reranker on all five metrics (N@5 +0.0434, CI [+0.0317,+0.0549], 4/4). But that
contrast cannot separate transfer from the mere presence of extra text, and a
control battery shows it does not survive:

| contrast | what it isolates | N@5 | verdict |
|---|---|---|---|
| cross-user only − no memory | any benefit at all | +0.0434 | SIG 4/4 |
| cross-user only − shuffled lessons | content, not text volume | +0.0115 | SIG 4/4 |
| FMRec (personal + cross) − personal only | **is cross-user additive?** | −0.0015 | **n.s. 2/4** |
| owner-filtered cross-user − personal | purified transfer | **−0.0249** | **SIG 0/4** |

The third row is the one that matters. Personal and cross-user memory are
**substitutes, not complements**: either alone delivers essentially the whole
gain, and combining them adds nothing. The fourth row is stronger still — when
the retrieval is forced to return only lessons written by *other* users, and
those lessons are guaranteed relevant to the candidate item, performance drops
below the personal-only arm and below shuffled lessons (−0.0128). The effect is
monotone in purity: the more strictly cross-user the content, the worse.

The mechanism is capacity, and it is measurable. Ablating the memory budget:

| budget step | H@1 | N@5 | N@10 |
|---|---|---|---|
| 0 → 1 fact | **+0.0558** | **+0.0458** | **+0.0361** SIG |
| 1 → 2.4 facts | +0.0032 | +0.0030 | +0.0023 n.s. |
| 2.4 personal → 1 personal + 2 cross | −0.0050 | −0.0015 | −0.0023 n.s. |

Roughly 94–95% of the entire memory effect is delivered by the **first single
lesson**. Every later slot is flat regardless of who wrote it. Because the
retrieval rule always spends slot 1 on the user's own lesson, cross-user content
is only ever evaluated in slots that do not move the metric — which explains
every cross-user null at once, without appealing to any property of the lessons.

A dose-response test confirms there is nothing accumulating in the later slots.
Holding composition at 1 personal + K cross-user lessons:

K=1→2: −0.0032 | K=2→3: +0.0047 SIG | K=3→5: −0.0007 | **K=1→5: +0.0008, n.s.**

The single significant step is non-monotone and cancels out. Adding four more
neighbours' lessons changes nothing.

## 4.3.2 Does collaborative similarity drive retrieval? (RQ2b)

We answer this with a design that removes every confound at once. The
`noself1` arm is configured identically to the personal-only arm — same
retrieval depth, same fact budget of one, same word and token limits, same
candidate sets (verified: 0 candidate and 0 ground-truth mismatches across all
4,000 paired users) — and differs only in *whose* lessons the retriever may
return. Because §4.3.1 establishes that essentially the whole memory effect is
delivered by the single first lesson, a budget of one places the contrast
exactly where the metric is known to move.

Two results follow.

**Cross-user knowledge transfers, but at a deficit.**

| contrast | H@1 | N@5 | N@10 | |
|---|---|---|---|---|
| one stranger's lesson vs no memory | +0.0365 | +0.0331 | **+0.0258** | SIG, 4/4 |
| one stranger's lesson vs one's own | −0.0192 | −0.0126 | **−0.0104** | SIG, **0/4** |

A single lesson written by another user beats the memory-free reranker on every
metric and every dataset. Transfer is therefore real, and reports of a pure null
would be wrong. But placed in the same slot under the same budget, it is
significantly *worse* than the user's own lesson, with no dataset favouring it.
The honest statement is a **deficit, not an absence**: other users' failures
carry usable signal, and consistently less of it than one's own.

**Neighbour identity contributes nothing.** Replacing the LightGCN-nearest
neighbours with randomly chosen users changes nothing: H@1 −0.0018, N@5 −0.0019,
N@10 −0.0011, not significant on any metric, 1/4 datasets, with the tightest
confidence intervals in the study (±0.002). Combined with the earlier finding
that LightGCN cosine is statistically indistinguishable from raw co-interaction
counts, the collaborative component of the retriever is not doing work: neither
the learned embedding nor the neighbour ranking it induces affects the outcome.

This overrides a weaker earlier contrast. With four facts packed and a personal
anchor present, top-K retrieval beat random neighbours by N@5 +0.0133 (SIG).
That effect does not survive isolation: with the anchor removed and the budget
held at one, it disappears entirely. We therefore attribute it to an interaction
among the packed facts — diversity or duplicate avoidance — rather than to
collaborative retrieval, and we report RQ2b as **unsupported**.

## 4.3.3 Robustness (RQ3)

**Backbone.** The memory effect is strongly backbone-dependent, and this is not
a headroom artifact. The two backbones start from almost the same memory-free
score (N@5: Qwen2.5-7B 0.4523, Gemma-3-4B 0.4509), yet Gemma gains roughly three
times as much from memory (N@10 +0.0362 vs +0.0135, both SIG, 4/4). The
lesson-content control separates them further: real lessons beat shuffled ones
by +0.0098 to +0.0154 (SIG, 4/4) on Gemma but are **n.s. on every metric** on
Qwen. The larger model largely ignores the injected context; the smaller one
uses it. Any claim about failure memory must therefore be stated per backbone.

**Retrieval configuration.** The method is insensitive to K over 1–5 (above) and
to the neighbour ranker (LightGCN ≈ raw co-occurrence). It is sensitive to only
one knob, the memory budget, and there only between 0 and 1.

---

## Numbers still needed before this section is final
1. Re-verified facts-per-user from `pack_memory_facts`, not the pool diagnostic,
   for the arms quoted in the budget table.
2. A 10-candidate MemRec lift for the cross-system column (T2.2).
