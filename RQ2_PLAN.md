# Plan to close RQ2

Written 2026-09-10. Companion to `COMPACT_HISTORY.md` (which holds all measured
numbers). This file is only about the one thing still open.

---

## 0. Where the paper actually stands

| RQ | Status |
|---|---|
| **RQ1** effectiveness vs baselines | ✅ effectively done. Memory beats no-memory everywhere (+0.0137 to +0.0361, significant, up to 6/6 datasets). MemRec's published lead came from Stage-W test-label leakage; with Stage-W off it drops sharply (confirmed by the user's own earlier run, being re-confirmed here on 4 datasets x 2 backbones). |
| **RQ3** robustness | ✅ largely done. Backbone varied (Qwen2.5-7B vs Gemma-3-4B, full 7-arm ladder, 4 datasets, n=1,000). Retrieval configuration covered by 5 distinct scopes (`full`, `same_user`, `shuffled`, `topk`, `topk_random`). Only a K-sweep is thin (2/12 runs) — optional. |
| **RQ2** cross-user failure transfer | ⚠️ **half open** — the subject of this file. |

---

## 1. Precisely what is broken

RQ2 asks two things. They have different answers.

| Half | Experiment | Result | Verdict |
|---|---|---|---|
| **(a) Does cross-user transfer add value at all?** | `full` (own + others') vs `same_user` (own only) | Gemma **-0.0025, 0/4**; Qwen +0.0002, 2/4; and null in 3 earlier sets | ❌ **THIS IS THE BROKEN ONE** |
| **(b) Does collaborative similarity help choose whose lesson?** | `topk` (3 most similar) vs `topk_random` (3 random) | Gemma **+0.0101 [+0.0049,+0.0154], 4/4** | ✅ already works |

### The arm ranking that explains it (Gemma, NDCG@10, mean of 4 datasets)

```
same_user    0.5986   <- BEST: personal lessons only
topk         0.5964
full         0.5962
profile      0.5876
shuffled     0.5865
topk_random  0.5863   <- random neighbours collapse to the "meaningless text" level
nomem        0.5602
```

Reading: **similarity is protective, not productive.** Choosing similar neighbours
keeps you at the personal-only level; choosing random ones drags you down to the
shuffled/profile level. Nothing lifts you *above* personal-only.

So the sentence the paper must be able to defend is:
*"a lesson learned from user A improves ranking for a different user B."*
Right now the data says it does not.

---

## 2. Five hypotheses for why (a) fails

Ordered by cost to test, not by likelihood.

### H1 — Budget cannibalization (cheap to test)
The prompt carries a fixed 3-4 facts. Every cross-user lesson **displaces** a
personal one. If personal lessons dominate, cross-user can only dilute. This
would produce exactly the observed pattern: `same_user` best, `full` slightly
below it.
**Prediction:** make cross-user *additive* rather than substitutive and the
deficit disappears.

### H2 — Lessons are model-level, not user-level
FMRec's lessons read like *"The ranker prioritized familiarity over explicitly
stated desires for a light, feel-good experience."* That describes **the LLM's
failure mode**, not this user's taste. If lessons are largely user-independent,
then (i) another user's lesson carries no extra information -> (a) null, while
(ii) a topically mismatched one still distracts -> random neighbours hurt.
Both observations fit.

### H3 — Wrong similarity space  ← most promising for a positive result
LightGCN cosine measures **interaction** similarity: who buys similar items.
Transfer needs **failure** similarity: who makes the same *kind* of mistake.
Two users can have near-identical purchase histories and fail for unrelated
reasons. Nothing in the current pipeline ever compares failures to failures.

### H4 — Personal memory already saturates the population  ← test is FREE
On the MEMCF datasets ~98-100% of users have their own lessons. If almost
everyone already has personal memory, nobody *needs* to borrow, so the average
cross-user effect is ~0 by construction. Cross-user transfer should matter for
**cold users** — those with few or no personal lessons.
Note: FMRec's own Books coverage is **59.6%**, so 40% of its users have no lesson
at all. The MEMCF datasets may simply lack the population where transfer pays off.

### H5 — Selection is confidence-based, not relevance-based
Each neighbour contributes its single highest-*confidence* lesson, with no check
that the lesson relates to the current candidate set. MEMCF's consensus gate did
add relevance but overshot (1.6% cross-user coverage; 0% on two datasets). The
middle of that range — say 20-40% coverage — has never been tested.

---

## 3. Experiments, in the order they should be run

### E1 — Cold-user stratification  (FREE, no GPU, do this first)
Re-analyse **existing** per-user results, split by how many personal lessons each
user has: 0, 1, 2, 3+. Compute `full - same_user` within each stratum.

- If the effect is strongly positive for the 0-lesson stratum and ~0 elsewhere,
  **H4 is confirmed** and RQ2 is answerable today: the claim becomes
  *"cross-user transfer rescues users without failure history"*, which is a
  sharper and more defensible claim than the current blanket one.
- If it is flat across all strata, H4 is dead and the problem is mechanistic
  (H2/H3/H5).

Needs: per-user memory-source counts. If per-user flags are absent from the
result JSONs, derive the personal-lesson count for each user from the memory
file (`i_temporal_nuser1000_8shards.memory.json`) and join on `user_id`.

**This single analysis decides the whole direction. Run it before any GPU job.**

### E2 — Additive budget  (1 extra arm x 4 datasets, ~2 h)
Arms on the same batch and backbone:
- `same3`  = personal only, `--max_memory_facts 3`   (reference)
- `full3`  = personal + cross, `--max_memory_facts 3` (current, substitutive)
- `full6`  = personal + cross, `--max_memory_facts 6` (additive)

If `full6 > same3` while `full3 ≈ same3`, H1 is confirmed and the fix is a
prompt-budget change, not a mechanism change.

### E3 — Coverage sweep  (4-6 arms x 4 datasets, ~4-6 h)
Sweep the relevance gate from MEMCF's strictest to FMRec's ungated setting so
cross-user coverage lands at roughly 1.6% / 10% / 25% / 40% / 67%. Plot
`full - same_user` against coverage. A hump in the middle would mean both
existing designs are on the wrong side of an optimum and neither paper found it.

Handles: `--graph_retrieval_scope dense_lgcn_userscore_consensus` with
`--consensus_min_users` and `--consensus_top_n_users`, versus
`dense_lgcn_fmrec_topk` at the ungated end.

### E4 — Failure-space retrieval  (new code, highest upside)
Retrieve neighbours by **lesson-embedding** similarity instead of LightGCN:
1. embed every lesson's text (any sentence encoder available offline);
2. for the target user, build a query from their own lesson(s), or from the
   candidate set when they have none;
3. retrieve the K nearest *lessons* (not users) and surface those.

This directly attacks H3 and is the only route that could make (a) positive by
design rather than by restriction. It also gives a clean ablation against the
LightGCN version already run — same budget, same arms, different similarity
space, so the comparison is exactly parallel to the `topk` vs `topk_random` pair
that already works.

### E5 — Lesson specificity audit  (analysis + optional re-distillation)
Classify lessons as item-grounded vs abstract (does the lesson name concrete
items/attributes, or only describe reasoning?). Then test whether the
item-grounded subset transfers while the abstract subset does not. If so, the
fix is in the **distillation prompt**, not in retrieval — regenerate lessons with
an instruction to anchor each one in concrete item attributes.

---

## 4. Decision tree

```
E1 (free, today)
├─ cold-user effect found        -> reframe RQ2 as cold-start transfer.
│                                   Add a cold-user-only table. RQ2 answered.
└─ flat across strata
   ├─ E2 shows full6 > same3     -> RQ2 was a budget artifact. Re-run main
   │                                table with the additive budget.
   └─ E2 flat
      ├─ E3 finds a coverage hump -> report the optimum; both prior designs
      │                              sat on the wrong side of it.
      └─ E3 flat
         -> mechanism is the problem. Run E4 (failure-space retrieval).
            If E4 also flat, RQ2 becomes a *negative result*: publish it with
            the full control battery (shuffled, random-neighbour, coverage
            sweep, two backbones) — that battery is stronger evidence than
            most positive claims in this literature.
```

---

## 5. If it stays negative — how to still finish the paper

The negative result is publishable **because of the controls already run**:

- two protocols x two backbones x 4-6 datasets, n=1,000, paired bootstrap
- a random-neighbour control that isolates similarity from content
- a shuffled-lesson control that isolates content from mere prompt padding
- a same-user arm that isolates cross-user from personal memory

Reframed contribution: *"failure memory helps (RQ1), its benefit comes from the
user's own failures rather than from cross-user transfer, and collaborative
similarity functions as a safeguard against harmful transfer rather than a
source of gain."* That is an honest, well-supported story, and the
similarity-as-safeguard finding (+0.0101, 4/4, significant) is novel on its own.

Two things must then be stated explicitly:
1. the backbone dependence (effects appear on Gemma-3-4B, vanish on Qwen2.5-7B);
2. that the shuffled control recovers a large share of the gain on Qwen, so the
   mechanism claim is only defensible on the weaker backbone.

---

## 6. Immediate next actions

1. **E1 stratification** — no GPU, do first, it redirects everything else.
2. Finish the running `memcfproto7_gemma` batch (28 runs) to complete the
   2x2 protocol x backbone matrix.
3. MemRec fair rerun on 4 datasets x 2 backbones (`eval_feedback: none`),
   to nail RQ1 shut.
4. Then E2, and E3/E4 depending on the decision tree.


---

# APPENDIX A — E1 RESULTS (run 2026-09-10)

**H4 is refuted, and the opposite is true.** Stratifying the 4,000 user-arm pairs
(4 datasets x 1,000 users) by how many failure lessons each user personally
authored:

### Cross-user contribution: `full` - `same_user`

| personal lessons | users | **Gemma** | Qwen |
|---|---|---|---|
| **0** | 97 | **-0.0467 [-0.0906, -0.0037] SIGNIFICANT** | -0.0023 n.s. |
| 1 | 491 | -0.0029 n.s. | +0.0169 n.s. |
| 2 | 1,185 | -0.0043 n.s. | +0.0036 n.s. |
| 3+ | 2,227 | +0.0005 n.s. | -0.0033 n.s. |
| ALL | 4,000 | -0.0025 n.s. | +0.0013 n.s. |

For a user with **zero** personal lessons the `same_user` arm is *no memory at
all*, so this row literally compares "other users' lessons" against "nothing".
On Gemma other users' lessons are **significantly worse than nothing**.
Cross-user transfer does not merely fail to help the users H4 predicted it would
rescue — it actively harms them.

### Similarity benefit: `topk` - `topk_random` (Gemma)

| personal lessons | users | diff | |
|---|---|---|---|
| **0** | 97 | **+0.0377 [+0.0126, +0.0661]** | ✅ SIGNIFICANT — largest of any stratum |
| 1 | 491 | +0.0108 | n.s. |
| 2 | 1,185 | +0.0030 | n.s. |
| 3+ | 2,227 | **+0.0125 [+0.0054, +0.0195]** | ✅ SIGNIFICANT |
| ALL | 4,000 | **+0.0101 [+0.0051, +0.0152]** | ✅ SIGNIFICANT |

### And the ceiling: `topk` - `same_user` (Gemma)

Every stratum n.s.; cold users **+0.0006 [-0.0351, +0.0363]**, ALL -0.0023.

## What this establishes

1. **Similarity selection matters most exactly where it should** — for users with
   no failure history of their own, picking similar neighbours instead of random
   ones is worth **+0.0377** NDCG@10. That is a specific, defensible RQ2 finding.
2. **But the best possible selection only reaches parity with having nothing**
   (+0.0006 for cold users). Cross-user lessons never carry information beyond
   what personal lessons already provide.
3. **Retrieval mechanism matters for cold users:** LightGCN top-K is harmless
   (+0.0006) while the graph-scoped `full` retrieval is harmful (-0.0467) for the
   same population. `full` picks its cross-user lessons by graph proximity and
   picks badly when the user has no history to anchor on.

## Revised diagnosis

H4 (population saturation) is dead. The evidence now points squarely at
**H2 / H3**: the lessons themselves do not carry transferable, user-specific
information. They describe *the ranker's* failure mode, so another user's lesson
adds nothing, and a mismatched one is pure distraction.

## Revised next steps

E1 sends us straight past E2/E3 to the mechanism questions:

1. **E4 (failure-space retrieval)** is now the priority, not the fallback.
   Retrieve neighbours by lesson-embedding similarity rather than LightGCN
   interaction similarity. The cold-user stratum is the sharpest test bed: it is
   where selection demonstrably matters most (+0.0377) and where the current
   ceiling is exactly zero. If any mechanism can push cold users above parity,
   this is the one, and that stratum is where it will show first.
2. **E5 (lesson specificity audit)** in parallel — if lessons are model-level
   rather than item-grounded, no retrieval mechanism can fix it and the fix
   belongs in the distillation prompt.
3. **E2 (additive budget)** is now low priority: `topk - same_user` is ~0 in
   every stratum, so budget displacement is not what is holding cross-user back.

## The paper-ready finding, if nothing further works

> Collaborative similarity governs whether cross-user failure transfer is *safe*,
> not whether it is *useful*. Selecting neighbours by collaborative similarity
> rather than at random is worth +0.0101 NDCG@10 overall and +0.0377 for users
> with no failure history of their own; yet even optimal selection only brings
> cross-user memory to parity with using the user's own lessons alone.

That is honest, fully controlled, and still novel.

---

# APPENDIX B — E5 RESULTS (2026-09-10): the mechanism is now explained

## The two systems write lessons at opposite extremes

| | MEMCF | FMRec |
|---|---|---|
| lessons audited | 2,153 (Software) / 2,462 (Video_Game) | 4,398 (Books) |
| avg words | 47.8 / 49.1 | 30.3 |
| **names its own exact item title** | **99.9%** | ~0% |
| contains a quoted phrase | 100% | rare |
| example | *"After preceding observed items 'Roxio Crunch Win/Mac', the recorded next item was 'Microsoft Office Outlook 2007 with Business Contact Manager', while the base ranker selected 'Microsoft Office Professional 2007'."* | *"The ranker prioritized familiarity based on past reading history over explicitly stated user desires for a light, feel-good experience."* |

H2 said lessons might be too abstract to transfer. For **MEMCF the opposite is
true — they are too specific.** FMRec sits at the other extreme.

## The smoking gun

Sampling 3 random cross-user lessons for each of 1,000 Software users and asking
whether the lesson's item is in that user's candidate set:

> **35 / 3,000 = 1.17%**

**98.8% of cross-user lessons talk about products the recipient will never be
asked to rank.** They are noise by construction.

This explains every observation at once:
- cross-user adds nothing on average -> 98.8% of it is irrelevant;
- it *harms* cold users (-0.0467) -> they receive nothing but irrelevant noise,
  with no personal lesson to dilute it;
- similarity still helps (+0.0377 for cold users) -> a similar neighbour's
  irrelevant lesson is at least in the right category, so it distracts less.

## Revised root cause and the fix

The bottleneck is **the level of abstraction of the lesson**, not the retrieval
mechanism. Retrieval cannot rescue content that is unusable:

- **MEMCF**: item-instance level. Only transfers if the recipient happens to face
  the same product — 1.17% of the time.
- **FMRec**: model-critique level. Always "applicable" but says nothing specific
  to this user, so it adds no information.

**The missing middle is attribute level**: express a lesson over *transferable*
properties — category, price tier, feature, genre, brand, format — instead of
exact titles or generic reasoning. e.g. *"prefers full application suites over
single-purpose utilities in office software"*. Such a lesson can apply to another
user's candidate set without naming any specific product.

### E6 (new, now the highest-value experiment)

Re-distill the existing failure events into **attribute-level** lessons and
re-run the same 7-arm ladder. Nothing else changes — same failures, same
retrieval, same budget, same backbone — so the contrast isolates lesson
abstraction exactly the way `topk` vs `topk_random` isolated selection.

Measure first, before any LLM re-distillation:
1. the applicability rate (currently 1.17%) using attribute overlap instead of
   item-id overlap — if attribute-level matching lifts it to 30-60%, the fix is
   confirmed cheaply and no re-distillation is needed to justify E6;
2. the cold-user stratum, which is where the ceiling is currently exactly zero.

E6 supersedes E4 in priority: failure-space retrieval cannot help while 98.8% of
the retrieved content is inapplicable regardless of who it came from.

---

# APPENDIX C — E6 STEP 1 (2026-09-10): the abstraction level is quantified

Question: if the same failures were expressed over *transferable* properties
instead of exact product titles, how much of a recipient's candidate set would a
cross-user lesson actually match? Measured on existing data, no LLM calls.

| dataset | exact **item id** | >=1 shared title token | **>=2 shared title tokens** | top-level **category** |
|---|---|---|---|---|
| Software | 1.17% | 77.0% | **54.7%** | 95.4% |
| Video_Game | 0.33% | 64.1% | **34.6%** | 91.6% |
| Prime_Pantry | 0.50% | 73.2% | **44.8%** | 100% |
| Industrial & Sci. | 0.27% | 79.1% | **52.7%** | 81% |

## Category is NOT the answer — it is the other failure mode

Category looks like a fix (95-100% applicable) but is an artifact of coarseness:

| dataset | distinct categories | dominant one |
|---|---|---|
| Software | 18 | **95.4% = "software"** |
| Video_Game | 20 | **91.6% = "video games"** |
| **Prime_Pantry** | **1** | **100% = "unknown"** |
| Industrial & Sci. | 23 | 56.3% + 25.0% = 81% in two |

Nearly every item shares one top-level category, so a category-level lesson
matches everything and therefore discriminates nothing. That is exactly FMRec's
failure mode at the opposite end: always applicable, never informative.

## The usable middle

**>=2 shared meaningful title tokens: 35-55%.** That is the honest target zone —
roughly a **40x** increase in applicable cross-user content over the current
1.17%, while staying discriminative (unlike category).

So E6 is confirmed as the right intervention, but with a sharpened spec:

- ❌ do **not** re-distill to category level — it destroys discrimination;
- ✅ re-distill to **product-type / brand / feature phrases** — the level that
  title tokens approximate here (e.g. *"office suite"*, *"antivirus"*,
  *"wireless controller"*, *"full version vs upgrade"*).

Caveat: title-token overlap is a proxy. Real attribute lessons depend on the
distillation prompt extracting the right attributes; 35-55% bounds the
opportunity rather than guaranteeing it.

## E6 step 2 (unchanged, now well-specified)

Re-distill the existing failure events into attribute-level lessons and re-run
the same 7-arm ladder on Gemma. Everything else held fixed, so the contrast
isolates lesson abstraction exactly as `topk` vs `topk_random` isolated
selection. Primary read-outs: `full - same_user` overall, and the cold-user
stratum where the current ceiling is exactly zero.

---

# APPENDIX D — TWO GPU-FREE EXPERIMENTS (2026-09-10)

Retrieval for `dense_lgcn_fmrec_topk` is deterministic, so it can be replayed
offline to recover exactly which cross-user lessons each user received. That
makes both of these possible without a single GPU hour.

## EXP-1b — condition on applicability: INCONCLUSIVE

Split `topk - same_user` by whether any surfaced cross-user lesson actually
mentions an item in that user's candidate set.

| subset | n | diff | 95% CI | |
|---|---|---|---|---|
| applicable | 124 | +0.0017 | [-0.0309, +0.0346] | n.s. |
| not applicable | 3,876 | -0.0024 | [-0.0077, +0.0030] | n.s. |

Per dataset the applicable group is only 9-69 users, and even pooled the
interval is +/-0.033 wide. **This neither rescues nor refutes cross-user** — it is
simply underpowered, because applicability is ~1%. To test it properly the
applicable population has to be manufactured (see the oracle arm).

## EXP-2b — do similar users fail alike? 2.31x, significant

First attempt used a category-pair signature and was degenerate (random pairs
already overlapped 0.69-1.00; Prime_Pantry exactly 1.00 because every category is
"unknown"). Re-run on the lesson's own prefer/avoid/evidence vocabulary:

| dataset | neighbours | random | ratio |
|---|---|---|---|
| Software | 0.2155 | 0.0964 | 2.24x |
| Prime_Pantry | 0.0961 | 0.0462 | 2.08x |
| Video_Game | 0.0776 | 0.0318 | 2.44x |
| Industrial & Sci. | 0.0706 | 0.0250 | 2.82x |
| **pooled** | **0.1144** | **0.0496** | **2.31x**, CI [+0.0628,+0.0667] ✅ |

**Important caveat — do not overclaim this.** Inspecting the fields shows
`evidence_terms` is just the *tokenised product titles*, and `prefer`/`avoid` are
exact product titles. So the 2.31x is **shared product vocabulary**, i.e.
confirmation that LightGCN retrieves users operating in the same product space.
That is a *necessary* condition for transfer, not proof that the underlying
failure *reasoning* is shared. C2 is weakened, not refuted.

## The most actionable finding: the schema is already right, the filler is not

| field | actual content |
|---|---|
| `prefer` | exact product title |
| `avoid` | exact product title |
| `evidence_terms` | tokens of those titles |
| `factual_statement` | the verbose title-naming sentence |
| **`applies_if`** | **EMPTY** |
| **`failure_type`** | `temporal_next_item_ranking_failure` — identical for every lesson |

The memory schema already reserves slots for generalisable conditions
(`applies_if`, `do_not_apply_if`, `overgeneralization_risk`) and the distillation
never fills them; `failure_type` is a constant, carrying zero information.

**So E6 is not "invent a new lesson format" — it is "populate the fields the
schema already has".** That is a distillation-prompt change, and the retrieval,
budget and ladder can all stay exactly as they are, keeping the comparison clean.

## Updated priority

1. **E6** — re-distill filling `applies_if` / `failure_type` / attribute-level
   `prefer`-`avoid` instead of product titles. Everything else held fixed.
2. **Oracle arm** — force-feed each user the cross-user lesson that *does* hit
   their candidate set. Bounds the ceiling; EXP-1b could not because the natural
   applicable population is only ~1%.
3. **`cross_only` vs `nomem`** — cross-user in isolation rather than as an
   increment. `cross_user_only` scope already exists in the codebase.

---

# APPENDIX E — RQ2 IS POSITIVE AFTER ALL (2026-09-11)

## The correction

Every earlier conclusion that "cross-user transfer does not work" rested on one
contrast: `full - same_user`. That contrast can only detect a **complement** — it
asks whether cross-user adds *on top of* personal memory. It is blind to a
**substitute**. The isolation arm had never been run.

It has now. `cross_user_only` removes the user's own lessons entirely and gives
them only other people's, on Gemma under the FMRec protocol, n=1,000 x 4 datasets:

| contrast | mean | 95% CI | dir | |
|---|---|---|---|---|
| **vs NO MEMORY** | **+0.0333** | [+0.0249, +0.0417] | **4/4** | ✅ SIG |
| **vs SHUFFLED lessons** | **+0.0070** | [+0.0009, +0.0129] | **4/4** | ✅ SIG |
| **vs profile-only** | **+0.0059** | [+0.0000, +0.0118] | 3/4 | ✅ SIG (borderline) |
| vs personal-only | -0.0051 | [-0.0103, +0.0003] | 0/4 | n.s. |
| vs full memory | -0.0026 | [-0.0069, +0.0018] | 1/4 | n.s. |

**The shuffled control is the one that matters.** Shuffled lessons are the same
quantity of text drawn from the same pool, only attached to the wrong users.
Cross-user-only beats them by +0.0070, significant on all four datasets. So the
gain is not "more plausible text in the prompt" — **whose lesson it is matters**.

## The corrected RQ2 answer

> **Failure knowledge does transfer across users.** Other people's lessons, with
> the user's own removed, beat no memory by +0.0333 and beat shuffled lessons by
> +0.0070, both significant on 4/4 datasets.
>
> **It is redundant with personal memory, not additive.** cross-user-only is
> statistically indistinguishable from personal-only (-0.0051 n.s., 0/4) and from
> full memory (-0.0026 n.s.). Combining the two adds nothing
> (`full - same_user` = -0.0010 n.s.).

Personal and cross-user failure memory are **substitutes**: either alone carries
the benefit, together they do not stack. That is why the incremental contrast
read as zero for so long.

## What this does to the earlier appendices

- §4 / App. A-D still stand as *measurements*, but their framing was wrong.
  "Cross-user adds nothing" should read **"cross-user adds nothing *on top of*
  personal memory"**, which is a statement about redundancy, not about transfer.
- The cold-user harm (-0.0467) was specific to the **graph-scoped `full`**
  retrieval, which picks badly when a user has no history to anchor on. The
  LightGCN-based `cross_user_only` does not show that failure mode.
- The 1.17% applicability figure and E6 remain relevant to *how much* can be
  transferred, but are no longer needed to rescue RQ2 — it is already positive.

## The 2x2 is complete: backbone decides, protocol does not

`topk` vs `topk_random` (CF proof), n=1,000 x 4 datasets:

| settings | Qwen2.5-7B | Gemma-3-4B |
|---|---|---|
| MEMCF (20 cand, 3 lessons) | -0.0012 ❌ | **+0.0119 [+0.0056,+0.0180] ✅ 3/4** |
| FMRec (10 cand, 4 lessons) | -0.0013 ❌ | **+0.0101 [+0.0049,+0.0154] ✅ 4/4** |

Two protocols, same verdict per backbone. **The effect is a property of the
backbone, not of the evaluation protocol** — a clean RQ3 answer nobody has
published.

## New ladder: MEMCF settings on Gemma

| contrast | mean | CI | dir | |
|---|---|---|---|---|
| memory vs no memory | **+0.0606** | [+0.0507,+0.0705] | 4/4 | ✅ largest measured |
| real vs shuffled | **+0.0118** | [+0.0048,+0.0184] | 4/4 | ✅ mechanism holds |
| CF: similar vs random | **+0.0119** | [+0.0056,+0.0180] | 3/4 | ✅ |
| **FMRec design vs MEMCF design** | **+0.0085** | [+0.0022,+0.0151] | 3/4 | ✅ **first separation** |
| facts over profile | +0.0060 | [-0.0006,+0.0128] | 3/4 | n.s. |
| cross-user over personal-only | -0.0010 | [-0.0069,+0.0047] | 1/4 | n.s. |

Note the last row: **FMRec's top-K retrieval significantly beats MEMCF's `full`
scope** here (+0.0085). In every previous configuration the two designs were tied.
