# FMRec — ablation & significance tables

Generated 2026-09-11. Fills the gap left by the paper: Table 3 reports point
estimates with no intervals, and section 4.3 (RQ2/RQ3) is empty.

**Setup.** FMRec's retrieval rule (top-K similar users by LightGCN cosine, no
gate, 1 personal + 3 collaborative lessons) run inside MEMCF's harness on
**Gemma-3-4B** — FMRec's own backbone — under FMRec's protocol (10 candidates,
max 4 lessons), n=1,000 users on each of 4 Amazon datasets. Paired bootstrap,
5,000 resamples, seed 2027. Metrics are the paper's own (H@K, N@K).

**Caveat to state in the paper.** These are Amazon datasets, not InstructRec
Books/GoodReads/MovieTV. They measure FMRec's *mechanism* under controlled
conditions against a full control battery; they are not a drop-in replacement
for Table 3. Reproducing on InstructRec is blocked because the server's
firewall denies googleusercontent.com (see COMPACT_HISTORY.md section 13).

========================================================================================================
TABLE A -- absolute scores, Gemma-3-4B, n=1,000 per dataset, mean over 4 datasets
arm                       H@1      H@3      N@3      H@5      N@5
Vanilla (no mem)       0.2400   0.4818   0.3782   0.6583   0.4509
profile only           0.2795   0.5250   0.4210   0.6875   0.4877
shuffled lessons       0.2802   0.5202   0.4187   0.6765   0.4828
random neighbours      0.2778   0.5230   0.4189   0.6837   0.4849
cross-user only        0.2893   0.5327   0.4297   0.6895   0.4942
personal only          0.2990   0.5422   0.4390   0.6905   0.4997
FMRec (topk)           0.2940   0.5340   0.4327   0.6933   0.4982

========================================================================================================
TABLE B -- paired differences with 95%% bootstrap CI (5,000 resamples, seed 2027)

  RQ1  FMRec vs Vanilla LLM   [FMRec (topk) - Vanilla (no mem)]
    H@1   +0.0540  CI [+0.0395, +0.0685]  SIGNIFICANT  dir 4/4
    H@3   +0.0523  CI [+0.0360, +0.0680]  SIGNIFICANT  dir 4/4
    N@3   +0.0545  CI [+0.0416, +0.0671]  SIGNIFICANT  dir 4/4
    H@5   +0.0350  CI [+0.0203, +0.0500]  SIGNIFICANT  dir 3/4
    N@5   +0.0473  CI [+0.0359, +0.0587]  SIGNIFICANT  dir 4/4

       vs profile text only   [FMRec (topk) - profile only]
    H@1   +0.0145  CI [+0.0037, +0.0255]  SIGNIFICANT  dir 4/4
    H@3   +0.0090  CI [-0.0030, +0.0208]  n.s.         dir 3/4
    N@3   +0.0117  CI [+0.0028, +0.0206]  SIGNIFICANT  dir 4/4
    H@5   +0.0057  CI [-0.0053, +0.0170]  n.s.         dir 3/4
    N@5   +0.0105  CI [+0.0028, +0.0183]  SIGNIFICANT  dir 4/4

  ***  vs shuffled lessons  (does content matter?)   [FMRec (topk) - shuffled lessons]
    H@1   +0.0137  CI [+0.0030, +0.0250]  SIGNIFICANT  dir 4/4
    H@3   +0.0138  CI [+0.0022, +0.0255]  SIGNIFICANT  dir 2/4
    N@3   +0.0140  CI [+0.0054, +0.0226]  SIGNIFICANT  dir 4/4
    H@5   +0.0168  CI [+0.0053, +0.0285]  SIGNIFICANT  dir 4/4
    N@5   +0.0154  CI [+0.0076, +0.0234]  SIGNIFICANT  dir 4/4

  RQ2b vs random neighbours (does similarity matter?)   [FMRec (topk) - random neighbours]
    H@1   +0.0163  CI [+0.0065, +0.0262]  SIGNIFICANT  dir 4/4
    H@3   +0.0110  CI [+0.0002, +0.0218]  SIGNIFICANT  dir 3/4
    N@3   +0.0138  CI [+0.0055, +0.0220]  SIGNIFICANT  dir 3/4
    H@5   +0.0095  CI [-0.0020, +0.0210]  n.s.         dir 3/4
    N@5   +0.0133  CI [+0.0060, +0.0209]  SIGNIFICANT  dir 3/4

  RQ2a cross-user alone vs no memory   [cross-user only - Vanilla (no mem)]
    H@1   +0.0492  CI [+0.0345, +0.0643]  SIGNIFICANT  dir 4/4
    H@3   +0.0510  CI [+0.0348, +0.0673]  SIGNIFICANT  dir 4/4
    N@3   +0.0515  CI [+0.0383, +0.0648]  SIGNIFICANT  dir 4/4
    H@5   +0.0312  CI [+0.0160, +0.0460]  SIGNIFICANT  dir 3/4
    N@5   +0.0434  CI [+0.0317, +0.0549]  SIGNIFICANT  dir 4/4

  ***  cross-user alone vs shuffled   [cross-user only - shuffled lessons]
    H@1   +0.0090  CI [-0.0025, +0.0205]  n.s.         dir 3/4
    H@3   +0.0125  CI [+0.0000, +0.0248]  n.s.         dir 3/4
    N@3   +0.0110  CI [+0.0015, +0.0202]  SIGNIFICANT  dir 4/4
    H@5   +0.0130  CI [+0.0010, +0.0248]  SIGNIFICANT  dir 4/4
    N@5   +0.0115  CI [+0.0031, +0.0197]  SIGNIFICANT  dir 4/4

       cross-user on top of personal   [FMRec (topk) - personal only]
    H@1   -0.0050  CI [-0.0155, +0.0050]  n.s.         dir 0/4
    H@3   -0.0083  CI [-0.0198, +0.0030]  n.s.         dir 1/4
    N@3   -0.0063  CI [-0.0147, +0.0021]  n.s.         dir 1/4
    H@5   +0.0027  CI [-0.0080, +0.0138]  n.s.         dir 3/4
    N@5   -0.0015  CI [-0.0089, +0.0060]  n.s.         dir 2/4

## How to read this for the paper

| Claim | Evidence | Verdict |
|---|---|---|
| RQ1 FMRec > Vanilla LLM | all 5 metrics significant, 4/4 datasets, N@5 +0.0473 | strong |
| Lesson *content* matters (not just extra text) | beats shuffled on **all 5 metrics**, N@5 +0.0154 | strong |
| RQ2a failure knowledge transfers across users | cross-user alone beats no-memory on **all 5**, N@5 +0.0434; and beats shuffled on 3/5 | strong |
| RQ2b collaborative similarity helps selection | beats random neighbours on 4/5, N@5 +0.0133 | good |
| Cross-user is *additive* on top of personal | all 5 metrics n.s. | **NOT supported** |

The last row is the honest limitation: personal and cross-user memory are
**substitutes**, not complements. Either alone delivers the gain; combining
them adds nothing. State this explicitly rather than letting a reader assume
the contributions stack.
