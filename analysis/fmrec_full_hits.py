#!/usr/bin/env python3
"""FULL FMRec arm only, all 8 metrics, grouped by dataset."""
import glob, json, os
D="/home/ubuntu/24nam.nh/video_games_data"
MK=[("hit@1","H@1"),("hit@3","H@3"),("hit@5","H@5"),("hit@10","H@10"),
    ("ndcg@3","N@3"),("ndcg@5","N@5"),("ndcg@10","N@10")]
DS=["Software_1000u","Prime_Pantry_1000u","Video_Game","Industrial_and_Scientific_1000u"]
CELLS=[("Gemma 20-cand","ladder_memcfproto_gemma_1000u","fmrec_topk_gk",()),
       ("Gemma 10-cand","proto7_gemma_1000u","proto7_gemma_topk.",("random",)),
       ("Qwen  20-cand","fmrec_ablation_1000u","scopedense_lgcn_fmrec_topk",("random",)),
       ("Qwen  10-cand","proto7_qwen_1000u","fmrec_topk_gk",())]
NOMEM={"ladder_memcfproto_gemma_1000u":"nomemory","proto7_gemma_1000u":"proto7_gemma_nomem",
       "fmrec_ablation_1000u":"fmrec1k_nomem","proto7_qwen_1000u":"nomemory"}
def load(p):
    r=json.load(open(p,encoding="utf-8"))
    if isinstance(r,dict): r=r.get("results",r.get("users",[]))
    return r
def find(root,ds,nd,ex=()):
    for p in sorted(glob.glob(os.path.join(D,"evaluation_results_"+root,ds,"*.json"))):
        if p.endswith(".summary.json"): continue
        b=os.path.basename(p)
        if nd in b and not any(e in b for e in ex): return p
hdr="%-15s "+" ".join(["%7s"]*len(MK))
for ds in DS:
    print("="*88); print("## %s"%ds)
    print(hdr % ("protocol",*[l for _,l in MK]))
    for lbl,root,nd,ex in CELLS:
        p=find(root,ds,nd,ex)
        if not p: print("%-15s  (missing)"%lbl); continue
        rows=load(p)
        v=[sum((x.get("metrics") or {}).get(k) or 0 for x in rows)/len(rows) for k,_ in MK]
        print(("%-15s "+" ".join(["%7.4f"]*len(MK))) % (lbl,*v))
    # paired deltas vs no-memory
    print("%-15s %s"%("","-- paired delta vs no memory --"))
    for lbl,root,nd,ex in CELLS:
        p=find(root,ds,nd,ex); q=find(root,ds,NOMEM[root])
        if not(p and q): continue
        A={str(x["user_id"]):(x.get("metrics") or {}) for x in load(p)}
        B={str(x["user_id"]):(x.get("metrics") or {}) for x in load(q)}
        u=sorted(set(A)&set(B))
        v=[sum((A[x].get(k) or 0)-(B[x].get(k) or 0) for x in u)/len(u) for k,_ in MK]
        print(("%-15s "+" ".join(["%+7.4f"]*len(MK))) % (lbl,*v))
    print()
print("="*88); print("## MEAN OVER THE 4 DATASETS")
print(hdr % ("protocol",*[l for _,l in MK]))
for lbl,root,nd,ex in CELLS:
    acc=[[] for _ in MK]
    for ds in DS:
        p=find(root,ds,nd,ex)
        if not p: continue
        rows=load(p)
        for i,(k,_) in enumerate(MK):
            acc[i].append(sum((x.get("metrics") or {}).get(k) or 0 for x in rows)/len(rows))
    if not acc[0]: continue
    print(("%-15s "+" ".join(["%7.4f"]*len(MK))) % (lbl,*[sum(a)/len(a) for a in acc]))
