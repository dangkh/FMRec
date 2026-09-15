#!/usr/bin/env python3
"""Verify two arms are actually comparable before trusting a paired contrast:
same users, same candidate set, same ground truth, same candidate count."""
import glob, json, os, sys
D="/home/ubuntu/24nam.nh/video_games_data"
DS="Software_1000u"
def rows(p):
    r=json.load(open(p,encoding="utf-8"))
    if isinstance(r,dict): r=r.get("results",r.get("users",[]))
    return {str(x["user_id"]):x for x in r}
def find(root,needle,ex=()):
    for p in sorted(glob.glob(os.path.join(root,DS,"*.json"))):
        if p.endswith(".summary.json"): continue
        b=os.path.basename(p)
        if needle in b and not any(x in b for x in ex): return p
A=find(D+"/evaluation_results_screen300_gemma","scr300_noself3.",("rand",))
B=find(D+"/evaluation_results_fmrec_ablation_1000u","memcf_nomemory_")
print("A =",os.path.basename(A)); print("B =",os.path.basename(B))
Ar,Br=rows(A),rows(B); u=sorted(set(Ar)&set(Br))
print("paired users:",len(u))
print("\nper-user record keys (A):",sorted(Ar[u[0]].keys()))
print("per-user record keys (B):",sorted(Br[u[0]].keys()))
def cands(r):
    for k in ("candidates","candidate_items","candidate_ids","candidate_list"):
        if k in r: return r[k]
def gt(r):
    for k in ("ground_truth","target_item","target","gt_item","positive_item"):
        if k in r: return r[k]
nc=ng=nlen=0
lensA={}; lensB={}
for x in u:
    ca,cb=cands(Ar[x]),cands(Br[x]); ga,gb=gt(Ar[x]),gt(Br[x])
    if ca is not None and cb is not None:
        lensA[len(ca)]=lensA.get(len(ca),0)+1; lensB[len(cb)]=lensB.get(len(cb),0)+1
        if list(ca)!=list(cb): nc+=1
        if len(ca)!=len(cb): nlen+=1
    if ga is not None and gb is not None and ga!=gb: ng+=1
print("\ncandidate-list mismatches:",nc,"  length mismatches:",nlen,"  ground-truth mismatches:",ng)
print("candidate-count histogram A:",lensA,"  B:",lensB)
for mk in ("hit@1","ndcg@5","ndcg@10"):
    a=sum((Ar[x].get("metrics") or {}).get(mk) or 0 for x in u)/len(u)
    b=sum((Br[x].get("metrics") or {}).get(mk) or 0 for x in u)/len(u)
    print("%-8s A(noself3)=%.4f  B(nomemory)=%.4f  diff=%+.4f" % (mk,a,b,a-b))
sa=json.load(open(A.replace(".json",".summary.json"),encoding="utf-8"))
sb=json.load(open(B.replace(".json",".summary.json"),encoding="utf-8"))
for tag,s in (("A",sa),("B",sb)):
    cfg=s.get("config") or s.get("args") or {}
    print(tag,"summary metrics:",{k:round(v,4) for k,v in (s.get("metrics") or {}).items() if isinstance(v,(int,float))})
    print(tag,"n_users:",s.get("number_of_users_evaluated"),
          " maxneg:",cfg.get("max_negative_candidates"),
          " split:",cfg.get("eval_split")," negmode:",cfg.get("candidate_negative_mode"))
