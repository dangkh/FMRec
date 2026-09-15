#!/usr/bin/env python3
"""The three-system table, broken out per dataset. Qwen2.5-7B, 20 candidates,
n=1,000. Each system is read against ITS OWN no-memory arm: FMRec and MEMCF
share candidate sets with each other, MemRec does not share with either."""
import glob, json, math, os
D="/home/ubuntu/24nam.nh/video_games_data"
DS=["Software_1000u","Prime_Pantry_1000u","Video_Game","Industrial_and_Scientific_1000u"]
FR=D+"/evaluation_results_fmrec_ablation_1000u"
MP=D+"/evaluation_results_memcf_a_mainpaper_rebuild_v2"
MR=D+"/evaluation_results_memrec_1k_static_trainhistory_fair"
MR2=MR+"_rerun_failed7_v2"
KEYS=["ndcg@1","ndcg@3","ndcg@5"]
def load(p):
    r=json.load(open(p,encoding="utf-8"))
    if isinstance(r,dict): r=r.get("results",r.get("users",[]))
    return r
def find(root,ds,nd,ex=()):
    for p in sorted(glob.glob(os.path.join(root,ds,"*.json"))):
        if p.endswith(".summary.json"): continue
        b=os.path.basename(p)
        if nd in b and not any(e in b for e in ex): return p
def mean(root,ds,nd,ex=()):
    p=find(root,ds,nd,ex)
    if not p: return None
    rows=load(p)
    return {k:sum((x.get("metrics") or {}).get(k) or 0 for x in rows)/len(rows) for k in KEYS}
def ndcg(pred,gt,k):
    for i,x in enumerate(pred[:k]):
        if x in gt: return 1.0/math.log2(i+2)
    return 0.0
def mrmean(ds,v):
    for root in (MR,MR2):
        g=glob.glob("%s/%s/%s*.ranking.json"%(root,ds,v))
        if g:
            rows=json.load(open(g[0],encoding="utf-8"))
            return {("ndcg@%d"%k):sum(ndcg(x["predictions"],set(x["ground_truth"]),k)
                    for x in rows)/len(rows) for k in (1,3,5)}
    return None
def show(lbl,m,base=None):
    if m is None: print("%-24s %8s %8s %8s   %8s %8s %8s"%(lbl,"-","-","-","","","")); return
    g=[("%8.4f"%m[k]) for k in KEYS]
    d=["%8s"%""]*3
    if base: d=[("%+8.4f"%(m[k]-base[k])) for k in KEYS]
    print("%-24s %s %s %s   %s %s %s"%(lbl,*g,*d))
TOT={}
for ds in DS:
    print("="*86); print("## %s   (Qwen2.5-7B, 20 candidates, n=1,000)"%ds)
    print("%-24s %8s %8s %8s   %8s %8s %8s"%("system / arm","N@1","N@3","N@5","dN@1","dN@3","dN@5"))
    fn=mean(FR,ds,"fmrec1k_nomem"); ft=mean(FR,ds,"scopedense_lgcn_fmrec_topk",("random",))
    mn=mean(MP,ds,"nomemory");      mf=mean(MP,ds,"scopefull")
    rn=mrmean(ds,"no_memory");      rr=mrmean(ds,"memrec_read")
    show("no memory (ours)",fn); show("FMRec top-K",ft,fn)
    show("no memory (MEMCF run)",mn); show("MEMCF full",mf,mn)
    show("no memory (MemRec)",rn); show("MemRec read (W off)",rr,rn)
    for nm,(a,b) in (("FMRec",(ft,fn)),("MEMCF",(mf,mn)),("MemRec",(rr,rn))):
        if a and b:
            TOT.setdefault(nm,{k:[] for k in KEYS})
            for k in KEYS: TOT[nm][k].append(a[k]-b[k])
    print()
print("="*86); print("## LIFT SUMMARY across the 4 datasets")
print("%-10s %-8s %9s %9s %9s %9s   %s"%("system","metric","Software","Pantry","VideoGame","Industrial","mean"))
for nm in ("FMRec","MEMCF","MemRec"):
    if nm not in TOT: continue
    for k,lbl in zip(KEYS,("N@1","N@3","N@5")):
        v=TOT[nm][k]
        print("%-10s %-8s %+9.4f %+9.4f %+9.4f %+9.4f   %+.4f  (%d/4)"
              %(nm,lbl,*v,sum(v)/len(v),sum(1 for x in v if x>0)))
