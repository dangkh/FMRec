#!/usr/bin/env python3
"""N@1 / N@3 / N@5 for (A) FMRec design vs MEMCF design, and (B) each system's
lift over its own no-memory arm. MemRec stores no N@1/N@3, so they are
recomputed here from its raw predictions."""
import glob, json, math, os, random
BOOT,SEED=5000,2027
D="/home/ubuntu/24nam.nh/video_games_data"
DS=["Software_1000u","Prime_Pantry_1000u","Video_Game","Industrial_and_Scientific_1000u"]
PR=D+"/evaluation_results_proto7_gemma_1000u"        # Gemma, 10 cand
FR=D+"/evaluation_results_fmrec_ablation_1000u"      # Qwen,  20 cand
MR=D+"/evaluation_results_memrec_1k_static_trainhistory_fair"
MR2=MR+"_rerun_failed7_v2"
MET=[("ndcg@1","N@1"),("ndcg@3","N@3"),("ndcg@5","N@5")]
def load(p):
    r=json.load(open(p,encoding="utf-8"))
    if isinstance(r,dict): r=r.get("results",r.get("users",[]))
    return {str(x["user_id"]):(x.get("metrics") or {}) for x in r}
def find(root,ds,nd,ex=()):
    for p in sorted(glob.glob(os.path.join(root,ds,"*.json"))):
        if p.endswith(".summary.json"): continue
        b=os.path.basename(p)
        if nd in b and not any(e in b for e in ex): return p
def strat(gs,seed=SEED,n=BOOT):
    r=random.Random(seed); m=[]
    for _ in range(n):
        m.append(sum(sum(g[r.randrange(len(g))] for _ in range(len(g)))/len(g) for g in gs)/len(gs))
    m.sort(); return m[int(.025*n)],m[int(.975*n)-1]
def contrast(root,a,b,aex=(),bex=(),label=""):
    pool={mk:[] for mk,_ in MET}; ok=0
    for ds in DS:
        pa,pb=find(root,ds,a,aex),find(root,ds,b,bex)
        if not(pa and pb): continue
        A,B=load(pa),load(pb); u=sorted(set(A)&set(B)); ok+=1
        for mk,_ in MET:
            pool[mk].append([(A[x].get(mk) or 0)-(B[x].get(mk) or 0) for x in u])
    if not ok: print("   %s: MISSING"%label); return
    out=[]
    for mk,ml in MET:
        gs=pool[mk]; mean=sum(sum(g)/len(g) for g in gs)/len(gs)
        lo,hi=strat(gs); pos=sum(1 for g in gs if sum(g)/len(g)>0)
        out.append("%s %+.4f [%+.4f,%+.4f] %d/%d%s"%(ml,mean,lo,hi,pos,len(gs)," SIG" if (lo>0 or hi<0) else ""))
    print("   %-34s %s"%(label," | ".join(out)))
def absmean(root,nd,ex=()):
    r={}
    for mk,_ in MET:
        vals=[]
        for ds in DS:
            p=find(root,ds,nd,ex)
            if not p: continue
            A=load(p); vals.append(sum(v.get(mk) or 0 for v in A.values())/len(A))
        r[mk]=sum(vals)/len(vals) if vals else None
    return r
print("### A. THE TWO VERSIONS, head to head (paired, same users, same candidates)")
print("  Gemma-3-4B, 10 candidates:")
contrast(PR,"proto7_gemma_topk.","proto7_gemma_full",("random",),(),"FMRec topk - MEMCF full")
print("\n### B. absolute N@1/N@3/N@5, Gemma 10-cand")
for nd,ex,lbl in (("proto7_gemma_nomem",(),"no memory"),("proto7_gemma_full",(),"MEMCF full"),
                  ("proto7_gemma_topk.",("random",),"FMRec topk")):
    m=absmean(PR,nd,ex)
    print("   %-16s N@1 %.4f  N@3 %.4f  N@5 %.4f"%(lbl,m["ndcg@1"] or 0,m["ndcg@3"] or 0,m["ndcg@5"] or 0))
print("\n### C. lift over own no-memory, Qwen 20-cand (MemRec-comparable protocol)")
contrast(FR,"scopedense_lgcn_fmrec_topk","fmrec1k_nomem",("random",),(),"FMRec - no memory")
def ndcg(pred,gt,k):
    for i,x in enumerate(pred[:k]):
        if x in gt: return 1.0/math.log2(i+2)
    return 0.0
def mrrows(ds,v):
    for root in (MR,MR2):
        g=glob.glob("%s/%s/%s*.ranking.json"%(root,ds,v))
        if g: return json.load(open(g[0],encoding="utf-8"))
print("   %-34s"%"MemRec read - no memory (recomputed):",end="")
pool={1:[],3:[],5:[]}
for ds in DS:
    a,b=mrrows(ds,"memrec_read"),mrrows(ds,"no_memory")
    if not(a and b): continue
    A={str(x["user_id"]):x for x in a}; B={str(x["user_id"]):x for x in b}
    u=sorted(set(A)&set(B))
    for k in (1,3,5):
        pool[k].append([ndcg(A[x]["predictions"],set(A[x]["ground_truth"]),k)
                       -ndcg(B[x]["predictions"],set(B[x]["ground_truth"]),k) for x in u])
out=[]
for k in (1,3,5):
    gs=pool[k]; mean=sum(sum(g)/len(g) for g in gs)/len(gs); lo,hi=strat(gs)
    pos=sum(1 for g in gs if sum(g)/len(g)>0)
    out.append("N@%d %+.4f [%+.4f,%+.4f] %d/%d%s"%(k,mean,lo,hi,pos,len(gs)," SIG" if (lo>0 or hi<0) else ""))
print(" "+" | ".join(out))
