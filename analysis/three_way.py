#!/usr/bin/env python3
"""All three systems, absolute N@1/N@3/N@5, at the SAME protocol (20 candidates,
n=1,000, 4 datasets). Ours are paired with each other; MemRec is not paired with
ours (different users and candidates), so its column is read as lift, not level."""
import glob, json, math, os, random
BOOT,SEED=5000,2027
D="/home/ubuntu/24nam.nh/video_games_data"
DS=["Software_1000u","Prime_Pantry_1000u","Video_Game","Industrial_and_Scientific_1000u"]
FR=D+"/evaluation_results_fmrec_ablation_1000u"
MP=D+"/evaluation_results_memcf_a_mainpaper_rebuild_v2"
LG=D+"/evaluation_results_ladder_memcfproto_gemma_1000u"
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
def absm(root,nd,ex=()):
    out={}
    for mk,_ in MET:
        v=[]
        for ds in DS:
            p=find(root,ds,nd,ex)
            if p:
                A=load(p); v.append(sum(x.get(mk) or 0 for x in A.values())/len(A))
        out[mk]=sum(v)/len(v) if v else None
    return out
def ndcg(pred,gt,k):
    for i,x in enumerate(pred[:k]):
        if x in gt: return 1.0/math.log2(i+2)
    return 0.0
def mrabs(v):
    out={1:[],3:[],5:[]}
    for ds in DS:
        rows=None
        for root in (MR,MR2):
            g=glob.glob("%s/%s/%s*.ranking.json"%(root,ds,v))
            if g: rows=json.load(open(g[0],encoding="utf-8")); break
        if not rows: continue
        for k in (1,3,5):
            out[k].append(sum(ndcg(x["predictions"],set(x["ground_truth"]),k) for x in rows)/len(rows))
    return {k:(sum(vs)/len(vs) if vs else None) for k,vs in out.items()}
print("### ALL THREE SYSTEMS -- 20 candidates, n=1,000, same 4 datasets")
print("%-12s %-22s %8s %8s %8s   %8s %8s %8s" % ("backbone","system / arm","N@1","N@3","N@5","dN@1","dN@3","dN@5"))
def row(bk,lbl,m,base=None):
    g=lambda k: ("%8.4f"%m[k]) if m.get(k) is not None else "%8s"%"-"
    d=["%8s"%""]*3
    if base:
        d=[("%+8.4f"%(m[k]-base[k])) if (m.get(k) is not None and base.get(k) is not None) else "%8s"%"-"
           for k,_ in MET]
    print("%-12s %-22s %s %s %s   %s %s %s" % (bk,lbl,g("ndcg@1"),g("ndcg@3"),g("ndcg@5"),*d))
qn=absm(FR,"fmrec1k_nomem"); qf=absm(FR,"scopedense_lgcn_fmrec_topk",("random",))
row("Qwen2.5-7B","no memory",qn); row("Qwen2.5-7B","FMRec top-K",qf,qn)
mn=absm(MP,"nomemory"); mf=absm(MP,"scopefull")
row("Qwen2.5-7B","MEMCF no memory",mn); row("Qwen2.5-7B","MEMCF full",mf,mn)
a=mrabs("no_memory"); b=mrabs("memrec_read")
A={"ndcg@1":a[1],"ndcg@3":a[3],"ndcg@5":a[5]}; B={"ndcg@1":b[1],"ndcg@3":b[3],"ndcg@5":b[5]}
row("Qwen2.5-7B","MemRec no memory",A); row("Qwen2.5-7B","MemRec read (W off)",B,A)
print()
gn=absm(LG,"nomemory"); gf=absm(LG,"scopefull"); gt=absm(LG,"fmrec_topk_gk")
row("Gemma-3-4B","no memory",gn); row("Gemma-3-4B","MEMCF full",gf,gn); row("Gemma-3-4B","FMRec top-K",gt,gn)
print("\n### Are our two 20-cand roots paired with each other?")
for ds in DS[:2]:
    p,q=find(FR,ds,"fmrec1k_nomem"),find(MP,ds,"nomemory")
    if not(p and q): print("   %s missing"%ds); continue
    P,Q=json.load(open(p,encoding="utf-8")),json.load(open(q,encoding="utf-8"))
    if isinstance(P,dict): P=P.get("results",[])
    if isinstance(Q,dict): Q=Q.get("results",[])
    Pd={str(x["user_id"]):x for x in P}; Qd={str(x["user_id"]):x for x in Q}
    u=sorted(set(Pd)&set(Qd))
    same=sum(1 for x in u if list(Pd[x].get("candidate_item_ids") or [])==list(Qd[x].get("candidate_item_ids") or []))
    print("   %-26s shared=%d  identical candidate lists=%d" % (ds,len(u),same))
