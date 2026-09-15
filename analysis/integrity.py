#!/usr/bin/env python3
"""Is memrec_read sound, and is it really comparable to FMRec?
Checks: completeness, silent-fallback rate, facet coverage, and -- decisively --
whether the two systems put the SAME candidate items in front of the SAME users."""
import glob, json, os
D="/home/ubuntu/24nam.nh/video_games_data"
MR=D+"/evaluation_results_memrec_1k_static_trainhistory_fair"
MR2=MR+"_rerun_failed7_v2"
FR=D+"/evaluation_results_fmrec_ablation_1000u"
DS=["Software_1000u","Prime_Pantry_1000u","Video_Game","Industrial_and_Scientific_1000u"]
def mrrows(ds,v):
    for root in (MR,MR2):
        g=glob.glob("%s/%s/%s*.ranking.json"%(root,ds,v))
        if g: return json.load(open(g[0],encoding="utf-8")), os.path.basename(root)
    return None,None
def fmrows(ds,nd,ex=()):
    for p in sorted(glob.glob(os.path.join(FR,ds,"*.json"))):
        if p.endswith(".summary.json"): continue
        b=os.path.basename(p)
        if nd in b and not any(e in b for e in ex):
            r=json.load(open(p,encoding="utf-8"))
            if isinstance(r,dict): r=r.get("results",r.get("users",[]))
            return r
print("=== A. memrec_read integrity (4 datasets) ===")
print("%-26s %5s %6s %8s %7s %7s %s" % ("dataset","n","cand","fallback","facets","H@20","source"))
for ds in DS:
    rows,src=mrrows(ds,"memrec_read")
    if not rows: print("%-26s MISSING"%ds); continue
    nc={len(x["candidates"]) for x in rows}
    fb=sum(1 for x in rows if list(x["predictions"])==list(x["candidates"]))
    nf=sum(1 for x in rows if x.get("facets"))
    h20=sum((x.get("metrics") or {}).get("H@20") or 0 for x in rows)/len(rows)
    dup=sum(1 for x in rows if len(set(x["candidates"]))!=len(x["candidates"]))
    gtin=sum(1 for x in rows if set(x["ground_truth"])&set(x["candidates"]))
    print("%-26s %5d %6s %7.1f%% %7d %7.3f %s" % (ds[:26],len(rows),sorted(nc),100*fb/len(rows),nf,h20,src[-14:]))
    if dup or gtin!=len(rows):
        print("     !! dup-candidate users=%d  gt-in-candidates=%d/%d" % (dup,gtin,len(rows)))
print("\n=== B. Do MemRec and FMRec show the SAME candidates to the SAME users? ===")
print("%-26s %7s %9s %9s %9s" % ("dataset","shared","same set","same GT","MR-only avg"))
for ds in DS:
    rows,_=mrrows(ds,"memrec_read"); fr=fmrows(ds,"fmrec1k_nomem")
    if not rows or not fr: print("%-26s MISSING"%ds); continue
    M={str(x["user_id"]):x for x in rows}
    F={str(x["user_id"]):x for x in fr}
    u=sorted(set(M)&set(F))
    if not u:
        print("%-26s %7d  (no overlapping user ids -- MemRec remaps ids?)"%(ds[:26],0)); continue
    same=gtsame=0; extra=0
    for x in u:
        a=set(M[x]["candidates"]); b=set(F[x].get("candidate_item_ids") or [])
        if a==b: same+=1
        extra+=len(a-b)
        ga=set(M[x]["ground_truth"]); gb=set(F[x].get("ground_truth_item_ids") or [])
        if ga==gb: gtsame+=1
    print("%-26s %7d %8.1f%% %8.1f%% %9.2f" % (ds[:26],len(u),100*same/len(u),100*gtsame/len(u),extra/len(u)))
