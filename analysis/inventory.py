#!/usr/bin/env python3
"""Catalogue of every MEMCF/FMRec result set: protocol, size, and arms present.
Backbone is inferred from the root name ('gemma' => Gemma-3-4B, else Qwen2.5-7B)
and is marked (?) where the name carries no tag."""
import glob, json, os
D="/home/ubuntu/24nam.nh/video_games_data"
SCOPES=[("nomemory","nomem"),("profileonly","prof"),("scopefull","full"),
        ("scopesame_user","same"),("scopeshuffled_memory","shuf"),
        ("scopecross_user_only","cross"),("scopeoracle_cross_candidate","oracle"),
        ("fmrec_topk_noself_random","nsRand"),("fmrec_topk_noself","noself"),
        ("fmrec_topk_random","tkRand"),("fmrec_topk_shared","shared"),
        ("fmrec_topk_popular","popular"),("fmrec_topk","topk"),
        ("userscore_consensus","consensus")]
def scope_of(b):
    for pat,lbl in SCOPES:
        if pat in b: return lbl
    return None
rows=[]
for r in sorted(glob.glob(D+"/evaluation_results_*")):
    if not os.path.isdir(r) or "memrec" in os.path.basename(r): continue
    arms={}; dss=set(); ncs=set(); ns=set()
    for p in glob.glob(r+"/*/*.json"):
        if p.endswith(".summary.json"): continue
        b=os.path.basename(p); s=scope_of(b)
        if not s: continue
        try: d=json.load(open(p,encoding="utf-8"))
        except Exception: continue
        if isinstance(d,dict): d=d.get("results",d.get("users",[]))
        if not d: continue
        nc=d[0].get("num_candidates") or len(d[0].get("candidate_item_ids") or [])
        arms.setdefault(s,set()).add(os.path.basename(os.path.dirname(p)))
        dss.add(os.path.basename(os.path.dirname(p))); ncs.add(nc); ns.add(len(d))
    if not arms: continue
    n=max(ns) if ns else 0
    name=os.path.basename(r)[19:]
    bk="Gemma" if "gemma" in name.lower() else "Qwen?"
    rows.append((n,name,bk,sorted(ncs),len(dss),arms))
rows.sort(key=lambda x:(-x[0],x[1]))
print("### PAPER-GRADE (n >= 500)")
hdr=False
for n,name,bk,ncs,nds,arms in rows:
    if n<500: continue
    if not hdr:
        print("%-42s %-6s %5s %4s %3s  %s"%("result set","LLM","cand","n","ds","arms")); hdr=True
    print("%-42s %-6s %5s %4d %3d  %s"%(name[:42],bk,",".join(map(str,ncs)),n,nds,
          " ".join(sorted(arms,key=lambda a:[l for _,l in SCOPES].index(a)))))
print()
print("### SMALLER / EXPLORATORY (n < 500):  %d result sets"%sum(1 for r in rows if r[0]<500))
for n,name,bk,ncs,nds,arms in rows:
    if n>=500: continue
    print("    %-44s n=%-4d ds=%-2d %s"%(name[:44],n,nds," ".join(sorted(arms))))
