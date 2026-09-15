#!/usr/bin/env python3
"""Every run of the FULL FMRec arm (scope dense_lgcn_fmrec_topk, no _random /
_noself / _shared / _popular variant), on every dataset it has ever been run on,
with its paired no-memory baseline from the same root where one exists."""
import glob, json, os
D="/home/ubuntu/24nam.nh/video_games_data"
MK=[("hit@1","H@1"),("ndcg@3","N@3"),("ndcg@5","N@5"),("ndcg@10","N@10")]
BAD=("_random","_noself","_shared","_popular","_consensus")
def load(p):
    r=json.load(open(p,encoding="utf-8"))
    if isinstance(r,dict): r=r.get("results",r.get("users",[]))
    return r
def met(rows,k): return sum((x.get("metrics") or {}).get(k) or 0 for x in rows)/len(rows)
hits=[]
for r in sorted(glob.glob(D+"/evaluation_results_*")):
    if not os.path.isdir(r) or "memrec" in os.path.basename(r): continue
    root=os.path.basename(r)[19:]
    for p in sorted(glob.glob(r+"/*/*.json")):
        if p.endswith(".summary.json"): continue
        b=os.path.basename(p)
        if "fmrec_topk" not in b: continue
        if any(x in b for x in BAD): continue
        try: rows=load(p)
        except Exception: continue
        if not rows: continue
        ds=os.path.basename(os.path.dirname(p))
        nc=rows[0].get("num_candidates") or len(rows[0].get("candidate_item_ids") or [])
        # paired no-memory in the same root+dataset
        base=None
        for q in sorted(glob.glob(os.path.join(r,ds,"*.json"))):
            if q.endswith(".summary.json"): continue
            if "nomemory" in os.path.basename(q):
                try:
                    br=load(q)
                    if br: base=br
                except Exception: pass
                break
        hits.append((root,ds,os.path.basename(r).find("gemma")>=0,nc,len(rows),rows,base,b))
print("%-34s %-26s %-6s %4s %5s   %7s %7s %7s %7s   %s"
      % ("result set","dataset","LLM","cand","n","H@1","N@3","N@5","N@10","dN@10 vs no-mem"))
seen=set()
for root,ds,isg,nc,n,rows,base,b in sorted(hits,key=lambda x:(-x[4],x[0],x[1])):
    key=(root,ds)
    if key in seen: continue
    seen.add(key)
    vals=[met(rows,k) for k,_ in MK]
    d=""
    if base:
        u=set(str(x["user_id"]) for x in rows)&set(str(x["user_id"]) for x in base)
        if u:
            A={str(x["user_id"]):(x.get("metrics") or {}) for x in rows}
            B={str(x["user_id"]):(x.get("metrics") or {}) for x in base}
            d="%+.4f"%(sum((A[x].get("ndcg@10") or 0)-(B[x].get("ndcg@10") or 0) for x in u)/len(u))
    print("%-34s %-26s %-6s %4d %5d   %7.4f %7.4f %7.4f %7.4f   %s"
          % (root[:34],ds[:26],"Gemma" if isg else "Qwen?",nc,n,*vals,d))
print()
ds_all=sorted({h[1] for h in hits})
print("datasets FMRec has ever been run on (%d): %s" % (len(ds_all), ", ".join(ds_all)))
