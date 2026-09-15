#!/usr/bin/env python3
"""Is memrec_full actually reranking, or falling back to candidate order?
If predictions == candidates verbatim, the LLM output was discarded."""
import glob, json, os
R="/home/ubuntu/24nam.nh/video_games_data/evaluation_results_memrec_1k_static_trainhistory_fair"
R2=R+"_rerun_failed7_v2"
DS=["Software_1000u","Prime_Pantry_1000u","Video_Game","Industrial_and_Scientific_1000u",
    "CDs_and_Vinyl_1000u","Digital_Music_1000u"]
print("%-24s %-12s %6s %8s %8s %8s" % ("dataset","arm","n","identity","n_facets","H@1"))
for ds in DS:
    for v in ("no_memory","memrec_read","memrec_full"):
        p=None
        for root in (R,R2):
            g=glob.glob("%s/%s/%s*.ranking.json"%(root,ds,v))
            if g: p=g[0]; break
        if not p: continue
        rows=json.load(open(p,encoding="utf-8"))
        ident=sum(1 for x in rows if list(x.get("predictions") or [])==list(x.get("candidates") or []))
        nf=sum(1 for x in rows if x.get("facets"))
        h1=sum((x.get("metrics") or {}).get("H@1") or 0 for x in rows)/max(len(rows),1)
        print("%-24s %-12s %6d %7.1f%% %8d %8.4f"
              % (ds[:24],v,len(rows),100.0*ident/max(len(rows),1),nf,h1))
    print()
