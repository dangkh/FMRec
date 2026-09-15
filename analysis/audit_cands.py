#!/usr/bin/env python3
"""Candidate-set audit. A paired contrast is only valid between arms with the
SAME candidate count; mixing a 10-candidate arm with a 20-candidate one produces
a huge spurious 'effect' that is really just task difficulty."""
import glob, json, os
D="/home/ubuntu/24nam.nh/video_games_data"
DS="Software_1000u"
ROOTS=["evaluation_results_screen300_gemma","evaluation_results_proto7_gemma_1000u",
       "evaluation_results_same1_gemma_1000u","evaluation_results_fmrec_ablation_1000u",
       "evaluation_results_p1_gemma_1000u"]
print("%-42s %-58s %6s %5s %8s" % ("root","arm file (trimmed)","users","cand","N@10"))
for r in ROOTS:
    for p in sorted(glob.glob(os.path.join(D,r,DS,"*.json"))):
        if p.endswith(".summary.json"): continue
        try:
            rows=json.load(open(p,encoding="utf-8"))
        except Exception as e:
            print("%-42s READ FAIL %s" % (r,e)); continue
        if isinstance(rows,dict): rows=rows.get("results",rows.get("users",[]))
        if not rows: continue
        h={}
        for x in rows:
            n=x.get("num_candidates") or len(x.get("candidate_item_ids") or [])
            h[n]=h.get(n,0)+1
        n10=sum((x.get("metrics") or {}).get("ndcg@10") or 0 for x in rows)/len(rows)
        b=os.path.basename(p)
        tag=b.split("_promptcompact_score_")[-1].replace(".json","")[:56]
        print("%-42s %-58s %6d %5s %8.4f" % (r[19:], tag, len(rows),
              ",".join(str(k) for k in sorted(h)), n10))
