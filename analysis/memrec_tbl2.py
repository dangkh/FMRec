#!/usr/bin/env python3
import glob, json, os
R="/home/ubuntu/24nam.nh/video_games_data/evaluation_results_memrec_1k_static_trainhistory_fair"
R2=R+"_rerun_failed7_v2"
DS=["Software_1000u","Prime_Pantry_1000u","Video_Game","Industrial_and_Scientific_1000u",
    "CDs_and_Vinyl_1000u","Digital_Music_1000u"]
def get(ds,v):
    for root in (R,R2):
        p=glob.glob("%s/%s/%s*.summary.json"%(root,ds,v))
        if p:
            d=json.load(open(p[0])); return d.get("metrics") or {}
    return {}
print("%-26s %-12s %7s %7s %7s" % ("dataset","arm","H@1","N@5","N@10"))
lift={"H@1":[], "NDCG@5":[], "NDCG@10":[]}
for ds in DS:
    nm=get(ds,"no_memory"); rd=get(ds,"memrec_read"); fl=get(ds,"memrec_full")
    for v,m in (("no_memory",nm),("read(W off)",rd),("full(W on)",fl)):
        if not m: print("%-26s %-12s %7s" % (ds[:26],v,"MISSING")); continue
        print("%-26s %-12s %7.4f %7.4f %7.4f" % (ds[:26],v,m.get("H@1",0),m.get("NDCG@5",0),m.get("NDCG@10",0)))
    if nm and rd:
        for k in lift: lift[k].append(rd.get(k,0)-nm.get(k,0))
    print()
print("="*60)
print("MemRec lift of Stage-R (read, W off) over its OWN no-memory arm")
for k,lbl in (("H@1","H@1"),("NDCG@5","N@5"),("NDCG@10","N@10")):
    v=lift[k]; pos=sum(1 for x in v if x>0)
    print("  %-5s mean %+0.4f   datasets favourable %d/%d   values %s"
          % (lbl, sum(v)/len(v), pos, len(v), " ".join("%+.4f"%x for x in v)))
print()
print("RANDOM-RANKING reference at 20 candidates: H@1 0.050  H@3 0.150  H@5 0.250  H@10 0.500")
