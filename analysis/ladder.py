#!/usr/bin/env python3
"""The three-rung ladder: no memory < one stranger's lesson < one's own lesson.
All three arms are 10-candidate, n=1,000, tb420, and paired on the same users."""
import glob, json, os, random
BOOT,SEED=5000,2027
D="/home/ubuntu/24nam.nh/video_games_data"
DS=["Software_1000u","Prime_Pantry_1000u","Video_Game","Industrial_and_Scientific_1000u"]
MET=[("hit@1","H@1"),("ndcg@5","N@5"),("ndcg@10","N@10")]
ARMS={"nomem":(D+"/evaluation_results_proto7_gemma_1000u","proto7_gemma_nomem",()),
      "stranger":(D+"/evaluation_results_noself1_gemma","ns1_noself1.",("rand",)),
      "own":(D+"/evaluation_results_same1_gemma_1000u","same1g_same1",())}
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
data={ds:{k:load(find(rt,ds,nd,ex)) for k,(rt,nd,ex) in ARMS.items()} for ds in DS}
print("%-10s %8s %8s %8s" % ("arm","H@1","N@5","N@10"))
abs_={k:{} for k in ARMS}
for k in ("nomem","stranger","own"):
    row=[]
    for mk,_ in MET:
        v=sum(sum(data[ds][k][x].get(mk) or 0 for x in data[ds][k])/len(data[ds][k]) for ds in DS)/4
        abs_[k][mk]=v; row.append(v)
    print("%-10s %8.4f %8.4f %8.4f" % (k,*row))
print("\n%-28s %9s %9s %9s" % ("gain over no-memory","H@1","N@5","N@10"))
for k in ("stranger","own"):
    cells=[]
    for mk,_ in MET:
        gs=[[ (data[ds][k][x].get(mk) or 0)-(data[ds]["nomem"][x].get(mk) or 0)
              for x in sorted(set(data[ds][k])&set(data[ds]["nomem"]))] for ds in DS]
        mean=sum(sum(g)/len(g) for g in gs)/4; lo,hi=strat(gs)
        cells.append("%+.4f%s" % (mean,"*" if (lo>0 or hi<0) else " "))
    print("%-28s %9s %9s %9s" % (k,*cells))
print("\n%-28s %9s %9s %9s" % ("stranger retains ... of own","H@1","N@5","N@10"))
cells=[]
for mk,_ in MET:
    gs_s=sum(abs_["stranger"][mk]-abs_["nomem"][mk] for _ in [0])
    gs_o=sum(abs_["own"][mk]-abs_["nomem"][mk] for _ in [0])
    cells.append("%.0f%%" % (100*gs_s/gs_o) if gs_o else "n/a")
print("%-28s %9s %9s %9s" % ("",*cells))
