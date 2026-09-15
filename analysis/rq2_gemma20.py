#!/usr/bin/env python3
"""RQ2 on Gemma-3-4B at 20 candidates -- the harder task, where more headroom
might expose a cross-user effect that the 10-candidate protocol could not."""
import glob, json, os, random
BOOT,SEED,PERM=5000,2027,5000
D="/home/ubuntu/24nam.nh/video_games_data"
R=D+"/evaluation_results_ladder_memcfproto_gemma_1000u"
DS=["Software_1000u","Prime_Pantry_1000u","Video_Game","Industrial_and_Scientific_1000u"]
MET=[("ndcg@1","N@1"),("ndcg@3","N@3"),("ndcg@5","N@5"),("ndcg@10","N@10")]
def load(p):
    r=json.load(open(p,encoding="utf-8"))
    if isinstance(r,dict): r=r.get("results",r.get("users",[]))
    return {str(x["user_id"]):x for x in r}
def find(ds,nd,ex=()):
    for p in sorted(glob.glob(os.path.join(R,ds,"*.json"))):
        if p.endswith(".summary.json"): continue
        b=os.path.basename(p)
        if nd in b and not any(e in b for e in ex): return p
def strat(gs,seed=SEED,n=BOOT):
    r=random.Random(seed); m=[]
    for _ in range(n):
        m.append(sum(sum(g[r.randrange(len(g))] for _ in range(len(g)))/len(g) for g in gs)/len(gs))
    m.sort(); return m[int(.025*n)],m[int(.975*n)-1]
def perm(gs,seed=SEED,n=PERM):
    obs=abs(sum(sum(g)/len(g) for g in gs)/len(gs)); r=random.Random(seed); c=0
    for _ in range(n):
        s=sum(sum(x if r.random()<.5 else -x for x in g)/len(g) for g in gs)/len(gs)
        if abs(s)>=obs-1e-12: c+=1
    return (c+1)/(n+1)
ARMS={"topk":("scopedense_lgcn_fmrec_topk",("_random",)),
      "random":("scopedense_lgcn_fmrec_topk_random",()),
      "same":("scopesame_user",()), "shuf":("scopeshuffled_memory",()),
      "nomem":("nomemory",()), "prof":("profileonly",())}
data={}
for ds in DS:
    data[ds]={}
    for k,(nd,ex) in ARMS.items():
        p=find(ds,nd,ex)
        if p: data[ds][k]=load(p)
CON=[("topk","nomem","memory vs none (sanity)"),
     ("topk","shuf","lesson CONTENT matters"),
     ("topk","random","RQ2b: similar vs RANDOM neighbours"),
     ("topk","same","RQ2a: cross-user ON TOP of personal"),
     ("same","nomem","personal alone vs none")]
for A,B,why in CON:
    have=[ds for ds in DS if A in data[ds] and B in data[ds]]
    if not have: print("\n=== %s - %s MISSING ==="%(A,B)); continue
    print("\n=== %s - %s   (%s) ===" % (A,B,why))
    pool={mk:[] for mk,_ in MET}; bad=0
    for ds in have:
        Ad,Bd=data[ds][A],data[ds][B]; u=sorted(set(Ad)&set(Bd))
        bad+=sum(1 for x in u if (Ad[x].get("num_candidates") or 0)!=(Bd[x].get("num_candidates") or 0))
        for mk,_ in MET:
            pool[mk].append([((Ad[x].get("metrics") or {}).get(mk) or 0)
                            -((Bd[x].get("metrics") or {}).get(mk) or 0) for x in u])
    if bad: print("   !! %d candidate-count mismatches -- NOT PAIRABLE"%bad); continue
    for mk,ml in MET:
        gs=pool[mk]; mean=sum(sum(g)/len(g) for g in gs)/len(gs)
        lo,hi=strat(gs); p=perm(gs); pos=sum(1 for g in gs if sum(g)/len(g)>0)
        print("   %-5s %+.4f [%+.4f,%+.4f] p=%.4f %d/%d %s"
              % (ml,mean,lo,hi,p,pos,len(gs),"SIG" if (lo>0 or hi<0) else "n.s."))
