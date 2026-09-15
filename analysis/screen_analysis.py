#!/usr/bin/env python3
"""screen300 (noself + K sweep) and noself1, against 1,000-user arms subset to
the same users. Every contrast is paired per user, bootstrapped within dataset
and pooled stratified, plus a 5,000-flip permutation test.

The decisive contrast is NOSELF1 vs SAME1: one fact, slot 1 -- the only slot C2
shows to matter -- differing solely in whose lesson occupies it."""
import glob, json, os, random, re, sys
BOOT, SEED, PERM = 5000, 2027, 5000
D  = "/home/ubuntu/24nam.nh/video_games_data"
DS = ["Software_1000u","Prime_Pantry_1000u","Video_Game","Industrial_and_Scientific_1000u"]
MET= [("hit@1","H@1"),("ndcg@5","N@5"),("ndcg@10","N@10")]

SC = D+"/evaluation_results_screen300_gemma"
NS = D+"/evaluation_results_noself1_gemma"
PR = D+"/evaluation_results_proto7_gemma_1000u"
S1 = D+"/evaluation_results_same1_gemma_1000u"
AB = D+"/evaluation_results_fmrec_ablation_1000u"

# arm -> (root, filename needle, excluded needles)
ARMS = {
 "noself3":    (SC,"scr300_noself3.",("rand",)), "noself3rand":(SC,"scr300_noself3rand",()),
 "k1":(SC,"scr300_k1",()), "k2":(SC,"scr300_k2",()), "k3":(SC,"scr300_k3",()), "k5":(SC,"scr300_k5",()),
 "noself1":    (NS,"ns1_noself1.",("rand",)),     "noself1rand":(NS,"ns1_noself1rand",()),
 "same1":      (S1,"same1g_same1",()),
 "topk":       (PR,"proto7_gemma_topk.",("random",)),
 "nomemory":   (PR,"proto7_gemma_nomem",()),   # 10-candidate baseline; AB root is 20-cand
 "same_user":  (PR,"proto7_gemma_same",()),
}
CONTRASTS = [
 ("noself1","same1",     "DECISIVE: 1 cross fact vs 1 personal fact, same slot"),
 ("noself1","nomemory",  "does a stranger's single lesson help at all?"),
 ("noself1","noself1rand","is the cross-user pick collaborative or arbitrary?"),
 ("noself3","same_user", "3 cross vs 3 personal"),
 ("noself3","noself3rand","3 cross: nearest vs random neighbours"),
 ("noself3","nomemory",  "3 cross facts vs no memory"),
 ("k2","k1","dose +1 cross"), ("k3","k2","dose +1 cross"), ("k5","k3","dose +2 cross"),
 ("k5","k1","dose: 5 cross vs 1 cross"),
]

def load(p):
    rows=json.load(open(p,encoding="utf-8"))
    if isinstance(rows,dict): rows=rows.get("results",rows.get("users",[]))
    return {str(r["user_id"]):{"m":(r.get("metrics") or {}),
            "nc":r.get("num_candidates") or len(r.get("candidate_item_ids") or []),
            "gt":tuple(r.get("ground_truth_item_ids") or ())} for r in rows}
def find(root,ds,needle,ex=()):
    for p in sorted(glob.glob(os.path.join(root,ds,"*.json"))):
        if p.endswith(".summary.json"): continue
        b=os.path.basename(p)
        if needle in b and not any(x in b for x in ex): return p
def boot(d,seed=SEED,n=BOOT):
    if len(d)<3: return float("nan"),float("nan"),False
    r=random.Random(seed);k=len(d)
    m=sorted(sum(d[r.randrange(k)] for _ in range(k))/k for _ in range(n))
    return m[int(.025*n)],m[int(.975*n)-1],(m[int(.025*n)]>0 or m[int(.975*n)-1]<0)
def strat(gs,seed=SEED,n=BOOT):
    gs=[g for g in gs if g]
    if not gs: return float("nan"),float("nan"),False
    r=random.Random(seed); m=[]
    for _ in range(n):
        m.append(sum(sum(g[r.randrange(len(g))] for _ in range(len(g)))/len(g) for g in gs)/len(gs))
    m.sort(); return m[int(.025*n)],m[int(.975*n)-1],(m[int(.025*n)]>0 or m[int(.975*n)-1]<0)
def perm(gs,seed=SEED,n=PERM):
    """sign-flip test on the stratified pooled mean"""
    gs=[g for g in gs if g]
    if not gs: return float("nan")
    obs=abs(sum(sum(g)/len(g) for g in gs)/len(gs)); r=random.Random(seed); c=0
    for _ in range(n):
        s=sum(sum(x if r.random()<.5 else -x for x in g)/len(g) for g in gs)/len(gs)
        if abs(s)>=obs-1e-12: c+=1
    return (c+1)/(n+1)

def cfg_of(path):
    """pull the confoundable knobs out of the run filename"""
    b=os.path.basename(path)
    g={}
    for k in ("mf","mw","tb","gk","nk"):
        m=re.search(r"_%s(\d+)_" % k, b)
        if m: g[k]=m.group(1)
    return g

CFG={ds:{} for ds in DS}
data={ds:{} for ds in DS}
for ds in DS:
    for k,(root,nd,ex) in ARMS.items():
        p=find(root,ds,nd,ex)
        if p:
            data[ds][k]=load(p); CFG[ds][k]=cfg_of(p)
print("arms found per dataset:")
for ds in DS: print("  %-32s %s" % (ds, " ".join(sorted(data[ds]))))

for A,B,why in CONTRASTS:
    have=[ds for ds in DS if A in data[ds] and B in data[ds]]
    if not have:
        print("\n=== %s - %s : MISSING ===" % (A,B)); continue
    print("\n" + "="*78)
    print("=== %s  -  %s   (%s) ===" % (A,B,why))
    ca,cb=CFG[have[0]].get(A,{}),CFG[have[0]].get(B,{})
    diff={k for k in set(ca)|set(cb) if ca.get(k)!=cb.get(k)}
    if diff:
        print("  !! CONFOUNDED: arms differ on %s  (%s vs %s) -- this is not a clean"
              % (sorted(diff), {k:ca.get(k) for k in sorted(diff)},
                 {k:cb.get(k) for k in sorted(diff)}))
        print("     ownership contrast; the knob difference is an alternative explanation.")
    pool={mk:[] for mk,_ in MET}
    for ds in have:
        Ad,Bd=data[ds][A],data[ds][B]; u=sorted(set(Ad)&set(Bd))
        if not u: continue
        bad_nc=sum(1 for x in u if Ad[x]["nc"]!=Bd[x]["nc"])
        bad_gt=sum(1 for x in u if Ad[x]["gt"]!=Bd[x]["gt"])
        if bad_nc or bad_gt:
            print("  %-32s SKIPPED - NOT PAIRABLE: %d candidate-count, %d ground-truth mismatches"
                  % (ds,bad_nc,bad_gt)); continue
        out=[]
        for mk,ml in MET:
            d=[(Ad[x]["m"].get(mk) or 0.0)-(Bd[x]["m"].get(mk) or 0.0) for x in u]
            lo,hi,sig=boot(d); pool[mk].append(d)
            out.append("%s %+.4f [%+.4f,%+.4f]%s" % (ml,sum(d)/len(d),lo,hi," SIG" if sig else ""))
        print("  %-32s n=%-4d %s" % (ds,len(u)," | ".join(out)))
    print("  " + "-"*74)
    for mk,ml in MET:
        gs=pool[mk]
        if not gs: continue
        mean=sum(sum(g)/len(g) for g in gs)/len(gs)
        lo,hi,sig=strat(gs); p=perm(gs)
        pos=sum(1 for g in gs if sum(g)/len(g)>0)
        print("  POOLED %-5s %+.4f [%+.4f,%+.4f] perm p=%.4f  %d/%d datasets  %s"
              % (ml,mean,lo,hi,p,pos,len(gs),"SIG" if sig else "n.s."))
