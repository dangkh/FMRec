#!/usr/bin/env python3
import glob, json, os
D="/home/ubuntu/24nam.nh/video_games_data"
DS=["Software_1000u","Prime_Pantry_1000u","Video_Game","Industrial_and_Scientific_1000u"]
MK=[("hit@1","H@1"),("ndcg@1","N@1"),("ndcg@3","N@3"),("ndcg@5","N@5"),("ndcg@10","N@10")]
CELLS={"Gemma 10-cand":("proto7_gemma_1000u",{"no memory":("proto7_gemma_nomem",()),
        "profile only":("proto7_gemma_profile",()),"shuffled":("proto7_gemma_shuffled",()),
        "random nbrs":("proto7_gemma_topkrand",()),"personal only":("proto7_gemma_same",()),
        "FMRec top-K":("proto7_gemma_topk.",("random",)),"MEMCF full":("proto7_gemma_full",())}),
 "Qwen 10-cand":("proto7_qwen_1000u",{"no memory":("nomemory",()),"profile only":("profileonly",()),
        "shuffled":("scopeshuffled_memory",()),"random nbrs":("fmrec_topk_random",()),
        "personal only":("scopesame_user",()),"FMRec top-K":("fmrec_topk_gk",()),
        "MEMCF full":("scopefull",())}),
 "Gemma 20-cand":("ladder_memcfproto_gemma_1000u",{"no memory":("nomemory",()),
        "profile only":("profileonly",()),"shuffled":("scopeshuffled_memory",()),
        "random nbrs":("fmrec_topk_random",()),"personal only":("scopesame_user",()),
        "FMRec top-K":("fmrec_topk_gk",()),"MEMCF full":("scopefull",())}),
 "Qwen 20-cand":("fmrec_ablation_1000u",{"no memory":("fmrec1k_nomem",()),
        "random nbrs":("fmrec_topk_random",()),"FMRec top-K":("scopedense_lgcn_fmrec_topk",("random",))}),
}
ORDER=["no memory","profile only","shuffled","random nbrs","personal only","FMRec top-K","MEMCF full"]
def load(p):
    r=json.load(open(p,encoding="utf-8"))
    if isinstance(r,dict): r=r.get("results",r.get("users",[]))
    return r
def find(root,ds,nd,ex=()):
    for p in sorted(glob.glob(os.path.join(D,"evaluation_results_"+root,ds,"*.json"))):
        if p.endswith(".summary.json"): continue
        b=os.path.basename(p)
        if nd in b and not any(e in b for e in ex): return p
for cell,(root,arms) in CELLS.items():
    print("="*74); print("## %s   (%s)"%(cell,root))
    print("%-16s %7s %7s %7s %7s %7s"%("arm",*[l for _,l in MK]))
    base=None
    for a in ORDER:
        if a not in arms: continue
        nd,ex=arms[a]; vals={k:[] for k,_ in MK}
        for ds in DS:
            p=find(root,ds,nd,ex)
            if not p: continue
            rows=load(p)
            for k,_ in MK: vals[k].append(sum((x.get("metrics") or {}).get(k) or 0 for x in rows)/len(rows))
        if not vals["ndcg@5"]: print("%-16s  (missing)"%a); continue
        m={k:sum(v)/len(v) for k,v in vals.items()}
        if a=="no memory": base=m
        d=""
        if base and a!="no memory": d="   dN@10 %+.4f"%(m["ndcg@10"]-base["ndcg@10"])
        print("%-16s %7.4f %7.4f %7.4f %7.4f %7.4f%s"%(a,*[m[k] for k,_ in MK],d))
    print()
