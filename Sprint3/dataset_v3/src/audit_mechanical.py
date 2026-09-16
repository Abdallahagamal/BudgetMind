import json, re, collections, unicodedata, hashlib
V3='Sprint3/dataset_v3/complexity_dataset_v3_3.jsonl'
rows=[json.loads(l) for l in open(V3) if l.strip()]
N=len(rows); L=['Low','Medium','High']
flags=collections.defaultdict(list)          # task_id -> [flag,...]
def F(tid,f): flags[tid].append(f)

print("="*100); print("  6B. MALFORMED / INTEGRITY"); print("="*100)
INV=re.compile('[​-‏‪-‮⁠-⁯﻿­]|[\U000E0000-\U000E007F]')
CTRL=re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
checks=collections.Counter()
for r in rows:
    t=r['text']; tid=r['task_id']
    if not t or not t.strip(): F(tid,'EMPTY'); checks['empty']+=1
    if INV.search(t): F(tid,'HIDDEN_UNICODE'); checks['hidden_unicode']+=1
    if CTRL.search(t): F(tid,'CONTROL_CHARS'); checks['control_chars']+=1
    if t!=unicodedata.normalize('NFKC',t): F(tid,'NOT_NFKC'); checks['not_nfkc']+=1
    if '�' in t: F(tid,'REPLACEMENT_CHAR'); checks['replacement_char']+=1
    if re.search(r'\.\.\.$|\[\.\.\.|truncated',t,re.I): F(tid,'TRUNCATION_MARKER'); checks['truncation_marker']+=1
    if len(t.strip())<15: F(tid,'VERY_SHORT'); checks['very_short']+=1
    # construction-note / metadata leakage into text
    if re.search(r'\b(task_id|source_meta|rubric_scores|annotation_version|row_ref)\b',t): F(tid,'METADATA_IN_TEXT'); checks['metadata_in_text']+=1
    # duplicated text within the row (paste artefact)
    h=t.strip()
    if len(h)>60 and h[:len(h)//2].strip()==h[len(h)//2:].strip(): F(tid,'SELF_DUPLICATED'); checks['self_duplicated']+=1
for k,v in checks.most_common(): print(f"  {k:<22}{v:>5}")
if not checks: print("  no integrity flags")

print("\n"+"="*100); print("  6A. DUPLICATES / NEAR-DUPLICATES"); print("="*100)
norm=[re.sub(r'\s+',' ',unicodedata.normalize('NFKC',r['text'])).strip().lower() for r in rows]
dupe=collections.Counter(norm)
exact=[k for k,v in dupe.items() if v>1]
print(f"  exact duplicate texts (normalised): {len(exact)}")
for e in exact[:5]:
    ids=[rows[i]['task_id'] for i,x in enumerate(norm) if x==e]
    print(f"    {ids} :: {e[:80]}")
    for i in ids[1:]: F(i,'EXACT_DUPLICATE')
TOK=re.compile(r'[a-z0-9]+',re.I)
tok=[{w.lower() for w in TOK.findall(t) if len(w)>2} for t in norm]
bysrc=collections.defaultdict(list)
for i,r in enumerate(rows): bysrc[r['source']].append(i)
pairs=[]
for s,idx in bysrc.items():
    for a in range(len(idx)):
        for b in range(a+1,len(idx)):
            i,j=idx[a],idx[b]
            u=tok[i]|tok[j]
            if not u: continue
            J=len(tok[i]&tok[j])/len(u)
            if J>=0.75: pairs.append((round(J,3),rows[i]['task_id'],rows[j]['task_id'],s,
                                      rows[i]['complexity'],rows[j]['complexity']))
print(f"  near-duplicate pairs (within-source Jaccard>=0.75): {len(pairs)}")
bys=collections.Counter(p[3] for p in pairs)
for k,v in bys.most_common(): print(f"    {k:<44}{v:>4} pairs")
lab=[p for p in pairs if p[4]!=p[5]]
print(f"  near-dup pairs with DIFFERENT labels: {len(lab)}")
for p in lab[:8]: print(f"    {p}")

print("\n"+"="*100); print("  6D. RUBRIC-SCORE -> BAND CONSISTENCY (rows that have scores)"); print("="*100)
def band_of(rs):
    d=[rs['D1'],rs['D2'],rs['D3'],rs['D4'],rs['D5'],rs['D6']]; k=rs['K']; tot=sum(d)
    b='Low' if tot<=3 else ('Medium' if tot<=7 else 'High')
    o={'Low':0,'Medium':1,'High':2}; inv={v:kk for kk,v in o.items()}; l=o[b]; fired=[]
    if d[5]==2 and l<1: l=1; fired.append('O1')
    if d[3]==2 and l<1: l=1; fired.append('O2')
    if d[1]==2 and d[4]==2 and l<2: l=2; fired.append('O3')
    if k==1 and l<1: l=1; fired.append('K1')
    if k==2 and l<2: l=2; fired.append('K2')
    return inv[l],tot,fired
bad=0; totmismatch=0
for r in rows:
    rs=r['rubric_scores']
    if not rs: continue
    b,tot,fired=band_of(rs)
    if b!=r['complexity']:
        bad+=1; F(r['task_id'],'BAND_MISMATCH')
        if bad<=10: print(f"    {r['task_id']} scores={[rs[k] for k in ('D1','D2','D3','D4','D5','D6')]} K={rs['K']} "
                          f"-> rubric says {b}, label says {r['complexity']}")
    if rs.get('total') is not None and rs['total']!=tot:
        totmismatch+=1; F(r['task_id'],'TOTAL_MISMATCH')
    if not (0<=rs['K']<=2) or any(not (0<=rs[k]<=2) for k in ('D1','D2','D3','D4','D5','D6')):
        F(r['task_id'],'SCORE_OUT_OF_RANGE')
print(f"  band mismatches: {bad} of {sum(1 for r in rows if r['rubric_scores'])} scored rows")
print(f"  stored total != recomputed: {totmismatch}")
print(f"  K=1 rows: {sum(1 for r in rows if r['rubric_scores'] and r['rubric_scores']['K']==1)}   "
      f"K=2 rows: {sum(1 for r in rows if r['rubric_scores'] and r['rubric_scores']['K']==2)}")

print("\n"+"="*100); print("  6E. SOURCE LEAKAGE (n-grams that identify a source)"); print("="*100)
docs=[re.findall(r"[a-z0-9']+",t) for t in norm]; src=[r['source'] for r in rows]
cnt=collections.Counter(); per=collections.defaultdict(collections.Counter)
for d,s in zip(docs,src):
    seen=set()
    for n_ in (1,2,3):
        for i in range(len(d)-n_+1):
            g=' '.join(d[i:i+n_])
            if g in seen: continue
            seen.add(g); cnt[g]+=1; per[g][s]+=1
tot_s=collections.Counter(src); leaks=[]
for g,c in cnt.items():
    if c<8: continue
    s,k=per[g].most_common(1)[0]
    if k/c>=0.90 and k/tot_s[s]>=0.90: leaks.append((round(k/c,2),round(k/tot_s[s],2),k,s,g))
leaks.sort(key=lambda x:-x[2])
bysrc_leak=collections.Counter(x[3] for x in leaks)
print(f"  n-grams with >=0.90 precision AND >=0.90 recall for a source: {len(leaks)}")
for k,v in bysrc_leak.most_common(): print(f"    {k:<44}{v:>4} n-grams  e.g. {[x[4] for x in leaks if x[3]==k][:4]}")
json.dump({k:v for k,v in flags.items()}, open('mech_flags.json','w'))  # gitignored side-output
print(f"\n  rows carrying >=1 mechanical flag: {len(flags)}")
print(f"  flag types: {dict(collections.Counter(f for v in flags.values() for f in v))}")
