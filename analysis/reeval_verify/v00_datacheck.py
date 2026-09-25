import numpy as np, collections
import os; D=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
cfg={'D1':('data_1','cb1','n2v_embeddings/cb1_embeddings/cb1_hop1_128_1.2_1.2.txt'),
     'D2':('data_2','cb2','n2v_embeddings/cb2_embeddings/cb2_hop1_128_1.2_1.2.txt'),
     'MP':('data_mp','mp','n2v_embeddings/mp_embeddings/mp_128_1.2_1.2.txt')}
for ds,(dd,tag,ef) in cfg.items():
    nm=[l.strip() for l in open(f'{D}/datasets/{dd}/names.txt') if l.strip()]
    s=set(nm)
    lines=[l.strip() for l in open(f'{D}/datasets/{dd}/interactions.txt') if l.strip()]
    rows=[l.split(',') for l in lines]
    ncols=collections.Counter(len(r) for r in rows)
    rows2=[(r[0].strip(),r[1].strip()) for r in rows]
    raw_ws=sum(1 for r in rows if r[0]!=r[0].strip() or r[1]!=r[1].strip())
    inboth=[(a,b) for a,b in rows2 if a in s and b in s]
    dset=set(inboth)
    selfp=sum(1 for a,b in dset if a==b)
    asym=sum(1 for a,b in dset if (b,a) not in dset)
    und=set(tuple(sorted(p)) for p in dset if p[0]!=p[1])
    outside=len(rows2)-len(inboth)
    emb_ids=[l.split(' ')[0] for l in open(f'{D}/{ef}')]
    print(ds,'names',len(nm),'uniq',len(s),'lines',len(lines),'ncols',dict(ncols),'ws',raw_ws,'outside-names',outside,
          'directed-in',len(dset),'dups',len(inboth)-len(dset),'self',selfp,'asym',asym,'unordered',len(und),
          'density',len(und)/(len(nm)*(len(nm)-1)/2))
    print('  emb ids',len(emb_ids),'set==names',set(emb_ids)==s,'same order',emb_ids==nm)
    # paths
    P={}
    for l in open(f'{D}/meta_paths/{tag}_3_paths.txt'):
        r=l.strip('\n').split(','); P[(r[0],r[1])]=np.array(r[-1].split(' '),float)
    missing=sum(1 for a in nm for b in nm if (a,b) not in P)
    asymP=max(np.abs(P[(a,b)]-P[(b,a)]).max() for a in nm for b in nm if a<b and (a,b) in P and (b,a) in P)
    print('  paths entries',len(P),'missing',missing,'max |P(a,b)-P(b,a)|',asymP,
          'self nonzero', sum(1 for a in nm if P[(a,a)].any()))
    S=np.loadtxt(f'{D}/datasets/{dd}/sim_arr.txt'); print('  S shape',S.shape,'asym',np.abs(S-S.T).max())
