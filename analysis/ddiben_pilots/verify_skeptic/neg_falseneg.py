import numpy as np
from collections import defaultdict
d='DDI-Bench/DDI_Ben/DDI_Ben/data/twosides_random/'
files=['train','valid_S0','test_S0','valid_S1','test_S1','valid_S2','test_S2']
pos_ord=defaultdict(lambda: np.zeros(209,int)); pos_un=defaultdict(lambda: np.zeros(209,int))
rows={}
for f in files:
    rows[f]=[]
    for l in open(d+f+'.txt'):
        h,t,r,p=l.split(); h,t,p=int(h),int(t),int(p); v=np.array(r.split(','),int)
        rows[f].append((h,t,v,p))
        if p==1: pos_ord[(h,t)]|=v; pos_un[frozenset((h,t))]|=v
te=rows['test_S1']; n=0; ordc=0; unc=0; lab_over=0; lab_tot=0
for h,t,v,p in te:
    if p: continue
    n+=1
    if (h,t) in pos_ord: ordc+=1
    k=frozenset((h,t))
    if k in pos_un:
        unc+=1; lab_over+=int((pos_un[k]&v).sum())
    lab_tot+=int(v.sum())
print('test_S1 negs',n,'ordered pair is a positive somewhere',ordc,'unordered',unc,'; neg (row,label) cells that are positive for SAME label on that unordered pair:',lab_over,'of',lab_tot, f'({100*lab_over/lab_tot:.2f}%)')
