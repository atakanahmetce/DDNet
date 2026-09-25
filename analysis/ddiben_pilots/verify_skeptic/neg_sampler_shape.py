"""How are TWOSIDES S1 negative drugs distributed? uniform over train_set/test_set drugs, or degree-proportional?"""
import numpy as np
from collections import Counter
from scipy.stats import spearmanr, chisquare
ROOT='DDI-Bench/DDI_Ben/DDI_Ben/data/'
for SPLIT in ['twosides_random','twosides_cluster']:
    d=ROOT+SPLIT+'/'
    def load(fn):
        return [(int(a),int(b),int(p)) for a,b,_,p in (l.split() for l in open(d+fn))]
    tr=load('train.txt'); known=set(np.loadtxt(d+'train_set.txt',dtype=int).tolist()); newt=set(np.loadtxt(d+'test_set.txt',dtype=int).tolist())
    deg=Counter()
    for a,b,p in tr:
        if p==1: deg[a]+=1; deg[b]+=1
    for S in ['S0','S1','S2']:
        te=load(f'test_{S}.txt')
        posk=Counter(); negk=Counter(); posn=Counter(); negn=Counter()
        for a,b,p in te:
            for x in (a,b):
                (posk if p==1 else negk)[x]+= (x in known)
                (posn if p==1 else negn)[x]+= (x not in known)
        K=sorted(known); Nw=sorted(newt)
        nk=np.array([negk[x] for x in K]); pk=np.array([posk[x] for x in K]); dg=np.array([deg[x] for x in K])
        out=f'{SPLIT} {S}: '
        if nk.sum()>0:
            out+=f'neg-known count: mean {nk.mean():.2f} sd {nk.std():.2f} (Poisson-uniform sd ~{np.sqrt(nk.mean()):.2f}); chi2 uniform p={chisquare(nk).pvalue:.3g}; spearman(neg count, train deg)={spearmanr(nk,dg)[0]:.3f}; spearman(pos count, train deg)={spearmanr(pk,dg)[0]:.3f}; mean train deg of known drug in pos rows={np.average(dg,weights=pk):.1f} vs neg rows={np.average(dg,weights=nk):.1f}; '
        nn=np.array([negn[x] for x in Nw]); pn=np.array([posn[x] for x in Nw])
        if nn.sum()>0:
            out+=f'neg-new count over test_set: mean {nn.mean():.2f} sd {nn.std():.2f}; chi2 uniform p={chisquare(nn).pvalue:.3g}; spearman(neg,pos count)={spearmanr(nn,pn)[0]:.3f}'
        print(out)
