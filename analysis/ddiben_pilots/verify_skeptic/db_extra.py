# extra checks: rows touching train_set drugs absent from train.txt; row- and drug-cluster bootstrap of S1/S2 macro-F1 for the ME rule
import numpy as np, importlib.util, sys
sys.argv=['x','drugbank_random']
src=open('drugbank_me_indep.py').read().split('# sensitivity on VALID only')[0].replace("print(f, res[f], flush=True)","pass")
ns={}; exec(src, ns)
data, sets, pred = ns['data'], ns['sets'], ns['pred']
from sklearn.metrics import f1_score
tr=data['train']; filed=set(tr[:,:2].ravel().tolist()); ghost=sets['train']-filed
rng=np.random.default_rng(0)
for f in ['test_S1','test_S2']:
    e=data[f]; y=e[:,2]; p=pred(e)
    g=np.isin(e[:,0],list(ghost))|np.isin(e[:,1],list(ghost))
    print(f,'rows touching the',len(ghost),'train_set drugs with no train.txt rows:',int(g.sum()))
    bs=[f1_score(y[i],p[i],average='macro') for i in (rng.integers(0,len(y),len(y)) for _ in range(200))]
    # drug-cluster bootstrap over NEW drugs
    newd=np.where(np.isin(e[:,0],list(sets['train'])),e[:,1],e[:,0]) if f=='test_S1' else e[:,0]
    ud=np.unique(newd); idx_by={d:np.flatnonzero(newd==d) for d in ud}
    bd=[]
    for _ in range(200):
        s=rng.choice(ud,len(ud)); ii=np.concatenate([idx_by[d] for d in s]); bd.append(f1_score(y[ii],p[ii],average='macro'))
    print(f,'macroF1',round(100*f1_score(y,p,average='macro'),2),'row-bootstrap 95%',np.round(100*np.percentile(bs,[2.5,97.5]),1),'new-drug(head for S2)-cluster bootstrap 95%',np.round(100*np.percentile(bd,[2.5,97.5]),1))
