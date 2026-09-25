# degree floor under TextDDI's twosides evaluator variant (first 200 labels only; TextDDI/twosides/evaluate_twosides.py:186-205)
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
d='DDI-Bench/DDI_Ben/DDI_Ben/data/twosides_random/'
def load(f):
    H,T,L,P=[],[],[],[]
    for l in open(d+f):
        h,t,r,p=l.split(); H.append(int(h)); T.append(int(t)); L.append(np.array(r.split(','),int)); P.append(int(p))
    return np.array(H),np.array(T),np.stack(L),np.array(P)
h,t,L,P=load('train.txt'); known=np.zeros(645,bool); known[np.loadtxt(d+'train_set.txt',dtype=int)]=True
C=np.zeros((645,209)); np.add.at(C,h[P==1],L[P==1]); np.add.at(C,t[P==1],L[P==1]); G=np.log1p(C*known[:,None])
for S in ['S0','S1']:
    th,tt,tL,tP=load(f'test_{S}.txt'); s=G[th]+G[tt]
    pos=tP==1; neg=~pos
    roc=[];pr=[]
    for r in range(200):
        idx=tL[pos][:,r]>0
        lab=[1]*idx.sum()+[0]*idx.sum(); sc=list(s[pos][idx,r])+list(s[neg][idx,r])
        roc.append(roc_auc_score(lab,sc)); pr.append(average_precision_score(lab,sc))
    print(S,'TextDDI-200-label variant ROC',round(100*np.mean(roc),2),'PR',round(100*np.mean(pr),2))
