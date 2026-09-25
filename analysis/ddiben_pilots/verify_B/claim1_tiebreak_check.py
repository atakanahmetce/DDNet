"""Check whether the small S2 gap (my pilot-recipe re-implementation 22.31/40.11/22.97 vs pilot 22.3/40.5/23.5)
comes from kNN tie-breaking / float32 similarity: float32 matrix-product Tanimoto + default (quicksort) argsort."""
import sys; sys.path.insert(0, '.')
import pickle, numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
from common import *
R, NE = 86, 1710
data = load_drugbank(); sets = load_sets('drugbank'); tr = data['train']
x = pickle.load(open(REPO + '/initial/drugbank/DB_molecular_feats.pkl', 'rb'))
F = (np.array([np.asarray(v, dtype=np.float32) for v in x['Morgan_Features']]) > 0).astype(np.float32)
for dtype, kind in [(np.float64, 'stable'), (np.float32, 'stable'), (np.float32, 'quicksort'), (np.float64, 'quicksort')]:
    Fd = F.astype(dtype); inter = Fd @ Fd.T; c = Fd.sum(1); Tm = inter / np.maximum(c[:, None] + c[None, :] - inter, 1)
    Ch = np.zeros((NE, R)); Ct = np.zeros((NE, R)); np.add.at(Ch, (tr[:, 0], tr[:, 2]), 1); np.add.at(Ct, (tr[:, 1], tr[:, 2]), 1)
    prior = np.bincount(tr[:, 2], minlength=R) + 1.0; prior /= prior.sum()
    ts = np.array(sorted(sets['train'])); known = np.zeros(NE, bool); known[ts] = True
    def dist(M):
        out = (M + prior) / (M.sum(1, keepdims=True) + 1)
        for d in np.where(~known)[0]:
            s = Tm[d, ts]; nb = ts[np.argsort(-s, kind=kind)[:10]]; w = Tm[d, nb]
            a = (w[:, None] * M[nb]).sum(0) / max(w.sum(), 1e-9) * 10
            out[d] = (a + prior) / (a.sum() + 1)
        return out
    Ph, Pt = dist(Ch), dist(Ct)
    res = []
    for s in ['test_S1', 'test_S2']:
        a = data[s]; p = np.argmax(Ph[a[:, 0]] * Pt[a[:, 1]] / prior, 1); y = a[:, 2]
        res.append(f"{s} {100*f1_score(y,p,average='macro'):.2f}/{100*np.mean(y==p):.2f}/{100*cohen_kappa_score(y,p):.2f}")
    print(dtype.__name__, kind, ' | '.join(res))
