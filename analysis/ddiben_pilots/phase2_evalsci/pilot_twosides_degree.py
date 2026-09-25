"""Pilot: degree/popularity main-effects on DDI-Ben TWOSIDES (multi-label, pos + corrupted-neg rows).
Score for label j on pair (a,b) = g_j(a) + g_j(b), g = log(1 + #train positive rows of the drug carrying j)
(label-specific) or overall degree; for drugs unseen in training, g is the Tanimoto-kNN mean over train drugs (k=10)
('kNN') or 0 ('zero'). Metric exactly as DDI-Ben trainer.py (mean over labels of ROC-AUC / PR-AUC)."""
import sys, pickle, json, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT, SPLIT = sys.argv[1], sys.argv[2]
d = f'{ROOT}/data/{SPLIT}/'
def load(fn):
    H, T, L, P = [], [], [], []
    for line in open(d + fn):
        h, t, r, p = line.strip().split(' ')
        H.append(int(h)); T.append(int(t)); L.append(np.array(r.split(','), dtype=np.int8)); P.append(int(p))
    return np.array(H), np.array(T), np.stack(L), np.array(P)
h, t, L, P = load('train.txt')
train_set = np.array(sorted(set(np.loadtxt(d + 'train_set.txt', dtype=int).tolist())))
x = pickle.load(open(f'{ROOT}/data/initial/twosides/DB_molecular_feats.pkl', 'rb'))
F = np.array(x if not isinstance(x, dict) else list(x.values())[0], dtype=np.float32)
F = (F > 0).astype(np.float32)
inter = F @ F.T; c = F.sum(1); Tm = inter / np.maximum(c[:, None] + c[None, :] - inter, 1)
N, R = F.shape[0], L.shape[1] - 0
pos = P == 1
C = np.zeros((N, R)); np.add.at(C, h[pos], L[pos]); np.add.at(C, t[pos], L[pos])
deg = C.sum(1)
known = np.zeros(N, bool); known[train_set] = True
def knn(v):
    out = v.copy()
    for i in np.flatnonzero(~known):
        s = Tm[i, train_set]; nb = train_set[np.argsort(-s)[:10]]; w = Tm[i, nb]
        out[i] = (w[:, None] * v[nb]).sum(0) / max(w.sum(), 1e-9) if v.ndim == 2 else (w * v[nb]).sum() / max(w.sum(), 1e-9)
    return out
G_lab = {'zero': np.log1p(C * known[:, None]), 'kNN': np.log1p(knn(C))}
G_deg = {'zero': np.log1p(deg * known), 'kNN': np.log1p(knn(deg))}
res = {}
for S in ['S0', 'S1', 'S2']:
    th, tt, tL, tP = load(f'test_{S}.txt')
    # composition of negatives: how many known drugs per row
    nk = known[th].astype(int) + known[tt].astype(int)
    comp = {f'pos_known{k}': int(((tP == 1) & (nk == k)).sum()) for k in range(3)}
    comp.update({f'neg_known{k}': int(((tP == 0) & (nk == k)).sum()) for k in range(3)})
    r = {'composition': comp}
    for name, sc in [('label-degree zero', lambda: G_lab['zero'][th] + G_lab['zero'][tt]),
                     ('label-degree kNN', lambda: G_lab['kNN'][th] + G_lab['kNN'][tt]),
                     ('overall-degree zero', lambda: np.repeat((G_deg['zero'][th] + G_deg['zero'][tt])[:, None], R, 1)),
                     ('overall-degree kNN', lambda: np.repeat((G_deg['kNN'][th] + G_deg['kNN'][tt])[:, None], R, 1))]:
        s = sc(); aucs, aps = [], []
        for j in range(R):
            w = np.flatnonzero(tL[:, j] == 1)
            if len(w) == 0: continue
            y = tP[w]
            if y.min() == y.max(): continue
            aucs.append(roc_auc_score(y, s[w, j])); aps.append(average_precision_score(y, s[w, j]))
        r[name] = dict(ROC_AUC=round(100 * np.mean(aucs), 1), PR_AUC=round(100 * np.mean(aps), 1), n_labels=len(aucs))
    res[S] = r; print(SPLIT, S, r, flush=True)
json.dump(res, open(f'out/phase2/evalsci/degree_{SPLIT}.json', 'w'), indent=1)
