"""Pilot 1 - diagnostic: does structure (ECFP4) or SimKNN add anything BEYOND the strongest popularity signal?
Popularity signal = out-of-sample DrugBank degree of each drug ([diag]: not available for a genuinely new drug).
Node (S1/S2) folds only, same folds/subsample seeds as p1_eval.py.   usage: python p1_beyond_pop.py SAMPLE
"""
import os, sys, json, time
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/out'
NAME = sys.argv[1]; MAXTR = int(os.environ.get('MAXTR', 50000))
z = np.load(f'{OUT}/prep.npz'); idx = z['S_' + NAME]; n = len(idx)
F = z['F'][idx].astype(np.float32); Dsc = z['D'][idx]; TC = z['TC'][idx].astype(float)
inter = F @ F.T; c = F.sum(1); T = inter / np.maximum(c[:, None] + c[None, :] - inter, 1); np.fill_diagonal(T, 1.0)
Pg = np.load(f'{OUT}/unordered_pairs_global.npy'); gidx = z['pool_gidx'][idx]
posmap = -np.ones(len(z['all_ddi']), dtype=int); posmap[gidx] = np.arange(n)
pa, pb = posmap[Pg[:, 0]], posmap[Pg[:, 1]]
Y = np.zeros((n, n), dtype=np.int8); ins = (pa >= 0) & (pb >= 0); Y[pa[ins], pb[ins]] = 1; Y[pb[ins], pa[ins]] = 1
one = (pa >= 0) ^ (pb >= 0); ext = np.bincount(np.where(pa[one] >= 0, pa[one], pb[one]), minlength=n).astype(float)
A, B = np.triu_indices(n, 1); y = Y[A, B].astype(int)
lext = np.log1p(ext)


def blocks(names, ii):
    a, b = A[ii], B[ii]; out = []
    for nm in names:
        if nm == 'ext': out += [np.minimum(lext[a], lext[b])[:, None], np.maximum(lext[a], lext[b])[:, None]]
        if nm == 'ecfp': out += [F[a] + F[b], T[a, b][:, None]]
        if nm == 'prox': out += [np.minimum(TC[a], TC[b])[:, None], np.maximum(TC[a], TC[b])[:, None],
                                 np.minimum(Dsc[a], Dsc[b]), np.maximum(Dsc[a], Dsc[b])]
    return np.hstack(out).astype(np.float32)


MODELS = {'ext-degree (HGB)': ['ext'], 'ext-degree + proxies (HGB)': ['ext', 'prox'],
          'ext-degree + ECFP4 (HGB)': ['ext', 'ecfp']}
res = {'sample': NAME, 'node': []}
for f, (trn, ten) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(n))):
    t0 = time.time()
    intr = np.zeros(n, bool); intr[trn] = True; ai, bi = intr[A], intr[B]
    tr = np.flatnonzero(ai & bi); tests = {'S1': np.flatnonzero(ai ^ bi), 'S2': np.flatnonzero(~ai & ~bi)}
    ytr = y[tr]
    rs = np.random.RandomState(1000 + f + 100)
    sub = np.sort(rs.choice(tr, MAXTR, replace=False)) if len(tr) > MAXTR else tr
    sc = {k: {} for k in tests}
    for m, bl in MODELS.items():
        h = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False,
                                           random_state=0).fit(blocks(bl, sub), y[sub])
        for k, te in tests.items():
            sc[k][m] = np.concatenate([h.predict_proba(blocks(bl, te[s:s + 40000]))[:, 1] for s in range(0, len(te), 40000)])
    # SimKNN (same definition as p1_eval.py) and a fit-free rank average with the external degree product
    Ytr = np.zeros((n, n)); Mtr = np.zeros((n, n))
    Ytr[A[tr], B[tr]] = ytr; Ytr[B[tr], A[tr]] = ytr; Mtr[A[tr], B[tr]] = 1; Mtr[B[tr], A[tr]] = 1
    cnt = np.bincount(A[tr], minlength=n) + np.bincount(B[tr], minlength=n); known = cnt > 0
    W = np.zeros((n, n))
    for a in range(n):
        cand = np.flatnonzero(known & (np.arange(n) != a)); nb = cand[np.argsort(-T[a, cand], kind='stable')[:10]]
        W[a, nb] = T[a, nb]
        if known[a]: W[a, a] = 1.0
    num = W @ Ytr @ W.T; den = W @ Mtr @ W.T; K = np.where(den > 0, num / np.maximum(den, 1e-12), ytr.mean())
    out = {}
    for k, te in tests.items():
        ep = ext[A[te]] * ext[B[te]]; kn = K[A[te], B[te]]
        sc[k]['ext-degree product (raw)'] = ep
        sc[k]['rank-avg(ext-degree product, SimKNN)'] = rankdata(ep) + rankdata(kn)
        for m, s in sc[k].items():
            out.setdefault(m, {})[k] = {'AUROC': roc_auc_score(y[te], s), 'AP': average_precision_score(y[te], s)}
    res['node'].append(out)
    json.dump(res, open(f'{OUT}/beyond_pop_{NAME}.json', 'w'), indent=1, default=float)
    print(f'  node fold {f} {time.time() - t0:.0f}s', flush=True)
for sp in ['S1', 'S2']:
    print('--', NAME, sp)
    for m in res['node'][0]:
        au = np.array([fr[m][sp]['AUROC'] for fr in res['node']]); ap = np.array([fr[m][sp]['AP'] for fr in res['node']])
        print(f'   {m:40s} AUROC {au.mean():.3f}±{au.std(ddof=1):.3f}  AP {ap.mean():.3f}±{ap.std(ddof=1):.3f}')
