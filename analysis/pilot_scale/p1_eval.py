"""Pilot 1 - step 2: warm / S1 / S2 evaluation of popularity proxies vs structure on one drug sample.

usage: python p1_eval.py SAMPLE    (SAMPLE in U1000, H300, U300, U300b, U300c; see p1_prep.py)
env:   MAXTR   max training rows per fold for every HGB model (same seeded subsample for all HGBs; default 100000)
       HEAVY   1 = run the ECFP-containing HGBs (default 1)
       EXTRA   1 = also run 'HGB ECFP+proxies+shared' (default 0)
       ECFP_PROD 1 = include fp_a*fp_b in the ECFP block (default 0: for tree models it is exactly 1[fp_a+fp_b==2],
                 i.e. redundant; dropped to halve HGB cost. Checked on 4 H300 warm folds, see RESULTS.md)
       MAXIT   HGB iterations (default 150, lr 0.1, 31 leaves, no early stopping)
Unit = unordered drug pair {a,b}, a != b; label 1 iff the pair is in DrugBank (either orientation);
negatives = every other pair inside the sample.
warm: KFold(5, shuffle, seed 0) over pairs.  S1/S2: KFold(5, shuffle, seed 0) over drugs;
train = pairs with both drugs in train drugs, S1 = exactly one held-out drug, S2 = both held out.
All label-derived quantities (degree prior, Vilar, SimKNN, identity LR, HGB) use TRAINING pairs only.
The '[diag]' external degree = # DrugBank partners OUTSIDE the sample: uses no within-sample label, but is
DrugBank knowledge that a genuinely new drug would not have -> diagnostic of popularity, not a method.
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/out'
NAME = sys.argv[1]
MAXTR = int(os.environ.get('MAXTR', 100000))
HEAVY = os.environ.get('HEAVY', '1') == '1'
EXTRA = os.environ.get('EXTRA', '0') == '1'
ECFP_PROD = os.environ.get('ECFP_PROD', '0') == '1'   # fp_a*fp_b == 1[fp_a+fp_b == 2]: exactly redundant for trees
MAXIT = int(os.environ.get('MAXIT', 150))

z = np.load(f'{OUT}/prep.npz')
idx = z['S_' + NAME]; n = len(idx)
F = z['F'][idx].astype(np.float32)
Dsc = z['D'][idx]
TC = z['TC'][idx].astype(float)
npool = len(z['pool'])
I = sp.csr_matrix((np.ones(len(z['ti_r'])), (z['ti_r'], z['ti_c'])), shape=(npool, int(z['n_prot'])))[idx]
SH = (I @ I.T).toarray()
JA = SH / np.maximum(TC[:, None] + TC[None, :] - SH, 1)
inter = F @ F.T; c = F.sum(1)
T = inter / np.maximum(c[:, None] + c[None, :] - inter, 1)
np.fill_diagonal(T, 1.0)

# labels + external degree
Pg = np.load(f'{OUT}/unordered_pairs_global.npy')
gidx = z['pool_gidx'][idx]
posmap = -np.ones(len(z['all_ddi']), dtype=int); posmap[gidx] = np.arange(n)
pa, pb = posmap[Pg[:, 0]], posmap[Pg[:, 1]]
Y = np.zeros((n, n), dtype=np.int8)
ins = (pa >= 0) & (pb >= 0)
Y[pa[ins], pb[ins]] = 1; Y[pb[ins], pa[ins]] = 1
one = (pa >= 0) ^ (pb >= 0)
ext = np.bincount(np.where(pa[one] >= 0, pa[one], pb[one]), minlength=n).astype(float)

A, B = np.triu_indices(n, 1)
y = Y[A, B].astype(int)
npair = len(y)
print(f'== {NAME}: drugs={n} pairs={npair} prevalence={y.mean():.4f} MAXTR={MAXTR}', flush=True)

# drug-level external-degree quintile (diagnostic strata)
qb = np.searchsorted(np.quantile(ext, [0.2, 0.4, 0.6, 0.8]), ext, side='right')
strat = np.minimum(qb[A], qb[B]) * 5 + np.maximum(qb[A], qb[B])
ext_prod = ext[A] * ext[B]


def blocks(names, ii):
    a, b = A[ii], B[ii]
    out = []
    for nm in names:
        if nm == 'ecfp':
            fa, fb = F[a], F[b]
            out += ([fa + fb, fa * fb] if ECFP_PROD else [fa + fb]) + [T[a, b][:, None]]
        elif nm == 'tc':
            out += [np.minimum(TC[a], TC[b])[:, None], np.maximum(TC[a], TC[b])[:, None]]
        elif nm == 'desc':
            out += [np.minimum(Dsc[a], Dsc[b]), np.maximum(Dsc[a], Dsc[b])]
        elif nm.startswith('d'):  # single descriptor dK
            k = int(nm[1:])
            out += [np.minimum(Dsc[a, k], Dsc[b, k])[:, None], np.maximum(Dsc[a, k], Dsc[b, k])[:, None]]
        elif nm == 'sh':
            out += [SH[a, b][:, None], JA[a, b][:, None]]
    return np.hstack(out).astype(np.float32)


HGB_MODELS = {
    'P: target count (HGB)': ['tc'],
    'P: MW (HGB)': ['d0'], 'P: logP (HGB)': ['d1'], 'P: rot. bonds (HGB)': ['d2'], 'P: ring count (HGB)': ['d3'],
    'P: 4 descriptors (HGB)': ['desc'],
    'P: proxies = tc+4 desc (HGB)': ['tc', 'desc'],
    'T: shared targets (HGB)': ['sh'],
    'T: tc + shared targets (HGB)': ['tc', 'sh'],
    'T: proxies + shared targets (HGB)': ['tc', 'desc', 'sh'],
}
if HEAVY:
    HGB_MODELS.update({'S: ECFP4 (HGB)': ['ecfp'], 'S: ECFP4 + tc (HGB)': ['ecfp', 'tc'],
                       'S: ECFP4 + proxies (HGB)': ['ecfp', 'tc', 'desc']})
    if EXTRA:
        HGB_MODELS['S: ECFP4 + proxies + shared (HGB)'] = ['ecfp', 'tc', 'desc', 'sh']


def hgb():
    return HistGradientBoostingClassifier(max_iter=MAXIT, learning_rate=0.1, max_leaf_nodes=31,
                                          early_stopping=False, random_state=0)


def predict_chunks(m, names, ii, chunk=40000):
    return np.concatenate([m.predict_proba(blocks(names, ii[s:s + chunk]))[:, 1] for s in range(0, len(ii), chunk)])


def strat_auc(yt, s, st):
    """pos-neg-pair-weighted mean of within-stratum AUROCs (strata = unordered pair of ext-degree quintiles)."""
    num = den = 0.0
    for k in np.unique(st):
        m = st == k; yk = yt[m]; npos = yk.sum(); nneg = len(yk) - npos
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg; num += w * roc_auc_score(yk, s[m]); den += w
    return num / den if den > 0 else np.nan


def evaluate(tr, tests, fold, warm):
    out = {}; t0 = time.time(); timing = {}
    ytr = y[tr]
    scores = {k: {} for k in tests}

    def add(name, svals):
        for (k, te), s in zip(tests.items(), svals):
            scores[k][name] = s

    # ---- (a) degree prior from training labels
    cnt = np.bincount(A[tr], minlength=n) + np.bincount(B[tr], minlength=n)
    pc = np.bincount(A[tr], weights=ytr, minlength=n) + np.bincount(B[tr], weights=ytr, minlength=n)
    r = np.where(cnt > 0, pc / np.maximum(cnt, 1), ytr.mean())
    add('B: degree prior, train labels (product)', [r[A[te]] * r[B[te]] for te in tests.values()])
    add('B: degree prior, train labels (sum)', [r[A[te]] + r[B[te]] for te in tests.values()])
    # ---- raw unsupervised pair scores
    add('U: Tanimoto ECFP4 (raw)', [T[A[te], B[te]] for te in tests.values()])
    add('U: target-count product (raw)', [(TC[A[te]] + 1) * (TC[B[te]] + 1) for te in tests.values()])
    add('U: # shared targets (raw)', [SH[A[te], B[te]] for te in tests.values()])
    add('U: Jaccard of target sets (raw)', [JA[A[te], B[te]] for te in tests.values()])
    add('[diag] external DrugBank degree product', [ext_prod[te] for te in tests.values()])
    # ---- (c) Vilar-style propagation and SimKNN (training labels only)
    Ptr = np.zeros((n, n), dtype=np.float32)
    trp = tr[ytr == 1]
    Ptr[A[trp], B[trp]] = 1; Ptr[B[trp], A[trp]] = 1
    Vmax = np.zeros((n, n), dtype=np.float32)
    for a in range(n):
        Na = np.flatnonzero(Ptr[a])
        if len(Na):
            Vmax[a] = T[:, Na].max(1)
    Ssum = Ptr @ T.astype(np.float32); dg = Ptr.sum(1)
    add('N: Vilar max-Tanimoto to partners', [np.maximum(Vmax[A[te], B[te]], Vmax[B[te], A[te]]) for te in tests.values()])
    add('N: Vilar mean-Tanimoto to partners',
        [(Ssum[A[te], B[te]] + Ssum[B[te], A[te]]) / np.maximum(dg[A[te]] + dg[B[te]], 1) for te in tests.values()])
    Ytr = np.zeros((n, n)); Mtr = np.zeros((n, n))
    Ytr[A[tr], B[tr]] = ytr; Ytr[B[tr], A[tr]] = ytr; Mtr[A[tr], B[tr]] = 1; Mtr[B[tr], A[tr]] = 1
    known = cnt > 0
    W = np.zeros((n, n))
    for a in range(n):
        cand = np.flatnonzero(known & (np.arange(n) != a))
        nb = cand[np.argsort(-T[a, cand], kind='stable')[:10]]
        W[a, nb] = T[a, nb]
        if known[a]:
            W[a, a] = 1.0
    num = W @ Ytr @ W.T; den = W @ Mtr @ W.T
    K = np.where(den > 0, num / np.maximum(den, 1e-12), ytr.mean())
    add('N: SimKNN k=10 (2-sided)', [K[A[te], B[te]] for te in tests.values()])
    timing['unsup'] = time.time() - t0
    # ---- (f) identity one-hot LR (warm only)
    if warm:
        t1 = time.time()
        rows = np.repeat(np.arange(len(tr)), 2)
        Xid = sp.csr_matrix((np.ones(2 * len(tr)), (rows, np.stack([A[tr], B[tr]], 1).ravel())), shape=(len(tr), n))
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xid, ytr)
        wv = lr.coef_.ravel()
        add('B: identity one-hot LR', [lr.intercept_[0] + wv[A[te]] + wv[B[te]] for te in tests.values()])
        timing['idLR'] = time.time() - t1
    # ---- (b,d,e) HGB models on one shared seeded training subsample
    rs = np.random.RandomState(1000 + fold + (0 if warm else 100))
    sub = np.sort(rs.choice(tr, MAXTR, replace=False)) if len(tr) > MAXTR else tr
    for mname, bl in HGB_MODELS.items():
        t1 = time.time()
        m = hgb().fit(blocks(bl, sub), y[sub])
        add(mname, [predict_chunks(m, bl, te) for te in tests.values()])
        timing[mname] = time.time() - t1
    # ---- metrics
    for k, te in tests.items():
        yt = y[te]; ref_p = scores[k].get('P: proxies = tc+4 desc (HGB)'); ref_e = ext_prod[te]
        for mname, s in scores[k].items():
            d = dict(AUROC=roc_auc_score(yt, s), AP=average_precision_score(yt, s),
                     sAUROC=strat_auc(yt, s, strat[te]))
            if np.std(s) > 0:
                d['rho_ext'] = spearmanr(s, ref_e)[0]
                if ref_p is not None:
                    d['rho_proxy'] = spearmanr(s, ref_p)[0]
            out.setdefault(mname, {})[k] = d
    out['_sizes'] = {k: {'n': int(len(te)), 'prev': float(y[te].mean())} for k, te in tests.items()}
    out['_sizes']['train'] = {'n': int(len(tr)), 'prev': float(ytr.mean()), 'hgb_rows': int(len(sub))}
    out['_timing'] = timing
    return out


res = {'sample': NAME, 'n': n, 'pairs': int(npair), 'prevalence': float(y.mean()), 'MAXTR': MAXTR, 'MAXIT': MAXIT,
       'ECFP_PROD': ECFP_PROD, 'EXTRA': EXTRA,
       'drug_stats': {'median_ext_degree': float(np.median(ext)), 'median_targets': float(np.median(TC)),
                      'frac_zero_targets': float((TC == 0).mean()),
                      'median_sample_degree': float(np.median(Y.sum(1)))},
       'warm': [], 'node': []}
fn = f'{OUT}/eval_{NAME}' + os.environ.get('OUTSUFFIX', '') + '.json'
NODE_FOLDS = [int(x) for x in os.environ.get('NODE_FOLDS', '0,1,2,3,4').split(',')]  # used to split a slow sample over processes
for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(npair))):
    if os.environ.get('SKIP_WARM') == '1':
        break
    t0 = time.time()
    res['warm'].append(evaluate(tr, {'warm': te}, f, True))
    print(f'  warm fold {f} {time.time() - t0:.0f}s  ' +
          ' '.join(f'{k[:14]}={v:.0f}' for k, v in res['warm'][-1]['_timing'].items() if v > 2), flush=True)
    json.dump(res, open(fn, 'w'), indent=1, default=float)
for f, (trn, ten) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(n))):
    if f not in NODE_FOLDS:
        continue
    t0 = time.time()
    intr = np.zeros(n, bool); intr[trn] = True
    ai, bi = intr[A], intr[B]
    tr = np.flatnonzero(ai & bi); s1 = np.flatnonzero(ai ^ bi); s2 = np.flatnonzero(~ai & ~bi)
    res['node'].append(evaluate(tr, {'S1': s1, 'S2': s2}, f, False)); res['node'][-1]['_fold'] = f
    print(f'  node fold {f} {time.time() - t0:.0f}s sizes {res["node"][-1]["_sizes"]}', flush=True)
    json.dump(res, open(fn, 'w'), indent=1, default=float)

for split in (['S1', 'S2'] if os.environ.get('SKIP_WARM') == '1' else ['warm', 'S1', 'S2']):
    src = res['warm'] if split == 'warm' else res['node']
    print(f'-- {NAME} {split} prev {np.mean([fr["_sizes"][split]["prev"] for fr in src]):.3f}')
    for mth in src[0]:
        if mth.startswith('_'):
            continue
        au = np.array([fr[mth][split]['AUROC'] for fr in src]); ap = np.array([fr[mth][split]['AP'] for fr in src])
        sa = np.array([fr[mth][split]['sAUROC'] for fr in src])
        print(f'   {mth:42s} AUROC {au.mean():.3f}±{au.std(ddof=1):.3f}  AP {ap.mean():.3f}±{ap.std(ddof=1):.3f}  sAUROC {sa.mean():.3f}')
