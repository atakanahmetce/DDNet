"""Task 3: leakage-free protocols.

Unit of prediction = UNORDERED drug pair {a,b}, a != b (no self pairs); label = OR of both directions.
Splits (5 folds each, seed 0):
  warm : random 5-fold over unordered pairs (a mirrored pair cannot straddle folds - it is one row)
  S1   : node-level 5-fold; train = pairs with both drugs in train-drugs;
         test = pairs with EXACTLY one drug from the held-out drug fold
  S2   : same node folds / same trained model; test = pairs with BOTH drugs held out
Feature sets / methods:
  DDNet-sym  [e_a+e_b, e_a*e_b, |e_a-e_b|, path1..3]  -> RF(100,balanced), HGB, FFNN-fixed
  RandomVec-sym + path (control: node2vec vectors replaced by i.i.d. Gaussian vectors) -> HGB, FFNN-fixed
  Identity MLP: FFNN-fixed on multi-hot identity (can do matrix completion, unlike additive LR)
  DDNet-concat (paper layout [e_a,e_b,path]) trained on both orientations of each TRAIN pair,
               test score = mean over both orientations -> HGB
  Emb-sym only (HGB), Path only (HGB)
  Morgan ECFP4 1024 bits from drugs.json SMILES (RDKit): [fp_a+fp_b (0/1/2), Tanimoto] -> HGB
  Identity (multi-hot onehot(a)+onehot(b)) -> LR, RF
  Degree (train-only per-drug positive rate; product / sum), raw similarity S,
  SimKNN (two-sided similarity-weighted neighbour average of TRAIN labels, k=10)
  DrugBank-global-degree product: EXTERNAL / leaky diagnostic (uses DDI counts from outside the dataset)
usage: python s03_proper.py DS [hop]      env RF_JOBS, TORCH_THREADS, SYM_RF (0 = skip RF), WARM_FOLDS (<5 = first k warm folds)
"""
import os, sys, json, time
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp
import torch
from common import *
import nets

RF_JOBS = int(os.environ.get('RF_JOBS', 1))
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 1)))


def hgb():
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, early_stopping=False, random_state=0)


def rf():
    return RandomForestClassifier(n_estimators=100, n_jobs=RF_JOBS, class_weight='balanced', random_state=0)


def skl(model, Xtr, ytr, Xtests):
    with threadpool_limits(1):
        model.fit(Xtr, ytr)
        return [model.predict_proba(X)[:, 1] for X in Xtests]


def run(ds, hop=1):
    nm = names(ds); n = len(nm)
    ids, E = load_emb(emb_file(ds, hop))
    Em = np.array([E[d] for d in nm])
    P = load_paths(ds, nm)
    Y = label_matrix(ds, nm, symmetric=True)
    S = sim(ds); Ss = (S + S.T) / 2
    F, bad = morgan_fp(nm)
    Ff = F.astype(np.float32)
    inter = Ff @ Ff.T; c = Ff.sum(1); T = inter / np.maximum(c[:, None] + c[None, :] - inter, 1)
    db = drugbank_pairs()
    import pandas as pd
    und = pd.DataFrame(np.sort(db.values, axis=1), columns=['a', 'b']).drop_duplicates()
    und = und[und.a != und.b]
    gdeg = pd.concat([und.a, und.b]).value_counts()
    g = np.array([gdeg.get(d, 0) for d in nm], dtype=float)

    A, B = np.triu_indices(n, 1)
    y = Y[A, B].astype(int)
    npair = len(y)
    EA, EB = Em[A], Em[B]
    path = P[A, B]
    X_sym = np.hstack([EA + EB, EA * EB, np.abs(EA - EB), path]).astype(np.float32)
    X_emb = X_sym[:, :-3]
    # control: i.i.d. Gaussian 128-D 'embedding' per drug (pure identity code, no graph information)
    Rm = np.random.RandomState(123).normal(size=Em.shape) * Em.std()
    RA, RB = Rm[A], Rm[B]
    X_rnd = np.hstack([RA + RB, RA * RB, np.abs(RA - RB), path]).astype(np.float32)
    X_cat_ab = np.hstack([EA, EB, path]).astype(np.float32)
    X_cat_ba = np.hstack([EB, EA, path]).astype(np.float32)
    X_fp = np.hstack([F[A] + F[B], T[A, B][:, None]]).astype(np.float32)
    rows = np.repeat(np.arange(npair), 2); cols = np.stack([A, B], 1).ravel()
    X_id = sp.csr_matrix((np.ones(2 * npair, dtype=np.float32), (rows, cols)), shape=(npair, n))
    print(f'== {ds} hop{hop}: drugs={n} pairs={npair} pos={y.mean():.3f} fp_fail={bad}', flush=True)

    def evaluate(tr, tests):
        """tr: train pair idx; tests: dict name->pair idx. Returns {method: {split: metrics}}"""
        out = {}
        ytr = y[tr]
        def add(name, str_, ste_list, thr=None, prob=True):
            if thr is None:
                thr = 0.5 if prob else best_threshold(ytr, str_)
            out[name] = {k: all_metrics(y[te], (s >= thr).astype(int) if prob else (s > thr).astype(int), s)
                         for (k, te), s in zip(tests.items(), ste_list)}
        tl = list(tests.values())
        # --- learned models
        if os.environ.get('SYM_RF', '1') == '1':
            add('DDNet-sym RF', None, skl(rf(), X_sym[tr], ytr, [X_sym[te] for te in tl]))
        add('DDNet-sym HGB', None, skl(hgb(), X_sym[tr], ytr, [X_sym[te] for te in tl]))
        sc = StandardScaler().fit(X_sym[tr])
        m = nets.train_fixed(sc.transform(X_sym[tr]), ytr, seed=0)
        add('DDNet-sym FFNN-fixed', None, [nets.predict_fixed(m, sc.transform(X_sym[te]))[1] for te in tl])
        with threadpool_limits(1):
            mh = hgb().fit(np.vstack([X_cat_ab[tr], X_cat_ba[tr]]), np.concatenate([ytr, ytr]))
            s_list = [(mh.predict_proba(X_cat_ab[te])[:, 1] + mh.predict_proba(X_cat_ba[te])[:, 1]) / 2 for te in tl]
        add('DDNet-concat both-orient. HGB', None, s_list)
        add('RandomVec-sym + path HGB (control)', None, skl(hgb(), X_rnd[tr], ytr, [X_rnd[te] for te in tl]))
        sc = StandardScaler().fit(X_rnd[tr])
        m = nets.train_fixed(sc.transform(X_rnd[tr]), ytr, seed=0)
        add('RandomVec-sym + path FFNN-fixed (control)', None, [nets.predict_fixed(m, sc.transform(X_rnd[te]))[1] for te in tl])
        m = nets.train_fixed(X_id[tr].toarray(), ytr, seed=0)
        add('Identity MLP (FFNN-fixed on multi-hot)', None, [nets.predict_fixed(m, X_id[te].toarray())[1] for te in tl])
        add('Emb-sym only HGB', None, skl(hgb(), X_emb[tr], ytr, [X_emb[te] for te in tl]))
        add('Path only HGB', None, skl(hgb(), path[tr], ytr, [path[te] for te in tl]))
        add('Morgan FP HGB', None, skl(hgb(), X_fp[tr], ytr, [X_fp[te] for te in tl]))
        add('Identity LR', None, skl(LogisticRegression(C=1.0, max_iter=3000), X_id[tr], ytr, [X_id[te] for te in tl]))
        if os.environ.get('ID_RF', '0') == '1':
            add('Identity RF', None, skl(rf(), X_id[tr], ytr, [X_id[te] for te in tl]))
        # --- score baselines (threshold = max-MCC on training pairs)
        cnt = np.bincount(A[tr], minlength=n) + np.bincount(B[tr], minlength=n)
        pc = np.bincount(A[tr], weights=ytr, minlength=n) + np.bincount(B[tr], weights=ytr, minlength=n)
        r = np.where(cnt > 0, pc / np.maximum(cnt, 1), ytr.mean())
        add('Degree (product)', r[A[tr]] * r[B[tr]], [r[A[te]] * r[B[te]] for te in tl], prob=False)
        add('Degree (sum)', r[A[tr]] + r[B[tr]], [r[A[te]] + r[B[te]] for te in tl], prob=False)
        add('Raw similarity S', Ss[A[tr], B[tr]], [Ss[A[te], B[te]] for te in tl], prob=False)
        # SimKNN: two-sided neighbour smoothing of the TRAINING label matrix
        Ytr = np.zeros((n, n)); Mtr = np.zeros((n, n))
        Ytr[A[tr], B[tr]] = ytr; Ytr[B[tr], A[tr]] = ytr; Mtr[A[tr], B[tr]] = 1; Mtr[B[tr], A[tr]] = 1
        known = cnt > 0
        W = np.zeros((n, n))
        for a in range(n):
            cand = np.where(known & (np.arange(n) != a))[0]
            nb = cand[np.argsort(-Ss[a, cand])[:10]]
            W[a, nb] = Ss[a, nb]
            if known[a]:
                W[a, a] = max(Ss[a, a], Ss[a].max())
        num = W @ Ytr @ W.T; den = W @ Mtr @ W.T
        K = np.where(den > 0, num / np.maximum(den, 1e-12), ytr.mean())
        add('SimKNN (k=10)', K[A[tr], B[tr]], [K[A[te], B[te]] for te in tl], prob=False)
        add('[leaky] DrugBank global degree product', g[A[tr]] * g[B[tr]], [g[A[te]] * g[B[te]] for te in tl], prob=False)
        return out

    res = {'ds': ds, 'hop': hop, 'n': n, 'pairs': int(npair), 'warm': [], 'node': []}
    for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(npair))):
        if f >= int(os.environ.get('WARM_FOLDS', 5)):
            break
        t0 = time.time()
        res['warm'].append(evaluate(tr, {'warm': te}))
        print(f'  warm fold {f} {time.time() - t0:.0f}s', flush=True)
        json.dump(res, open(f'{OUT}/out/proper_{ds}_hop{hop}.json', 'w'), indent=1, default=float)
    for f, (trn, ten) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(n))):
        t0 = time.time()
        intr = np.zeros(n, bool); intr[trn] = True
        a_in, b_in = intr[A], intr[B]
        tr = np.where(a_in & b_in)[0]
        s1 = np.where(a_in ^ b_in)[0]
        s2 = np.where(~a_in & ~b_in)[0]
        r = evaluate(tr, {'S1': s1, 'S2': s2})
        r['_sizes'] = {'train': int(len(tr)), 'S1': int(len(s1)), 'S2': int(len(s2)),
                       'S1_pos': float(y[s1].mean()), 'S2_pos': float(y[s2].mean())}
        res['node'].append(r)
        print(f'  node fold {f} {time.time() - t0:.0f}s sizes {r["_sizes"]}', flush=True)
        json.dump(res, open(f'{OUT}/out/proper_{ds}_hop{hop}.json', 'w'), indent=1, default=float)
    # quick summary
    for split in ['warm', 'S1', 'S2']:
        src = res['warm'] if split == 'warm' else res['node']
        print(f'-- {ds} {split}')
        for mth in src[0]:
            if mth.startswith('_'):
                continue
            v = [fr[mth][split] for fr in src]
            au = np.array([x['AUROC'] for x in v]); mc = np.array([x['MCC'] for x in v])
            ap = np.array([x['AUPRC'] for x in v]); f1 = np.array([x['F1'] for x in v])
            print(f'   {mth:42s} AUROC {au.mean():.3f}±{au.std(ddof=1):.3f}  AUPRC {ap.mean():.3f}  '
                  f'F1 {f1.mean():.3f}  MCC {mc.mean():.3f}±{mc.std(ddof=1):.3f}')
    return res


if __name__ == '__main__':
    ds = sys.argv[1]
    hop = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run(ds, hop)
