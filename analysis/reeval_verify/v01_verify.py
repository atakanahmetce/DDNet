"""Independent re-implementation (does NOT import anything from ../reeval) of the decision-relevant
DDNet comparisons.

usage: python v01_verify.py DS PROTO [NFOLDS_TO_RUN]
  DS    : D1 | MP
  PROTO : paper    -> (d) ordered pairs incl. self pairs, [e_src,e_tgt,path], row-L2 normalised,
                      random 5-fold over ORDERED rows (mirror pairs can straddle folds) + a
                      "mirror-grouped" variant where (a,b),(b,a) always share a fold (isolates mirror leakage)
          proper   -> (a) warm 5-fold over UNORDERED pairs (i<j) and drug-disjoint node 5-fold (S1/S2)
Seeds deliberately differ from the original re-evaluation (they used KFold seed 0): here 2026.
Methods: DDNet features + RandomForest(100, balanced); degree from TRAINING labels; one-hot identity + RF;
         (proper only) external DrugBank degree: full (leaky) and restricted to partners OUTSIDE the dataset
         (cannot contain any in-dataset label).
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, auc)

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f'{HERE}/out', exist_ok=True)
SEED = 2026
NJOBS = int(os.environ.get('NJOBS', 4))

SPEC = {
    'D1': dict(d='datasets/data_1', emb='n2v_embeddings/cb1_embeddings/cb1_hop1_128_1.2_1.2.txt',
               paths='meta_paths/cb1_3_paths.txt'),
    'MP': dict(d='datasets/data_mp', emb='n2v_embeddings/mp_embeddings/mp_128_1.2_1.2.txt',
               paths='meta_paths/mp_3_paths.txt'),
}


def load(ds):
    sp = SPEC[ds]
    drugs = [x.strip() for x in open(f"{ROOT}/{sp['d']}/names.txt") if x.strip()]
    ix = {d: i for i, d in enumerate(drugs)}
    n = len(drugs)
    E = np.full((n, 128), np.nan)
    for line in open(f"{ROOT}/{sp['emb']}"):
        k, v = line.rstrip('\n').split(' ', 1)
        E[ix[k]] = np.array(v.split(','), dtype=float)
    assert not np.isnan(E).any()
    P = np.full((n, n, 3), np.nan)
    for line in open(f"{ROOT}/{sp['paths']}"):
        parts = line.rstrip('\n').split(',')
        P[ix[parts[0]], ix[parts[1]]] = np.array(parts[2].split(' '), dtype=float)
    assert not np.isnan(P).any()
    Yd = np.zeros((n, n), dtype=np.int8)
    for line in open(f"{ROOT}/{sp['d']}/interactions.txt"):
        a, b = line.strip().split(',')[:2]
        if a in ix and b in ix:
            Yd[ix[a], ix[b]] = 1
    return drugs, E, P, Yd


def hard_auprc(y, pred):
    """paper way: sklearn auc(recall, precision) of the PR curve computed from HARD labels."""
    pr, rc, _ = precision_recall_curve(y, pred)
    return auc(rc, pr)


def metr(y, score, pred):
    y = np.asarray(y); score = np.asarray(score, float); pred = np.asarray(pred).astype(int)
    one = len(np.unique(y)) > 1
    return dict(AUROC=roc_auc_score(y, score) if one else np.nan,
                AUPRC=average_precision_score(y, score) if one else np.nan,
                F1=f1_score(y, pred, zero_division=0), MCC=matthews_corrcoef(y, pred),
                AUPRC_hard=hard_auprc(y, pred), pos=float(y.mean()), n=int(len(y)))


def mcc_threshold(y, s):
    best, bt = -9, None
    for t in np.unique(np.quantile(s, np.linspace(0.005, 0.995, 199))):
        m = matthews_corrcoef(y, (s > t).astype(int))
        if m > best:
            best, bt = m, t
    return bt


def score_baseline(ytr, str_, ste, yte):
    t = mcc_threshold(ytr, str_)
    if t is None or np.unique(str_).size == 1:
        return metr(yte, ste, np.zeros_like(yte))
    return metr(yte, ste, (ste > t).astype(int))


def rf(seed):
    return RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=NJOBS, random_state=seed)


def per_drug_rate(a, b, ylab, n):
    """positive rate of each drug over the given (training) pairs; drugs with no pair get NaN."""
    cnt = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    pos = np.bincount(a, weights=ylab, minlength=n) + np.bincount(b, weights=ylab, minlength=n)
    r = np.full(n, np.nan)
    r[cnt > 0] = pos[cnt > 0] / cnt[cnt > 0]
    return r


def folds_of(m, k, rng):
    f = np.empty(m, dtype=int)
    f[rng.permutation(m)] = np.arange(m) % k
    return f


# ================================================================ (d) paper-style
def run_paper(ds, nrun):
    drugs, E, P, Yd = load(ds)
    n = len(drugs)
    src = np.repeat(np.arange(n), n); tgt = np.tile(np.arange(n), n)
    X = np.hstack([E[src], E[tgt], P[src, tgt]])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)          # sklearn Normalizer (row-wise L2)
    y = Yd[src, tgt].astype(int)
    OH = np.zeros((len(y), 2 * n), dtype=np.float32)
    OH[np.arange(len(y)), src] = 1; OH[np.arange(len(y)), n + tgt] = 1
    print(f'[paper {ds}] n={n} rows={len(y)} pos={y.mean():.4f} self-rows={int((src == tgt).sum())}', flush=True)
    rng = np.random.default_rng(SEED)
    fold_rows = folds_of(len(y), 5, rng)
    # mirror-grouped: key = unordered pair
    key = np.minimum(src, tgt) * n + np.maximum(src, tgt)
    uk, inv = np.unique(key, return_inverse=True)
    fold_grp = folds_of(len(uk), 5, rng)[inv]
    res = {'ds': ds, 'rows': int(len(y)), 'pos': float(y.mean()), 'random': [], 'mirror_grouped': []}
    for scheme, fv in [('random', fold_rows), ('mirror_grouped', fold_grp)]:
        for f in range(nrun):
            t0 = time.time()
            te = np.where(fv == f)[0]; tr = np.where(fv != f)[0]
            # sanity: how many test rows have their mirror in train
            mir = tgt[te] * n + src[te]
            in_train = np.zeros(n * n, bool); in_train[tr] = True   # row id == src*n+tgt
            frac_mirror_in_train = float(in_train[mir].mean())
            R = {'_mirror_in_train': frac_mirror_in_train, '_n_test': int(len(te))}
            m = rf(f).fit(X[tr], y[tr]); s = m.predict_proba(X[te])[:, 1]
            R['DDNet RF'] = metr(y[te], s, (s >= 0.5).astype(int))
            r = per_drug_rate(src[tr], tgt[tr], y[tr], n); r = np.where(np.isnan(r), y[tr].mean(), r)
            R['Degree product (train labels)'] = score_baseline(y[tr], r[src[tr]] * r[tgt[tr]], r[src[te]] * r[tgt[te]], y[te])
            m = rf(f).fit(OH[tr], y[tr]); s = m.predict_proba(OH[te])[:, 1]
            R['One-hot identity RF'] = metr(y[te], s, (s >= 0.5).astype(int))
            R['All-positive'] = metr(y[te], np.zeros(len(te)), np.ones(len(te)))
            res[scheme].append(R)
            print(f'  {scheme} fold {f} ({time.time() - t0:.0f}s, mirror-in-train {frac_mirror_in_train:.3f}): ' +
                  ' | '.join(f"{k}: AUROC {v['AUROC']:.3f} AP {v['AUPRC']:.3f} F1 {v['F1']:.3f} MCC {v['MCC']:.3f} hAP {v['AUPRC_hard']:.3f}"
                             for k, v in R.items() if not k.startswith('_')), flush=True)
            json.dump(res, open(f'{HERE}/out/paper_{ds}.json', 'w'), indent=1, default=float)
    return res


# ================================================================ (a) warm vs drug-disjoint
def drugbank_degrees(drugs):
    db = pd.read_csv(f'{ROOT}/drug-drug_interaction_Drugbank.csv')
    a = db.iloc[:, 0].astype(str).values; b = db.iloc[:, 1].astype(str).values
    lo = np.where(a < b, a, b); hi = np.where(a < b, b, a)
    und = pd.DataFrame({'lo': lo, 'hi': hi}).drop_duplicates()
    und = und[und.lo != und.hi]
    full = pd.concat([und.lo, und.hi]).value_counts()
    ds = set(drugs)
    # partners outside the dataset only
    out_mask_lo = ~und.hi.isin(ds)   # lo's partner is hi
    out_mask_hi = ~und.lo.isin(ds)
    outside = pd.concat([und.lo[out_mask_lo], und.hi[out_mask_hi]]).value_counts()
    g_full = np.array([full.get(d, 0) for d in drugs], float)
    g_out = np.array([outside.get(d, 0) for d in drugs], float)
    return g_full, g_out


def run_proper(ds, nrun):
    drugs, E, P, Yd = load(ds)
    n = len(drugs)
    A, B = np.triu_indices(n, 1)
    y = ((Yd[A, B] + Yd[B, A]) > 0).astype(int)
    EA, EB, PP = E[A], E[B], P[A, B]
    X_sym = np.hstack([EA + EB, EA * EB, np.abs(EA - EB), PP])
    X_ab = np.hstack([EA, EB, PP]); X_ba = np.hstack([EB, EA, PP])
    MH = np.zeros((len(y), n), dtype=np.float32); MH[np.arange(len(y)), A] = 1; MH[np.arange(len(y)), B] = 1
    g_full, g_out = drugbank_degrees(drugs)
    print(f'[proper {ds}] n={n} pairs={len(y)} pos={y.mean():.4f}  DrugBank deg median full={np.median(g_full):.0f} '
          f'outside-only={np.median(g_out):.0f}', flush=True)

    def evaluate(tr, tests, seed):
        out = {}
        ytr = y[tr]
        def put(name, scores, prob=True, train_scores=None):
            out[name] = {}
            for (k, te), s in zip(tests.items(), scores):
                if prob:
                    out[name][k] = metr(y[te], s, (s >= 0.5).astype(int))
                else:
                    out[name][k] = score_baseline(ytr, train_scores, s, y[te])
        m = rf(seed).fit(X_sym[tr], ytr)
        put('DDNet-sym RF', [m.predict_proba(X_sym[te])[:, 1] for te in tests.values()])
        m = rf(seed).fit(np.vstack([X_ab[tr], X_ba[tr]]), np.concatenate([ytr, ytr]))
        put('DDNet-concat both-orient RF', [(m.predict_proba(X_ab[te])[:, 1] + m.predict_proba(X_ba[te])[:, 1]) / 2
                                            for te in tests.values()])
        m = rf(seed).fit(MH[tr], ytr)
        put('One-hot identity RF', [m.predict_proba(MH[te])[:, 1] for te in tests.values()])
        r = per_drug_rate(A[tr], B[tr], ytr, n); r = np.where(np.isnan(r), ytr.mean(), r)
        put('Degree product (train labels)', [r[A[te]] * r[B[te]] for te in tests.values()], prob=False,
            train_scores=r[A[tr]] * r[B[tr]])
        put('DrugBank degree product, outside-dataset partners only', [g_out[A[te]] * g_out[B[te]] for te in tests.values()],
            prob=False, train_scores=g_out[A[tr]] * g_out[B[tr]])
        put('[leaky] DrugBank degree product, full', [g_full[A[te]] * g_full[B[te]] for te in tests.values()],
            prob=False, train_scores=g_full[A[tr]] * g_full[B[tr]])
        return out

    rng = np.random.default_rng(SEED)
    res = {'ds': ds, 'pairs': int(len(y)), 'pos': float(y.mean()), 'warm': [], 'node': []}
    fw = folds_of(len(y), 5, rng)
    for f in range(nrun):
        t0 = time.time()
        te = np.where(fw == f)[0]; tr = np.where(fw != f)[0]
        res['warm'].append(evaluate(tr, {'warm': te}, f))
        print(f'  warm fold {f} {time.time() - t0:.0f}s ' + ' | '.join(
            f"{k}: {v['warm']['AUROC']:.3f}/{v['warm']['MCC']:.3f}" for k, v in res['warm'][-1].items()), flush=True)
        json.dump(res, open(f'{HERE}/out/proper_{ds}.json', 'w'), indent=1, default=float)
    fn = folds_of(n, 5, rng)
    for f in range(5):
        t0 = time.time()
        held = fn == f
        tr = np.where(~held[A] & ~held[B])[0]
        s1 = np.where(held[A] ^ held[B])[0]
        s2 = np.where(held[A] & held[B])[0]
        # explicit drug-disjointness checks
        train_drugs = set(A[tr]) | set(B[tr])
        s2_drugs = set(A[s2]) | set(B[s2])
        assert not (train_drugs & s2_drugs), 'S2 not drug-disjoint'
        assert all((A[i] in train_drugs) != (B[i] in train_drugs) for i in s1[:2000])
        R = evaluate(tr, {'S1': s1, 'S2': s2}, 100 + f)
        R['_sizes'] = dict(train=int(len(tr)), S1=int(len(s1)), S2=int(len(s2)), held_drugs=int(held.sum()),
                           S2_pos=float(y[s2].mean()))
        res['node'].append(R)
        print(f'  node fold {f} {time.time() - t0:.0f}s sizes {R["_sizes"]} ' + ' | '.join(
            f"{k}: S1 {v['S1']['AUROC']:.3f} S2 {v['S2']['AUROC']:.3f}/{v['S2']['MCC']:.3f}"
            for k, v in R.items() if not k.startswith('_')), flush=True)
        json.dump(res, open(f'{HERE}/out/proper_{ds}.json', 'w'), indent=1, default=float)
    return res


if __name__ == '__main__':
    ds, proto = sys.argv[1], sys.argv[2]
    nrun = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    (run_paper if proto == 'paper' else run_proper)(ds, nrun)
