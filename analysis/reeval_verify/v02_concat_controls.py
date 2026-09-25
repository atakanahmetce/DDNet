"""Controls for the strongest learner found in v01 (RandomForest on the paper's concatenated layout,
trained on both orientations, test score = mean of both orientations). Same folds as v01 (seed 2026).
Feature sets (all through the identical RF-concat pipeline):
  DDNet      [e_a, e_b, path]
  RandVec    [r_a, r_b, path]    r = iid Gaussian, same per-coordinate scale as node2vec (seed 7)  -> identity code
  RandVec-np [r_a, r_b]          (no path)
  OneHot     [onehot(a), onehot(b)]                                                           -> pure identity
  Morgan     [fp_a, fp_b, Tanimoto(a,b)]   RDKit ECFP4 1024 bits from drugs.json SMILES
usage: python v02_concat_controls.py DS [NWARM]
"""
import sys, os, json, time
import numpy as np
from v01_verify import load, metr, rf, folds_of, SEED, HERE, ROOT


def morgan(drugs):
    import json as js
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
    sm = {d['id']: d['smiles'] for d in js.load(open(f'{ROOT}/drugs.json'))}
    F = np.zeros((len(drugs), 1024), np.float32); bad = 0
    for i, d in enumerate(drugs):
        m = Chem.MolFromSmiles(sm[d]) if sm.get(d) else None
        if m is None:
            bad += 1; continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        F[i] = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord('0')
    return F, bad


def run(ds, nwarm):
    drugs, E, P, Yd = load(ds)
    n = len(drugs)
    A, B = np.triu_indices(n, 1)
    y = ((Yd[A, B] + Yd[B, A]) > 0).astype(int)
    Rv = np.random.default_rng(7).normal(size=E.shape) * E.std(0, keepdims=True)
    OH = np.eye(n, dtype=np.float32)
    F, bad = morgan(drugs)
    inter = F @ F.T; c = F.sum(1); T = inter / np.maximum(c[:, None] + c[None, :] - inter, 1)
    PP = P[A, B]
    print(f'[v02 {ds}] n={n} pairs={len(y)} morgan_fail={bad}', flush=True)

    def feats(name, a, b, pp, tt):
        if name == 'DDNet concat RF':
            return np.hstack([E[a], E[b], pp])
        if name == 'RandVec+path concat RF':
            return np.hstack([Rv[a], Rv[b], pp])
        if name == 'RandVec (no path) concat RF':
            return np.hstack([Rv[a], Rv[b]])
        if name == 'One-hot concat RF':
            return np.hstack([OH[a], OH[b]])
        if name == 'Morgan FP concat RF':
            return np.hstack([F[a], F[b], tt[:, None]])
        raise KeyError(name)
    names = ['DDNet concat RF', 'RandVec+path concat RF', 'RandVec (no path) concat RF', 'One-hot concat RF',
             'Morgan FP concat RF']

    def evaluate(tr, tests, seed):
        out = {}
        a, b = A[tr], B[tr]
        for nm_ in names:
            Xtr = np.vstack([feats(nm_, a, b, PP[tr], T[a, b]), feats(nm_, b, a, PP[tr], T[a, b])])
            m = rf(seed).fit(Xtr, np.concatenate([y[tr], y[tr]]))
            out[nm_] = {}
            for k, te in tests.items():
                s = (m.predict_proba(feats(nm_, A[te], B[te], PP[te], T[A[te], B[te]]))[:, 1] +
                     m.predict_proba(feats(nm_, B[te], A[te], PP[te], T[A[te], B[te]]))[:, 1]) / 2
                out[nm_][k] = metr(y[te], s, (s >= 0.5).astype(int))
        return out

    rng = np.random.default_rng(SEED)
    fw = folds_of(len(y), 5, rng)       # identical to v01 (same rng call order)
    fn = folds_of(n, 5, rng)
    res = {'ds': ds, 'warm': [], 'node': []}
    for f in range(nwarm):
        t0 = time.time()
        te = np.where(fw == f)[0]; tr = np.where(fw != f)[0]
        res['warm'].append(evaluate(tr, {'warm': te}, f))
        print(f'  warm fold {f} {time.time() - t0:.0f}s ' + ' | '.join(
            f"{k}: {v['warm']['AUROC']:.3f}/{v['warm']['MCC']:.3f}" for k, v in res['warm'][-1].items()), flush=True)
        json.dump(res, open(f'{HERE}/out/concat_controls_{ds}.json', 'w'), indent=1, default=float)
    for f in range(5):
        t0 = time.time()
        held = fn == f
        tr = np.where(~held[A] & ~held[B])[0]; s1 = np.where(held[A] ^ held[B])[0]; s2 = np.where(held[A] & held[B])[0]
        assert not ((set(A[tr]) | set(B[tr])) & (set(A[s2]) | set(B[s2])))
        res['node'].append(evaluate(tr, {'S1': s1, 'S2': s2}, 100 + f))
        print(f'  node fold {f} {time.time() - t0:.0f}s ' + ' | '.join(
            f"{k}: S1 {v['S1']['AUROC']:.3f} S2 {v['S2']['AUROC']:.3f}/{v['S2']['MCC']:.3f}"
            for k, v in res['node'][-1].items()), flush=True)
        json.dump(res, open(f'{HERE}/out/concat_controls_{ds}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
