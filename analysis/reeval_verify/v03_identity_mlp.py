"""Independent check of the 'identity MLP' warm-split claim (their torch FFNN on multi-hot: AUROC 0.976 D1,
0.982 MP) with sklearn's MLPClassifier on the multi-hot drug identity of unordered pairs. Same warm folds as v01.
usage: python v03_identity_mlp.py DS [NWARM]
"""
import sys, json, time
import numpy as np
from sklearn.neural_network import MLPClassifier
from v01_verify import load, metr, folds_of, SEED, HERE


def run(ds, nwarm):
    drugs, E, P, Yd = load(ds)
    n = len(drugs)
    A, B = np.triu_indices(n, 1)
    y = ((Yd[A, B] + Yd[B, A]) > 0).astype(int)
    MH = np.zeros((len(y), n), np.float32); MH[np.arange(len(y)), A] = 1; MH[np.arange(len(y)), B] = 1
    rng = np.random.default_rng(SEED)
    fw = folds_of(len(y), 5, rng)
    fn = folds_of(n, 5, rng)
    res = {'warm': [], 'node': []}
    for f in range(nwarm):
        t0 = time.time()
        te = np.where(fw == f)[0]; tr = np.where(fw != f)[0]
        m = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256, max_iter=60,
                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, random_state=f)
        m.fit(MH[tr], y[tr]); s = m.predict_proba(MH[te])[:, 1]
        r = metr(y[te], s, (s >= 0.5).astype(int)); res['warm'].append(r)
        print(f'{ds} warm fold {f} {time.time() - t0:.0f}s AUROC {r["AUROC"]:.3f} AUPRC {r["AUPRC"]:.3f} '
              f'F1 {r["F1"]:.3f} MCC {r["MCC"]:.3f} (iters {m.n_iter_})', flush=True)
    for f in range(5):
        held = fn == f
        tr = np.where(~held[A] & ~held[B])[0]; s1 = np.where(held[A] ^ held[B])[0]; s2 = np.where(held[A] & held[B])[0]
        m = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256, max_iter=60,
                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, random_state=f)
        m.fit(MH[tr], y[tr])
        r = {k: metr(y[te], m.predict_proba(MH[te])[:, 1], (m.predict_proba(MH[te])[:, 1] >= 0.5).astype(int))
             for k, te in [('S1', s1), ('S2', s2)]}
        res['node'].append(r)
        print(f'{ds} node fold {f} S1 AUROC {r["S1"]["AUROC"]:.3f} MCC {r["S1"]["MCC"]:.3f} | '
              f'S2 AUROC {r["S2"]["AUROC"]:.3f} MCC {r["S2"]["MCC"]:.3f}', flush=True)
    json.dump(res, open(f'{HERE}/out/identity_mlp_{ds}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
