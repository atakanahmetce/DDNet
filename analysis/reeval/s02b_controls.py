"""Controls under the as-in-paper (leaky) protocol, same folds as s02_reproduce.py:
  RandomVec + path : node2vec vectors replaced by i.i.d. Gaussian 128-D vectors (identity codes, no graph info)
                     -> HGB and FFNN-fixed
  Identity MLP     : FFNN-fixed on [onehot(src), onehot(tgt)]
usage: python s02b_controls.py CONFIG [...]    env MAXFOLDS, TORCH_THREADS
"""
import os, sys, json, time
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
from common import *
import nets
from s02_reproduce import CONFIGS

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 1)))
MAXFOLDS = int(os.environ.get('MAXFOLDS', 5))


def run(cfg_name):
    cfg = CONFIGS[cfg_name]; ds = cfg['ds']
    ids, E = load_emb(emb_file(ds, cfg['hop'], 128, '1.2', '1.2', cfg['variant']))
    n = len(ids); Em = np.array([E[d] for d in ids])
    Rm = np.random.RandomState(123).normal(size=Em.shape) * Em.std()
    P = load_paths(ds, ids)
    Yd = label_matrix(ds, ids, symmetric=False)
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing='ij'); I = I.ravel(); J = J.ravel()
    X = np.hstack([Rm[I], Rm[J], P[I, J]]).astype(np.float32)
    y = Yd[I, J].astype(int)
    OH = np.zeros((len(y), 2 * n), dtype=np.float32)
    OH[np.arange(len(y)), I] = 1; OH[np.arange(len(y)), n + J] = 1
    out = {'config': cfg_name, 'folds': []}
    for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
        if f >= MAXFOLDS:
            break
        t0 = time.time(); R = {}
        with threadpool_limits(1):
            m = HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=0).fit(X[tr], y[tr])
            R['RandomVec + path HGB (control)'] = all_metrics(y[te], m.predict(X[te]), m.predict_proba(X[te])[:, 1])
        sc = StandardScaler().fit(X[tr])
        m = nets.train_fixed(sc.transform(X[tr]), y[tr], seed=f)
        h, s = nets.predict_fixed(m, sc.transform(X[te]))
        R['RandomVec + path FFNN-fixed (control)'] = all_metrics(y[te], h, s)
        m = nets.train_fixed(OH[tr], y[tr], seed=f)
        h, s = nets.predict_fixed(m, OH[te])
        R['Identity MLP (FFNN-fixed on one-hot)'] = all_metrics(y[te], h, s)
        out['folds'].append(R)
        print(f'{cfg_name} fold {f}: {time.time() - t0:.0f}s ' + '  '.join(
            f"{k}: F1={v['F1']:.3f} MCC={v['MCC']:.3f} AUROC={v['AUROC']:.3f}" for k, v in R.items()), flush=True)
        json.dump(out, open(f'{OUT}/out/controls_{cfg_name}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    for c in sys.argv[1:]:
        run(c)
