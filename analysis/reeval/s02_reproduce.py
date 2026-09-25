"""Tasks 1+2: reproduce DDNet 'as in paper' + trivial baselines under the SAME (leaky) protocol.

Protocol (featurization/f_mat_generation.gen_matrix + model_process/data_loader.get_data):
  rows  = ALL ordered pairs (src, tgt) over the embedding-file drugs, INCLUDING self pairs
  X     = [emb(src), emb(tgt), path(src,tgt)]  (128+128+3)
  y     = 1 if (src,tgt) in interactions.txt (directed; labels are stored in both orientations)
  prep  = sklearn Normalizer (row-wise L2), the get_data default (normal=True)
  split = shuffled 5-fold KFold (seed 0); fold 1 doubles as 'the single 80/20 random split'
Models: RF(100, class_weight=balanced), HGB (stand-in for sklearn GradientBoosting, see RESULTS.md),
        FFNN-orig (Softmax(dim=0)+BCELoss+thr 0.5/len(X)), FFNN-fixed (BCEWithLogits, thr 0.5, StandardScaler).
Baselines: all-positive, degree (train-only per-drug positive rate; product / sum), raw similarity S,
        one-hot identity [onehot(src), onehot(tgt)] with LR and RF, path-only (HGB), embedding-only (HGB).

usage: python s02_reproduce.py CONFIG [CONFIG ...]   (CONFIG in MP D1h1 D1h2 D2h1 D2h2 D1q1 D2q1)
env:   RF_JOBS (default 1), TORCH_THREADS (default 1), FOLDS (default 5)
"""
import os, sys, json, time
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.model_selection import KFold
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import torch
from common import *
import nets

RF_JOBS = int(os.environ.get('RF_JOBS', 1))
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 1)))
NFOLD = int(os.environ.get('FOLDS', 5))
MAXFOLDS = int(os.environ.get('MAXFOLDS', 5))   # run only the first k folds of the 5-fold split (compute budget)

CONFIGS = {
    'MP':   dict(ds='MP', hop=1, variant='', ablate=True),
    'D1h1': dict(ds='D1', hop=1, variant='', ablate=True),
    'D1h2': dict(ds='D1', hop=2, variant='', ablate=False),
    'D2h1': dict(ds='D2', hop=1, variant='', ablate=True),
    'D2h2': dict(ds='D2', hop=2, variant='', ablate=False),
    'D1q1': dict(ds='D1', hop=1, variant='q', ablate=True),   # IQR-'outlier'-filtered embedding file (71 drugs)
    'D2q1': dict(ds='D2', hop=1, variant='q', ablate=True),   # (73 drugs)
}


def hgb():
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, early_stopping=False, random_state=0)


def rf():
    return RandomForestClassifier(n_estimators=100, n_jobs=RF_JOBS, class_weight='balanced', random_state=0)


def fit_eval_sklearn(model, Xtr, ytr, Xte, yte):
    with threadpool_limits(1):
        model.fit(Xtr, ytr)
        s = model.predict_proba(Xte)[:, 1]
        p = model.predict(Xte)
    return all_metrics(yte, p, s)


def run(cfg_name):
    cfg = CONFIGS[cfg_name]
    ds = cfg['ds']
    path = emb_file(ds, cfg['hop'], 128, '1.2', '1.2', cfg['variant'])
    ids, E = load_emb(path)
    n = len(ids)
    Em = np.array([E[d] for d in ids])
    P = load_paths(ds, ids)
    Yd = label_matrix(ds, ids, symmetric=False)      # directed, exactly as get_label
    S = sim(ds)
    nm = names(ds); pos = {d: i for i, d in enumerate(nm)}
    Sx = S[np.ix_([pos[d] for d in ids], [pos[d] for d in ids])]
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing='ij'); I = I.ravel(); J = J.ravel()
    X = np.hstack([Em[I], Em[J], P[I, J]]).astype(np.float32)
    y = Yd[I, J].astype(int)
    Xn = Normalizer().fit_transform(X).astype(np.float32)
    print(f'== {cfg_name}: file={os.path.basename(path)} drugs={n} rows={len(y)} pos_rate={y.mean():.3f}', flush=True)
    kf = KFold(NFOLD, shuffle=True, random_state=0)
    out = {'config': cfg_name, 'file': os.path.basename(path), 'n_drugs': n, 'rows': int(len(y)),
           'pos_rate': float(y.mean()), 'folds': []}
    for f, (tr, te) in enumerate(kf.split(X)):
        if f >= MAXFOLDS:
            break
        t0 = time.time()
        R = {}
        # ---------------- DDNet models
        R['DDNet RF'] = fit_eval_sklearn(rf(), Xn[tr], y[tr], Xn[te], y[te])
        R['DDNet GB(HGB)'] = fit_eval_sklearn(hgb(), Xn[tr], y[tr], Xn[te], y[te])
        m = nets.train_orig(Xn[tr], y[tr], seed=f)
        h, s = nets.predict_orig(m, Xn[te])
        R['DDNet FFNN-orig'] = all_metrics(y[te], h, s)
        R['DDNet FFNN-orig']['pred_pos_rate'] = float(h.mean())
        sc = StandardScaler().fit(X[tr])
        m = nets.train_fixed(sc.transform(X[tr]), y[tr], seed=f)
        h, s = nets.predict_fixed(m, sc.transform(X[te]))
        R['DDNet FFNN-fixed'] = all_metrics(y[te], h, s)
        # ---------------- trivial baselines
        R['All-positive'] = all_metrics(y[te], np.ones(len(te), int), np.zeros(len(te)))
        # degree: per-drug positive rate from TRAINING rows only (drug as src or tgt)
        cnt = np.bincount(I[tr], minlength=n) + np.bincount(J[tr], minlength=n)
        posc = np.bincount(I[tr], weights=y[tr], minlength=n) + np.bincount(J[tr], weights=y[tr], minlength=n)
        r = np.where(cnt > 0, posc / np.maximum(cnt, 1), y[tr].mean())
        for nmk, fn in [('Degree (product)', lambda a, b: a * b), ('Degree (sum)', lambda a, b: a + b)]:
            st, se = fn(r[I[tr]], r[J[tr]]), fn(r[I[te]], r[J[te]])
            t = best_threshold(y[tr], st)
            R[nmk] = all_metrics(y[te], (se > t).astype(int), se)
        st, se = Sx[I[tr], J[tr]], Sx[I[te], J[te]]
        t = best_threshold(y[tr], st)
        R['Raw similarity S'] = all_metrics(y[te], (se > t).astype(int), se)
        if cfg['ablate']:
            OH = np.zeros((len(y), 2 * n), dtype=np.float32)
            OH[np.arange(len(y)), I] = 1; OH[np.arange(len(y)), n + J] = 1
            R['One-hot identity LR'] = fit_eval_sklearn(LogisticRegression(C=1.0, max_iter=3000), OH[tr], y[tr], OH[te], y[te])
            R['One-hot identity RF'] = fit_eval_sklearn(rf(), OH[tr], y[tr], OH[te], y[te])
            R['Path features only (HGB)'] = fit_eval_sklearn(hgb(), X[tr][:, -3:], y[tr], X[te][:, -3:], y[te])
            R['Embeddings only (HGB)'] = fit_eval_sklearn(hgb(), X[tr][:, :-3], y[tr], X[te][:, :-3], y[te])
            if n < 250 and os.environ.get('EMB_RF', '1') == '1':
                R['Embeddings only (RF)'] = fit_eval_sklearn(rf(), Xn[tr][:, :-3], y[tr], Xn[te][:, :-3], y[te])
        out['folds'].append(R)
        print(f'  fold {f}: {time.time() - t0:.0f}s  ' + '  '.join(
            f"{k}: F1={v['F1']:.3f} MCC={v['MCC']:.3f} AUROC={v['AUROC']:.3f}" for k, v in R.items()), flush=True)
        json.dump(out, open(f'{OUT}/out/reproduce_{cfg_name}.json', 'w'), indent=1, default=float)
    return out


if __name__ == '__main__':
    for c in sys.argv[1:]:
        run(c)
