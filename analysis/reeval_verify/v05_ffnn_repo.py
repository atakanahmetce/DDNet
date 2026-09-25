"""Run the ORIGINAL repo FFNN code (model_train.train / evaluate_score, imported read-only from /home/user/DDNet,
no bytecode written) on the D1 hop-1 paper-style matrix (ordered pairs incl. self, [e_s,e_t,path], Normalizer
fitted on train as in get_data), 80/20 random split, for the settings the report / notebook may have used:
  isValid=False epochs=100 (notebook), isValid=False epochs=200 (train() default), isValid=True epochs=200.
Also reports AUROC from the continuous softmax output (not in the repo). usage: python v05_ffnn_repo.py [seed ...]
"""
import sys, os, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_auc_score
from model_train import train, evaluate_score
from v01_verify import load, HERE

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 4)))
drugs, E, P, Yd = load('D1')
n = len(drugs)
src = np.repeat(np.arange(n), n); tgt = np.tile(np.arange(n), n)
X = np.hstack([E[src], E[tgt], P[src, tgt]]); y = Yd[src, tgt].astype(float)
out = []
for seed in [int(s) for s in sys.argv[1:]] or [0]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    sc = Normalizer().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    for isValid, ep in [(False, 100), (False, 200), (True, 200)]:
        torch.manual_seed(seed)
        t0 = time.time()
        import io, contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            m = train(Xtr, ytr, ml_type='deep', epochs=ep, isValid=isValid)
        sc_, f = evaluate_score(m, Xte, yte, isDeep=True, preds=True)
        m.eval()
        with torch.no_grad():
            z = m(torch.from_numpy(Xte).float()).numpy().ravel()
        r = dict(seed=seed, isValid=isValid, epochs=ep, F1=sc_['F1-Score'], MCC=sc_['MCC'], AUPRC_hard=sc_['AUPRC'],
                 Precision=sc_['Precision'], pred_pos=float(f.mean()), AUROC_score=roc_auc_score(yte, z),
                 secs=time.time() - t0)
        out.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        json.dump(out, open(f'{HERE}/out/ffnn_repo_D1.json', 'w'), indent=1)
json.dump(out, open(f'{HERE}/out/ffnn_repo_D1.json', 'w'), indent=1)
