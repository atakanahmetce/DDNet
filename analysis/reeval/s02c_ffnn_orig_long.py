"""Sensitivity: does longer training of the ORIGINAL FFNN (Softmax(dim=0), BCELoss, thr 0.5/len(X))
reach the paper's FFNN numbers?  D1 hop-1, fold 1 of the as-in-paper split; checkpoints every 100 epochs.
(In the repo, deep_model(isValid=True) performs 20 optimiser steps per epoch -> 4000 steps for 200 epochs.)
"""
import json, time
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import Normalizer
from common import *
import nets

torch.set_num_threads(1)
ds = 'D1'
ids, E = load_emb(emb_file(ds, 1)); n = len(ids); Em = np.array([E[d] for d in ids])
P = load_paths(ds, ids); Yd = label_matrix(ds, ids, symmetric=False)
I, J = np.meshgrid(np.arange(n), np.arange(n), indexing='ij'); I = I.ravel(); J = J.ravel()
X = Normalizer().fit_transform(np.hstack([Em[I], Em[J], P[I, J]])).astype(np.float32)
y = Yd[I, J].astype(int)
tr, te = next(KFold(5, shuffle=True, random_state=0).split(X))
torch.manual_seed(0)
Xt = torch.from_numpy(X[tr]); yt = torch.from_numpy(y[tr].astype(np.float32)).reshape(-1, 1)
model = nets.Net(X.shape[1], 128, 5, 0.4, softmax=True)
opt = torch.optim.Adam(model.parameters(), lr=0.0007, weight_decay=1e-5)
crit = nn.BCELoss()
out = []
t0 = time.time()
for ep in range(1, 1001):
    model.train(); opt.zero_grad()
    loss = crit(model(Xt), yt); loss.backward(); opt.step()
    if ep % 100 == 0:
        h, s = nets.predict_orig(model, X[te])
        m = all_metrics(y[te], h, s); m['epoch'] = ep; m['pred_pos_rate'] = float(h.mean()); m['loss'] = float(loss)
        out.append(m)
        print(f"ep {ep} {time.time() - t0:.0f}s F1={m['F1']:.3f} MCC={m['MCC']:.3f} AUROC={m['AUROC']:.3f} "
              f"AUPRC_hard={m['AUPRC_hard']:.3f} predpos={m['pred_pos_rate']:.3f}", flush=True)
json.dump(out, open(f'{OUT}/out/ffnn_orig_long_D1h1.json', 'w'), indent=1, default=float)
