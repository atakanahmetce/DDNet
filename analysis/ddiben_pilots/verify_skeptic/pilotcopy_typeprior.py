"""Pilot (evaluation-science lens): how much of DDI-Ben DrugBank S1/S2 macro-F1 does a pair-agnostic
'main-effects' lookup reproduce?  Reads DDI-Bench data (LARS-research/DDI-Bench @ dfbeeab) read-only.
Baselines (all use TRAINING triples only; no learning beyond counting + kNN):
  G  : global majority type
  K  : known drug's role-specific type distribution (argmax) -- S1 only; ignores the new drug entirely
  KN : product of role-specific type distributions, known drug (counted) x new drug (Tanimoto-kNN of train drugs)
  NN : both drugs via Tanimoto-kNN role distributions (S2; also reported for S1)
Macro-F1 / accuracy / kappa computed exactly as DDI-Ben trainer.py (sklearn average='macro', cohen_kappa).
"""
import sys, pickle, json, numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
ROOT = sys.argv[1]; SPLIT = sys.argv[2]  # drugbank_random | drugbank_cluster
K_NN = 10; R = 86; ALPHA = 1.0
d = f'{ROOT}/data/{SPLIT}/'
tr = np.loadtxt(d + 'train.txt', dtype=int)
train_set = set(np.loadtxt(d + 'train_set.txt', dtype=int).tolist())
x = pickle.load(open(f'{ROOT}/data/initial/drugbank/DB_molecular_feats.pkl', 'rb'))
F = np.array([np.asarray(v, dtype=np.float32) for v in x['Morgan_Features']]); F = (F > 0).astype(np.float32)
inter = F @ F.T; c = F.sum(1); T = inter / np.maximum(c[:, None] + c[None, :] - inter, 1)
N = F.shape[0]
Hc = np.zeros((N, R)); Tc = np.zeros((N, R))
np.add.at(Hc, (tr[:, 0], tr[:, 2]), 1); np.add.at(Tc, (tr[:, 1], tr[:, 2]), 1)
prior = np.bincount(tr[:, 2], minlength=R) + 1.0; prior /= prior.sum()
def norm(M): M = M + ALPHA * prior[None, :]; return M / M.sum(1, keepdims=True)
Hp, Tp = norm(Hc), norm(Tc)
tr_idx = np.array(sorted(train_set))
def knn_dist(M_counts, i):
    s = T[i, tr_idx].copy()
    nb = tr_idx[np.argsort(-s)[:K_NN]]; w = T[i, nb]
    return norm((w[:, None] * M_counts[nb]).sum(0, keepdims=True) / max(w.sum(), 1e-9) * 10)[0]
cacheH, cacheT = {}, {}
def role_dist(i, role, known):
    if known: return Hp[i] if role == 'h' else Tp[i]
    cache, M = (cacheH, Hc) if role == 'h' else (cacheT, Tc)
    if i not in cache: cache[i] = knn_dist(M, i)
    return cache[i]
out = {}
for S in ['S0', 'S1', 'S2']:
    te = np.loadtxt(d + f'test_{S}.txt', dtype=int)
    y = te[:, 2]; preds = {'G': [], 'K': [], 'KN': [], 'NN': []}
    for h, t, r in te:
        kh, kt = h in train_set, t in train_set
        ph, pt = role_dist(h, 'h', kh), role_dist(t, 't', kt)
        preds['G'].append(int(np.argmax(prior)))
        if S == 'S1':
            preds['K'].append(int(np.argmax(ph if kh else pt)))
        elif S == 'S0':
            preds['K'].append(int(np.argmax(ph * pt / prior)))
        preds['KN'].append(int(np.argmax(ph * pt / prior)))
        phn = role_dist(h, 'h', False) if True else ph; ptn = role_dist(t, 't', False)
        preds['NN'].append(int(np.argmax(phn * ptn / prior)))
    res = {}
    for k, p in preds.items():
        if len(p) == 0: continue
        res[k] = dict(macroF1=round(100 * f1_score(y, p, average='macro'), 1), acc=round(100 * accuracy_score(y, p), 1),
                      kappa=round(100 * cohen_kappa_score(y, p), 1))
    out[S] = dict(n=int(len(te)), **res)
    print(SPLIT, S, len(te), res, flush=True)
json.dump(out, open(f'out/phase3/skeptic_anchor/pilotcopy_typeprior_{SPLIT}.json', 'w'), indent=1)
