"""Method-lens pilot (candidate C, gate G1 preview): does CROssBAR v1 target annotation predict an UNSEEN drug's
role-specific DDI-type distribution better than chemical structure?  DDI-Ben DrugBank (LARS-research/DDI-Bench @ dfbeeab,
read-only copy in phase2/pilot2/repos).  Main-effects 'KN' rule from phase2/evalsci/pilot_typeprior.py:
  pred(h,t) = argmax_r  p_head(h)_r * p_tail(t)_r / prior_r
known drug -> counted role distribution from TRAINING triples; unseen drug -> similarity-weighted kNN (k=10) over TRAINING drugs.
Kernels for the unseen drug:  tani (Morgan Tanimoto, = evalsci baseline), tgt (Jaccard of CROssBAR v1 target accessions,
from DDNet data/drugs.json), mix (0.5*tani + 0.5*tgt), maxk (max of both).  Drugs without targets fall back to tani in tgt.
No tuning (k=10, alpha=1, weights fixed a priori). Metrics as DDI-Ben trainer.py (sklearn macro-F1, acc, kappa)."""
import sys, json, pickle, numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
ROOT = 'DDI-Bench/DDI_Ben/DDI_Ben'
OUT = 'out/phase2/methodC/'
K_NN, R, ALPHA = 10, 86, 1.0
x = pickle.load(open(f'{ROOT}/data/initial/drugbank/DB_molecular_feats.pkl', 'rb'))
ids = list(x['DrugBank ID']); N = len(ids)
F = (np.array([np.asarray(v, dtype=np.float32) for v in x['Morgan_Features']]) > 0).astype(np.float32)
inter = F @ F.T; c = F.sum(1); TAN = inter / np.maximum(c[:, None] + c[None, :] - inter, 1)
drugs = {d['id']: d for d in json.load(open('../../data/drugs.json'))}
acc_sets = [set(drugs.get(i, {}).get('accessions') or []) for i in ids]
vocab = sorted(set().union(*acc_sets)); vi = {a: j for j, a in enumerate(vocab)}
A = np.zeros((N, len(vocab)), np.float32)
for i, s in enumerate(acc_sets):
    for a in s: A[i, vi[a]] = 1
has_t = A.sum(1) > 0
ia = A @ A.T; ca = A.sum(1); TGT = ia / np.maximum(ca[:, None] + ca[None, :] - ia, 1)
print(f'drugs {N}; with >=1 v1 target {has_t.sum()} ({100*has_t.mean():.1f}%); target vocab {len(vocab)}', flush=True)
KERNELS = {'tani': TAN, 'tgt': np.where(has_t[:, None] & has_t[None, :], TGT, TAN),
           'mix': np.where(has_t[:, None] & has_t[None, :], 0.5 * TAN + 0.5 * TGT, TAN)}
KERNELS['maxk'] = np.maximum(TAN, KERNELS['tgt'])
res_all = {}
for SPLIT in ['drugbank_random', 'drugbank_cluster']:
    d = f'{ROOT}/data/{SPLIT}/'
    tr = np.loadtxt(d + 'train.txt', dtype=int)
    train_set = np.array(sorted(set(np.loadtxt(d + 'train_set.txt', dtype=int).tolist())))
    known = np.zeros(N, bool); known[train_set] = True
    Hc = np.zeros((N, R)); Tc = np.zeros((N, R))
    np.add.at(Hc, (tr[:, 0], tr[:, 2]), 1); np.add.at(Tc, (tr[:, 1], tr[:, 2]), 1)
    prior = np.bincount(tr[:, 2], minlength=R) + 1.0; prior /= prior.sum()
    norm = lambda M: (M + ALPHA * prior[None, :]) / (M + ALPHA * prior[None, :]).sum(1, keepdims=True)
    Hp, Tp = norm(Hc), norm(Tc)
    res = {}
    for kname, S_ in KERNELS.items():
        Sk = S_[:, train_set]
        nb = np.argsort(-Sk, 1)[:, :K_NN]; w = np.take_along_axis(Sk, nb, 1)
        def knn(M):
            agg = (w[:, :, None] * M[train_set][nb]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9) * 10
            return norm(agg)
        Hn, Tn = knn(Hc), knn(Tc)
        res[kname] = {}
        for S in ['S1', 'S2']:
            te = np.loadtxt(d + f'test_{S}.txt', dtype=int); h, t, y = te[:, 0], te[:, 1], te[:, 2]
            ph = np.where(known[h][:, None], Hp[h], Hn[h]); pt = np.where(known[t][:, None], Tp[t], Tn[t])
            p = np.argmax(ph * pt / prior[None, :], 1)
            newd = np.where(known[h], t, h) if S == 'S1' else h
            cov = float(has_t[newd].mean())
            res[kname][S] = dict(macroF1=round(100 * f1_score(y, p, average='macro'), 1), acc=round(100 * accuracy_score(y, p), 1),
                                 kappa=round(100 * cohen_kappa_score(y, p), 1), n=int(len(te)), new_drug_target_cov=round(100*cov, 1))
        print(SPLIT, kname, res[kname], flush=True)
    res_all[SPLIT] = res
json.dump(res_all, open(OUT + 'kg_roleprior.json', 'w'), indent=1)
