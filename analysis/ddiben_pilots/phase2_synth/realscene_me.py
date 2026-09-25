"""Synthesis check: main-effects (NN = both-drug Tanimoto-10NN role-type product) on DDI-Ben 'Real Scene'
(approval-time) split. Real Scene test drugs are disjoint from every train/valid file, so it is an S2-only test.
Same estimator as phase2/evalsci/pilot_typeprior.py (k=10, alpha=1, untuned). Reads DDI-Bench read-only."""
import pickle, json, numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
B = 'DDI-Bench'
RS = B + '/Real Scene/'
x = pickle.load(open(B + '/DDI_Ben/DDI_Ben/data/initial/drugbank/DB_molecular_feats.pkl', 'rb'))
F = np.array([np.asarray(v, dtype=np.float32) for v in x['Morgan_Features']]); F = (F > 0).astype(np.float32)
inter = F @ F.T; c = F.sum(1); T = inter / np.maximum(c[:, None] + c[None, :] - inter, 1)
N, R, K, A = F.shape[0], 86, 10, 1.0
def run(train_files, test_file):
    tr = np.concatenate([np.loadtxt(RS + f, dtype=int) for f in train_files])
    trd = np.array(sorted(set(tr[:, 0]) | set(tr[:, 1])))
    Hc = np.zeros((N, R)); Tc = np.zeros((N, R))
    np.add.at(Hc, (tr[:, 0], tr[:, 2]), 1); np.add.at(Tc, (tr[:, 1], tr[:, 2]), 1)
    prior = np.bincount(tr[:, 2], minlength=R) + 1.0; prior /= prior.sum()
    norm = lambda M: (M + A * prior) / (M + A * prior).sum(-1, keepdims=True)
    def kd(M, i):
        s = T[i, trd]; nb = trd[np.argsort(-s)[:K]]; w = T[i, nb]
        return norm((w[:, None] * M[nb]).sum(0) / max(w.sum(), 1e-9) * 10)
    te = np.loadtxt(RS + test_file, dtype=int); y = te[:, 2]
    pG = np.full(len(y), int(np.argmax(prior)))
    pNN = np.array([int(np.argmax(kd(Hc, h) * kd(Tc, t) / prior)) for h, t, _ in te])
    m = lambda p: dict(macroF1=round(100 * f1_score(y, p, average='macro'), 1), acc=round(100 * accuracy_score(y, p), 1), kappa=round(100 * cohen_kappa_score(y, p), 1))
    return dict(n=int(len(y)), n_train=int(len(tr)), majority=m(pG), ME_NN=m(pNN))
out = {'test<-all_train': run(['train_1.txt', 'train_2.txt', 'train_3.txt'], 'test.txt'),
       'test<-all_train+valid': run(['train_1.txt', 'train_2.txt', 'train_3.txt', 'valid_1.txt', 'valid_2.txt', 'valid_3.txt'], 'test.txt'),
       'valid_1<-train_1': run(['train_1.txt'], 'valid_1.txt'), 'valid_2<-train_2': run(['train_2.txt'], 'valid_2.txt'), 'valid_3<-train_3': run(['train_3.txt'], 'valid_3.txt')}
print(json.dumps(out, indent=1))
json.dump(out, open('out/phase2/synth/realscene_me.json', 'w'), indent=1)
