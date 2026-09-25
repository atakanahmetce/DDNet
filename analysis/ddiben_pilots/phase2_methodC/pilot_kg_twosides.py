"""Method-lens pilot 2: DDI-Ben TWOSIDES (multi-label, pos + corrupted negatives). Per-label main-effect score
g_j(a)+g_j(b), g = log1p(#train positive rows of the drug carrying label j) for known drugs; for UNSEEN drugs g is a
similarity-weighted kNN (k=10) mean over training drugs using (tani) Morgan Tanimoto [= evalsci baseline 'label-degree kNN'],
(tgt) Jaccard of CROssBAR v1 target accessions, (mix) 0.5/0.5.  TWOSIDES CIDs are mapped to DrugBank IDs by InChIKey
connectivity block (RDKit) against DDNet data/drugs.json SMILES; unmapped or target-less drugs fall back to Tanimoto.
Metric as DDI-Ben trainer.py (mean over labels of ROC-AUC / PR-AUC). No tuning."""
import json, pickle, numpy as np
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = 'DDI-Bench/DDI_Ben/DDI_Ben'
OUT = 'out/phase2/methodC/'
cid2id = json.load(open(f'{ROOT}/data/initial/twosides/cid2id.json')); cid2smi = json.load(open(f'{ROOT}/data/initial/twosides/cid2smiles.json'))
N = len(cid2id); id2cid = {v: k for k, v in cid2id.items()}
def ikb(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try: return Chem.MolToInchiKey(m).split('-')[0]
    except Exception: return None
db = {}
for d in json.load(open('../../data/drugs.json')):
    if d.get('accessions'):
        k = ikb(d['smiles'])
        if k: db.setdefault(k, set()).update(d['accessions'])
acc = []
for i in range(N):
    k = ikb(cid2smi[id2cid[i]]); acc.append(db.get(k, set()))
has_t = np.array([len(s) > 0 for s in acc]); print(f'TWOSIDES drugs {N}; mapped with >=1 v1 target {has_t.sum()} ({100*has_t.mean():.1f}%)', flush=True)
vocab = sorted(set().union(*acc)); vi = {a: j for j, a in enumerate(vocab)}
A = np.zeros((N, max(len(vocab), 1)), np.float32)
for i, s in enumerate(acc):
    for a in s: A[i, vi[a]] = 1
ia = A @ A.T; ca = A.sum(1); TGT = ia / np.maximum(ca[:, None] + ca[None, :] - ia, 1)
x = pickle.load(open(f'{ROOT}/data/initial/twosides/DB_molecular_feats.pkl', 'rb'))
F = (np.array(x, dtype=np.float32) > 0).astype(np.float32)
inter = F @ F.T; c = F.sum(1); TAN = inter / np.maximum(c[:, None] + c[None, :] - inter, 1)
both = has_t[:, None] & has_t[None, :]
KER = {'tani': TAN, 'tgt': np.where(both, TGT, TAN), 'mix': np.where(both, 0.5 * TAN + 0.5 * TGT, TAN)}
def load(fn):
    H, T, L, P = [], [], [], []
    for line in open(fn):
        h, t, r, p = line.strip().split(' ')
        H.append(int(h)); T.append(int(t)); L.append(np.array(r.split(','), dtype=np.int8)); P.append(int(p))
    return np.array(H), np.array(T), np.stack(L), np.array(P)
out = {}
for SPLIT in ['twosides_random', 'twosides_cluster']:
    d = f'{ROOT}/data/{SPLIT}/'
    h, t, L, P = load(d + 'train.txt'); R = L.shape[1]
    train_set = np.array(sorted(set(np.loadtxt(d + 'train_set.txt', dtype=int).tolist())))
    known = np.zeros(N, bool); known[train_set] = True
    pos = P == 1; C = np.zeros((N, R)); np.add.at(C, h[pos], L[pos]); np.add.at(C, t[pos], L[pos])
    out[SPLIT] = {}
    for kn, S_ in KER.items():
        Sk = S_[:, train_set]; nb = np.argsort(-Sk, 1)[:, :10]; w = np.take_along_axis(Sk, nb, 1)
        Cn = (w[:, :, None] * C[train_set][nb]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9)
        G = np.log1p(np.where(known[:, None], C, Cn))
        out[SPLIT][kn] = {}
        for S in ['valid_S1', 'valid_S2', 'test_S1', 'test_S2']:
            th, tt, tL, tP = load(d + f'{S}.txt'); s = G[th] + G[tt]; aucs, aps = [], []
            for j in range(R):
                ww = np.flatnonzero(tL[:, j] == 1)
                if len(ww) == 0: continue
                y = tP[ww]
                if y.min() == y.max(): continue
                aucs.append(roc_auc_score(y, s[ww, j])); aps.append(average_precision_score(y, s[ww, j]))
            out[SPLIT][kn][S] = dict(ROC_AUC=round(100 * np.mean(aucs), 1), PR_AUC=round(100 * np.mean(aps), 1))
        print(SPLIT, kn, out[SPLIT][kn], flush=True)
json.dump(out, open(OUT + 'kg_twosides.json', 'w'), indent=1)
