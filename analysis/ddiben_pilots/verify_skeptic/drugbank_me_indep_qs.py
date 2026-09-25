"""Independent re-implementation (skeptic) of CLAIM 1: training-free role-specific main-effects rule on DDI-Ben DrugBank.
Written from the claim text, not copied from the pilot. Reads DDI-Bench read-only.
score(r | h,t) = p_head(r|h) * p_tail(r|t) / p(r); known drug: smoothed empirical role distribution from train.txt;
new drug: Tanimoto(Morgan, binarised)-weighted mean of the k most similar TRAIN drugs' role COUNTS, then smoothed.
Evaluated under three label/row conventions:
  file   : label = 3rd column of the test file, all rows  (EmerGNN/DrugBank/base_model.py:119-141; TextDDI evaluate.py:288)
  ddiben : literal DDI_Ben/trainer.py path: label = argmax of multi-hot sr2o_all[(h,t)] over ALL split files
           (data_process.py:57-95, 390-395; trainer.py:177-193) and DataLoader(drop_last=True, bs=128) (data_process.py:327-334)
Also reports valid_S* to check that nothing was tuned on test, plus a small k/alpha sensitivity grid on valid."""
import sys, json, pickle, numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
ROOT = 'DDI-Bench/DDI_Ben/DDI_Ben'
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'drugbank_random'
D = f'{ROOT}/data/{SPLIT}/'
R, NENT = 86, 1710
files = ['train', 'valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']
data = {f: np.loadtxt(D + f + '.txt', dtype=np.int64) for f in files}
sets = {s: set(np.loadtxt(D + f'{s}_set.txt', dtype=int).tolist()) for s in ['train', 'valid', 'test']}
tr = data['train']
train_drugs = set(tr[:, 0].tolist()) | set(tr[:, 1].tolist())
info = dict(train_set_eq_train_file=train_drugs == sets['train'],
            overlaps={a + '&' + b: len(sets[a] & sets[b]) for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]})
# composition check of each eval file
for f in files[1:]:
    e = data[f]; kn = np.isin(e[:, 0], list(sets['train'])).astype(int) + np.isin(e[:, 1], list(sets['train'])).astype(int)
    info[f] = {f'known{k}': int((kn == k).sum()) for k in range(3)}
x = pickle.load(open(f'{ROOT}/data/initial/drugbank/DB_molecular_feats.pkl', 'rb'))
info['node_id_is_range'] = bool(np.array_equal(np.asarray(x['Node ID']), np.arange(NENT)))
B = np.stack([np.asarray(v) for v in x['Morgan_Features']]) > 0
B = B.astype(np.float64)
inter = B @ B.T; pop = B.sum(1)
union = pop[:, None] + pop[None, :] - inter
TAN = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

def build(train_rows):
    head = np.zeros((NENT, R)); tail = np.zeros((NENT, R))
    for h, t, r in train_rows: head[h, r] += 1; tail[t, r] += 1
    pri = (np.bincount(train_rows[:, 2], minlength=R) + 1.0); pri = pri / pri.sum()
    return head, tail, pri

def predictor(train_rows, known, k=10, alpha=1.0, scale=10.0):
    head, tail, pri = build(train_rows)
    kn_arr = np.array(sorted(known))
    def smooth(c): v = c + alpha * pri; return v / v.sum()
    memo = {}
    def dist(drug, role):
        key = (drug, role)
        if key in memo: return memo[key]
        M = head if role == 0 else tail
        if drug in known:
            v = smooth(M[drug])
        else:
            s = TAN[drug, kn_arr]; top = np.argsort(-s)[:k]; nb = kn_arr[top]; w = s[top]
            avg = (w[:, None] * M[nb]).sum(0) / max(w.sum(), 1e-12)
            v = smooth(avg * scale)
        memo[key] = v; return v
    def predict(rows):
        return np.array([int(np.argmax(dist(h, 0) * dist(t, 1) / pri)) for h, t in rows[:, :2]])
    return predict

# DDI_Ben trainer label convention
sr2o = defaultdict(set)
for f in files:
    for h, t, r in data[f]: sr2o[(int(h), int(t))].add(int(r))
def ddiben_label(rows):
    return np.array([min(sr2o[(int(h), int(t))]) for h, t in rows[:, :2]])  # argmax of multi-hot = smallest index
def metrics(y, p):
    return dict(macroF1=round(100 * f1_score(y, p, average='macro'), 2), acc=round(100 * accuracy_score(y, p), 2),
                kappa=round(100 * cohen_kappa_score(y, p), 2), n=int(len(y)))
known = sets['train']
pred = predictor(tr, known)
res = {'info': info}
for f in files[1:]:
    e = data[f]; p = pred(e)
    n_keep = len(e) - len(e) % 128
    yl = ddiben_label(e)
    res[f] = dict(file=metrics(e[:, 2], p), ddiben=metrics(yl[:n_keep], p[:n_keep]),
                  n_rows_multi_rel=int(sum(len(sr2o[(int(h), int(t))]) > 1 for h, t in e[:, :2])),
                  n_label_changed=int((yl != e[:, 2]).sum()))
    print(f, res[f], flush=True)
# sensitivity on VALID only
grid = {}
for k in [5, 10, 20]:
    for a in [0.3, 1.0, 3.0]:
        pr = predictor(tr, known, k=k, alpha=a)
        grid[f'k{k}_a{a}'] = {f: metrics(data[f][:, 2], pr(data[f]))['macroF1'] for f in ['valid_S1', 'valid_S2', 'test_S1', 'test_S2']}
        print(k, a, grid[f'k{k}_a{a}'], flush=True)
res['grid_macroF1'] = grid
json.dump(res, open(f'out/phase3/skeptic_anchor/me_indep_quicksort_{SPLIT}.json', 'w'), indent=1, default=str)
