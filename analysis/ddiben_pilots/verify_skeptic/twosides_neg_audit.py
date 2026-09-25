"""Skeptic audit of DDI-Ben TWOSIDES negatives (read-only on DDI-Bench data).
For each split file: rows alternate (pos, neg)? neg label vector == pos label vector?
Does the neg keep a drug of its paired pos? Which drugs does it keep? Are negatives ever true positives?
Also recompute the known-drug label-degree floor with (a) the pilot's aggregation and
(b) a literal re-implementation of DDI_Ben/trainer.py:188-215 incl. sr2o_all label lookup (data_process.py:77-95,
TestDataset data_process.py:390-395) and DataLoader drop_last=True (data_process.py:327-334)."""
import sys, json, numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = 'DDI-Bench/DDI_Ben/DDI_Ben'
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'twosides_random'
d = f'{ROOT}/data/{SPLIT}/'

def load(fn):
    rows = []
    for line in open(d + fn):
        h, t, r, p = line[:-1].split(' ')
        rows.append((int(h), int(t), tuple(int(x) for x in r.split(',')) + (int(p),)))
    return rows

files = ['train', 'valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']
data = {f: load(f + '.txt') for f in files}
sets = {s: set(np.loadtxt(d + f'{s}_set.txt', dtype=int).tolist()) for s in ['train', 'valid', 'test']}
train_drugs_in_file = set(h for h, t, _ in data['train']) | set(t for h, t, _ in data['train'])
out = {'train_set_equals_train_file_drugs': sets['train'] == train_drugs_in_file,
       'set_sizes': {k: len(v) for k, v in sets.items()},
       'set_overlaps': {'train&valid': len(sets['train'] & sets['valid']), 'train&test': len(sets['train'] & sets['test']),
                        'valid&test': len(sets['valid'] & sets['test'])}}
# positive pair universe (unordered) over ALL files
pos_pairs = set()
for f in files:
    for h, t, r in data[f]:
        if r[-1] == 1: pos_pairs.add(frozenset((h, t)))
known = sets['train']
for f in files:
    rows = data[f]
    P = np.array([r[-1] for _, _, r in rows])
    alt = bool(np.all(P[0::2] == 1) and np.all(P[1::2] == 0))
    same_lab = all(rows[i][2][:-1] == rows[i + 1][2][:-1] for i in range(0, len(rows) - 1, 2))
    keep = Counter(); keep_known = 0; keep_new = 0; neg_is_pos = 0; nk = Counter()
    for i in range(0, len(rows) - 1, 2):
        ph, pt, _ = rows[i]; nh, nt, _ = rows[i + 1]
        shared = {ph, pt} & {nh, nt}
        keep[len(shared)] += 1
        pk = [x for x in (ph, pt) if x in known]; pn = [x for x in (ph, pt) if x not in known]
        if any(x in (nh, nt) for x in pk): keep_known += 1
        if any(x in (nh, nt) for x in pn): keep_new += 1
        if frozenset((nh, nt)) in pos_pairs: neg_is_pos += 1
        nk[(int(nh in known) + int(nt in known))] += 1
    out[f] = dict(n=len(rows), alternating_pos_neg=alt, neg_label_vec_equals_pos=same_lab,
                  neg_shares_k_drugs_with_pos=dict(keep), neg_keeps_a_known_drug_of_pos=keep_known,
                  neg_keeps_a_new_drug_of_pos=keep_new, neg_pair_is_a_positive_somewhere=neg_is_pos,
                  neg_num_known_drugs=dict(nk))

# ---- degree floor: pilot aggregation vs literal DDI_Ben trainer ----
N = 645; R = 209
C = np.zeros((N, R))
for h, t, r in data['train']:
    if r[-1] == 1:
        v = np.array(r[:-1]); C[h] += v; C[t] += v
kn = np.zeros(N, bool); kn[list(known)] = True
G = np.log1p(C * kn[:, None])
# sr2o_all exactly as data_process.py:37-81 (non-MSTE branch)
sr2o = defaultdict(set)
for f in files:
    for h, t, r in data[f]: sr2o[(h, t)].add(r)
sr2o_all = {k: list(v) for k, v in sr2o.items()}
def trainer_metrics(rows, score_fn, drop_last=True, bs=128):
    n = len(rows) - (len(rows) % bs if drop_last else 0)
    rows = rows[:n]
    lab = np.array([np.array(sr2o_all[(h, t)])[0] for h, t, _ in rows])  # TestDataset: np.array(ele['label'])[0]
    pred = np.stack([score_fn(h, t) for h, t, _ in rows])
    roc, prc = [], []
    for j in range(R):  # pred has 209 cols (trainer.py:175-176 slices to num_rel)
        w = np.where(lab[:, j] == 1)[0]
        y = lab[w, j] * lab[w, -1]; s = pred[w, j]
        if y.shape[0] == 0: roc.append(0); prc.append(0); continue
        roc.append(roc_auc_score(y, s)); prc.append(average_precision_score(y, s))
    return dict(ROC_AUC=round(100 * np.mean(roc), 2), PR_AUC=round(100 * np.mean(prc), 2), n_rows=n,
                n_rows_label_mismatch=int(sum(tuple(l) != r for l, (_, _, r) in zip(lab, rows))))
def pilot_metrics(rows, score_fn):
    lab = np.array([r for _, _, r in rows]); pred = np.stack([score_fn(h, t) for h, t, _ in rows])
    roc, prc = [], []
    for j in range(R):
        w = np.flatnonzero(lab[:, j] == 1)
        if len(w) == 0: continue
        y = lab[w, -1]
        if y.min() == y.max(): continue
        roc.append(roc_auc_score(y, pred[w, j])); prc.append(average_precision_score(y, pred[w, j]))
    return dict(ROC_AUC=round(100 * np.mean(roc), 2), PR_AUC=round(100 * np.mean(prc), 2), n_labels=len(roc))
score = lambda h, t: G[h] + G[t]
# EmerGNN evaluator (EmerGNN/TWOSIDES/base_model.py:116-164): split pos/neg rows, per label over pos rows with label r and paired negs
def emergnn_metrics(rows, score_fn):
    pos = [x for x in rows if x[2][-1] == 1]; neg = [x for x in rows if x[2][-1] == 0]
    L = np.array([r[:-1] for _, _, r in pos]); ps = np.stack([score_fn(h, t) for h, t, _ in pos]); ns = np.stack([score_fn(h, t) for h, t, _ in neg])
    roc, prc = [], []
    for j in range(R):
        idx = L[:, j] > 0
        if idx.sum() == 0: continue
        lab = [1] * idx.sum() + [0] * idx.sum(); sc = list(ps[idx, j]) + list(ns[idx, j])
        roc.append(roc_auc_score(lab, sc)); prc.append(average_precision_score(lab, sc))
    return dict(ROC_AUC=round(100 * np.mean(roc), 2), PR_AUC=round(100 * np.mean(prc), 2), n_labels=len(roc))
out['degree_floor'] = {}
for S in ['S0', 'S1', 'S2']:
    for part in ['valid', 'test']:
        rows = data[f'{part}_{S}']
        out['degree_floor'][f'{part}_{S}'] = dict(pilot_agg=pilot_metrics(rows, score),
                                                  ddiben_trainer_literal=trainer_metrics(rows, score),
                                                  emergnn_evaluator=emergnn_metrics(rows, score))
# control: random known drug swap score -- degree of the NEG's known drug vs POS's known drug
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(f'out/phase3/skeptic_anchor/twosides_audit_{SPLIT}.json', 'w'), indent=1, default=str)
