"""Independent re-implementation of CLAIM 2 (training-free partner label-degree score on DDI-Ben TWOSIDES random).

Score for a test row (h, t) and label j = number of distinct TRAINING partners of the row's known (train) drug
that share a POSITIVE training pair carrying label j. (In S2 no drug is known -> score 0 for all rows.)

Metrics:
  'ddiben'  exact replica of DDI_Ben/DDI_Ben/trainer.py::predict for twosides (non-MSTE models):
            label vector+flag taken from np.array(list(sr2o_all[(h,t)]))[0] (Data_record), rows in file order,
            DataLoader drop_last=True (batch 128), for each of the 209 label columns: rows with label_j==1,
            target = label_j * flag, roc_auc_score / average_precision_score, 0 if no rows, mean over 209.
  'emergnn' EmerGNN/TWOSIDES/base_model.py::evaluate: i-th positive paired with i-th negative, per label r over
            positives carrying r (+ the paired negatives), skip labels with no rows, mean.
  'within'  within-partner AUROC: for each label, only compare rows that share the same known drug.
"""
import sys, json
from collections import defaultdict as ddict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = 'DDI-Bench/DDI_Ben/DDI_Ben'
DATA = REPO + '/data/twosides_random'
NLAB = 209
SPLITS = ['train', 'valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']


def load(split):
    H, Tt, Y, P = [], [], [], []
    for line in open(f'{DATA}/{split}.txt'):
        h, t, r, p = line[:-1].split(' ')
        H.append(int(h)); Tt.append(int(t)); Y.append(tuple(int(x) for x in r.split(','))); P.append(int(p))
    return np.array(H), np.array(Tt), Y, np.array(P)


D = {s: load(s) for s in SPLITS}
train_set = set(np.loadtxt(f'{DATA}/train_set.txt', dtype=int).ravel().tolist())

# ---- training label-degree ----
h, t, Y, P = D['train']
Ymat = np.array(Y, dtype=np.int8)
partners = ddict(lambda: ddict(set))  # drug -> label -> partner set
pos_idx = np.where(P == 1)[0]
for i in pos_idx:
    for j in np.nonzero(Ymat[i])[0]:
        partners[h[i]][j].add(t[i]); partners[t[i]][j].add(h[i])
NENT = 645
DEG = np.zeros((NENT, NLAB))
for d, dd in partners.items():
    for j, s in dd.items():
        DEG[d, j] = len(s)
# row-count variant (non-distinct) and label-agnostic total degree for diagnostics
DEG_rows = np.zeros((NENT, NLAB))
np.add.at(DEG_rows, h[pos_idx], Ymat[pos_idx]); np.add.at(DEG_rows, t[pos_idx], Ymat[pos_idx])
tot_partners = np.zeros(NENT)
for d in range(NENT):
    s = set()
    for j in partners[d]:
        s |= partners[d][j]
    tot_partners[d] = len(s)

# ---- sr2o_all exactly as Data_record builds it for twosides, non-MSTE ----
sr2o = ddict(set)
for s in SPLITS:
    hh, tt, YY, PP = D[s]
    for a, b, y, p in zip(hh, tt, YY, PP):
        sr2o[(int(a), int(b))].add(tuple(list(y) + [int(p)]))
sr2o_all = {k: list(v) for k, v in sr2o.items()}


def known_drug(hh, tt):
    kh = np.isin(hh, list(train_set)); kt = np.isin(tt, list(train_set))
    kd = np.where(kh, hh, np.where(kt, tt, -1))
    return kd, kh, kt


def scores_for(split, kind='deg'):
    hh, tt, YY, PP = D[split]
    kd, kh, kt = known_drug(hh, tt)
    M = {'deg': DEG, 'deg_rows': DEG_rows}.get(kind)
    S = np.zeros((len(hh), NLAB))
    if kind in ('deg', 'deg_rows'):
        S[kh] += M[hh[kh]]; S[kt] += M[tt[kt]]
    elif kind == 'total_degree':
        S[kh] += tot_partners[hh[kh]][:, None]; S[kt] += tot_partners[tt[kt]][:, None]
    elif kind == 'random':
        S = np.random.default_rng(0).random((len(hh), NLAB))
    return S, kd


def eval_ddiben(split, S):
    hh, tt, YY, PP = D[split]
    lab = np.array([np.array(sr2o_all[(int(a), int(b))])[0] for a, b in zip(hh, tt)])  # (n, 210)
    file_lab = np.array([list(y) + [int(p)] for y, p in zip(YY, PP)])
    n_sub = int((lab != file_lab).any(1).sum())
    n = (len(hh) // 128) * 128
    lab = lab[:n]; pr = S[:n]
    roc, prc = [], []
    for j in range(NLAB):
        w = np.where(lab[:, j] == 1)[0]
        y = lab[w, j] * lab[w, -1]
        if len(w) == 0:
            roc.append(0); prc.append(0); continue
        roc.append(roc_auc_score(y, pr[w, j])); prc.append(average_precision_score(y, pr[w, j]))
    return dict(roc=100 * np.mean(roc), prc=100 * np.mean(prc), n_rows=n, n_label_substituted=n_sub,
                n_labels_zero=int(sum(1 for j in range(NLAB) if (lab[:, j] == 1).sum() == 0)))


def eval_emergnn(split, S):
    hh, tt, YY, PP = D[split]
    Yv = np.array(YY)
    pos = np.where(PP == 1)[0]; neg = np.where(PP == 0)[0]
    lab = Yv[pos]
    roc, prc = [], []
    for r in range(NLAB):
        idx = lab[:, r] > 0
        if idx.sum() == 0:
            continue
        sc = np.concatenate([S[pos[idx], r], S[neg[idx], r]])
        y = np.r_[np.ones(idx.sum()), np.zeros(idx.sum())]
        roc.append(roc_auc_score(y, sc)); prc.append(average_precision_score(y, sc))
    return dict(roc=100 * np.mean(roc), prc=100 * np.mean(prc), n_labels=len(roc))


def eval_within(split, S, kd):
    """Within-known-drug AUROC per label: pairs (pos,neg) compared only if same known drug.
    Returns pooled pair-level AUROC and mean of per-(label,drug) AUROCs, plus coverage."""
    hh, tt, YY, PP = D[split]
    Yv = np.array(YY)
    wins = ties = tot = 0
    per = []
    labels_cov = 0
    for j in range(NLAB):
        rows = np.where(Yv[:, j] == 1)[0]
        if len(rows) == 0:
            continue
        got = False
        for d in np.unique(kd[rows]):
            r = rows[kd[rows] == d]
            y = PP[r]
            if y.min() == y.max():
                continue
            got = True
            sp = S[r[y == 1], j]; sn = S[r[y == 0], j]
            cmp = sp[:, None] - sn[None, :]
            w = (cmp > 0).sum(); ti = (cmp == 0).sum(); n = cmp.size
            wins += w; ties += ti; tot += n
            per.append((w + 0.5 * ti) / n)
        labels_cov += got
    # between-partner: fraction of all pos-neg comparisons (same label) that are within-partner
    all_pairs = 0
    for j in range(NLAB):
        rows = np.where(Yv[:, j] == 1)[0]
        all_pairs += int((PP[rows] == 1).sum()) * int((PP[rows] == 0).sum())
    return dict(pooled_auroc=100 * (wins + 0.5 * ties) / tot if tot else float('nan'),
                mean_group_auroc=100 * np.mean(per) if per else float('nan'),
                n_groups=len(per), n_labels_with_groups=labels_cov,
                within_pairs=int(tot), all_same_label_pairs=all_pairs,
                frac_ties_within=(ties / tot if tot else float('nan')))


if __name__ == '__main__':
    out = {}
    for split in ['valid_S1', 'test_S1', 'valid_S2', 'test_S2', 'valid_S0', 'test_S0']:
        for kind in ['deg', 'deg_rows', 'total_degree', 'random']:
            if split.endswith('S0') and kind != 'deg':
                continue
            S, kd = scores_for(split, kind)
            if split.endswith('S0'):
                # both drugs known in S0: sum of both drugs' label degrees
                hh, tt, _, _ = D[split]; S = DEG[hh] + DEG[tt]; kd = np.full(len(hh), -1)
            r = dict(ddiben=eval_ddiben(split, S), emergnn=eval_emergnn(split, S))
            if split.endswith('S1'):
                r['within'] = eval_within(split, S, kd)
            out[f'{split}/{kind}'] = r
            print(f'== {split} score={kind}')
            for k, v in r.items():
                print('   ', k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    json.dump(out, open('out/phase3/anchor_reverify/twosides_results.json', 'w'), indent=1)
