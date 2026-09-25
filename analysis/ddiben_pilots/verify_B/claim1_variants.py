"""CLAIM 1 robustness sweep (vectorised). Same rule as claim1_drugbank.py, with more design choices varied:

  fp       : rdkit Morgan r=2 2048 bits | rdkit r=2 1024 | DDI-Ben's own pkl Morgan_Features (binarised)
  k        : 5, 10, 20
  smoothing: none | add-alpha (alpha=1, 0.1, 0.01) | Dirichlet toward p(r) with strength m (m=1, 10)
  knn pool : 'role'  = k nearest among training drugs having >=1 triple in the needed role
             'all'   = k nearest among all drugs in train.txt; neighbours without that role are skipped
                       (if none of the k has it -> p(r))
  knn agg  : 'mean'  = mean of neighbours' normalised distributions
             'wmean' = Tanimoto-weighted mean
             'pool'  = summed neighbour counts, then normalised (+ same smoothing)
Evaluator 'trainer' = DDI_Ben trainer.py (min-relation label from sr2o_all, drop_last batches of 128).
"""
import itertools
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, "out/phase3/indep_verify_c1c2")
from common import (NUM_ENT, NUM_REL, OUT, load_drugbank, morgan_bitvects,
                    provided_morgan_drugbank, smiles_drugbank, tanimoto_matrix)

R, NE = NUM_REL["drugbank"], NUM_ENT["drugbank"]
data = load_drugbank()
tr = data["train"]
Ch = np.zeros((NE, R)); Ct = np.zeros((NE, R))
np.add.at(Ch, (tr[:, 0], tr[:, 2]), 1)
np.add.at(Ct, (tr[:, 1], tr[:, 2]), 1)
C = {"h": Ch, "t": Ct}
prior = np.bincount(tr[:, 2], minlength=R) / len(tr)
train_drugs = np.array(sorted(set(tr[:, 0]) | set(tr[:, 1])))
is_train = np.zeros(NE, bool); is_train[train_drugs] = True

sr2o_all = defaultdict(set)
for s in data:
    for h, t, r in data[s]:
        sr2o_all[(h, t)].add(r)

smi = smiles_drugbank()
FPS = {"rdkit2048": morgan_bitvects(smi, NE, 2, 2048)[0],
       "rdkit1024": morgan_bitvects(smi, NE, 2, 1024)[0],
       "pkl1024": provided_morgan_drugbank()}
all_drugs = list(range(NE))
SIM = {name: tanimoto_matrix(f, all_drugs, list(train_drugs)) for name, f in FPS.items()}  # (NE, n_train)
for name in SIM:
    SIM[name][train_drugs, np.arange(len(train_drugs))] = -2  # never pick yourself


def smooth(counts, sm):
    kind, v = sm
    n = counts.sum(-1, keepdims=True)
    if kind == "none":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, counts / np.maximum(n, 1e-300), np.nan)
    if kind == "add":
        return (counts + v) / (n + v * R)
    if kind == "dir":
        return (counts + v * R * prior) / (n + v * R)


def drug_role_dist(role, sm, fp, k, pool_kind, agg):
    """Distribution for every drug id (NE, R) in the given role."""
    Cr = C[role]
    own_n = Cr.sum(1)
    Pown = smooth(Cr, sm)  # nan rows for drugs with no data (no smoothing) -- replaced below
    S = SIM[fp]
    has_role = own_n[train_drugs] > 0
    out = np.empty((NE, R))
    for d in range(NE):
        if is_train[d] and own_n[d] > 0:
            out[d] = Pown[d]
            continue
        s = S[d].copy()
        if pool_kind == "role":
            s[~has_role] = -3
        if s.max() < 0:
            out[d] = prior
            continue
        nb_idx = np.argsort(-s, kind="stable")[:k]
        if pool_kind == "all":
            nb_idx = nb_idx[has_role[nb_idx]]
            if len(nb_idx) == 0:
                out[d] = prior
                continue
        nb = train_drugs[nb_idx]
        w = np.clip(s[nb_idx], 1e-6, None)
        if agg == "mean":
            out[d] = Pown[nb].mean(0)
        elif agg == "wmean":
            out[d] = (Pown[nb] * w[:, None]).sum(0) / w.sum()
        elif agg == "pool":
            out[d] = smooth(Cr[nb].sum(0), sm)
    return out


def evaluate(arr, Ph, Pt):
    h, t = arr[:, 0], arr[:, 1]
    sc = np.log(Ph[h] + 1e-12) + np.log(Pt[t] + 1e-12) - np.log(prior)[None]
    pred = sc.argmax(1)
    y = np.array([min(sr2o_all[(a, b)]) for a, b in zip(h, t)])
    m = (len(y) // 128) * 128
    y, pred = y[:m], pred[:m]
    return (100 * f1_score(y, pred, average="macro"), 100 * np.mean(y == pred), 100 * cohen_kappa_score(y, pred))


smooths = [("none", 0), ("add", 1.0), ("add", 0.1), ("add", 0.01), ("dir", 1.0), ("dir", 10.0)]
res = []
for fp, sm, pool_kind, agg in itertools.product(FPS, smooths, ["role", "all"], ["mean", "wmean", "pool"]):
    for k in [5, 10, 20]:
        Ph = drug_role_dist("h", sm, fp, k, pool_kind, agg)
        Pt = drug_role_dist("t", sm, fp, k, pool_kind, agg)
        row = dict(fp=fp, smooth=f"{sm[0]}{sm[1] if sm[0] != 'none' else ''}", pool=pool_kind, agg=agg, k=k)
        for s in ["valid_S1", "test_S1", "valid_S2", "test_S2"]:
            f1, acc, kap = evaluate(data[s], Ph, Pt)
            row[s] = [round(f1, 2), round(acc, 2), round(kap, 2)]
        res.append(row)
        print(json.dumps(row), flush=True)
json.dump(res, open(f"{OUT}/claim1_variants.json", "w"), indent=0)
