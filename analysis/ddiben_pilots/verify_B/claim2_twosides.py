"""CLAIM 2: training-free partner label-degree score on DDI-Ben TWOSIDES random split.

deg_j(d) = number of distinct training partners x with a POSITIVE train.txt row (d,x) or (x,d) carrying label j
          (variant 'edges': number of positive rows instead of distinct partners;
           variant 'train+valid': graph = positive rows of train.txt + valid_S0/S1/S2 files)
score(h,t,j) = deg_j(h) + deg_j(t)   (in S1 exactly one of h,t is a training drug; the new drug has degree 0,
                                      so this equals the known drug's label-j degree)

Evaluators:
  trainer   : exact DDI_Ben trainer.py::predict for twosides: label vector looked up as
              np.array(list(sr2o_all[(h,t)]))[0] (sets built in the trainer's insertion order), loader batch 128
              shuffle=False drop_last=True, for each of the 209 labels j: rows with label[:,j]==1, target=label[j]*label[-1],
              roc_auc_score / average_precision_score, label with no rows contributes 0, mean over 209 labels.
  rowexact  : same per-label aggregation but label vector taken from the row itself and every row kept
              (== EmerGNN/TWOSIDES/base_model.py::evaluate on these files, which skips empty labels).
  pooled    : one ROC/PR over all (row, j) entries with the row carrying label j (target = p); raw degree as score.
  pooled_norm: same, degree divided by max_d deg_j(d) (per-label scale removed).
  fullmicro : standard multilabel micro average over the full (row x 209) matrix, target = Y[row,j]*p[row].
"""
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "out/phase3/indep_verify_c1c2")
from common import NUM_ENT, NUM_REL, OUT, load_sets, load_twosides, morgan_bitvects, smiles_twosides, tanimoto_matrix

R, NE = NUM_REL["twosides"], NUM_ENT["twosides"]
D = load_twosides()
sets = load_sets("twosides")
ORDER = ["train", "valid_S0", "test_S0", "valid_S1", "test_S1", "valid_S2", "test_S2"]

# ---------------- replicate trainer label lookup ----------------
sr2o = defaultdict(set)
H, T, Y, P = D["train"]
for h, t, y, p in zip(H, T, Y, P):
    sr2o[(int(h), int(t))].add(tuple(int(v) for v in y) + (int(p),))
for s in ORDER[1:]:
    H, T, Y, P = D[s]
    for h, t, y, p in zip(H, T, Y, P):
        sr2o[(int(h), int(t))].add(tuple(int(v) for v in y) + (int(p),))
sr2o_all = {k: list(v) for k, v in sr2o.items()}


def trainer_label(s):
    H, T, Y, P = D[s]
    L = np.array([sr2o_all[(int(h), int(t))][0] for h, t in zip(H, T)], dtype=np.int64)
    amb = sum(len(sr2o_all[(int(h), int(t))]) > 1 for h, t in zip(H, T))
    wrong = int((L != np.concatenate([Y, P[:, None]], 1)).any(1).sum())
    return L, amb, wrong


# ---------------- degree tables ----------------
def degree(files, kind="partners"):
    part = defaultdict(set)
    deg = np.zeros((NE, R))
    for s in files:
        H, T, Y, P = D[s]
        for h, t, y, p in zip(H, T, Y, P):
            if p != 1:
                continue
            for j in np.nonzero(y)[0]:
                if kind == "edges":
                    deg[h, j] += 1; deg[t, j] += 1
                else:
                    part[(h, j)].add(t); part[(t, j)].add(h)
    if kind == "partners":
        for (d, j), xs in part.items():
            deg[d, j] = len(xs)
    return deg


DEG = {"partners_train": degree(["train"]), "edges_train": degree(["train"], "edges"),
       "partners_train+valid": degree(["train", "valid_S0", "valid_S1", "valid_S2"])}


# ---------------- metrics ----------------
def per_label(Lmat, S):
    """Lmat: (n, 210) label vectors incl. p flag; S: (n, 209) scores. Mirrors trainer.py."""
    roc, prc, nl, n_single = [], [], 0, 0
    for j in range(R):
        w = np.where(Lmat[:, j] == 1)[0]
        if len(w) == 0:
            roc.append(0.0); prc.append(0.0); continue
        y = Lmat[w, j] * Lmat[w, -1]
        if len(np.unique(y)) < 2:  # would raise in sklearn; count it
            n_single += 1
            roc.append(np.nan); prc.append(np.nan); continue
        nl += 1
        roc.append(roc_auc_score(y, S[w, j])); prc.append(average_precision_score(y, S[w, j]))
    roc, prc = np.array(roc), np.array(prc)
    return dict(roc=100 * np.nanmean(roc), pr=100 * np.nanmean(prc), labels_scored=nl,
                labels_empty=int(sum(1 for j in range(R) if (Lmat[:, j] == 1).sum() == 0)), labels_single_class=n_single,
                roc_nonempty_only=100 * np.nanmean(roc[[(Lmat[:, j] == 1).sum() > 0 for j in range(R)]]),
                pr_nonempty_only=100 * np.nanmean(prc[[(Lmat[:, j] == 1).sum() > 0 for j in range(R)]]))


def pooled(Y, P, S, norm=None):
    r, c = np.nonzero(Y)
    sc = S[r, c] if norm is None else S[r, c] / np.maximum(norm[c], 1e-12)
    y = P[r]
    return dict(roc=100 * roc_auc_score(y, sc), pr=100 * average_precision_score(y, sc), n=int(len(y)))


def fullmicro(Y, P, S):
    y = (Y * P[:, None]).ravel()
    return dict(roc=100 * roc_auc_score(y, S.ravel()), pr=100 * average_precision_score(y, S.ravel()))


def all_metrics(s, S, H=None, T=None, Y=None, P=None, Ltr=None):
    if H is None:
        H, T, Y, P = D[s]
    out = {}
    if Ltr is not None:
        m = (len(H) // 128) * 128
        out["trainer"] = per_label(Ltr[:m], S[:m])
    out["rowexact"] = per_label(np.concatenate([Y, P[:, None]], 1), S)
    out["pooled"] = pooled(Y, P, S)
    out["pooled_norm"] = pooled(Y, P, S, norm=S.max(0) if S.max() > 0 else None)
    out["fullmicro"] = fullmicro(Y, P, S)
    return out


# ---------------- kNN-transferred scores (context only) ----------------
fps, nfail = morgan_bitvects(smiles_twosides(), NE, 2, 2048)
train_drugs = np.array(sorted(sets["train"]))
SIM = tanimoto_matrix(fps, list(range(NE)), list(train_drugs))
SIM[train_drugs, np.arange(len(train_drugs))] = -2
# adjacency by label (train positives)
A = np.zeros((NE, NE, R), dtype=np.uint8)
H, T, Y, P = D["train"]
for h, t, y, p in zip(H, T, Y, P):
    if p == 1:
        A[h, t] |= y; A[t, h] |= y


def knn(d, k):
    return train_drugs[np.argsort(-SIM[d], kind="stable")[:k]]


def score_knn_deg(H, T, deg, k):
    """new drug -> mean label-degree of its k nearest training drugs; known drug -> its own degree."""
    cache = {}
    def g(d):
        if d not in cache:
            cache[d] = deg[d] if d in sets["train"] else deg[knn(d, k)].mean(0)
        return cache[d]
    return np.array([g(h) + g(t) for h, t in zip(H, T)])


def score_knn_link(H, T, k):
    """S1 only: fraction of the new drug's k nearest training drugs that have a label-j train edge with the known drug."""
    out = np.zeros((len(H), R))
    for i, (h, t) in enumerate(zip(H, T)):
        known, new = (h, t) if h in sets["train"] else (t, h)
        out[i] = A[knn(new, k), known].mean(0)
    return out


results = {}
print("RDKit fp failures (twosides):", nfail)
for s in ["valid_S1", "test_S1", "valid_S2", "test_S2"]:
    H, T, Y, P = D[s]
    Ltr, amb, wrong = trainer_label(s)
    print(f"\n=== {s}: rows={len(H)} pos={P.sum()} neg={(P==0).sum()} | pairs with >1 label tuple in sr2o_all: {amb}, "
          f"rows whose trainer-looked-up label != own row: {wrong}; rows scored by trainer: {(len(H)//128)*128}")
    for name, deg in DEG.items():
        S = deg[H] + deg[T]
        res = all_metrics(s, S, Ltr=Ltr)
        results[(s, "degree_" + name)] = res
        print(f"  degree[{name:22s}] trainer ROC/PR={res['trainer']['roc']:.2f}/{res['trainer']['pr']:.2f} "
              f"(empty labels={res['trainer']['labels_empty']}, 1-class={res['trainer']['labels_single_class']}; nonempty-only "
              f"{res['trainer']['roc_nonempty_only']:.2f}/{res['trainer']['pr_nonempty_only']:.2f}) | rowexact "
              f"{res['rowexact']['roc']:.2f}/{res['rowexact']['pr']:.2f} (nonempty-only {res['rowexact']['roc_nonempty_only']:.2f}/"
              f"{res['rowexact']['pr_nonempty_only']:.2f}) | pooled {res['pooled']['roc']:.2f}/{res['pooled']['pr']:.2f} | "
              f"pooled_norm {res['pooled_norm']['roc']:.2f}/{res['pooled_norm']['pr']:.2f} | fullmicro "
              f"{res['fullmicro']['roc']:.2f}/{res['fullmicro']['pr']:.2f}", flush=True)
    for k in [5, 10, 20]:
        S = score_knn_deg(H, T, DEG["partners_train"], k)
        res = all_metrics(s, S, Ltr=Ltr)
        results[(s, f"knn_degree_k{k}")] = res
        print(f"  knn-degree k={k:2d}                trainer ROC/PR={res['trainer']['roc']:.2f}/{res['trainer']['pr']:.2f} | rowexact "
              f"{res['rowexact']['roc_nonempty_only']:.2f}/{res['rowexact']['pr_nonempty_only']:.2f} | pooled {res['pooled']['roc']:.2f}/{res['pooled']['pr']:.2f}", flush=True)
        if s.endswith("S1"):
            S = score_knn_link(H, T, k)
            res = all_metrics(s, S, Ltr=Ltr)
            results[(s, f"knn_link_k{k}")] = res
            print(f"  knn-link   k={k:2d}                trainer ROC/PR={res['trainer']['roc']:.2f}/{res['trainer']['pr']:.2f} | rowexact "
                  f"{res['rowexact']['roc_nonempty_only']:.2f}/{res['rowexact']['pr_nonempty_only']:.2f} | pooled {res['pooled']['roc']:.2f}/{res['pooled']['pr']:.2f}", flush=True)

json.dump({f"{a}|{b}": v for (a, b), v in results.items()}, open(f"{OUT}/claim2_results.json", "w"), indent=1)
print("saved")
