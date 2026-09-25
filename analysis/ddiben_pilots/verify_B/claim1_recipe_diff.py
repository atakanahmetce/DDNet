"""CLAIM 1 discrepancy analysis. Written AFTER my own numbers were produced (claim1_drugbank.py / claim1_variants.py),
then reading the pilot's pilot_typeprior.py to identify its exact choices, re-implemented here in my own code.

Pilot choices identified (differ from the plain wording of the claim):
  (a) global prior p(r) = add-1 smoothed train frequency
  (b) per-drug role distribution = (counts + c * p(r)) / (n + c) with c = 1 (one pseudo-count spread by the prior),
      i.e. a weak Dirichlet-toward-prior smoothing, not raw empirical frequencies and not add-1 Laplace
  (c) 'known' = membership in train_set.txt (1,367 ids; 29 of them have no train triple -> prior)
  (d) new drug: k nearest in train_set (incl. drugs with no triples in that role) by Tanimoto on DDI-Ben's
      binarised 1024-bit pkl Morgan features; similarity-WEIGHTED MEAN OF COUNTS (not of normalised
      distributions), multiplied by 10, then (b)
  (e) evaluation on the row's own label, all rows (no drop_last, no min-label lookup)
This script varies c, k, count-scale, fp and evaluator around that recipe.
"""
import itertools
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, "out/phase3/indep_verify_c1c2")
from common import NUM_ENT, NUM_REL, load_drugbank, load_sets, morgan_bitvects, provided_morgan_drugbank, smiles_drugbank, tanimoto_matrix

R, NE = NUM_REL["drugbank"], NUM_ENT["drugbank"]
data = load_drugbank(); sets = load_sets("drugbank"); tr = data["train"]
Ch = np.zeros((NE, R)); Ct = np.zeros((NE, R))
np.add.at(Ch, (tr[:, 0], tr[:, 2]), 1); np.add.at(Ct, (tr[:, 1], tr[:, 2]), 1)
prior1 = (np.bincount(tr[:, 2], minlength=R) + 1.0); prior1 /= prior1.sum()
train_set = np.array(sorted(sets["train"]))
sr2o_all = defaultdict(set)
for s in data:
    for h, t, r in data[s]:
        sr2o_all[(h, t)].add(r)
FPS = {"pkl1024": provided_morgan_drugbank(), "rdkit2048": morgan_bitvects(smiles_drugbank(), NE, 2, 2048)[0]}
SIM = {n: tanimoto_matrix(f, list(range(NE)), list(train_set)) for n, f in FPS.items()}


def dists(M, c, k, fp, scale, known_mask):
    n = M.sum(1, keepdims=True)
    own = (M + c * prior1) / np.maximum(n + c, 1e-300)
    out = own.copy()
    S = SIM[fp]
    for d in np.where(~known_mask)[0]:
        s = S[d]
        idx = np.argsort(-s, kind="stable")[:k]
        w = s[idx]
        agg = (w[:, None] * M[train_set[idx]]).sum(0) / max(w.sum(), 1e-9) * scale
        out[d] = (agg + c * prior1) / (agg.sum() + c) if (agg.sum() + c) > 0 else prior1
    return out


known_mask = np.zeros(NE, bool); known_mask[train_set] = True


def run(c, k, fp, scale):
    Ph = dists(Ch, c, k, fp, scale, known_mask); Pt = dists(Ct, c, k, fp, scale, known_mask)
    out = {}
    for s in ["valid_S1", "test_S1", "valid_S2", "test_S2"]:
        a = data[s]
        sc = np.log(Ph[a[:, 0]] + 1e-12) + np.log(Pt[a[:, 1]] + 1e-12) - np.log(prior1)
        p = sc.argmax(1)
        y_row = a[:, 2]
        y_tr = np.array([min(sr2o_all[(h, t)]) for h, t in a[:, :2]]); m = (len(a) // 128) * 128
        out[s] = {"row": (100 * f1_score(y_row, p, average="macro"), 100 * np.mean(y_row == p), 100 * cohen_kappa_score(y_row, p)),
                  "trainer": (100 * f1_score(y_tr[:m], p[:m], average="macro"), 100 * np.mean(y_tr[:m] == p[:m]), 100 * cohen_kappa_score(y_tr[:m], p[:m]))}
    return out


print("fp        c     k  scale | " + " | ".join(f"{s} row F1/acc/kap  (trainer)" for s in ["valid_S1", "test_S1", "valid_S2", "test_S2"]))
for fp, c, k, scale in itertools.chain(
        [("pkl1024", 1.0, 10, 10.0)],  # pilot recipe
        itertools.product(["pkl1024", "rdkit2048"], [0.0, 1.0, 10.0, 86.0], [5, 10, 20], [10.0]),
        itertools.product(["pkl1024"], [1.0], [10], [1.0, 100.0])):
    o = run(c, k, fp, scale)
    print(f"{fp:9s} {c:5.1f} {k:3d} {scale:6.1f} | " + " | ".join(
        "{:.2f}/{:.2f}/{:.2f} ({:.2f}/{:.2f}/{:.2f})".format(*o[s]["row"], *o[s]["trainer"]) for s in ["valid_S1", "test_S1", "valid_S2", "test_S2"]), flush=True)
