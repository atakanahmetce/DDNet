"""CLAIM 1: training-free additive 'main-effects' rule on DDI-Ben DrugBank random split.

score(r | h, t) = log p_head(r | h) + log p_tail(r | t) - log p(r)
  * known drug (appears in train.txt with >=1 triple in the needed role): its own empirical
    role-specific relation distribution from train.txt
  * new drug (or known drug with no triple in that role): mean of the role-specific distributions of
    its k most Tanimoto-similar (Morgan) training drugs among those having >=1 triple in that role
  * p(r): relation frequency in train.txt
  * smoothing alpha: per-drug add-alpha on counts (alpha=0: raw; a 1e-12 floor inside the log only
    breaks all-zero ties, i.e. prefers relations with fewer zero factors)

Two evaluators:
  'trainer' : exactly DDI_Ben/DDI_Ben/trainer.py::predict for drugbank: label = argmax of the multi-hot
              of sr2o_all[(h,t)] (= smallest relation id among all relations of the ordered pair in
              train + every valid/test file), TestDataset loader with batch_size=128, shuffle=False,
              drop_last=True (so the trailing len % 128 rows are not scored);
              sklearn f1_score(average='macro'), accuracy, cohen_kappa_score.
  'rowlabel': the relation on the row itself, all rows (as EmerGNN/DrugBank/base_model.py::evaluate).
"""
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, "out/phase3/indep_verify_c1c2")
from common import (NUM_ENT, NUM_REL, OUT, load_drugbank, load_sets, morgan_bitvects,
                    provided_morgan_drugbank, smiles_drugbank, tanimoto_matrix)

R = NUM_REL["drugbank"]
NE = NUM_ENT["drugbank"]
data = load_drugbank()
sets = load_sets("drugbank")
tr = data["train"]

# ---------------- counts ----------------
Ch = np.zeros((NE, R)); Ct = np.zeros((NE, R))
np.add.at(Ch, (tr[:, 0], tr[:, 2]), 1)
np.add.at(Ct, (tr[:, 1], tr[:, 2]), 1)
prior = np.bincount(tr[:, 2], minlength=R) / len(tr)
train_drugs = np.array(sorted(set(tr[:, 0]) | set(tr[:, 1])))
pool = {"h": np.array([d for d in train_drugs if Ch[d].sum() > 0]),
        "t": np.array([d for d in train_drugs if Ct[d].sum() > 0])}
C = {"h": Ch, "t": Ct}
print(f"train triples {len(tr)}, train drugs {len(train_drugs)}, head pool {len(pool['h'])}, tail pool {len(pool['t'])}")

# ---------------- labels as the DDI_Ben trainer builds them ----------------
sr2o_all = defaultdict(set)
for s in data:
    for h, t, r in data[s]:
        sr2o_all[(h, t)].add(r)


def trainer_labels(arr):
    return np.array([min(sr2o_all[(h, t)]) for h, t, _ in arr])


# ---------------- fingerprints ----------------
fps_all = {}
fps_all["rdkit_r2_2048"], nfail = morgan_bitvects(smiles_drugbank(), NE, 2, 2048)
print("rdkit Morgan fp failures:", nfail)
fps_all["ddiben_pkl_morgan1024"] = provided_morgan_drugbank()

eval_drugs = sorted(set(np.concatenate([data[s][:, :2].ravel() for s in data if s != "train"])))
knn_cache = {}
for fpname, fps in fps_all.items():
    for role in ["h", "t"]:
        S = tanimoto_matrix(fps, eval_drugs, pool[role])
        # exclude self (for known drugs lacking the role this matters only if the drug is in the pool,
        # which by construction it is not; keep for safety)
        for i, d in enumerate(eval_drugs):
            S[i, pool[role] == d] = -2
        # stable order: similarity desc, then drug id asc
        order = np.argsort(-S, axis=1, kind="stable")  # pool is id-sorted -> ties broken by id
        knn_cache[(fpname, role)] = ({d: i for i, d in enumerate(eval_drugs)}, order, S)


def role_dist(d, role, alpha, k, fpname, stats):
    Cr = C[role]
    n = Cr[d].sum()
    if n > 0:
        stats["own"] += 1
        return (Cr[d] + alpha) / (n + alpha * R)
    idx, order, S = knn_cache[(fpname, role)]
    i = idx[d]
    if S[i].max() < 0:  # no fingerprint
        stats["prior_fallback"] += 1
        return prior.copy()
    nb = pool[role][order[i, :k]]
    stats["knn_known" if d in set_train_drugs else "knn_new"] += 1
    P = (Cr[nb] + alpha) / (Cr[nb].sum(1, keepdims=True) + alpha * R)
    return P.mean(0)


set_train_drugs = set(train_drugs.tolist())


def predict(arr, alpha, k, fpname, mode="full"):
    stats = defaultdict(int)
    cache = {}
    preds = np.empty(len(arr), dtype=np.int64)
    allzero = 0
    for n, (h, t, _) in enumerate(arr):
        key_h, key_t = (h, "h"), (t, "t")
        if key_h not in cache:
            cache[key_h] = role_dist(h, "h", alpha, k, fpname, stats)
        if key_t not in cache:
            cache[key_t] = role_dist(t, "t", alpha, k, fpname, stats)
        ph, pt = cache[key_h], cache[key_t]
        if mode == "full":
            prod = ph * pt / prior
            if prod.max() <= 0:
                allzero += 1
            sc = np.log(ph + 1e-12) + np.log(pt + 1e-12) - np.log(prior)
        elif mode == "known_only":  # only the known (train) drug's role distribution
            known_h = h in set_train_drugs
            sc = np.log((ph if known_h else pt) + 1e-12)
        elif mode == "prior":
            sc = np.log(prior)
        preds[n] = int(np.argmax(sc))
    return preds, dict(stats), allzero


def metrics(y, p):
    return dict(f1=100 * f1_score(y, p, average="macro"), acc=100 * float(np.mean(y == p)),
                kappa=100 * cohen_kappa_score(y, p), n=int(len(y)))


results = []
eval_splits = ["valid_S1", "test_S1", "valid_S2", "test_S2", "valid_S0", "test_S0"]
lab = {s: {"rowlabel": data[s][:, 2], "trainer": trainer_labels(data[s])} for s in eval_splits}
for s in eval_splits:
    print(s, "rows", len(data[s]), "rows whose trainer label != row label:", int((lab[s]["rowlabel"] != lab[s]["trainer"]).sum()),
          "rows scored by trainer (drop_last,128):", (len(data[s]) // 128) * 128)

configs = [(fp, k, a, "full") for fp in fps_all for k in [5, 10, 20] for a in [0.0, 1.0]]
configs += [("rdkit_r2_2048", 10, 0.0, "known_only"), ("rdkit_r2_2048", 10, 0.0, "prior")]
for fp, k, a, mode in configs:
    for s in eval_splits:
        p, st, az = predict(data[s], a, k, fp, mode)
        for ev in ["trainer", "rowlabel"]:
            y = lab[s][ev]
            if ev == "trainer":
                m = (len(y) // 128) * 128
                met = metrics(y[:m], p[:m])
            else:
                met = metrics(y, p)
            row = dict(fp=fp, k=k, alpha=a, mode=mode, split=s, evaluator=ev, **met, allzero_rows=az, **{"n_" + kk: v for kk, v in st.items()})
            results.append(row)
        print(f"{fp:22s} k={k:2d} a={a} {mode:10s} {s:9s} trainer F1/acc/kappa = "
              f"{results[-2]['f1']:.2f}/{results[-2]['acc']:.2f}/{results[-2]['kappa']:.2f} | rowlabel "
              f"{results[-1]['f1']:.2f}/{results[-1]['acc']:.2f}/{results[-1]['kappa']:.2f}  allzero={az} {st}", flush=True)

json.dump(results, open(f"{OUT}/claim1_results.json", "w"), indent=1)
print("saved", f"{OUT}/claim1_results.json")
