"""CLAIM 2 mechanism + partner-matched negatives on DDI-Ben TWOSIDES random S1 (valid and test).

Part A: how DDI-Ben S1 negatives are formed (row 2i = positive, row 2i+1 = its negative; checked).
Part B: partner-matched negatives. For every S1 positive row (new n, known k, label vector Y) and every
        label j with Y_j = 1, build a negative (n', k) with n' drawn uniformly (seed 0) from the same held-out
        drug set as n (test_set for test, valid_set for valid), n' != n, such that (n', k) has NO recorded
        label-j positive in any file (train + all valid/test files, both orientations); n' takes n's position.
        Evaluated per label j (rows carrying j), exactly like the DDI-Ben evaluator, and pooled.
        Variant B2 (row level, DDI-Ben format): one negative per positive row, n' with no recorded positive
        of any label with k, carrying the positive's label vector; evaluated with the trainer-style per-label
        metric (no drop_last here since the file is synthetic).
Scores: partner-degree deg_j(known) (+ deg_j(new)=0), knn-degree (new drug gets the mean deg of its k nearest
        training drugs), knn-link (fraction of new drug's k nearest training drugs with a label-j train edge to k).
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "out/phase3/indep_verify_c1c2")
from common import NUM_ENT, NUM_REL, OUT, load_sets, load_twosides, morgan_bitvects, smiles_twosides, tanimoto_matrix

R, NE = NUM_REL["twosides"], NUM_ENT["twosides"]
D = load_twosides()
sets = load_sets("twosides")
train_set = sets["train"]
rng = np.random.default_rng(0)

# all recorded positives, any file, symmetric, per label
POSJ = np.zeros((NE, NE, R), dtype=np.uint8)
for s, (H, T, Y, P) in D.items():
    m = P == 1
    np.bitwise_or.at(POSJ, (H[m], T[m]), Y[m])
    np.bitwise_or.at(POSJ, (T[m], H[m]), Y[m])
ANYPOS = POSJ.any(2)

# train-only adjacency & degrees
A = np.zeros((NE, NE, R), dtype=np.uint8)
H, T, Y, P = D["train"]
m = P == 1
np.bitwise_or.at(A, (H[m], T[m]), Y[m]); np.bitwise_or.at(A, (T[m], H[m]), Y[m])
DEG = A.sum(1).astype(float)  # distinct partners per label (A is 0/1 per (d,x,j))

fps, _ = morgan_bitvects(smiles_twosides(), NE, 2, 2048)
train_drugs = np.array(sorted(train_set))
SIM = tanimoto_matrix(fps, list(range(NE)), list(train_drugs))
SIM[train_drugs, np.arange(len(train_drugs))] = -2
KNN = {k: {d: train_drugs[np.argsort(-SIM[d], kind="stable")[:k]] for d in range(NE)} for k in [5, 10, 20]}


def scores(Hs, Ts, which):
    out = np.zeros((len(Hs), R))
    for i, (h, t) in enumerate(zip(Hs, Ts)):
        known, new = (h, t) if h in train_set else (t, h)
        if which == "partner_degree":
            out[i] = DEG[h] + DEG[t]
        elif which.startswith("knn_degree"):
            k = int(which.split("k")[-1])
            out[i] = DEG[known] + DEG[KNN[k][new]].mean(0)
        elif which.startswith("knn_newside_degree"):  # new-drug side only
            k = int(which.split("k")[-1])
            out[i] = DEG[KNN[k][new]].mean(0)
        elif which.startswith("knn_link"):
            k = int(which.split("k")[-1])
            out[i] = A[KNN[k][new], known].mean(0)
    return out


def per_label(Y, P, S):
    roc, pr = [], []
    for j in range(R):
        w = np.where(Y[:, j] == 1)[0]
        if len(w) == 0:
            roc.append(0.0); pr.append(0.0); continue
        roc.append(roc_auc_score(P[w], S[w, j])); pr.append(average_precision_score(P[w], S[w, j]))
    return 100 * np.mean(roc), 100 * np.mean(pr)


SCORES = ["partner_degree", "knn_degree_k10", "knn_newside_degree_k10", "knn_link_k5", "knn_link_k10", "knn_link_k20"]
report = {}
for s, held in [("valid_S1", sets["valid"]), ("test_S1", sets["test"])]:
    H, T, Y, P = D[s]
    held_list = np.array(sorted(held))
    print(f"\n===== {s} =====")
    # ---------------- Part A ----------------
    cnt = Counter(); pos_known_deg, neg_known_deg = [], []
    neg_is_recorded_pos = 0; neg_label_overlap = 0
    for i in range(len(H) // 2):
        h, t, nh, nt = H[2 * i], T[2 * i], H[2 * i + 1], T[2 * i + 1]
        assert P[2 * i] == 1 and P[2 * i + 1] == 0 and (Y[2 * i] == Y[2 * i + 1]).all()
        new_pos_head = h not in train_set
        n, k = (h, t) if new_pos_head else (t, h)
        nn_, kk = (nh, nt) if nh not in train_set else (nt, nh)
        cnt[("new drug kept" if nn_ == n else "new drug replaced",
             "known drug kept" if kk == k else "known drug replaced",
             "new-drug position kept" if ((nh not in train_set) == new_pos_head) else "new-drug position flipped")] += 1
        neg_is_recorded_pos += int(ANYPOS[nh, nt])
        neg_label_overlap += int((POSJ[nh, nt] & Y[2 * i]).any())
        js = np.nonzero(Y[2 * i])[0]
        pos_known_deg.append(DEG[k, js].mean()); neg_known_deg.append(DEG[kk, js].mean())
    print("negative construction:", dict(cnt))
    print(f"negatives whose pair has ANY recorded positive: {neg_is_recorded_pos}; "
          f"with a recorded positive sharing one of the row's labels: {neg_label_overlap}")
    print(f"mean over rows of mean_j deg_j(known): positives {np.mean(pos_known_deg):.2f}, negatives {np.mean(neg_known_deg):.2f}")
    # how the negative's known drug is drawn: correlation of its frequency with overall train degree
    negk = Counter(); posk = Counter()
    for i in range(len(H) // 2):
        for (a, b), c in [((H[2 * i], T[2 * i]), posk), ((H[2 * i + 1], T[2 * i + 1]), negk)]:
            c[a if a in train_set else b] += 1
    totdeg = ANYPOS[:, train_drugs].sum(1)  # not used for scoring; descriptive
    tdeg = np.array([A[d].any(1).sum() for d in train_drugs])
    fneg = np.array([negk[d] for d in train_drugs]); fpos = np.array([posk[d] for d in train_drugs])
    print(f"Spearman(train degree, #times as known drug): positives {np.corrcoef(np.argsort(np.argsort(tdeg)), np.argsort(np.argsort(fpos)))[0,1]:.3f}, "
          f"negatives {np.corrcoef(np.argsort(np.argsort(tdeg)), np.argsort(np.argsort(fneg)))[0,1]:.3f}; distinct known drugs pos {len(posk)} neg {len(negk)}")
    rep = {"construction": {" / ".join(k): v for k, v in cnt.items()}, "neg_any_recorded_pos": neg_is_recorded_pos,
           "neg_label_overlap": neg_label_overlap}
    # original DDI-Ben negatives, for reference (all rows, per label, row labels)
    for sc in SCORES:
        S = scores(H, T, sc)
        rep[f"orig|{sc}"] = per_label(Y, P, S)
        print(f"  original negatives   {sc:24s} per-label ROC/PR = {rep[f'orig|{sc}'][0]:.2f}/{rep[f'orig|{sc}'][1]:.2f}")

    # ---------------- Part B: per-(row,label) partner-matched negatives ----------------
    posrows = np.where(P == 1)[0]
    rows_h, rows_t, rows_j, rows_p, rows_grp = [], [], [], [], []
    n_fail = 0
    for i in posrows:
        h, t = H[i], T[i]
        new_head = h not in train_set
        n, k = (h, t) if new_head else (t, h)
        for j in np.nonzero(Y[i])[0]:
            cand = held_list[(held_list != n) & (POSJ[held_list, k, j] == 0)]
            if len(cand) == 0:
                n_fail += 1; continue
            n2 = rng.choice(cand)
            rows_h += [h, n2 if new_head else k]; rows_t += [t, k if new_head else n2]
            rows_j += [j, j]; rows_p += [1, 0]
    rows_h, rows_t, rows_j, rows_p = map(np.array, (rows_h, rows_t, rows_j, rows_p))
    print(f"  matched (row,label) pairs built: {len(rows_p)//2}, failed (no eligible n'): {n_fail}")
    for sc in SCORES:
        Sfull = scores(rows_h, rows_t, sc)
        sj = Sfull[np.arange(len(rows_j)), rows_j]
        roc, pr = [], []
        for j in range(R):
            w = rows_j == j
            if w.sum() == 0:
                roc.append(0.0); pr.append(0.0); continue
            roc.append(roc_auc_score(rows_p[w], sj[w])); pr.append(average_precision_score(rows_p[w], sj[w]))
        pl = (100 * np.mean(roc), 100 * np.mean(pr))
        po = (100 * roc_auc_score(rows_p, sj), 100 * average_precision_score(rows_p, sj))
        ties = float(np.mean(sj[rows_p == 1] == sj[rows_p == 0]))
        rep[f"matched_rowlabel|{sc}"] = {"per_label": pl, "pooled": po, "frac_pos_neg_tied": ties}
        print(f"  matched (row,label)  {sc:24s} per-label ROC/PR = {pl[0]:.2f}/{pl[1]:.2f} | pooled {po[0]:.2f}/{po[1]:.2f} | pos==neg score in {100*ties:.1f}% of pairs")

    # ---------------- Part B2: row-level matched negatives (DDI-Ben file format) ----------------
    bh, bt, bY, bP = [], [], [], []
    n_fail2 = 0
    for i in posrows:
        h, t = H[i], T[i]
        new_head = h not in train_set
        n, k = (h, t) if new_head else (t, h)
        cand = held_list[(held_list != n) & (~ANYPOS[held_list, k])]
        if len(cand) == 0:
            n_fail2 += 1; continue
        n2 = rng.choice(cand)
        bh += [h, n2 if new_head else k]; bt += [t, k if new_head else n2]; bY += [Y[i], Y[i]]; bP += [1, 0]
    bh, bt, bY, bP = np.array(bh), np.array(bt), np.array(bY), np.array(bP)
    print(f"  row-level matched negatives built: {len(bP)//2}, failed: {n_fail2}")
    for sc in SCORES:
        S = scores(bh, bt, sc)
        pl = per_label(bY, bP, S)
        r, c = np.nonzero(bY)
        po = (100 * roc_auc_score(bP[r], S[r, c]), 100 * average_precision_score(bP[r], S[r, c]))
        rep[f"matched_row|{sc}"] = {"per_label": pl, "pooled": po}
        print(f"  matched (row-level)  {sc:24s} per-label ROC/PR = {pl[0]:.2f}/{pl[1]:.2f} | pooled {po[0]:.2f}/{po[1]:.2f}")
    report[s] = rep

json.dump(report, open(f"{OUT}/claim2_matched_results.json", "w"), indent=1, default=float)
print("saved")
