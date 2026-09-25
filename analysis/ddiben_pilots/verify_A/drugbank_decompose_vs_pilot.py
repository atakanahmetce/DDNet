"""Written AFTER my own numbers, to explain the gap to the pilot's 57.0/67.0/60.0.
Pilot (phase2/evalsci/pilot_typeprior.py 'KN') differs from the literal claim text in:
  (s) Dirichlet shrinkage of every role distribution toward the train prior: (counts + 1*prior)/(n+1)
  (w) new-drug distribution = similarity-WEIGHTED mean of neighbour COUNTS (x10), not an unweighted mean of
      normalised distributions; then (s) applied
  (p) neighbour pool = all 1367 train_set.txt ids (29 of which have no training triples)
This script toggles each ingredient in my own implementation."""
import numpy as np
from drugbank_main_effects import (T, train, train_drugs, train_set_file, MF, NENT, NREL, Hc, Tc, prior,
                                   tanimoto_binary, evaluate)

pool_all = np.array(sorted(train_set_file))
SPL = ['valid_S1', 'test_S1', 'valid_S2', 'test_S2', 'test_S0']
prior_s = np.bincount(train[:, 2], minlength=NREL) + 1.0; prior_s /= prior_s.sum()  # pilot's +1-smoothed prior


def build(shrink, agg, pool, weighted, k=10, pri=prior):
    sim = tanimoto_binary(MF, MF[pool])
    is_known = np.zeros(NENT, bool); is_known[train_drugs] = True
    out = []
    for C in (Hc, Tc):
        P = np.zeros((NENT, NREL))
        for d in range(NENT):
            if is_known[d] or d in train_set_file:
                c = C[d].copy()
            else:
                order = np.argsort(-sim[d], kind='stable')[:k]
                nb = pool[order]; w = sim[d, order] if weighted else np.ones(k)
                if agg == 'counts':
                    c = (w[:, None] * C[nb]).sum(0) / max(w.sum(), 1e-9) * k
                else:  # mean of normalised distributions (neighbours with 0 role count skipped)
                    rows = C[nb]; n = rows.sum(1); ok = n > 0
                    c = ((w[ok, None] * rows[ok] / n[ok, None]).sum(0) / max(w[ok].sum(), 1e-9)) if ok.any() else pri.copy()
            if shrink > 0:
                c = c + shrink * pri
            s = c.sum()
            P[d] = c / s if s > 0 else pri
        out.append(P)
    return out


def run(tag, pri=prior, **kw):
    Ph, Pt = build(pri=pri, **kw)
    cells = []
    for sp in SPL:
        rows = T[sp]
        p = (Ph[rows[:, 0]] * Pt[rows[:, 1]] / pri[None, :]).argmax(1)
        em, dd = evaluate(sp, p)
        cells.append(f"{sp}: {em['f1']:.2f}/{em['acc']:.2f}/{em['kappa']:.2f} [ddiben-eval {dd['f1']:.2f}/{dd['acc']:.2f}/{dd['kappa']:.2f}]")
    print(f'{tag:62s}\n    ' + '\n    '.join(cells))


if __name__ == '__main__':
    run('literal: no shrink, mean of dists, unweighted, pool=train drugs', shrink=0, agg='dists', pool=train_drugs, weighted=False)
    run('+ pool = train_set.txt ids', shrink=0, agg='dists', pool=pool_all, weighted=False)
    run('+ shrink 1*prior only', shrink=1, agg='dists', pool=train_drugs, weighted=False)
    run('+ weighted-count aggregation only', shrink=0, agg='counts', pool=train_drugs, weighted=True)
    run('+ unweighted-count aggregation only', shrink=0, agg='counts', pool=train_drugs, weighted=False)
    run('shrink + unweighted counts', shrink=1, agg='counts', pool=train_drugs, weighted=False)
    run('shrink + weighted counts, pool=train drugs', shrink=1, agg='counts', pool=train_drugs, weighted=True)
    run('PILOT-EQUIVALENT: shrink + weighted counts + pool=train_set, +1 prior', pri=prior_s, shrink=1, agg='counts', pool=pool_all, weighted=True)
