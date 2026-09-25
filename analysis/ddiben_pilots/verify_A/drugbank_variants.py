"""Sensitivity of CLAIM-1 numbers to under-specified implementation details of the rule.
Imports data/metric helpers from drugbank_main_effects.py (my own file)."""
import numpy as np
from drugbank_main_effects import (T, train, train_drugs, MF, NENT, NREL, Hc, Tc, prior, tanimoto_binary,
                                   evaluate, norm_rows)

sim = tanimoto_binary(MF, MF[train_drugs])
is_train = np.zeros(NENT, bool); is_train[train_drugs] = True
new = np.where(~is_train)[0]
SPL = ['valid_S1', 'test_S1', 'valid_S2', 'test_S2']


def dists(agg='mean_dist', pool='all', rolespec=True, known_zero='prior', k=10):
    Ph_emp = norm_rows(Hc); Pt_emp = norm_rows(Tc)
    Pall = norm_rows(Hc + Tc)
    Ph = np.full((NENT, NREL), np.nan); Pt = np.full((NENT, NREL), np.nan)
    for P, Pemp in ((Ph, Ph_emp), (Pt, Pt_emp)):
        P[is_train] = Pemp[is_train]
        miss = is_train & np.isnan(P).any(1)
        P[miss] = prior if known_zero == 'prior' else Pall[miss]
    for d in new:
        for P, C, Pemp in ((Ph, Hc, Ph_emp), (Pt, Tc, Pt_emp)):
            if not rolespec:
                C = Hc + Tc; Pemp = Pall
            cand = np.arange(len(train_drugs))
            if pool == 'role_present':
                cand = cand[C[train_drugs].sum(1) > 0]
            order = cand[np.argsort(-sim[d, cand], kind='stable')[:k]]
            nb = train_drugs[order]
            if agg == 'mean_dist':
                rows = Pemp[nb]; ok = ~np.isnan(rows).any(1)
                P[d] = rows[ok].mean(0) if ok.any() else prior
            elif agg == 'sum_counts':
                c = C[nb].sum(0)
                P[d] = c / c.sum() if c.sum() > 0 else prior
    return Ph, Pt


def pred_rows(rows, Ph, Pt, zero_fallback='argmax0', eps=0.0):
    ph = Ph[rows[:, 0]] + eps; pt = Pt[rows[:, 1]] + eps
    S = ph * pt / prior[None, :]
    p = S.argmax(1)
    z = S.max(1) <= 0
    if zero_fallback == 'sum':
        p[z] = ((ph + pt)[z]).argmax(1)
    elif zero_fallback == 'prior':
        p[z] = prior.argmax()
    return p


def show(tag, Ph, Pt, **kw):
    s = []
    for sp in SPL:
        em, dd = evaluate(sp, pred_rows(T[sp], Ph, Pt, **kw))
        s.append(f"{sp}: {em['f1']:.2f}/{em['acc']:.2f}/{em['kappa']:.2f}")
    print(f'{tag:70s} ' + ' | '.join(s))


if __name__ == '__main__':
    print('EmerGNN-style eval (file labels, all rows); F1/Acc/Kappa')
    base = dists()
    show('primary (mean of role dists, pool=all train drugs, argmax0 fallback)', *base)
    show('primary + all-zero fallback argmax(ph+pt)', *base, zero_fallback='sum')
    show('primary + all-zero fallback most frequent class', *base, zero_fallback='prior')
    show('primary + eps=1e-12 on probabilities', *base, eps=1e-12)
    show('primary + eps=1e-6', *base, eps=1e-6)
    show('primary + eps=1e-3', *base, eps=1e-3)
    show('known zero-role -> pooled (head+tail) dist', *dists(known_zero='pooled'))
    show('neighbour pool restricted to drugs present in that role', *dists(pool='role_present'))
    show('neighbour aggregation = summed counts, then normalise', *dists(agg='sum_counts'))
    show('sum_counts + role_present pool', *dists(agg='sum_counts', pool='role_present'))
    show('NOT role-specific for new drug (pooled head+tail of neighbours)', *dists(rolespec=False))
    show('NOT role-specific, sum_counts', *dists(rolespec=False, agg='sum_counts'))
