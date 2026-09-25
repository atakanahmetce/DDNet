"""Diagnostics on how DDI-Ben TWOSIDES S1/S2 negatives are formed (my own analysis)."""
import numpy as np
from collections import Counter
from scipy.stats import spearmanr, chisquare
from twosides_partner_degree import D, train_set, tot_partners, DEG, known_drug, SPLITS

allpos = set()
for s in SPLITS:
    hh, tt, _, PP = D[s]
    for a, b, p in zip(hh, tt, PP):
        if p == 1: allpos.add((a, b)); allpos.add((b, a))
for split in ['valid_S1', 'test_S1']:
    hh, tt, YY, PP = D[split]
    kd, kh, kt = known_drug(hh, tt)
    nd = np.where(kh, tt, hh)
    pos = PP == 1; neg = PP == 0
    pk, nk = kd[pos], kd[neg]; pn, nn = nd[pos], nd[neg]
    print(f'== {split}: {pos.sum()} pos / {neg.sum()} neg (row 2i = positive, row 2i+1 = its negative)')
    print('  negatives keeping the positive\'s known drug:', int((pk == nk).sum()), '| keeping its new drug:', int((pn == nn).sum()),
          '| keeping both:', int(((pk == nk) & (pn == nn)).sum()), '| keeping neither:', int(((pk != nk) & (pn != nn)).sum()))
    print('  known drug in the same slot (head/tail) as in positive:', int((kh[pos] == kh[neg]).sum()))
    print('  negative pairs that are positives somewhere in DDI-Ben twosides_random:', int(sum((a, b) in allpos for a, b in zip(hh[neg], tt[neg]))))
    print('  mean train total-partner count of known drug: pos %.1f  neg %.1f' % (tot_partners[pk].mean(), tot_partners[nk].mean()))
    print('  distinct known drugs: pos %d  neg %d  (train drugs %d)' % (len(set(pk)), len(set(nk)), len(train_set)))
    tr = np.array(sorted(train_set)); cp = Counter(pk); cn = Counter(nk)
    fp = np.array([cp.get(d, 0) for d in tr]); fn = np.array([cn.get(d, 0) for d in tr])
    print('  Spearman(train degree, #appearances as known drug): pos %.3f  neg %.3f' % (spearmanr(tot_partners[tr], fp)[0], spearmanr(tot_partners[tr], fn)[0]))
    print('  chi-square vs uniform over train drugs, negatives: p=%.3g ; positives: p=%.3g' % (chisquare(fn).pvalue, chisquare(fp).pvalue))
    cnn = Counter(nn); cpn = Counter(pn)
    print('  distinct new drugs: pos %d neg %d; max/min appearances of a new drug in negatives %d/%d, in positives %d/%d' % (len(cpn), len(cnn), max(cnn.values()), min(cnn.values()), max(cpn.values()), min(cpn.values())))
    # label-specific: fraction of (row,label) where known drug has zero training degree for that label
    Yv = np.array(YY)
    r, c = np.nonzero(Yv)
    z = DEG[kd[r], c] == 0
    print('  share of (row,label) cells with zero label-degree: pos %.3f  neg %.3f' % (z[PP[r] == 1].mean(), z[PP[r] == 0].mean()))
