"""Independent set-based check: how many DDI-Ben TWOSIDES negatives are pairs with a recorded positive (any file, either orientation),
and how many share >=1 of the labels they are evaluated on."""
import sys; sys.path.insert(0, 'out/phase3/indep_verify_c1c2')
from collections import defaultdict
import numpy as np
from common import load_twosides
D = load_twosides(); pos = defaultdict(set)
for s, (H, T, Y, P) in D.items():
    for h, t, y, p in zip(H, T, Y, P):
        if p == 1:
            js = set(np.nonzero(y)[0].tolist()); pos[(h, t)] |= js; pos[(t, h)] |= js
for s in ['valid_S1', 'test_S1', 'valid_S2', 'test_S2', 'test_S0', 'train']:
    H, T, Y, P = D[s]; anyp = ov = ovlab = tot = 0
    for h, t, y, p in zip(H, T, Y, P):
        if p == 0:
            tot += 1
            if (h, t) in pos:
                anyp += 1; o = pos[(h, t)] & set(np.nonzero(y)[0].tolist())
                if o: ov += 1; ovlab += len(o)
    print(s, 'negatives', tot, 'recorded-positive pair', anyp, 'shares>=1 evaluated label', ov,
          '(row,label) negative entries that are recorded positives', ovlab, 'of', int(Y[P == 0].sum()))
