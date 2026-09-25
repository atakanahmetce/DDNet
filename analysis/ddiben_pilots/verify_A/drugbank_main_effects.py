"""Independent re-implementation of CLAIM 1 (training-free 'main-effects' rule on DDI-Ben DrugBank).

Rule: for each role (head/tail) the drug's empirical relation-type distribution from TRAIN triples.
Known drug -> its own role distribution. New drug -> unweighted mean of the role distributions
of its k=10 most Tanimoto-similar training drugs (benchmark-provided Morgan features).
Prediction = argmax_r p(r|head) * p(r|tail) / p(r), p(r) = train relation frequency.

Metrics reproduced two ways:
  'emergnn'  : label = relation in the file row, all rows, sklearn f1(macro)/acc/kappa
               (EmerGNN/DrugBank/base_model.py::evaluate; TextDDI uses f1(average=None).mean(), identical)
  'ddiben'   : DDI_Ben/DDI_Ben/trainer.py::predict: label = argmax of multi-hot of ALL relations of
               (h,t) over train+valid_S*+test_S* (sr2o_all), test DataLoader drop_last=True with batch 128.
"""
import sys, json, pickle, itertools
from collections import defaultdict as ddict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

REPO = 'DDI-Bench/DDI_Ben/DDI_Ben'
DATA = REPO + '/data/drugbank_random'
NREL = 86
NENT = 1710
SPLITS = ['train', 'valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']

T = {s: np.loadtxt(f'{DATA}/{s}.txt', dtype=int).reshape(-1, 3) for s in SPLITS}
train = T['train']
train_drugs = np.array(sorted(set(train[:, :2].ravel().tolist())))
train_set_file = set(np.loadtxt(f'{DATA}/train_set.txt', dtype=int).ravel().tolist())
assert set(train_drugs.tolist()) <= train_set_file

# ---------------- fingerprints ----------------
with open(REPO + '/data/initial/drugbank/DB_molecular_feats.pkl', 'rb') as f:
    X = pickle.load(f, encoding='utf-8')
assert (np.asarray(X['Node ID'], dtype=int) == np.arange(NENT)).all()
MF = np.array([np.asarray(v, dtype=float) for v in X['Morgan_Features']])  # (1710, 1024) counts


def tanimoto_binary(A, B):
    A = (A > 0).astype(np.float64); B = (B > 0).astype(np.float64)
    inter = A @ B.T
    na = A.sum(1)[:, None]; nb = B.sum(1)[None, :]
    return inter / np.maximum(na + nb - inter, 1e-12)


def tanimoto_count(A, B):
    # generalized (min/max) Tanimoto on count vectors
    out = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        mn = np.minimum(A[i][None, :], B).sum(1)
        mx = np.maximum(A[i][None, :], B).sum(1)
        out[i] = mn / np.maximum(mx, 1e-12)
    return out


def rdkit_fp(n_bits=2048, radius=2):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
    smi = json.load(open(REPO + '/data/initial/drugbank/id2smiles.json'))
    F = np.zeros((NENT, n_bits))
    bad = []
    for i in range(NENT):
        m = Chem.MolFromSmiles(smi[str(i)].strip())
        if m is None:
            m = Chem.MolFromSmiles(smi[str(i)].strip(), sanitize=False)
            if m is not None:
                try:
                    m.UpdatePropertyCache(strict=False); Chem.GetSymmSSSR(m)
                except Exception:
                    m = None
        if m is None:
            bad.append(i); continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        F[i, list(fp.GetOnBits())] = 1
    return F, bad


# ---------------- role distributions ----------------
Hc = np.zeros((NENT, NREL)); Tc = np.zeros((NENT, NREL))
np.add.at(Hc, (train[:, 0], train[:, 2]), 1)
np.add.at(Tc, (train[:, 1], train[:, 2]), 1)
prior = np.bincount(train[:, 2], minlength=NREL).astype(float); prior /= prior.sum()


def norm_rows(C, alpha=0.0):
    C = C + alpha
    s = C.sum(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.where(s > 0, C / np.where(s > 0, s, 1), np.nan)
    return P  # nan rows = drug never seen in that role


def build_role_dists(sim, k=10, alpha=0.0, weighted=False, zero_role='prior', neigh_zero='skip'):
    """Return Ph, Pt (NENT x NREL) usable for any drug id.
    sim: NENT x len(train_drugs) similarity.
    zero_role: what a KNOWN drug that never occurred in the role gets: 'prior' | 'other_role'
    neigh_zero: neighbours with no occurrence in the role: 'skip' (average over the others) | 'prior'."""
    Ph_emp = norm_rows(Hc, alpha); Pt_emp = norm_rows(Tc, alpha)
    Ph = np.full((NENT, NREL), np.nan); Pt = np.full((NENT, NREL), np.nan)
    is_train = np.zeros(NENT, bool); is_train[train_drugs] = True
    # known drugs
    for P, Pemp, Poth in ((Ph, Ph_emp, Pt_emp), (Pt, Pt_emp, Ph_emp)):
        P[is_train] = Pemp[is_train]
        miss = is_train & np.isnan(P).any(1)
        if zero_role == 'prior':
            P[miss] = prior
        elif zero_role == 'other_role':
            P[miss] = Poth[miss]
    # new drugs via kNN over train drugs
    new = np.where(~is_train)[0]
    stats = {'n_new': len(new), 'neigh_role_missing': 0, 'all_neigh_missing': 0}
    for d in new:
        s = sim[d]
        order = np.argsort(-s, kind='stable')[:k]
        nb = train_drugs[order]; w = s[order] if weighted else np.ones(k)
        for P, Pemp in ((Ph, Ph_emp), (Pt, Pt_emp)):
            rows = Pemp[nb]
            ok = ~np.isnan(rows).any(1)
            stats['neigh_role_missing'] += int((~ok).sum())
            if neigh_zero == 'prior':
                rows = np.where(ok[:, None], rows, prior[None, :]); ok = np.ones(k, bool)
            if ok.sum() == 0:
                stats['all_neigh_missing'] += 1
                P[d] = prior
            else:
                ww = w[ok]
                P[d] = (rows[ok] * ww[:, None]).sum(0) / max(ww.sum(), 1e-12)
    return Ph, Pt, stats


def predict(rows, Ph, Pt):
    S = Ph[rows[:, 0]] * Pt[rows[:, 1]] / prior[None, :]
    allzero = (S.max(1) <= 0).sum()
    return S.argmax(1), int(allzero)


# DDI_Ben trainer label (sr2o_all multi-hot argmax -> smallest relation index)
sr2o_all = ddict(set)
for s in SPLITS:
    for h, t, r in T[s]:
        sr2o_all[(h, t)].add(int(r))


def metrics(y, p):
    return dict(f1=100 * f1_score(y, p, average='macro'), acc=100 * accuracy_score(y, p),
                kappa=100 * cohen_kappa_score(y, p), n=int(len(y)))


def evaluate(split, pred):
    rows = T[split]
    y_file = rows[:, 2]
    em = metrics(y_file, pred)
    y_ddi = np.array([min(sr2o_all[(h, t)]) for h, t, _ in rows])
    n_keep = (len(rows) // 128) * 128
    dd = metrics(y_ddi[:n_keep], pred[:n_keep])
    return em, dd


def run(tag, sim, **kw):
    Ph, Pt, st = build_role_dists(sim, **kw)
    out = {'tag': tag, 'cfg': kw, 'stats': st}
    for split in ['valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']:
        pred, az = predict(T[split], Ph, Pt)
        em, dd = evaluate(split, pred)
        out[split] = {'emergnn': em, 'ddiben': dd, 'all_zero_score_rows': az}
    return out


def fmt(m):
    return f"F1 {m['f1']:.2f} / Acc {m['acc']:.2f} / K {m['kappa']:.2f} (n={m['n']})"


if __name__ == '__main__':
    results = []
    sim_bin = tanimoto_binary(MF, MF[train_drugs])
    configs = [
        ('PRIMARY bench-Morgan1024 binary-Tanimoto, k=10, unweighted, no smoothing', sim_bin, dict(k=10)),
        ('bench-Morgan binary, zero_role=other_role', sim_bin, dict(k=10, zero_role='other_role')),
        ('bench-Morgan binary, neigh_zero=prior', sim_bin, dict(k=10, neigh_zero='prior')),
        ('bench-Morgan binary, similarity-weighted', sim_bin, dict(k=10, weighted=True)),
        ('bench-Morgan binary, Laplace alpha=1', sim_bin, dict(k=10, alpha=1.0)),
        ('bench-Morgan binary, alpha=0.01', sim_bin, dict(k=10, alpha=0.01)),
    ]
    sim_cnt = tanimoto_count(MF, MF[train_drugs])
    configs.append(('bench-Morgan COUNT (min/max) Tanimoto, k=10', sim_cnt, dict(k=10)))
    F, bad = rdkit_fp()
    print('RDKit Morgan r2/2048: unparsable ids', bad)
    sim_rd = tanimoto_binary(F, F[train_drugs])
    configs.append(('RDKit Morgan r2 2048-bit binary Tanimoto, k=10', sim_rd, dict(k=10)))
    for kk in (1, 5, 20):
        configs.append((f'bench-Morgan binary, k={kk}', sim_bin, dict(k=kk)))
    for tag, sim, kw in configs:
        r = run(tag, sim, **kw)
        results.append(r)
        print('\n==', tag, r['stats'])
        for split in ['valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']:
            print(f"  {split:9s} EmerGNN-eval: {fmt(r[split]['emergnn'])} | DDI_Ben-trainer-eval: {fmt(r[split]['ddiben'])} | all-zero rows {r[split]['all_zero_score_rows']}")
    json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else 'out/phase3/anchor_reverify/drugbank_results.json', 'w'), indent=1)
