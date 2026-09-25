"""Shared loaders / metrics / split utilities for the DDNet re-evaluation.

Read-only access to /home/user/DDNet/data. Nothing here writes into the repo.
"""
import os, re, json, time
import numpy as np
import pandas as pd
from sklearn import metrics

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, 'out'), exist_ok=True)

DS_DIR = {'D1': 'data_1', 'D2': 'data_2', 'MP': 'data_mp'}
DS_TAG = {'D1': 'cb1', 'D2': 'cb2', 'MP': 'mp'}


def names(ds):
    return [l.strip() for l in open(f'{DATA}/datasets/{DS_DIR[ds]}/names.txt') if l.strip()]


def sim(ds):
    return np.loadtxt(f'{DATA}/datasets/{DS_DIR[ds]}/sim_arr.txt')


def directed_labels(ds):
    """Set of directed (a,b) pairs with both drugs in names (exactly like creating_paths.get_label)."""
    s = set(names(ds))
    out = set()
    for l in open(f'{DATA}/datasets/{DS_DIR[ds]}/interactions.txt'):
        r = l.strip().split(',')
        if len(r) >= 2 and r[0] in s and r[1] in s:
            out.add((r[0], r[1]))
    return out


def label_matrix(ds, order=None, symmetric=True):
    order = order or names(ds)
    idx = {d: i for i, d in enumerate(order)}
    Y = np.zeros((len(order), len(order)), dtype=np.int8)
    for a, b in directed_labels(ds):
        if a in idx and b in idx:
            Y[idx[a], idx[b]] = 1
            if symmetric:
                Y[idx[b], idx[a]] = 1
    np.fill_diagonal(Y, 0)
    return Y


def emb_file(ds, hop=1, dim=128, p='1.2', q='1.2', variant=''):
    tag = DS_TAG[ds]
    if ds == 'MP':
        return f'{DATA}/n2v_embeddings/mp_embeddings/mp_{dim}_{p}_{q}.txt'
    v = f'{variant}_' if variant else ''
    return f'{DATA}/n2v_embeddings/{tag}_embeddings/{tag}_{v}hop{hop}_{dim}_{p}_{q}.txt'


def load_emb(path):
    """Returns (ordered list of ids as in file, dict id->np.array)."""
    ids, E = [], {}
    for row in open(path):
        row = re.split(' ', row.strip('\n'))
        ids.append(row[0])
        E[row[0]] = np.array(re.split(',', row[1]), dtype=float)
    return ids, E


def load_paths(ds, order):
    """n x n x 3 array of meta-path features aligned to `order`."""
    idx = {d: i for i, d in enumerate(order)}
    n = len(order)
    P = np.zeros((n, n, 3))
    miss = 0
    for l in open(f'{DATA}/meta_paths/{DS_TAG[ds]}_3_paths.txt'):
        r = l.strip('\n').split(',')
        a, b = r[0], r[1]
        if a in idx and b in idx:
            P[idx[a], idx[b]] = np.array(r[-1].split(' '), dtype=float)
        else:
            miss += 1
    return P


def smiles():
    dj = json.load(open(f'{DATA}/drugs.json'))
    return {d['id']: d['smiles'] for d in dj}


def morgan_fp(order, nbits=1024, radius=2):
    """ECFP4-like 1024-bit Morgan fingerprints (same call as model_process/smiles2vec.py)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
    sm = smiles()
    F = np.zeros((len(order), nbits), dtype=np.uint8)
    bad = []
    for i, d in enumerate(order):
        m = Chem.MolFromSmiles(sm.get(d, '')) if sm.get(d) else None
        if m is None:
            bad.append(d)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        F[i] = np.array(list(fp.ToBitString()), dtype=np.uint8)
    return F, bad


def drugbank_pairs():
    db = pd.read_csv(f'{DATA}/drug-drug_interaction_Drugbank.csv')
    return db


# --------------------------------------------------------------------- metrics
def paper_auc_hard(y, f):
    """Exactly evaluation_functions.classif_AUC / classif_AUPRC called with HARD labels f."""
    fpr, tpr, _ = metrics.roc_curve(y, f, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    pr, rc, _ = metrics.precision_recall_curve(y, f, pos_label=1)
    auprc = metrics.auc(rc, pr)
    return auc, auprc


def all_metrics(y, pred, score=None):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    d = dict(
        F1=metrics.f1_score(y, pred, zero_division=0),
        MCC=metrics.matthews_corrcoef(y, pred),
        Prec=metrics.precision_score(y, pred, zero_division=0),
        Rec=metrics.recall_score(y, pred, zero_division=0),
    )
    d['AUROC_hard'], d['AUPRC_hard'] = paper_auc_hard(y, pred)
    if score is not None and len(np.unique(y)) > 1:
        d['AUROC'] = metrics.roc_auc_score(y, score)
        d['AUPRC'] = metrics.average_precision_score(y, score)
    else:
        d['AUROC'] = np.nan
        d['AUPRC'] = np.nan
    d['pos_rate'] = y.mean()
    return d


def best_threshold(y, s, crit='mcc'):
    """Threshold on training scores maximising MCC (used for score-only baselines)."""
    qs = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
    best, bt = -2, qs[0]
    for t in qs:
        p = (s > t).astype(int)
        v = metrics.matthews_corrcoef(y, p) if crit == 'mcc' else metrics.f1_score(y, p)
        if v > best:
            best, bt = v, t
    return bt


def fmt(m, s=None, k=3):
    if s is None or np.isnan(s):
        return f'{m:.{k}f}'
    return f'{m:.{k}f}±{s:.{k}f}'


def summarise(rows, keys=('F1', 'MCC', 'Prec', 'AUROC', 'AUPRC', 'AUROC_hard', 'AUPRC_hard')):
    df = pd.DataFrame(rows)
    return {k: (df[k].mean(), df[k].std(ddof=1) if len(df) > 1 else np.nan) for k in keys if k in df}


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        print(f'[{self.name}] {time.time() - self.t:.1f}s', flush=True)
