"""Shared loaders for the independent re-implementation of the two DDI-Ben anchor claims.

Data are read directly from the (read-only) DDI-Bench clone; file formats follow
DDI_Ben/DDI_Ben/utils.py::load_data and data_process.py::Data_record.
"""
import json
import os
import pickle
import warnings

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

REPO = ("out/phase2/"
        "pilot2/repos/LARS-research_DDI-Bench/DDI_Ben/DDI_Ben/data")
OUT = "out/phase3/indep_verify_c1c2"
SPLITS = ["train", "valid_S0", "test_S0", "valid_S1", "test_S1", "valid_S2", "test_S2"]
NUM_ENT = {"drugbank": 1710, "twosides": 645}
NUM_REL = {"drugbank": 86, "twosides": 209}


def load_sets(ds):
    d = os.path.join(REPO, f"{ds}_random")
    return {s: set(int(x) for x in open(os.path.join(d, f"{s}_set.txt")).read().split())
            for s in ["train", "valid", "test"]}


def load_drugbank():
    """Returns dict split -> int array (n,3) of (head, tail, rel), in file order."""
    d = os.path.join(REPO, "drugbank_random")
    out = {}
    for s in SPLITS:
        out[s] = np.array([[int(x) for x in l.split()] for l in open(os.path.join(d, f"{s}.txt"))], dtype=np.int64)
    return out


def load_twosides():
    """Returns dict split -> (h, t, Y (n,209) uint8, p (n,)) in file order."""
    d = os.path.join(REPO, "twosides_random")
    out = {}
    for s in SPLITS:
        H, T, Y, P = [], [], [], []
        for l in open(os.path.join(d, f"{s}.txt")):
            h, t, r, p = l[:-1].split(" ") if l.endswith("\n") else l.split(" ")
            H.append(int(h)); T.append(int(t)); P.append(int(p))
            Y.append([int(v) for v in r.split(",")])
        out[s] = (np.array(H), np.array(T), np.array(Y, dtype=np.uint8), np.array(P))
    return out


def smiles_drugbank():
    return {int(k): v for k, v in json.load(open(os.path.join(REPO, "initial/drugbank/id2smiles.json"))).items()}


def smiles_twosides():
    cid2id = json.load(open(os.path.join(REPO, "initial/twosides/cid2id.json")))
    cid2smi = json.load(open(os.path.join(REPO, "initial/twosides/cid2smiles.json")))
    return {cid2id[c]: cid2smi[c] for c in cid2smi}


def morgan_bitvects(id2smi, n_ent, radius=2, nbits=2048):
    """RDKit Morgan bit vectors; None where the SMILES cannot be parsed (even unsanitized)."""
    fps = [None] * n_ent
    n_fail = 0
    for i in range(n_ent):
        smi = id2smi.get(i, "")
        m = Chem.MolFromSmiles(smi.strip()) if smi else None
        if m is None and smi:
            m = Chem.MolFromSmiles(smi.strip(), sanitize=False)
            if m is not None:
                try:
                    m.UpdatePropertyCache(strict=False)
                    Chem.GetSymmSSSR(m)
                except Exception:
                    m = None
        if m is None:
            n_fail += 1
            continue
        fps[i] = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
    return fps, n_fail


def provided_morgan_drugbank():
    """DDI-Ben's own DB_molecular_feats.pkl Morgan_Features (1024-d counts) -> binarized RDKit bitvects."""
    x = pickle.load(open(os.path.join(REPO, "initial/drugbank/DB_molecular_feats.pkl"), "rb"), encoding="utf-8")
    fps = []
    for v in x["Morgan_Features"]:
        v = np.asarray(v) > 0
        bv = DataStructs.ExplicitBitVect(len(v))
        for b in np.nonzero(v)[0]:
            bv.SetBit(int(b))
        fps.append(bv)
    return fps


def tanimoto_matrix(fps, rows, cols):
    """Tanimoto between fps[rows] and fps[cols]; rows with None fp get all -1 (=> no neighbours)."""
    S = np.full((len(rows), len(cols)), -1.0)
    col_fps = [fps[c] for c in cols]
    ok_cols = np.array([f is not None for f in col_fps])
    col_fps_ok = [f for f in col_fps if f is not None]
    for i, r in enumerate(rows):
        if fps[r] is None:
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[r], col_fps_ok))
        S[i, ok_cols] = sims
    return S
