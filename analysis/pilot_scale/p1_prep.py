"""Pilot 1 - step 1: per-drug feature cache + drug samples.

Reads (read-only) /home/user/DDNet/data/{drug-drug_interaction_Drugbank.csv, drugs.json}.
Writes out/prep.npz (pool drugs, ECFP4, descriptors, target counts, target incidence),
out/unordered_pairs.npy (global unordered DDI pairs, as pool/global indices), out/samples.json.
Seeds: U1000 -> RandomState(0), H300 -> RandomState(1), U300 -> RandomState(2),
extra uniform-300 replicates U300b/U300c -> RandomState(3)/(4).
"""
import json, os
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')

db = pd.read_csv(f'{DATA}/drug-drug_interaction_Drugbank.csv')
und = pd.DataFrame(np.sort(db.values, axis=1), columns=['a', 'b']).drop_duplicates()
und = und[und.a != und.b]
all_ddi = sorted(set(und.a) | set(und.b))
gdeg = pd.concat([und.a, und.b]).value_counts()
print(f'DDI rows {len(db)}, unordered pairs {len(und)}, drugs {len(all_ddi)}')

drugs = json.load(open(f'{DATA}/drugs.json'))
info = {d['id']: d for d in drugs}
with_smiles = [d for d in all_ddi if d in info and info[d].get('smiles')]
print(f'DDI drugs in drugs.json with SMILES: {len(with_smiles)}')

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
pool, F, D, TC, TG, bad = [], [], [], [], [], []
for d in with_smiles:
    m = Chem.MolFromSmiles(info[d]['smiles'])
    if m is None:
        bad.append(d); continue
    pool.append(d)
    F.append(gen.GetFingerprintAsNumPy(m).astype(np.uint8))
    D.append([Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcNumRotatableBonds(m),
              rdMolDescriptors.CalcNumRings(m)])
    acc = sorted(set(info[d].get('accessions') or []))
    TC.append(len(acc)); TG.append(acc)
print(f'RDKit parse failures: {len(bad)} -> pool {len(pool)}')
F = np.array(F); D = np.array(D, dtype=float); TC = np.array(TC)
prots = sorted(set(p for t in TG for p in t)); pidx = {p: i for i, p in enumerate(prots)}
ti_r = np.concatenate([[i] * len(t) for i, t in enumerate(TG)]).astype(int)
ti_c = np.array([pidx[p] for t in TG for p in t], dtype=int)
g_all = np.array([gdeg.get(d, 0) for d in all_ddi], dtype=float)
g_pool = np.array([gdeg.get(d, 0) for d in pool], dtype=float)
q75 = np.percentile(g_all, 75)
print(f'global degree: median {np.median(g_all):.0f}, q75 {q75:.0f} (over {len(all_ddi)} DDI drugs); '
      f'pool median {np.median(g_pool):.0f}; pool drugs with 0 targets: {(TC == 0).sum()}')

# global pair list in pool index space (only pairs with both drugs in pool) + all-drug index for external degree
gi = {d: i for i, d in enumerate(all_ddi)}
P_all = np.stack([und.a.map(gi).values, und.b.map(gi).values], 1).astype(np.int32)
np.save(f'{OUT}/unordered_pairs_global.npy', P_all)
pool_gidx = np.array([gi[d] for d in pool])

rng_specs = {'U1000': (0, 1000, 'uniform'), 'H300': (1, 300, 'hub'), 'U300': (2, 300, 'uniform'),
             'U300b': (3, 300, 'uniform'), 'U300c': (4, 300, 'uniform')}
top = np.where(g_pool >= q75)[0]
print(f'pool drugs in top global-degree quartile (deg >= {q75:.0f}): {len(top)}')
samples = {}
for name, (seed, k, kind) in rng_specs.items():
    rs = np.random.RandomState(seed)
    src = np.arange(len(pool)) if kind == 'uniform' else top
    samples[name] = sorted(rs.choice(src, k, replace=False).tolist())
json.dump({'pool_size': len(pool), 'q75_global_degree': float(q75), 'n_top_quartile_in_pool': int(len(top)),
           'rdkit_failures': bad, 'n_ddi_drugs': len(all_ddi), 'n_unordered_pairs': int(len(und)),
           'samples': {k: [pool[i] for i in v] for k, v in samples.items()}}, open(f'{OUT}/samples.json', 'w'))
np.savez_compressed(f'{OUT}/prep.npz', pool=np.array(pool), F=F, D=D, TC=TC, ti_r=ti_r, ti_c=ti_c,
                    n_prot=len(prots), g_pool=g_pool, g_all=g_all, pool_gidx=pool_gidx,
                    all_ddi=np.array(all_ddi), **{f'S_{k}': np.array(v) for k, v in samples.items()})
for k, v in samples.items():
    gv = g_pool[v]
    pct = (g_all[None, :] <= gv[:, None]).mean(1) * 100
    print(f'{k}: n={len(v)} median global degree {np.median(gv):.0f} (median pct {np.median(pct):.0f}), '
          f'median targets {np.median(TC[v]):.0f}')
