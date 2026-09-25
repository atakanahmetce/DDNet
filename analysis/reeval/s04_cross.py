"""Task 4: cross-dataset D1 -> D2 ('new drug' experiment of the paper) + diagnosis of the
per-ego-subgraph node2vec embeddings (alignment across datasets, comparability within a dataset).
env: RF_JOBS, TORCH_THREADS
"""
import os, json, time
import numpy as np, networkx as nx
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.manifold import SpectralEmbedding
import torch
from common import *
import nets

RF_JOBS = int(os.environ.get('RF_JOBS', 1))
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 1)))
res = {}


def ordered_matrix(ds, hop):
    ids, E = load_emb(emb_file(ds, hop))
    n = len(ids); Em = np.array([E[d] for d in ids])
    P = load_paths(ds, ids)
    Yd = label_matrix(ds, ids, symmetric=False)
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing='ij'); I = I.ravel(); J = J.ravel()
    X = np.hstack([Em[I], Em[J], P[I, J]]).astype(np.float32)
    return ids, Em, X, Yd[I, J].astype(int), I, J


# ------------------------------------------------------------------ 1. cross-dataset as in paper
shared = sorted(set(names('D1')) & set(names('D2')))
print('shared drugs', len(shared))
for hop in [1, 2]:
    ids1, E1, X1, y1, I1, J1 = ordered_matrix('D1', hop)
    ids2, E2, X2, y2, I2, J2 = ordered_matrix('D2', hop)
    sh = np.array([d in set(ids1) for d in ids2])
    ptype = np.where(sh[I2] & sh[J2], 'both_shared', np.where(sh[I2] | sh[J2], 'one_shared', 'none_shared'))
    X1n, X2n = Normalizer().fit_transform(X1), Normalizer().fit_transform(X2)
    scores = {}
    with threadpool_limits(1):
        m = RandomForestClassifier(100, n_jobs=RF_JOBS, class_weight='balanced', random_state=0).fit(X1n, y1)
        scores['DDNet RF'] = (m.predict(X2n), m.predict_proba(X2n)[:, 1])
        m = HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=0).fit(X1n, y1)
        scores['DDNet GB(HGB)'] = (m.predict(X2n), m.predict_proba(X2n)[:, 1])
    m = nets.train_orig(X1n, y1)
    scores['DDNet FFNN-orig'] = nets.predict_orig(m, X2n)
    sc = StandardScaler().fit(X1)
    m = nets.train_fixed(sc.transform(X1), y1)
    scores['DDNet FFNN-fixed'] = nets.predict_fixed(m, sc.transform(X2))
    with threadpool_limits(1):
        m = HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=0).fit(X1[:, -3:], y1)
        scores['Path only (HGB)'] = (m.predict(X2[:, -3:]), m.predict_proba(X2[:, -3:])[:, 1])
        m = HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=0).fit(X1[:, :-3], y1)
        scores['Embeddings only (HGB)'] = (m.predict(X2[:, :-3]), m.predict_proba(X2[:, :-3])[:, 1])
    # degree learned on D1, transferred to D2 by drug identity (only shared drugs have one)
    n1 = len(ids1)
    cnt = np.bincount(I1, minlength=n1) + np.bincount(J1, minlength=n1)
    pc = np.bincount(I1, weights=y1, minlength=n1) + np.bincount(J1, weights=y1, minlength=n1)
    r1 = dict(zip(ids1, pc / cnt))
    r2 = np.array([r1.get(d, y1.mean()) for d in ids2])
    st = np.array(list(r1.values()))
    s_tr = (pc / cnt)[I1] * (pc / cnt)[J1]
    t = best_threshold(y1, s_tr)
    s_te = r2[I2] * r2[J2]
    scores['Degree from D1 (product)'] = ((s_te > t).astype(int), s_te)
    S1m, S2m = sim('D1'), sim('D2')
    p1 = {d: i for i, d in enumerate(names('D1'))}; p2 = {d: i for i, d in enumerate(names('D2'))}
    S1x = S1m[np.ix_([p1[d] for d in ids1], [p1[d] for d in ids1])]
    S2x = S2m[np.ix_([p2[d] for d in ids2], [p2[d] for d in ids2])]
    t = best_threshold(y1, S1x[I1, J1])
    scores['Raw similarity S'] = ((S2x[I2, J2] > t).astype(int), S2x[I2, J2])
    scores['All-positive'] = (np.ones(len(y2), int), np.zeros(len(y2)))
    out = {}
    for k, (h, s) in scores.items():
        out[k] = {'all': all_metrics(y2, h, s)}
        for pt in ['both_shared', 'one_shared', 'none_shared']:
            msk = ptype == pt
            out[k][pt] = all_metrics(y2[msk], h[msk], s[msk])
        print(f'hop{hop} {k:28s} ' + '  '.join(f"{pt}: F1={v['F1']:.3f} MCC={v['MCC']:.3f} AUROC={v['AUROC']:.3f}"
                                                 for pt, v in out[k].items()), flush=True)
    out['_sizes'] = {pt: int((ptype == pt).sum()) for pt in ['both_shared', 'one_shared', 'none_shared']}
    res[f'cross_hop{hop}'] = out

# ------------------------------------------------------------------ 2. alignment of shared-drug embeddings
rng = np.random.RandomState(0)


def unit(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


for hop in [1, 2]:
    _, E1d = load_emb(emb_file('D1', hop)); _, E2d = load_emb(emb_file('D2', hop))
    A = unit(np.array([E1d[d] for d in shared])); B = unit(np.array([E2d[d] for d in shared]))
    same = np.sum(A * B, 1)
    C = A @ B.T
    diff = C[~np.eye(len(shared), dtype=bool)]
    within1 = (A @ A.T)[~np.eye(len(shared), dtype=bool)]
    # Procrustes (full) residual vs permutation null
    def resid(Aa, Bb):
        R, _ = orthogonal_procrustes(Aa, Bb)
        return np.linalg.norm(Aa @ R - Bb) / np.linalg.norm(Bb)
    r_true = resid(A, B)
    r_null = np.array([resid(A, B[rng.permutation(len(B))]) for _ in range(200)])
    # held-out: fit on half, evaluate matched vs mismatched cosine on other half (100 random halvings)
    ho_same, ho_diff, ho_rank = [], [], []
    for _ in range(100):
        p = rng.permutation(len(shared)); tr, te = p[:23], p[23:]
        R, _ = orthogonal_procrustes(A[tr], B[tr])
        Ct = unit(A[te] @ R) @ B[te].T
        ho_same.append(np.diag(Ct).mean()); ho_diff.append(Ct[~np.eye(len(te), dtype=bool)].mean())
        ho_rank.append(np.mean([(Ct[i] >= Ct[i, i]).sum() for i in range(len(te))]))   # 1 = correct match ranked first
    res[f'align_hop{hop}'] = dict(
        cos_same_drug_mean=float(same.mean()), cos_same_drug_median=float(np.median(same)),
        cos_same_drug_min=float(same.min()), cos_same_drug_max=float(same.max()),
        cos_different_drugs_mean=float(diff.mean()), cos_within_D1_mean=float(within1.mean()),
        mw_p_same_vs_diff=float(stats.mannwhitneyu(same, diff, alternative='greater').pvalue),
        procrustes_resid_true=float(r_true), procrustes_resid_null_mean=float(r_null.mean()),
        procrustes_resid_null_p05=float(np.quantile(r_null, 0.05)),
        procrustes_perm_p=float((np.sum(r_null <= r_true) + 1) / (len(r_null) + 1)),
        heldout_cos_matched=float(np.mean(ho_same)), heldout_cos_mismatched=float(np.mean(ho_diff)),
        heldout_mean_rank_of_true_match=float(np.mean(ho_rank)), heldout_rank_chance=float((23 + 1) / 2))
    print(f'align hop{hop}', json.dumps(res[f'align_hop{hop}'], indent=1), flush=True)

# ------------------------------------------------------------------ 3. comparability within a dataset
def nn_overlap(M1, M2, k=10):
    n = len(M1); ov = []
    for i in range(n):
        a = [j for j in np.argsort(-M1[i]) if j != i][:k]
        b = [j for j in np.argsort(-M2[i]) if j != i][:k]
        ov.append(len(set(a) & set(b)))
    return float(np.mean(ov)), float(k * k / (n - 1))


for ds, hops in [('D1', [1, 2]), ('D2', [1, 2]), ('MP', [1])]:
    nm = names(ds); n = len(nm)
    S = sim(ds); Ss = (S + S.T) / 2
    Y = label_matrix(ds, nm)
    iu = np.triu_indices(n, 1)
    G = nx.read_edgelist(f'{DATA}/edgelists/{DS_TAG[ds]}_edgelist_sim3.edgelist', nodetype=str, data=(('weight', float),))
    G.remove_edges_from(nx.selfloop_edges(G)); G.add_nodes_from(nm)
    Adj = nx.to_numpy_array(G, nodelist=nm)
    sdeg = Adj.astype(bool).sum(1); ddeg = Y.sum(1)
    # global spectral embedding of the same weighted graph for comparison
    se = SpectralEmbedding(n_components=min(32, n - 2), affinity='precomputed', random_state=0).fit_transform(Adj + 1e-6)
    Cse = unit(se) @ unit(se).T
    for hop in hops:
        _, Ed = load_emb(emb_file(ds, hop))
        Em = np.array([Ed[d] for d in nm])
        Cm = unit(Em) @ unit(Em).T
        norms = np.linalg.norm(Em, axis=1)
        d = dict(
            spearman_cosEmb_vs_S=float(stats.spearmanr(Cm[iu], Ss[iu])[0]),
            spearman_cosEmb_vs_DDIlabel=float(stats.spearmanr(Cm[iu], Y[iu])[0]),
            auroc_cosEmb_for_DDI=float(metrics.roc_auc_score(Y[iu], Cm[iu])),
            nn10_overlap_emb_vs_S=nn_overlap(Cm, Ss),
            mean_offdiag_cos=float(Cm[iu].mean()),
            spearman_norm_vs_simdegree=float(stats.spearmanr(norms, sdeg)[0]),
            spearman_norm_vs_DDIdegree=float(stats.spearmanr(norms, ddeg)[0]),
            global_spectral_spearman_cos_vs_S=float(stats.spearmanr(Cse[iu], Ss[iu])[0]),
            global_spectral_nn10_overlap_vs_S=nn_overlap(Cse, Ss),
        )
        res[f'comparability_{ds}_hop{hop}'] = d
        print(ds, hop, json.dumps(d), flush=True)
    if ds != 'MP':
        _, Ea = load_emb(emb_file(ds, 1)); _, Eb = load_emb(emb_file(ds, 2))
        c = [float(unit(Ea[x][None])[0] @ unit(Eb[x][None])[0]) for x in nm]
        res[f'hop1_vs_hop2_same_drug_cos_{ds}'] = float(np.mean(c))

json.dump(res, open(f'{OUT}/out/cross.json', 'w'), indent=1, default=float)
