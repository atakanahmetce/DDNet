"""Task 5: characterise D1 / D2 / MP (density, degrees, DrugBank context, similarity graph,
whether sim_arr carries DDI information)."""
import json
import numpy as np, pandas as pd, networkx as nx
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LinearRegression
from common import *

res = {}
db = drugbank_pairs()
# global DrugBank degree (unique unordered partners)
und = pd.DataFrame(np.sort(db.values, axis=1), columns=['a', 'b']).drop_duplicates()
und = und[und.a != und.b]
gdeg = pd.concat([und.a, und.b]).value_counts()
print('DrugBank file: drugs', len(gdeg), 'unordered pairs', len(und), 'median degree', gdeg.median())
res['drugbank'] = dict(n_drugs=int(len(gdeg)), n_pairs=int(len(und)), median_deg=float(gdeg.median()),
                       mean_deg=float(gdeg.mean()), q75=float(gdeg.quantile(.75)), q90=float(gdeg.quantile(.9)))

sets = {ds: set(names(ds)) for ds in DS_DIR}
res['overlap'] = {'D1&D2': len(sets['D1'] & sets['D2']), 'D1&MP': len(sets['D1'] & sets['MP']),
                  'D2&MP': len(sets['D2'] & sets['MP'])}
print(res['overlap'])

for ds in ['D1', 'D2', 'MP']:
    nm = names(ds); n = len(nm)
    Y = label_matrix(ds)                      # symmetrised, zero diagonal
    iu = np.triu_indices(n, 1)
    y = Y[iu]
    dens = y.mean()
    ddeg = Y.sum(1)
    r = dict(n=n, pos_unordered=int(y.sum()), density=float(dens),
             allpos_F1=float(2 * dens / (1 + dens)),
             ds_deg_min=int(ddeg.min()), ds_deg_median=float(np.median(ddeg)), ds_deg_max=int(ddeg.max()),
             ds_isolated=int((ddeg == 0).sum()))
    # global degree of chosen drugs
    g = np.array([gdeg.get(d, 0) for d in nm], dtype=float)
    r['gdeg_median'] = float(np.median(g)); r['gdeg_mean'] = float(g.mean())
    r['gdeg_median_pct_rank'] = float(stats.percentileofscore(gdeg.values, np.median(g)))
    r['frac_in_global_top25pct'] = float((g >= gdeg.quantile(.75)).mean())
    r['spearman_dsdeg_vs_gdeg'] = float(stats.spearmanr(ddeg, g)[0])
    # transductive "oracle" degree scores (uses the full label matrix -> upper bound of degree signal)
    r['AUROC_degprod_oracle'] = float(roc_auc_score(y, np.outer(ddeg, ddeg)[iu]))
    r['AUROC_globaldeg_prod'] = float(roc_auc_score(y, np.outer(g, g)[iu]))
    # similarity
    S = sim(ds)
    Ss = (S + S.T) / 2
    r['sim_asym_maxdiff'] = float(np.abs(S - S.T).max())
    r['sim_diag_min'] = float(S.diagonal().min())
    r['AUROC_S'] = float(roc_auc_score(y, Ss[iu]))
    r['AUPRC_S'] = float(average_precision_score(y, Ss[iu]))
    P = load_paths(ds, nm)
    for k in range(3):
        r[f'AUROC_path{k+1}'] = float(roc_auc_score(y, P[..., k][iu]))
    # Morgan cosine recomputed from SMILES; is sim_arr = a*cos + b*X ?
    F, bad = morgan_fp(nm)
    Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    C = Fn @ Fn.T
    r['AUROC_morgan_cos'] = float(roc_auc_score(y, C[iu]))
    r['corr_S_vs_morgancos'] = float(np.corrcoef(Ss[iu], C[iu])[0, 1])
    lr = LinearRegression().fit(C[iu].reshape(-1, 1), Ss[iu])
    r['S_on_cos_slope'], r['S_on_cos_intercept'] = float(lr.coef_[0]), float(lr.intercept_)
    X2 = 2 * S - C                           # implied 2nd (SIMCOMP) component if S = 0.5*cos + 0.5*X
    r['implied_X_range'] = [float(X2[~np.eye(n, dtype=bool)].min()), float(X2[~np.eye(n, dtype=bool)].max())]
    r['implied_X_diag_values'] = sorted(set(np.round(X2.diagonal(), 3).tolist()))[:6]
    Xs = (X2 + X2.T) / 2
    r['AUROC_implied_X'] = float(roc_auc_score(y, Xs[iu]))
    # DDI-profile Jaccard similarity (label derived) vs S and vs Morgan cosine
    inter = Y.astype(float) @ Y.T.astype(float)
    un = ddeg[:, None] + ddeg[None, :] - inter
    J = np.where(un > 0, inter / np.maximum(un, 1), 0)
    r['spearman_S_vs_DDIprofileJaccard'] = float(stats.spearmanr(Ss[iu], J[iu])[0])
    r['spearman_cos_vs_DDIprofileJaccard'] = float(stats.spearmanr(C[iu], J[iu])[0])
    # similarity graph as built by the repo (edgelist sim3, threshold 0.3, undirected, last weight wins)
    G = nx.read_edgelist(f'{DATA}/edgelists/{DS_TAG[ds]}_edgelist_sim3.edgelist', nodetype=str,
                         data=(('weight', float),))
    G.remove_edges_from(nx.selfloop_edges(G))
    G.add_nodes_from(nm)
    for thr, lab in [(0.3, 'thr0.3'), (0.5, 'thr0.5')]:
        H = nx.Graph(); H.add_nodes_from(nm)
        H.add_edges_from((a, b, d) for a, b, d in G.edges(data=True) if d['weight'] > thr)
        degs = np.array([H.degree(v) for v in nm])
        ego1 = np.array([len(list(H.neighbors(v))) + 1 for v in nm])
        ego2 = np.array([len(nx.single_source_shortest_path_length(H, v, cutoff=2)) for v in nm])
        r[f'G_{lab}'] = dict(edges=H.number_of_edges(), density=H.number_of_edges() / (n * (n - 1) / 2),
                             isolated=int((degs == 0).sum()), components=nx.number_connected_components(H),
                             mean_deg=float(degs.mean()), ego1_mean_nodes=float(ego1.mean()),
                             ego2_mean_nodes=float(ego2.mean()), ego2_frac_of_graph=float(ego2.mean() / n))
        A = nx.to_numpy_array(H, nodelist=nm)
        r[f'G_{lab}']['DDI_rate_on_sim_edges'] = float(Y[iu][A[iu] > 0].mean()) if A[iu].sum() else np.nan
        r[f'G_{lab}']['DDI_rate_off_sim_edges'] = float(Y[iu][A[iu] == 0].mean())
        r[f'G_{lab}']['spearman_simdeg_vs_ddideg'] = float(stats.spearmanr(degs, ddeg)[0])
    res[ds] = r
    print(ds, json.dumps(r, indent=1, default=float))

json.dump(res, open(f'{OUT}/out/characterize.json', 'w'), indent=1, default=float)

# degree histogram table (quantiles)
rows = []
for ds in ['D1', 'D2', 'MP']:
    nm = names(ds)
    g = np.array([gdeg.get(d, 0) for d in nm])
    rows.append([ds] + [int(np.quantile(g, q)) for q in [0, .25, .5, .75, 1]])
rows.append(['DrugBank(all)'] + [int(gdeg.quantile(q)) for q in [0, .25, .5, .75, 1]])
pd.DataFrame(rows, columns=['set', 'min', 'q25', 'median', 'q75', 'max']).to_csv(f'{OUT}/out/gdeg_quantiles.csv', index=False)
print(pd.DataFrame(rows, columns=['set', 'min', 'q25', 'median', 'q75', 'max']))
