"""Pilot 1 - drug-level check: how do the drug-side proxies relate to full-DrugBank DDI degree (3,618-drug pool)?"""
import numpy as np, json, os
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(f'{HERE}/out/prep.npz'); TC = z['TC'].astype(float); g = z['g_pool']; D = z['D']
out = {'n_pool': int(len(TC)), 'TC_quantiles_25_50_75_90_99': np.percentile(TC, [25, 50, 75, 90, 99]).tolist(),
       'frac_zero_targets': float((TC == 0).mean()),
       'rho_TC_degree': float(spearmanr(TC, g)[0]),
       'median_degree_zero_targets': float(np.median(g[TC == 0])), 'median_degree_nonzero_targets': float(np.median(g[TC > 0]))}
for k, nm in enumerate(['MW', 'logP', 'rotB', 'rings']):
    out[f'rho_{nm}_degree'] = float(spearmanr(D[:, k], g)[0])
# nonlinear: 5-fold CV HGB regression of log degree from proxies (drug level)
X = np.column_stack([TC, D]); lg = np.log1p(g)
for nm, cols in [('tc', [0]), ('desc', [1, 2, 3, 4]), ('tc+desc', [0, 1, 2, 3, 4])]:
    p = cross_val_predict(HistGradientBoostingRegressor(max_iter=150, random_state=0), X[:, cols], lg,
                          cv=KFold(5, shuffle=True, random_state=0))
    out[f'cv_rho_pred_logdeg_from_{nm}'] = float(spearmanr(p, lg)[0])
json.dump(out, open(f'{HERE}/out/proxy_corr.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
