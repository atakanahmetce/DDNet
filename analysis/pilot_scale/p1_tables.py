"""Pilot 1 - step 3: tables from out/eval_<SAMPLE>.json -> out/tables.md"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/out'
SAMPLES = [s for s in ['U1000', 'U300', 'U300b', 'U300c', 'H300'] if os.path.exists(f'{OUT}/eval_{s}.json')]
R = {s: json.load(open(f'{OUT}/eval_{s}.json')) for s in SAMPLES}
# merge node folds that were run in a separate process (OUTSUFFIX=_nodefold<k>)
import glob
for s in SAMPLES:
    for fnp in sorted(glob.glob(f'{OUT}/eval_{s}_nodefold*.json')):
        part = json.load(open(fnp))
        have = {fr.get('_fold', i) for i, fr in enumerate(R[s]['node'])}
        for fr in part['node']:
            if fr['_fold'] not in have:
                R[s]['node'].append(fr)
SPL = ['warm', 'S1', 'S2']
L = []


def vals(r, m, sp, key):
    src = r['warm'] if sp == 'warm' else r['node']
    v = [fr[m][sp].get(key, np.nan) for fr in src if m in fr and sp in fr[m]]
    return np.array(v, dtype=float)


def ms(v, d=3):
    if len(v) == 0 or np.all(np.isnan(v)):
        return '–'
    return f'{np.nanmean(v):.{d}f} ± {np.nanstd(v, ddof=1):.{d}f}' if len(v) > 1 else f'{v[0]:.{d}f}'


L.append('## T1. Samples and split sizes (mean over folds)\n')
L.append('| Sample | drugs | pairs | prevalence | median out-of-sample DrugBank degree | median #targets | % drugs with 0 targets | folds warm/node | warm test n (prev) | S1 n (prev) | S2 n (prev) | train rows (HGB rows) warm / node |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
for s, r in R.items():
    ds = r['drug_stats']
    def sz(sp):
        src = r['warm'] if sp == 'warm' else r['node']
        return f"{np.mean([f['_sizes'][sp]['n'] for f in src]):.0f} ({np.mean([f['_sizes'][sp]['prev'] for f in src]):.3f})"
    trw = r['warm'][0]['_sizes']['train']; trn = r['node'][0]['_sizes']['train'] if r['node'] else None
    L.append(f"| {s} | {r['n']} | {r['pairs']} | {r['prevalence']:.3f} | {ds['median_ext_degree']:.0f} | {ds['median_targets']:.0f} | "
             f"{100 * ds['frac_zero_targets']:.0f} | {len(r['warm'])}/{len(r['node'])} | {sz('warm')} | "
             f"{sz('S1') if r['node'] else '–'} | {sz('S2') if r['node'] else '–'} | {trw['n']} ({trw['hgb_rows']}) / "
             f"{(str(trn['n']) + ' (' + str(trn['hgb_rows']) + ')') if trn else '–'} |")

methods = [m for m in R[SAMPLES[0]]['warm'][0] if not m.startswith('_')]
for key, lab in [('AUROC', 'AUROC'), ('AP', 'Average precision (compare with prevalence in T1)')]:
    for s, r in R.items():
        L.append(f'\n## T2-{key}. {s}: {lab}, mean ± std over folds\n')
        L.append('| Method | warm | S1 | S2 |'); L.append('|---|---|---|---|')
        for m in methods:
            L.append(f'| {m} | ' + ' | '.join(ms(vals(r, m, sp, key)) for sp in SPL) + ' |')

L.append('\n## T3. Degree-stratified AUROC (pos-neg pairs compared only within the same stratum of out-of-sample DrugBank degree quintiles; 15 unordered strata) - mean over folds\n')
L.append('| Method | ' + ' | '.join(f'{s} {sp}' for s in R for sp in SPL) + ' |')
L.append('|---|' + '---|' * (3 * len(R)))
for m in methods:
    L.append(f'| {m} | ' + ' | '.join(f"{np.nanmean(vals(r, m, sp, 'sAUROC')):.3f}" if len(vals(r, m, sp, 'sAUROC')) else '–'
                                     for s, r in R.items() for sp in SPL) + ' |')

L.append('\n## T4. Paired per-fold differences (AUROC and AP; mean ± std; #folds with diff > 0 / #folds)\n')
pairs = [('S: ECFP4 + proxies (HGB)', 'P: proxies = tc+4 desc (HGB)'),
         ('S: ECFP4 (HGB)', 'P: proxies = tc+4 desc (HGB)'),
         ('S: ECFP4 + tc (HGB)', 'S: ECFP4 (HGB)'),
         ('S: ECFP4 + proxies (HGB)', 'S: ECFP4 (HGB)'),
         ('T: proxies + shared targets (HGB)', 'P: proxies = tc+4 desc (HGB)'),
         ('N: SimKNN k=10 (2-sided)', 'P: proxies = tc+4 desc (HGB)'),
         ('N: SimKNN k=10 (2-sided)', 'B: degree prior, train labels (product)'),
         ('P: proxies = tc+4 desc (HGB)', 'B: degree prior, train labels (product)')]
L.append('| A − B | sample | ' + ' | '.join(f'{sp} ΔAUROC | {sp} ΔAP' for sp in SPL) + ' |')
L.append('|---|---|' + '---|---|' * 3)
for a, b in pairs:
    for s, r in R.items():
        cells = []
        for sp in SPL:
            for key in ['AUROC', 'AP']:
                va, vb = vals(r, a, sp, key), vals(r, b, sp, key)
                if len(va) == 0 or len(va) != len(vb):
                    cells.append('–'); continue
                d = va - vb
                cells.append(f'{d.mean():+.3f} ± {d.std(ddof=1):.3f} ({(d > 0).sum()}/{len(d)})')
        L.append(f'| {a} − {b} | {s} | ' + ' | '.join(cells) + ' |')

L.append('\n## T5. Share of above-chance AUROC reached without structure: (AUROC_proxy − 0.5) / (AUROC_ECFP − 0.5)\n')
L.append('| sample | split | ECFP4 (HGB) | proxies tc+4desc (HGB) | target count only (HGB) | 4 descriptors (HGB) | share proxies | share tc | share desc |')
L.append('|---|---|---|---|---|---|---|---|---|')
for s, r in R.items():
    for sp in SPL:
        e = np.mean(vals(r, 'S: ECFP4 (HGB)', sp, 'AUROC')); p = np.mean(vals(r, 'P: proxies = tc+4 desc (HGB)', sp, 'AUROC'))
        t = np.mean(vals(r, 'P: target count (HGB)', sp, 'AUROC')); dd = np.mean(vals(r, 'P: 4 descriptors (HGB)', sp, 'AUROC'))
        f = lambda x: f'{(x - 0.5) / (e - 0.5):.2f}' if e > 0.5 else '–'
        L.append(f'| {s} | {sp} | {e:.3f} | {p:.3f} | {t:.3f} | {dd:.3f} | {f(p)} | {f(t)} | {f(dd)} |')

L.append('\n## T6. Spearman correlation of test scores with (i) the out-of-sample DrugBank degree product and (ii) the proxies-HGB score (mean over folds)\n')
L.append('| Method | ' + ' | '.join(f'{s} {sp} ρ_ext / ρ_proxy' for s in R for sp in ['S1', 'S2']) + ' |')
L.append('|---|' + '---|' * (2 * len(R)))
for m in methods:
    cells = []
    for s, r in R.items():
        for sp in ['S1', 'S2']:
            a, b = vals(r, m, sp, 'rho_ext'), vals(r, m, sp, 'rho_proxy')
            cells.append(f"{np.nanmean(a):.2f} / {np.nanmean(b):.2f}" if len(a) and not np.all(np.isnan(a)) else '–')
    L.append(f'| {m} | ' + ' | '.join(cells) + ' |')

L.append('\n## T7. HGB fit+predict time per fold (s, mean)\n')
L.append('| sample | ' + ' | '.join(['warm ECFP4', 'node ECFP4', 'warm proxies', 'node proxies']) + ' |'); L.append('|---|---|---|---|---|')
for s, r in R.items():
    g = lambda src, k: np.mean([f['_timing'].get(k, np.nan) for f in src]) if src else np.nan
    L.append(f"| {s} | {g(r['warm'], 'S: ECFP4 (HGB)'):.0f} | {g(r['node'], 'S: ECFP4 (HGB)'):.0f} | "
             f"{g(r['warm'], 'P: proxies = tc+4 desc (HGB)'):.1f} | {g(r['node'], 'P: proxies = tc+4 desc (HGB)'):.1f} |")
# ---- compact headline table (written to compact.md)
C = []
KEY = ['B: degree prior, train labels (product)', 'B: identity one-hot LR', '[diag] external DrugBank degree product',
       'U: Tanimoto ECFP4 (raw)', 'U: target-count product (raw)', 'N: Vilar max-Tanimoto to partners',
       'N: Vilar mean-Tanimoto to partners', 'N: SimKNN k=10 (2-sided)', 'P: target count (HGB)',
       'P: 4 descriptors (HGB)', 'P: proxies = tc+4 desc (HGB)', 'T: proxies + shared targets (HGB)',
       'S: ECFP4 (HGB)', 'S: ECFP4 + tc (HGB)', 'S: ECFP4 + proxies (HGB)']
for key, lab in [('AUROC', 'AUROC'), ('AP', 'AP')]:
    C.append(f'\n### {lab} (mean ± std over folds; prevalence: ' + ', '.join(f"{s} {R[s]['prevalence']:.3f}" for s in R) + ')\n')
    C.append('| Method | ' + ' | '.join(f'{s} {sp}' for s in R for sp in SPL) + ' |')
    C.append('|---|' + '---|' * (3 * len(R)))
    for m in KEY:
        C.append(f'| {m} | ' + ' | '.join(ms(vals(r, m, sp, key), 3) for s, r in R.items() for sp in SPL) + ' |')
BP = {s: json.load(open(f'{OUT}/beyond_pop_{s}.json')) for s in R if os.path.exists(f'{OUT}/beyond_pop_{s}.json')}
if BP:
    C.append('\n### Beyond the strongest popularity signal ([diag] out-of-sample DrugBank degree; p1_beyond_pop.py), AUROC / AP, mean ± std over node folds\n')
    ms_ = list(next(iter(BP.values()))['node'][0].keys())
    C.append('| Method | ' + ' | '.join(f'{s} {sp} (folds)' for s in BP for sp in ['S1', 'S2']) + ' |')
    C.append('|---|' + '---|' * (2 * len(BP)))
    for m in ms_:
        cells = []
        for s, b in BP.items():
            for sp in ['S1', 'S2']:
                au = np.array([fr[m][sp]['AUROC'] for fr in b['node']]); ap = np.array([fr[m][sp]['AP'] for fr in b['node']])
                cells.append(f'{ms(au)} / {ms(ap)} ({len(au)})')
        C.append(f'| {m} | ' + ' | '.join(cells) + ' |')
if BP:
    C.append('\n### Paired per-fold gain over HGB on out-of-sample degree alone (mean ± sd; #folds > 0)\n')
    C.append('| sample | split | +ECFP4 ΔAUROC | +ECFP4 ΔAP | +proxies ΔAUROC | +proxies ΔAP |'); C.append('|---|---|---|---|---|---|')
    for s_, b in BP.items():
        for sp in ['S1', 'S2']:
            cells = []
            for m in ['ext-degree + ECFP4 (HGB)', 'ext-degree + proxies (HGB)']:
                for k in ['AUROC', 'AP']:
                    d = np.array([f[m][sp][k] - f['ext-degree (HGB)'][sp][k] for f in b['node']])
                    cells.append(f'{d.mean():+.3f} ± {d.std(ddof=1):.3f} ({(d > 0).sum()}/{len(d)})')
            C.append(f'| {s_} | {sp} | ' + ' | '.join(cells) + ' |')
open(f'{OUT}/compact.md', 'w').write('\n'.join(C) + '\n')
open(f'{OUT}/tables.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
