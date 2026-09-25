"""Build markdown tables from out/*.json -> out/tables.md (pasted into RESULTS.md)."""
import json, os, glob
import numpy as np
from common import OUT

O = f'{OUT}/out'
L = []

PAPER = {  # (MCC, F1, AUPRC[hard-label], Precision) from the 2022 report
    'MP': {'DDNet RF': (0.26, 0.53, 0.67, None), 'DDNet GB(HGB)': (None, 0.73, 0.78, None),
           'DDNet FFNN-orig': (0.72, 0.85, 0.87, None)},
    'D1h1': {'DDNet RF': (0.37, 0.82, 0.85, 0.71), 'DDNet GB(HGB)': (0.46, 0.84, 0.88, 0.80), 'DDNet FFNN-orig': (0.71, 0.90, 0.92, 0.86)},
    'D1h2': {'DDNet RF': (0.32, 0.79, 0.83, 0.70), 'DDNet GB(HGB)': (0.48, 0.83, 0.87, 0.79), 'DDNet FFNN-orig': (0.24, 0.79, 0.82, 0.65)},
    'D2h1': {'DDNet RF': (0.49, 0.83, 0.87, 0.78), 'DDNet GB(HGB)': (0.52, 0.85, 0.88, 0.81), 'DDNet FFNN-orig': (0.68, 0.91, 0.93, 0.88)},
    'D2h2': {'DDNet RF': (0.28, 0.81, 0.84, 0.71), 'DDNet GB(HGB)': (0.44, 0.84, 0.87, 0.78), 'DDNet FFNN-orig': (0.54, 0.86, 0.89, 0.79)},
}
PAPER_CROSS = {1: {'DDNet RF': (0.16, 0.80, 0.83, 0.67), 'DDNet GB(HGB)': (0.20, 0.76, 0.83, 0.73), 'DDNet FFNN-orig': (0.13, 0.64, 0.79, 0.73)},
               2: {'DDNet RF': (0.13, 0.72, 0.78, 0.60), 'DDNet GB(HGB)': (0.056, 0.12, 0.64, 0.68), 'DDNet FFNN-orig': (0.10, 0.79, 0.83, 0.66)}}


def ms(v, k=3):
    v = np.array(v, dtype=float)
    if len(v) == 1:
        return f'{v[0]:.{k}f}'
    return f'{np.nanmean(v):.{k}f} ± {np.nanstd(v, ddof=1):.{k}f}'


def p(x):
    return '–' if x is None else f'{x:.2f}'


# ------------------------------------------------------------------ reproduction
L.append('## Reproduction tables (as-in-paper protocol)\n')
for cfg in ['MP', 'D1h1', 'D1h2', 'D2h1', 'D2h2', 'D1q1', 'D2q1']:
    f = f'{O}/reproduce_{cfg}.json'
    if not os.path.exists(f):
        L.append(f'### {cfg}: NOT RUN\n'); continue
    r = json.load(open(f))
    folds = r['folds']
    L.append(f"### {cfg} — `{r['file']}`, {r['n_drugs']} drugs, {r['rows']} ordered rows (incl. self), "
             f"positive rate {r['pos_rate']:.3f}, {len(folds)} folds\n")
    L.append('| Method | F1 | MCC | Precision | AUPRC (paper way, hard labels) | AUROC (hard) | AUROC (scores) | AUPRC (scores, AP) | fold-1 = 80/20 split F1 / MCC | paper MCC / F1 / AUPRC / Prec |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for m in folds[0]:
        g = lambda k: [fo[m][k] for fo in folds]
        pp = PAPER.get(cfg, {}).get(m)
        pstr = ' / '.join(p(x) for x in pp) if pp else ''
        L.append(f"| {m} | {ms(g('F1'))} | {ms(g('MCC'))} | {ms(g('Prec'))} | {ms(g('AUPRC_hard'))} | {ms(g('AUROC_hard'))} | "
                 f"{ms(g('AUROC'))} | {ms(g('AUPRC'))} | {folds[0][m]['F1']:.3f} / {folds[0][m]['MCC']:.3f} | {pstr} |")
    fc = f'{O}/controls_{cfg}.json'
    if os.path.exists(fc):
        cf = json.load(open(fc))['folds']
        for m in cf[0]:
            g = lambda k: [fo[m][k] for fo in cf]
            L.append(f"| {m} [{len(cf)} folds] | {ms(g('F1'))} | {ms(g('MCC'))} | {ms(g('Prec'))} | {ms(g('AUPRC_hard'))} | {ms(g('AUROC_hard'))} | "
                     f"{ms(g('AUROC'))} | {ms(g('AUPRC'))} | {cf[0][m]['F1']:.3f} / {cf[0][m]['MCC']:.3f} |  |")
    if 'pred_pos_rate' in folds[0].get('DDNet FFNN-orig', {}):
        L.append(f"\nFFNN-orig predicted-positive rate: {ms([fo['DDNet FFNN-orig']['pred_pos_rate'] for fo in folds])}\n")
    L.append('')

# ------------------------------------------------------------------ proper protocols
L.append('## Leakage-free protocols (unordered pairs, no self pairs)\n')
for f in sorted(glob.glob(f'{O}/proper_*.json')):
    r = json.load(open(f))
    if not r['node']:
        continue
    sizes = [fo['_sizes'] for fo in r['node']]
    L.append(f"### {r['ds']} (hop-{r['hop']} embeddings) — {r['n']} drugs, {r['pairs']} unordered pairs; "
             f"per node-fold: train≈{np.mean([s['train'] for s in sizes]):.0f}, S1≈{np.mean([s['S1'] for s in sizes]):.0f}, "
             f"S2≈{np.mean([s['S2'] for s in sizes]):.0f} pairs; warm folds={len(r['warm'])}, node folds={len(r['node'])}\n")
    L.append('| Method | warm AUROC | warm AUPRC | warm F1 | warm MCC | S1 AUROC | S1 AUPRC | S1 MCC | S2 AUROC | S2 AUPRC | S2 MCC |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for m in r['warm'][0]:
        w = [fo[m]['warm'] for fo in r['warm']]
        s1 = [fo[m]['S1'] for fo in r['node']]
        s2 = [fo[m]['S2'] for fo in r['node']]
        g = lambda lst, k: [x[k] for x in lst]
        L.append(f"| {m} | {ms(g(w, 'AUROC'))} | {ms(g(w, 'AUPRC'))} | {ms(g(w, 'F1'))} | {ms(g(w, 'MCC'))} | "
                 f"{ms(g(s1, 'AUROC'))} | {ms(g(s1, 'AUPRC'))} | {ms(g(s1, 'MCC'))} | "
                 f"{ms(g(s2, 'AUROC'))} | {ms(g(s2, 'AUPRC'))} | {ms(g(s2, 'MCC'))} |")
    L.append(f"\nPositive rate: warm {np.mean([fo[list(fo)[0]]['warm']['pos_rate'] for fo in r['warm']]):.3f}, "
             f"S1 {np.mean([s['S1_pos'] for s in sizes]):.3f}, S2 {np.mean([s['S2_pos'] for s in sizes]):.3f} "
             f"(= AUPRC of a random ranker).\n")

# ------------------------------------------------------------------ cross
f = f'{O}/cross.json'
if os.path.exists(f):
    r = json.load(open(f))
    L.append('## Cross-dataset D1 → D2\n')
    for hop in [1, 2]:
        c = r.get(f'cross_hop{hop}')
        if not c:
            continue
        sz = c['_sizes']
        L.append(f"### hop-{hop} (train all D1 ordered pairs, test all D2 ordered pairs; D2 pair types: "
                 f"both drugs also in D1 = {sz['both_shared']}, one = {sz['one_shared']}, none = {sz['none_shared']})\n")
        L.append('| Method | all F1 | all MCC | all AUROC | all AUPRC(AP) | all AUPRC(hard) | both-shared AUROC | one-shared AUROC | none-shared AUROC | none-shared MCC | paper MCC / F1 / AUPRC / Prec |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for m, v in c.items():
            if m.startswith('_'):
                continue
            pp = PAPER_CROSS[hop].get(m)
            L.append(f"| {m} | {v['all']['F1']:.3f} | {v['all']['MCC']:.3f} | {v['all']['AUROC']:.3f} | {v['all']['AUPRC']:.3f} | "
                     f"{v['all']['AUPRC_hard']:.3f} | {v['both_shared']['AUROC']:.3f} | {v['one_shared']['AUROC']:.3f} | "
                     f"{v['none_shared']['AUROC']:.3f} | {v['none_shared']['MCC']:.3f} | {' / '.join(p(x) for x in pp) if pp else ''} |")
        L.append('')
    L.append('### Alignment of the 46 shared drugs\n')
    L.append('| hop | cos(same drug D1 vs D2) mean [min,max] | cos(different drugs) mean | Procrustes residual (true) | Procrustes residual (perm. null mean, 5%) | perm p | held-out cos matched / mismatched | held-out mean rank of true match (chance) |')
    L.append('|---|---|---|---|---|---|---|---|')
    for hop in [1, 2]:
        a = r.get(f'align_hop{hop}')
        if not a:
            continue
        L.append(f"| {hop} | {a['cos_same_drug_mean']:.3f} [{a['cos_same_drug_min']:.2f}, {a['cos_same_drug_max']:.2f}] | "
                 f"{a['cos_different_drugs_mean']:.3f} | {a['procrustes_resid_true']:.3f} | {a['procrustes_resid_null_mean']:.3f}, "
                 f"{a['procrustes_resid_null_p05']:.3f} | {a['procrustes_perm_p']:.3f} | {a['heldout_cos_matched']:.3f} / "
                 f"{a['heldout_cos_mismatched']:.3f} | {a['heldout_mean_rank_of_true_match']:.1f} ({a['heldout_rank_chance']:.1f}) |")
    L.append('\n### Are per-ego embeddings comparable within a dataset?\n')
    L.append('| dataset/hop | Spearman(cos emb, S) | NN@10 overlap emb vs S (chance) | AUROC of cos(emb) for DDI | mean off-diag cos | Spearman(norm, sim-degree) | Spearman(norm, DDI-degree) | global spectral emb: Spearman(cos,S) / NN@10 |')
    L.append('|---|---|---|---|---|---|---|---|')
    for k, d in r.items():
        if not k.startswith('comparability_'):
            continue
        L.append(f"| {k.replace('comparability_', '')} | {d['spearman_cosEmb_vs_S']:.3f} | {d['nn10_overlap_emb_vs_S'][0]:.2f} ({d['nn10_overlap_emb_vs_S'][1]:.2f}) | "
                 f"{d['auroc_cosEmb_for_DDI']:.3f} | {d['mean_offdiag_cos']:.3f} | {d['spearman_norm_vs_simdegree']:.3f} | "
                 f"{d['spearman_norm_vs_DDIdegree']:.3f} | {d['global_spectral_spearman_cos_vs_S']:.3f} / {d['global_spectral_nn10_overlap_vs_S'][0]:.2f} |")
    for k, v in r.items():
        if k.startswith('hop1_vs_hop2'):
            L.append(f'\n{k}: {v:.3f}')

open(f'{O}/tables.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))


# =================================================================== compact summaries -> out/compact.md
C = []


def load_rep(cfg):
    f = f'{O}/reproduce_{cfg}.json'
    if not os.path.exists(f):
        return None
    folds = json.load(open(f))['folds']
    fc = f'{O}/controls_{cfg}.json'
    ctrl = json.load(open(fc))['folds'] if os.path.exists(fc) else None
    return folds, ctrl


def mean_of(folds, m, k):
    v = [fo[m][k] for fo in folds if m in fo]
    return np.mean(v) if v else np.nan


C.append('### A. As-in-paper protocol: our re-run vs the 2022 report\n')
C.append('| Data | Model | ours F1 | ours MCC | ours AUPRC (hard labels, paper way) | ours AUROC (scores) | paper F1 | paper MCC | paper AUPRC | folds |')
C.append('|---|---|---|---|---|---|---|---|---|---|')
for cfg in ['MP', 'D1h1', 'D2h1', 'D1h2', 'D2h2']:
    r = load_rep(cfg)
    if not r:
        continue
    folds, _ = r
    for m in ['DDNet RF', 'DDNet GB(HGB)', 'DDNet FFNN-orig', 'DDNet FFNN-fixed']:
        pp = PAPER.get(cfg, {}).get(m, (None, None, None, None))
        C.append(f"| {cfg} | {m} | {mean_of(folds, m, 'F1'):.3f} | {mean_of(folds, m, 'MCC'):.3f} | {mean_of(folds, m, 'AUPRC_hard'):.3f} | "
                 f"{mean_of(folds, m, 'AUROC'):.3f} | {p(pp[1])} | {p(pp[0])} | {p(pp[2])} | {len(folds)} |")

C.append('\n### B. Same (leaky) protocol: DDNet vs trivial baselines and identity controls (mean over folds)\n')
meths = ['All-positive', 'Raw similarity S', 'Path features only (HGB)', 'Degree (product)', 'One-hot identity LR',
         'One-hot identity RF', 'DDNet RF', 'DDNet GB(HGB)', 'Embeddings only (HGB)', 'DDNet FFNN-orig', 'DDNet FFNN-fixed',
         'RandomVec + path HGB (control)', 'RandomVec + path FFNN-fixed (control)', 'Identity MLP (FFNN-fixed on one-hot)']
cfgs = ['MP', 'D1h1', 'D2h1', 'D1q1', 'D2q1']
C.append('| Method | ' + ' | '.join(f'{c} AUROC / MCC / F1 / AUPRC-hard' for c in cfgs) + ' |')
C.append('|---|' + '---|' * len(cfgs))
for m in meths:
    cells = []
    for cfg in cfgs:
        r = load_rep(cfg)
        if not r:
            cells.append('n/a'); continue
        folds, ctrl = r
        src = folds if m in folds[0] else (ctrl if ctrl and m in ctrl[0] else None)
        if src is None:
            cells.append('–'); continue
        cells.append(f"{mean_of(src, m, 'AUROC'):.3f} / {mean_of(src, m, 'MCC'):.3f} / {mean_of(src, m, 'F1'):.3f} / {mean_of(src, m, 'AUPRC_hard'):.3f}")
    C.append(f'| {m} | ' + ' | '.join(cells) + ' |')

C.append('\n### C. Leakage-free protocols: AUROC (mean ± std over folds)\n')
pm = ['DDNet-sym RF', 'DDNet-sym HGB', 'DDNet-sym FFNN-fixed', 'DDNet-concat both-orient. HGB', 'Emb-sym only HGB',
      'RandomVec-sym + path HGB (control)', 'RandomVec-sym + path FFNN-fixed (control)', 'Identity MLP (FFNN-fixed on multi-hot)',
      'Identity LR', 'Degree (product)', 'Morgan FP HGB', 'SimKNN (k=10)', 'Raw similarity S', 'Path only HGB',
      '[leaky] DrugBank global degree product']
PR = {}
for ds in ['D1', 'D2', 'MP']:
    f = f'{O}/proper_{ds}_hop1.json'
    if os.path.exists(f):
        PR[ds] = json.load(open(f))
hdr = []
for ds in PR:
    hdr += [f'{ds} warm', f'{ds} S1', f'{ds} S2']
C.append('| Method | ' + ' | '.join(hdr) + ' |')
C.append('|---|' + '---|' * len(hdr))
for m in pm:
    cells = []
    for ds, r in PR.items():
        for split in ['warm', 'S1', 'S2']:
            src = r['warm'] if split == 'warm' else r['node']
            v = [fo[m][split]['AUROC'] for fo in src if m in fo]
            cells.append(ms(v, 3) if v else '–')
    C.append(f'| {m} | ' + ' | '.join(cells) + ' |')
C.append('\nMCC (mean over folds) for the same methods:\n')
C.append('| Method | ' + ' | '.join(hdr) + ' |')
C.append('|---|' + '---|' * len(hdr))
for m in pm:
    cells = []
    for ds, r in PR.items():
        for split in ['warm', 'S1', 'S2']:
            src = r['warm'] if split == 'warm' else r['node']
            v = [fo[m][split]['MCC'] for fo in src if m in fo]
            cells.append(f'{np.mean(v):.3f}' if v else '–')
    C.append(f'| {m} | ' + ' | '.join(cells) + ' |')
C.append('\nFolds used: ' + ', '.join(f"{ds}: warm {len(r['warm'])}, node {len(r['node'])}" for ds, r in PR.items()))
open(f'{O}/compact.md', 'w').write('\n'.join(C) + '\n')
print('\n'.join(C))
