"""Aggregate reeval_verify/out/*.json into mean ± std tables (printed as markdown)."""
import json, os, glob
import numpy as np

O = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')


def ms(v):
    v = np.array(v, float)
    return f'{v.mean():.3f}±{v.std(ddof=1):.3f}' if len(v) > 1 else f'{v[0]:.3f}'


print('## (d) paper-style ordered pairs incl. self pairs, RF / baselines')
for ds in ['D1', 'MP']:
    f = f'{O}/paper_{ds}.json'
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    for scheme in ['random', 'mirror_grouped']:
        folds = r[scheme]
        print(f'\n### {ds} {scheme} ({len(folds)} folds, mirror-in-train {np.mean([x["_mirror_in_train"] for x in folds]):.3f})')
        print('| method | AUROC | AUPRC(AP) | F1 | MCC | hard-label AUPRC |')
        print('|---|---|---|---|---|---|')
        for m in [k for k in folds[0] if not k.startswith('_')]:
            g = lambda k: [x[m][k] for x in folds]
            print(f"| {m} | {ms(g('AUROC'))} | {ms(g('AUPRC'))} | {ms(g('F1'))} | {ms(g('MCC'))} | {ms(g('AUPRC_hard'))} |")

print('\n## (a)-(c) warm unordered vs drug-disjoint (S1/S2)')
for ds in ['D1', 'MP']:
    for pref in ['proper', 'concat_controls']:
        f = f'{O}/{pref}_{ds}.json'
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        print(f"\n### {ds} {pref} (warm folds {len(r['warm'])}, node folds {len(r['node'])})")
        print('| method | warm AUROC | warm AP | warm F1 | warm MCC | S1 AUROC | S1 MCC | S2 AUROC | S2 AP | S2 F1 | S2 MCC |')
        print('|---|---|---|---|---|---|---|---|---|---|---|')
        for m in [k for k in r['warm'][0] if not k.startswith('_')]:
            w = [x[m]['warm'] for x in r['warm']]
            s1 = [x[m]['S1'] for x in r['node'] if m in x]
            s2 = [x[m]['S2'] for x in r['node'] if m in x]
            g = lambda L, k: [x[k] for x in L]
            print(f"| {m} | {ms(g(w,'AUROC'))} | {ms(g(w,'AUPRC'))} | {ms(g(w,'F1'))} | {ms(g(w,'MCC'))} | "
                  f"{ms(g(s1,'AUROC'))} | {ms(g(s1,'MCC'))} | {ms(g(s2,'AUROC'))} | {ms(g(s2,'AUPRC'))} | "
                  f"{ms(g(s2,'F1'))} | {ms(g(s2,'MCC'))} |")
        if 'node' in r and r['node'] and '_sizes' in r['node'][0]:
            print('S2 sizes / pos:', [(x['_sizes']['S2'], round(x['_sizes']['S2_pos'], 3)) for x in r['node']])

for ds in ['D1', 'MP']:
    f = f'{O}/identity_mlp_{ds}.json'
    if os.path.exists(f):
        r = json.load(open(f))
        g = lambda L, k: [x[k] for x in L]
        print(f"\n### {ds} identity MLP (sklearn): warm AUROC {ms(g(r['warm'],'AUROC'))} MCC {ms(g(r['warm'],'MCC'))} F1 {ms(g(r['warm'],'F1'))} | "
              f"S1 AUROC {ms([x['S1']['AUROC'] for x in r['node']])} | S2 AUROC {ms([x['S2']['AUROC'] for x in r['node']])}")
