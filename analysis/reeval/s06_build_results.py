"""Assemble RESULTS.md from results_template.md + generated tables + narrative blocks in narrative/*.md."""
import json, os, re
from common import OUT

O = f'{OUT}/out'
t = open(f'{OUT}/results_template.md').read()
tables = open(f'{O}/tables.md').read()
compact = open(f'{O}/compact.md').read()

# split compact into A / B / C
parts = re.split(r'(?m)^### ', compact)
secs = {p[0]: '### ' + p for p in parts if p.strip()}
t = t.replace('{{COMPACT_A}}', secs.get('A', ''))
t = t.replace('{{COMPACT_B}}', secs.get('B', ''))
t = t.replace('{{COMPACT_C}}', secs.get('C', ''))

# cross section = from '## Cross-dataset' to end of tables.md ; FULL = the rest
i = tables.find('## Cross-dataset')
cross = tables[i:] if i >= 0 else '(cross-dataset run missing)'
full = tables[:i] if i >= 0 else tables
t = t.replace('{{CROSS_TABLES}}', cross.replace('## Cross-dataset D1 → D2\n', ''))
t = t.replace('{{FULL}}', full.replace('## ', '### '))

# FFNN long
f = f'{O}/ffnn_orig_long_D1h1.json'
if os.path.exists(f):
    r = json.load(open(f))
    L = ['| epochs (full-batch steps) | F1 | MCC | Precision | AUROC (scores) | hard-label AUPRC | predicted-positive rate |', '|---|---|---|---|---|---|---|']
    for m in r:
        L.append(f"| {m['epoch']} | {m['F1']:.3f} | {m['MCC']:.3f} | {m['Prec']:.3f} | {m['AUROC']:.3f} | {m['AUPRC_hard']:.3f} | {m['pred_pos_rate']:.3f} |")
    L.append('\n(Report, D1 hop-1 FFNN: F1 0.90, MCC 0.71, AUPRC 0.92, precision 0.86. True positive rate 0.58.)')
    t = t.replace('{{FFNN_LONG}}', '\n'.join(L))
else:
    t = t.replace('{{FFNN_LONG}}', '(not run)')

# MP folds
f = f'{O}/proper_MP_hop1.json'
if os.path.exists(f):
    r = json.load(open(f))
    t = t.replace('{{MP_FOLDS}}', f"{len(r['warm'])} warm and {len(r['node'])} node folds")

for key in ['TAKEAWAYS', 'PROPER_FINDINGS', 'CROSS_FINDINGS', 'PUBLISH', 'CAVEATS']:
    fn = f'{OUT}/narrative/{key.lower()}.md'
    t = t.replace('{{' + key + '}}', open(fn).read().strip() if os.path.exists(fn) else f'(missing {key})')

open(f'{OUT}/RESULTS.md', 'w').write(t)
print('RESULTS.md written', len(t))
