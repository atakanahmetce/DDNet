"""Pilot: 'new-drug reliance' swap test on DDI-Ben DrugBank S1/S2 with a DRIFT-like Morgan MLP.
Train on train.txt; model selection on valid_S1 macro-F1; then on test_S1:
  normal | swap-new (new drug replaced by a random other *test* drug, role kept) | swap-known (known drug replaced by random train drug)
and on the 'atypical' subset (label != known drug's modal role-specific training type).  CPU only, seed 0."""
import sys, pickle, json, numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
torch.set_num_threads(2); torch.manual_seed(0); rng = np.random.RandomState(0)
ROOT, SPLIT = sys.argv[1], sys.argv[2]; d = f'{ROOT}/data/{SPLIT}/'; R = 86
x = pickle.load(open(f'{ROOT}/data/initial/drugbank/DB_molecular_feats.pkl', 'rb'))
F = torch.tensor(np.array([np.asarray(v, dtype=np.float32) for v in x['Morgan_Features']]) > 0, dtype=torch.float32)
ld = lambda f: np.loadtxt(d + f, dtype=int)
tr = ld('train.txt'); train_set = np.array(sorted(set(ld('train_set.txt').tolist()))); test_set = np.array(sorted(set(ld('test_set.txt').tolist())))
known = np.zeros(F.shape[0], bool); known[train_set] = True
def feats(h, t):
    a, b = F[h], F[t]; return torch.cat([a, b, (a - b).abs(), a * b], 1)
net = nn.Sequential(nn.Linear(4 * F.shape[1], 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, R))
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
def predict(h, t):
    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(h), 8192):
            out.append(net(feats(torch.tensor(h[s:s+8192]), torch.tensor(t[s:s+8192]))).argmax(1).numpy())
    return np.concatenate(out)
def metr(y, p): return dict(macroF1=round(100*f1_score(y, p, average='macro'), 1), acc=round(100*accuracy_score(y, p), 1), kappa=round(100*cohen_kappa_score(y, p), 1))
va = ld('valid_S1.txt'); best, best_state = -1, None
H, T, Y = torch.tensor(tr[:, 0]), torch.tensor(tr[:, 1]), torch.tensor(tr[:, 2])
for ep in range(25):
    net.train(); perm = torch.randperm(len(tr))
    for s in range(0, len(tr), 512):
        i = perm[s:s+512]; opt.zero_grad()
        loss = nn.functional.cross_entropy(net(feats(H[i], T[i])), Y[i]); loss.backward(); opt.step()
    f1v = f1_score(va[:, 2], predict(va[:, 0], va[:, 1]), average='macro')
    if f1v > best: best, best_state, best_ep = f1v, {k: v.clone() for k, v in net.state_dict().items()}, ep
    print(f'ep {ep} loss {loss.item():.3f} valid_S1 macroF1 {100*f1v:.1f}', flush=True)
net.load_state_dict(best_state)
# modal role-specific type of known drug (for atypical subset)
Hc = np.zeros((F.shape[0], R)); Tc = np.zeros((F.shape[0], R)); np.add.at(Hc, (tr[:, 0], tr[:, 2]), 1); np.add.at(Tc, (tr[:, 1], tr[:, 2]), 1)
out = {'best_epoch': best_ep, 'valid_S1_macroF1': round(100*best, 1)}
for S in ['S0', 'S1', 'S2']:
    te = ld(f'test_{S}.txt'); h, t, y = te[:, 0].copy(), te[:, 1].copy(), te[:, 2]
    r = {'normal': metr(y, predict(h, t))}
    if S == 'S1':
        kh = known[h]
        newd = np.where(kh, t, h); knd = np.where(kh, h, t)
        sw_new = rng.choice(test_set, len(te)); sw_kn = rng.choice(train_set, len(te))
        hn, tn = np.where(kh, h, sw_new), np.where(kh, sw_new, t)
        hk, tk = np.where(kh, sw_kn, h), np.where(kh, t, sw_kn)
        r['swap-new'] = metr(y, predict(hn, tn)); r['swap-known'] = metr(y, predict(hk, tk))
        modal = np.where(kh, Hc[h].argmax(1), Tc[t].argmax(1))
        atyp = y != modal; p = predict(h, t)
        r['atypical_frac'] = round(100*atyp.mean(), 1)
        r['normal_on_atypical'] = metr(y[atyp], p[atyp]); r['normal_on_typical'] = metr(y[~atyp], p[~atyp])
        r['agree_with_modal_lookup_pct'] = round(100*(p == modal).mean(), 1)
    if S == 'S2':
        sw = rng.choice(test_set, len(te)); r['swap-tail'] = metr(y, predict(h, sw)); r['swap-head'] = metr(y, predict(sw, t))
    out[S] = r; print(SPLIT, S, r, flush=True)
json.dump(out, open(f'out/phase2/evalsci/swap_{SPLIT}.json', 'w'), indent=1)
