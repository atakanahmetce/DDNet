"""FFNN variants.

orig  : faithful copy of model_train/learning_functions.Net + deep_model(isValid=False) +
        evaluation_functions.evaluate_score(isDeep=True):
          Softmax(dim=0) over the *batch*, BCELoss, full-batch Adam, 200 epochs,
          hard label = out >= 0.5/len(X) computed over the whole test set at once.
fixed : same layer stack, no softmax, BCEWithLogitsLoss, mini-batch Adam, threshold 0.5 on sigmoid.
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)


class Net(nn.Module):
    def __init__(self, in_size, hid_size=128, layer_size=5, p=0.4, out_size=1, softmax=True):
        super().__init__()
        self.layer_size = layer_size
        self.lf = nn.Linear(in_size, hid_size)
        self.lm = nn.Linear(hid_size, hid_size)
        self.lo = nn.Linear(hid_size, out_size)
        self.dropout = nn.Dropout(p)
        self.use_softmax = softmax
        self.softmax = nn.Softmax(dim=0)
        self.leaky = nn.LeakyReLU()

    def logits(self, X):
        first_layer = self.dropout(self.leaky(self.lf(X)))
        layer = self.leaky(self.lm(first_layer))
        for i in range(3, self.layer_size):          # same (shared) lm layer re-applied, as in repo
            layer = self.leaky(self.lm(layer))
        return self.lo(self.dropout(layer))

    def forward(self, X):
        z = self.logits(X)
        return self.softmax(z) if self.use_softmax else z


def train_orig(X, y, epochs=200, p=0.4, lr=0.0007, eps=1e-5, n_layer=5, hid=128, seed=0):
    torch.manual_seed(seed)
    X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
    model = Net(X.shape[1], hid, n_layer, p, softmax=True)
    crit = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=eps)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X)
        loss = crit(out, y)
        loss.backward()
        opt.step()
    return model


def predict_orig(model, X):
    """Returns (hard labels exactly as evaluate_score, continuous score = logit)."""
    X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        z = model.logits(X)
        out = torch.softmax(z, dim=0)
    hard = np.where(out.numpy().ravel() < 0.5 / len(X), 0, 1)
    return hard, z.numpy().ravel()


def train_fixed(X, y, epochs=40, p=0.4, lr=1e-3, eps=1e-5, n_layer=5, hid=128, bs=512, seed=0,
                pos_weight=None):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
    model = Net(Xt.shape[1], hid, n_layer, p, softmax=False)
    pw = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=eps)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            loss = crit(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    return model


def predict_fixed(model, X):
    X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        z = model(X).numpy().ravel()
    prob = 1 / (1 + np.exp(-z))
    return (prob >= 0.5).astype(int), prob
