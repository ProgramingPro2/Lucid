"""Round 3: rigorously test the trend-direction hypothesis."""
import sys; sys.path.insert(0, ".")
import numpy as np
from src.conceptpoc.persistence import load_joint
from src.conceptpoc.codec.jepa_codec import JepaCodec
from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.data.corpus import load_documents

def load_trajs(n_docs=400):
    model, tok = load_joint("artifacts/jepa_v3_128t/jepa/joint_robust.pt", device="cpu")
    codec = JepaCodec(model, tok, 128, device="cpu")
    docs = load_documents("data/austen_corpus.txt")
    arrs = []
    for d in docs[:n_docs]:
        s = [x for x in d.sentences if x.strip()]
        if len(s) >= 8:
            arrs.append(codec.encode(s))
    return arrs, codec.dim

trajs, dim = load_trajs()
print(f"trajectories={len(trajs)} dim={dim}")

def delta_cos_consistency():
    rng = np.random.RandomState(0)
    within = []
    all_deltas = np.concatenate([t[1:]-t[:-1] for t in trajs])
    for t in trajs:
        d = t[1:] - t[:-1]
        dn = d/(np.linalg.norm(d,axis=1,keepdims=True)+1e-9)
        if len(dn) >= 2:
            within.append((dn[:-1]*dn[1:]).sum(1).mean())
    idx1 = rng.randint(0, len(all_deltas), 2000)
    idx2 = rng.randint(0, len(all_deltas), 2000)
    a = all_deltas[idx1]; b = all_deltas[idx2]
    an = a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-9)
    bn = b/(np.linalg.norm(b,axis=1,keepdims=True)+1e-9)
    w = float(np.mean(within)); ac = float((an*bn).sum(1).mean())
    print(f"A. trend consistency: within-doc delta-delta cos={w:.4f} "
          f"across-docs={ac:.4f} signal={w-ac:+.4f}")

delta_cos_consistency()

# B. fit ridge: predict next delta direction from prev delta direction
try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import normalize
    Xs, ys = [], []
    for t in trajs:
        d = t[1:]-t[:-1]
        dn = normalize(d)
        for i in range(len(dn)-1):
            Xs.append(dn[i]); ys.append(dn[i+1])
    X = np.array(Xs); y = np.array(ys)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X)); cut = int(0.8*len(X))
    tr, te = idx[:cut], idx[cut:]
    m = Ridge(alpha=1.0).fit(X[tr], y[tr])
    pred = normalize(m.predict(X[te]))
    d = (pred*y[te]).sum(1)/(np.linalg.norm(pred,axis=1)*np.linalg.norm(y[te],axis=1)+1e-9)
    c = float(d.mean())
    rb = normalize(np.random.RandomState(1).randn(len(te), dim))
    d0 = (rb*y[te]).sum(1)/(np.linalg.norm(rb,axis=1)*np.linalg.norm(y[te],axis=1)+1e-9)
    c0 = float(d0.mean())
    print(f"B. fitted trend predictor: test cos={c:.4f} | random-dir={c0:.4f} gain={c-c0:+.4f}")
except ImportError:
    print("B. sklearn not available; skip fitted test")
