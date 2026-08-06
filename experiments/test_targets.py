"""Experiment matrix: which 'forward JEPA' target is learnable without collapse?

We compare candidate TARGETS for forward prediction, measured on real concept
vectors. Learnability = adjacent-target cosine; signal = adj minus shuffled.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from src.conceptpoc.persistence import load_joint
from src.conceptpoc.codec.jepa_codec import JepaCodec
from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.data.corpus import load_documents


def load_trajs(n_docs=300):
    model, tok = load_joint("artifacts/jepa_v3_128t/jepa/joint_robust.pt", device="cpu")
    codec = JepaCodec(model, tok, 128, device="cpu")
    docs = load_documents("data/austen_corpus.txt")
    arrs = []
    for d in docs[:n_docs]:
        s = [x for x in d.sentences if x.strip()]
        if len(s) >= 6:
            arrs.append(codec.encode(s))
    return arrs, codec.dim


def run(name, fn):
    """fn(traj) -> (P, T) aligned contiguous arrays of matching leading dim."""
    adj, shuf = [], []
    rng = np.random.RandomState(0)
    for traj in trajs:
        try:
            P, T = fn(traj)
        except Exception:
            continue
        if P is None or len(P) < 2 or T.shape[0] != P.shape[0]:
            continue
        n = np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1) + 1e-9
        adj.append(((P * T).sum(1) / n).mean())
        ridx = rng.permutation(T.shape[0])
        shuf.append(((P * T[ridx]).sum(1) / n).mean())
    A, S = float(np.mean(adj)), float(np.mean(shuf))
    print(f"  {name:24s} adj={A:.4f}  shuffled={S:.4f}  signal={A-S:+.4f}")


# candidates -----------------------------------------------------------
def exact_next(traj):       return traj[:-1], traj[1:]
def avg_next2(traj):        return traj[:-2], (traj[1:-1] + traj[2:]) / 2.0
def avg_next3(traj):        return traj[:-3], (traj[1:-2] + traj[2:-1] + traj[3:]) / 3.0
def delta(traj):            d = traj[1:] - traj[:-1];  return d[:-1], d[1:]
def recenter_delta(traj):   # predict change, but relative to a rolling centroid
    c = traj.mean(0)
    d = traj[1:] - traj[:-1]
    return d[:-1], d[1:]
def direction(traj):        # predict next DIRECTION (normalized change)
    d = traj[1:] - traj[:-1]
    dn = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9)
    return dn[:-1], dn[1:]


trajs, dim = load_trajs()
print(f"trajectories={len(trajs)} dim={dim}\n")
print("forward-prediction target learnability:")
run("exact next (baseline)", exact_next)
run("avg next 2 (smoother)", avg_next2)
run("avg next 3 (smoother)", avg_next3)
run("delta (change)", delta)
run("delta recentred", recenter_delta)
run("direction (unit d)", direction)
