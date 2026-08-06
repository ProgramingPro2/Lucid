"""Round 2: novel target ideas for forward JEPA, measured on real vectors.

Round 1 showed exact-next has NEGATIVE signal (adj 0.037 < shuffled 0.146) --
the over-spread space makes the next vector systematically dissimilar. New
ideas attack this:

A. removal of global offset (center within doc): maybe doc-mean removal
   reveals local adjacency that the global cloud hides.
B. learned-metric / contrastive: instead of raw cosine, ask whether ordering
   of candidate sentences is learnable (relative, not absolute).
C. predicting a linear TRANSFORM of next (low-rank) that is the *complement*
   of the common direction.
D. k-step skip with interpolation (predict mid-forward target).
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
    adj, shuf, ranks = [], [], []
    rng = np.random.RandomState(0)
    for traj in trajs:
        try:
            P, T = fn(traj)
        except Exception:
            continue
        if P is None or len(P) < 2 or P.shape[0] != T.shape[0]:
            continue
        n = np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1) + 1e-9
        adj.append(((P * T).sum(1) / n).mean())
        ridx = rng.permutation(T.shape[0])
        shuf.append(((P * T[ridx]).sum(1) / n).mean())
        try:
            r = np.linalg.matrix_rank(T - T.mean(0), tol=1e-2)
            ranks.append(r)
        except Exception:
            pass
    print(f"  {name:26s} adj={np.mean(adj):.4f} shuf={np.mean(shuf):.4f} "
          f"signal={np.mean(adj)-np.mean(shuf):+.4f} rank={np.mean(ranks) if ranks else '?'}")

# ---- candidates ------------------------------------------------------
def centered_adj(traj):
    # remove doc mean so global cloud offset is gone; predict within-doc deviation
    c = traj.mean(0)
    z = traj - c
    return z[:-1], z[1:]

def centered_avg2(traj):
    c = traj.mean(0); z = traj - c
    return z[:-2], (z[1:-1] + z[2:]) / 2.0

def skip2(traj):
    # predict the sentence after next (2-step), harder but different
    return traj[:-2], traj[2:]

def pseudo_inverse_next(traj):
    # predict next in the space orthogonal to the common doc direction
    c = traj.mean(0); c_hat = c / (np.linalg.norm(c) + 1e-9)
    z = traj - np.outer(traj @ c_hat, c_hat)  # remove component along doc mean dir
    return z[:-1], z[1:]

def first_vs_rest(traj):
    # trend: predict the 'slow' start-to-end direction incrementally
    P, T = traj[:-1], traj[1:]
    # target = signed projection along the doc's own trend direction
    trend = traj[-1] - traj[0]
    tn = trend / (np.linalg.norm(trend) + 1e-9)
    # predict scalar progress? -> make a 1-D target: projection of next onto trend
    t_1d = (T @ tn)[:, None] * tn
    return P, t_1d

trajs, dim = load_trajs()
print(f"trajectories={len(trajs)} dim={dim}\nround-2 ideas:")
run("doc-centered adj", centered_adj)
run("doc-centered avg2", centered_avg2)
run("skip-2 (harder)", skip2)
run("orthogonal-to-mean", pseudo_inverse_next)
run("trend-projection (1d)", first_vs_rest)
