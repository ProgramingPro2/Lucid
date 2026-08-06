import sys; sys.path.insert(0, ".")
import numpy as np
from src.conceptpoc.persistence import load_joint
from src.conceptpoc.codec.jepa_codec import JepaCodec
from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.data.corpus import load_documents

model, tok = load_joint("artifacts/jepa_v3_128t/jepa/joint_robust.pt", device="cpu")
codec = JepaCodec(model, tok, 128, device="cpu")
docs = load_documents("data/austen_corpus.txt")
# find a doc with many sentences
big = None
for d in docs:
    s = [x for x in d.sentences if x.strip()]
    if len(s) >= 6:
        big = s; break
v = codec.encode(big)
print("num sents", len(big), "shape", v.shape)
a, b = v[:-1], v[1:]
print("norms a", np.round(np.linalg.norm(a, axis=1), 3)[:6])
print("dot", np.round((a*b).sum(-1), 4)[:6])
denom = np.linalg.norm(a,-1)*np.linalg.norm(b,-1)
print("denom", np.round(denom,4)[:6])
c = (a*b).sum(-1)/denom
print("cos", np.round(c,4)[:6])
print("any nan in v:", np.isnan(v).any())
