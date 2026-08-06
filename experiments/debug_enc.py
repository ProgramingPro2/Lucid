import sys; sys.path.insert(0, ".")
import numpy as np
from src.conceptpoc.persistence import load_joint
from src.conceptpoc.codec.jepa_codec import JepaCodec
from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.data.corpus import load_documents

model, tok = load_joint("artifacts/jepa_v3_128t/jepa/joint_robust.pt", device="cpu")
codec = JepaCodec(model, tok, 128, device="cpu")
docs = load_documents("data/austen_corpus.txt")
s = [x for x in docs[0].sentences if x.strip()][:5]
v = codec.encode(s)
print("shape", v.shape)
print("norms", np.linalg.norm(v, axis=1)[:5])
print("means", v.mean(axis=1)[:5])
print("raw sample[0][:5]", v[0][:5])
# cosine between adjacent
a, b = v[:-1], v[1:]
c = (a*b).sum(-1)/(np.linalg.norm(a,-1)*np.linalg.norm(b,-1)+1e-9)
print("adj cosines", c)
