"""Ceiling gate for the routed encoder: adjacent-sentence cosine in corr channel.

The single-space encoder gave ~0.054. The routing hypothesis says the corr
channel (which the predictor now operates on) should show MUCH higher
adjacent-sentence cosine, because anti-collapse no longer flattens it.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
from src.conceptpoc.persistence import load_joint
from src.conceptpoc.codec.jepa_codec import JepaCodec
from src.conceptpoc.jepa.tokenizer import Tokenizer
from src.conceptpoc.data.corpus import load_documents


def main():
    model, tok = load_joint("artifacts/jepa_route_v1/jepa/joint.pt", device="cpu")
    codec = JepaCodec(model, tok, 128, device="cpu")
    cd = model.cfg.corr_dim if model.cfg.channel_split else model.cfg.dim
    print(f"channel_split={model.cfg.channel_split} corr_dim={cd} dim={model.cfg.dim}")

    docs = load_documents("data/corpus_1m.txt")
    # sample docs like original ceiling test
    cosines_full, cosines_corr = [], []
    n = 0
    for d in docs[:400]:
        sents = d.sentences[:50]
        for i in range(1, len(sents)):
            if n >= 2000:
                break
            v0 = codec.encode([sents[i-1]])[0]
            v1 = codec.encode([sents[i]])[0]
            c_full = float(v0 @ v1) / (np.linalg.norm(v0)*np.linalg.norm(v1))
            c_corr = float(v0[:cd] @ v1[:cd]) / (np.linalg.norm(v0[:cd])*np.linalg.norm(v1[:cd])+1e-9)
            cosines_full.append(c_full)
            cosines_corr.append(c_corr)
            n += 1
    a = np.array(cosines_full); b = np.array(cosines_corr)
    print(f"sampled {n} adjacent pairs")
    print(f"FULL space   adjacent-sentence cosine: mean={a.mean():.4f}")
    print(f"CORR channel adjacent-sentence cosine: mean={b.mean():.4f}")
    print(f"  (old single-space full was 0.054)")
    print("\nGATE:")
    if b.mean() > 0.054:
        print(f"  PASS: corr-channel adjacent cosine {b.mean():.4f} > 0.054 wall "
              f"({b.mean()/0.054:.1%}) -- routing broke the predictability wall!")
    else:
        print(f"  FAIL: corr-channel {b.mean():.4f} still at/below wall")


if __name__ == "__main__":
    main()
