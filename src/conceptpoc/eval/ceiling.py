"""Measure: how predictable are adjacent sentences in concept space?

If the mean cosine between *ground-truth* consecutive sentence embeddings is
low (~0.05), then the predictor's 0.048 cosine is near the physical ceiling and
chasing better next-sentence prediction is futile. If it's high (>0.15), the
reasoner has real headroom. This decides whether to invest in a bigger predictor.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
from src.conceptpoc.codec.jepa_codec import JepaCodec
from src.conceptpoc.data.corpus import load_documents
from src.conceptpoc.persistence import load_joint


def main() -> None:
    model, tok = load_joint("artifacts/jepa_v3_128t/jepa/joint_robust.pt")
    codec = JepaCodec(model, tok, max_tokens=128, device="cpu")
    docs = load_documents("data/corpus_1m.txt")
    print(f"loaded {len(docs)} documents")

    cosines = []
    order_hist = {}  # how often next is more similar to current than drawn random
    n = 0
    for d in docs[:200]:
        sents = d.sentences[:50]
        for i in range(1, len(sents)):
            if n >= 2000:
                break
            v0 = codec.encode([sents[i - 1]])[0]
            v1 = codec.encode([sents[i]])[0]
            dot = float(v0 @ v1)
            n0 = float(np.linalg.norm(v0))
            n1 = float(np.linalg.norm(v1))
            if n0 < 1e-8 or n1 < 1e-8:
                continue
            cosines.append(dot / (n0 * n1))
            n += 1
    a = np.array(cosines)
    print(f"sampled {len(a)} adjacent sentence pairs")
    print(f"adjacent-sentence cosine: mean={a.mean():.4f} median={np.median(a):.4f} std={a.std():.4f}")
    print(f"  p25={np.percentile(a,25):.4f} p75={np.percentile(a,75):.4f}")
    # Compare to predictor's own val_sample_cosine of ~0.048.
    print(f"predictor val_sample_cosine was ~0.048 -> ratio to ceiling={a.mean():.1%} "
          f"(1.0 = at ceiling, 0.25 = 4x headroom)")


if __name__ == "__main__":
    main()
