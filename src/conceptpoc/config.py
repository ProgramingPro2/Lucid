"""Central configuration.

Defaults are tuned for a single 24GB GPU (RTX 3090) but run on CPU for tests.
Override any field via a JSON file passed to ``Config.load`` or by constructing
``Config`` directly.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _default_artifacts_dir() -> Path:
    return Path(os.environ.get("CONCEPTPOC_ARTIFACTS", "artifacts")).resolve()


@dataclass
class CodecConfig:
    # "sentence-transformer" (default, runs anywhere), "sonar" (generative
    # decoding, optional install), "hashing" (deterministic, no downloads), or
    # "jepa" (our own jointly-trained encoder/decoder; loads a saved checkpoint).
    kind: str = "sentence-transformer"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Dimensionality for the hashing codec only; real codecs report their own.
    hashing_dim: int = 256
    # Size of the sentence bank used by the retrieval decoder.
    decoder_bank_size: int = 50_000
    device: str = "auto"
    # Checkpoint produced by `train-joint`; loaded when kind == "jepa".
    # If None, defaults to <artifacts_dir>/jepa/joint.pt.
    jepa_checkpoint: str | None = None


@dataclass
class PredictorConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 64
    # Loss weighting between MSE and (1 - cosine) on the predicted next vector.
    cosine_weight: float = 1.0
    mse_weight: float = 1.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_steps: int = 100
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    seed: int = 0


@dataclass
class DataConfig:
    corpus_path: str = "data/sample_corpus.txt"
    min_sentence_chars: int = 12
    max_sentence_chars: int = 400
    # Concept-sequence length fed to the predictor during training.
    context_len: int = 16


@dataclass
class JepaConfig:
    """Joint self-supervised encoder/decoder + predictor (the "own codec" path).

    The design is built around defeating representation collapse: a
    reconstruction anchor, an EMA stop-grad prediction target, and VICReg-style
    variance/covariance regularization. See docs/ARCHITECTURE.md.
    """

    # Concept (bottleneck) dimensionality.
    dim: int = 512
    # Tokenizer: a pretrained BPE name (e.g. "gpt2") or "char" for the offline
    # dependency-free fallback used in tests.
    tokenizer: str = "gpt2"
    max_tokens: int = 48

    # Encoder transformer.
    enc_layers: int = 4
    enc_heads: int = 8
    enc_ff_mult: int = 4
    # Decoder transformer (autoregressive, conditioned on the concept vector).
    dec_layers: int = 4
    dec_heads: int = 8
    dec_ff_mult: int = 4
    d_model: int = 512
    dropout: float = 0.1

    # Curriculum: reconstruction-only warmup, then ramp the prediction weight.
    recon_warmup_epochs: int = 3
    alpha_warmup_epochs: int = 3
    alpha_max: float = 1.0

    # Anti-collapse.
    ema_momentum: float = 0.99
    var_weight: float = 1.0
    cov_weight: float = 0.04
    var_margin: float = 1.0
    # Warn if the mean per-dim std of the unnormalized embedding falls below
    # this (collapse alarm). Healthy training pushes it toward var_margin.
    collapse_std_threshold: float = 0.1

    # Optimization.
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 12
    grad_clip: float = 1.0
    context_len: int = 16
    seed: int = 0


@dataclass
class LoopConfig:
    # Reward-ranked fine-tuning (RAFT/RWR): sample N rollouts per prompt, keep
    # the top fraction by reward, fine-tune the predictor on them.
    rollouts_per_prompt: int = 8
    keep_fraction: float = 0.25
    rollout_steps: int = 6
    iterations: int = 3
    finetune_epochs: int = 2
    # Blend of LLM-judge vs human reward when both are present (0..1 -> human).
    human_weight: float = 0.5


@dataclass
class Config:
    artifacts_dir: Path = field(default_factory=_default_artifacts_dir)
    codec: CodecConfig = field(default_factory=CodecConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    jepa: JepaConfig = field(default_factory=JepaConfig)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        cfg = cls()
        if path is None:
            return cfg
        raw = json.loads(Path(path).read_text())
        return cfg.merged(raw)

    def merged(self, raw: dict) -> "Config":
        def apply(obj, updates: dict):
            for key, value in updates.items():
                if not hasattr(obj, key):
                    raise KeyError(f"Unknown config key: {key}")
                current = getattr(obj, key)
                if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                    apply(current, value)
                elif key == "artifacts_dir":
                    setattr(obj, key, Path(value).resolve())
                else:
                    setattr(obj, key, value)

        apply(self, raw)
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["artifacts_dir"] = str(self.artifacts_dir)
        return d

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
