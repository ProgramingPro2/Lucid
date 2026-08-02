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
    # Generative decode sampling (jepa codec only). 0.0 temperature = greedy
    # (most faithful). top_k/top_p restrict the candidate set when sampling.
    decode_temperature: float = 0.0
    decode_top_k: int = 0
    decode_top_p: float = 0.0


@dataclass
class DecoderRobustConfig:
    """Contingency decoder fine-tune on predicted/perturbed concept vectors."""

    epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-4
    perturb_sigma: float = 0.08
    scheduled_sampling_start: float = 0.0
    scheduled_sampling_end: float = 0.5
    sampling_steps: int = 15


@dataclass
class DiffusionConfig:
    """Concept-space conditional diffusion head (replaces point regression)."""

    num_train_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # "epsilon" (predict noise) or "x0" (predict clean vector).
    predict_type: str = "epsilon"
    sampler: str = "ddim"
    sampling_steps: int = 30
    head_hidden: int = 1024
    head_layers: int = 3
    time_dim: int = 128
    # Optional extra Gaussian jitter after sampling (maps to rollout temperature).
    sample_jitter: float = 0.0


@dataclass
class PredictorConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 64
    # "regression" (legacy MSE+cosine point estimate) or "diffusion".
    head: str = "regression"
    # Loss weighting between MSE and (1 - cosine) on the predicted next vector.
    cosine_weight: float = 1.0
    mse_weight: float = 1.0
    # E10 conditioning (default off = exact legacy behavior). When enabled, the
    # predictor gains a learned segment/role embedding (0=instruction,
    # 1=response) and/or a theme-vector projection added to every position.
    use_segment_embedding: bool = False
    use_theme: bool = False
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


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
    # Anti-anisotropy (Wang & Isola uniformity) on the normalized concept
    # vectors. VICReg above is mean-invariant and does not prevent angular
    # collapse into a narrow cone (which inflates next_vector_cosine while
    # retrieval_top1 stays ~0). Default 0.0 keeps legacy behavior; experiment
    # configs enable it (e.g. 1.0) to keep the directional space spread out.
    uniformity_weight: float = 0.0
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
    # Held-out fraction of windows used to monitor prediction generalization and
    # select the best checkpoint (the encoder can over-train on reconstruction).
    val_fraction: float = 0.1
    # Stop early after this many prediction-phase epochs without val improvement
    # (0 disables early stopping; the best-by-val checkpoint is kept regardless).
    patience: int = 0


@dataclass
class InstructionConfig:
    """E10 instruction/theme conditioning data + training (default off).

    All network/HF/LLM access is lazy and opt-in via the explicit ``source`` and
    ``author_with_llm`` flags so nothing downloads or calls a server unless asked.
    """

    # "synthesize" (build from the local corpus with templates) or "borrow"
    # (download a small open instruction set via HF datasets).
    source: str = "synthesize"
    # Cache of encoded (instruction, response) concept pairs.
    cache_path: str = "artifacts/instructions/pairs.npz"

    # --- borrow ---
    hf_dataset: str = "databricks/databricks-dolly-15k"
    hf_split: str = "train"
    max_pairs: int = 256
    # Per-record field names (Dolly uses instruction/response; Alpaca too).
    instruction_field: str = "instruction"
    response_field: str = "response"

    # --- synthesize ---
    templates: list[str] = field(
        default_factory=lambda: ["Continue:", "Summarize:", "Write about {theme}"]
    )
    themes: list[str] = field(
        default_factory=lambda: ["nature", "the city", "memory", "the sea"]
    )
    max_response_sentences: int = 8
    # Opt-in LLM-authored instructions via the lemonade client wiring (lazy).
    author_with_llm: bool = False

    # Cap on sentences encoded per side, keeps windows within max_seq_len.
    max_instruction_sentences: int = 4


@dataclass
class LoopConfig:
    # Reward-ranked fine-tuning (RAFT/RWR): sample N rollouts per prompt, keep
    # the top fraction by reward, fine-tune the predictor on them.
    rollouts_per_prompt: int = 8
    keep_fraction: float = 0.25
    rollout_steps: int = 6
    iterations: int = 3
    finetune_epochs: int = 2
    # Number of prompts drawn (spread across the cached set) for a RAFT run.
    # Larger than the old hardcoded 8 so alignment does not overfit a tiny slice.
    n_prompts: int = 64
    # Blend of LLM-judge vs human reward when both are present (0..1 -> human).
    human_weight: float = 0.5


@dataclass
class EvalConfig:
    """Per-capability generation eval harness (see ``eval/harness.py``).

    ``eval_path`` may be a single ``.jsonl`` file or a directory of them (one
    file per capability). ``output_dir`` is resolved relative to
    ``artifacts_dir`` when not absolute. ``rating_app_url`` is only used by the
    optional ``--push-rating-app`` / ``--merge-human`` flags.
    """

    eval_path: str = "data/eval"
    rollout_steps: int = 6
    output_dir: str = "eval"
    rating_app_url: str = "http://localhost:8000"
    # When True, ``run_harness`` loads cached corpus vectors and reports
    # manifold-proximity / collapse metrics alongside judge scores.
    compute_latent_diagnostics: bool = True


@dataclass
class GpuGateConfig:
    """Polite GPU sharing: pause training while other processes actively compute.

    The trigger is per-process **compute (SM) utilization** via ``nvidia-smi
    pmon``, not VRAM, so a model merely parked in VRAM but idle (e.g. an LLM
    server between requests) is ignored. When other processes' combined SM usage
    exceeds ``foreign_util_pct``, the trainer offloads to CPU (releasing VRAM and
    dropping our GPU utilization to ~0) and polls every ``poll_interval_s``
    seconds, resuming after ``clear_polls`` consecutive readings below
    ``resume_util_pct``. Disabled by default; enable in a config or via the
    ``--gpu-gate`` CLI flag.
    """

    enabled: bool = False
    # Pause when other processes' combined SM utilization is at/above this %.
    # Set above idle desktop rendering noise and below an active game/LLM load.
    foreign_util_pct: float = 40.0
    # Resume once foreign SM utilization drops below this % (hysteresis).
    resume_util_pct: float = 20.0
    # pmon snapshots to average per probe (each ~1s); smooths transient spikes.
    pmon_samples: int = 2
    # Minimum wall-clock seconds between probes while training (bounds overhead).
    check_interval_s: float = 20.0
    # Seconds between probes while paused.
    poll_interval_s: float = 15.0
    # Consecutive clear readings required before resuming (debounces flapping).
    clear_polls: int = 3
    # Optional secondary trigger: also pause if other processes hold this many MB
    # of VRAM (0 = disabled, the default, so idle parked models do not pause us).
    mem_trigger_mb: float = 0.0
    # Offload model+optimizer to CPU and free VRAM when pausing (vs. compute-only
    # pause that keeps VRAM reserved). Recommended True so games can use the VRAM.
    release_memory: bool = True
    # GPU index to watch; None = the training device's index.
    device_index: int | None = None


@dataclass
class Config:
    artifacts_dir: Path = field(default_factory=_default_artifacts_dir)
    codec: CodecConfig = field(default_factory=CodecConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    jepa: JepaConfig = field(default_factory=JepaConfig)
    gpu_gate: GpuGateConfig = field(default_factory=GpuGateConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    instruction: InstructionConfig = field(default_factory=InstructionConfig)
    # Diffusion head hyperparameters (also nested under predictor.diffusion).
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    decoder_robust: DecoderRobustConfig = field(default_factory=DecoderRobustConfig)

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
