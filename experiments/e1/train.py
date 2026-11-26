"""E1 training harness: one arm, one seed, on a real causal LM.

This is the machinery that produces the numbers, not a result by itself. It
implements the protocol in ``docs/PROTOCOL.md`` §5:

* the **student** is a Qwen3 base model with LoRA adapters and an optional
  inserted :class:`~experiments.e1.block.JEPABlock` (or a parameter-matched
  head for A'/B');
* the **teacher** is the *same weights* with the adapter disabled and no block
  contribution — no EMA in the main arms (the protocol's EMA curves are a G6
  mechanism check, not the confirmatory contrast);
* the JEPA objective predicts the teacher's hidden state at ``t + delta`` from
  the student's state at ``t``;
* the block output projection is zero-initialised, so A, B and E are
  bit-identical at step 0.

Evaluation returns the two pre-registered endpoints:

``P1`` context gain = ``NLL(shuffled context) - NLL(true context)`` and
``P3`` necessity = ``P1(normal) - P1(block ablated)``.

Everything is seeded through :mod:`experiments.protocol.seeds`, so arms do not
share or shift each other's data order.

Usage (from the repository root)::

    PYTHONPATH=src .venv/bin/python -m experiments.e1.train \
        --arm A --seed 0 --tokens 2000000 \
        --corpus data/corpus_v2.txt \
        --out artifacts/e1/A_seed0.json

Use ``--tiny`` for a CPU smoke test that never downloads a model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.e1.arms import by_name, build_head  # noqa: E402
from experiments.e1.block import JEPABlock  # noqa: E402
from experiments.e1.data import PackedCorpus, target_labels  # noqa: E402
from experiments.e1.objective import (  # noqa: E402
    combined_loss,
    context_gain,
    representation_stats,
)
from experiments.protocol.manifest import (  # noqa: E402
    RunManifest,
    capture_git_commit,
    hash_config,
    spawn_keys_for_seed,
)
from experiments.protocol.qualify import NanCounter, microbatch_nonfinite  # noqa: E402
from experiments.protocol.seeds import spawn_key, token_stream_sha256  # noqa: E402

# Arm -> (block, jepa, head, target_mode). ``by_name`` owns the protocol table;
# this map only says how each role is wired.
_ARM_WIRING: dict[str, dict[str, object]] = {
    "A": {"block": True, "jepa": True, "head": False, "scramble": False},
    "B": {"block": True, "jepa": False, "head": False, "scramble": False},
    "A_prime": {"block": False, "jepa": True, "head": True, "scramble": False},
    "B_prime": {"block": False, "jepa": False, "head": True, "scramble": False},
    "E": {"block": True, "jepa": True, "head": False, "scramble": True},
    "D": {"block": False, "jepa": False, "head": False, "scramble": False},
    "N": {"block": True, "jepa": False, "head": False, "scramble": False},
    "PC_hi": {"block": False, "jepa": False, "head": False, "scramble": False},
    "PC_lo": {"block": False, "jepa": False, "head": False, "scramble": False},
}


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen3-1.7B-Base"
    arm: str = "A"
    seed: int = 0
    tokens: int = 2_000_000
    context: int = 512
    batch_size: int = 2
    grad_accum: int = 4
    lr: float = 1e-4
    jepa_weight: float = 0.1
    delta: int = 8
    layer: int | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lambda_var: float = 1.0
    lambda_cov: float = 0.04
    use_cosine: bool = True
    eval_items: int = 64
    eval_prefix: int = 384
    eval_target: int = 128
    eval_batch: int = 4
    dtype: str = "bfloat16"
    device: str = "cuda"
    log_every: int = 25
    max_steps: int | None = None
    # Data construction.
    corpus: str = "data/corpus_v2.txt"
    corpus_sep: str = "===DOC==="
    test_fraction: float = 0.1
    # The split and the eval items must be identical for every arm and every
    # seed (protocol §5.5: "one fixed test set"). Only the *training* order is
    # keyed by the run seed.
    split_seed: int = 20260601
    eval_seed: int = 20260602
    max_docs: int | None = None
    max_tokens: int | None = None
    cache: bool = True
    cache_dir: str = "artifacts/e1/cache"
    # Held-out dev fraction of the *full* corpus. The sealed test split is
    # taken first, at ``test_fraction``, and is not resized to make room.
    dev_fraction: float = 0.05
    # docs/EXPERIMENTS.md §9: checkpoint every 500 steps, keep all of them,
    # and always keep a final checkpoint. There is no best-checkpoint
    # selection; choosing one by its score is optional stopping.
    checkpoint_every: int = 500
    checkpoint_dir: str = "artifacts/e1/checkpoints"


# --------------------------------------------------------------------------
# Model plumbing
# --------------------------------------------------------------------------


def find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the first ``ModuleList`` of transformer layers in ``model``.

    Works whether ``model`` is a bare ``*ForCausalLM`` or a PEFT wrapper,
    without hard-coding the wrapper's attribute names (which change between
    transformers versions).
    """
    for _, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return layers
    raise ValueError("could not find the decoder layer list in the model")


class E1Model(torch.nn.Module):
    """A base LM plus the E1 block/head, with an adapter-free teacher path.

    The block is applied by a forward hook on layer ``layer_index``. Its output
    is added to that layer's hidden state *before* the model stores it in
    ``hidden_states``, so ``hidden_states[layer_index + 1]`` is the post-block
    student state. The teacher runs exactly the same hook but with
    ``_teacher_mode`` set, so it gets the untouched base state.
    """

    def __init__(
        self,
        base: torch.nn.Module,
        *,
        block: JEPABlock | None = None,
        head: torch.nn.Module | None = None,
        layer_index: int | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.block = block
        self.head = head
        layers = find_decoder_layers(base)
        if layer_index is None:
            layer_index = len(layers) // 2
        if not 0 <= layer_index < len(layers):
            raise ValueError(
                f"layer_index {layer_index} outside 0..{len(layers) - 1}"
            )
        self.layer_index = int(layer_index)
        self._teacher_mode = False
        self._stats: dict[str, float] = {}
        self._handle = layers[self.layer_index].register_forward_hook(self._hook)

    def remove_hooks(self) -> None:
        self._handle.remove()

    def _hook(self, module, args, output):  # noqa: ANN001 - torch API
        hidden = output[0] if isinstance(output, tuple) else output
        if self.block is not None and not self._teacher_mode:
            delta = self.block(hidden)
            # The washout diagnostic is the block's contribution on a *clean*
            # forward. ``evaluate`` then ablates the block and forwards again;
            # writing stats on that second pass stores the zero contribution
            # and every run reports block_fraction 0. An ablated forward must
            # leave the last clean measurement in place.
            if not self.training and not self.block._ablated:
                with torch.no_grad():
                    self._stats["block_rms"] = float(
                        delta.detach().pow(2).mean().sqrt()
                    )
                    self._stats["residual_rms"] = float(
                        hidden.detach().pow(2).mean().sqrt()
                    )
            hidden = hidden + delta
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden
        return output

    def forward(
        self, input_ids: torch.Tensor, *, teacher: bool = False, **kwargs
    ):
        """Run one forward pass and return the raw model output.

        ``teacher=True`` disables the adapter and the block contribution. It is
        wrapped in ``no_grad`` because the teacher is never differentiated.
        """
        self._teacher_mode = teacher
        call = dict(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )
        if teacher and hasattr(self.base, "disable_adapter"):
            with self.base.disable_adapter(), torch.no_grad():
                return self.base(**call)
        if teacher:
            with torch.no_grad():
                return self.base(**call)
        return self.base(**call)

    # -- state extraction ---------------------------------------------------

    def student_states(self, hidden_states) -> torch.Tensor:  # noqa: ANN001
        h = hidden_states[self.layer_index + 1]
        return self.head(h) if self.head is not None else h

    def teacher_states(self, hidden_states) -> torch.Tensor:  # noqa: ANN001
        return hidden_states[self.layer_index + 1]

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


def build_base(cfg: TrainConfig, *, tiny: bool = False):
    """Load the backbone (or build a tiny one for smoke tests)."""
    if tiny:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(cfg.seed)
        config = Qwen3Config(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=2048,
        )
        return Qwen3ForCausalLM(config)

    from transformers import AutoModelForCausalLM

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    kwargs: dict = {"torch_dtype": dtype}
    if cfg.device == "cuda":
        kwargs["device_map"] = {"": 0}
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    if cfg.device != "cuda":
        base.to(cfg.device)
    return base


def attach_lora(
    base: torch.nn.Module, cfg: TrainConfig, *, tiny: bool = False
) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    if tiny:
        # The tiny model is plain ``nn.Linear`` under the same attribute names,
        # so the same target list works and the test exercises the real path.
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base, lora)


def build_e1_model(
    cfg: TrainConfig, *, tiny: bool = False
) -> tuple[E1Model, object]:
    """Build the student and return ``(wrapper, tokenizer_or_None)``."""
    base = build_base(cfg, tiny=tiny)
    base = attach_lora(base, cfg, tiny=tiny)
    # ``use_cache`` must stay off while training; PEFT models pass kwargs through.
    base.config.use_cache = False

    wiring = _ARM_WIRING[cfg.arm]
    d_model = int(base.config.hidden_size)
    base_dtype = next(base.parameters()).dtype

    block = None
    if wiring["block"]:
        torch.manual_seed(spawn_key(cfg.seed, "block_init"))
        block = JEPABlock(
            d_model,
            n_heads=_n_heads(d_model),
            zero_init_output=cfg.arm != "N",
        ).to(dtype=base_dtype)
        if cfg.arm == "N":
            # Nuisance arm: a *frozen random* nonlinearity. Zero-init would
            # make it the identity, which is not the nuisance we want to size.
            for p in block.parameters():
                p.requires_grad_(False)

    head = None
    if wiring["head"]:
        torch.manual_seed(spawn_key(cfg.seed, "head_init"))
        head = build_head(d_model).to(dtype=base_dtype)

    model = E1Model(base, block=block, head=head, layer_index=cfg.layer)
    model.to(cfg.device)
    return model, None


def _n_heads(d_model: int) -> int:
    for h in (16, 12, 8, 4, 2):
        if d_model % h == 0:
            return h
    return 1


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


def read_documents(path: str, sep: str = "===DOC===") -> list[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    parts = text.split(sep)
    return [p.strip() for p in parts if p.strip()]


def load_corpus(
    cfg: TrainConfig, tokenizer, *, tiny: bool = False
) -> tuple[PackedCorpus, PackedCorpus, PackedCorpus]:
    if tiny:
        # The synthetic corpus is part of the *data*, not the run: it must be
        # identical for every arm and seed so the tiny path honours §5.5 too.
        rng = np.random.default_rng(cfg.split_seed)
        docs = [
            list(rng.integers(0, 512, size=1200, dtype=np.int64))
            for _ in range(40)
        ]
        corpus = PackedCorpus(docs, eos_id=0)
    else:
        import hashlib
        from pathlib import Path

        texts = read_documents(cfg.corpus, cfg.corpus_sep)
        if cfg.max_docs is not None:
            texts = texts[: cfg.max_docs]

        cache_path = None
        if cfg.cache:
            key_src = "|".join(
                [
                    str(Path(cfg.corpus).resolve()),
                    str(Path(cfg.corpus).stat().st_mtime_ns),
                    cfg.model_id,
                    str(cfg.max_docs),
                    str(cfg.max_tokens),
                    str(cfg.context + cfg.eval_prefix + cfg.eval_target),
                ]
            )
            key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
            cache_path = Path(cfg.cache_dir) / f"corpus_{key}.npz"

        if cache_path is not None and cache_path.exists():
            corpus = PackedCorpus.from_cache(cache_path)
            print(f"[data] loaded cached corpus {cache_path}", flush=True)
        else:
            corpus = PackedCorpus.from_texts(
                texts,
                tokenizer,
                max_docs=cfg.max_docs,
                max_tokens=cfg.max_tokens,
                min_tokens=cfg.context + cfg.eval_prefix + cfg.eval_target,
            )
            if cache_path is not None:
                corpus.save(cache_path)
                print(
                    f"[data] cached {len(corpus.docs)} docs / "
                    f"{corpus.n_tokens} tokens at {cache_path}",
                    flush=True,
                )
    return corpus.split_with_dev(
        cfg.test_fraction, cfg.dev_fraction, seed=cfg.split_seed
    )


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def _batch_tokens(cfg: TrainConfig) -> int:
    return cfg.batch_size * cfg.grad_accum * cfg.context


def _n_steps(cfg: TrainConfig) -> int:
    steps = math.ceil(cfg.tokens / _batch_tokens(cfg))
    return min(steps, cfg.max_steps) if cfg.max_steps is not None else steps


def save_checkpoint(model: E1Model, directory: Path, *, step: int) -> Path:
    """Write one checkpoint and return its directory.

    Every call creates a new directory. Nothing here deletes an older
    checkpoint or picks a best one: docs/EXPERIMENTS.md §9 forbids selecting
    the checkpoint by its score, and the analysis reads only the final one.
    The adapter (or the full module, if this is not a PEFT model) and the
    inserted block/head are stored separately, because the block is not part
    of the backbone the adapter wraps.
    """
    directory.mkdir(parents=True, exist_ok=True)
    base = model.base
    if hasattr(base, "save_pretrained"):
        base.save_pretrained(directory / "adapter")
    else:
        torch.save(base.state_dict(), directory / "base.pt")
    if model.block is not None:
        torch.save(model.block.state_dict(), directory / "block.pt")
    if model.head is not None:
        torch.save(model.head.state_dict(), directory / "head.pt")
    (directory / "trainer_state.json").write_text(
        json.dumps({"step": step}) + "\n", encoding="utf-8"
    )
    return directory


def _checkpoint_root(cfg: TrainConfig) -> Path:
    return Path(cfg.checkpoint_dir) / f"{cfg.arm}_s{cfg.seed}"


@torch.no_grad()
def held_out_nll(cfg: TrainConfig, model: E1Model, corpus: PackedCorpus) -> float:
    """Mean token NLL of a small fixed batch from ``corpus``.

    This is the cheap dev measurement the convergence rule reads at each log
    interval. The batch is redrawn from ``split_seed`` every call, so the
    number is comparable across steps. It must not be called on the sealed
    test split: test is read once per claim, at the end, by :func:`evaluate`.
    """
    was_training = model.training
    model.eval()
    try:
        rng = np.random.default_rng(cfg.split_seed)
        batches = []
        for _ in range(2):
            x_np, _ = corpus.train_batch(rng, cfg.batch_size, cfg.context)
            input_ids = torch.as_tensor(x_np, dtype=torch.long, device=cfg.device)
            logits = model(input_ids).logits
            logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            target = input_ids[:, 1:]
            token = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            batches.append(-token.mean())
        return float(torch.stack(batches).mean())
    finally:
        model.train(was_training)


def _corpus_sha256(corpus: PackedCorpus) -> str:
    """Hash the packed training stream in stored document order.

    The order is the split, not the seed-dependent shuffle, so every arm of a
    seed hashes the same bytes. The sealed test documents are not included:
    they are not part of the stream the model trains on.
    """
    def _tokens():
        for doc in corpus.docs:
            yield from doc

    return token_stream_sha256(_tokens())


def train(
    cfg: TrainConfig,
    model: E1Model,
    train_corpus: PackedCorpus,
    *,
    dev_corpus: PackedCorpus | None = None,
    tiny: bool = False,
) -> dict:
    """Run the arm's optimisation loop and return a metrics dictionary."""
    wiring = _ARM_WIRING[cfg.arm]
    jepa_on = bool(wiring["jepa"]) and cfg.jepa_weight > 0.0
    if wiring["scramble"] and cfg.batch_size * cfg.grad_accum < 2:
        raise ValueError("arm E needs at least two sequences to scramble targets")
    if cfg.checkpoint_every < 1:
        raise ValueError(
            f"checkpoint_every must be positive, got {cfg.checkpoint_every}"
        )
    lr = 0.0 if cfg.arm == "PC_lo" else cfg.lr

    params = model.trainable_parameters()
    if not params:
        raise RuntimeError("no trainable parameters (is LoRA attached?)")
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    rng = np.random.default_rng(spawn_key(cfg.seed, "data"))
    n_steps = _n_steps(cfg)
    tokens_seen = 0
    history: list[dict] = []
    started = time.time()
    model.train()
    nan_counter = NanCounter()

    global_step = 0
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        step_lm = 0.0
        step_jepa = 0.0
        step_diag: dict | None = None
        step_stats: dict[str, float] | None = None
        micro_done = 0
        tainted = False
        will_log = (step + 1) % cfg.log_every == 0 or (step + 1) == n_steps
        for micro in range(cfg.grad_accum):
            x_np, _ = train_corpus.train_batch(
                rng, cfg.batch_size, cfg.context
            )
            input_ids = torch.as_tensor(
                x_np, dtype=torch.long, device=cfg.device
            )
            out = model(input_ids)
            logits = out.logits
            lm = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )
            loss = lm
            diag = None
            if jepa_on:
                s = model.student_states(out.hidden_states)
                t_out = model(input_ids, teacher=True)
                t_states = model.teacher_states(t_out.hidden_states)
                if wiring["scramble"]:
                    # Each row predicts another row's future. Roll by one keeps
                    # the distribution of targets but destroys correspondence.
                    t_states = torch.roll(t_states, shifts=1, dims=0)
                jepa, diag = combined_loss(
                    s.float(),
                    t_states.float(),
                    cfg.delta,
                    lambda_var=cfg.lambda_var,
                    lambda_cov=cfg.lambda_cov,
                    use_cosine=cfg.use_cosine,
                )
                loss = loss + cfg.jepa_weight * jepa
                step_jepa += float(jepa.detach())
            # A non-finite loss or gradient is discarded and counted. Applying
            # it would poison the weights, and an arm that skips more often
            # than another has been trained on a different distribution
            # (docs/PROTOCOL.md §10). The whole optimiser step is skipped: a
            # bad microbatch has already been added into the accumulated grad.
            if microbatch_nonfinite(params, loss):
                nan_counter.record()
                opt.zero_grad(set_to_none=True)
                tainted = True
                tokens_seen += int(input_ids.numel())
                micro_done += 1
                break
            (loss / cfg.grad_accum).backward()
            if microbatch_nonfinite(params):
                nan_counter.record()
                opt.zero_grad(set_to_none=True)
                tainted = True
                tokens_seen += int(input_ids.numel())
                micro_done += 1
                break
            step_lm += float(lm.detach())
            step_diag = diag
            tokens_seen += int(input_ids.numel())
            micro_done += 1
            if will_log and micro == cfg.grad_accum - 1:
                step_stats = representation_stats(
                    model.student_states(out.hidden_states)
                )

        if tainted:
            global_step += 1
            if will_log:
                print(
                    f"[{cfg.arm} s{cfg.seed}] step {global_step}/{n_steps} "
                    f"skipped non-finite microbatch "
                    f"(nan_skips={nan_counter.total})",
                    flush=True,
                )
            continue

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        global_step += 1

        if global_step % cfg.checkpoint_every == 0:
            # Kept alongside every other interval checkpoint. Not compared.
            save_checkpoint(
                model,
                _checkpoint_root(cfg) / f"step_{global_step:06d}",
                step=global_step,
            )

        if global_step % cfg.log_every == 0 or global_step == n_steps:
            denom = max(micro_done, 1)
            val_nll = (
                held_out_nll(cfg, model, dev_corpus)
                if dev_corpus is not None
                else None
            )
            rec = {
                "step": global_step,
                "lm": step_lm / denom,
                "jepa": step_jepa / denom if jepa_on else None,
                "var": None if step_diag is None else step_diag.get("var"),
                "cov": None if step_diag is None else step_diag.get("cov"),
                "std_z": None if step_stats is None else step_stats["std_z"],
                "effective_rank": (
                    None if step_stats is None else step_stats["effective_rank"]
                ),
                "val_nll": val_nll,
                "tokens": tokens_seen,
                "seconds": round(time.time() - started, 2),
                "nan_skips": nan_counter.total,
            }
            history.append(rec)
            extra = ""
            if rec["jepa"] is not None:
                extra += f" jepa={rec['jepa']:.4f}"
            if rec["var"] is not None:
                extra += f" var={rec['var']:.4f} cov={rec['cov']:.4f}"
            if rec["std_z"] is not None:
                extra += (
                    f" std_z={rec['std_z']:.4f}"
                    f" erank={rec['effective_rank']:.2f}"
                )
            if val_nll is not None:
                extra += f" val_nll={val_nll:.4f}"
            print(
                f"[{cfg.arm} s{cfg.seed}] step {global_step}/{n_steps} "
                f"lm={rec['lm']:.4f}{extra} tokens={tokens_seen}",
                flush=True,
            )

    # The final checkpoint is always written, including when the last step
    # was also an interval checkpoint. Analysis uses this path and no other.
    final_checkpoint = save_checkpoint(
        model, _checkpoint_root(cfg) / "final", step=global_step
    )
    elapsed = time.time() - started
    return {
        "steps": global_step,
        "tokens_seen": tokens_seen,
        "lr": lr,
        "jepa_on": jepa_on,
        "seconds": round(elapsed, 2),
        "tokens_per_second": tokens_seen / elapsed if elapsed > 0 else 0.0,
        "history": history,
        "nan_skips": nan_counter.total,
        "final_checkpoint": str(final_checkpoint),
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def masked_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean negative log-likelihood over the labelled positions."""
    valid = labels != -100
    if int(valid.sum()) == 0:
        raise ValueError("no labelled positions")
    logp = torch.log_softmax(logits.float(), dim=-1)
    safe = labels.clamp_min(0)
    token = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return float(-(token * valid).sum() / valid.sum())


@torch.no_grad()
def evaluate(
    cfg: TrainConfig, model: E1Model, items, *, tiny: bool = False
) -> dict:
    """Compute P1 (context gain) and P3 (necessity) on paired eval items."""
    model.eval()
    wiring = _ARM_WIRING[cfg.arm]
    has_block = wiring["block"]
    has_head = wiring["head"]

    gains: list[float] = []
    nll_true_all: list[float] = []
    nll_shuf_all: list[float] = []
    ablated_gains: list[float] = []
    # Snapshot of the last *clean* forward. The ablated re-forward must not
    # be what the washout diagnostic reports.
    clean_stats: dict[str, float] = {}

    for start in range(0, len(items), cfg.eval_batch):
        batch = items[start : start + cfg.eval_batch]
        true_ids = torch.tensor(
            [it.true_ids for it in batch], dtype=torch.long, device=cfg.device
        )
        shuf_ids = torch.tensor(
            [it.shuffled_ids for it in batch], dtype=torch.long, device=cfg.device
        )
        labels = torch.tensor(
            [target_labels(it.true_ids, it.target_start) for it in batch],
            dtype=torch.long,
            device=cfg.device,
        )

        def nll_for(ids: torch.Tensor) -> list[float]:
            out = model(ids)
            return [
                masked_nll(out.logits[row], labels[row])
                for row in range(ids.size(0))
            ]

        nll_true = nll_for(true_ids)
        nll_shuf = nll_for(shuf_ids)
        if model.block is not None and not model.block._ablated and model._stats:
            clean_stats = dict(model._stats)
        gains.extend(
            float(g) for g in context_gain(
                torch.tensor(nll_true), torch.tensor(nll_shuf)
            )
        )
        nll_true_all.extend(nll_true)
        nll_shuf_all.extend(nll_shuf)

        if has_block and model.block is not None:
            model.block.set_ablation(True)
            try:
                ab_true = nll_for(true_ids)
                ab_shuf = nll_for(shuf_ids)
            finally:
                model.block.set_ablation(False)
            ablated_gains.extend(
                float(g) for g in context_gain(
                    torch.tensor(ab_true), torch.tensor(ab_shuf)
                )
            )

    gains_arr = np.asarray(gains, dtype=float)
    result: dict = {
        "n_items": len(gains_arr),
        "nll_true": float(np.mean(nll_true_all)),
        "nll_shuffled": float(np.mean(nll_shuf_all)),
        "context_gain_mean": float(gains_arr.mean()),
        "context_gain_sd": float(gains_arr.std(ddof=1)) if len(gains_arr) > 1 else 0.0,
        "context_gain_per_item": gains,
        "has_block": has_block,
        "has_head": has_head,
    }
    if ablated_gains:
        ab = np.asarray(ablated_gains, dtype=float)
        necessity = gains_arr - ab
        result.update(
            {
                "context_gain_ablated_mean": float(ab.mean()),
                "necessity_mean": float(necessity.mean()),
                "necessity_sd": (
                    float(necessity.std(ddof=1)) if len(necessity) > 1 else 0.0
                ),
                "necessity_per_item": necessity.tolist(),
            }
        )
        # Washout diagnostic: if the block's contribution is a tiny fraction of
        # the residual, a null means "washed out", not "unhelpful". Read the
        # snapshot taken before ablation, not the live stats the ablated
        # forward would have overwritten.
        if "block_rms" in clean_stats and "residual_rms" in clean_stats:
            result["block_rms"] = clean_stats["block_rms"]
            result["residual_rms"] = clean_stats["residual_rms"]
            result["block_fraction"] = (
                clean_stats["block_rms"]
                / max(clean_stats["residual_rms"], 1e-12)
            )
    return result


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _peak_vram_gb(device: str) -> float:
    """Peak allocated VRAM in GiB, or 0 when the run is not on CUDA.

    ``max_memory_allocated`` is the quantity the manifest records. A CPU run
    has no VRAM figure; 0 is an honest reading, not a missing field.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    return 0.0


def run(cfg: TrainConfig, *, tiny: bool = False, tokenizer=None) -> dict:
    started = time.perf_counter()
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(spawn_key(cfg.seed, "lora_init"))
    model, _ = build_e1_model(cfg, tiny=tiny)
    if not tiny:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    train_corpus, dev_corpus, test_corpus = load_corpus(cfg, tokenizer, tiny=tiny)
    eval_items = test_corpus.eval_items(
        cfg.eval_items,
        prefix=cfg.eval_prefix,
        target=cfg.eval_target,
        seed=cfg.eval_seed,
    )
    # Every arm must see the same evaluation items. Hash them so the manifest
    # can prove it rather than assert it.
    import hashlib

    digest = hashlib.sha256()
    for it in eval_items:
        digest.update(bytes(str(it.target_start), "ascii"))
        digest.update(np.asarray(it.true_ids, dtype=np.int64).tobytes())
        digest.update(np.asarray(it.shuffled_ids, dtype=np.int64).tobytes())
    eval_fingerprint = digest.hexdigest()

    train_metrics = train(
        cfg, model, train_corpus, dev_corpus=dev_corpus, tiny=tiny
    )
    eval_metrics = evaluate(cfg, model, eval_items, tiny=tiny)
    wall_clock_s = time.perf_counter() - started
    record = RunManifest(
        run_id=f"e1/{cfg.arm}/s{cfg.seed}",
        git_commit=capture_git_commit(_REPO_ROOT),
        config_sha256=hash_config(asdict(cfg)),
        corpus_sha256=_corpus_sha256(train_corpus),
        spawn_keys=spawn_keys_for_seed(cfg.seed),
        token_budget=int(cfg.tokens),
        final_checkpoint=str(train_metrics["final_checkpoint"]),
        wall_clock_s=float(wall_clock_s),
        peak_vram_gb=_peak_vram_gb(cfg.device),
        nan_skips=int(train_metrics["nan_skips"]),
        arms=[cfg.arm],
        notes=f"arm={cfg.arm} seed={cfg.seed}",
    )
    return {
        "config": asdict(cfg),
        "arm": cfg.arm,
        "role": by_name(cfg.arm).role,
        "seed": cfg.seed,
        "manifest": {
            "layer_index": model.layer_index,
            "d_model": int(model.base.config.hidden_size),
            "n_trainable": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "n_block": (
                sum(p.numel() for p in model.block.parameters())
                if model.block is not None
                else 0
            ),
            "n_head": (
                sum(p.numel() for p in model.head.parameters())
                if model.head is not None
                else 0
            ),
            "eval_fingerprint": eval_fingerprint,
            "train_docs": len(train_corpus.docs),
            "dev_docs": len(dev_corpus.docs),
            "test_docs": len(test_corpus.docs),
        },
        "run_manifest": record.to_dict(),
        "train": train_metrics,
        "eval": eval_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="A", choices=list(_ARM_WIRING))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokens", type=int, default=2_000_000)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--corpus", default="data/corpus_v2.txt")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--jepa-weight", type=float, default=0.1)
    p.add_argument("--delta", type=int, default=8)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--eval-items", type=int, default=64)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--checkpoint-dir", default="artifacts/e1/checkpoints")
    p.add_argument("--dev-fraction", type=float, default=0.05)
    p.add_argument("--tiny", action="store_true", help="CPU smoke test, no download")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    cfg = TrainConfig(
        model_id=args.model,
        arm=args.arm,
        seed=args.seed,
        tokens=args.tokens,
        context=args.context,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        jepa_weight=args.jepa_weight,
        delta=args.delta,
        layer=args.layer,
        eval_items=args.eval_items,
        eval_prefix=min(384, args.context // 2),
        eval_target=min(128, args.context // 4),
        device=args.device,
        dtype=args.dtype,
        log_every=args.log_every,
        max_steps=args.max_steps,
        corpus=args.corpus,
        max_docs=args.max_docs,
        max_tokens=args.max_tokens,
        dev_fraction=args.dev_fraction,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(
        f"E1 arm={cfg.arm} seed={cfg.seed} model={cfg.model_id} "
        f"tokens={cfg.tokens} device={cfg.device}",
        flush=True,
    )
    result = run(cfg, tiny=args.tiny)
    ev = result["eval"]
    record = result["run_manifest"]
    history = result["train"]["history"]
    val_nll = history[-1]["val_nll"] if history else None
    print("-" * 60)
    print(f"P1 context gain : {ev['context_gain_mean']:+.5f} nats/token")
    if "necessity_mean" in ev:
        print(f"P3 necessity    : {ev['necessity_mean']:+.5f} nats/token")
    if "block_fraction" in ev:
        print(f"block fraction  : {ev['block_fraction']:.4f} of residual RMS")
    if val_nll is not None:
        print(f"val NLL         : {val_nll:.5f}")
    print(f"checkpoint      : {record['final_checkpoint']}")
    print("manifest fields :")
    for name in (
        "run_id",
        "git_commit",
        "config_sha256",
        "corpus_sha256",
        "spawn_keys",
        "token_budget",
        "final_checkpoint",
        "wall_clock_s",
        "peak_vram_gb",
        "nan_skips",
        "arms",
        "notes",
    ):
        print(f"  {name}: {record[name]}")
    print("-" * 60)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        manifest_path = out.with_suffix(".manifest.json")
        RunManifest.from_dict(record).to_json(manifest_path)
        print(f"wrote {out}")
        print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
