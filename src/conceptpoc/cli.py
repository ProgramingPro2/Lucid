"""Command-line entrypoint tying the pipeline together.

Subcommands:
  prepare     encode the corpus, cache vectors, fit the decoder bank
  train       train the concept predictor (on a frozen codec)
  train-joint train your own encoder/decoder + predictor jointly (kind='jepa')
  eval        report reconstruction + next-vector accuracy
  generate    roll out a continuation (--prompt) or instruction response
  improve     run the reward-ranked self-improvement loop
  build-instructions  build + cache conditioned instruction concept pairs (E10)
  train-conditioned   train the segment/theme-conditioned predictor (E10)
  train-diffusion   train a concept-space diffusion predictor (replaces regression)
  train-decoder-robust  decoder fine-tune on predicted/perturbed vectors (contingency)

Run with: python -m conceptpoc.cli <subcommand> [--config config.json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .codec import build_codec
from .config import Config
from .data.cache import EmbeddingCache
from .data.corpus import load_documents
from .data.dataset import ConceptSequenceDataset
from .generate.rollout import generate_rollout, rollout_to_text
from .loader import (
    artifact_paths as _paths,
    build_codec_with_bank as _build_codec_with_bank,
    codec_and_predictor as _codec_and_predictor,
    ensure_jepa_checkpoint as _ensure_jepa_checkpoint,
    is_jepa as _is_jepa,
)
from .persistence import (
    save_decoder_bank,
    save_diffusion_predictor,
    save_joint,
    save_predictor,
    write_json,
)
from .progress import log_stage


def cmd_prepare(cfg: Config, *, progress: bool = True) -> None:
    cfg.ensure_dirs()
    docs = load_documents(
        cfg.data.corpus_path, cfg.data.min_sentence_chars, cfg.data.max_sentence_chars
    )
    if _is_jepa(cfg):
        _ensure_jepa_checkpoint(cfg)
    codec = build_codec(cfg.codec)
    cache = EmbeddingCache(cfg.artifacts_dir / "cache")
    codec_id = f"{cfg.codec.kind}:{cfg.codec.model_name}"
    corpus = cache.build(docs, codec, codec_id, progress=progress)
    if hasattr(codec, "fit_decoder"):
        bank = corpus.sentences[: cfg.codec.decoder_bank_size]
        bank_vecs = corpus.vectors[: cfg.codec.decoder_bank_size]
        save_decoder_bank(_paths(cfg)["bank"], bank, bank_vecs)
    print(
        f"prepared {len(docs)} docs, {len(corpus.sentences)} sentences, dim={corpus.dim}"
    )


def cmd_train(cfg: Config, *, progress: bool = True) -> None:
    cfg.ensure_dirs()
    docs = load_documents(
        cfg.data.corpus_path, cfg.data.min_sentence_chars, cfg.data.max_sentence_chars
    )
    codec = build_codec(cfg.codec)
    cache = EmbeddingCache(cfg.artifacts_dir / "cache")
    codec_id = f"{cfg.codec.kind}:{cfg.codec.model_name}"
    corpus = cache.build(docs, codec, codec_id, progress=progress)
    dataset = ConceptSequenceDataset(corpus, context_len=cfg.data.context_len)

    from .predictor.train import train_predictor

    model, history = train_predictor(
        dataset, cfg.predictor, cfg.train, progress=progress, gate=_build_gate(cfg, progress=progress)
    )
    save_predictor(_paths(cfg)["predictor"], model)
    best = (
        f"best val cosine={history.best_val_cosine:.3f} (epoch {history.best_epoch})"
        if history.best_epoch > 0
        else f"final val cosine={history.val_cosine[-1]:.3f}"
    )
    print(f"trained predictor ({model.num_parameters():,} params); {best}")


def cmd_train_delta(cfg: Config, *, progress: bool = True) -> None:
    """Train the delta (change-direction) forward predictor."""
    cfg.ensure_dirs()
    docs = load_documents(
        cfg.data.corpus_path, cfg.data.min_sentence_chars, cfg.data.max_sentence_chars
    )
    codec = build_codec(cfg.codec)
    cache = EmbeddingCache(cfg.artifacts_dir / "cache")
    codec_id = f"{cfg.codec.kind}:{cfg.codec.model_name}"
    corpus = cache.build(docs, codec, codec_id, progress=progress)

    from .predictor.train_delta import train_delta_predictor

    model, history = train_delta_predictor(
        corpus,
        cfg.predictor,
        cfg.train,
        progress=progress,
        gate=_build_gate(cfg, progress=progress),
        db_path=cfg.artifacts_dir / "train.db",
        context_len=cfg.data.context_len,
    )
    from .persistence import save_predictor

    out = cfg.artifacts_dir / "delta_predictor.pt"
    save_predictor(out, model)
    best = history.get("best_val_cosine", float("nan"))
    print(
        f"trained delta predictor ({model.num_parameters():,} params); "
        f"best_val_cosine={best:.4f}"
    )
    print(f"saved to {out}")


def cmd_train_diffusion(cfg: Config, *, progress: bool = True) -> None:
    cfg.ensure_dirs()
    cfg.predictor.head = "diffusion"
    docs = load_documents(
        cfg.data.corpus_path, cfg.data.min_sentence_chars, cfg.data.max_sentence_chars
    )
    codec = _build_codec_with_bank(cfg, [s for d in docs for s in d.sentences])
    cache = EmbeddingCache(cfg.artifacts_dir / "cache")
    codec_id = f"{cfg.codec.kind}:{cfg.codec.model_name}"
    corpus = cache.build(docs, codec, codec_id, progress=progress)
    dataset = ConceptSequenceDataset(corpus, context_len=cfg.data.context_len)

    from .predictor.train_diffusion import train_diffusion_predictor

    model, history = train_diffusion_predictor(
        dataset,
        cfg.predictor,
        cfg.train,
        progress=progress,
        gate=_build_gate(cfg, progress=progress),
        db_path=cfg.artifacts_dir / "train.db",
    )
    out = _paths(cfg)["predictor_diffusion"]
    save_diffusion_predictor(out, model)
    best = (
        f"best val sample cosine={history.best_val_sample_cosine:.3f} "
        f"(epoch {history.best_epoch})"
        if history.best_epoch > 0
        else f"final val sample cosine={history.val_sample_cosine[-1]:.3f}"
    )
    print(f"trained diffusion predictor ({model.num_parameters():,} params); {best}")
    print(f"saved to {out}")


def cmd_train_decoder_robust(cfg: Config, *, progress: bool = True) -> None:
    """Decoder-only robustness fine-tune (contingency for off-manifold samples)."""
    cfg.ensure_dirs()
    docs = load_documents(
        cfg.data.corpus_path, cfg.data.min_sentence_chars, cfg.data.max_sentence_chars
    )
    _ensure_jepa_checkpoint(cfg)
    from .jepa.train_decoder_robust import train_decoder_robust
    from .persistence import load_diffusion_predictor, load_joint, save_joint

    joint_path = Path(cfg.codec.jepa_checkpoint or cfg.artifacts_dir / "jepa" / "joint.pt")
    model, tokenizer = load_joint(joint_path)
    diffusion = None
    diff_path = _paths(cfg)["predictor_diffusion"]
    if diff_path.exists():
        diffusion = load_diffusion_predictor(diff_path)
    model, history = train_decoder_robust(
        model,
        tokenizer,
        docs,
        cfg.jepa,
        cfg.decoder_robust,
        diffusion=diffusion,
        progress=progress,
        gate=_build_gate(cfg, progress=progress),
    )
    out = cfg.artifacts_dir / "jepa" / "joint_robust.pt"
    save_joint(out, model, cfg.jepa.tokenizer)
    print(
        f"decoder-robust fine-tune done; final recon_acc={history.recon_acc[-1]:.3f}; "
        f"saved to {out}"
    )


def _build_gate(cfg: Config, *, progress: bool):
    if not cfg.gpu_gate.enabled:
        return None
    import torch

    from .gpu_gate import GpuGate

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate = GpuGate(cfg.gpu_gate, dev, progress=progress)
    if not gate.active:
        log_stage(
            "gpu-gate requested but no CUDA device is available; running normally.",
            enabled=progress,
        )
        return None
    log_stage(
        f"gpu-gate active: pausing when other processes use "
        f">={cfg.gpu_gate.foreign_util_pct:.0f}% GPU compute.",
        enabled=progress,
    )
    return gate


def cmd_train_joint(cfg: Config, *, progress: bool = True) -> None:
    cfg.ensure_dirs()
    docs = load_documents(
        cfg.data.corpus_path, cfg.data.min_sentence_chars, cfg.data.max_sentence_chars
    )
    from .jepa.train_joint import train_joint

    model, _, history = train_joint(
        docs,
        cfg.jepa,
        cfg.predictor,
        progress=progress,
        gate=_build_gate(cfg, progress=progress),
        db_path=cfg.artifacts_dir / "train.db",
        corpus_path=str(cfg.data.corpus_path),
    )
    ckpt = cfg.artifacts_dir / "jepa" / "joint.pt"
    save_joint(ckpt, model, cfg.jepa.tokenizer)
    metrics = {
        "final": history.latest(),
        "std_z_per_epoch": [round(s, 4) for s in history.std_z],
        "aniso_per_epoch": [round(a, 4) for a in history.aniso],
        "recon_acc_per_epoch": [round(a, 4) for a in history.recon_acc],
        "pred_per_epoch": [round(p, 4) for p in history.pred],
        "val_pred_cos_per_epoch": [round(v, 4) for v in history.val_pred_cos],
        "best_epoch": history.best_epoch,
        "best_val_pred_cos": round(history.best_val_cos, 4) if history.best_epoch > 0 else None,
        "alpha_per_epoch": history.alpha,
    }
    write_json(cfg.artifacts_dir / "jepa" / "metrics.json", metrics)
    best = (
        f" best epoch={history.best_epoch} (val_pred_cos={history.best_val_cos:.3f})"
        if history.best_epoch > 0
        else ""
    )
    print(
        f"trained joint JEPA codec; final recon_acc={history.recon_acc[-1]:.3f} "
        f"std_z={history.std_z[-1]:.3f}{best}"
    )
    if history.collapsed:
        print(
            "WARNING: representation collapse detected (std_z below threshold). "
            "Increase var_weight, lower alpha_max, or lengthen recon_warmup_epochs."
        )


def cmd_eval(cfg: Config, *, progress: bool = True) -> None:
    from .eval.metrics import next_vector_accuracy, reconstruction_quality

    docs = load_documents(cfg.data.corpus_path)
    codec, model = _codec_and_predictor(cfg, [s for d in docs for s in d.sentences])
    cache = EmbeddingCache(cfg.artifacts_dir / "cache")
    corpus = cache.build(
        docs, codec, f"{cfg.codec.kind}:{cfg.codec.model_name}", progress=progress
    )
    dataset = ConceptSequenceDataset(corpus, context_len=cfg.data.context_len)

    sample = corpus.sentences[: min(200, len(corpus.sentences))]
    log_stage(
        f"eval: reconstruction ({len(sample)} items)",
        enabled=progress,
    )
    recon = reconstruction_quality(codec, sample, progress=progress)
    log_stage(
        f"eval: next-vector accuracy (up to 512 items)",
        enabled=progress,
    )
    acc = next_vector_accuracy(
        model, dataset, bank_vectors=corpus.vectors, progress=progress
    )

    metrics = {
        "reconstruction_cosine": recon.mean_cosine,
        "next_vector_cosine": acc.mean_cosine,
        "next_vector_centered_cosine": acc.centered_cosine,
        "anisotropy_floor": acc.anisotropy,
        "next_vector_retrieval_top1": acc.retrieval_top1,
        "next_vector_retrieval_top5": acc.retrieval_top5,
        "next_vector_mrr": acc.mrr,
        "eval_items": acc.n,
    }
    write_json(_paths(cfg)["metrics"], metrics)
    print(metrics)


def cmd_generate(
    cfg: Config,
    prompt: str | None,
    steps: int,
    instruction: str | None = None,
    theme: str | None = None,
) -> None:
    from .data.segment import segment_sentences

    docs = load_documents(cfg.data.corpus_path)
    fit_texts = [s for d in docs for s in d.sentences]

    if instruction is not None:
        codec = _build_codec_with_bank(cfg, fit_texts)
        model = _load_conditioned_predictor(cfg)
        from .generate.rollout import generate_conditioned

        instr_sents = segment_sentences(instruction) or [instruction]
        instr_vecs = codec.encode(instr_sents)
        theme_vec = codec.encode([theme])[0] if theme else None
        roll = generate_conditioned(
            model, instr_vecs, steps=steps, theme_vec=theme_vec, temperature=0.0
        )
        text = rollout_to_text(roll, codec)
        print("INSTRUCTION:", instruction)
        if theme:
            print("THEME:", theme)
        print("RESPONSE:")
        for line in text:
            print("  -", line)
        return

    codec, model = _codec_and_predictor(cfg, fit_texts)
    seed_sents = segment_sentences(prompt) or [prompt]
    seed_vecs = codec.encode(seed_sents)
    roll = generate_rollout(model, seed_vecs, steps=steps, temperature=0.1)
    text = rollout_to_text(roll, codec)
    print("PROMPT:", prompt)
    print("CONTINUATION:")
    for line in text:
        print("  -", line)


def _conditioned_predictor_path(cfg: Config) -> Path:
    return cfg.artifacts_dir / "predictor_conditioned.pt"


def cmd_generate_hybrid(cfg: Config, prompt: str, steps: int, writer_path: str | None = None) -> None:
    """Hybrid generation: JEPA reasoner predicts next concept, writer renders it.

    Flow:
      1. Encode the prompt sentences into context concept vectors (JEPA encoder).
      2. Diffusion predictor samples the NEXT concept vector from that context.
      3. The fluent writer (GPT-2-scale) renders that vector into natural text.
      Repeat for ``steps``.
    """
    import torch
    import numpy as np

    from .data.segment import segment_sentences
    from .persistence import load_joint, load_diffusion_predictor, load_writer
    from .codec.jepa_codec import JepaCodec
    from .jepa.tokenizer import Tokenizer
    from .generate.rollout import _predict_next

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Frozen JEPA reasoner: encoder (concepts) + diffusion predictor.
    jepa_ckpt = cfg.writer.jepa_checkpoint or cfg.codec.jepa_checkpoint or \
        str(cfg.artifacts_dir / "jepa" / "joint_robust.pt")
    model, tok = load_joint(jepa_ckpt)
    model.to(dev).eval()
    codec = JepaCodec(model, tok, cfg.jepa.max_tokens, device=str(dev))
    pred_path = cfg.artifacts_dir / "predictor_diffusion.pt"
    predictor = load_diffusion_predictor(pred_path, device=str(dev))

    # 2. Fluent writer (Phase-2 conditioned).
    writer_path = writer_path or (cfg.artifacts_dir / "writer_cond.pt")
    writer = load_writer(writer_path, device=str(dev))

    # 3. Encode prompt to seed context.
    seed_sents = segment_sentences(prompt) or [prompt]
    seed_vecs = [codec.encode(seed_sents)]
    import numpy as _np
    context = [_np.asarray(v, dtype=_np.float32).reshape(1, -1)[0] for v in seed_vecs]

    print("PROMPT:", prompt)
    print("HYBRID CONTINUATION:")
    for i in range(steps):
        # Build context window for the predictor (last max_seq_len vectors).
        window_v = _np.stack(context[-cfg.predictor.max_seq_len:])
        ctx = torch.from_numpy(window_v).unsqueeze(0).to(dev)
        pred = _predict_next(predictor, ctx).squeeze(0).cpu().numpy().reshape(1, -1)
        # Render the predicted next-vector with the fluent writer.
        concept = torch.from_numpy(pred[0]).unsqueeze(0).to(dev)
        token_ids = writer.generate(
            concept, max_tokens=cfg.jepa.max_tokens,
            bos_id=tok.bos_id, eos_id=tok.eos_id, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.3,
        )[0].tolist()
        text = tok.decode(token_ids)
        print("  -", text)
        context.append(pred[0])


def cmd_generate_delta(cfg: Config, prompt: str, steps: int,
                       writer_path: str | None = None,
                       delta_path: str | None = None,
                       step_size: float = 1.0) -> None:
    """Delta generation: iterate the *change direction*, not the destination.

    Flow:
      1. Encode prompt into context concept vectors (JEPA encoder).
      2. Compute deltas (change directions) from recent context vectors.
      3. DeltaPredictor predicts the NEXT change direction.
      4. next_vec = current_vec + step*d_pred, renormalized; writer renders it.
      5. Append encoded result; repeat.
    The delta predictor was empirically shown to learn the forward change signal
    (ridge ~0.44) where exact-next was anti-predictive (~0.05 wall).
    """
    import torch
    import numpy as np

    from .data.segment import segment_sentences
    from .persistence import load_joint, load_predictor, load_writer
    from .codec.jepa_codec import JepaCodec
    from .jepa.tokenizer import Tokenizer
    from .predictor.delta_model import apply_delta

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jepa_ckpt = cfg.writer.jepa_checkpoint or cfg.codec.jepa_checkpoint or \
        str(cfg.artifacts_dir / "jepa" / "joint_robust.pt")
    model, tok = load_joint(jepa_ckpt)
    model.to(dev).eval()
    codec = JepaCodec(model, tok, cfg.jepa.max_tokens, device=str(dev))

    delta_path = delta_path or str(cfg.artifacts_dir / "delta_predictor.pt")
    delta_pred = load_predictor(delta_path, device=str(dev))
    delta_pred.to(dev).eval()
    writer_path = writer_path or (cfg.artifacts_dir / "writer_cond.pt")
    writer = load_writer(writer_path, device=str(dev))

    seed_sents = segment_sentences(prompt) or [prompt]
    context = [np.asarray(codec.encode(seed_sents), dtype=np.float32).reshape(1, -1)[0]
               for _ in ()] or list(np.asarray(codec.encode(seed_sents), dtype=np.float32))

    print("PROMPT:", prompt)
    print("DELTA CONTINUATION:")
    for i in range(steps):
        while len(context) < 3:
            # need at least 2 prior vectors to form a delta
            context.append(context[-1])
        # window of deltas from recent context (need >=2 prior vectors)
        last = context[-3:]
        # build delta sequence from the context (d_j = z[j+1]-z[j])
        zs = np.stack(last)
        deltas = []
        for j in range(len(zs) - 1):
            d = zs[j+1] - zs[j]
            n = np.linalg.norm(d)
            deltas.append(d / n if n > 0 else d)
        deltas = deltas[-cfg.predictor.max_seq_len:]
        ctx = torch.from_numpy(np.stack(deltas)).unsqueeze(0).to(dev)  # (1, L, dim)
        d_pred = delta_pred(ctx).squeeze(0).detach().cpu().numpy()  # (dim,)
        # apply change to the current vector
        cur = zs[-1]
        nxt = apply_delta(torch.from_numpy(cur).unsqueeze(0),
                          torch.from_numpy(d_pred).unsqueeze(0),
                          step=step_size).squeeze(0).cpu().numpy()
        concept = torch.from_numpy(nxt).unsqueeze(0).to(dev)
        token_ids = writer.generate(
            concept, max_tokens=cfg.jepa.max_tokens,
            bos_id=tok.bos_id, eos_id=tok.eos_id, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.3,
        )[0].tolist()
        text = tok.decode(token_ids)
        print("  -", text)
        context.append(nxt)


def cmd_generate_trajectory(cfg: Config, prompt: str, rounds: int,
                            writer_path: str | None = None,
                            planner_path: str | None = None) -> None:
    """Trajectory (plan-then-write) generation.

    Flow:
      1. Encode the prompt into context concept vectors (JEPA encoder).
      2. Planner emits a K-vector *trajectory* in latent space (the plan).
      3. Writer realizes each planned vector into a sentence; encoded sentence
         feeds back into context for the next round.
    The key difference vs one-step hybrid: the planner commits to a multi-step
    direction *before* writing, so the whole passage should track one topic.
    """
    import torch
    import numpy as np

    from .data.segment import segment_sentences
    from .persistence import load_joint, load_writer, load_planner
    from .codec.jepa_codec import JepaCodec
    from .jepa.tokenizer import Tokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jepa_ckpt = cfg.writer.jepa_checkpoint or cfg.codec.jepa_checkpoint or \
        str(cfg.artifacts_dir / "jepa" / "joint_robust.pt")
    model, tok = load_joint(jepa_ckpt)
    model.to(dev).eval()
    codec = JepaCodec(model, tok, cfg.jepa.max_tokens, device=str(dev))

    writer_path = writer_path or (cfg.artifacts_dir / "writer_cond.pt")
    writer = load_writer(writer_path, device=str(dev))
    planner_path = planner_path or (cfg.artifacts_dir / "planner.pt")
    planner = load_planner(planner_path, device=str(dev))
    C = planner.C
    K = planner.K

    # Build seed context from prompt sentences (pad up to C by repeating).
    seed_sents = segment_sentences(prompt) or [prompt]
    seed_vecs = [np.asarray(codec.encode(seed_sents), dtype=np.float32).reshape(-1)]
    while len(seed_vecs) < C:
        seed_vecs.append(seed_vecs[-1].copy())
    context = seed_vecs[-C:]

    print("PROMPT:", prompt)
    print("TRAJECTORY CONTINUATION (planner plans ahead K per round):")
    for _ in range(rounds):
        # Planner sees the last C concept vectors and emits K future concepts.
        window = torch.from_numpy(np.stack(context[-C:])).unsqueeze(0).to(dev)
        traj = planner.plan(window).squeeze(0).cpu().numpy()  # (K, dim)
        for v in traj:
            concept = torch.from_numpy(v).unsqueeze(0).to(dev)
            token_ids = writer.generate(
                concept, max_tokens=cfg.jepa.max_tokens,
                bos_id=tok.bos_id, eos_id=tok.eos_id, temperature=0.7, top_k=50,
                top_p=0.9, repetition_penalty=1.3,
            )[0].tolist()
            text = tok.decode(token_ids)
            print("  -", text)
            # Feed the realized sentence back into context.
            nv = np.asarray(codec.encode([text]), dtype=np.float32).reshape(-1)
            context.append(nv)
            if len(context) > C:
                context.pop(0)


def _load_conditioned_predictor(cfg: Config):
    from .persistence import load_predictor

    path = _conditioned_predictor_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"no conditioned predictor at {path}; run 'train-conditioned' first"
        )
    return load_predictor(path)


def _build_synthetic(cfg: Config, codec, *, progress: bool):
    from .data.instruction import build_synthetic_instructions

    icfg = cfg.instruction
    docs = load_documents(
        cfg.data.corpus_path,
        cfg.data.min_sentence_chars,
        cfg.data.max_sentence_chars,
    )
    # Bound encoding cost: at most max_pairs corpus passages become synth pairs.
    docs = docs[: max(1, icfg.max_pairs)]
    return build_synthetic_instructions(
        docs,
        codec,
        templates=icfg.templates,
        themes=icfg.themes,
        max_response_sentences=icfg.max_response_sentences,
        max_instruction_sentences=icfg.max_instruction_sentences,
        author_with_llm=icfg.author_with_llm,
    )


def _build_borrowed(cfg: Config, codec, *, progress: bool):
    """Borrowed HF subset; returns ``None`` (with a warning) if HF is unavailable."""
    from .data.instruction import build_borrowed_instructions

    icfg = cfg.instruction
    try:
        return build_borrowed_instructions(
            codec,
            icfg.hf_dataset,
            split=icfg.hf_split,
            max_pairs=icfg.max_pairs,
            instruction_field=icfg.instruction_field,
            response_field=icfg.response_field,
            max_response_sentences=icfg.max_response_sentences,
            max_instruction_sentences=icfg.max_instruction_sentences,
        )
    except Exception as exc:  # noqa: BLE001 - HF/network failures are expected offline
        log_stage(
            f"borrow unavailable ({type(exc).__name__}: {exc}); "
            "falling back to synthesized pairs only.",
            enabled=progress,
        )
        return None


def cmd_build_instructions(cfg: Config, *, progress: bool = True) -> None:
    cfg.ensure_dirs()
    icfg = cfg.instruction
    source = icfg.source.lower()
    if source not in {"borrow", "synthesize", "both"}:
        raise ValueError(f"unknown instruction source: {icfg.source!r}")

    from .data.instruction import combine_instruction_datasets

    codec = _build_codec_with_bank(cfg, _corpus_sentences(cfg))

    parts: list = []
    sizes: dict[str, int] = {}
    if source in {"borrow", "both"}:
        borrowed = _build_borrowed(cfg, codec, progress=progress)
        if borrowed is not None:
            parts.append(borrowed)
            sizes["borrow"] = len(borrowed)
    if source in {"synthesize", "both"} or not parts:
        synth = _build_synthetic(cfg, codec, progress=progress)
        parts.append(synth)
        sizes["synthesize"] = len(synth)

    dataset = combine_instruction_datasets(parts) if len(parts) > 1 else parts[0]

    out = Path(icfg.cache_path)
    if not out.is_absolute():
        out = cfg.artifacts_dir / out
    dataset.save(out)
    breakdown = ", ".join(f"{k}={v}" for k, v in sizes.items())
    print(
        f"built {len(dataset)} instruction pairs (requested={icfg.source}; "
        f"{breakdown}; dim={dataset.dim}, themed={dataset.has_theme}) -> {out}"
    )


def _corpus_sentences(cfg: Config) -> list[str]:
    docs = load_documents(
        cfg.data.corpus_path, cfg.data.min_sentence_chars, cfg.data.max_sentence_chars
    )
    return [s for d in docs for s in d.sentences]


def cmd_train_conditioned(cfg: Config, *, progress: bool = True) -> None:
    cfg.ensure_dirs()
    icfg = cfg.instruction
    cache = Path(icfg.cache_path)
    if not cache.is_absolute():
        cache = cfg.artifacts_dir / cache
    if not cache.exists():
        raise FileNotFoundError(
            f"no instruction cache at {cache}; run 'build-instructions' first"
        )

    from .data.instruction import InstructionConceptDataset
    from .predictor.train_conditioned import train_conditioned

    dataset = InstructionConceptDataset.load(cache)
    model, history = train_conditioned(
        dataset,
        cfg.predictor,
        cfg.train,
        progress=progress,
        gate=_build_gate(cfg, progress=progress),
    )
    save_predictor(_conditioned_predictor_path(cfg), model)
    best = (
        f"best val cosine={history.best_val_cosine:.3f} (epoch {history.best_epoch})"
        if history.best_epoch > 0
        else "no validation split"
    )
    print(
        f"trained conditioned predictor ({model.num_parameters():,} params); {best}"
    )


def _spread_sample(seq: list, n: int) -> list:
    """Return up to ``n`` items evenly spread across ``seq`` (order preserved).

    Used to pick RAFT prompts from the whole cached set instead of just the
    first few, so alignment does not overfit one contiguous slice.
    """
    total = len(seq)
    if n <= 0 or total <= n:
        return list(seq)
    if n == 1:
        return [seq[0]]
    step = (total - 1) / (n - 1)
    idx = sorted({round(i * step) for i in range(n)})
    return [seq[i] for i in idx]


def cmd_improve(cfg: Config, *, conditioned: bool = False, progress: bool = True) -> None:
    from .reward.judge import build_judge

    judge = build_judge()
    if conditioned:
        _cmd_improve_conditioned(cfg, judge, progress=progress)
        return

    from .loop.self_improve import run_self_improvement

    docs = load_documents(cfg.data.corpus_path)
    codec, model = _codec_and_predictor(cfg, [s for d in docs for s in d.sentences])

    usable_docs = [d for d in docs if len(d.sentences) >= 2]
    chosen = _spread_sample(usable_docs, cfg.loop.n_prompts)
    prompts = [codec.encode(d.sentences[:2]) for d in chosen]
    model, report = run_self_improvement(
        model,
        codec,
        prompts,
        judge,
        cfg.loop,
        cfg.predictor,
        cfg.train,
        progress=progress,
    )
    if not _is_jepa(cfg):
        save_predictor(_paths(cfg)["predictor"], model)
    print("mean reward per iteration:", [round(r, 3) for r in report.mean_reward_per_iter])
    print("improved:", report.improved)


def _cmd_improve_conditioned(cfg: Config, judge, *, progress: bool = True) -> None:
    from .data.instruction import InstructionConceptDataset
    from .loader import codec_and_conditioned_predictor
    from .loop.self_improve import (
        ConditionedPrompt,
        run_conditioned_self_improvement,
    )

    icfg = cfg.instruction
    cache = Path(icfg.cache_path)
    if not cache.is_absolute():
        cache = cfg.artifacts_dir / cache
    if not cache.exists():
        raise FileNotFoundError(
            f"no instruction cache at {cache}; run 'build-instructions' first"
        )

    codec, model = codec_and_conditioned_predictor(cfg, None)
    dataset = InstructionConceptDataset.load(cache)

    # Snapshot the SFT model (current canonical) before RAFT mutates it, so the
    # held-out keep-best gate can compare SFT against RAFT honestly.
    sft_path = cfg.artifacts_dir / "predictor_conditioned_sft.pt"
    save_predictor(sft_path, model)

    prompts = []
    for pair in _spread_sample(dataset.pairs, cfg.loop.n_prompts):
        text = " ".join(codec.decode(pair.instruction_vecs))
        prompts.append(
            ConditionedPrompt(
                instruction_vecs=pair.instruction_vecs,
                theme_vec=pair.theme_vec,
                text=text,
            )
        )

    model, report = run_conditioned_self_improvement(
        model,
        codec,
        prompts,
        judge,
        cfg.loop,
        cfg.predictor,
        cfg.train,
        progress=progress,
    )
    raft_path = cfg.artifacts_dir / "predictor_conditioned_raft.pt"
    save_predictor(raft_path, model)
    print("mean reward per iteration:", [round(r, 3) for r in report.mean_reward_per_iter])
    print("improved:", report.improved)

    winner, scores = _keep_best_conditioned(
        cfg, judge, sft_path, raft_path, progress=progress
    )
    print(
        f"held-out keep-best (judge={_judge_backend()}): "
        f"sft={scores['sft']:.4f} raft={scores['raft']:.4f} -> canonical={winner}"
    )


def _judge_backend() -> str:
    import os

    return os.environ.get("JUDGE_BACKEND", "heuristic").lower()


def _pick_winner(scores: dict[str, float]) -> str:
    """Keep RAFT only if it strictly beats SFT on held-out; SFT wins ties."""
    return "raft" if scores.get("raft", float("-inf")) > scores.get("sft", float("-inf")) else "sft"


def _keep_best_conditioned(
    cfg: Config, judge, sft_path: Path, raft_path: Path, *, progress: bool = True
) -> tuple[str, dict[str, float]]:
    """Score SFT vs RAFT on the held-out eval set and keep the winner canonical.

    Each candidate is copied to the canonical conditioned checkpoint, scored with
    ``run_harness(..., conditioned=True)`` under the configured judge, and the
    held-out winner (not training reward) becomes canonical. Both variants stay
    on disk. The canonical checkpoint is restored to SFT if scoring fails.
    """
    import shutil

    from .eval.harness import run_harness

    canonical = _conditioned_predictor_path(cfg)
    backend = _judge_backend()
    steps = cfg.eval.rollout_steps
    eval_path = cfg.eval.eval_path
    scores: dict[str, float] = {}
    try:
        for label, ckpt in (("sft", sft_path), ("raft", raft_path)):
            shutil.copyfile(ckpt, canonical)
            _, report = run_harness(
                cfg,
                eval_path=eval_path,
                steps=steps,
                judge=judge,
                judge_backend=backend,
                config_label=f"conditioned-{label}",
                progress=progress,
                conditioned=True,
            )
            scores[label] = report["overall"]["mean_score"]
        winner = _pick_winner(scores)
        shutil.copyfile(sft_path if winner == "sft" else raft_path, canonical)
        return winner, scores
    except Exception:
        shutil.copyfile(sft_path, canonical)
        raise


def _resolve_out_dir(cfg: Config, override: str | None) -> Path:
    target = override if override is not None else cfg.eval.output_dir
    p = Path(target)
    return p if p.is_absolute() else cfg.artifacts_dir / p


def cmd_writer_pretrain(cfg: Config, *, progress: bool = True) -> None:
    """Phase-1: pretrain the fluent writer as a next-token LM."""
    from .writer.train_phase1 import train_phase1
    from .persistence import save_writer

    model, tok = train_phase1(cfg, progress=progress)
    out = cfg.artifacts_dir / "writer_p1.pt"
    save_writer(
        out,
        model,
        {
            "vocab_size": tok.vocab_size,
            "dim": cfg.writer.dim,
            "n_layers": cfg.writer.n_layers,
            "n_heads": cfg.writer.n_heads,
            "ff_mult": cfg.writer.ff_mult,
            "max_seq_len": cfg.writer.seq_len + 8,
            "concept_dim": cfg.jepa.dim,
            "use_concept": False,
            "use_film": False,
        },
    )
    print(f"writer phase-1 done; {model.num_parameters():,} params; saved to {out}")


def cmd_writer_condition(cfg: Config, *, progress: bool = True) -> None:
    """Phase-2: condition the writer on JEPA concept vectors."""
    from .writer.train_phase2 import train_phase2
    from .persistence import save_writer

    model, tok = train_phase2(cfg, progress=progress)
    # Phase-2.5 two-channel checkpoint; never overwrite writer_p2_film/writer_cond.pt (A/B control).
    out = cfg.artifacts_dir / "writer_two_channel.pt"
    save_writer(
        out,
        model,
        {
            "vocab_size": tok.vocab_size,
            "dim": cfg.writer.dim,
            "n_layers": cfg.writer.n_layers,
            "n_heads": cfg.writer.n_heads,
            "ff_mult": cfg.writer.ff_mult,
            "max_seq_len": cfg.writer.seq_len + 8,
            "concept_dim": cfg.jepa.dim,
            "use_concept": cfg.writer.use_concept,
            "use_film": cfg.writer.use_film,
            "two_channel": True,
        },
    )
    print(f"writer phase-2.5 done; {model.num_parameters():,} params; saved to {out}")


def cmd_eval_harness(cfg: Config, args, *, progress: bool = True) -> None:
    import os

    from .eval.harness import (
        format_summary,
        run_harness,
        write_report,
    )
    from .reward.judge import build_judge

    judge_backend = os.environ.get("JUDGE_BACKEND", "heuristic").lower()
    steps = args.steps if args.steps is not None else cfg.eval.rollout_steps
    eval_path = args.eval_set if args.eval_set is not None else cfg.eval.eval_path

    if args.compare:
        _run_compare(cfg, args, steps, eval_path, judge_backend, progress=progress)
        return

    judge = build_judge()
    results, report = run_harness(
        cfg,
        eval_path=eval_path,
        steps=steps,
        judge=judge,
        judge_backend=judge_backend,
        config_label=getattr(args, "config", None),
        progress=progress,
        conditioned=getattr(args, "conditioned", False),
    )

    rating_url = args.rating_app_url or cfg.eval.rating_app_url
    if args.push_rating_app:
        from .eval.rating_client import push_to_rating_app

        n = push_to_rating_app(results, rating_url)
        log_stage(f"pushed {n} items to rating app at {rating_url}", enabled=progress)
    if args.merge_human:
        from .eval.rating_client import merge_human_scores

        n = merge_human_scores(results, rating_url)
        report = _rebuild_report(results, report)
        log_stage(f"merged human scores for {n} items", enabled=progress)

    out_dir = _resolve_out_dir(cfg, args.out)
    json_path, _ = write_report(report, out_dir)
    print(format_summary(report))
    print(f"\nreport written to {json_path}")


def _rebuild_report(results, report: dict) -> dict:
    from .eval.harness import build_report

    return build_report(
        results,
        steps=report["steps"],
        judge_backend=report["judge_backend"],
        config=report["config"],
    )


def _run_compare(cfg, args, steps, eval_path, judge_backend, *, progress: bool) -> None:
    from .eval.harness import run_harness, write_report
    from .reward.judge import build_judge

    judge = build_judge()
    runs: list[tuple[str, dict]] = []
    for config_path in args.compare:
        sub_cfg = Config.load(config_path)
        _, report = run_harness(
            sub_cfg,
            eval_path=eval_path,
            steps=steps,
            judge=judge,
            judge_backend=judge_backend,
            config_label=config_path,
            progress=progress,
        )
        write_report(report, _resolve_out_dir(sub_cfg, args.out))
        runs.append((config_path, report))

    caps = sorted({c for _, r in runs for c in r["per_capability"]})
    header = f"{'config':<40}" + "".join(f"{c[:14]:>16}" for c in caps) + f"{'overall':>16}"
    lines = ["Compare (mean judge score per capability)", header, "-" * len(header)]
    for path, report in runs:
        cells = "".join(
            f"{report['per_capability'].get(c, {}).get('mean_score', float('nan')):>16.3f}"
            for c in caps
        )
        overall = report["overall"]["mean_score"]
        lines.append(f"{path:<40}{cells}{overall:>16.3f}")
    table = "\n".join(lines)
    print(table)

    out_dir = _resolve_out_dir(cfg, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "compare.json",
        {"capabilities": caps, "runs": {p: r for p, r in runs}},
    )
    (out_dir / "compare.txt").write_text(table + "\n", encoding="utf-8")
    print(f"\ncompare report written to {out_dir / 'compare.json'}")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    # default=SUPPRESS so a subparser parsing no --config does not clobber a value
    # already set by the top-level parser when --config precedes the subcommand.
    common.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="path to JSON config override",
    )
    common.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress progress bars and per-epoch logging",
    )
    common.add_argument(
        "--gpu-gate",
        action="store_true",
        help="pause training (offloading to CPU) while other processes use the GPU",
    )

    p = argparse.ArgumentParser(prog="conceptpoc", parents=[common])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", parents=[common])
    sub.add_parser("train", parents=[common])
    sub.add_parser("train-delta", parents=[common])
    sub.add_parser("train-diffusion", parents=[common])
    sub.add_parser("train-decoder-robust", parents=[common])
    sub.add_parser("train-joint", parents=[common])
    sub.add_parser("eval", parents=[common])
    g = sub.add_parser("generate", parents=[common])
    g.add_argument("--prompt", default=None, help="continuation seed text")
    g.add_argument(
        "--instruction", default=None, help="conditioned generation instruction"
    )
    g.add_argument(
        "--theme", default=None, help="optional theme vector for conditioned generation"
    )
    g.add_argument("--steps", type=int, default=6)
    gh = sub.add_parser("generate-hybrid", parents=[common])
    gh.add_argument("--prompt", required=True, help="continuation seed text")
    gh.add_argument("--steps", type=int, default=6)
    gh.add_argument("--writer", default=None, help="path to Phase-2 writer checkpoint")
    gt = sub.add_parser("generate-trajectory", parents=[common])
    gt.add_argument("--prompt", required=True, help="continuation seed text")
    gt.add_argument("--rounds", type=int, default=2)
    gt.add_argument("--writer", default=None, help="path to Phase-2 writer checkpoint")
    gt.add_argument("--planner", default=None, help="path to trajectory planner checkpoint")
    gd = sub.add_parser("generate-delta", parents=[common])
    gd.add_argument("--prompt", required=True, help="continuation seed text")
    gd.add_argument("--steps", type=int, default=6)
    gd.add_argument("--writer", default=None, help="path to Phase-2 writer checkpoint")
    gd.add_argument("--delta", default=None, help="path to delta predictor checkpoint")
    gd.add_argument("--step-size", type=float, default=1.0, help="delta step magnitude")
    imp = sub.add_parser("improve", parents=[common])
    imp.add_argument(
        "--conditioned",
        action="store_true",
        help="run RAFT over instruction-conditioned rollouts (E10 alignment)",
    )
    sub.add_parser("build-instructions", parents=[common])
    sub.add_parser("train-conditioned", parents=[common])
    sub.add_parser("writer-pretrain", parents=[common])
    sub.add_parser("writer-condition", parents=[common])

    eh = sub.add_parser("eval-harness", parents=[common])
    eh.add_argument(
        "--eval-set", default=None, help="path to a .jsonl file or dir (overrides config)"
    )
    eh.add_argument("--steps", type=int, default=None, help="rollout steps per prompt")
    eh.add_argument("--out", default=None, help="output dir for report.json")
    eh.add_argument(
        "--push-rating-app",
        action="store_true",
        help="POST each generated item to the rating app (/api/items)",
    )
    eh.add_argument(
        "--merge-human",
        action="store_true",
        help="pull /api/export and merge human scores by (prompt, text)",
    )
    eh.add_argument(
        "--rating-app-url", default=None, help="rating app base URL (overrides config)"
    )
    eh.add_argument(
        "--compare",
        nargs="+",
        default=None,
        metavar="CONFIG",
        help="config paths to evaluate side by side and tabulate",
    )
    eh.add_argument(
        "--conditioned",
        action="store_true",
        help="load the conditioned predictor and drive instruction/themed sets "
        "with generate_conditioned (E10)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.load(getattr(args, "config", None))
    progress = not getattr(args, "quiet", False)
    if getattr(args, "gpu_gate", False):
        cfg.gpu_gate.enabled = True
    if args.command == "prepare":
        cmd_prepare(cfg, progress=progress)
    elif args.command == "train":
        cmd_train(cfg, progress=progress)
    elif args.command == "train-delta":
        cmd_train_delta(cfg, progress=progress)
    elif args.command == "train-diffusion":
        cmd_train_diffusion(cfg, progress=progress)
    elif args.command == "train-decoder-robust":
        cmd_train_decoder_robust(cfg, progress=progress)
    elif args.command == "train-joint":
        cmd_train_joint(cfg, progress=progress)
    elif args.command == "eval":
        cmd_eval(cfg, progress=progress)
    elif args.command == "generate":
        if args.prompt is None and args.instruction is None:
            raise SystemExit("generate requires --prompt or --instruction")
        cmd_generate(cfg, args.prompt, args.steps, args.instruction, args.theme)
    elif args.command == "generate-hybrid":
        cmd_generate_hybrid(cfg, args.prompt, args.steps, writer_path=args.writer)
    elif args.command == "generate-trajectory":
        cmd_generate_trajectory(cfg, args.prompt, args.rounds,
                                writer_path=args.writer, planner_path=args.planner)
    elif args.command == "generate-delta":
        cmd_generate_delta(cfg, args.prompt, args.steps,
                           writer_path=args.writer, delta_path=args.delta,
                           step_size=args.step_size)
    elif args.command == "improve":
        cmd_improve(
            cfg, conditioned=getattr(args, "conditioned", False), progress=progress
        )
    elif args.command == "build-instructions":
        cmd_build_instructions(cfg, progress=progress)
    elif args.command == "train-conditioned":
        cmd_train_conditioned(cfg, progress=progress)
    elif args.command == "writer-pretrain":
        cmd_writer_pretrain(cfg, progress=progress)
    elif args.command == "writer-condition":
        cmd_writer_condition(cfg, progress=progress)
    elif args.command == "eval-harness":
        cmd_eval_harness(cfg, args, progress=progress)


if __name__ == "__main__":
    main()
