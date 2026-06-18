"""Command-line entrypoint tying the pipeline together.

Subcommands:
  prepare     encode the corpus, cache vectors, fit the decoder bank
  train       train the concept predictor (on a frozen codec)
  train-joint train your own encoder/decoder + predictor jointly (kind='jepa')
  eval        report reconstruction + next-vector accuracy
  generate    roll out a continuation from a text prompt
  improve     run the reward-ranked self-improvement loop

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
from .persistence import (
    load_decoder_bank,
    load_predictor,
    save_decoder_bank,
    save_joint,
    save_predictor,
    write_json,
)


def _paths(cfg: Config) -> dict[str, Path]:
    a = cfg.artifacts_dir
    return {
        "predictor": a / "predictor.pt",
        "bank": a / "decoder_bank.npz",
        "metrics": a / "metrics.json",
    }


def _is_jepa(cfg: Config) -> bool:
    return cfg.codec.kind.lower() == "jepa"


def _ensure_jepa_checkpoint(cfg: Config) -> None:
    if not cfg.codec.jepa_checkpoint:
        cfg.codec.jepa_checkpoint = str(cfg.artifacts_dir / "jepa" / "joint.pt")


def _build_codec_with_bank(cfg: Config, fit_texts: list[str] | None):
    if _is_jepa(cfg):
        _ensure_jepa_checkpoint(cfg)
        return build_codec(cfg.codec)
    codec = build_codec(cfg.codec)
    if hasattr(codec, "fit_decoder"):
        if fit_texts is not None:
            bank = fit_texts[: cfg.codec.decoder_bank_size]
            codec.fit_decoder(bank)
        else:
            paths = _paths(cfg)
            if paths["bank"].exists():
                texts, vecs = load_decoder_bank(paths["bank"])
                codec.decoder.fit(texts, vecs)
    return codec


def _codec_and_predictor(cfg: Config, fit_texts: list[str] | None):
    """Return (codec, predictor). For 'jepa' the predictor is the jointly
    trained one carried by the codec; otherwise it is loaded from predictor.pt."""
    codec = _build_codec_with_bank(cfg, fit_texts)
    if _is_jepa(cfg):
        return codec, codec.predictor
    return codec, load_predictor(_paths(cfg)["predictor"])


def cmd_prepare(cfg: Config) -> None:
    cfg.ensure_dirs()
    docs = load_documents(
        cfg.data.corpus_path, cfg.data.min_sentence_chars, cfg.data.max_sentence_chars
    )
    codec = build_codec(cfg.codec)
    cache = EmbeddingCache(cfg.artifacts_dir / "cache")
    codec_id = f"{cfg.codec.kind}:{cfg.codec.model_name}"
    corpus = cache.build(docs, codec, codec_id)
    if hasattr(codec, "fit_decoder"):
        bank = corpus.sentences[: cfg.codec.decoder_bank_size]
        bank_vecs = corpus.vectors[: cfg.codec.decoder_bank_size]
        save_decoder_bank(_paths(cfg)["bank"], bank, bank_vecs)
    print(
        f"prepared {len(docs)} docs, {len(corpus.sentences)} sentences, dim={corpus.dim}"
    )


def cmd_train(cfg: Config) -> None:
    cfg.ensure_dirs()
    docs = load_documents(
        cfg.data.corpus_path, cfg.data.min_sentence_chars, cfg.data.max_sentence_chars
    )
    codec = build_codec(cfg.codec)
    cache = EmbeddingCache(cfg.artifacts_dir / "cache")
    codec_id = f"{cfg.codec.kind}:{cfg.codec.model_name}"
    corpus = cache.build(docs, codec, codec_id)
    dataset = ConceptSequenceDataset(corpus, context_len=cfg.data.context_len)

    from .predictor.train import train_predictor

    model, history = train_predictor(dataset, cfg.predictor, cfg.train)
    save_predictor(_paths(cfg)["predictor"], model)
    print(
        f"trained predictor ({model.num_parameters():,} params); "
        f"final val cosine={history.val_cosine[-1]:.3f}"
    )


def cmd_train_joint(cfg: Config) -> None:
    cfg.ensure_dirs()
    docs = load_documents(
        cfg.data.corpus_path, cfg.data.min_sentence_chars, cfg.data.max_sentence_chars
    )
    from .jepa.train_joint import train_joint

    model, _, history = train_joint(docs, cfg.jepa, cfg.predictor)
    ckpt = cfg.artifacts_dir / "jepa" / "joint.pt"
    save_joint(ckpt, model, cfg.jepa.tokenizer)
    metrics = {
        "final": history.latest(),
        "std_z_per_epoch": [round(s, 4) for s in history.std_z],
        "recon_acc_per_epoch": [round(a, 4) for a in history.recon_acc],
        "pred_per_epoch": [round(p, 4) for p in history.pred],
        "alpha_per_epoch": history.alpha,
    }
    write_json(cfg.artifacts_dir / "jepa" / "metrics.json", metrics)
    print(
        f"trained joint JEPA codec; final recon_acc={history.recon_acc[-1]:.3f} "
        f"std_z={history.std_z[-1]:.3f}"
    )
    if history.collapsed:
        print(
            "WARNING: representation collapse detected (std_z below threshold). "
            "Increase var_weight, lower alpha_max, or lengthen recon_warmup_epochs."
        )


def cmd_eval(cfg: Config) -> None:
    from .eval.metrics import next_vector_accuracy, reconstruction_quality

    docs = load_documents(cfg.data.corpus_path)
    codec, model = _codec_and_predictor(cfg, [s for d in docs for s in d.sentences])
    cache = EmbeddingCache(cfg.artifacts_dir / "cache")
    corpus = cache.build(docs, codec, f"{cfg.codec.kind}:{cfg.codec.model_name}")
    dataset = ConceptSequenceDataset(corpus, context_len=cfg.data.context_len)

    sample = corpus.sentences[: min(200, len(corpus.sentences))]
    recon = reconstruction_quality(codec, sample)
    acc = next_vector_accuracy(model, dataset, bank_vectors=corpus.vectors)

    metrics = {
        "reconstruction_cosine": recon.mean_cosine,
        "next_vector_cosine": acc.mean_cosine,
        "next_vector_retrieval_top1": acc.retrieval_top1,
        "eval_items": acc.n,
    }
    write_json(_paths(cfg)["metrics"], metrics)
    print(metrics)


def cmd_generate(cfg: Config, prompt: str, steps: int) -> None:
    docs = load_documents(cfg.data.corpus_path)
    codec, model = _codec_and_predictor(cfg, [s for d in docs for s in d.sentences])

    from .data.segment import segment_sentences

    seed_sents = segment_sentences(prompt) or [prompt]
    seed_vecs = codec.encode(seed_sents)
    roll = generate_rollout(model, seed_vecs, steps=steps, temperature=0.1)
    text = rollout_to_text(roll, codec)
    print("PROMPT:", prompt)
    print("CONTINUATION:")
    for line in text:
        print("  -", line)


def cmd_improve(cfg: Config) -> None:
    from .loop.self_improve import run_self_improvement
    from .reward.judge import build_judge

    docs = load_documents(cfg.data.corpus_path)
    codec, model = _codec_and_predictor(cfg, [s for d in docs for s in d.sentences])

    prompts = [codec.encode(d.sentences[:2]) for d in docs[:8] if len(d.sentences) >= 2]
    judge = build_judge()
    model, report = run_self_improvement(
        model, codec, prompts, judge, cfg.loop, cfg.predictor, cfg.train
    )
    if not _is_jepa(cfg):
        save_predictor(_paths(cfg)["predictor"], model)
    print("mean reward per iteration:", [round(r, 3) for r in report.mean_reward_per_iter])
    print("improved:", report.improved)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="path to JSON config override")

    p = argparse.ArgumentParser(prog="conceptpoc", parents=[common])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", parents=[common])
    sub.add_parser("train", parents=[common])
    sub.add_parser("train-joint", parents=[common])
    sub.add_parser("eval", parents=[common])
    g = sub.add_parser("generate", parents=[common])
    g.add_argument("--prompt", required=True)
    g.add_argument("--steps", type=int, default=6)
    sub.add_parser("improve", parents=[common])
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    if args.command == "prepare":
        cmd_prepare(cfg)
    elif args.command == "train":
        cmd_train(cfg)
    elif args.command == "train-joint":
        cmd_train_joint(cfg)
    elif args.command == "eval":
        cmd_eval(cfg)
    elif args.command == "generate":
        cmd_generate(cfg, args.prompt, args.steps)
    elif args.command == "improve":
        cmd_improve(cfg)


if __name__ == "__main__":
    main()
