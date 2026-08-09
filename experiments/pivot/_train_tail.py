def _trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _grads_finite(params: list[nn.Parameter]) -> bool:
    for p in params:
        if p.grad is None:
            continue
        if not bool(torch.isfinite(p.grad).all().item()):
            return False
    return True


def train(
    cfg: dict[str, Any],
    *,
    resume: bool = True,
    max_steps: int | None = None,
    profile: bool = False,
    skip_baseline: bool = False,
    log_path: Path | None = None,
) -> dict[str, Any]:
    _seed_all(int(cfg.get("seed", 42)))
    out_dir = _ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_path = out_dir / "train.log"
    else:
        log_path = Path(log_path)
    device = _device("auto")

    zbank = _ROOT / cfg["zbank"]
    if not zbank.exists():
        from experiments.pivot.encode_bank import build_bank

        print("[stageA] zbank missing; building...", flush=True)
        build_bank(out_path=zbank)

    print(f"[stageA] loading train pairs from {zbank}...", flush=True)
    train_ds = ParaPairDataset(_ROOT / cfg["corpus"], zbank, split="train")
    print("[stageA] loading held pairs...", flush=True)
    held_ds = ParaPairDataset(_ROOT / cfg["corpus"], zbank, split="held")
    n_paras = len(train_ds)
    print(
        f"[stageA] train pairs={n_paras} held={len(held_ds)} "
        f"(need ≥{cfg.get('min_paragraphs', 150000)})",
        flush=True,
    )
    if n_paras < int(cfg.get("min_paragraphs", 150000)):
        print(
            f"[stageA] WARNING: train pairs {n_paras} < min_paragraphs; "
            "continuing anyway (anti-starvation still applies via ≥2 epochs).",
            flush=True,
        )

    backbone, tokenizer, backbone_name = _resolve_backbone(cfg, device)
    backbone = _attach_lora(backbone, cfg)
    adapter = SoftPrefixAdapter(
        z_dim=int(cfg["z_dim"]),
        d_model=int(cfg["d_model"]),
        n_prefix=int(cfg["n_prefix"]),
        hidden=int(cfg.get("adapter_hidden", 512)),
    ).to(device)
    print(
        f"[stageA] adapter params={human_param_count(adapter.n_trainable())}",
        flush=True,
    )
    model = StageAModel(backbone, adapter)
    model.train()

    if bool(cfg.get("pretokenize", True)):
        cache_name = (
            f"pretok_{backbone_name.replace('/', '_')}"
            f"_ctx{cfg['ctx_max']}_np{cfg['n_prefix']}.pt"
        )
        train_ds.build_pretoken_cache(tokenizer, cfg, cache_path=out_dir / cache_name)

    opt = torch.optim.AdamW(
        _param_groups(model, cfg),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    trainable = _trainable_params(model)

    db_path = _ROOT / cfg.get("db_path", str(out_dir / "checkpoints.sqlite"))
    db = CheckpointDB(db_path, keep_last=3)
    run_id = db.latest_run() if resume else None
    start_epoch = 0
    global_step = 0
    flat_holder = _FlatState()
    if run_id is None:
        run_id = db.start_run(
            str(cfg["corpus"]),
            backbone_name,
            int(cfg.get("seed", 42)),
            int(cfg["ctx_max"]),
            int(cfg["n_prefix"]),
            {**cfg, "backbone_resolved": backbone_name},
        )
        print(f"[stageA] new run_id={run_id}", flush=True)
    else:
        start_epoch, global_step = _load_resume(db, run_id, model, opt, device)

    gate = _maybe_gate(device, bool(cfg.get("gpu_gate", True)))
    nw = int(cfg.get("num_workers", 0))
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("micro_batch", 1)),
        shuffle=True,
        num_workers=nw,
        collate_fn=_make_collate(tokenizer, cfg),
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
    )
    grad_accum = int(cfg.get("grad_accum", 64))
    epochs = int(cfg.get("epochs", 2))
    kappa = float(cfg.get("kappa_aug", 700.0))
    drop_p = float(cfg.get("prefix_drop_p", 0.10))
    log_every = int(cfg.get("log_every", 20))
    eval_every = int(cfg.get("eval_every_steps", 500))
    clip = float(cfg.get("grad_clip", 1.0))
    profile_every = int(cfg.get("profile_every_micro", 1 if profile else 0))
    micro_log_every = int(cfg.get("micro_log_every", 16 if profile else 0))

    baseline_ce = float("nan")
    if not skip_baseline:
        baseline_losses: list[float] = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 32:
                    break
                z = batch["z"].to(device)
                drop = torch.ones(z.size(0), dtype=torch.bool, device=device)
                out = model(
                    z,
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device),
                    drop_mask=drop,
                )
                baseline_losses.append(float(out.loss.item()))
        baseline_ce = float(np.mean(baseline_losses)) if baseline_losses else float("nan")
        model.train()
        (out_dir / "baseline_uncond.json").write_text(
            json.dumps(
                {"n": len(baseline_losses), "ce": baseline_ce, "backbone": backbone_name},
                indent=2,
            )
            + "\n"
        )
    else:
        bpath = out_dir / "baseline_uncond.json"
        if bpath.exists():
            try:
                baseline_ce = float(json.loads(bpath.read_text())["ce"])
            except Exception:  # noqa: BLE001
                baseline_ce = float("nan")

    codec: JepaCodec | None = None
    try:
        jepa_path = _ROOT / cfg.get("jepa", "artifacts/jepa_repair_1m/jepa/joint.pt")
        jm, jt = load_joint(str(jepa_path), device="cpu")
        codec = JepaCodec(jm, jt, max_tokens=128, device="cpu")
    except Exception as e:  # noqa: BLE001
        print(f"[stageA] JEPA codec for eval skipped: {e}", flush=True)

    history: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    stop = {"flag": False}
    nan_skips = 0
    grad_nan_skips = 0

    def _sig(_s, _f):
        stop["flag"] = True
        print("[stageA] interrupt flagged; will checkpoint after step", flush=True)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    t_run = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        def log(msg: str) -> None:
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()

        log(
            f"start run={run_id} backbone={backbone_name} "
            f"train_pairs={n_paras} epochs={epochs} "
            f"eff_batch={grad_accum * int(cfg.get('micro_batch', 1))} "
            f"resume_epoch={start_epoch} step={global_step} vram={_vram_gb():.2f}GB "
            f"baseline_uncond_ce={baseline_ce:.4f} kappa={kappa} "
            f"profile={int(profile)} pretoken={int(train_ds._tok_cache is not None)}"
        )

        for epoch in range(start_epoch, epochs):
            model.train()
            running = 0.0
            n_loss = 0
            tok_window = 0
            t_window = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            micro = 0
            loader_iter = iter(loader)
            bi = -1
            while True:
                bi += 1
                if stop["flag"] or (max_steps is not None and global_step >= max_steps):
                    break

                t0 = time.perf_counter()
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                t_data = (time.perf_counter() - t0) * 1000.0

                if gate is not None:
                    gate.step(model, opt)

                t1 = time.perf_counter()
                z = batch["z"].to(device, non_blocking=True)
                z_tilde = fiber_corrupt(z, kappa=kappa)
                z_ok = bool(torch.isfinite(z_tilde).all().item())
                if not z_ok:
                    z_tilde = F.normalize(z.float(), dim=-1)
                drop = torch.rand(z.size(0), device=device) < drop_p
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                n_tgt_tok = int((labels != -100).sum().item())
                if n_tgt_tok == 0:
                    continue
                t_fiber = (time.perf_counter() - t1) * 1000.0

                t2 = time.perf_counter()
                out = model(z_tilde, input_ids, attn, labels, drop_mask=drop)
                raw_loss = out.loss
                if device.type == "cuda" and profile:
                    torch.cuda.synchronize()
                t_fwd = (time.perf_counter() - t2) * 1000.0

                if raw_loss is None or not bool(torch.isfinite(raw_loss).item()):
                    nan_skips += 1
                    opt.zero_grad(set_to_none=True)
                    micro = 0
                    if nan_skips <= 20 or nan_skips % 50 == 0:
                        log(
                            f"NAN_SKIP micro_bi={bi} step={global_step} "
                            f"nan_skips={nan_skips} z_finite={int(z_ok)} "
                            f"(raw_loss guard; accumulation reset)"
                        )
                    continue

                t3 = time.perf_counter()
                loss = raw_loss / grad_accum
                loss.backward()
                if device.type == "cuda" and profile:
                    torch.cuda.synchronize()
                t_bwd = (time.perf_counter() - t3) * 1000.0
                micro += 1

                loss_val = float(raw_loss.detach().item())
                running += loss_val
                n_loss += 1
                tok_window += n_tgt_tok

                t_opt = 0.0
                did_step = False
                if micro >= grad_accum:
                    t4 = time.perf_counter()
                    if not _grads_finite(trainable):
                        grad_nan_skips += 1
                        opt.zero_grad(set_to_none=True)
                        micro = 0
                        log(
                            f"GRAD_NAN_SKIP step_pending={global_step + 1} "
                            f"grad_nan_skips={grad_nan_skips} "
                            f"(optimizer step aborted; grads cleared)"
                        )
                        t_opt = (time.perf_counter() - t4) * 1000.0
                    else:
                        nn.utils.clip_grad_norm_(trainable, clip)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        micro = 0
                        global_step += 1
                        did_step = True
                        if device.type == "cuda" and profile:
                            torch.cuda.synchronize()
                        t_opt = (time.perf_counter() - t4) * 1000.0

                        wall = time.perf_counter() - t_window
                        tps = tok_window / max(wall, 1e-6)
                        avg = running / max(n_loss, 1)
                        if global_step % log_every == 0 or global_step <= 5 or profile:
                            log(
                                f"epoch={epoch} step={global_step} loss={avg:.4f} "
                                f"tok/s={tps:.1f} step_ms={wall * 1000:.0f} "
                                f"vram={_vram_gb():.2f}GB "
                                f"nan_skips={nan_skips} grad_nan_skips={grad_nan_skips}"
                                + (
                                    f" (baseline_uncond_ce={baseline_ce:.4f})"
                                    if global_step <= 5
                                    else ""
                                )
                            )
                            history.append(
                                {
                                    "epoch": epoch,
                                    "step": global_step,
                                    "loss": avg,
                                    "tok_s": tps,
                                    "step_ms": wall * 1000.0,
                                    "vram_gb": _vram_gb(),
                                }
                            )
                            running = 0.0
                            n_loss = 0
                            tok_window = 0
                            t_window = time.perf_counter()

                        if global_step % eval_every == 0:
                            eval_path = out_dir / f"eval_samples_step{global_step:06d}.jsonl"
                            try:
                                log(f"eval_sample_log begin n={cfg.get('eval_n', 64)}")
                                eval_sample_log(
                                    model,
                                    tokenizer,
                                    codec,
                                    held_ds,
                                    eval_path,
                                    n=int(cfg.get("eval_n", 64)),
                                    device=device,
                                )
                            except Exception as e:  # noqa: BLE001
                                log(f"eval_sample_log failed: {e}")

                if profile and profile_every > 0 and (
                    bi % profile_every == 0 or did_step or bi < 5
                ):
                    row = {
                        "bi": bi,
                        "step": global_step,
                        "micro": micro,
                        "data_ms": t_data,
                        "fiber_ms": t_fiber,
                        "fwd_ms": t_fwd,
                        "bwd_ms": t_bwd,
                        "opt_ms": t_opt,
                        "loss": loss_val,
                        "tgt_tok": n_tgt_tok,
                        "did_step": did_step,
                    }
                    profile_rows.append(row)
                    if bi < 8 or did_step or (micro_log_every and bi % micro_log_every == 0):
                        log(
                            f"prof bi={bi} step={global_step} micro={micro}/"
                            f"{grad_accum} data_ms={t_data:.1f} fiber_ms={t_fiber:.1f} "
                            f"fwd_ms={t_fwd:.1f} bwd_ms={t_bwd:.1f} opt_ms={t_opt:.1f} "
                            f"loss={loss_val:.4f} tgt_tok={n_tgt_tok}"
                        )

            meta = {
                "backbone": backbone_name,
                "epoch": epoch,
                "step": global_step,
                "loss_last": history[-1]["loss"] if history else None,
            }
            ckpt_path = out_dir / f"checkpoint_epoch{epoch:02d}.pt"
            flat_holder.set_state(_flat_trainable_state(model, global_step))
            db.record_epoch(run_id, epoch, meta)
            db.save_checkpoint(run_id, epoch, flat_holder, opt, device="cpu")
            _save_pt(ckpt_path, model, opt, epoch, global_step, meta)
            log(f"checkpoint epoch={epoch} → {ckpt_path}")

            do_epoch_eval = not profile and not (
                max_steps is not None and global_step >= max_steps
            )
            if do_epoch_eval:
                eval_path = out_dir / f"eval_samples_epoch{epoch:02d}.jsonl"
                try:
                    log(f"epoch eval_sample_log begin n={cfg.get('eval_n', 64)}")
                    eval_sample_log(
                        model,
                        tokenizer,
                        codec,
                        held_ds,
                        eval_path,
                        n=int(cfg.get("eval_n", 64)),
                        device=device,
                    )
                except Exception as e:  # noqa: BLE001
                    log(f"epoch eval_sample_log failed: {e}")

            if stop["flag"] or (max_steps is not None and global_step >= max_steps):
                break

        latest = out_dir / "checkpoint_latest.pt"
        import shutil

        existing = sorted(out_dir.glob("checkpoint_epoch*.pt"))
        if existing:
            shutil.copy2(existing[-1], latest)
        log(
            f"done elapsed_h={(time.time() - t_run) / 3600:.2f} steps={global_step} "
            f"nan_skips={nan_skips} grad_nan_skips={grad_nan_skips}"
        )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if profile_rows:
        (out_dir / "profile_micros.json").write_text(
            json.dumps(profile_rows, indent=2) + "\n"
        )
    return {
        "run_id": run_id,
        "backbone": backbone_name,
        "steps": global_step,
        "history": history,
        "out_dir": str(out_dir),
        "vram_gb": _vram_gb(),
        "nan_skips": nan_skips,
        "grad_nan_skips": grad_nan_skips,
        "profile_rows": profile_rows,
    }


def baseline_unconditioned_loss(
    cfg: dict[str, Any], n_batches: int = 32
) -> float:
    """Cheap no-condition CE baseline (prefix always dropped) for first-epoch compare."""
    device = _device("auto")
    zbank = _ROOT / cfg["zbank"]
    ds = ParaPairDataset(_ROOT / cfg["corpus"], zbank, split="train")
    backbone, tokenizer, name = _resolve_backbone(cfg, device)
    adapter = SoftPrefixAdapter(
        z_dim=int(cfg["z_dim"]),
        d_model=int(getattr(backbone.config, "hidden_size", cfg["d_model"])),
        n_prefix=int(cfg["n_prefix"]),
        hidden=int(cfg.get("adapter_hidden", 512)),
    ).to(device)
    cfg = dict(cfg)
    cfg["d_model"] = int(backbone.config.hidden_size)
    model = StageAModel(backbone, adapter).eval()
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        collate_fn=_make_collate(tokenizer, cfg),
    )
    losses: list[float] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            z = batch["z"].to(device)
            drop = torch.ones(z.size(0), dtype=torch.bool, device=device)
            out = model(
                z,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
                drop_mask=drop,
            )
            losses.append(float(out.loss.item()))
    mean = float(np.mean(losses)) if losses else float("nan")
    print(
        f"[stageA] no-condition baseline CE ({n_batches} batches, {name}): {mean:.4f}",
        flush=True,
    )
    del model, backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mean


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pivot Stage-A QLoRA + soft-prefix trainer")
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--no-gpu-gate", action="store_true")
    p.add_argument("--profile", action="store_true", help="Per-micro timing logs")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--log-path", type=Path, default=None)
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--no-pretokenize", action="store_true")
    args = p.parse_args(argv)
    print(f"[stageA] boot config={args.config} resume={not args.no_resume}", flush=True)
    cfg = _load_cfg(args.config)
    if args.no_gpu_gate:
        cfg["gpu_gate"] = False
    if args.kappa is not None:
        cfg["kappa_aug"] = float(args.kappa)
    if args.no_pretokenize:
        cfg["pretokenize"] = False
    if args.baseline_only:
        baseline_unconditioned_loss(cfg)
        return
    train(
        cfg,
        resume=not args.no_resume,
        max_steps=args.max_steps,
        profile=args.profile,
        skip_baseline=args.skip_baseline,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
