# Determinism forensics, 2026-11-26

Read-only investigation of PID 552497 on the shared RTX 3090 (24,576 MiB). No process was signalled. Source files were not edited. The process exited by itself at 12:01:10 EST after writing a complete result.

Clock used below: America/New_York. Wifi-off is 23:30. At 12:05:08, 11.41 hours of the window remained.

## 1. What PID 552497 was

It was the first of a determinism pair at the operating point, not a repeat of an earlier probe and not the 30M-token arm. It was left alone because it was on that config and was advancing. It then finished.

Command line, from `/proc/552497/cmdline`:

```
/home/jpearce/Documents/Lucid/.venv/bin/python -u -m experiments.e1.train \
  --arm D --seed 0 --tokens 2000000 --context 512 --batch-size 2 --grad-accum 4 \
  --max-docs 4000 --step-trace \
  --corpus /home/jpearce/Documents/Lucid/data/corpus_v2.txt \
  --checkpoint-dir /home/jpearce/Documents/Lucid/artifacts/e1/harness_validation/repeat \
  --out /home/jpearce/Documents/Lucid/artifacts/e1/harness_validation/D_seed0_run1.json
```

| Fact | Value |
|---|---|
| Start | 2026-11-26 11:44:48 EST (`btime` + `starttime` / 100) |
| End of `run()` | 2026-11-26 12:01:10 EST (start + `wall_clock_s`) |
| Wall clock | 981.882 s (16.36 min) |
| cwd | `/tmp/lucid-e1-run` (git worktree, commit `bdd612a`, clean) |
| Parent | PID 551787, `/bin/bash -l`, now gone |
| stdout and stderr | `artifacts/e1/harness_validation/D_seed0_run1.log` |
| Arm / seed | D / 0. Arm D is plain LoRA: no block, no JEPA, no head |
| Model / dtype | `Qwen/Qwen3-1.7B-Base`, bfloat16, CUDA, determinism flags on |
| Shape | context 512, batch 2, grad_accum 4 → 4,096 tokens per optimiser step |
| Budget | `--tokens 2000000`. The arm table is 30,000,000; the log warning is the harness saying so. This process did not run 30M |
| Steps | 489 = ceil(2,000,000 / 4,096). Tokens seen 2,002,944. `max_steps` unset |
| When first observed (~12:00) | step 475/489, 1,945,600 tokens, still running |
| Finish | step 489/489, `nan_skips` 0, both JSON files parse |

Result files, both valid JSON:

- `artifacts/e1/harness_validation/D_seed0_run1.json` (40,407 bytes)
- `artifacts/e1/harness_validation/D_seed0_run1.manifest.json`
- checkpoint `artifacts/e1/harness_validation/repeat/D_s0/final` (`trainer_state.json` says step 489)
- a second copy `artifacts/e1/harness_validation/repeat/D_s0_run1/final`, different inode, same SHA-256 of `adapter_model.safetensors` (`baba5ff93b16cc58d62031b48ee23c1965f4b74d4a1521586c37387eaa830402`). No repository script names that path. Timestamps match the original, which is what `cp -a` preserves. Keep both until run 2 has been compared. A later run that passes the same `--checkpoint-dir .../repeat` overwrites `repeat/D_s0/final` only.

Manifest of this run: `peak_vram_gb` 8.861928939819336, `config_sha256` `afd0627f8c28565ac3285c77dd24f66a878fa02373c11b18d42c3bb5fc491559`, `git_commit` `bdd612a`, `git_dirty` false, `n_trainable` 17,432,576.

Throughput, from `train.history[].seconds` (the train-loop clock, which includes the dev eval every 25 steps) and from `wall_clock_s`:

| Window | tok/s |
|---|---|
| Train loop, 2,002,944 tokens / 967.55 s | 2,070 |
| Wall, including load and final eval, / 981.882 s | 2,040 |
| Steps 1–25, before llama-server was resident | 3,157 |
| Steps 25–475, llama-server resident | 2,046 |

The first 25 steps match `probe_tps_on` (deterministic on, same shape, 100 steps, 3,178 tok/s over its train loop). `probe_tps_off` (same shape, determinism off) was 3,452 tok/s. The drop after step 25 lines up with llama-server PID 631297 attaching (~4,572 MiB from t+73 s) and staying for the rest of the run. `--step-trace` does not explain it: the fast interval is the traced run, before the other GPU tenant arrived.

`probe_d` and `probe_tps_on` are not a pair and are not this run. `probe_d` is `max_steps=3` (12,288 tokens, `step_trace` on). `probe_tps_on` is `max_steps=100` (409,600 tokens, `step_trace` off). Their histories agree with this run on the four shared log rows (steps 25, 50, 75, 100: `lm`, `tokens`, `val_nll` all equal) and then stop. That agreement is the same trajectory observed twice for 100 steps. It is not a finished determinism test at 2M tokens.

## 2. Memory

Verdict: **(b) wasteful but safe** for a single run of this shape. The footprint is flat, it is the same number the short probes already recorded, and this 489-step run finished. It is over the 4–8 GB sharing budget once the caching allocator's reservation is counted, and a second Lucid job does not fit. It is not on a path that grows into an OOM of its own output.

Two different instruments, one run:

| Instrument | Number | What it measures |
|---|---|---|
| `torch.cuda.max_memory_allocated`, stored as `peak_vram_gb` | 8.861928939819336 GiB = 9,515,423,744 bytes = 9,074.6 MiB | Peak bytes PyTorch gave to tensors. Authoritative for "how big was the working set" |
| `nvidia-smi` for PID 552497 | 9,816 MiB for 3 samples, then **11,732 MiB for 94 samples**, never higher | What the card actually lost. The caching allocator does not return the gap |

The same allocated peak, to every stored digit, is in `probe_d.manifest.json`, `probe_tps_on.manifest.json`, `probe_tps_off.manifest.json`, and `D_seed0_run1.manifest.json`. A 3-step run and a 489-step run peak at the same byte count, so the peak is hit in the first steps and does not grow with step count, with `--step-trace`, or with the determinism flags (`probe_tps_off` had them off).

The same 11,732 MiB nvidia-smi reservation was also the high-water mark of two earlier Lucid PIDs in `vram_procs.log` (533739 and 544798), each of which went 9,816 then 11,732. 11.7 GB is the operating-point reservation, not a leak unique to the long run.

Gap held and not used by tensors: 11,732 − 9,074.6 = **2,657 MiB**.

Where the 9,074.6 MiB allocated goes. Arm D trains LoRA only (`n_trainable` 17,432,576, `n_block` 0, `n_head` 0). The adapter file is fp32 (17,432,576 × 4 = 69,730,304 parameter bytes inside a 69,782,384-byte safetensors file). `train()` builds `AdamW` on `trainable_parameters()` only, `betas=(0.9, 0.95)`, and calls `zero_grad(set_to_none=True)`.

| Piece | Size |
|---|---|
| Base checkpoint `model.safetensors` (bf16 weights that have to stay resident; LoRA still backprops through them) | 3,441,185,608 bytes = 3,281.8 MiB |
| LoRA fp32 parameters | 66.5 MiB |
| LoRA fp32 gradient, present during backward, dropped between steps | 66.5 MiB |
| Adam moments `exp_avg` + `exp_avg_sq`, fp32 | 133.0 MiB |
| Optimizer side at the backward peak | 266 MiB |
| Residual: allocated peak minus weights minus that optimizer side | 9,074.6 − 3,281.8 − 266 = **5,527 MiB** |

That residual is one microbatch, not four. The loop calls `backward()` inside the `grad_accum` loop, so the autograd graph of microbatch *k* is freed before microbatch *k+1* is built. Gradient accumulation adds LoRA-sized gradient buffers, which are already inside the 266 MiB, and does not stack four activation graphs.

Of the 5,527 MiB, the shapes in the forward account for two small named tensors and leave the rest as backward-saved activations plus CUDA workspace:

- bf16 logits, batch 2 × 512 × vocab 151,936 × 2 bytes = 297 MiB. The loss then upcasts; that workspace sits in the residual, not beside it as a second full copy I could measure.
- `output_hidden_states=True` is unconditional in `E1Model.forward`, including arm D, whose loss uses only logits. The tuple is 29 tensors × 2 × 512 × 2048 × 2 bytes = 116 MiB, and those tensors are largely the activations autograd already holds.

A full-model AdamW is ruled out by the parameter count and by the arithmetic. Moments alone for the 1.72B bf16 checkpoint would be 1,720,592,804 × 8 bytes = 12.82 GiB, before gradients and before the 3.28 GiB of weights. That does not fit in the measured 9.07 GiB allocated peak, and `n_trainable` would be ~1.72B rather than 17.4M.

There is no gradient checkpointing in this trainer. The ~5.1 GiB of unnamed residual is the cost of saving the 28-layer backward at batch 2, context 512. That is inherent to this loop at this shape. It is waste relative to a 4–8 GB sharing cap: allocated is already 8.86 GiB, and the card loses 11.46 GiB.

Card total during the run (`vram.csv`), after the process had settled at 11,732 MiB and the extra tenant below had gone: median **19,048 MiB**, max 19,053 MiB, so about 5,528 MiB free. The process's own line never moved.

The 23,019 MiB card spike at t+144 s (sample 1795711632) was other tenants on top of a flat 11,732 MiB, not this process growing. At 1795711571 the compute processes were:

| PID | MiB | What |
|---|---|---|
| 552497 | 11,732 | this run, already at its plateau |
| 631297 | 4,570 | llama-server, just loaded (it had been absent at t+0) |
| 554993 | 3,966 | a short-lived `.venv/bin/python`, 10 samples, ~90 s, gone by ~1795711652. Cmdline was already gone when inspected |
| desktop / capture compute apps | ~1,503 | EA Desktop, sunshine, two small pythons |

Plus the graphics clients that do not appear in the compute-apps list, this sums to the observed ~23.0 GiB. Headroom at that moment was about 1.5 GiB. The training run survived it. A plan that counts on that overlap is not safe; the training run by itself, held at 11,732 MiB with ~5.5 GiB free beside llama-server and the desktop, is.

Classification against the three labels:

- (a) benign would require the job to sit inside the 4–8 GB cap the card was being shared under. 11,732 MiB reserved does not.
- (b) wasteful but safe: 2,657 MiB sits in the allocator above the tensor peak, the tensor peak itself is ~5.5 GiB of activations for a frozen 3.3 GiB base, and neither number grows. A solo run of this shape, including a 30M-token run of the same shape, does not OOM itself. This run is the evidence: 489 steps, peak identical to the 3-step probe, clean JSON, checkpoint step 489.
- (c) does not apply to this job alone. It applies to putting a second Lucid train of this shape on the card. See §5.

## 3. Stop decision

Left it running. No signal was sent.

At the time of the decision it was step 475/489 of the config a determinism pair actually needs, losses were finite, the nvidia-smi line had been flat at 11,732 MiB since t+33 s, and the output JSON did not yet exist because the trainer writes it once, after the loop. Stopping it would have thrown away a run that was about to answer the question. It then exited on its own. GPU-minutes lost to a kill: **0**. GPU-minutes consumed by the completed run: **16.36**.

SIGTERM is not safe on this trainer, which is a separate fact from the decision. `experiments/e1/train.py` installs no signal handler and no `atexit` hook (checked on the worktree copy that actually ran, commit `bdd612a`; it matches the workspace file). The harvest supervisor's per-item flush does not exist here.

- `checkpoint_every` defaults to 500 and this run is 489 steps, so the interval checkpoint never fires. The only checkpoint is `save_checkpoint` of `.../final`, after the loop returns.
- The result JSON is `Path.write_text` of the whole document, once, after eval. `write_text` truncates and then writes. A signal before that line leaves no file. A signal during it can leave a partial file that a later reader could mistake for a result.
- The `BrokenPipeError` on fd 44 in the log is the inductor compile worker at interpreter shutdown. The same line is in the finished probe logs. It is after `wrote D_seed0_run1.json`. Both JSON files parse.

## 4. What an honest validation requires, and whether it fits

One run is done. The test is a second process with the same training inputs, compared against this run. `probe_d` vs `probe_tps_on` is not that test: different `max_steps`, different `step_trace`, and neither is 489 steps.

Run 2 has to match every field that changes a number: arm D, seed 0, `--tokens 2000000`, `--context 512`, `--batch-size 2`, `--grad-accum 4`, `--max-docs 4000`, `--step-trace`, `--dtype bfloat16`, the same corpus file, determinism left at its default (on), `--log-every` left at 25, no `--max-steps`. Launch it from `/tmp/lucid-e1-run` at `bdd612a` so the code matches the manifest (`git_dirty` false). `hash_config` hashes the whole config, including `checkpoint_dir`, so the two `config_sha256` values will differ if the checkpoint directory differs. That difference is required, because the directories must differ, and it is not a training difference. Compare the fields above, not the hashes.

Compare, from the two JSON files and the two `adapter_model.safetensors`:

1. Every logged loss. `train.history` has 20 rows (steps 25, 50, …, 475, 489), and the value to compare is `lm` plus `val_nll` and `val_context_gain`. The log line prints `lm` to 4 decimals; the JSON holds the full float (`2.54296875`, not the printed `2.5430`). `--step-trace` also stored all 489 steps: `step_trace.loss_sum_f64_hex` is the f64 bit pattern of the **sum** of the microbatch losses (the logged `lm` is that sum divided by the microbatch count), `loss_xor` for this run is `4026380000000000`, `init_param_xor` is `10b3af00f9855552`, `batch_xor` is `0b1431514485c50e`, `tainted_steps` is empty. A batch-xor mismatch means the data order moved before the loss did.
2. Eval, elementwise. Arm D stores one per-item array, `eval.context_gain_per_item`, length 64. `nll_true` and `nll_shuffled` are means only; the per-item NLLs are computed and discarded. There is no `necessity_per_item` on arm D because there is no block.
3. Checkpoint tensors with `torch.equal` on every tensor in `adapter_model.safetensors`. The base weights are frozen and are not in the checkpoint. This run's final file is the SHA-256 above, 69,782,384 bytes, step 489.

Cost, from this run, which is the only full-length measurement:

| | Time |
|---|---|
| Run 1, already spent, measured wall | 16.36 min |
| Run 2, same card conditions (llama-server resident, ~2,070 tok/s) | ~16.4 min |
| Run 2 if the card is as quiet as `probe_tps_on` (3,178 tok/s, determinism on) | ~11 min (489/100 × 128.9 s train, plus the ~13 s load and final eval that probe measured once) |
| Pair, total, at the measured rate | ~33 GPU-min |

Plan on 16 minutes for run 2 while llama-server PID 631297 is up. Do not kill llama-server as part of this; it is not this job.

The 30M-token arm is a different run. Same shape, so the same memory, about 15× the steps: 30,000,000 / 2,070 tok/s ≈ 4.03 h at the rate this run actually held. It does not answer the determinism question any better than the 2M pair, because the GEMM shape is the same. It is a duration test. It should not start before run 2 exists.

Window, serial, because §5 says only one Lucid train fits:

| Job | Hours |
|---|---|
| Run 2 | 0.27 |
| E4-S1 | 4 |
| E2 smoke | 1 |
| Sum | 5.27 |
| Remaining at 12:05 | 11.41 |

All three fit, in that order, with about 6.1 hours spare. They do not fit beside each other. Run 2 goes first because run 1 is already sitting on disk and is not a result until the twin exists, and because it is the short job. E4-S1 second, E2 smoke third. Adding the 30M arm (~4.0 h at the measured rate) after those three is 9.3 hours, still inside 11.4, and still only if nothing else takes the card. It is not the next thing to launch.

## 5. Concurrency the arbiter should use

**Maximum safe concurrent Lucid GPU jobs of this shape: 1.**

Card: 24,576 MiB. Use the nvidia-smi reservation, 11,732 MiB, as the per-job cost. The 8.86 GiB figure is the tensor peak of the same job, and the allocator does not give the 2,657 MiB difference back.

Non-Lucid floor, measured at 12:05 after this job had exited: GPU used 7,310 MiB, of which llama-server PID 631297 held 4,572 MiB, so the desktop and capture stack held 7,310 − 4,572 = **2,738 MiB**.

| Combination | Sum (MiB) | Against 24,576 |
|---|---|---|
| 1 × 11,732 + llama 4,572 + desktop 2,738 | 19,042 | free 5,534. Matches the in-run median of 19,048 used |
| 2 × 11,732 + llama 4,572 + desktop 2,738 | 30,774 | over by 6,198 |
| 2 × 11,732 + desktop 2,738, llama stopped | 26,202 | over by 1,626 |
| 2 × 9,074.6 allocated + llama 4,572 + desktop 2,738 | 25,459 | still over by 883. Switching the accounting to the tensor peak does not create a second slot while llama-server and the desktop stay |
| This job + the observed llama-server high-water of 8,594 MiB (PID 525555 in the same log) + desktop 2,738 | 23,064 | free 1,512. This is the 23.0 GiB spike, and it is the edge that already happened |

One Lucid train co-resides with the current llama-server and the desktop, with ~5.5 GiB free. That free space does not hold a second train (it needs 11,732 MiB). Two trains do not fit even if llama-server is gone. Anything that needs more than ~5.5 GiB while this shape and llama-server and the desktop are up does not fit either. The short 3,966 MiB python that overlapped this run is what pushed the card to 23.0 GiB; that overlap should not be repeated on purpose.

## 6. The one next action on this card

Launch run 2 now, from `/tmp/lucid-e1-run`, before E4-S1 or anything else takes the GPU. Same flags as §1, with a new output path and a new checkpoint directory so `repeat/D_s0/final` and the spare copy `repeat/D_s0_run1/final` stay untouched:

```
cd /tmp/lucid-e1-run
/home/jpearce/Documents/Lucid/.venv/bin/python -u -m experiments.e1.train \
  --arm D --seed 0 --tokens 2000000 --context 512 --batch-size 2 --grad-accum 4 \
  --max-docs 4000 --step-trace \
  --corpus /home/jpearce/Documents/Lucid/data/corpus_v2.txt \
  --checkpoint-dir /home/jpearce/Documents/Lucid/artifacts/e1/harness_validation/repeat_run2 \
  --out /home/jpearce/Documents/Lucid/artifacts/e1/harness_validation/D_seed0_run2.json
```

Then compare history, `step_trace`, `context_gain_per_item`, and `torch.equal` on the two adapter files, as in §4. Expect about 16 minutes while llama-server stays resident. The card at 12:05 had no Lucid train on it (GPU util 0, 16,806 MiB free).
