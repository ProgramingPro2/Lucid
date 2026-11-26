# E1 harness evidence, 2026-11-26

Commands below were run on this host (America/New_York) from the repository root with `.venv/bin/python`. They check the harness fixes that voided the 2026-11-20 pilot: the inserted block's causal mask, a matched short repeat, the arm-table token budget, the pilot sample-size helper, a real run manifest and checkpoint, a non-zero washout reading, the harvest quiet window, the live corpus file, and the unit suite. The training commands use the tiny random Qwen3 smoke model on CPU. They are not a result about whether the block is load-bearing, and they are not a certification of the context-512 bf16 GPU operating point. A separate GPU validation is in flight. Trainer manifests written during this capture record git commit `77d808171f2683342408f38bdd1979a628f589b5`. The branch moved to `2019328b928409fae335a51d9974c911718401f3` while the capture was in progress; these commands were not re-run on that commit.

## 1. Causal mask

The tiny E1 model is built on CPU. The inserted block's output projection is drawn from a normal distribution, so the block is not the identity. One token, at position 20 of a length-32 batch, is changed. Logits at positions 0 through 19 are compared to the unperturbed forward.

```bash
.venv/bin/python - <<'PY'
import torch
from experiments.e1.block import JEPABlock
from experiments.e1.train import E1Model, TrainConfig, attach_lora, build_base

torch.manual_seed(0)
cfg = TrainConfig(arm="A", seed=0, device="cpu", dtype="float32")
base = attach_lora(build_base(cfg, tiny=True), cfg, tiny=True)
block = JEPABlock(64, n_heads=8, zero_init_output=False)
with torch.no_grad():
    block.out.weight.normal_()
    block.out.bias.normal_()
model = E1Model(base, block=block).eval()

g = torch.Generator().manual_seed(0)
ids = torch.randint(0, 512, (2, 32), generator=g)
changed_pos = 20
perturbed = ids.clone()
perturbed[:, changed_pos] = (perturbed[:, changed_pos] + 17) % 512

with torch.no_grad():
    logits = model(ids).logits
    logits_p = model(perturbed).logits

earlier = (logits[:, :changed_pos] - logits_p[:, :changed_pos]).abs()
changed = (logits[:, changed_pos] - logits_p[:, changed_pos]).abs()
print(f"device={next(model.parameters()).device.type} dtype={next(model.parameters()).dtype}")
print(f"batch={tuple(ids.shape)} changed_position={changed_pos}")
print(f"token_actually_changed={not torch.equal(ids[:, changed_pos], perturbed[:, changed_pos])}")
print(f"out_weight_abs_sum={float(block.out.weight.abs().sum())}")
print(f"out_is_all_zero={bool(torch.count_nonzero(block.out.weight) == 0)}")
print(f"earlier_positions=0..{changed_pos - 1}")
print(f"earlier_equal={torch.equal(logits[:, :changed_pos], logits_p[:, :changed_pos])}")
print(f"past_max={float(earlier.max())}")
print(f"past_max_is_exactly_0={float(earlier.max()) == 0.0}")
print(f"changed_position_equal={torch.equal(logits[:, changed_pos], logits_p[:, changed_pos])}")
print(f"changed_position_abs_max={float(changed.max())}")
print(f"changed_position_abs_mean={float(changed.mean())}")
PY
```

```text
device=cpu dtype=torch.float32
batch=(2, 32) changed_position=20
token_actually_changed=True
out_weight_abs_sum=3347.422607421875
out_is_all_zero=False
earlier_positions=0..19
earlier_equal=True
past_max=0.0
past_max_is_exactly_0=True
changed_position_equal=False
changed_position_abs_max=0.1390606164932251
changed_position_abs_mean=0.024404380470514297
```

Exit code 0. Stderr then printed `Exception ignored in: <_io.BufferedWriter name=25>` and `BrokenPipeError: [Errno 32] Broken pipe` during interpreter shutdown. That line is not a failed comparison.

What this proves: with a non-zero output projection, changing token 20 leaves logits at positions 0–19 bit-identical (`past_max` is exactly 0.0) and moves the logits at position 20 (max absolute change 0.1390606164932251).

What this does not prove: the same bitwise result on Qwen3-1.7B in bf16, or that the language-model loss of a trained block is useful. A zero-init block would pass this check without being causal; the output projection here is not zero.

## 2. Repeatability

Two separate processes, same command, arm D, seed 0, determinism on. Both were limited to 4 steps. The trainer has no CLI flags for `eval_prefix` and `eval_target`; with `--context 32` it stores `eval_prefix = min(384, 16) = 16` and `eval_target = min(128, 8) = 8`. Both saved configs show those values. `log_every` is the CLI default, 25, so the history records the final step only. Both processes were given the same checkpoint directory so the hashed config would not differ by an output path.

```bash
.venv/bin/python -m experiments.e1.train \
  --arm D --seed 0 --tiny \
  --tokens 4096 --context 32 --batch-size 2 --grad-accum 1 \
  --max-steps 4 --eval-items 4 \
  --device cpu --dtype float32 --deterministic --log-every 25 \
  --checkpoint-dir /tmp/e1-demo-evidence/d-same-ckpts \
  --out /tmp/e1-demo-evidence/d-same-1.json
.venv/bin/python -m experiments.e1.train \
  --arm D --seed 0 --tiny \
  --tokens 4096 --context 32 --batch-size 2 --grad-accum 1 \
  --max-steps 4 --eval-items 4 \
  --device cpu --dtype float32 --deterministic --log-every 25 \
  --checkpoint-dir /tmp/e1-demo-evidence/d-same-ckpts \
  --out /tmp/e1-demo-evidence/d-same-2.json
```

Both logs, trimmed to the step line (each also printed the arm-table warning for `--tokens 4096` versus 30000000, then exit 0):

```text
[D s0] step 4/4 lm=6.2346 std_z=0.0276 erank=27.75 val_nll=6.2744 val_cg=0.0184 tokens=256
P1 context gain : +0.02629 nats/token
```

Comparison of the two JSON results:

```text
config_fields_equal True
  arm='D'
  seed=0
  tokens=4096
  context=32
  batch_size=2
  grad_accum=1
  max_steps=4
  eval_items=4
  eval_prefix=16
  eval_target=8
  device='cpu'
  dtype='float32'
  deterministic=True
  log_every=25
  checkpoint_dir='/tmp/e1-demo-evidence/d-same-ckpts'
config_diff_keys []
steps 4 4
history_steps [4] [4]
final_lm_run1 6.234643459320068
final_lm_run2 6.234643459320068
final_lm_equal True
context_gain_per_item_run1 [-0.014926433563232422, 0.09246492385864258, -0.015810012817382812, 0.04341602325439453]
context_gain_per_item_run2 [-0.014926433563232422, 0.09246492385864258, -0.015810012817382812, 0.04341602325439453]
context_gain_per_item_equal True
eval_equal True
config_sha256_run1 83da468deec4bf10535a1d614224c6c95641ca3d5e2f2afcc0e89f3a7df2264a
config_sha256_run2 83da468deec4bf10535a1d614224c6c95641ca3d5e2f2afcc0e89f3a7df2264a
config_sha256_equal True
manifest_diff_keys ['wall_clock_s']
  manifest wall_clock_s: 20.964114266913384
  manifest wall_clock_s: 22.92455826420337
git_commit 77d808171f2683342408f38bdd1979a628f589b5 dirty True
determinism_enabled True True
```

Both manifests recorded `git_dirty` true and the same `git_diff_sha256` `51a29faac5db5a823899f89ba65de4d372b1f51df61d056d5ba3aa95814934f2`. At that moment the dirty paths were the corpus files another agent was editing, not `experiments/e1`. Arm D's eval dict has no necessity array; the per-item array it does have is `context_gain_per_item`, and the whole eval dict matched.

An earlier pair, about eight minutes before, used the same training knobs and the same 4-step cap but different checkpoint directories. Its config hashes differed only in `checkpoint_dir`. Final loss was again `6.234643459320068` on both sides, and the same four context-gain values matched. That pair's manifests recorded a clean tree at the same commit.

What this proves: two fresh CPU processes, arm D, seed 0, determinism on, same 4-step config, produced the same final logged LM loss and the same per-item context-gain array, exactly.

What this does not prove: repeatability at the confirmatory operating point. CPU tiny float32, context 32, batch 2, four steps, is not a certification of context 512, bf16, Qwen3-1.7B-Base, gradient accumulation greater than 1, or a 30M-token run. A separate GPU validation is in flight. This pair is arm D, which has no inserted block.

## 3. Token budget single source of truth

```bash
.venv/bin/python - <<'PY'
from experiments.e1.arms import TOKENS_PER_RUN, all_arms, resolve_token_budget, schedule_rows
from experiments.e1.train import build_arg_parser

print(f"TOKENS_PER_RUN={TOKENS_PER_RUN}")
print("arm token_budget resolve(None) warning")
for arm in all_arms():
    tokens, warning = resolve_token_budget(arm.name, None)
    print(f"{arm.name:8} {arm.token_budget:10d} {tokens:10d} warning={warning}")
for row in schedule_rows():
    print(f"{row.pattern:16} count={row.count:3d} token_budget={row.token_budget}")
action = build_arg_parser()._option_string_actions["--tokens"]
print(f"argparse_default={action.default!r}")
print(f"argparse_help={action.help}")
PY
.venv/bin/python -m experiments.e1.train --help | rg -n "token_budget|--tokens|30M"
```

```text
TOKENS_PER_RUN=30000000
arm token_budget resolve(None) warning
A          30000000   30000000 warning=None
B          30000000   30000000 warning=None
A_prime    30000000   30000000 warning=None
B_prime    30000000   30000000 warning=None
E          30000000   30000000 warning=None
D          30000000   30000000 warning=None
N                 0          0 warning=None
PC_hi       2000000    2000000 warning=None
PC_lo       2000000    2000000 warning=None
e1/trunk/s*      count= 32 token_budget=30000000
e1/*/s*.joint    count= 32 token_budget=30000000
e1/A/s*.dose*    count= 48 token_budget=30000000
argparse_default=None
argparse_help=override the arm-table token budget (default: by_name(arm).token_budget)
```

From `--help` (the docstring is wrapped into the usage text):

```text
The default token count is that arm's ``token_budget``;
``--tokens`` is an override and prints a warning when it disagrees with the
arm table.
The token count defaults to ``by_name(arm).token_budget`` (30M for A).
  --tokens TOKENS       override the arm-table token budget (default:
                        by_name(arm).token_budget)
```

What this proves: `TOKENS_PER_RUN` is 30000000, arms A, B, A′, B′, E, and D use it, and a missing `--tokens` resolves to that arm's own budget with no warning. The CLI default is `None`, and the help text says the default is `by_name(arm).token_budget` rather than a literal. N is 0 because it does not train. PC_hi and PC_lo are 2000000 because that is their assay budget in the same table.

What this does not prove: that every launcher omits the override. Passing `--tokens 2000000` still runs, and it warns. The old shared 2M default was dangerous because the voided pilot trained 2,002,944 tokens, 0.135 epochs of its 14,821,030-token train split, instead of the pre-registered 30M budget (`docs/runs/2026-11-20-e1-pilot.md`).

## 4. Sample-size reporting

The on-disk pilot summary was only read. `summarise` was called in process and was not asked to rewrite `summary.json`.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
raw = json.loads(Path("artifacts/e1/pilot/summary.json").read_text())
print(json.dumps(raw["A_minus_B"], indent=2))
PY
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from experiments.e1.pilot import DEFAULT_PLAN, summarise
summary = summarise(Path("artifacts/e1/pilot"), DEFAULT_PLAN, 0.0112)
ab = summary["A_minus_B"]
print(json.dumps(ab, indent=2))
print(
    "estimated n from a within-design paired SD "
    "(not a run-level variance), "
    f"{ab['n_pairs']} seeds, inflation {ab['pilot_sd_inflation']:.2f}, "
    f"floor 24, cap 64, MPID=0.0112: "
    f"{ab['estimated_n_within_design']}"
)
PY
```

Stored `artifacts/e1/pilot/summary.json`, key `A_minus_B`:

```text
{
  "n_pairs": 4,
  "mean": 0.0004563541151583195,
  "sd": 0.0014652899559288886,
  "per_seed": {
    "0": 0.0017165038734674454,
    "1": 0.0017321966588497162,
    "2": -0.0008817985653877258,
    "3": -0.0007414855062961578
  },
  "required_n_at_pilot_sd": 1
}
```

`experiments.e1.pilot.summarise` on the same directory, MPID 0.0112:

```text
{
  "n_pairs": 4,
  "mean": 0.0004563541151583195,
  "sd": 0.0014652899559288886,
  "per_seed": {
    "0": 0.0017165038734674454,
    "1": 0.0017321966588497162,
    "2": -0.0008817985653877258,
    "3": -0.0007414855062961578
  },
  "pilot_sd_inflation": 1.7275872940849268,
  "estimated_n_within_design": 24,
  "variance_note": "within-design paired SD across seeds, not a run-level variance"
}
estimated n from a within-design paired SD (not a run-level variance), 4 seeds, inflation 1.73, floor 24, cap 64, MPID=0.0112: 24
```

What this proves: recomputing from the saved pilot JSON replaces `required_n_at_pilot_sd: 1` with `pilot_sd_inflation` 1.7275872940849268 (1.73 at two decimals, the 4-seed factor) and `estimated_n_within_design` 24, which is the floor the helper applies.

What this does not prove: that 24 is a fresh power calculation from a run-level variance. The old "required n = 1" was wrong because A and B shared data order and init, so the per-seed spread was a within-design variance, not a run-level one, and no small-pilot inflation was applied. The stored summary file still contains the old field; this capture did not rewrite it.

## 5. Run manifest and checkpoints

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/e1/harness_validation/probe_tps_on.manifest.json")
data = json.loads(p.read_text())
print("keys", sorted(data))
for name in (
    "determinism", "git_dirty", "git_diff_sha256", "nan_skips",
    "peak_vram_gb", "final_checkpoint", "spawn_keys",
):
    print(f"{name}={data[name]!r}")
print(f"token_budget={data['token_budget']!r}")
print(f"run_id={data['run_id']!r}")
print(f"git_commit={data['git_commit']!r}")
PY
find artifacts/e1/harness_validation/probe_tps_on/D_s0/final -type f -printf '%s\t%p\n' | sort
cat artifacts/e1/harness_validation/probe_tps_on/D_s0/final/trainer_state.json
git grep -n "torch.save" 60eeade^ -- experiments/e1/train.py || echo "NO torch.save in experiments/e1/train.py at 60eeade^"
git grep -n "torch.save" 60eeade -- experiments/e1/train.py
printf '' | sha256sum
```

```text
keys ['arms', 'config_sha256', 'corpus_sha256', 'determinism', 'final_checkpoint', 'git_commit', 'git_diff_sha256', 'git_dirty', 'nan_skips', 'notes', 'peak_vram_gb', 'run_id', 'spawn_keys', 'token_budget', 'wall_clock_s']
determinism={'cublas_workspace_config': ':4096:8', 'cudnn_allow_tf32': False, 'cudnn_benchmark': False, 'cudnn_deterministic': True, 'deterministic_algorithms': True, 'deterministic_algorithms_warn_only': False, 'enabled': True, 'float32_matmul_precision': 'highest', 'matmul_allow_tf32': False}
git_dirty=False
git_diff_sha256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
nan_skips=0
peak_vram_gb=8.861928939819336
final_checkpoint='/home/jpearce/Documents/Lucid/artifacts/e1/harness_validation/probe_tps_on/D_s0/final'
spawn_keys={'block_init': 15203753715784122704, 'data': 17695361009812855374, 'decode': 3475641800253881583, 'dropout': 3517829066415317750, 'head_init': 11469141701766028007, 'lora_init': 16178048237645280395}
token_budget=2000000
run_id='e1/D/s0'
git_commit='bdd612a475743b853739d77d1593799f178b9589'
1098	artifacts/e1/harness_validation/probe_tps_on/D_s0/final/adapter/adapter_config.json
14	artifacts/e1/harness_validation/probe_tps_on/D_s0/final/trainer_state.json
5194	artifacts/e1/harness_validation/probe_tps_on/D_s0/final/adapter/README.md
69782384	artifacts/e1/harness_validation/probe_tps_on/D_s0/final/adapter/adapter_model.safetensors
{"step": 100}
NO torch.save in experiments/e1/train.py at 60eeade^
60eeade:experiments/e1/train.py:447:        torch.save(base.state_dict(), directory / "base.pt")
60eeade:experiments/e1/train.py:449:        torch.save(model.block.state_dict(), directory / "block.pt")
60eeade:experiments/e1/train.py:451:        torch.save(model.head.state_dict(), directory / "head.pt")
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  -
```

`git_diff_sha256` matches the SHA-256 of the empty input, and `git_dirty` is false. The probe log's first line is the override warning: `--tokens 2000000` for arm D against the arm-table budget 30000000. The same log records step 100/100 and `tokens=409600`. The checkpoint directory contains adapter weights (`adapter_model.safetensors`, 69782384 bytes) and `trainer_state.json`. There is no `block.pt` because this probe is arm D. Before commit `60eeade`, `experiments/e1/train.py` contained no `torch.save` call, so no run had a model attached and every pilot run failed the reproducibility rule.

What this proves: a real GPU probe wrote a manifest with the reproducibility fields, including determinism on, a clean git diff hash, `nan_skips` 0, peak VRAM 8.861928939819336 GiB, a final checkpoint path, and the six spawn keys, and that directory holds adapter weights plus `trainer_state.json`.

What this does not prove: that the checkpoint reloads to the same loss, or that a 30M-token run has been saved. This probe's recorded budget is the 2M override, and it saw 409600 tokens.

## 6. Washout diagnostic

`rg block_fraction artifacts/e1/harness_validation` found no matches. The GPU probe is arm D, which has no block, so it cannot record `block_rms`. A tiny arm A train was run instead. Step count is `ceil(1024 / (32 * 2 * 1)) = 16`.

```bash
.venv/bin/python -m experiments.e1.train \
  --arm A --seed 0 --tiny \
  --tokens 1024 --context 32 --batch-size 2 --grad-accum 1 \
  --eval-items 2 --device cpu --dtype float32 --deterministic \
  --log-every 1000 \
  --checkpoint-dir /tmp/e1-demo-evidence/a-wash-ckpts \
  --out /tmp/e1-demo-evidence/a-wash.json
```

```text
[A s0] step 16/16 lm=6.2286 jepa=1.7891 var=0.9700 cov=0.0628 std_z=0.0300 erank=21.31 val_nll=6.2739 val_cg=0.0191 tokens=1024
P1 context gain : +0.05584 nats/token
P3 necessity    : +0.01411 nats/token
block fraction  : 0.0903 of residual RMS
```

From the written eval object: `block_rms=0.0030598887764100276`, `residual_rms=0.033871740508493675`, `block_fraction=0.09033751234728343`, `has_block=True`. Exit code 0.

What this proves: after 16 steps, a clean eval forward records a non-zero block RMS and a block fraction of about 0.0903 of residual RMS. The diagnostic is no longer stuck at 0.

What this does not prove: anything about washout on Qwen3-1.7B. The previous diagnostic always recorded 0.0 because the ablated forward overwrote the clean forward's stats, so the one number that distinguishes "the block was washed out" from "the block was useless" was itself broken. This tiny run only shows that the overwrite is gone.

## 7. Corpus harvester resilience

```bash
date
.venv/bin/python scripts/harvest_supervisor.py --dry-run --at 11:40 --at 02:00 --at 23:29 --at 23:59
stat -c '%y  %s bytes  %n' artifacts/corpus/harvest_2026b.log
rg -n "Errno 101|aborted|host refused|Network is unreachable" artifacts/corpus/harvest_2026b.log
```

```text
Thu Nov 26 11:45:50 AM EST 2026
quiet 23:30-06:00 local is 6.5 h; allowed window is 17.5 h
would run: /home/jpearce/Documents/Lucid/.venv/bin/python -m experiments.corpus.harvest --server biorxiv --cutoff 2025-06-01 --out /home/jpearce/Documents/Lucid/artifacts/corpus/biorxiv_2026.jsonl --max-docs 1000000 --delay 3.0 --resume
2026-11-26T11:40-05:00 allowed; would start the harvester immediately; quiet begins at 23:30
2026-11-26T02:00-05:00 quiet; would sleep until 2026-11-26T06:00-05:00 and start nothing
2026-11-26T23:29-05:00 allowed; would start the harvester immediately; quiet begins at 23:30
2026-11-26T23:59-05:00 quiet; would sleep until 2026-11-27T06:00-05:00 and start nothing
2026-11-25 23:32:28.578433250 -0500  1159 bytes  artifacts/corpus/harvest_2026b.log
6:[harvest] host refused; stopping rather than retrying: HostRefused: host refused https://www.biorxiv.org/content/early/2025/07/01/2025.02.06.636914.source.xml (status=None): <urlopen error [Errno 101] Network is unreachable> | cause=URLError: <urlopen error [Errno 101] Network is unreachable>
17:    "error:TransientRequestError: transient failure after retries for https://www.biorxiv.org/content/early/2025/07/01/2025.03.11.642697.source.xml (status=None): <urlopen error [Errno 101] N": 1
21:  "aborted": "host_refused"
```

What this proves: on the local clock, 11:40 and 23:29 are inside the allowed window and 02:00 and 23:59 are inside the quiet window. A start at 02:00 would sleep until 06:00 and start nothing; a start at 23:59 would sleep until 06:00 the next day. The host's network is off 23:30–06:00 local every night, so a harvest that is still on the network at 23:30 dies. `harvest_2026b.log` was last written at 23:32:28 on 2026-11-25 and records an abort with `Errno 101 Network is unreachable`.

What this does not prove: that the supervisor process itself was running at 23:32 last night. The log is the harvest abort. The supervisor's job is to keep a later harvest out of that window.

## 8. Live corpus status

Read-only. The parser skips a line that is not valid JSON, which is how a torn final line is tolerated. This read had no such line. The file was not written.

```text
path=artifacts/corpus/biorxiv_2026.jsonl
bytes=38980662
mtime_ns=1795711506815564440
parsed_rows=891
blank_lines=0
distinct_doc_id=891
duplicate_extra_rows=0
unparsed_lines=0
status_counts
     576  licence_excluded
     237  ok
      69  abstract_only
       7  error:RuntimeError
       1  error:HTTPError: HTTP Error 404: Not Found
       1  error:TransientRequestError: transient failure after retries for https://www.biorxiv.org/content/early/2025/07/01/2025.03.11.642697.source.xml (status=None): <urlopen error [Errno 101] N
fulltext_ok=237
error_star_rows=9
ok_token_est_n=237
ok_token_est_median_chars_over_4=17076.0
ok_token_est_total_chars_over_4=4480896.0
e4s1_needs=192
e4s1_192_fulltext_ok_reached=True
```

The mtime is 2026-11-26 11:45:06 EST. Token estimates are `len(text) / 4` on rows whose `meta.fulltext` is `ok`, not a tokenizer count. A later `stat` of the same file, at 11:46:12, showed 39106318 bytes, so the harvest was still appending after this snapshot. Nine `error:*` rows are in the file (7 `error:RuntimeError`, one HTTP 404, one truncated `TransientRequestError`). Those nine must not enter a build. A gate fix for exactly this is in flight by another agent. The stored transient-error status string itself ends at `Errno 101] N`.

What this proves: at 11:45:06 the file held 891 distinct documents, 237 with full text `ok`, which is past the 192 full-text documents E4-S1 asks for, and 9 error rows.

What this does not prove: that those 237 documents survive the licence gate, the cutoff, deduplication, or a section-text requirement, or that a build has been cut from them. `licence_excluded` is a separate status (576 rows) and is not counted as full text ok.

## 9. Test totals

```bash
.venv/bin/python -m pytest -q
```

`pyproject.toml` already sets `addopts = "-q"`, so this invocation is quiet twice. Pytest then prints progress and warnings and does not print an `N passed in T` line. The run was started in a tmux session created at 11:45:33 EST. The output file's mtime is 2026-11-26 11:47:32.951 -0500, about 120 seconds later. That interval is the file's, not a figure pytest printed. Exit code 0. Progress lines contain 420 `.` marks and no `F`, `E`, `s`, or `x`.

```text
........................................................................ [ 17%]
........................................................................ [ 34%]
........................................................................ [ 51%]
........................................................................ [ 68%]
........................................................................ [ 85%]
............................................................             [100%]
=============================== warnings summary ===============================
tests/test_e1_train.py: 20 warnings
  /home/jpearce/Documents/Lucid/.venv/lib/python3.13/site-packages/peft/utils/save_and_load.py:372: UserWarning: Could not find a config file in  - will assume that the vocabulary was not modified.
    warnings.warn(

tests/test_rating_app.py::test_full_rating_flow
  /home/jpearce/Documents/Lucid/tests/test_rating_app.py:25: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
EXIT:0
```

The exact last line pytest printed is `-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html`. There is no count-and-time line in the capture.

Two later attempts to force the single-quiet summary (`pytest -q -o addopts=`) were terminated with signal 15 (exit 143) while the host load average was above 40. They are not test failures. A collect-only a few minutes later, after another agent had edited `tests/test_corpus_pipeline.py`, reported `423 tests collected`. The 420 figure is the suite that finished at 11:45, not that later tree.

What this proves: the suite that actually finished exited 0, with 420 passing progress marks and the two warnings above.

What this does not prove: a pytest-reported duration, or that the suite still passes on the files another agent was editing after 11:47.

## What remains unverified

- GPU operating-point repeatability. Context 512, bf16, Qwen3-1.7B-Base, and the confirmatory batch and accumulation settings were not repeated here. Section 2 is a 4-step CPU tiny float32 run at context 32. A separate GPU validation is in flight.
- The 30M-token arm. The arm table says 30000000. Nothing in this note trained that many tokens. The GPU probe's manifest records `token_budget` 2000000, and its log shows 409600 tokens seen.
- The corpus gate fix. Nine `error:*` rows are still in `artifacts/corpus/biorxiv_2026.jsonl`. Another agent is editing the gate. This note did not run a build and did not write that file.
- Cutoff provenance. Commit `2019328b928409fae335a51d9974c911718401f3` ("record the operative harvest cutoff beside the proxy") landed while this capture was in progress. This note did not check a model card, a paper, or a weight upload against that proxy.
