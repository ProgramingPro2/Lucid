# Stop point, 2026-11-26

Handoff written after every Lucid harvest, train, and test process found still running had been stopped. Nothing was restarted. The harvest timer was not re-enabled. Nothing was pushed.

Clock is America/New_York. The snapshot commits below are local only.

## What was stopped

Before this pass, the orchestrator had already SIGTERM'd the bioRxiv harvest (pid 503191), its supervisor (503190), and an E1 training job (672593). Those were not running when this pass started. The harvest supervisor's tmux pane shows the child exiting `rc=-15` at 12:03:22 and the supervisor then exiting on its own.

During this pass, other agents were still launching jobs into the same tree. Each of these was stopped:

| What | Signal | Result |
|---|---|---|
| `.venv/bin/python -m pytest` in this repo (several, including a full suite and `tests/test_gpu_lock.py`) | SIGTERM | exited |
| `experiments/designs/necessity_threshold.py` (CPU assay, relaunched after the first kill) | SIGTERM | exited |
| `experiments.e1.train` arm D writing `D_seed0_run2.json` (twice; the second via `/tmp/run_e1_d2.sh`) | SIGTERM | exit 143, during weight load, before step 1 |
| tmux `e1-vram` sampler (`while true; nvidia-smi; sleep 10`) | SIGTERM ignored by the interactive shell; SIGINT via the pane | loop stopped, shell left at a prompt |
| `/tmp/run_e1_siginfo.py`, which is arm D again and installs a handler that logs SIGTERM and keeps training. It also ignores SIGINT and SIGQUIT (`SigIgn` included those bits) | SIGTERM, then SIGINT, then SIGHUP | process gone after SIGHUP. SIGKILL was not used |

Final check after the snapshot commits: no `/home/jpearce/Documents/Lucid/.venv/bin/python` process was running.

Idle tmux sessions were left in place. They are not holding a job (pane command is `bash`): `lucid-harvest-supervisor`, `e1-probe`, `e1-pytest`, `e1-pytest-summary`, `e1-tps-on`, `e1-vram`, `e1-d-run2`, `collapse-diag`. Unrelated sessions (TalonTodo, design runs) were not touched.

## Verification commands

Processes (after the kills; a mid-pass scan still showed the siginfo trainer and a pytest, which were then stopped):

```
ps -eo pid,etime,cmd | grep '/home/jpearce/Documents/Lucid/.venv/bin/python'
```

Empty at the end of this pass.

```
systemctl --user is-enabled lucid-harvest.timer
not-found
systemctl --user is-active lucid-harvest.timer
failed
systemctl --user is-active lucid-harvest.service
inactive
```

`systemctl --user status lucid-harvest.timer` reports `Loaded: not-found` and, at 12:03:09, `lucid-harvest.timer: Unit to trigger vanished` / `Failed with result 'resources'`. The unit file is not linked. `~/.config/systemd/user/lucid-harvest.service` is still a symlink to `scripts/systemd/lucid-harvest.service`. The service is inactive. `list-timers` shows `lucid-harvest.timer` with no next elapse.

The unstaged edit to `scripts/systemd/lucid-harvest.service` is three comment lines above `ExecStart`. It does not change the command and it does not arm the timer.

GPU compute apps at the start of the pass were not Lucid training: ComfyUI, VibeVoice, Sunshine, EA Desktop, and lemonade `llama-server`. They were left alone.

## Corpus

Re-counted locally, not from the orchestrator's note:

| Check | Value |
|---|---|
| File | `artifacts/corpus/biorxiv_2026.jsonl` |
| Bytes | 61,429,531 |
| Rows | 1378 |
| Malformed JSON lines | 0 |
| Trailing newline | yes |
| Last `doc_id` | `biorxiv:10.1101/2025.05.11.653310` |
| `artifacts/corpus/biorxiv_2026.jsonl.cursors.json` | `{"biorxiv\|2025-07-01\|2025-07-07": 1265, "biorxiv\|2025-07-08\|2025-07-14": 30}` |

`git stash list` is empty.

## How to re-arm the harvest later

Do not do this as part of resuming code work. The timer is what starts the supervisor at 06:05 America/New_York.

```
systemctl --user link /home/jpearce/Documents/Lucid/scripts/systemd/lucid-harvest.timer
systemctl --user daemon-reload
systemctl --user enable --now lucid-harvest.timer
```

The service unit is already linked. `Persistent=false`, so a missed 06:05 slot does not fire late.

## Branches

`main` is `b98cc32`, on origin. `backup/pre-lfs-migration` (`8d7dbae`) is not on origin and has no commits `main` lacks.

Published, and containing commits `main` does not:

| Branch | Tip | On origin | Commits `main` lacks |
|---|---|---|---|
| `cursor/corpus-gate-defects-52f3` | `2019328` | yes, same tip | 9, from `60eeade` through `2019328` |
| `cursor/corpus-harvest-resume-364f` | `b5d5d90` | yes | 1 (`b5d5d90`) |
| `origin/cursor/e1-harness-fixes-6f2d` | `2019328` | yes | same 9 as corpus-gate |
| local `cursor/e1-harness-fixes-6f2d` | `bdd612a` | yes, but **behind** origin by the two commits `77d8081` and `2019328` | those commits are already on origin |

Local branch names that are not on origin. Several were created during this stop and point at a commit that is already on origin, so the name is new and the commits are not:

| Branch | Tip | Unique? |
|---|---|---|
| `cursor/e4s1-interface-harness-3081` | `2019328` | no unique commits; the tip is corpus-gate |
| `cursor/shared-trunk` | `d8ef861` | **yes.** `2019328` plus snapshot `d8ef861` (below). Not pushed |
| `cursor/analysis-freeze-d274` | snapshot commit that adds this file | **yes.** Parent is `35112d3`, which is not on origin |
| `cursor/corpus-drop-audit` | `35112d3` | the one commit `35112d3` is not on origin |
| `cursor/e2-latent-emission` | `35112d3` | same commit, not on origin |
| `cursor/g0-assay-prep-8bba` | `35112d3` | same |
| `cursor/g5-olmo-harness` | `35112d3` | same. Worktree `/tmp/lucid-g5-olmo` is clean: no OLMo code was written |
| `cursor/split-and-hash-2764` | `35112d3` | same |
| `cursor/integration-guard` | `147fa4c` | **yes.** Cut from `main` (`b98cc32`), plus snapshot `147fa4c`. Not pushed |

`35112d3` (`docs: record captured E1 harness evidence for 2026-11-26`) adds only `docs/runs/2026-11-26-demo-evidence.md` (456 lines). It is corpus-gate plus that file. It is the parent of the main snapshot commit. It is not on origin.

Worktrees:

| Path | HEAD |
|---|---|
| `/home/jpearce/Documents/Lucid` | `cursor/analysis-freeze-d274` |
| `/tmp/lucid-e1-run` | detached `bdd612a`, clean |
| `/tmp/lucid-g5-olmo` | `cursor/g5-olmo-harness`, clean |
| `/tmp/lucid-integration-guard` | `cursor/integration-guard` at `147fa4c` |
| `/tmp/lucid-shared-trunk` | `cursor/shared-trunk` at `d8ef861`, with `experiments/e1/train.py` still modified and unstaged (+210/−4) |

## Snapshot commits (not pushed)

1. `147fa4c` on `cursor/integration-guard`. `scripts/check_stop_points.py`, `scripts/integrate_branches.sh`, `tests/test_stop_points.py`, `tests/test_integrate_branches.py`.
2. `d8ef861` on `cursor/shared-trunk`. `experiments/e1/trunk.py`, `tests/test_e1_trunk.py`. The `experiments/e1/train.py` edit in that worktree was not staged.
3. The commit that adds this file, on `cursor/analysis-freeze-d274`. It contains the untracked modules and notes that were in the main worktree. It does not contain edits to files that were already tracked. `git show --stat` of that commit is the exact file list. Agents were still creating files while this note was being written, so a path missing from the prose below may still be in that stat.

## Dangling objects

`git fsck --lost-found --no-reflogs` reported 3 dangling commits, 10 dangling blobs, and 3 dangling trees. `git stash list` is empty; one dangling commit is an old stash that was dropped.

| Commit | Date | What it is | Recovery |
|---|---|---|---|
| `d500f1639bf93be41ec3339f809851a6ae01096f` | 2026-08-05 | Stash "WIP on main" touching `src/conceptpoc/predictor/train_delta.py`: import `asdict`, and use it for `predictor_cfg`. That change is already in the current file | `git branch recovered-d500f16 d500f1639bf93be41ec3339f809851a6ae01096f` |
| `36a120b0eb70806cd6330f331aaa0c0d94872caa` | 2026-11-18 | Pre-rebase "experiments: E0 rehearsal…". Not an ancestor of `main`. Source diff against the rebased `8d7dbae` (which is on `main`) is `.gitattributes`, `LICENSE`, and a 4-line `README.md` change | `git branch recovered-36a120b 36a120b0eb70806cd6330f331aaa0c0d94872caa` |
| `d45e77d76011ee461eb2802a23795b9be1dec781` | 2026-11-18 | Rebase-finish of that same experiments commit, before LFS pointer conversion. Diff against `8d7dbae` is `.gitattributes` plus two parquet files becoming LFS pointers | `git branch recovered-d45e77d d45e77d76011ee461eb2802a23795b9be1dec781` |

The 10 dangling blobs are old markdown drafts (FINDINGS, README, PLAN, EXPERIMENTS, PROTOCOL, CORPUS, LIMITATIONS, IDEAS). None match a file in the current tree. They are not today's modules. Show one with `git cat-file -p <sha>`. SHAs: `0199040fde058d3147ee82a91112529be34bc855`, `e74cf23263c64b6bbf2528c31bcf984c40b82688`, `13c392e0e329cea83e76f6ba13353462ca81ce9b`, `1f74587ad7dc4712e0e9cc6a7a28582b49a66d1c`, `2ead982d4e299ffb8b0320f75cfb9b3d640d4c4d`, `5736e9e60dee4ce465c61a4d9d731834ffe8db4b`, `52ce3b00a3d16b4ff7189bcb8579b195cc0d669d`, `5ddb7972ffc8921489c301aff76a715d1f7e3259`, `7d546af0beabd71de78fd2fa920cb0eceefaf1a9`, `a49c7b74e603020f51b7ee812f8e7f039629b2d3`.

## Determinism repeat

Run 1 is a valid, finished result. Run 2 is an incomplete kill, not a corrupt result file.

Run 1:

- `artifacts/e1/harness_validation/D_seed0_run1.json` parses. 40,407 bytes. Arm D, seed 0. `train.steps` 489, `tokens_seen` 2,002,944, `history` length 20, last step 489, `nan_skips` 0, `wall_clock_s` 981.881670675939.
- `D_seed0_run1.manifest.json` parses. `git_commit` `bdd612a475743b853739d77d1593799f178b9589`, `git_dirty` false, `nan_skips` 0, `peak_vram_gb` 8.861928939819336.
- Checkpoints `artifacts/e1/harness_validation/repeat/D_s0/final` and `repeat/D_s0_run1/final`. Both `trainer_state.json` files are `{"step": 489}`. Both `adapter_model.safetensors` files are 69,782,384 bytes, mtime 2026-11-26 12:01:04, SHA-256 `baba5ff93b16cc58d62031b48ee23c1965f4b74d4a1521586c37387eaa830402`.

Run 2:

- `D_seed0_run2.json` does not exist. No manifest.
- `D_seed0_run2.log` starts 12:08:44, loads weights, loads the corpus cache, then `EXIT:143` at 12:09:50 (SIGTERM during this pass). `D_seed0_run2.killed.log` is an earlier attempt that also died during weight load and has no `EXIT` line.
- The run-1 adapter files were not rewritten (mtime still 12:01:04, hashes still match). The repeat directory still has only `D_s0` and `D_s0_run1`.
- Discard the run-2 logs. They are not a second sample. A future run 2 needs a new `--out` and a checkpoint directory that is not `repeat/`, or it will overwrite `repeat/D_s0/final` if it is allowed to finish. Launch it from `/tmp/lucid-e1-run` at `bdd612a` so the code matches the run-1 manifest. The comparison procedure is in `docs/runs/2026-11-26-determinism-forensics.md`.

## Untracked source that was still being written

These were untracked in the main worktree and are what the main snapshot commit is for. "Complete" here means the file ends on a real last line (`main`, `return`, or a finished function) and was not a stub. It does not mean the experiment has been run.

| Path | State |
|---|---|
| `experiments/e4/arms.py`, `data.py`, `drift.py`, `fit.py`, `generate.py`, `evaluate.py` | Partial harness, modules present and closed. No E4-S1 run artifact. This is the E4-S1 code that landed |
| `experiments/e2/latent.py`, `codebook.py`, `gate.py`, `arms.py`, `flops.py`, `analysis.py`, `chat.py` | Partial. Latent-emission arm is on disk as these modules. No E2 run artifact from today |
| `experiments/e0/assay.py`, `collect.py`, `criteria.py`, `dose.py`, `items.py`, `synthetic.py` | Partial G0 harness. `assay.py` defaults to a CPU dry run and refuses `--execute` when the corpus is too small. Not a finished G0 |
| `experiments/designs/necessity_threshold.py` | Complete CPU toy (H1–H5, `main`). It was executed and SIGTERM'd. No `necessity_threshold.json` result was on disk |
| `experiments/corpus/split.py`, `tests/test_corpus_split.py` | Module present, ends in `main`. This is the split/hash deliverable |
| `scripts/gpu_lock.py`, `tests/test_gpu_lock.py` | Module present, ends in `main`. A pytest of it was in progress and was stopped; pass/fail was not recorded |
| `docs/runs/2026-11-26-determinism-forensics.md` | Complete note. Run 1 forensics and the run-2 plan |

## Tracked files with unstaged edits (not in any snapshot commit)

Main worktree, against `35112d3`. Left unstaged on purpose. `git diff` is the text:

- `experiments/corpus/nll.py` — large expansion. The docstring now describes the memorisation screen. This is the contamination-screen work, written into the existing scorer rather than a new `contamination.py`.
- `experiments/corpus/pipeline.py`
- `experiments/corpus/build.py`
- `experiments/corpus/harvest.py`
- `experiments/e0/__init__.py`
- `scripts/harvest_supervisor.py`
- `scripts/systemd/lucid-harvest.service` — comments only, see above
- `tests/test_corpus_pipeline.py`

`/tmp/lucid-shared-trunk`, against `d8ef861`:

- `experiments/e1/train.py` — +210/−4, unstaged. Do not check out another branch in that worktree until this diff is committed or copied. A checkout that overwrites it drops the edit.

## Deliverables

| Item | On disk |
|---|---|
| E4-S1 harness | Partial. Modules listed above. No run |
| E2 latent arm | Partial. `experiments/e2/latent.py` and the siblings. No run |
| G0 assay | Partial. `experiments/e0/assay.py` and siblings. `experiments/e0/rehearsal.py` is the old rehearsal and says it is not G0 |
| G5 OLMo harness | Absent. Branch `cursor/g5-olmo-harness` and a clean worktree exist. No module |
| Contamination screen | Partial, and only as the unstaged edit to `experiments/corpus/nll.py`. No `contamination.py` |
| Split / hash | Present. `experiments/corpus/split.py` plus `tests/test_corpus_split.py`, in the main snapshot |
| Nightly corpus audit | Absent. No `nightly_corpus_audit.py` |
| GPU lock | Present. `scripts/gpu_lock.py` plus `tests/test_gpu_lock.py`, in the main snapshot. Test run was killed |
| Integration script | Present, in `147fa4c` on `cursor/integration-guard`: `scripts/integrate_branches.sh` and `tests/test_integrate_branches.py` |
| Analysis / verdict | Partial. `experiments/e4/evaluate.py` (G3) and `experiments/e2/analysis.py`, `experiments/e0/criteria.py`. No separate tool, no run |
| Collapse-metric diagnosis | Unknown. tmux session `collapse-diag` is an idle shell. No new module was found |
| Drop-path audit | Unknown as code. Branch `cursor/corpus-drop-audit` exists and points at `35112d3` (the demo-evidence doc only). No drop-path module was in the untracked set |
| Gate red-team | Outside the repo only: `/tmp/gate_probe.py` (16,525 bytes, 12:13). Docstring: "Adversarial probe. Prints observed gate outputs. Not a test." Not committed |
| FDA register | Specified in `docs/CORPUS.md`. Fetch helper is the existing `experiments/corpus/sources.py` openFDA function. No sealed FDA artifact was found |
| Demo evidence | Committed as `docs/runs/2026-11-26-demo-evidence.md` in `35112d3`. Raw small-run files are under `/tmp/e1-demo-evidence/` (not in git) |
| Design doc | No new design doc. `docs/PROTOCOL.md` and `docs/EXPERIMENTS.md` are the design. Module docstrings carry the new detail |
| Stop-point checker | Present, in `147fa4c`: `scripts/check_stop_points.py`, `tests/test_stop_points.py` |
| Shared trunk | Present, in `d8ef861`: `experiments/e1/trunk.py`, `tests/test_e1_trunk.py`. The `train.py` hookup is the unstaged diff in that worktree |

## Outside git, still on this machine

These are not in a commit. `/tmp` is cleared on reboot.

- `/tmp/gate_probe.py` (16,525 bytes)
- `/tmp/lucid-corpus-gate-check.py` (2,519 bytes)
- `/tmp/run_e1_d2.sh` (814 bytes) and `/tmp/run_e1_siginfo.py` (3,190 bytes) — launchers for the killed run 2, not results
- `/tmp/e1-demo-evidence/` — small CPU/GPU probe outputs. The write-up of them is the committed demo-evidence note
- `/tmp/e1_det_*.json` and the matching `.pt` files from earlier in the day (about 12:58–13:00 UTC-5, ~115 KB json / 132 KB pt)

## What is not recoverable from git

- Edits that never reached disk. Several agents were mid-task and had not saved when they were stopped. There is no object for those.
- The unstaged diffs listed above, if a later checkout or `git restore` overwrites them. They are only in the two worktrees.
- `/tmp/gate_probe.py` if `/tmp` is cleared before someone copies it in.
- A finished determinism run 2. It was never produced. Run 1 does not become a pair by itself.

## First step when resuming

Read this file, leave `lucid-harvest.timer` disabled, and do not treat determinism run 2 as data. When GPU work actually resumes, the first job is a fresh arm-D run 2 from `/tmp/lucid-e1-run` at `bdd612a`, with a new output path and a checkpoint directory that is not `artifacts/e1/harness_validation/repeat`.
