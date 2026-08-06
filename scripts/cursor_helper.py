#!/usr/bin/env python3
"""Reliable Cursor Agent orchestrator.

WHY: running `cursor-agent -p --mode plan ...` and scraping stdout/JSON/JSONL
transcripts is fragile — when Cursor uses its CreatePlan tool, the plan lands in
a tool call that never reaches stdout, so text and even json output modes miss
it. The deterministic fix: tell the agent to WRITE its deliverable to a file
(via its Write tool), then read that file back. This wrapper implements that
contract and handles the background run + file-verification.

Usage:
  python cursor_helper.py run --prompt-file /tmp/p.txt --out /tmp/plan.md \
      [--model claude-sonnet-5-thinking-medium] [--mode plan] [--workspace ~/proj] \
      [--append-instruction] [--timeout 900]

It appends an instruction telling the agent to save the final deliverable to
--out, launches cursor-agent (subprocess), waits for the file to appear with
real content, and prints the file path + the agent's exit status. Returns
non-zero if the file never materialized with a minimum size.

Also exposes `transcript <session-id>` to extract the last agent text from the
JSONL transcript as a fallback.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURSOR_AGENT = os.environ.get("CURSOR_AGENT", "cursor-agent")
TRANSCRIPT_DIR = os.path.expanduser(
    os.environ.get("CURSOR_TRANSCRIPTS",
                   "~/.cursor/projects/home-jpearce-Documents-Lucid/agent-transcripts"))


def file_instruction(out_path: str) -> str:
    return (
        "\n\nIMPORTANT OUTPUT CONTRACT: write your complete final deliverable to "
        f"the file `{out_path}` using your Write tool (create directories if needed). "
        "Do NOT rely on stdout or the final chat message for the deliverable — save it "
        "to that file. If the deliverable is a plan, write the full plan markdown there. "
        "After writing, reply with just: DONE <absolute-path>"
    )


def run(args):
    if os.path.exists(args.out):
        os.remove(args.out)  # clear stale target so we can detect a fresh write
    with open(args.prompt_file) as f:
        prompt = f.read().strip()
    prompt += file_instruction(args.out)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cmd = [CURSOR_AGENT, "-p", "--model", args.model, "--output-format", "json",
           "--trust", "--workspace", args.workspace]
    if args.mode in ("plan", "ask"):
        cmd += ["--mode", args.mode]
    cmd.append(prompt)
    print(f"[cursor-helper] launching: {' '.join(cmd[:6])}... (mode={args.mode} model={args.model})")
    fd = open(args.log, "w") if args.log else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=fd, stderr=subprocess.STDOUT, text=True)

    # Poll for the deliverable file + process exit.
    deadline = time.time() + args.timeout
    seen_exit = False
    while time.time() < deadline:
        if not seen_exit and proc.poll() is not None:
            seen_exit = True
            print(f"[cursor-helper] agent exited rc={proc.returncode}")
            # give it a moment to flush the file
            time.sleep(2)
        if os.path.exists(args.out):
            size = os.path.getsize(args.out)
            if size >= args.min_bytes:
                print(f"[cursor-helper] deliverable written: {args.out} ({size} bytes)")
                if args.log:
                    print(f"[cursor-helper] log: {args.log}")
                return 0
        time.sleep(3)

    # Timed out: if the agent is still alive, kill it.
    if proc.poll() is None:
        print("[cursor-helper] TIMEOUT - terminating agent")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    print("[cursor-helper] FAILED: deliverable not written within timeout")
    return 1


def transcript(args):
    path = os.path.join(TRANSCRIPT_DIR, args.session, f"{args.session}.jsonl")
    if not os.path.exists(path):
        print(f"no transcript at {path}")
        return 1
    texts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msg = obj.get("message")
            if not msg:
                continue
            content = msg.get("content") if isinstance(msg, dict) else msg
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
    if args.plan_only and texts:
        # print the last substantive text block (the final answer/plan)
        print(texts[-1])
    else:
        print("\n---\n".join(texts))
    return 0


def main():
    p = argparse.ArgumentParser(description="Reliable Cursor Agent orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--prompt-file", required=True)
    r.add_argument("--out", required=True, help="file Cursor must write the deliverable to")
    r.add_argument("--model", default="claude-sonnet-5-thinking-medium")
    r.add_argument("--mode", default="write", choices=["write", "plan", "ask"])
    r.add_argument("--workspace", default=ROOT)
    r.add_argument("--timeout", type=int, default=900)
    r.add_argument("--min-bytes", type=int, default=200)
    r.add_argument("--log", default=None)
    r.set_defaults(func=run)

    t = sub.add_parser("transcript")
    t.add_argument("--session", required=True)
    t.add_argument("--plan-only", action="store_true")
    t.set_defaults(func=transcript)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
