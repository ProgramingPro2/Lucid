# splicer: replace train() onward in train_stage_a.py
from pathlib import Path
import ast

path = Path("/home/jpearce/Documents/Lucid/experiments/pivot/train_stage_a.py")
text = path.read_text()
marker = "\ndef train(cfg: dict[str, Any], *, resume: bool = True, max_steps: int | None = None) -> dict[str, Any]:"
idx = text.find(marker)
if idx < 0:
    raise SystemExit("train marker not found")
head = text[:idx]
# Drop prior helper defs if we already inserted them before train
for name in ("def _trainable_params", "def _grads_finite"):
    j = head.rfind("\n" + name)
    if j >= 0 and j > head.rfind("\ndef eval_sample_log"):
        head = head[:j]

tail_path = Path("/home/jpearce/Documents/Lucid/experiments/pivot/_train_tail.py")
tail = tail_path.read_text()
path.write_text(head + "\n" + tail)
ast.parse(path.read_text())
print("ok", path.stat().st_size)
