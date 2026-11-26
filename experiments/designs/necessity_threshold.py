"""When does a side channel stop being decorative?

CPU toy. No pretrained model, no GPU. The task is small enough that a
one-layer transformer can solve either path; the only thing that varies is
whether the text path already contains the answer.

Each example has a topic. The target is 12 tokens drawn uniformly from that
topic's private 4-token set, so a model that knows the topic reaches
log(4) = 1.386 nats and a model that does not cannot. The text path, when
open, shows those 4 tokens in the context, and copying them solves the loss.
The side channel is a frozen random code for the topic, passed through a
zero-initialised linear adapter and then either prepended as one token
("prefix") or added at every position ("film"). Zero init makes the channel
invisible at step 0, which is the same starting point as E1's block.

Pre-registered predictions, fixed before the run:

H1  closed + CE, prefix. The text path is blank, so the channel is the only
    route. On a closed test, NLL(true) < 2.0, necessity
    NLL(zero) - NLL(true) > 1.0 nats, and shuffled-topic necessity > 1.0 nats.
    Both interventions have to hurt. They are not required to hurt by the
    same amount: a confident wrong topic scores worse than a zero vector,
    and that is content sensitivity, not a slot effect. An earlier wording
    required the two necessities to agree within 0.25 nats. A 150-step,
    one-seed smoke (not this run) scored shuffle worse than zero by several
    nats while NLL(true) had already reached the log(4) floor, so that
    wording rejects the mechanism it names. It was replaced with the two
    one-sided tests above before the three-seed run.

H2  open + CE, prefix. The four answer tokens are in the context. Necessity
    on an open test < 0.15 nats, and the same weights evaluated on a closed
    test stay within 0.5 nats of the no-channel closed floor. The channel
    was never needed, so it was never learned.

H3  open + JEPA, prefix. An auxiliary MSE pulls the adapter output onto the
    frozen code, so the write side is supervised whether or not the reader
    uses it. Adapter cosine to the code > 0.9, and necessity on the open
    test is still < 0.15. A loaded channel that the loss can route around
    stays decorative. This is the result that would make a JEPA term, by
    itself, the wrong tool for Q1.

H4  partial, q = 0.3, CE, prefix. Each of the four answer tokens is shown
    independently with probability 0.3, so P(at least one identifier) =
    1 - 0.7**4 = 0.760. Necessity on the open subset of that distribution
    (examples that still show at least one answer token) < 0.30 nats.
    Sprinkled dropout is not a bottleneck, because one surviving token
    still names the topic. The closed subset is reported separately and
    is not part of H4.

H5  closed + CE, film. Same bottleneck as H1, position-uniform add instead
    of a prefix token. This task is a position-independent unigram, which a
    uniform add can carry, so a film success does NOT speak to W3. The arm
    exists to check that the bottleneck result is not an artifact of the
    prefix. Recorded, not interpreted as a rendering result.

Assay, not a hypothesis. open + no channel must reach NLL < 2.0 on an open
test (the copy path is solvable), and closed + no channel must stay above
3.5 on a closed test (blanks do not name the topic). If either fails the
run is void and H1–H5 are not interpreted.

A discarded run. The first execution used a bidirectional encoder and put
the labels in the input, so NLL collapsed to 0 by reading the answer. That
run is not evidence. Target positions are now blank inputs, labels are
held out, and attention is causal.

A miss on H1 or H2 kills the design in the toy regime. A miss on H3 means
this reader uses an informative channel even when a copy path exists, so
the real model needs a harder gap than "one extra vector." A miss on H4
means 30% masking already forces the channel on examples that still have
a text path, and the literature's "partial dropout is not enough" warning
does not survive this task.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Refuse the GPU even if one is free. Other jobs are using it.
torch.cuda.is_available = lambda: False  # type: ignore[method-assign]

K = 16
M = 4
CTX = 8
TGT = 12
BLANK = K
N_CONTENT = K * M
VOCAB = K + 1 + N_CONTENT  # indicators, blank, content
CONTENT0 = K + 1
D = 64
HEADS = 4
LAYERS = 1
STEPS = 200
BATCH = 128
LR = 3e-3
SEEDS = (0, 1, 2)
EVAL_N = 1024
JEPA_WEIGHT = 1.0

OUT = Path(__file__).with_name("necessity_threshold.json")


def content_id(topic: np.ndarray, slot: np.ndarray) -> np.ndarray:
    return CONTENT0 + topic * M + slot


class CausalBlock(nn.Module):
    """One pre-norm transformer block. Causal SDPA, no dropout.

    ``nn.TransformerEncoder`` with a float mask was ~30x slower on this CPU
    and the ``is_causal`` flag is rejected unless a mask is also passed.
    SDPA's own causal kernel is the fast path.
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.d = d
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        b, t, _ = q.shape
        hd = self.d // self.heads

        def split(z: torch.Tensor) -> torch.Tensor:
            return z.view(b, t, self.heads, hd).transpose(1, 2)

        y = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d)
        x = x + self.out(y)
        return x + self.ff(self.ln2(x))


class Reader(nn.Module):
    def __init__(self, mode: str, seed: int):
        super().__init__()
        self.mode = mode
        g = torch.Generator()
        g.manual_seed(seed)
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(CTX + TGT + 1, D)
        nn.init.normal_(self.tok.weight, std=0.02, generator=g)
        nn.init.normal_(self.pos.weight, std=0.02, generator=g)
        code = torch.randn(K, D, generator=g)
        code = code / code.norm(dim=-1, keepdim=True)
        self.register_buffer("codebook", code)
        self.adapter = nn.Linear(D, D, bias=False)
        nn.init.zeros_(self.adapter.weight)
        self.blocks = nn.ModuleList(CausalBlock(D, HEADS) for _ in range(LAYERS))
        self.head = nn.Linear(D, VOCAB)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def channel_out(self, topic: torch.Tensor, intervention: str) -> torch.Tensor:
        if intervention == "shuffle":
            topic = (topic + 7) % K
        raw = self.adapter(self.codebook[topic])
        if intervention == "zero":
            raw = torch.zeros_like(raw)
        return raw

    def forward(self, tokens: torch.Tensor, topic: torch.Tensor, intervention: str) -> torch.Tensor:
        b, t = tokens.shape
        pos = torch.arange(1, t + 1, device=tokens.device)
        x = self.tok(tokens) + self.pos(pos)
        c = self.channel_out(topic, intervention)
        if self.mode == "prefix":
            prefix = (c + self.pos.weight[0]).unsqueeze(1)
            seq = torch.cat([prefix, x], dim=1)
        elif self.mode == "film":
            seq = x + c.unsqueeze(1)
        elif self.mode == "none":
            seq = x
        else:
            raise ValueError(self.mode)
        h = seq
        for block in self.blocks:
            h = block(h)
        if self.mode == "prefix":
            h = h[:, 1:]
        return self.head(h)


def make_batch(
    rng: np.random.Generator,
    n: int,
    policy: str,
    q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return tokens, labels, topics, and a closed-example mask.

    Target positions are blank in the input. The labels are held out, so a
    bidirectional read of the input cannot see the answer, and neither can
    a causal one. policy "bernoulli": each of the four answer tokens is
    shown in the context with probability q. "closed" and "open" are the
    q = 0 and q = 1 ends of that.
    """
    topic = rng.integers(0, K, size=n)
    slot = rng.integers(0, M, size=(n, TGT))
    labels = content_id(topic[:, None], slot)
    tokens = np.full((n, CTX + TGT), BLANK, dtype=np.int64)
    show = np.zeros((n, M), dtype=bool)
    if policy == "bernoulli":
        show = rng.random((n, M)) < q
    elif policy == "open":
        show[:] = True
    elif policy == "closed":
        pass
    else:
        raise ValueError(policy)
    for s in range(M):
        ids = content_id(topic, np.full(n, s))
        tokens[:, s] = np.where(show[:, s], ids, BLANK)
    closed = ~show.any(axis=1)
    return tokens, labels, topic, closed


def nll_of(
    model: Reader,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    topic: torch.Tensor,
    intervention: str,
) -> float:
    logits = model(tokens, topic, intervention)
    pred = logits[:, CTX - 1 : CTX - 1 + TGT, :]
    return float(F.cross_entropy(pred.reshape(-1, VOCAB), labels.reshape(-1)).item())


@torch.no_grad()
def evaluate(model: Reader, seed: int, policy: str, q: float) -> dict[str, float]:
    rng = np.random.default_rng(10_000 + seed)
    model.eval()
    acc = {"true": 0.0, "zero": 0.0, "shuffle": 0.0}
    seen = 0
    while seen < EVAL_N:
        n = min(BATCH, EVAL_N - seen)
        tokens, labels, topic, _closed = make_batch(rng, n, policy, q)
        tt = torch.from_numpy(tokens)
        lab = torch.from_numpy(labels)
        tp = torch.from_numpy(topic.copy())
        for name in acc:
            acc[name] += nll_of(model, tt, lab, tp, name) * n
        seen += n
    out = {k: v / seen for k, v in acc.items()}
    out["necessity_zero"] = out["zero"] - out["true"]
    out["necessity_shuffle"] = out["shuffle"] - out["true"]
    # Mixed policies: necessity on the fully-closed and fully-open subsets.
    if policy == "bernoulli":
        rng2 = np.random.default_rng(20_000 + seed)
        buckets = {True: {"true": 0.0, "zero": 0.0, "n": 0}, False: {"true": 0.0, "zero": 0.0, "n": 0}}
        drawn = 0
        while drawn < EVAL_N:
            n = BATCH
            tokens, labels, topic, closed = make_batch(rng2, n, policy, q)
            tt = torch.from_numpy(tokens)
            lab = torch.from_numpy(labels)
            tp = torch.from_numpy(topic.copy())
            log_true = _per_example_nll(model, tt, lab, tp, "true")
            log_zero = _per_example_nll(model, tt, lab, tp, "zero")
            for flag, lt, lz in zip(closed, log_true, log_zero):
                b = buckets[bool(flag)]
                b["true"] += float(lt)
                b["zero"] += float(lz)
                b["n"] += 1
            drawn += n
        for flag, key in ((True, "closed_subset"), (False, "open_subset")):
            b = buckets[flag]
            if b["n"]:
                out[f"{key}_nll_true"] = b["true"] / b["n"]
                out[f"{key}_necessity"] = (b["zero"] - b["true"]) / b["n"]
                out[f"{key}_n"] = b["n"]
    return out


def _per_example_nll(
    model: Reader,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    topic: torch.Tensor,
    intervention: str,
) -> np.ndarray:
    logits = model(tokens, topic, intervention)
    pred = logits[:, CTX - 1 : CTX - 1 + TGT, :]
    per = F.cross_entropy(pred.reshape(-1, VOCAB), labels.reshape(-1), reduction="none")
    return per.reshape(tokens.shape[0], TGT).mean(dim=1).numpy()


def adapter_stats(model: Reader) -> dict[str, float]:
    with torch.no_grad():
        w = model.adapter.weight
        pred = model.adapter(model.codebook)
        tgt = model.codebook
        cos = F.cosine_similarity(pred, tgt, dim=-1).mean()
        return {
            "adapter_frob": float(w.norm().item()),
            "adapter_cosine": float(cos.item()),
        }


def train_one(seed: int, mode: str, policy: str, q: float, jepa: bool) -> dict:
    torch.manual_seed(seed)
    model = Reader(mode if mode != "none" else "none", seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    model.train()
    t0 = time.perf_counter()
    last_ce = None
    last_jepa = 0.0
    for step in range(STEPS):
        tokens, labels, topic, _ = make_batch(rng, BATCH, policy, q)
        tt = torch.from_numpy(tokens)
        lab = torch.from_numpy(labels)
        tp = torch.from_numpy(topic.copy())
        if mode == "none":
            logits = model(tt, tp, "zero")
            jepa_loss = tt.new_zeros(())
        else:
            logits = model(tt, tp, "true")
            if jepa:
                pred_c = model.adapter(model.codebook[tp])
                jepa_loss = F.mse_loss(pred_c, model.codebook[tp].detach())
            else:
                jepa_loss = tt.new_zeros(())
        pred = logits[:, CTX - 1 : CTX - 1 + TGT, :]
        ce = F.cross_entropy(pred.reshape(-1, VOCAB), lab.reshape(-1))
        loss = ce + (JEPA_WEIGHT * jepa_loss if jepa else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_ce = float(ce.item())
        last_jepa = float(jepa_loss.item()) if jepa else 0.0
    seconds = time.perf_counter() - t0
    model.eval()
    row = {
        "seed": seed,
        "mode": mode,
        "policy": policy,
        "q": q,
        "jepa": jepa,
        "steps": STEPS,
        "seconds": round(seconds, 2),
        "final_ce": last_ce,
        "final_jepa_mse": last_jepa,
        **adapter_stats(model),
        "on_open": evaluate(model, seed, "open", 1.0),
        "on_closed": evaluate(model, seed, "closed", 0.0),
    }
    if policy == "bernoulli":
        row["on_partial"] = evaluate(model, seed, "bernoulli", q)
    return row


def mean_std(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if len(a) > 1 else 0.0), "min": float(a.min()), "max": float(a.max())}


def summarise(rows: list[dict]) -> dict:
    def grab(path: tuple[str, ...]) -> dict[str, float]:
        vals = []
        for r in rows:
            cur = r
            for p in path:
                cur = cur[p]
            vals.append(float(cur))
        return mean_std(vals)

    keys = {
        "final_ce": ("final_ce",),
        "adapter_frob": ("adapter_frob",),
        "adapter_cosine": ("adapter_cosine",),
        "open_nll": ("on_open", "true"),
        "open_necessity_zero": ("on_open", "necessity_zero"),
        "open_necessity_shuffle": ("on_open", "necessity_shuffle"),
        "closed_nll": ("on_closed", "true"),
        "closed_necessity_zero": ("on_closed", "necessity_zero"),
        "closed_necessity_shuffle": ("on_closed", "necessity_shuffle"),
        "closed_nll_zeroed": ("on_closed", "zero"),
    }
    out = {name: grab(path) for name, path in keys.items()}
    if rows and rows[0]["policy"] == "bernoulli":
        out["partial_necessity_zero"] = grab(("on_partial", "necessity_zero"))
        out["partial_nll"] = grab(("on_partial", "true"))
        out["partial_closed_subset_necessity"] = grab(("on_partial", "closed_subset_necessity"))
        out["partial_open_subset_necessity"] = grab(("on_partial", "open_subset_necessity"))
    return out


ARMS = [
    {"name": "closed_ce_prefix", "mode": "prefix", "policy": "closed", "q": 0.0, "jepa": False},
    {"name": "open_ce_prefix", "mode": "prefix", "policy": "open", "q": 1.0, "jepa": False},
    {"name": "open_jepa_prefix", "mode": "prefix", "policy": "open", "q": 1.0, "jepa": True},
    {"name": "closed_jepa_prefix", "mode": "prefix", "policy": "closed", "q": 0.0, "jepa": True},
    {"name": "partial03_ce_prefix", "mode": "prefix", "policy": "bernoulli", "q": 0.3, "jepa": False},
    {"name": "closed_ce_film", "mode": "film", "policy": "closed", "q": 0.0, "jepa": False},
    {"name": "open_ce_film", "mode": "film", "policy": "open", "q": 1.0, "jepa": False},
    {"name": "closed_none", "mode": "none", "policy": "closed", "q": 0.0, "jepa": False},
    {"name": "open_none", "mode": "none", "policy": "open", "q": 1.0, "jepa": False},
]


def judge(summary: dict[str, dict]) -> dict[str, dict]:
    """Apply H1–H5 to the summarised arms. A miss is a result."""
    c = summary["closed_ce_prefix"]
    o = summary["open_ce_prefix"]
    j = summary["open_jepa_prefix"]
    p = summary["partial03_ce_prefix"]
    f = summary["closed_ce_film"]
    floor = summary["closed_none"]["closed_nll"]["mean"]
    h1 = (
        c["closed_nll"]["mean"] < 2.0
        and c["closed_necessity_zero"]["mean"] > 1.0
        and c["closed_necessity_shuffle"]["mean"] > 1.0
    )
    h2 = (
        o["open_necessity_zero"]["mean"] < 0.15
        and abs(o["closed_nll"]["mean"] - floor) < 0.5
    )
    h3 = j["adapter_cosine"]["mean"] > 0.9 and j["open_necessity_zero"]["mean"] < 0.15
    h4 = p["partial_open_subset_necessity"]["mean"] < 0.30
    assay_ok = summary["open_none"]["open_nll"]["mean"] < 2.0 and floor > 3.5
    return {
        "assay_ok": bool(assay_ok),
        "H1_closed_ce_necessary": {"pass": bool(h1) if assay_ok else None, "closed_nll": c["closed_nll"]["mean"], "necessity_zero": c["closed_necessity_zero"]["mean"], "necessity_shuffle": c["closed_necessity_shuffle"]["mean"]},
        "H2_open_ce_decorative": {"pass": bool(h2) if assay_ok else None, "open_necessity": o["open_necessity_zero"]["mean"], "closed_nll": o["closed_nll"]["mean"], "floor": floor},
        "H3_jepa_write_without_read": {"pass": bool(h3) if assay_ok else None, "adapter_cosine": j["adapter_cosine"]["mean"], "open_necessity": j["open_necessity_zero"]["mean"]},
        "H4_partial_dropout_not_enough": {"pass": bool(h4) if assay_ok else None, "open_subset_necessity": p["partial_open_subset_necessity"]["mean"], "closed_subset_necessity": p["partial_closed_subset_necessity"]["mean"]},
        "H5_film_closed_recorded": {"closed_nll": f["closed_nll"]["mean"], "necessity_zero": f["closed_necessity_zero"]["mean"]},
        "oracle_nats": math.log(M),
    }


def main() -> None:
    torch.set_num_threads(4)
    rows: list[dict] = []
    done: set[tuple[str, int]] = set()
    if OUT.exists():
        prior = json.loads(OUT.read_text())
        # A finished payload is not resumed; a partial one is.
        if prior.get("partial") and "verdict" not in prior:
            rows = list(prior.get("rows", []))
            done = {(r["arm"], int(r["seed"])) for r in rows}
            print(f"resuming, {len(done)} rows already on disk", flush=True)
    for arm in ARMS:
        arm_rows = []
        for seed in SEEDS:
            if (arm["name"], seed) in done:
                continue
            row = train_one(seed, arm["mode"], arm["policy"], arm["q"], arm["jepa"])
            row["arm"] = arm["name"]
            arm_rows.append(row)
            rows.append(row)
            print(
                f"{arm['name']} seed {seed}: ce {row['final_ce']:.3f} "
                f"open_nec {row['on_open']['necessity_zero']:+.3f} "
                f"closed_nll {row['on_closed']['true']:.3f} "
                f"closed_nec {row['on_closed']['necessity_zero']:+.3f} "
                f"closed_shuf {row['on_closed']['necessity_shuffle']:+.3f} "
                f"cos {row['adapter_cosine']:.3f} frob {row['adapter_frob']:.3f} "
                f"({row['seconds']:.1f}s)",
                flush=True,
            )
            # Write after every seed. A killed run must not take the rows with it.
            OUT.write_text(json.dumps({"partial": True, "rows": rows}, indent=2))
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    summary = {name: summarise(rs) for name, rs in by_arm.items()}
    verdict = judge(summary)
    payload = {
        "predictions": "H1-H5 in the module docstring, fixed before this run",
        "config": {
            "K": K, "M": M, "steps": STEPS, "batch": BATCH, "d": D,
            "layers": LAYERS, "seeds": list(SEEDS), "jepa_weight": JEPA_WEIGHT,
            "oracle_nats": math.log(M),
        },
        "summary": summary,
        "verdict": verdict,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
