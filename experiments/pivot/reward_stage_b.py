"""Stage-B hybrid reward: JEPA tracking + LM judge coherence + hard collapse gate.

R(x|z_cmd) = gate(x) * (w_z * f(cos(E(x_hat), z_cmd)) + w_j * J_coh(x))
Fable Stage-B spec §1 / §3. New code only; does not modify src/.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from conceptpoc.codec.jepa_codec import JepaCodec  # noqa: E402

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+")
_WS = re.compile(r"\s+")

COHERENCE_SYSTEM = (
    "You are a strict writing evaluator. You judge one paragraph at a time.\n"
    "Ignore the paragraph's topic and length. Judge ONLY: grammar, fluency,\n"
    "sentence-to-sentence flow, and whether it reads as sane, natural prose a\n"
    "competent human could have written. Long does not mean good. Lists,\n"
    "headers, or fragments are not paragraphs. Answer with a single digit."
)

COHERENCE_USER = (
    "PARAGRAPH:\n{generated_paragraph}\n\n"
    "Score the paragraph:\n"
    "1 = broken (ungrammatical, incoherent, or degenerate/repetitive)\n"
    "2 = poor (readable but disjointed or unnatural)\n"
    "3 = acceptable (minor awkwardness)\n"
    "4 = good (fluent, coherent)\n"
    "5 = excellent (indistinguishable from a well-edited human paragraph)\n"
    "Answer with one digit only."
)

TOPIC_SYSTEM = "You are a strict topical evaluator. Answer with a single digit."
TOPIC_USER = (
    "TARGET TOPIC (keywords): {keywords}\n\n"
    "PARAGRAPH:\n{generated_paragraph}\n\n"
    "Did the paragraph stay on the target topic?\n"
    "1 = unrelated  2 = tangential  3 = partially on topic\n"
    "4 = clearly on topic  5 = fully and specifically on topic\n"
    "Being long or well-written does NOT count as on-topic.\n"
    "Answer with one digit only."
)


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (float(np.linalg.norm(x)) + eps)
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return (x / n).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_l2(a), _l2(b)))


def f_ramp(c: float, floor: float = 0.35, ceiling: float = 0.62) -> float:
    """Saturating piecewise-linear ramp of cosine into [0, 1]."""
    if ceiling <= floor:
        raise ValueError("ceiling must be > floor")
    return float(np.clip((float(c) - floor) / (ceiling - floor), 0.0, 1.0))


def tokenize_words(text: str) -> list[str]:
    return [t for t in _WS.split((text or "").strip().lower()) if t]


@dataclass
class GateFlags:
    short: bool = False
    distinct2: bool = False
    fourgram: bool = False
    loop: bool = False
    non_ascii: bool = False
    list_fmt: bool = False

    @property
    def tripped(self) -> bool:
        return any(
            (self.short, self.distinct2, self.fourgram, self.loop, self.non_ascii, self.list_fmt)
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "short": self.short,
            "distinct2": self.distinct2,
            "fourgram": self.fourgram,
            "loop": self.loop,
            "non_ascii": self.non_ascii,
            "list_fmt": self.list_fmt,
        }


def distinct_n(tokens: Sequence[str], n: int = 2) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return float(len(set(grams)) / max(1, len(grams)))


def max_ngram_count(tokens: Sequence[str], n: int = 4) -> int:
    if len(tokens) < n:
        return 0
    counts: dict[tuple[str, ...], int] = {}
    best = 0
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i : i + n])
        counts[g] = counts.get(g, 0) + 1
        if counts[g] > best:
            best = counts[g]
    return int(best)


def longest_verbatim_loop(tokens: Sequence[str]) -> int:
    """Longest contiguous span that is immediately repeated (>=1 full repeat)."""
    n = len(tokens)
    best = 0
    # period length L: tokens[i:i+L] == tokens[i+L:i+2L]
    for L in range(1, n // 2 + 1):
        i = 0
        while i + 2 * L <= n:
            if tokens[i : i + L] == tokens[i + L : i + 2 * L]:
                run = L
                j = i + L
                while j + L <= n and tokens[j : j + L] == tokens[i : i + L]:
                    run += L
                    j += L
                if run > best:
                    best = run
                i = j
            else:
                i += 1
    return int(best)


def non_ascii_control_frac(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    for ch in text:
        o = ord(ch)
        if ch in "\n\t\r ":
            continue
        if o < 32 or o == 127 or o > 127:
            bad += 1
    return float(bad) / float(len(text))


def list_formatted(text: str, min_bullets: int = 3) -> bool:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    n = sum(1 for ln in lines if _BULLET.match(ln))
    return n >= min_bullets


def hard_collapse_gate(
    text: str,
    *,
    min_tokens: int = 40,
    min_distinct2: float = 0.35,
    max_4gram_reps: int = 4,
    max_loop_tokens: int = 30,
    max_non_ascii: float = 0.20,
    min_bullet_lines: int = 3,
) -> tuple[int, GateFlags, dict[str, float]]:
    """Return (gate in {0,1}, flags, surface stats). gate=0 if ANY rule trips."""
    toks = tokenize_words(text)
    flags = GateFlags(
        short=len(toks) < min_tokens,
        distinct2=distinct_n(toks, 2) < min_distinct2 if toks else True,
        fourgram=max_ngram_count(toks, 4) >= max_4gram_reps,
        loop=longest_verbatim_loop(toks) >= max_loop_tokens,
        non_ascii=non_ascii_control_frac(text) > max_non_ascii,
        list_fmt=list_formatted(text, min_bullets=min_bullet_lines),
    )
    stats = {
        "n_tokens": float(len(toks)),
        "distinct2": float(distinct_n(toks, 2)),
        "max_4gram": float(max_ngram_count(toks, 4)),
        "loop_tokens": float(longest_verbatim_loop(toks)),
        "non_ascii_frac": float(non_ascii_control_frac(text)),
    }
    gate = 0 if flags.tripped else 1
    return gate, flags, stats


def split_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(t) if p.strip()]
    return parts if parts else [t]


def encode_paragraph_pooled(codec: JepaCodec, text: str) -> np.ndarray:
    sents = split_sentences(text)
    if not sents:
        return np.zeros(int(getattr(codec, "dim", 512)), dtype=np.float32)
    if len(sents) > 32:
        sents = sents[:32]
    vecs = codec.encode(sents)
    return _l2(np.asarray(vecs, dtype=np.float32).mean(axis=0))


def jepa_cosine(codec: JepaCodec, text: str, z_cmd: np.ndarray) -> float:
    if not (text or "").strip():
        return float("nan")
    y = encode_paragraph_pooled(codec, text)
    return _cos(y, z_cmd)


def group_advantages(rewards: Sequence[float], eps: float = 1e-4) -> list[float]:
    """A_i = (r_i - mean) / (std + eps)."""
    r = np.asarray(list(rewards), dtype=np.float64)
    if r.size == 0:
        return []
    mu = float(r.mean())
    sd = float(r.std(ddof=0))
    return [float((x - mu) / (sd + eps)) for x in r]


def expected_digit_from_logits(
    logits_1d: torch.Tensor,
    digit_token_ids: Sequence[int],
    temperature: float = 1.0,
) -> float:
    """Softmax over tokens '1'..'5' → expected digit in [1,5], then map to [0,1] externally."""
    if len(digit_token_ids) != 5:
        raise ValueError("need exactly 5 digit token ids")
    idxs = torch.tensor(list(digit_token_ids), device=logits_1d.device, dtype=torch.long)
    scores = logits_1d.index_select(0, idxs).float() / max(1e-6, float(temperature))
    probs = torch.softmax(scores, dim=0)
    digits = torch.arange(1, 6, device=probs.device, dtype=torch.float32)
    return float((probs * digits).sum().item())


def digit_to_unit(e: float) -> float:
    return float(np.clip((float(e) - 1.0) / 4.0, 0.0, 1.0))


@dataclass
class RewardBreakdown:
    text: str
    c: float
    f: float
    j: float
    gate: int
    reward: float
    flags: dict[str, bool] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "c": self.c,
            "f": self.f,
            "j": self.j,
            "gate": self.gate,
            "reward": self.reward,
            "flags": self.flags,
            "stats": self.stats,
        }


class LogitJudge:
    """Frozen small instruct LM; logit-read digit scores (no sampling)."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        *,
        load_in_4bit: bool = True,
        chat_template: str = "auto",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_name = model_name
        self.device = device
        compute_dtype = torch.bfloat16
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": compute_dtype,
        }
        if load_in_4bit and device.type == "cuda":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            map_dev = device.index if device.index is not None else 0
            kwargs["device_map"] = {"": map_dev}
        else:
            kwargs["device_map"] = None

        print(f"[judge] loading {model_name} (4bit={load_in_4bit})...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not load_in_4bit or device.type != "cuda":
            self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.digit_ids = self._resolve_digit_ids()
        self.chat_template = chat_template
        self._on_device = True

    def _resolve_digit_ids(self) -> list[int]:
        ids: list[int] = []
        for d in "12345":
            # Prefer single-token encodings; try with leading space variants.
            cands = []
            for s in (d, f" {d}"):
                enc = self.tok.encode(s, add_special_tokens=False)
                if len(enc) == 1:
                    cands.append(enc[0])
            if not cands:
                enc = self.tok.encode(d, add_special_tokens=False)
                cands = [enc[0]]
            ids.append(int(cands[0]))
        return ids

    def _format_chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(self.tok, "apply_chat_template"):
            try:
                return self.tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        # Qwen-style fallback
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def to_cpu(self) -> None:
        if not self._on_device:
            return
        try:
            self.model.to("cpu")
        except Exception:
            pass
        self._on_device = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_device(self) -> None:
        if self._on_device:
            return
        try:
            self.model.to(self.device)
        except Exception:
            pass
        self._on_device = True

    @torch.no_grad()
    def score_coherence(
        self,
        texts: Sequence[str],
        *,
        temperature: float = 1.0,
        skip_empty: bool = True,
    ) -> list[float]:
        """Return J in [0,1] per text (nan for empty if skip_empty)."""
        out: list[float] = []
        for t in texts:
            if skip_empty and not (t or "").strip():
                out.append(float("nan"))
                continue
            prompt = self._format_chat(
                COHERENCE_SYSTEM,
                COHERENCE_USER.format(generated_paragraph=(t or "").strip()),
            )
            enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits[0, -1]
            e = expected_digit_from_logits(logits, self.digit_ids, temperature=temperature)
            out.append(digit_to_unit(e))
        return out

    @torch.no_grad()
    def score_topic(
        self,
        texts: Sequence[str],
        keywords: Sequence[str],
        *,
        temperature: float = 1.0,
    ) -> list[float]:
        kw = ", ".join(keywords)
        out: list[float] = []
        for t in texts:
            if not (t or "").strip():
                out.append(float("nan"))
                continue
            prompt = self._format_chat(
                TOPIC_SYSTEM,
                TOPIC_USER.format(keywords=kw, generated_paragraph=(t or "").strip()),
            )
            enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits[0, -1]
            e = expected_digit_from_logits(logits, self.digit_ids, temperature=temperature)
            out.append(digit_to_unit(e))
        return out


class HybridReward:
    """Compose gate + JEPA ramp + judge into scalar reward."""

    def __init__(
        self,
        codec: JepaCodec,
        judge: LogitJudge | None,
        *,
        w_z: float = 0.6,
        w_j: float = 0.4,
        f_floor: float = 0.35,
        f_ceiling: float = 0.62,
    ) -> None:
        self.codec = codec
        self.judge = judge
        self.w_z = float(w_z)
        self.w_j = float(w_j)
        self.f_floor = float(f_floor)
        self.f_ceiling = float(f_ceiling)

    def score_one(
        self,
        text: str,
        z_cmd: np.ndarray,
        *,
        j_override: float | None = None,
        skip_judge_if_gated: bool = True,
    ) -> RewardBreakdown:
        gate, flags, stats = hard_collapse_gate(text)
        if not (text or "").strip():
            return RewardBreakdown(
                text=text or "",
                c=float("nan"),
                f=0.0,
                j=0.0,
                gate=0,
                reward=0.0,
                flags=flags.as_dict(),
                stats=stats,
            )
        c = jepa_cosine(self.codec, text, z_cmd)
        f = f_ramp(c, self.f_floor, self.f_ceiling) if np.isfinite(c) else 0.0
        j = 0.0
        if j_override is not None:
            j = float(j_override)
        elif gate == 1 or not skip_judge_if_gated:
            if self.judge is None:
                j = 0.0
            else:
                j = float(self.judge.score_coherence([text])[0])
                if not np.isfinite(j):
                    j = 0.0
        r = float(gate) * (self.w_z * f + self.w_j * j)
        return RewardBreakdown(
            text=text,
            c=float(c) if np.isfinite(c) else float("nan"),
            f=float(f),
            j=float(j),
            gate=int(gate),
            reward=float(r),
            flags=flags.as_dict(),
            stats=stats,
        )

    def score_group(
        self,
        texts: Sequence[str],
        z_cmd: np.ndarray,
        *,
        skip_judge_if_gated: bool = True,
    ) -> list[RewardBreakdown]:
        gates: list[tuple[int, GateFlags, dict[str, float]]] = [
            hard_collapse_gate(t) for t in texts
        ]
        need_judge_idx = [
            i
            for i, (g, _, _) in enumerate(gates)
            if (g == 1 or not skip_judge_if_gated) and (texts[i] or "").strip()
        ]
        j_map: dict[int, float] = {}
        if self.judge is not None and need_judge_idx:
            js = self.judge.score_coherence([texts[i] for i in need_judge_idx])
            for i, jv in zip(need_judge_idx, js):
                j_map[i] = float(jv) if np.isfinite(jv) else 0.0

        out: list[RewardBreakdown] = []
        for i, text in enumerate(texts):
            gate, flags, stats = gates[i]
            if not (text or "").strip():
                out.append(
                    RewardBreakdown(
                        text=text or "",
                        c=float("nan"),
                        f=0.0,
                        j=0.0,
                        gate=0,
                        reward=0.0,
                        flags=flags.as_dict(),
                        stats=stats,
                    )
                )
                continue
            c = jepa_cosine(self.codec, text, z_cmd)
            f = f_ramp(c, self.f_floor, self.f_ceiling) if np.isfinite(c) else 0.0
            j = float(j_map.get(i, 0.0))
            r = float(gate) * (self.w_z * f + self.w_j * j)
            out.append(
                RewardBreakdown(
                    text=text,
                    c=float(c) if np.isfinite(c) else float("nan"),
                    f=float(f),
                    j=j,
                    gate=int(gate),
                    reward=float(r),
                    flags=flags.as_dict(),
                    stats=stats,
                )
            )
        return out
