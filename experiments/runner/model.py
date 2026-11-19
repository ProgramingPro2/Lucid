"""Model runner: load a backbone, generate, and measure.

This is the layer every arm calls. It exists so that the throughput and VRAM
numbers in ``docs/PROTOCOL.md`` §10 are measured, not assumed, and so that every
arm uses identical decode settings.

It deliberately does no science: no scoring, no arms, no training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import torch


@dataclass(frozen=True)
class GenConfig:
    max_new_tokens: int = 128
    do_sample: bool = False          # greedy by default; sampling must be explicit
    temperature: float = 1.0
    top_p: float = 1.0
    batch_size: int = 4
    seed: int | None = None          # only used when do_sample=True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenStats:
    n_prompts: int = 0
    prompt_tokens: int = 0
    new_tokens: int = 0
    seconds: float = 0.0
    peak_vram_gb: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / self.seconds if self.seconds > 0 else float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["tokens_per_second"] = self.tokens_per_second
        return d


class LM:
    """A thin wrapper around a causal LM.

    Left-padding is required for batched decoder-only generation, otherwise the
    shorter prompts are padded on the right and generation continues from pad
    tokens. That bug silently produces empty or truncated outputs, so the
    tokenizer's padding side is forced here rather than at each call site.
    """

    def __init__(
        self,
        model_id: str,
        *,
        load_in_4bit: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        kwargs: dict = {"torch_dtype": dtype}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["device_map"] = {"": device} if device == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if not load_in_4bit and device == "cuda":
            self.model.to(device)
        self.model.eval()

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _reset_vram(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _peak_vram_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024**3

    def generate(self, prompts: list[str], cfg: GenConfig) -> tuple[list[str], GenStats]:
        """Generate one continuation per prompt. Returns completions only."""
        stats = GenStats(n_prompts=len(prompts))
        out: list[str] = []
        self._reset_vram()

        gen_kwargs: dict = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if cfg.do_sample:
            gen_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)

        start = time.perf_counter()
        for i in range(0, len(prompts), cfg.batch_size):
            batch = prompts[i : i + cfg.batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(self.model.device)
            stats.prompt_tokens += int(enc["attention_mask"].sum())

            if cfg.seed is not None and cfg.do_sample:
                torch.manual_seed(cfg.seed + i)

            with torch.no_grad():
                ids = self.model.generate(**enc, **gen_kwargs)

            new = ids[:, enc["input_ids"].shape[1] :]
            stats.new_tokens += int((new != self.tokenizer.pad_token_id).sum())
            out.extend(self.tokenizer.batch_decode(new, skip_special_tokens=True))

        stats.seconds = time.perf_counter() - start
        stats.peak_vram_gb = self._peak_vram_gb()
        return out, stats
