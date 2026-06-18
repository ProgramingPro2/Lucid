"""Self-improvement loop via reward-ranked fine-tuning (RAFT / RWR).

Classic DPO assumes a policy with token log-probabilities; our policy is a
continuous regression model, so token-level DPO does not apply directly.
Instead we use reward-ranked fine-tuning, which fits this architecture cleanly:

  for each iteration:
    1. sample N rollouts per prompt (noise -> diversity)
    2. score each with the judge (and optional reward model), apply human weight
    3. keep the top fraction by reward
    4. fine-tune the predictor on the kept trajectories
    5. record the mean reward

If mean reward rises across iterations, the loop is closing. This is the
honest, measurable claim of the POC; it is not open-ended self-improvement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..codec.base import ConceptCodec
from ..config import LoopConfig, PredictorConfig, TrainConfig
from ..data.cache import CachedCorpus
from ..data.dataset import ConceptSequenceDataset
from ..predictor.model import ConceptPredictor
from ..predictor.train import train_predictor
from ..progress import iter_bar, log_stage
from ..reward.judge import Judge
from ..reward.preference import ScoredItem
from ..generate.rollout import generate_rollout, rollout_to_text


@dataclass
class LoopReport:
    mean_reward_per_iter: list[float] = field(default_factory=list)
    kept_per_iter: list[int] = field(default_factory=list)
    best_samples: list[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        r = self.mean_reward_per_iter
        return len(r) >= 2 and r[-1] > r[0]


def _trajectories_to_dataset(
    trajectories: list[np.ndarray], context_len: int
) -> ConceptSequenceDataset:
    vectors = np.concatenate(trajectories, axis=0).astype(np.float32)
    offsets = np.cumsum([0] + [t.shape[0] for t in trajectories]).astype(np.int64)
    corpus = CachedCorpus(
        vectors=vectors,
        sentences=[""] * vectors.shape[0],
        doc_offsets=offsets,
    )
    return ConceptSequenceDataset(corpus, context_len=context_len)


def run_self_improvement(
    model: ConceptPredictor,
    codec: ConceptCodec,
    prompts: list[np.ndarray],
    judge: Judge,
    loop_cfg: LoopConfig,
    predictor_cfg: PredictorConfig,
    train_cfg: TrainConfig,
    device: str = "cpu",
    seed: int = 0,
    progress: bool = True,
) -> tuple[ConceptPredictor, LoopReport]:
    rng = np.random.default_rng(seed)
    report = LoopReport()

    for iteration in range(loop_cfg.iterations):
        log_stage(
            f"improve: iteration {iteration + 1}/{loop_cfg.iterations}",
            enabled=progress,
        )
        items: list[ScoredItem] = []
        kept_trajectories: list[np.ndarray] = []

        prompt_iter = iter_bar(
            enumerate(prompts),
            desc="rollouts",
            total=len(prompts),
            enabled=progress,
            unit="prompt",
        )
        for p_idx, seed_vecs in prompt_iter:
            candidates: list[ScoredItem] = []
            for _ in range(loop_cfg.rollouts_per_prompt):
                roll = generate_rollout(
                    model,
                    seed_vecs,
                    steps=loop_cfg.rollout_steps,
                    temperature=0.15,
                    device=device,
                    rng=rng,
                )
                text = " ".join(rollout_to_text(roll, codec))
                result = judge.score(
                    prompt=" ".join(codec.decode(seed_vecs)) if seed_vecs.ndim > 1 else "",
                    output=text,
                    vectors=roll.generated_vectors,
                )
                candidates.append(
                    ScoredItem(
                        prompt_id=str(p_idx),
                        text=text,
                        vectors=roll.all_vectors,
                        reward=result.score,
                    )
                )
            items.extend(candidates)

            candidates.sort(key=lambda x: x.reward, reverse=True)
            keep_n = max(1, int(len(candidates) * loop_cfg.keep_fraction))
            for c in candidates[:keep_n]:
                kept_trajectories.append(c.vectors)

        mean_reward = float(np.mean([it.reward for it in items])) if items else 0.0
        report.mean_reward_per_iter.append(mean_reward)
        report.kept_per_iter.append(len(kept_trajectories))
        log_stage(
            f"improve: mean_reward={mean_reward:.3f} kept={len(kept_trajectories)}",
            enabled=progress,
        )

        best = max(items, key=lambda x: x.reward)
        report.best_samples.append(best.text)

        usable = [t for t in kept_trajectories if t.shape[0] >= 2]
        if usable:
            ds = _trajectories_to_dataset(usable, predictor_cfg.max_seq_len)
            ft_cfg = TrainConfig(
                batch_size=train_cfg.batch_size,
                lr=train_cfg.lr * 0.5,
                weight_decay=train_cfg.weight_decay,
                epochs=loop_cfg.finetune_epochs,
                warmup_steps=0,
                grad_clip=train_cfg.grad_clip,
                val_fraction=0.0 if len(ds) < 4 else 0.1,
                seed=seed,
            )
            if len(ds) >= 2:
                model, _ = train_predictor(
                    ds,
                    predictor_cfg,
                    ft_cfg,
                    device=device,
                    model=model,
                    progress=progress,
                )

    return model, report
