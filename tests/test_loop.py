from __future__ import annotations

from conceptpoc.config import LoopConfig, PredictorConfig, TrainConfig
from conceptpoc.loop.self_improve import run_self_improvement
from conceptpoc.predictor.model import ConceptPredictor
from conceptpoc.reward.judge import HeuristicJudge


def test_self_improvement_runs_end_to_end(fitted_codec, sentences):
    model = ConceptPredictor(
        concept_dim=fitted_codec.dim,
        cfg=PredictorConfig(d_model=32, n_layers=2, n_heads=4, max_seq_len=16),
    )
    prompts = [fitted_codec.encode(sentences[i : i + 2]) for i in range(0, 6, 2)]
    loop_cfg = LoopConfig(
        rollouts_per_prompt=4,
        keep_fraction=0.5,
        rollout_steps=4,
        iterations=2,
        finetune_epochs=1,
    )
    _, report = run_self_improvement(
        model,
        fitted_codec,
        prompts,
        HeuristicJudge(),
        loop_cfg,
        PredictorConfig(d_model=32, n_layers=2, n_heads=4, max_seq_len=16),
        TrainConfig(batch_size=8, epochs=1),
        device="cpu",
        progress=False,
    )
    assert len(report.mean_reward_per_iter) == 2
    assert len(report.best_samples) == 2
    assert all(0.0 <= r <= 1.0 for r in report.mean_reward_per_iter)
