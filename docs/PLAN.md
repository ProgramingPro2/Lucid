# Plan and Key Decisions

## Goal

Build a runnable proof-of-concept that demonstrates two things honestly:
1. Latent-concept prediction in language works (predict ideas, not tokens).
2. A closed LLM + human self-improvement loop measurably improves output quality.

Not a goal: human-level intelligence, or a guaranteed scaling path to it.

## Key decisions

- **Frozen codec, trained predictor.** Training a concept encoder/decoder from
  scratch would consume the whole compute budget before the loop is testable.
  We freeze a pretrained codec and train only a small predictor on cached
  embeddings, which fits a single 24GB GPU comfortably.

- **Retrieval decoder as the default.** Generative decoding of arbitrary concept
  vectors (SONAR) is heavy and degrades on unseen vectors. The default decoder
  snaps a predicted vector to the nearest real sentence in a bank. This always
  produces grammatical text and runs anywhere; SONAR is an optional upgrade.

- **Reward-ranked fine-tuning (RAFT/RWR), not token-DPO.** The policy is a
  continuous regression model, so token-level DPO does not apply directly. RAFT
  fits cleanly: sample, score, keep the best, fine-tune.

- **Offline-first reward.** A deterministic heuristic judge and statistical
  quality-control checks let the entire pipeline run with no API key or network.
  The OpenAI judge and human raters layer on top when available.

- **Single language stack for the app.** The rating app is FastAPI + SQLite to
  avoid a second toolchain. (Prisma/Node is a reasonable alternative if a richer
  app is wanted later.)

## Staging

1. Scaffold + codec round-trip.
2. Data pipeline + embedding cache.
3. Predictor training + latent rollout generation.
4. Reward layer (judge + human app + QC) and the self-improvement loop.
5. Stretch: experts + gating router.

## Decision point before scaling

Only after the loop shows rising reward on a real corpus should bigger GPUs be
rented. Revisit predictor size, corpus size, and whether the experts/router
stage earns its complexity at that point.
