# Limitations (read this)

This project is an honest proof-of-concept. It is easy to overstate what it
does, so this document is deliberately blunt.

## What it demonstrates
- Predicting sentence-level concept vectors and decoding them back to text.
- A closed reward loop (LLM-judge + human ratings -> reward-ranked fine-tuning)
  that can measurably raise output quality on a small corpus.
- The plumbing for human-in-the-loop rating with quality control over raters.

## What it does NOT do
- **It is not AGI and is not a path to it.** "Scaling this reaches human-level
  intelligence" is an unproven research bet, not something this code delivers.
- **It does not invent genuinely new concepts.** The retrieval decoder can only
  emit sentences already in its bank. Even the SONAR decoder generates within
  the distribution of its training. Open-ended concept formation is unsolved
  here.
- **Self-improvement is bounded.** The loop optimizes against the current judge
  and raters; it plateaus quickly and is only as good as those signals.

## Known failure modes
- **Reward hacking.** The predictor can learn to please a weak judge with
  fluent-but-empty output. The heuristic judge is especially gameable; combine
  with the LLM judge and humans for anything real.
- **Decoder fidelity.** A predicted vector may not correspond to any good
  sentence; retrieval snapping hides this by returning the nearest bank entry,
  which can drift off-topic.
- **Tiny-corpus overfitting.** On the bundled sample corpus, next-vector
  retrieval accuracy is low; this is expected. Use a substantial corpus for
  meaningful results.
- **Heuristic QC is statistical, not semantic.** It catches lazy/troll patterns
  (constant, too-fast, consensus-violating ratings) but not clever bad-faith
  ratings; an LLM QC pass can be layered on.

## Joint JEPA codec (training your own encoder/decoder)
- **Collapse is the default outcome without care.** The variance/covariance
  regularization, EMA stop-grad target, and reconstruction anchor are all
  required together. Watch the collapse alarm (`std_z`) every run; if it trends
  toward zero, the concept space is degenerating. Tune `var_weight`,
  `alpha_max`, and `recon_warmup_epochs`.
- **Single-vector reconstruction is lossy.** Squeezing a whole sentence through
  one vector and regenerating it word-by-word with a small model produces
  approximate paraphrases at best, and gibberish until trained enough. Expect
  worse decode quality than the frozen sentence-transformer + retrieval path,
  in exchange for owning (and being able to grow) the concept space.
- **It needs real training to be meaningful.** The bundled offline char-level
  runs are smoke tests. Use the BPE tokenizer, larger dims, and many epochs on a
  substantial corpus (on the 3090) before judging quality.
- **alpha is a knife edge.** Too much prediction weight smears or collapses the
  space; too little and it is just an autoencoder with a predictor attached.

## Honest scaling note
Each ingredient (a strong predictor, a large clean corpus, a reliable judge,
many trustworthy raters) is itself a significant effort. Treat the prototype as
a sandbox for the ideas, not as step one of a guaranteed climb.
