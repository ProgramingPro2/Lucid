"""Document-intact token streams for E1.

The endpoint is *context gain*: NLL of a target span given the true preceding
context minus NLL given a shuffled context from another document. That requires
document boundaries to be preserved, so this module never concatenates documents
into one undifferentiated stream. A training window is drawn inside one
document; an evaluation item pairs a true prefix with a prefix from a different
document.

Everything is deterministic given the seed and the corpus, and no tokenizer
special tokens are invented beyond the model's own EOS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvalItem:
    doc_index: int
    true_ids: list[int]
    shuffled_ids: list[int]
    target_start: int

    def __post_init__(self) -> None:
        if len(self.true_ids) != len(self.shuffled_ids):
            raise ValueError("true and shuffled contexts must align")
        if not (0 < self.target_start < len(self.true_ids)):
            raise ValueError("target_start must be inside the sequence")


class PackedCorpus:
    """A list of tokenised documents, each kept intact."""

    def __init__(self, docs: list[list[int]], *, eos_id: int | None = None) -> None:
        if not docs:
            raise ValueError("empty corpus")
        self.docs = [list(d) for d in docs if d]
        if not self.docs:
            raise ValueError("corpus has no non-empty documents")
        self.eos_id = eos_id
        self.n_tokens = sum(len(d) for d in self.docs)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        tokenizer,
        *,
        max_docs: int | None = None,
        max_tokens: int | None = None,
        min_tokens: int = 64,
    ) -> "PackedCorpus":
        """Tokenise documents and stop once ``max_tokens`` is reached.

        Documents shorter than ``min_tokens`` are skipped: they cannot supply a
        training window and would only contribute an EOS.
        """
        eos = tokenizer.eos_token_id
        docs: list[list[int]] = []
        total = 0
        for text in texts[:max_docs] if max_docs is not None else texts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if eos is not None:
                ids = ids + [eos]
            if len(ids) < min_tokens:
                continue
            docs.append(ids)
            total += len(ids)
            if max_tokens is not None and total >= max_tokens:
                break
        return cls(docs, eos_id=eos)

    def split(self, test_fraction: float = 0.1, seed: int = 0) -> tuple["PackedCorpus", "PackedCorpus"]:
        """Document-level split. No document appears in both halves."""
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1)")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.docs))
        n_test = max(1, int(round(len(self.docs) * test_fraction)))
        test_idx = set(int(i) for i in order[:n_test])
        train = [d for i, d in enumerate(self.docs) if i not in test_idx]
        test = [d for i, d in enumerate(self.docs) if i in test_idx]
        if not train or not test:
            raise ValueError("split produced an empty half; use more documents")
        return (
            PackedCorpus(train, eos_id=self.eos_id),
            PackedCorpus(test, eos_id=self.eos_id),
        )

    def train_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        context: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(input_ids, labels)`` of shape ``(B, context)``.

        The window is drawn entirely inside one document. ``labels`` are the
        next tokens, so no label crosses a document boundary.
        """
        eligible = [i for i, d in enumerate(self.docs) if len(d) > context]
        if not eligible:
            raise ValueError("no document is longer than the context window")
        x = np.empty((batch_size, context), dtype=np.int64)
        y = np.empty((batch_size, context), dtype=np.int64)
        for row in range(batch_size):
            i = int(eligible[int(rng.integers(0, len(eligible)))])
            doc = self.docs[i]
            start = int(rng.integers(0, len(doc) - context))
            x[row] = doc[start : start + context]
            y[row] = doc[start + 1 : start + context + 1]
        return x, y

    def eval_items(
        self,
        n_items: int,
        *,
        prefix: int = 384,
        target: int = 128,
        seed: int = 0,
    ) -> list[EvalItem]:
        """Paired true/shuffled-context items, one per document where possible.

        The target span is identical in both conditions; only the preceding
        context changes. That is what makes the difference a use-of-context
        measure rather than a difficulty measure.
        """
        span = prefix + target
        eligible = [i for i, d in enumerate(self.docs) if len(d) >= span + 1]
        if len(eligible) < 2:
            raise ValueError("need at least two documents long enough for an item")
        rng = np.random.default_rng(seed)
        picks = rng.permutation(len(eligible))
        items: list[EvalItem] = []
        for k in range(min(n_items, len(eligible))):
            i = eligible[int(picks[k])]
            doc = self.docs[i]
            start = int(rng.integers(0, len(doc) - span))
            true = doc[start : start + span]
            # A context of the same length from a different document.
            other_i = eligible[int(rng.integers(0, len(eligible)))]
            if other_i == i and len(eligible) > 1:
                other_i = eligible[(eligible.index(i) + 1) % len(eligible)]
            other = self.docs[other_i]
            other_start = int(rng.integers(0, len(other) - prefix))
            shuffled = other[other_start : other_start + prefix] + doc[
                start + prefix : start + span
            ]
            items.append(
                EvalItem(
                    doc_index=i,
                    true_ids=true,
                    shuffled_ids=shuffled,
                    target_start=prefix,
                )
            )
        return items


def target_labels(ids: list[int], target_start: int) -> list[int]:
    """Labels with the prefix masked and the target shifted by one.

    The model predicts token ``t+1`` from position ``t``. The first target token
    is predicted by the last prefix position, so it is included; the final token
    has no successor and is excluded.
    """
    if not 0 < target_start < len(ids):
        raise ValueError("target_start must be inside the sequence")
    labels = [-100] * len(ids)
    for pos in range(target_start - 1, len(ids) - 1):
        labels[pos] = ids[pos + 1]
    return labels
