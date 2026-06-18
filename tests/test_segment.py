from __future__ import annotations

from conceptpoc.data.segment import segment_sentences


def test_basic_split():
    text = "The sun rose over the hills. Birds began to sing. The day was warm."
    sents = segment_sentences(text, min_chars=5)
    assert len(sents) == 3


def test_abbreviation_not_split():
    text = "Dr. Smith arrived early. He greeted everyone warmly."
    sents = segment_sentences(text, min_chars=5)
    assert any("Dr. Smith arrived early" in s for s in sents)
    assert len(sents) == 2


def test_min_length_filter():
    text = "Hi. This is a much longer sentence that should be kept."
    sents = segment_sentences(text, min_chars=12)
    assert all(len(s) >= 12 for s in sents)
