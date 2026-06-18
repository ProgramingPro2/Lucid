from __future__ import annotations

from conceptpoc.progress import iter_bar, log_epoch, log_stage


def test_iter_bar_disabled_yields_all_items():
    items = list(iter_bar(range(5), desc="test", enabled=False))
    assert items == [0, 1, 2, 3, 4]


def test_log_stage_disabled_is_silent(capsys):
    log_stage("should not appear", enabled=False)
    assert capsys.readouterr().out == ""


def test_log_epoch_disabled_is_silent(capsys):
    log_epoch("train", 1, 10, {"loss": 0.5}, enabled=False)
    assert capsys.readouterr().out == ""


def test_log_epoch_enabled_prints_metrics(capsys):
    log_epoch("train", 2, 10, {"train_loss": 0.123, "val_cosine": 0.891})
    out = capsys.readouterr().out
    assert "[train] epoch 2/10" in out
    assert "train_loss=0.123" in out
    assert "val_cosine=0.891" in out
