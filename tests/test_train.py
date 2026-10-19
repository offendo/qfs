import json
import sqlite3

from rlmath.train import Trainer, TrainConfig


def test_short_run(tmp_path):
    cfg = TrainConfig(
        p=3, rounds=3, candidates=128, d_model=32, n_layers=2, n_heads=4, d_ff=64,
        train_steps=10, batch_size=16, min_pool=8, workers=2, device="cpu", out=str(tmp_path),
        db=str(tmp_path / "heights.sqlite"),
    )
    Trainer(cfg).run()
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert len(log) == 3
    assert log[-1]["best"] >= 2
    assert (tmp_path / "model.pt").exists()
    elites = [json.loads(line) for line in (tmp_path / "elites.jsonl").read_text().splitlines()]
    assert all(e["score"] >= 2 for e in elites)
    db = sqlite3.connect(tmp_path / "heights.sqlite")
    assert db.execute("SELECT COUNT(*) FROM heights").fetchone()[0] == log[-1]["oracle_calls"]
    assert db.execute("SELECT COUNT(*) FROM smoothness").fetchone()[0] > 0


def test_short_policy_gradient_run(tmp_path):
    cfg = TrainConfig(
        p=3, rounds=4, candidates=128, objective="pg", pg_group_size=4, d_model=32, n_layers=2, n_heads=4,
        d_ff=64, workers=2, device="cpu", out=str(tmp_path), db="",
    )
    trainer = Trainer(cfg)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.run()
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert len(log) == 4
    assert "edit_mean_reward" in log[-1] and 0 <= log[-1]["edit_improved"] <= 1
    assert any((a != b).any() for a, b in zip(before, trainer.model.parameters()))
    assert (tmp_path / "model.pt").exists()
