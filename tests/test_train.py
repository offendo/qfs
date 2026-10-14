import json

from rlmath.train import Trainer, TrainConfig


def test_short_run(tmp_path):
    cfg = TrainConfig(
        p=3, rounds=3, candidates=128, d_model=32, n_layers=2, n_heads=4, d_ff=64,
        train_steps=10, batch_size=16, min_pool=8, workers=2, device="cpu", out=str(tmp_path),
    )
    Trainer(cfg).run()
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert len(log) == 3
    assert log[-1]["best"] >= 2
    assert (tmp_path / "model.pt").exists()
    elites = [json.loads(line) for line in (tmp_path / "elites.jsonl").read_text().splitlines()]
    assert all(e["score"] >= 2 for e in elites)
