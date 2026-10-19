"""PatternBoost-style search loop: generate -> score -> local search -> keep elites -> retrain.

Each round:
  1. Propose candidates: edits of elite polynomials filled in by the model (the "f -> f'" move),
     polynomials generated from scratch by the model, and uniformly random ones.
  2. Score them with the Julia oracle (height, then a smoothness check for anything >= 2).
  3. Run hill climbing from the best new finds.
  4. Add everything good to the elite pool.
  5. Train the model: either a few hundred steps imitating the elite pool (--objective maskgit),
     or one policy-gradient step on this round's edits (--objective pg).

With --objective pg an edit's reward is (child score - parent score). Each parent and mask is
sampled --pg-group-size times, and a child's advantage is its reward minus the mean reward of its
group (REINFORCE with a group baseline, no critic). Generation from scratch has no parent, so it is
skipped in this mode.

Run `rlmath-train --help` for options. Everything is written to --out (default runs/<timestamp>):
  config.json, log.jsonl (one line per round), elites.jsonl (every new find), model.pt.
"""

import argparse
from dataclasses import asdict, dataclass, field, fields
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from rlmath.model import MaskedPolyTransformer, ModelConfig
from rlmath.oracle import HeightOracle
from rlmath.polynomials import augment, monomials, random_polys, to_string
from rlmath.search import ElitePool, Scorer, local_search

log = logging.getLogger("rlmath.train")


@dataclass
class TrainConfig:
    # problem
    p: int = 5
    n: int = 4
    d: int = 4
    cutoff: int = 10  # quartic K3 surfaces: finite heights are at most 10
    # Score for supersingular (infinite-height) finds. Default 1 = no better than ordinary: at p = 5
    # sparse supersingular quartics are easy to find and would flood the search. Set to 10 to target them.
    infinite_score: int = 1
    # budget
    rounds: int = 200
    max_oracle_calls: int = 0  # stop once this many polynomials were scored (0 = no limit)
    # proposals per round
    candidates: int = 1024
    frac_edit: float = 0.5  # edits of elites, filled in by the model
    frac_generate: float = 0.25  # generated from scratch by the model
    max_edit: int = 8  # an edit re-samples between 1 and this many coefficients
    parent_temperature: float = 1.0
    sample_temperature: float = 1.0
    use_model: bool = True  # False: edits are filled uniformly at random (model-free baseline)
    objective: str = "maskgit"  # "maskgit": imitate the elite pool; "pg": policy gradient on edits
    pg_group_size: int = 8  # children sampled per (parent, mask); their mean reward is the baseline
    # local search
    local_search_starts: int = 8
    local_search_steps: int = 5
    # elite pool
    per_level: int = 2000
    # model
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    # optimisation
    train_steps: int = 100  # per round
    min_pool: int = 64  # don't train until the pool has this many polynomials
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    # misc
    workers: int = 0  # Julia worker processes (0 = number of CPUs - 1)
    device: str = "auto"
    seed: int = 0
    out: str = ""
    db: str = "runs/heights.sqlite"  # every scored polynomial is saved here ("" to disable)
    checkpoint_every: int = 10


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EditBatch:
    """The model-filled edits of one round, kept so the policy gradient can replay the sampling."""

    parent_scores: np.ndarray  # (E,)
    height: np.ndarray  # (E,) the height token the fill was conditioned on
    traces: list = field(default_factory=list)  # (start, end, trace) per chunk of rows
    invalid: np.ndarray | None = None  # (E,) the child was the zero polynomial


class Trainer:
    def __init__(self, cfg: TrainConfig):
        if cfg.objective not in ("maskgit", "pg"):
            raise ValueError(f"unknown objective {cfg.objective!r}; use 'maskgit' or 'pg'")
        if cfg.objective == "pg" and cfg.pg_group_size < 2:
            raise ValueError("--pg-group-size must be at least 2 to have a baseline")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.device = pick_device(cfg.device)
        self.out = Path(cfg.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

        self.n_monomials = len(monomials(cfg.n, cfg.d))
        self.oracle = HeightOracle(cfg.p, cfg.n, cfg.d, cfg.cutoff, workers=cfg.workers or None, db=cfg.db or None)
        self.scorer = Scorer(
            self.oracle,
            infinite_score=cfg.infinite_score,
            past_cutoff_is_infinite=(cfg.n == 4 and cfg.d == 4 and cfg.cutoff >= 10),
        )
        self.pool = ElitePool(cfg.p, cfg.n, cfg.d, per_level=cfg.per_level)

        self.model = MaskedPolyTransformer(
            ModelConfig(
                p=cfg.p, n=cfg.n, d=cfg.d,
                max_height=max(cfg.cutoff + 1, cfg.infinite_score),
                d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                d_ff=cfg.d_ff, dropout=cfg.dropout,
            )
        ).to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
        )
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda step: min(1.0, (step + 1) / cfg.warmup_steps)
        )
        self.pg = cfg.objective == "pg" and cfg.use_model
        self.trained = self.pg  # the policy gradient starts from the untrained model
        self.edit_stats: dict = {}
        self.start = time.time()
        self.bar = None

    def phase(self, name: str):
        """Show what the current round is doing next to the progress bar."""
        if self.bar is not None:
            self.bar.set_postfix_str(f"best {self.pool.best} | {name}", refresh=True)
            self.bar.n = self.oracle.calls if self.cfg.max_oracle_calls else self.bar.n
            self.bar.refresh()

    # ---- proposals -------------------------------------------------------------------------

    def propose(self) -> tuple[np.ndarray, EditBatch | None]:
        """Candidates for this round, and (for the policy gradient) the record of the model's edits."""
        cfg = self.cfg
        if len(self.pool) == 0:
            return random_polys(self.rng, cfg.candidates, cfg.p, cfg.n, cfg.d), None
        group = cfg.pg_group_size if self.pg else 1
        n_edit = int(cfg.candidates * cfg.frac_edit) // group * group
        n_gen = int(cfg.candidates * cfg.frac_generate) if (cfg.use_model and self.trained and not self.pg) else 0
        n_rand = cfg.candidates - n_edit - n_gen

        parents, parent_scores = self.pool.sample_parents(
            self.rng, n_edit // group, cfg.parent_temperature, with_scores=True
        )
        k = self.rng.integers(1, cfg.max_edit + 1, size=n_edit // group)
        masked = np.zeros_like(parents, dtype=bool)
        for i, ki in enumerate(k):
            masked[i, self.rng.choice(self.n_monomials, size=ki, replace=False)] = True
        # In a policy-gradient group, every child comes from the same parent and mask.
        parents, parent_scores, masked = (np.repeat(x, group, axis=0) for x in (parents, parent_scores, masked))

        edit_batch = None
        if cfg.use_model and self.trained:
            # Ask for the best level seen so far.
            height = np.full(n_edit, self.pool.best)
            if self.pg:
                edit_batch = EditBatch(parent_scores, height)
            edits = self.fill(parents, masked, height, edit_batch.traces if edit_batch else None)
        else:
            edits = parents.copy()
            edits[masked] = self.rng.integers(0, cfg.p, size=int(masked.sum()))

        parts = [edits]
        if n_gen:
            blank = np.zeros((n_gen, self.n_monomials), dtype=np.int64)
            parts.append(self.fill(blank, np.ones_like(blank, dtype=bool), np.full(n_gen, self.pool.best)))
        parts.append(random_polys(self.rng, n_rand, cfg.p, cfg.n, cfg.d))
        out = np.concatenate(parts)
        if edit_batch is not None:
            edit_batch.invalid = ~edits.any(axis=1)
        out[~out.any(axis=1)] = random_polys(self.rng, int((~out.any(axis=1)).sum()), cfg.p, cfg.n, cfg.d)
        return out, edit_batch

    def fill(
        self, coeffs: np.ndarray, masked: np.ndarray, height: np.ndarray, traces: list | None = None
    ) -> np.ndarray:
        """Fill the masked entries with the model. If `traces` is a list, append (start, end, trace)
        for each chunk of rows so the policy gradient can replay the sampling."""
        self.model.eval()
        out = []
        for i in range(0, len(coeffs), 512):
            c = torch.as_tensor(coeffs[i : i + 512], device=self.device)
            m = torch.as_tensor(masked[i : i + 512], device=self.device)
            h = torch.as_tensor(height[i : i + 512], device=self.device)
            trace = [] if traces is not None else None
            out.append(self.model.fill(c, m, h, temperature=self.cfg.sample_temperature, trace=trace).cpu().numpy())
            if traces is not None:
                traces.append((i, i + len(c), trace))
        return np.concatenate(out)

    # ---- training --------------------------------------------------------------------------

    def train_model(self) -> float:
        cfg = self.cfg
        coeffs, scores = self.pool.arrays()
        if len(coeffs) < cfg.min_pool:
            return float("nan")
        self.model.train()
        losses = []
        for _ in range(cfg.train_steps):
            idx = self.rng.integers(len(coeffs), size=cfg.batch_size)
            batch = augment(self.rng, coeffs[idx], cfg.p, cfg.n, cfg.d)
            c = torch.as_tensor(batch, device=self.device)
            h = torch.as_tensor(scores[idx], device=self.device)
            loss = self.model.loss(c, h)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            self.sched.step()
            losses.append(loss.item())
        self.trained = True
        return float(np.mean(losses))

    def train_policy_gradient(self, edits: EditBatch | None, child_scores: np.ndarray) -> float:
        """One REINFORCE step on this round's model edits.

        reward = child score - parent score (a child that is the zero polynomial counts as 0);
        advantage = reward - mean reward of its group, divided by the batch's advantage std.
        Returns the surrogate loss (not comparable across rounds); NaN if there was nothing to train on.
        """
        cfg = self.cfg
        if edits is None or len(edits.parent_scores) == 0:
            return float("nan")
        n = len(edits.parent_scores)
        child = np.where(edits.invalid, 0, child_scores[:n])
        reward = (child - edits.parent_scores).astype(np.float64)
        self.edit_stats = {
            "edit_mean_reward": float(reward.mean()),
            "edit_improved": float((reward > 0).mean()),
            "edit_worse": float((reward < 0).mean()),
        }
        adv = reward.reshape(-1, cfg.pg_group_size)
        adv = (adv - adv.mean(axis=1, keepdims=True)).reshape(-1)
        std = adv.std()
        if std < 1e-8:  # every group had identical rewards: no signal
            return 0.0
        adv_t = torch.as_tensor(adv / std, dtype=torch.float32, device=self.device)

        # The sampler ran in eval mode (no dropout); replay in the same mode. The loss is linear in the
        # per-step log-probabilities, so backpropagate one step at a time to bound memory.
        self.model.eval()
        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        for start, end, trace in edits.traces:
            h = torch.as_tensor(edits.height[start:end], device=self.device)
            for step in trace:
                logp = self.model.trace_log_prob([step], h, cfg.sample_temperature)
                loss = -(adv_t[start:end] * logp).sum() / n
                loss.backward()
                total += loss.item()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.opt.step()
        self.sched.step()
        return total

    # ---- main loop -------------------------------------------------------------------------

    def record(self, new, round_):
        with open(self.out / "elites.jsonl", "a") as f:
            for e in new:
                f.write(json.dumps({
                    "score": e.score, "status": e.status, "round": round_,
                    "oracle_calls": self.oracle.calls, "seconds": round(time.time() - self.start, 1),
                    "coeffs": e.coeffs.tolist(), "poly": to_string(e.coeffs, self.cfg.n, self.cfg.d),
                }) + "\n")

    def run_round(self, round_: int) -> dict:
        cfg = self.cfg
        calls_before = self.oracle.calls
        self.phase("proposing")
        candidates, edit_batch = self.propose()
        self.phase(f"scoring {len(candidates)} candidates")
        scores, results = self.scorer(candidates)
        new = self.pool.add(candidates, scores, results, round_)

        # Hill-climb from the best of this round's new finds.
        starts = sorted(new, key=lambda e: -e.score)[: cfg.local_search_starts]
        for i, e in enumerate(starts):
            self.phase(f"local search {i + 1}/{len(starts)}")
            rows, s, res = local_search(self.scorer, e.coeffs, e.score, cfg.p, cfg.local_search_steps)
            new += self.pool.add(rows, s, res, round_)

        self.record(new, round_)
        self.phase("training model")
        if self.pg:
            loss = self.train_policy_gradient(edit_batch, scores)
        else:
            loss = self.train_model() if cfg.use_model else float("nan")
        found = {}
        for e in new:
            found[e.score] = found.get(e.score, 0) + 1
        return {
            "round": round_,
            "seconds": round(time.time() - self.start, 1),
            "oracle_calls": self.oracle.calls,
            "round_calls": self.oracle.calls - calls_before,
            "smooth_calls": self.oracle.smooth_calls,
            "best": self.pool.best,
            "new_by_score": dict(sorted(found.items())),
            "pool_by_score": self.pool.counts(),
            "loss": loss,
            **(self.edit_stats if self.pg else {}),
            "candidate_mean_score": float(scores.mean()),
            "supersingular_candidates": sum(r.status in ("inf", "past_cutoff") for r in results),
        }

    def run(self):
        cfg = self.cfg
        # Also log to <out>/train.log, so `tail -f` works however the run was started.
        handler = logging.FileHandler(self.out / "train.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logging.getLogger("rlmath").addHandler(handler)
        log.info("writing to %s; device %s; %d Julia workers", self.out, self.device, self.oracle.n_workers)
        # The bar counts oracle calls when there is a budget, else rounds. It turns itself off when
        # output isn't a terminal (e.g. redirected to a file); log lines are printed either way.
        by_calls = bool(cfg.max_oracle_calls)
        self.bar = tqdm(
            total=cfg.max_oracle_calls if by_calls else cfg.rounds,
            unit=" calls" if by_calls else " rounds",
            unit_scale=by_calls,
            dynamic_ncols=True,
            disable=None,
        )
        try:
            with logging_redirect_tqdm():
                for r in range(cfg.rounds):
                    stats = self.run_round(r)
                    with open(self.out / "log.jsonl", "a") as f:
                        f.write(json.dumps(stats) + "\n")
                    if not by_calls:
                        self.bar.update(1)
                    self.phase("round done")
                    rate = stats["oracle_calls"] / max(stats["seconds"], 1e-9)
                    log.info(
                        "round %d  %.0fs  calls %d (%.0f/s)  best %d  new %s  pool %s  loss %.3f%s",
                        r, stats["seconds"], stats["oracle_calls"], rate, stats["best"],
                        stats["new_by_score"], stats["pool_by_score"], stats["loss"],
                        "  edit reward %+.3f (up %.1f%%, down %.1f%%)" % (
                            stats["edit_mean_reward"], 100 * stats["edit_improved"], 100 * stats["edit_worse"]
                        ) if "edit_mean_reward" in stats else "",
                    )
                    if cfg.use_model and (r + 1) % cfg.checkpoint_every == 0:
                        torch.save({"config": asdict(cfg), "model": self.model.state_dict()}, self.out / "model.pt")
                    if cfg.max_oracle_calls and self.oracle.calls >= cfg.max_oracle_calls:
                        break
        finally:
            self.bar.close()
            self.oracle.close()
            if cfg.use_model:
                torch.save({"config": asdict(cfg), "model": self.model.state_dict()}, self.out / "model.pt")


def parse_args(argv=None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(TrainConfig):
        name = "--" + f.name.replace("_", "-")
        if f.type is bool or isinstance(f.default, bool):
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(name, type=type(f.default), default=f.default)
    return TrainConfig(**vars(parser.parse_args(argv)))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    Trainer(parse_args(argv)).run()


if __name__ == "__main__":
    main()
