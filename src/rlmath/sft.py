"""Supervised warm start for the policy: imitate edits that worked.

Trains the agent of rlmath.reinforce to put high probability on a child, given its parent, using
(parent, child) pairs whose child is at least as high as the parent (see rlmath.pairs and
experiments/count_pairs.py). The loss is  -log pi(child | parent)  summed over the coefficients.
The checkpoint can start a reinforcement-learning run:  rlmath-reinforce --init <out>.

    uv run rlmath-sft --pairs pairs_p3.npz --out runs/sft.pt --require ge
"""

from dataclasses import asdict, dataclass
import logging
import math
import time

import numpy as np
import torch
import tyro

from rlmath.pairs import Pairs
from rlmath.polynomials import variable_permutations
from rlmath.reinforce import Agent

log = logging.getLogger("rlmath.sft")


@dataclass
class Args:
    pairs: str  # npz written by experiments/count_pairs.py
    out: str  # checkpoint to write
    p: int = 3
    n: int = 4
    d: int = 4
    require: str = "ge"  # which pairs to imitate: "ge" child >= parent, "gt" child > parent, "all"
    min_parent: int = 2  # ... with parents of at least this score
    max_parent: int = 99  # ... and at most this score
    epochs: int = 8
    batch_size: int = 1024
    learning_rate: float = 3e-4
    warmup_steps: int = 200
    holdout: float = 0.02  # share of parents kept out to measure the loss on
    augment: bool = True  # rename the variables at random (the same way for parent and child)
    single: bool = False  # train the single-edit policy (rlmath-reinforce --single); the pairs are single changes
    seed: int = 1
    cuda: bool = True
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512


def select(pairs: Pairs, args: Args) -> np.ndarray:
    keep = (pairs.parent_score >= args.min_parent) & (pairs.parent_score <= args.max_parent)
    if args.require == "ge":
        keep &= pairs.child_score >= pairs.parent_score
    elif args.require == "gt":
        keep &= pairs.child_score > pairs.parent_score
    elif args.require != "all":
        raise ValueError("require must be 'ge', 'gt' or 'all'")
    return np.flatnonzero(keep)


def main(args: Args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    pairs = Pairs.load(args.pairs)
    chosen = select(pairs, args)
    if len(chosen) == 0:
        raise ValueError("no pairs match")
    # hold out whole parents, so the held-out loss is on polynomials the model has not seen
    held_parents = rng.random(len(pairs.digits)) < args.holdout
    test = chosen[held_parents[pairs.parent[chosen]]]
    train = chosen[~held_parents[pairs.parent[chosen]]]
    log.info("%d pairs selected (%d train, %d held out); parent scores %s", len(chosen), len(train), len(test),
             {int(s): int((pairs.parent_score[chosen] == s).sum()) for s in np.unique(pairs.parent_score[chosen])})

    table = torch.as_tensor(pairs.digits, device=device)
    parent = torch.as_tensor(pairs.parent, device=device).long()
    pos = torch.as_tensor(pairs.pos, device=device).long()
    new = torch.as_tensor(pairs.new, device=device).long()
    perms = torch.as_tensor(variable_permutations(args.n, args.d), device=device)

    def batch(idx: torch.Tensor, augment: bool):
        obs = table[parent[idx]].long()
        child = obs.clone()
        child[torch.arange(len(idx), device=device), pos[idx]] = new[idx]
        if augment:
            sigma = perms[torch.randint(len(perms), (len(idx),), device=device)]
            obs, child = obs.gather(1, sigma), child.gather(1, sigma)
        return obs, child

    agent = Agent(args.p, args.n, args.d, args.d_model, args.n_layers, args.n_heads, args.d_ff, args.single).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.learning_rate, weight_decay=0.0)
    steps_per_epoch = max(1, len(train) // args.batch_size)
    total = steps_per_epoch * args.epochs

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.learning_rate * (step + 1) / args.warmup_steps
        return args.learning_rate * 0.5 * (1 + math.cos(math.pi * (step - args.warmup_steps) / max(1, total - args.warmup_steps)))

    def held_out_nll() -> tuple[float, float]:
        agent.eval()
        nll, mass_changed = [], []
        with torch.no_grad():
            for chunk in torch.as_tensor(test, device=device).split(4096):
                obs, child = batch(chunk, augment=False)
                nll.append(-agent.child_log_prob(obs, child))
                # probability the model gives to some coefficient changing at all (how "edit-like" it is)
                keep = agent(obs).probs.gather(-1, obs.unsqueeze(-1)).squeeze(-1).prod(-1)
                mass_changed.append(torch.ones_like(keep) if args.single else 1 - keep)
        return torch.cat(nll).mean().item(), torch.cat(mass_changed).mean().item()

    train_t = torch.as_tensor(train, device=device)
    step, start = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        agent.train()
        losses = []
        order = train_t[torch.randperm(len(train_t), device=device)]
        for idx in order[: steps_per_epoch * args.batch_size].split(args.batch_size):
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            obs, child = batch(idx, args.augment)
            loss = -agent.child_log_prob(obs, child).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            step += 1
        if len(test):
            nll, changed = held_out_nll()
            log.info("epoch %d/%d  train nll %.3f  held-out nll %.3f  P(some coefficient changes) %.3f  [%.0fs]",
                     epoch, args.epochs, np.mean(losses), nll, changed, time.time() - start)
        else:
            log.info("epoch %d/%d  train nll %.3f  [%.0fs]", epoch, args.epochs, np.mean(losses), time.time() - start)
    torch.save({"args": asdict(args), "model": agent.state_dict()}, args.out)
    log.info("saved %s", args.out)


def cli():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
