"""REINFORCE for the polynomial-editing environment, laid out like a CleanRL script.

The agent reads a parent polynomial and writes a child: coefficients are embedded, encoded by a
transformer, and projected to p logits per coefficient. Each coefficient is sampled from its
softmax (the 35 coefficients are independent given the parent) and the log-probability of the child
is their sum. `rlmath.env.PolyEnv` returns the reward.

Episodes have one step, so the gradient is just  -(reward - baseline) * log pi(child | parent),
averaged over the batch, with one update per batch of fresh samples.

Each parent gets `children_per_parent` independent children, and the baseline is the mean reward of
that group, so a child is judged against other edits of the same parent. The reward contains
-score(parent), which is the same for the whole group and cancels. (With one child per parent the
baseline is the batch mean of the reward plus score(parent), which removes the same term.)

Curriculum: pool parents are capped at a height ceiling that starts at `curriculum_start` and rises by
one whenever the share of children that are at least as high as their parent, averaged over the last
`curriculum_window` iterations, reaches `curriculum_threshold`.

    uv run rlmath-reinforce --p 5 --total-timesteps 1000000 --track
"""

import copy
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import tyro

from rlmath.env import PolyEnv
from rlmath.polynomials import monomials

log = logging.getLogger("rlmath.reinforce")


@dataclass
class Args:
    exp_name: str = "reinforce"
    seed: int = 1
    cuda: bool = True
    track: bool = False  # log to Weights and Biases
    wandb_project_name: str = "rlmath"
    wandb_entity: str | None = None
    out: str = ""  # where model.pt goes (default runs/<run name>)
    save_every: int = 50  # iterations between checkpoints

    # environment
    p: int = 5
    n: int = 4
    d: int = 4
    cutoff: int = 10  # quartic K3 surfaces: finite heights are at most 10
    infinite_score: int = 1  # score of supersingular polynomials; 1 = not rewarded
    hamming_free: int = 8  # the child may change this many coefficients of the parent without penalty
    hamming_penalty: float = 0.5  # reward lost per coefficient changed beyond hamming_free (to be tuned)
    copy_penalty: float = 100.0  # reward lost when the child is exactly the parent
    workers: int = 0  # Julia worker processes (0 = number of CPUs - 1)
    db: str = "runs/heights.sqlite"  # every scored polynomial is saved here ("" to disable)
    parent_db_frac: float = 0.5  # fraction of parents drawn from the polynomials saved in db
    parent_min_height: int = 2  # ... that have at least this height
    parent_db_min_size: int = 1000  # parents are all random until the pool holds this many polynomials
    parent_temperature: float = 3.0  # pool parents: P(height level) ~ exp(height / this); lower favours higher
    curriculum_start: int = 2  # initial ceiling on the height of pool parents (0 = no curriculum)
    curriculum_threshold: float = 0.25  # raise the ceiling when this share of children are >= their parent
    curriculum_window: int = 20  # ... averaged over this many iterations

    # algorithm
    total_timesteps: int = 1_000_000  # children scored
    num_envs: int = 1024  # children scored per iteration
    children_per_parent: int = 128  # children sampled for each parent (num_envs / this parents per iteration)
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    norm_adv: bool = True  # divide the advantages by their standard deviation
    ent_coef: float = 0.0  # entropy is summed over the coefficients
    max_grad_norm: float = 0.5

    # policy class
    single: bool = False  # change exactly one coefficient: sample one (position, new value) pair from a single softmax

    # starting point
    init: str = ""  # checkpoint to start from (a model.pt, or one written by rlmath-sft); its model size must match
    kl_coef: float = 0.0  # penalty on KL(policy || the policy at the start), summed over coefficients; needs init

    # model
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512


class Agent(nn.Module):
    """Parent in, child out. By default the child is sampled coefficient by coefficient; with `single` it differs
    from the parent in exactly one coefficient, chosen with one softmax over the (position, new value) pairs."""

    def __init__(
        self, p: int, n: int, d: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, single: bool = False
    ):
        super().__init__()
        self.p, self.single = p, single
        self.register_buffer("exponents", torch.as_tensor(monomials(n, d)), persistent=False)
        self.coeff_emb = nn.Embedding(p, d_model)
        # One table per variable, indexed by exponent; a monomial's embedding is the sum over its variables.
        self.exp_emb = nn.ModuleList(nn.Embedding(d + 1, d_model) for _ in range(n))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.actor = nn.Linear(d_model, p)
        nn.init.orthogonal_(self.actor.weight, 0.01)  # start close to uniform
        nn.init.zeros_(self.actor.bias)

    def forward(self, obs: torch.Tensor) -> Categorical:
        """obs: (B, M) coefficients in 0..p-1. Returns an independent categorical per coefficient."""
        mono = sum(emb(self.exponents[:, j]) for j, emb in enumerate(self.exp_emb))
        h = self.norm(self.encoder(self.coeff_emb(obs) + mono))
        return Categorical(logits=self.actor(h))

    def policy(self, obs: torch.Tensor) -> Categorical:
        """The distribution the child is drawn from: per coefficient, or (with `single`) one over all
        (position, new value) pairs, where position * p + value indexes the pair and keeping a value is not allowed."""
        dist = self(obs)
        if not self.single:
            return dist
        logits = dist.logits.masked_fill(F.one_hot(obs, self.p).bool(), float("-inf"))
        return Categorical(logits=logits.reshape(len(obs), -1))

    def get_action(self, obs: torch.Tensor):
        dist = self.policy(obs)
        sample = dist.sample()
        if not self.single:
            return sample, dist.log_prob(sample).sum(-1), dist.entropy().sum(-1), dist
        child = obs.clone()
        child[torch.arange(len(obs), device=obs.device), sample // self.p] = sample % self.p
        # entropy() would multiply the masked -inf logits by their zero probability, which gives NaN gradients
        entropy = -(dist.probs * dist.logits.clamp(min=-1e4)).sum(-1)
        return child, dist.log_prob(sample), entropy, dist

    def child_log_prob(self, obs: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
        """Log-probability of `child` given `obs`. With `single`, child must differ from obs in exactly one coefficient."""
        dist = self.policy(obs)
        if not self.single:
            return dist.log_prob(child).sum(-1)
        changed = child != obs
        pos = changed.float().argmax(dim=1)
        return dist.log_prob(pos * self.p + child[torch.arange(len(obs), device=obs.device), pos])


def policy_kl(p: Categorical, q: Categorical) -> torch.Tensor:
    """KL(p || q) per parent: summed over the coefficients when the distributions are per coefficient."""
    # not torch's kl_divergence: where both distributions rule a value out it gives NaN gradients
    kl = (p.probs * (p.logits.clamp(min=-1e4) - q.logits.clamp(min=-1e4))).sum(-1)
    return kl.sum(-1) if kl.dim() > 1 else kl


def group_advantages(reward: torch.Tensor, group: int, norm: bool) -> torch.Tensor:
    """Reward minus the mean reward of its group (consecutive rows), divided by the batch std if `norm`.

    With a group of one child there is nothing to compare against, so the batch mean is used instead.
    """
    adv = reward.view(-1, group)
    adv = (adv - adv.mean(dim=1, keepdim=True)).reshape(-1) if group > 1 else (reward - reward.mean())
    return adv / (adv.std() + 1e-8) if norm else adv


def update_curriculum(ceiling: int, successes: list, success: float, window: int, threshold: float, cutoff: int) -> int:
    """Record this iteration's success rate; raise the ceiling by one when the last `window` average reaches `threshold`."""
    successes.append(success)
    if len(successes) >= window and np.mean(successes[-window:]) >= threshold and ceiling < cutoff:
        successes.clear()
        return ceiling + 1
    return ceiling


def main(args: Args):
    run_name = f"p{args.p}__{args.exp_name}__{args.seed}__{int(time.time())}"
    out = Path(args.out or f"runs/{run_name}")
    out.mkdir(parents=True, exist_ok=True)
    if args.track:
        import wandb

        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=asdict(args), name=run_name, save_code=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = PolyEnv(
        args.num_envs, args.p, args.n, args.d, cutoff=args.cutoff, infinite_score=args.infinite_score,
        hamming_penalty=args.hamming_penalty, hamming_free=args.hamming_free, copy_penalty=args.copy_penalty, workers=args.workers, db=args.db,
        parent_db_frac=args.parent_db_frac, parent_min_height=args.parent_min_height,
        parent_db_min_size=args.parent_db_min_size, parent_temperature=args.parent_temperature,
        parent_max_height=args.curriculum_start or None, children_per_parent=args.children_per_parent, seed=args.seed,
    )
    agent = Agent(args.p, args.n, args.d, args.d_model, args.n_layers, args.n_heads, args.d_ff, args.single).to(device)
    ref = None
    if args.init:
        ckpt = torch.load(args.init, map_location=device)
        for k in ("p", "n", "d", "d_model", "n_layers", "n_heads", "d_ff", "single"):
            if ckpt["args"].get(k, False if k == "single" else None) != getattr(args, k):
                raise ValueError(f"--init was trained with {k}={ckpt['args'].get(k)}, this run has {k}={getattr(args, k)}")
        agent.load_state_dict(ckpt["model"])
        if args.kl_coef:
            ref = copy.deepcopy(agent).requires_grad_(False)
    elif args.kl_coef:
        raise ValueError("--kl-coef needs --init")
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    num_iterations = args.total_timesteps // args.num_envs
    log.info("writing to %s; device %s; %d Julia workers", out, device, envs.oracle.n_workers)

    def save():
        torch.save({"args": asdict(args), "model": agent.state_dict()}, out / "model.pt")

    best = 0
    ceiling = args.curriculum_start  # 0 = no curriculum
    successes = []
    start = time.time()
    obs, _ = envs.reset(seed=args.seed)
    try:
        for iteration in range(1, num_iterations + 1):
            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / num_iterations) * args.learning_rate

            # One step of every episode: sample a child for each parent and score it.
            obs_t = torch.as_tensor(obs, device=device)
            action, logprob, entropy, dist = agent.get_action(obs_t)
            next_obs, reward, _, _, info = envs.step(action.cpu().numpy())
            child, parent = info["child_score"], info["parent_score"]

            # REINFORCE with the group baseline described at the top of the file.
            shaped = torch.as_tensor(reward + parent, dtype=torch.float32, device=device)
            advantages = group_advantages(shaped, args.children_per_parent, args.norm_adv)
            pg_loss = -(advantages * logprob).mean()
            loss = pg_loss - args.ent_coef * entropy.mean()
            kl_ref = torch.zeros((), device=device)
            if ref is not None:
                with torch.no_grad():
                    ref_dist = ref.policy(obs_t)
                kl_ref = policy_kl(dist, ref_dist).mean()
                loss = loss + args.kl_coef * kl_ref

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            # How far this one step moved the policy on the batch it was computed from: the exact KL
            # between the policy before and after, summed over the coefficients, per parent.
            with torch.no_grad():
                kl = policy_kl(Categorical(logits=dist.logits.detach()), agent.policy(obs_t))

            if ceiling and info["from_pool"].any():
                ceiling = update_curriculum(
                    ceiling, successes, float((child >= parent)[info["from_pool"]].mean()),
                    args.curriculum_window, args.curriculum_threshold, args.cutoff,
                )
                envs.parent_max_height = ceiling

            best = max(best, int(child.max()))
            hit = float((child > parent).reshape(-1, args.children_per_parent).any(axis=1).mean())
            global_step = iteration * args.num_envs
            metrics = {
                "charts/reward": float(reward.mean()),
                "charts/improvement": float((child - parent).mean()),  # reward without the penalties
                "charts/improved": float((child > parent).mean()),
                "charts/worse": float((child < parent).mean()),
                "charts/hamming": float(info["hamming"].mean()),
                "charts/copied": float((info["hamming"] == 0).mean()),
                "charts/group_hit_rate": hit,  # share of parents for which some child beat the parent
                "charts/parent_ceiling": ceiling,
                "charts/child_score": float(child.mean()),
                "charts/parent_score": float(parent.mean()),
                "charts/best_child_score": best,
                "charts/oracle_calls": envs.oracle.calls,
                "charts/parent_pool": len(envs.pool),
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/SPS": int(global_step / (time.time() - start)),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy.mean().item(),
                "losses/grad_norm": float(grad_norm),
                "losses/kl_step": kl.mean().item(),
                "losses/kl_step_max": kl.max().item(),
                "losses/kl_to_init": kl_ref.item(),
            }
            # The same numbers for parents drawn from the pool and for uniformly random ones.
            split = ""
            for name, mask in (("pool", info["from_pool"]), ("random", ~info["from_pool"])):
                if mask.any():
                    gain = (child - parent)[mask]
                    metrics.update({
                        f"{name}/fraction": float(mask.mean()),
                        f"{name}/improvement": float(gain.mean()),
                        f"{name}/improved": float((gain > 0).mean()),
                        f"{name}/worse": float((gain < 0).mean()),
                        f"{name}/parent_score": float(parent[mask].mean()),
                        f"{name}/hamming": float(info["hamming"][mask].mean()),
                    })
                    split += f"  {name} {gain.mean():+.2f} ({100 * (gain > 0).mean():.0f}% up, n {mask.sum()})"
            if args.track:
                wandb.log(metrics, step=global_step)
            log.info(
                "iter %d/%d  step %d  reward %+.3f  improvement %+.3f  child %.2f (parent %.2f)  improved %.1f%%  worse %.1f%%  "
                "hamming %.1f  copied %.1f%%  best %d  pool %d  calls %d  SPS %d  kl %.4f  ceil %d  hit %.0f%%%s",
                iteration, num_iterations, global_step, metrics["charts/reward"], metrics["charts/improvement"],
                metrics["charts/child_score"],
                metrics["charts/parent_score"], 100 * metrics["charts/improved"], 100 * metrics["charts/worse"],
                metrics["charts/hamming"], 100 * metrics["charts/copied"], best, len(envs.pool), metrics["charts/oracle_calls"],
                metrics["charts/SPS"], metrics["losses/kl_step"], ceiling, 100 * hit, split,
            )
            if iteration % args.save_every == 0:
                save()
            obs = next_obs
    finally:
        save()
        envs.close()
        if args.track:
            wandb.finish()


def cli():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
