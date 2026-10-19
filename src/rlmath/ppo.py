"""PPO for the polynomial-editing environment, after CleanRL's ppo.py.

The agent reads a parent polynomial and writes a child: coefficients are embedded, encoded by a
transformer, and projected to p logits per coefficient. Each coefficient is sampled from its
softmax (the 35 coefficients are independent given the parent), the log-probability is their sum,
and the reward is score(child) - score(parent) from `rlmath.env.PolyEnv`.

Episodes have one step, so there is no bootstrapping: the return is the reward and the advantage
is reward - value(parent).

    uv run rlmath-ppo --p 5 --total-timesteps 1000000 --track
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import tyro

from rlmath.env import PolyEnv
from rlmath.polynomials import monomials

log = logging.getLogger("rlmath.ppo")


@dataclass
class Args:
    exp_name: str = "ppo"
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
    workers: int = 0  # Julia worker processes (0 = number of CPUs - 1)
    db: str = "runs/heights.sqlite"  # every scored polynomial is saved here ("" to disable)
    parent_db_frac: float = 0.0  # fraction of parents drawn from the polynomials saved in db
    parent_min_height: int = 2  # ... that have at least this height

    # algorithm
    total_timesteps: int = 1_000_000  # children scored
    num_envs: int = 1024  # children scored per iteration (one parent each)
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 4
    num_minibatches: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.0  # entropy is summed over the coefficients, as in CleanRL's continuous PPO
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # model
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512


class Agent(nn.Module):
    def __init__(self, p: int, n: int, d: int, d_model: int, n_layers: int, n_heads: int, d_ff: int):
        super().__init__()
        self.register_buffer("exponents", torch.as_tensor(monomials(n, d)), persistent=False)
        self.coeff_emb = nn.Embedding(p, d_model)
        # One table per variable, indexed by exponent; a monomial's embedding is the sum over its variables.
        self.exp_emb = nn.ModuleList(nn.Embedding(d + 1, d_model) for _ in range(n))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )  # no dropout: PPO needs the policy to be the same function when it is sampled and when it is updated
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.actor = nn.Linear(d_model, p)
        self.critic = nn.Linear(d_model, 1)
        nn.init.orthogonal_(self.actor.weight, 0.01)  # start close to uniform, as CleanRL does
        nn.init.zeros_(self.actor.bias)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, M) coefficients in 0..p-1. Returns (B, M, d_model)."""
        mono = sum(emb(self.exponents[:, j]) for j, emb in enumerate(self.exp_emb))
        return self.norm(self.encoder(self.coeff_emb(obs) + mono))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encode(obs).mean(dim=1)).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        h = self.encode(obs)
        dist = Categorical(logits=self.actor(h))
        if action is None:
            action = dist.sample()
        value = self.critic(h.mean(dim=1)).squeeze(-1)
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value


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
        args.num_envs, args.p, args.n, args.d, args.cutoff, args.infinite_score, args.workers,
        args.db, args.parent_db_frac, args.parent_min_height, args.seed,
    )
    agent = Agent(args.p, args.n, args.d, args.d_model, args.n_layers, args.n_heads, args.d_ff).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    num_iterations = args.total_timesteps // args.num_envs
    minibatch_size = args.num_envs // args.num_minibatches
    log.info("writing to %s; device %s; %d Julia workers", out, device, envs.oracle.n_workers)

    def save():
        torch.save({"args": asdict(args), "model": agent.state_dict()}, out / "model.pt")

    best = 0
    start = time.time()
    obs, _ = envs.reset(seed=args.seed)
    try:
        for iteration in range(1, num_iterations + 1):
            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / num_iterations) * args.learning_rate

            # One step of every episode: sample a child for each parent and score it.
            obs_t = torch.as_tensor(obs, device=device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            next_obs, reward, _, _, info = envs.step(action.cpu().numpy())
            returns = torch.as_tensor(reward, device=device)
            advantages = returns - value

            # PPO update on this batch.
            clipfracs = []
            for _ in range(args.update_epochs):
                for mb in torch.randperm(args.num_envs, device=device).split(minibatch_size):
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs_t[mb], action[mb])
                    logratio = newlogprob - logprob[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                    mb_adv = advantages[mb]
                    if args.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                    v_loss = 0.5 * ((newvalue - returns[mb]) ** 2).mean()
                    loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            child, parent = info["child_score"], info["parent_score"]
            best = max(best, int(child.max()))
            global_step = iteration * args.num_envs
            metrics = {
                "charts/reward": float(reward.mean()),
                "charts/improved": float((child > parent).mean()),
                "charts/worse": float((child < parent).mean()),
                "charts/copy_fraction": float((action.cpu().numpy() == obs).mean()),
                "charts/child_score": float(child.mean()),
                "charts/parent_score": float(parent.mean()),
                "charts/best_child_score": best,
                "charts/oracle_calls": envs.oracle.calls,
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/SPS": int(global_step / (time.time() - start)),
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy.mean().item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": float(np.mean(clipfracs)),
            }
            if args.track:
                wandb.log(metrics, step=global_step)
            log.info(
                "iter %d/%d  step %d  reward %+.3f  improved %.1f%%  worse %.1f%%  copy %.2f  best %d  calls %d  SPS %d",
                iteration, num_iterations, global_step, metrics["charts/reward"], 100 * metrics["charts/improved"],
                100 * metrics["charts/worse"], metrics["charts/copy_fraction"], best, metrics["charts/oracle_calls"],
                metrics["charts/SPS"],
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
