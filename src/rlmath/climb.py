"""Hill-climbing as a game: several single-coefficient edits in a row, with the oracle's answer after each one.

`ClimbEnv` starts every trajectory at a parent polynomial. Each step the agent changes one coefficient and the child is
scored. With `accept="always"` the trajectory moves to every child, whatever its height (free exploration); with
`"not_lower"` it moves only to children that are not lower than the current polynomial. A trajectory has `steps` steps and
costs one oracle call per step. Rewards (`--reward`): "improve": step t earns max(0, height_t - height of the parent), so a
walk 6 -> 4 -> 1 -> 5 -> 8 -> 4 -> 9 earns 0, 0, 0, 2, 0, 3; "best": the gain in the best height reached so far; "child":
child height minus the current height. With one step and "child" this is the one-edit problem of rlmath.reinforce.

A step whose polynomial (up to scaling) already occurred in the trajectory, the parent included, is a repeat: `info["repeat"]`
marks it, and the trainer gives it a large negative reward (`--repeat-penalty`) instead of its improvement.

`rlmath-climb` trains the single-edit agent with REINFORCE on this game. The agent is told the current height and the
step number. The advantage of a step is the return still to come after it (final best minus best before it), minus the
mean of that over the trajectories that share the parent (same step), divided by the batch standard deviation.

    uv run rlmath-climb --p 3 --steps 8 --total-timesteps 1500000 --init sft.pt
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.distributions import Categorical
import tyro

from rlmath.env import PolyEnv
from rlmath.polynomials import normalize
from rlmath.reinforce import Agent, Args as ReinforceArgs, policy_kl, update_curriculum

log = logging.getLogger("rlmath.climb")


class ClimbEnv(PolyEnv):
    """Vector env of hill-climbing trajectories; `children_per_parent` trajectories share each parent.

    `reset` draws the parents; `step(children)` scores one child per trajectory. There is no autoreset: after
    `steps` steps call `reset` again. With `walk=False` a trajectory never moves, so every step is an independent
    child of the parent (best-of-k sampling).
    """

    def __init__(self, *args, steps: int = 8, walk: bool = True, accept: str = "always", **kwargs):
        super().__init__(*args, **kwargs)
        if accept not in ("always", "not_lower"):
            raise ValueError("accept must be 'always' or 'not_lower'")
        self.steps, self.walk, self.accept = steps, walk, accept

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.parents, self.parent_scores, self.from_pool = self._sample_parents()
        self.current, self.current_score = self.parents.copy(), self.parent_scores.copy()
        self.best, self.t = self.parent_scores.copy(), 0
        self.visited = [{row.tobytes()} for row in normalize(self.parents, self.p).astype(np.uint8)]
        return self.current.copy(), {}

    def step(self, actions: np.ndarray):
        child = np.asarray(actions, dtype=np.int64)
        child_score = self._score(child)
        self._remember(child, child_score)
        before, best_before = self.current_score.copy(), self.best.copy()
        repeat = np.zeros(self.num_envs, dtype=bool)
        for i, row in enumerate(normalize(child, self.p).astype(np.uint8)):
            key = row.tobytes()
            repeat[i] = key in self.visited[i]
            self.visited[i].add(key)
        accepted = child_score >= self.current_score if self.accept == "not_lower" else np.ones(self.num_envs, dtype=bool)
        if self.walk:
            self.current = np.where(accepted[:, None], child, self.current)
            self.current_score = np.where(accepted, child_score, self.current_score)
        self.best = np.maximum(self.best, child_score)
        self.t += 1
        info = {
            "child_score": child_score, "score_before": before, "best_before": best_before, "best": self.best.copy(),
            "accepted": accepted, "parent_score": self.parent_scores, "from_pool": self.from_pool, "repeat": repeat,
        }
        done = np.full(self.num_envs, self.t >= self.steps)
        reward = (self.best - best_before).astype(np.float32)
        return self.current.copy(), reward, done, np.zeros(self.num_envs, dtype=bool), info


def random_flips(rng: np.random.Generator, obs: np.ndarray, p: int) -> np.ndarray:
    """Change one random coefficient of every row to a different value."""
    child = obs.copy()
    pos = rng.integers(0, obs.shape[1], size=len(obs))
    rows = np.arange(len(obs))
    child[rows, pos] = (child[rows, pos] + rng.integers(1, p, size=len(obs))) % p
    return child


def play(env: ClimbEnv, propose) -> dict:
    """Play `env.steps` steps. propose(obs, score, t) -> children. Returns per-step arrays (steps, N) of what happened."""
    obs = env.current.copy()
    out = {k: [] for k in ("obs", "score", "child", "child_score", "accepted", "best_before", "best", "repeat")}
    for t in range(env.steps):
        score = env.current_score.copy()
        child = propose(obs, score, t)
        next_obs, _, _, _, info = env.step(child)
        for k, v in (("obs", obs), ("score", score), ("child", child), ("child_score", info["child_score"]),
                     ("accepted", info["accepted"]), ("best_before", info["best_before"]), ("best", info["best"]),
                     ("repeat", info["repeat"])):
            out[k].append(v)
        obs = next_obs
    return {k: np.stack(v) for k, v in out.items()}


def policy_proposer(agent: Agent, device):
    """Sample from the agent (given the current height and step number if it uses them)."""

    def propose(obs: np.ndarray, score: np.ndarray, t: int) -> np.ndarray:
        with torch.no_grad():
            ctx = (torch.as_tensor(score, device=device).long(), torch.full((len(obs),), t, device=device))
            return agent.get_action(torch.as_tensor(obs, device=device), ctx)[0].cpu().numpy()

    return propose


def no_replacement_proposer(agent: Agent | None, device, p: int):
    """Like `policy_proposer` (uniform over single flips if `agent` is None), but never proposes a polynomial that already
    occurred in the trajectory (the parent, earlier children, the polynomial the trajectory is standing on), up to scaling.

    Make a new one for every game: it remembers the polynomials seen so far.
    """
    visited: list[set] | None = None

    def codes(x: np.ndarray) -> np.ndarray:  # base-p integer of each row, scaled to start with a 1
        return (normalize(x, p).astype(np.int64) * p ** np.arange(x.shape[1], dtype=np.int64)).sum(axis=1)

    def propose(obs: np.ndarray, score: np.ndarray, t: int) -> np.ndarray:
        nonlocal visited
        n, m = obs.shape
        if visited is None:
            visited = [set() for _ in range(n)]
        for v, c in zip(visited, codes(obs)):
            v.add(int(c))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=device)
            if agent is None:
                logits = torch.zeros(n, m, p, device=device)
            else:
                ctx = (torch.as_tensor(score, device=device).long(), torch.full((n,), t, device=device))
                logits = agent(obs_t, ctx).logits
            logits = logits.masked_fill(torch.nn.functional.one_hot(obs_t, p).bool(), float("-inf")).reshape(n, m * p)
        # the polynomial reached by each of the m * p (position, value) choices
        cand = np.repeat(obs[:, None, :], m * p, axis=1)
        pos, val = np.divmod(np.arange(m * p), p)
        cand[:, np.arange(m * p), pos] = val
        cand_codes = codes(cand.reshape(n * m * p, m)).reshape(n, m * p)
        seen = np.array([[c in v for c in row] for row, v in zip(cand_codes.tolist(), visited)])
        logits = logits.masked_fill(torch.as_tensor(seen, device=device), float("-inf"))
        pick = Categorical(logits=logits).sample().cpu().numpy()
        child = obs.copy()
        child[np.arange(n), pick // p] = pick % p
        for v, c in zip(visited, codes(child)):
            v.add(int(c))
        return child

    return propose


@dataclass
class Args(ReinforceArgs):
    exp_name: str = "climb"
    steps: int = 8  # edits per trajectory (= oracle calls per trajectory)
    reward: str = "improve"  # "improve": max(0, child - parent) per step; "best": gain in best height; "child": child - current
    repeat_penalty: float = 500.0  # reward of a step whose polynomial already occurred in its trajectory (it replaces the improvement)
    repeat_clip: float = 10.0  # ... but in standard-deviation units its advantage is at least -this, so it cannot swamp the batch
    max_hours: float = 0.0  # stop after this much wall-clock time (0 = run all iterations); the learning rate decays over it
    snapshot_every: int = 0  # also keep a weights-only copy model_<iteration>.pt every this many iterations
    resume: str = ""  # continue from a model.pt written by this script (weights, optimizer and clock)
    accept: str = "always"  # "always": the trajectory moves to every child; "not_lower": only to children not lower than it
    single: bool = True
    context: bool = True  # tell the agent the current height and the step number
    num_envs: int = 256  # trajectories per iteration
    children_per_parent: int = 32  # trajectories that share a parent
    hamming_penalty: float = 0.0
    copy_penalty: float = 0.0


def main(args: Args):
    if not args.single:
        raise ValueError("the game needs single-coefficient edits")
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

    envs = ClimbEnv(
        args.num_envs, args.p, args.n, args.d, cutoff=args.cutoff, infinite_score=args.infinite_score, urls=args.urls,
        db=args.db, parent_db_frac=args.parent_db_frac, parent_min_height=args.parent_min_height,
        parent_db_min_size=args.parent_db_min_size, parent_temperature=args.parent_temperature,
        parent_max_height=args.curriculum_start or None, children_per_parent=args.children_per_parent, seed=args.seed,
        steps=args.steps, accept=args.accept,
    )
    agent = Agent(args.p, args.n, args.d, args.d_model, args.n_layers, args.n_heads, args.d_ff, True, args.context).to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location=device)
        for k in ("p", "n", "d", "d_model", "n_layers", "n_heads", "d_ff"):
            if ckpt["args"].get(k) != getattr(args, k):
                raise ValueError(f"--init was trained with {k}={ckpt['args'].get(k)}, this run has {k}={getattr(args, k)}")
        if not ckpt["args"].get("single", False):
            raise ValueError("--init must be a single-edit checkpoint")
        agent.load_state_dict(ckpt["model"], strict=False)  # a checkpoint without the context embeddings is fine
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    per_iteration = args.num_envs * args.steps  # oracle calls
    num_iterations = args.total_timesteps // per_iteration
    log.info("writing to %s; device %s; %d height servers; %d iterations", out, device, len(args.urls), num_iterations)

    progress = {"iteration": 0, "elapsed": 0.0, "ceiling": args.curriculum_start}

    def save():
        torch.save({"args": asdict(args), "model": agent.state_dict(), "optimizer": optimizer.state_dict(), **progress}, out / "model.pt")

    start_iteration, offset = 1, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration, offset = ckpt["iteration"] + 1, ckpt["elapsed"]
        args.curriculum_start = ckpt.get("ceiling", args.curriculum_start)
        log.info("resuming from iteration %d after %.1f hours", ckpt["iteration"], offset / 3600)
    ceiling, successes, best_seen, start = args.curriculum_start, [], 0, time.time() - offset
    envs.parent_max_height = ceiling or None
    propose = policy_proposer(agent, device)
    try:
        for iteration in range(start_iteration, num_iterations + 1):
            elapsed = time.time() - start
            if args.max_hours and elapsed >= args.max_hours * 3600:
                log.info("time limit reached after %d iterations", iteration - 1)
                break
            if args.anneal_lr:
                done = max((iteration - 1.0) / num_iterations, elapsed / (args.max_hours * 3600) if args.max_hours else 0.0)
                optimizer.param_groups[0]["lr"] = (1.0 - done) * args.learning_rate
            progress["iteration"], progress["elapsed"] = iteration - 1, elapsed
            envs.reset()
            game = play(envs, propose)
            steps, n = game["child"].shape[:2]
            if args.reward == "best":
                step_reward = game["best"] - game["best_before"]
            elif args.reward == "improve":
                step_reward = np.maximum(game["child_score"] - envs.parent_scores[None], 0)
            elif args.reward == "child":
                step_reward = game["child_score"] - game["score"]
            else:
                raise ValueError("reward must be 'improve', 'best' or 'child'")
            repeat = game["repeat"]
            step_reward = np.where(repeat, 0.0, step_reward)  # a repeat earns no improvement ...
            to_go = np.cumsum(step_reward[::-1], axis=0)[::-1].astype(np.float32)  # return from each step on

            ret = torch.as_tensor(to_go.copy(), device=device)  # (steps, N)
            adv = ret.view(steps, -1, args.children_per_parent)
            mean = adv.mean(dim=2, keepdim=True)
            adv = (adv - mean).reshape(steps, n)
            rep = torch.as_tensor(repeat, device=device)
            scale = (adv[~rep].std() + 1e-8) if args.norm_adv and (~rep).any() else 1.0
            # ... and is punished on its own step only (earlier steps are not blamed for it)
            penalty = ((-args.repeat_penalty - mean.reshape(steps, -1).repeat_interleave(args.children_per_parent, dim=1)) / scale).clamp(min=-args.repeat_clip)
            adv = torch.where(rep, penalty, adv / scale)

            obs = torch.as_tensor(game["obs"], device=device).reshape(steps * n, -1)
            child = torch.as_tensor(game["child"], device=device).reshape(steps * n, -1)
            height = torch.as_tensor(game["score"], device=device).long().reshape(-1)
            t = torch.arange(steps, device=device).repeat_interleave(n)
            dist = agent.policy(obs, (height, t))
            pos = (child != obs).float().argmax(dim=1)  # the one coefficient that changed
            logprob = dist.log_prob(pos * args.p + child[torch.arange(len(obs), device=device), pos])
            entropy = -(dist.probs * dist.logits.clamp(min=-1e4)).sum(-1)
            pg_loss = -(adv.reshape(-1) * logprob).mean()
            loss = pg_loss - args.ent_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                kl = policy_kl(Categorical(logits=dist.logits.detach()), agent.policy(obs, (height, t)))

            parent = envs.parent_scores
            gain = game["best"][-1] - parent
            success = gain > 0
            pool = envs.from_pool
            if ceiling and pool.any():
                ceiling = update_curriculum(
                    ceiling, successes, float(success[pool].mean()), args.curriculum_window, args.curriculum_threshold, args.cutoff
                )
                envs.parent_max_height = ceiling
            best_seen = max(best_seen, int(game["best"].max()))
            global_step = iteration * per_iteration
            group_hit = success.reshape(-1, args.children_per_parent).any(axis=1).mean()
            metrics = {
                "charts/reward": float(gain.mean()),  # best reached - parent
                "charts/return": float(step_reward.sum(axis=0).mean()),  # what the chosen reward adds up to
                "charts/repeat": float(repeat.mean()),
                "charts/improved_steps": float((game["child_score"] > envs.parent_scores[None]).mean()),
                "charts/success": float(success.mean()),
                "charts/success_step1": float((game["best"][0] > parent).mean()),
                "charts/group_hit_rate": float(group_hit),
                "charts/final_score": float(game["best"][-1].mean()),
                "charts/parent_score": float(parent.mean()),
                "charts/accepted": float(game["accepted"].mean()),
                "charts/parent_ceiling": ceiling,
                "charts/best_score": best_seen,
                "charts/oracle_calls": envs.oracle.calls,
                "charts/parent_pool": len(envs.pool),
                "charts/SPS": int(global_step / (time.time() - start)),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy.mean().item(),
                "losses/grad_norm": float(grad_norm),
                "losses/kl_step": kl.mean().item(),
            }
            if args.track:
                wandb.log(metrics, step=global_step)
            log.info(
                "iter %d/%d  step %d  gain %+.3f  repeats %.1f%%  success %.1f%% (step 1: %.1f%%)  final %.2f (parent %.2f)  accepted %.0f%%  "
                "best %d  pool %d  calls %d  SPS %d  kl %.4f  ceil %d  hit %.0f%%",
                iteration, num_iterations, global_step, metrics["charts/reward"], 100 * metrics["charts/repeat"], 100 * metrics["charts/success"],
                100 * metrics["charts/success_step1"], metrics["charts/final_score"], metrics["charts/parent_score"],
                100 * metrics["charts/accepted"], best_seen, len(envs.pool), metrics["charts/oracle_calls"],
                metrics["charts/SPS"], metrics["losses/kl_step"], ceiling, 100 * group_hit,
            )
            progress["iteration"], progress["elapsed"], progress["ceiling"] = iteration, time.time() - start, ceiling
            if iteration % args.save_every == 0:
                save()
            if args.snapshot_every and iteration % args.snapshot_every == 0:
                torch.save({"args": asdict(args), "model": agent.state_dict()}, out / f"model_{iteration}.pt")  # weights only, to evaluate
    finally:
        progress["elapsed"] = time.time() - start
        save()
        envs.close()
        if args.track:
            wandb.finish()


def cli():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
