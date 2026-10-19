"""Gymnasium environment: rewrite a polynomial, get rewarded for raising its height.

An episode is one step. The observation is a parent polynomial (its coefficient vector over F_p),
the action is a child polynomial (a full coefficient vector), and the reward is
score(child) - score(parent) - hamming_penalty * (number of coefficients where they differ)
- copy_penalty if the child is exactly the parent. The score is the quasi-F-split height of a smooth
polynomial; a singular or zero polynomial scores 0, so it is never worth producing.

`PolyEnv` is a vector env: its `num_envs` episodes run as one batch, scored with one call to the
height oracle. Every episode ends after its single step, so `step` returns the next episodes'
parents as the observation, and `reset` is only needed to start (and to reseed).

Parents are uniformly random polynomials, except for a fraction `parent_db_frac` that are drawn
from a pool of smooth polynomials of height at least `parent_min_height` (at least 2, the lowest
height that is checked for smoothness). The pool starts with the ones already saved in the SQLite
file `db` (see rlmath.store), read once when the environment is created, and grows with every such
polynomial the environment scores. Parents come from the pool only once it holds at least
`parent_db_min_size` polynomials; until then they are all random.
"""

import logging
import sqlite3

import gymnasium as gym
from gymnasium.vector.utils import batch_space
import numpy as np

from rlmath.oracle import HeightOracle, HeightResult
from rlmath.polynomials import monomials, normalize, random_polys

log = logging.getLogger(__name__)


def score(result: HeightResult, infinite_score: int, cutoff: int, past_cutoff_is_infinite: bool) -> int:
    """Turn an oracle result into a number (before the smoothness check).

    Finite height h scores h. Infinite height (supersingular) scores `infinite_score`. Invalid input
    and errors score 0.
    """
    if result.status == "ok":
        return result.height
    if result.status == "inf" or (result.status == "past_cutoff" and past_cutoff_is_infinite):
        return infinite_score
    if result.status == "past_cutoff":
        return cutoff + 1
    return 0


class ParentPool:
    """Distinct polynomials of known score, kept in memory for sampling parents."""

    def __init__(self, m: int):
        self.coeffs = np.empty((1024, m), dtype=np.uint8)
        self.scores = np.empty(1024, dtype=np.int64)
        self.keys: set[bytes] = set()

    def __len__(self) -> int:
        return len(self.keys)

    def add(self, coeffs: np.ndarray, scores: np.ndarray, p: int):
        # Scaling a polynomial doesn't change it, so store each scaled to start with a 1 (as the database does).
        for row, s in zip(normalize(coeffs, p).astype(np.uint8), scores):
            k = row.tobytes()
            if k in self.keys:
                continue
            n = len(self.keys)
            if n == len(self.coeffs):
                self.coeffs = np.concatenate([self.coeffs, np.empty_like(self.coeffs)])
                self.scores = np.concatenate([self.scores, np.empty_like(self.scores)])
            self.coeffs[n], self.scores[n] = row, s
            self.keys.add(k)

    def sample(self, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
        pick = rng.integers(len(self), size=count)
        return self.coeffs[pick].astype(np.int64), self.scores[pick]


class PolyEnv(gym.vector.VectorEnv):
    def __init__(
        self,
        num_envs: int,
        p: int = 5,
        n: int = 4,
        d: int = 4,
        cutoff: int = 10,
        infinite_score: int = 1,  # score of supersingular (infinite-height) polynomials; 1 = not rewarded
        hamming_penalty: float = 0.0,  # reward lost per coefficient where the child differs from the parent
        copy_penalty: float = 0.0,  # reward lost when the child is exactly the parent
        workers: int = 0,  # Julia worker processes (0 = number of CPUs - 1)
        db: str = "",  # SQLite file every scored polynomial is saved to, and parents are read from
        parent_db_frac: float = 0.0,  # fraction of parents drawn from the pool, once it is big enough
        parent_min_height: int = 2,  # a polynomial needs at least this score to enter the pool
        parent_db_min_size: int = 1000,  # parents are all random until the pool has this many polynomials
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.p, self.n, self.d = p, n, d
        self.cutoff, self.infinite_score = cutoff, infinite_score
        self.hamming_penalty, self.copy_penalty = hamming_penalty, copy_penalty
        # For quartic K3 surfaces a chain that runs past the cutoff is supersingular.
        self.past_cutoff_is_infinite = n == 4 and d == 4 and cutoff >= 10
        m = len(monomials(n, d))
        self.single_observation_space = gym.spaces.MultiDiscrete([p] * m)
        self.single_action_space = gym.spaces.MultiDiscrete([p] * m)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

        self.rng = np.random.default_rng(seed)
        self.oracle = HeightOracle(p, n, d, cutoff, workers=workers or None, db=db or None)
        self.parent_db_frac, self.parent_min_height = parent_db_frac, parent_min_height
        self.parent_db_min_size = max(1, parent_db_min_size)
        self.pool = ParentPool(m)
        if parent_db_frac and db:
            self._load_db(db)
        if parent_db_frac:
            log.info("parent pool: %d polynomials from the database (parents are random until %d)", len(self.pool), self.parent_db_min_size)

    def _load_db(self, db: str):
        query = (
            "SELECT DISTINCT h.coeffs, h.height FROM heights h JOIN smoothness s "
            "ON s.p = h.p AND s.n = h.n AND s.d = h.d AND s.coeffs = h.coeffs "
            "WHERE h.p = ? AND h.n = ? AND h.d = ? AND h.status = 'ok' AND s.smooth = 1 AND h.height >= ?"
        )
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60) as con:
            rows = con.execute(query, (self.p, self.n, self.d, self.parent_min_height)).fetchall()
        if rows:
            coeffs = np.stack([np.frombuffer(c, dtype=np.uint8) for c, _ in rows]).astype(np.int64)
            self.pool.add(coeffs, np.array([h for _, h in rows]), self.p)

    def _remember(self, coeffs: np.ndarray, scores: np.ndarray):
        """Add the scored polynomials that are good enough to the parent pool."""
        keep = scores >= self.parent_min_height
        if self.parent_db_frac and keep.any():
            self.pool.add(coeffs[keep], scores[keep], self.p)

    def _score(self, coeffs: np.ndarray) -> np.ndarray:
        """Scores for a batch. Polynomials that would score 2 or more must also be smooth, else 0."""
        results = self.oracle.heights(coeffs)
        s = np.array([score(r, self.infinite_score, self.cutoff, self.past_cutoff_is_infinite) for r in results])
        check = np.flatnonzero(s >= 2)
        if len(check):
            s[check[~np.array(self.oracle.smooth(coeffs[check]))]] = 0
        return s

    def _sample_parents(self) -> tuple[np.ndarray, np.ndarray]:
        parents = random_polys(self.rng, self.num_envs, self.p, self.n, self.d)
        scores = np.zeros(self.num_envs, dtype=np.int64)
        n_pool = 0
        if self.parent_db_frac and len(self.pool) >= self.parent_db_min_size:
            n_pool = int(self.rng.binomial(self.num_envs, self.parent_db_frac))
        if n_pool:
            parents[:n_pool], scores[:n_pool] = self.pool.sample(self.rng, n_pool)
        if n_pool < self.num_envs:
            scores[n_pool:] = self._score(parents[n_pool:])
            self._remember(parents[n_pool:], scores[n_pool:])
        return parents, scores

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.parents, self.parent_scores = self._sample_parents()
        return self.parents.copy(), {}

    def step(self, actions: np.ndarray):
        children = np.asarray(actions, dtype=np.int64)
        child_scores = self._score(children)
        self._remember(children, child_scores)
        hamming = (children != self.parents).sum(axis=1)
        copied = hamming == 0
        reward = child_scores - self.parent_scores - self.hamming_penalty * hamming - self.copy_penalty * copied
        info = {"parent_score": self.parent_scores, "child_score": child_scores, "hamming": hamming}
        reward = reward.astype(np.float32)
        self.parents, self.parent_scores = self._sample_parents()
        terminated = np.ones(self.num_envs, dtype=bool)
        return self.parents.copy(), reward, terminated, np.zeros(self.num_envs, dtype=bool), info

    def close_extras(self, **kwargs):
        self.oracle.close()
