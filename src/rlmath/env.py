"""Gymnasium environment: rewrite a polynomial, get rewarded for raising its height.

An episode is one step. The observation is a parent polynomial (its coefficient vector over F_p),
the action is a child polynomial (a full coefficient vector), and the reward is
score(child) - score(parent). The score is the quasi-F-split height of a smooth polynomial; a
singular or zero polynomial scores 0, so it is never worth producing.

`PolyEnv` is a vector env: its `num_envs` episodes run as one batch, scored with one call to the
height oracle. Every episode ends after its single step, so `step` returns the next episodes'
parents as the observation, and `reset` is only needed to start (and to reseed).

Parents are uniformly random polynomials, except for a fraction `parent_db_frac` that are drawn
from the smooth polynomials of height at least `parent_min_height` already saved in the SQLite
file `db` (see rlmath.store). The file is read once, when the environment is created.
"""

import logging
import sqlite3

import gymnasium as gym
from gymnasium.vector.utils import batch_space
import numpy as np

from rlmath.oracle import HeightOracle, HeightResult
from rlmath.polynomials import monomials, random_polys

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


class PolyEnv(gym.vector.VectorEnv):
    def __init__(
        self,
        num_envs: int,
        p: int = 5,
        n: int = 4,
        d: int = 4,
        cutoff: int = 10,
        infinite_score: int = 1,  # score of supersingular (infinite-height) polynomials; 1 = not rewarded
        workers: int = 0,  # Julia worker processes (0 = number of CPUs - 1)
        db: str = "",  # SQLite file every scored polynomial is saved to, and parents are read from
        parent_db_frac: float = 0.0,
        parent_min_height: int = 2,
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.p, self.n, self.d = p, n, d
        self.cutoff, self.infinite_score = cutoff, infinite_score
        # For quartic K3 surfaces a chain that runs past the cutoff is supersingular.
        self.past_cutoff_is_infinite = n == 4 and d == 4 and cutoff >= 10
        m = len(monomials(n, d))
        self.single_observation_space = gym.spaces.MultiDiscrete([p] * m)
        self.single_action_space = gym.spaces.MultiDiscrete([p] * m)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

        self.rng = np.random.default_rng(seed)
        self.oracle = HeightOracle(p, n, d, cutoff, workers=workers or None, db=db or None)
        self.parent_db_frac = parent_db_frac
        self.db_parents, self.db_scores = self._load_db_parents(db, parent_min_height) if parent_db_frac else (None, None)


    def _load_db_parents(self, db: str, min_height: int) -> tuple[np.ndarray, np.ndarray]:
        query = (
            "SELECT DISTINCT h.coeffs, h.height FROM heights h JOIN smoothness s "
            "ON s.p = h.p AND s.n = h.n AND s.d = h.d AND s.coeffs = h.coeffs "
            "WHERE h.p = ? AND h.n = ? AND h.d = ? AND h.status = 'ok' AND s.smooth = 1 AND h.height >= ?"
        )
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60) as con:
            rows = con.execute(query, (self.p, self.n, self.d, min_height)).fetchall()
        if not rows:
            log.warning("no polynomials of height >= %d in %s; parents are all random", min_height, db)
            return None, None
        coeffs = np.stack([np.frombuffer(c, dtype=np.uint8) for c, _ in rows]).astype(np.int64)
        return coeffs, np.array([h for _, h in rows])

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
        n_db = int(self.rng.binomial(self.num_envs, self.parent_db_frac)) if self.db_parents is not None else 0
        if n_db:
            pick = self.rng.integers(len(self.db_parents), size=n_db)
            parents[:n_db], scores[:n_db] = self.db_parents[pick], self.db_scores[pick]
        if n_db < self.num_envs:
            scores[n_db:] = self._score(parents[n_db:])
        return parents, scores

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.parents, self.parent_scores = self._sample_parents()
        return self.parents.copy(), {}

    def step(self, actions: np.ndarray):
        children = np.asarray(actions, dtype=np.int64)
        child_scores = self._score(children)
        reward = (child_scores - self.parent_scores).astype(np.float32)
        info = {"parent_score": self.parent_scores, "child_score": child_scores}
        self.parents, self.parent_scores = self._sample_parents()
        terminated = np.ones(self.num_envs, dtype=bool)
        return self.parents.copy(), reward, terminated, np.zeros(self.num_envs, dtype=bool), info

    def close_extras(self, **kwargs):
        self.oracle.close()
