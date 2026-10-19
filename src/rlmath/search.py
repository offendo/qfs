"""Scoring, the elite pool, and model-free local search."""

from dataclasses import dataclass, field

import numpy as np

from rlmath.oracle import HeightOracle, HeightResult
from rlmath.polynomials import canonical_key, single_edits


def score(result: HeightResult, infinite_score: int, cutoff: int, past_cutoff_is_infinite: bool) -> int:
    """Turn an oracle result into the number the search maximises (before the smoothness check).

    Finite height h scores h. Infinite height (supersingular) scores `infinite_score` (the training
    default is 1, i.e. not rewarded). Invalid input and errors score 0.
    """
    if result.status == "ok":
        return result.height
    if result.status == "inf" or (result.status == "past_cutoff" and past_cutoff_is_infinite):
        return infinite_score
    if result.status == "past_cutoff":
        return cutoff + 1
    return 0


@dataclass
class Scorer:
    oracle: HeightOracle
    infinite_score: int
    past_cutoff_is_infinite: bool
    smooth_from: int = 2  # check smoothness for anything scoring at least this

    def __call__(self, coeffs: np.ndarray) -> tuple[np.ndarray, list[HeightResult]]:
        """Scores for a batch. Singular polynomials with a high score are set to 0."""
        results = self.oracle.heights(coeffs)
        s = np.array(
            [score(r, self.infinite_score, self.oracle.cutoff, self.past_cutoff_is_infinite) for r in results]
        )
        check = np.flatnonzero(s >= self.smooth_from)
        if len(check):
            smooth = self.oracle.smooth(coeffs[check])
            s[check[~np.array(smooth)]] = 0
        return s, results


@dataclass
class Elite:
    coeffs: np.ndarray
    score: int
    status: str
    round: int


@dataclass
class ElitePool:
    """Best distinct polynomials found so far, capped per score level.

    "Distinct" means up to permuting variables, scaling variables and scaling the polynomial
    (see `canonical_key`), so near-copies of one surface are counted once.

    Keeping a cap per level (rather than one global top-k) keeps lower levels represented, so the
    model still sees what height 3 looks like once height 6 exists.
    """

    p: int
    n: int
    d: int
    per_level: int = 2000
    min_score: int = 2
    levels: dict[int, dict[bytes, Elite]] = field(default_factory=dict)

    def add(self, coeffs: np.ndarray, scores: np.ndarray, results: list[HeightResult], round_: int) -> list[Elite]:
        """Add the rows that qualify; returns the ones that are new."""
        new = []
        for row, s, r in zip(coeffs, scores, results):
            if s < self.min_score:
                continue
            k = canonical_key(row, self.p, self.n, self.d)
            level = self.levels.setdefault(int(s), {})
            if k in level:
                continue
            e = Elite(row.copy(), int(s), r.status, round_)
            level[k] = e
            new.append(e)
            if len(level) > self.per_level:  # drop the oldest
                level.pop(next(iter(level)))
        return new

    def __len__(self):
        return sum(len(v) for v in self.levels.values())

    @property
    def best(self) -> int:
        return max(self.levels, default=0)

    def counts(self) -> dict[int, int]:
        return {s: len(v) for s, v in sorted(self.levels.items())}

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        rows = [e for level in self.levels.values() for e in level.values()]
        return np.array([e.coeffs for e in rows]), np.array([e.score for e in rows])

    def sample_parents(
        self, rng: np.random.Generator, count: int, temperature: float = 1.0, with_scores: bool = False
    ):
        """Pick parents for editing, favouring higher levels: P(level s) ∝ exp(s / temperature).

        Returns the coefficients, and also their scores if `with_scores`.
        """
        levels = sorted(self.levels)
        w = np.exp((np.array(levels) - max(levels)) / temperature)
        picks = rng.choice(len(levels), size=count, p=w / w.sum())
        out = []
        for li in picks:
            level = list(self.levels[levels[li]].values())
            out.append(level[rng.integers(len(level))].coeffs)
        if with_scores:
            return np.array(out), np.array(levels)[picks]
        return np.array(out)


def local_search(scorer: Scorer, start: np.ndarray, start_score: int, p: int, max_steps: int = 5):
    """Steepest-ascent hill climbing over single-coefficient changes.

    Each step scores all M*(p-1) neighbours in one batch and moves to the best one if it is
    strictly better. Returns every polynomial scored along the way, so the elite pool sees them.
    """
    current, current_score = start.copy(), start_score
    seen_rows, seen_scores, seen_results = [], [], []
    for _ in range(max_steps):
        neighbours = single_edits(current, p)
        s, results = scorer(neighbours)
        seen_rows.append(neighbours)
        seen_scores.append(s)
        seen_results.extend(results)
        best = int(np.argmax(s))
        if s[best] <= current_score:
            break
        current, current_score = neighbours[best], int(s[best])
    if not seen_rows:
        return np.empty((0, len(start))), np.empty(0), []
    return np.concatenate(seen_rows), np.concatenate(seen_scores), seen_results
