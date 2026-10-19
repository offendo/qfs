import pytest

from rlmath.oracle import HeightOracle


@pytest.fixture(scope="session")
def oracle_factory():
    """Julia workers take a few seconds to start, so share them across tests."""
    made = {}

    def get(p, n=4, d=4, cutoff=10):
        k = (p, n, d, cutoff)
        if k not in made:
            made[k] = HeightOracle(p, n, d, cutoff, workers=2)
        return made[k]

    yield get
    for o in made.values():
        o.close()
