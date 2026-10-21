from pathlib import Path
import socket
import subprocess
import time

import pytest

from rlmath.oracle import HeightOracle

SERVER = Path(__file__).resolve().parents[1] / "Qfs.jl" / "Server.jl"
PORTS = (8080, 8081)


def listening(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="session", autouse=True)
def servers():
    """Start the height servers unless something is already listening."""
    procs = [subprocess.Popen(["julia", f"--project={SERVER.parent}", str(SERVER), str(port)]) for port in PORTS if not listening(port)]
    deadline = time.time() + 300
    while not all(listening(port) for port in PORTS):
        if time.time() > deadline:
            raise RuntimeError("height servers did not start")
        time.sleep(1)
    yield
    for proc in procs:
        proc.kill()


@pytest.fixture(scope="session")
def oracle_factory():
    made = {}

    def get(p, n=4, d=4, cutoff=10):
        k = (p, n, d, cutoff)
        if k not in made:
            made[k] = HeightOracle(p, n, d, cutoff, urls=[f"http://127.0.0.1:{port}" for port in PORTS])
        return made[k]

    yield get
    for o in made.values():
        o.close()
