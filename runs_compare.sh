#!/bin/sh
# p = 5 comparison at equal oracle budget: model vs local search only vs uniform random
set -e
B=100000
uv run rlmath-train --p 5 --max-oracle-calls $B --rounds 10000 --out runs/p5-model
uv run rlmath-train --p 5 --max-oracle-calls $B --rounds 10000 --no-use-model --out runs/p5-nomodel
uv run rlmath-baseline --p 5 --max-oracle-calls $B --out runs/p5-random
