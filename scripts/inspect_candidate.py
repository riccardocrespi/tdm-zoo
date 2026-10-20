"""Sample a candidate, verify its evaluation circuit, and print its structure.

Example:
    python scripts/inspect_candidate.py raa --p 101 --k 16 --set c=3 --set view=diagonal
    python scripts/inspect_candidate.py fft_style --p 2 --m 8 --k 32 --set secret=quasi_cyclic
    python scripts/inspect_candidate.py luof --p 101 --k 16 --backend cuda
"""

from __future__ import annotations

import argparse
import ast
import json
import time

import numpy as np

from tdm_zoo import GF, make_candidate
from tdm_zoo.ops import linalg
from tdm_zoo.tdm import CANDIDATES


def parse_option(text: str) -> tuple[str, object]:
    key, _, raw = text.partition("=")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    return key, value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("candidate", choices=CANDIDATES.names())
    parser.add_argument("--p", type=int, default=101)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="candidate option")
    args = parser.parse_args()

    field = GF(args.p, args.m, backend=args.backend)
    options = dict(parse_option(item) for item in args.set)
    candidate = make_candidate(args.candidate, field, k=args.k, **options)
    rng = np.random.default_rng(args.seed)

    start = time.perf_counter()
    product = candidate.sample_product(rng)
    sample = product.sample(rng)
    sampled = time.perf_counter() - start
    circuit = sample.evaluation_circuit
    vectors = field.random(rng, (args.k, 4))
    correct = np.array_equal(circuit.apply(vectors), field.matmul(sample.matrix, vectors))

    print(json.dumps(candidate.describe(), indent=2, default=str))
    print("layers:", ", ".join(f"{name}{shape}" for name, shape in product.layer_shapes))
    print("secret families:", [repr(s) for s in product.secrets])
    print(f"sampled in {sampled:.3f}s; rank(M) = {linalg.rank(field, sample.matrix)} / {args.k}")
    print(f"evaluation circuit correct on random vectors: {correct}")
    print(f"nominal circuit cost {circuit.cost} field ops vs dense 2k^2 = {2 * args.k**2}")


if __name__ == "__main__":
    main()
