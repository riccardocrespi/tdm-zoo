# tdm-zoo

A framework for building **trapdoored-matrix (TDM) candidates** over finite fields and
studying their pseudorandomness. The object of study is the layered product

```
M = A_1 · S_1 · A_2 · S_2 ··· A_ℓ · S_ℓ · A_{ℓ+1}
```

with public `A_i` (e.g. permutations) and secret, fast `S_i` (e.g. Toeplitz,
block-diagonal, weighted accumulators). The question is whether `M` is pseudorandom
given the public layers.

## Layout

The package is organised in layers; each layer depends only on the layers below it.

| Layer | Package | Contents |
|---|---|---|
| 0 | `tdm_zoo.arith` | modular arithmetic on int64 arrays, polynomials over F_p, prime-field linear-algebra backends (`numpy`, `flint`, `torch`/`cuda`) |
| 1 | `tdm_zoo.fields` | field-agnostic `Field` interface, `PrimeField` (F_p), `ExtensionField` (F_{p^m}) |
| 2 | `tdm_zoo.ops` | `linalg` (rank, rref, kernels, inverses), fast `LinearOperator`s, symbolic `MatrixMap`s, sparse polynomials |
| 3 | `tdm_zoo.tdm` | secret families, public layers, DI14 codes, `LayeredProduct`, named candidates |
| 4 | `tdm_zoo.attacks` | `Distinguisher` interface; `peeling` (Macaulay-form peeling attack) |
| – | `tdm_zoo.experiments` | paired structured/uniform trials, advantage estimates, JSONL results |

### Candidates (`tdm_zoo.tdm.candidates`)

| Name | Construction |
|---|---|
| `fft_style` | `S_1 Π_2 S_2 ··· Π_ℓ S_ℓ` with Toeplitz, Hankel or circulant secrets (or a mix, one per layer); `secret="quasi_cyclic"` gives `[I∣C_1] Π_2 C_2 ··· Π_ℓ [I//C_ℓ]` (EMVP Eq. (1)) |
| `raa` | `Dᵀ Π_{ℓ+1} S_ℓ Π_ℓ ··· S_1 Π_1 D` with weighted accumulators; `view="diagonal"` absorbs `C_n` into the publics so that the secrets are diagonal; option for secret permutations |
| `luof` | layers `S_i = Ext · H_i · Enc` from Druk–Ishai (encoder first, then the secret block-diagonal `H_i`, then the extractor); `view="split"` (secrets `H_i`, publics `A_i = Enc Π_i Ext`) or `"composite"`; `codes="di14"` (faithful structure) or `"random"` |
| `layered` | any registered square family per layer, with `permutation` / `random` / `random_invertible` / `identity` publics |
| `layered_tdm` | `A_1 S_1 ··· S_ℓ A_{ℓ+1}` with random invertible public `A_i`; `base_family` for `S_1, S_2`, `peel_family` for `S_3..S_ℓ` (needs a homogeneous inverse map); target of the peeling attack |

### Secret families (`tdm_zoo.tdm.families`)

`diagonal`, `block_diagonal`, `dense`, `weighted_accumulator`, `toeplitz`, `hankel`,
`circulant`, `quasi_cyclic`, plus `SandwichFamily` (`L·S·R` with fixed `L`, `R`).

Each family exposes three consistent views:

1. `sample_params(rng)`: the secret randomness (trapdoor);
2. `operator(params)`: a fast `LinearOperator` (its piece of the evaluation circuit);
3. `param_map`: a **linear** map `f` whose image contains every sample. When available,
   `inverse_map` gives `g` (whose image contains every inverse), and `inverse_params(params)`
   gives a witness `w*` with `g(w*) = S⁻¹`.

   For Toeplitz secrets, `g` is the paper's degree-2 Gohberg–Semencul map with witness
   `x = T⁻¹e₀`, `z = T⁻¹·Z·J·r` (where `r` is the first row of `T`). Hankel uses `J·g_T`;
   circulant and diagonal-type families use `g = f`.

The generic rank attack only needs view 3 together with the revealed public matrices.

## Peeling attack (`tdm_zoo.attacks.peeling`)

For `M = A_1 S_1 ··· S_ℓ A_{ℓ+1}` (candidate `layered_tdm`):

1. **Base relations** (`relations.py`): a basis `Q_1..Q_K` of the degree-`d` polynomials in the
   `n²` matrix entries that vanish on `F(X_1, X_2) = A_1 f_1(X_1) A_2 f_2(X_2) A_3`. This is the
   kernel of a monomial evaluation matrix at random points, re-verified on fresh points.
2. **Peeling system** (`system.py`): candidate layers `S_j(W_j) = A_{j+1}⁻¹ g_j(W_j)` for
   `j = ℓ..3`, and `p_i(W) = Q_i(vec(M · S_ℓ(W_ℓ) ··· S_3(W_3)))`. `g_j` is the family's
   `homogeneous_inverse_map`: linear for diagonal, block-diagonal, circulant, accumulator and
   dense families, and degree-2 homogenised Gohberg–Semencul for Toeplitz/Hankel (extra
   variable `t`, with `t = 1` at the witness). The coefficients of the `p_i` come from
   evaluating on a tensor grid and interpolating with per-block inverse Vandermonde matrices.
3. **Macaulay matrix** (`macaulay.py`): rows are `m · p_i` for multiplier monomials `m` of
   multidegree `a`; columns are the monomials of multidegree `d·e + a` (block `W_ℓ` most
   significant). `PeelingSystem.witness_vector` gives `v*` with `T(M) v* = 0`.
4. **Distinguisher** (`peeling_macaulay`): outputs "structured" iff `rank T(M) < rank T(U)`.

```python
from tdm_zoo.attacks import PeelingMacaulay
from tdm_zoo.experiments import run_paired
cand = make_candidate("layered_tdm", GF(2**31 - 1), k=4, ell=4)
run_paired(cand, PeelingMacaulay(d=3, multipliers=0), trials=5, seed=0).summary()
```

Dense matrices are capped by `max_entries` (default 2·10⁸). Use a large prime: the base
relations and the rank test are correct with high probability only over large fields
(Schwartz–Zippel).

## Conventions

- Vectors are columns. `LinearOperator.apply` accepts `(n,)` or a batch `(n, b)`. `A @ B`
  composes operators (B is applied first).
- Layers are indexed left to right as in the paper. The evaluation circuit applies
  `A_{ℓ+1}` first.
- Permutation matrices satisfy `Π e_a = e_{π(a)}`.
- Field elements are int64 values in `[0, q)`. An element of F_{p^m} is encoded as
  `Σ a_i p^i`, where the `a_i` are its coefficients in the polynomial basis of the
  lexicographically first monic irreducible polynomial (or a modulus you supply). Matrices
  over any field are therefore plain int64 numpy arrays.
- Randomness always comes from an explicit `numpy.random.Generator`, so seeds fully
  determine an experiment.

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # numpy, python-flint, pytest, ruff
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126   # optional CUDA backend
```

## Usage

```python
import numpy as np
from tdm_zoo import GF, make_candidate

F = GF(101)                       # GF(2, 8) for F_{2^8}; GF(101, backend="cuda") for GPU linear algebra
cand = make_candidate("raa", F, k=16, c=3, ell=3)
rng = np.random.default_rng(0)

product = cand.sample_product(rng)    # samples the public layers
sample = product.sample(rng)          # samples the secrets
sample.matrix                         # dense M
sample.apply(v)                       # fast evaluation via the trapdoor
view = product.public_view()          # what a distinguisher gets
```

```sh
.venv/bin/python scripts/inspect_candidate.py luof --k 16 --set view=composite
.venv/bin/python -m pytest
```

To run an attack, subclass `tdm_zoo.attacks.Distinguisher` (`prepare(view)` once per
public sample, then `decide(state, M, rng)`) and call
`tdm_zoo.experiments.run_paired(candidate, distinguisher, trials, seed)`.

## Extending

- **New secret family:** subclass `SecretFamily` in `tdm/families/` and decorate it with
  `@SECRET_FAMILIES.register("name")`. Implement `nparams`, `sample_params`, `operator`
  and `_param_map`, and optionally `_inverse_map` / `inverse_params`.
  `tests/test_families.py` checks the three views against each other.
- **New candidate:** subclass `Candidate` in `tdm/candidates/`, register it with
  `@CANDIDATES.register("name")`, and return a `LayeredProduct` from `sample_product`.
- **New attack:** subclass `Distinguisher` in `attacks/` and register it with
  `@DISTINGUISHERS.register("name")`.

## Notes on the DI14 implementation

`tdm_zoo.tdm.di14` follows the structure of Druk–Ishai (Lemma 4 recursive codes,
Lemma 5 distance amplification, Theorem 3 extractor as a transposed code). It makes
three substitutions:

- random graphs replace the explicit lossless expanders of Capalbo et al.;
- the base code `C_0` is a random rate-1/4 code;
- lengths are reached by zero-padding messages and puncturing the extractor's code.
