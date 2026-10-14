# Why the fast height step gives the same answer

This note proves that `height_fast` (appendix of `PLAN.md`) returns exactly what
`quasiFSplitHeight_CY_lift` in `Qfs/src/Qfs.jl` returns, for every homogeneous `f` of degree `n`
in `n` variables over `F_p` and every `cutoff`. The proof is about the code: it uses only what the
Julia functions compute, not any theorem from the KTY paper.

## 1. What changed, exactly

Both functions do the same steps:

1. Return 1 if `f` is F-split.
2. Build a map `θ`.
3. Set `g₂ = θ(f^(p-1))`. Then, for `k = 2, 3, …, cutoff`: return `cutoff + 2` if `g_k = 0`;
   return `k` if `g_k ∉ m^[p]`; otherwise set `g_{k+1} = θ(g_k)`.
4. Return `cutoff + 1`.

The only difference is the map `θ`. Write `u` for `polynomial_frobenius_generator(p, ·)` and `Δ`
for `Δ₁l(p, ·)`.

| | code | in symbols |
|---|---|---|
| original | `Δ = Δ₁l(p, f^(p-1))`; `θ(a) = u(Δ * a)` | `θ_old(g) = u(Δ(f^(p-1)) · g)` |
| new | `D = Δ₁l(p, f)`; `θ(g) = -f^(p-2) * u(D * g)` | `θ_new(g) = −f^(p-2) · u(Δ(f) · g)` |

(The prototype also drops the `isHomog` check. That check should stay in the real code. The proof
assumes `f` is homogeneous of degree `n`, which is exactly when the original doesn't return `-1`.)

**Where the speed comes from.** `Δ(f^(p-1))` is a polynomial of degree `np(p-1)`, built as the
`p`-th power of a polynomial with about `p³` terms. For `n = 4`, `p = 7` that is degree 168 and
about 700,000 terms. `Δ(f)` has degree `np` (28 for `n = 4`, `p = 7`) and at most 4,495 terms. The
old method spent almost all its time building `Δ(f^(p-1))` and then multiplying by it at every
step.

## 2. Why they agree (the idea)

- `Δ` behaves almost like a derivative. It satisfies a product rule mod `p`, and that rule gives
  `Δ(f^(p-1)) = −(f^(p-2))^p · Δ(f) + b^p` for some polynomial `b`. The `b^p` correction exists
  because `Δ` depends a little on how a polynomial is written down (which integer lift, which
  terms), and that dependence is always a `p`-th power.
- The map `u` pulls `p`-th powers out: `u(a^p · g) = a · u(g)`.
- Putting these together gives `θ_old(g) = θ_new(g) + b · u(g)`. The maps differ, but only by a
  multiple of `u(g)`.
- The algorithm only applies `θ` to a `g` it has just checked lies in `m^[p]`. For the polynomials
  that occur here, that check is the same as `u(g) = 0`. So the extra term is always zero at the
  moment `θ` is applied, and both versions produce the same sequence `g₂, g₃, …`.

The two maps really are different: in the numerical check below the correction `b^p` was nonzero
for every random quartic at `p = 3, 5, 7`. They agree on every input the algorithm actually feeds
them.

## 3. Detailed proof

**Notation.** `R = F_p[x₁, …, x_n]`, `m = (x₁, …, x_n)`, and `1 = (1, …, 1)`. For an integer
polynomial `A ∈ Z[x]`, `Ā` is its reduction mod `p`. `R^p = {a^p : a ∈ R}`. Since `c^p = c` for
`c ∈ F_p`, we have `(Σ a_m x^m)^p = Σ a_m x^(pm)` in `R`. So `R^p` is exactly the set of
polynomials in which every exponent is divisible by `p`, and it is closed under addition and
multiplication.

### 3.1 The two maps as the code computes them

**`u`.** `polynomial_frobenius_splitting` with indices `p-1` keeps the terms `a_m x^m` with
`m ≡ (p-1)·1 (mod p)` and sends each one to `a_m x^((m − (p-1)·1)/p)`. So `u` is additive.

**`Δ`.** For `g ∈ R`, `Δ₁l` takes the lift `G = Σᵢ cᵢ x^(mᵢ) ∈ Z[x]` with `cᵢ ∈ {0, …, p-1}` (one
term per monomial of `g`) and returns

  `Δ(g) = reduction mod p of (G^p − Σᵢ cᵢ^p x^(p·mᵢ)) / p`.

This division is exact. Mod `p`, `G^p ≡ Σ cᵢ^p x^(p mᵢ)` because the multinomial coefficients of
the cross terms are divisible by `p`.

### 3.2 Lemma 1: `u` pulls out `p`-th powers

For `a, g ∈ R`, `u(a^p · g) = a · u(g)`.

*Proof.* Write `a = Σ_i a_i x^i` and `g = Σ_j g_j x^j`, so `a^p g = Σ_{i,j} a_i g_j x^(p i + j)`.
Since `p i + j ≡ j (mod p)`, the condition `p i + j ≡ (p-1)·1` is the same as
`j ≡ (p-1)·1`. When it holds, `(p i + j − (p-1)·1)/p = i + (j − (p-1)·1)/p`. Summing gives
`u(a^p g) = Σ_i a_i x^i · Σ_{j ≡ (p-1)1} g_j x^((j−(p-1)1)/p) = a · u(g)`.
(Additivity of `u` handles the case where several pairs `(i, j)` hit the same monomial.) ∎

### 3.3 Lemma 2: `Δ` is a `p`-derivation, up to `p`-th powers

Let `φ : Z[x] → Z[x]` be the substitution `x_k ↦ x_k^p` with coefficients left alone. It is a ring
homomorphism, and `φ(A) ≡ A^p (mod p)` by the remark on `R^p` above. So

  `δ(A) := (φ(A) − A^p) / p`

is an integer polynomial for every `A ∈ Z[x]`.

**(a) Product rule.** `δ(AB) = A^p δ(B) + B^p δ(A) + p δ(A) δ(B)`.
*Proof:* `φ(AB) = φ(A)φ(B) = (A^p + pδA)(B^p + pδB)`. Subtract `(AB)^p` and divide by `p`. ∎

**(b) Powers.** For `k ≥ 1`, `δ(A^k) ≡ k · A^(p(k-1)) · δ(A) (mod p)`.
*Proof:* by induction on `k` using (a). The `pδδ` term vanishes mod `p`, and
`A^p · k A^(p(k-1)) δA + A^(pk) δA = (k+1) A^(pk) δA`. ∎

**(c) Changing the lift changes `δ` by a `p`-th power.** If `A' = A + pH`, then
`δ(A') ≡ δ(A) + H̄^p (mod p)`.
*Proof:* `A'^p = A^p + p·A^(p-1)·pH + (terms divisible by p²) ≡ A^p (mod p²)`, and
`φ(A') = φ(A) + pφ(H)`. So `δ(A') − δ(A) ≡ φ(H) (mod p)`, and `φ(H)` reduces to `H̄^p`. ∎

**(d) `Δ` versus `δ`.** For `g ∈ R` and **any** integer lift `A` of `g`, `Δ(g) ≡ −δ(A) mod R^p`.
That is, `Δ(g) + δ(A)` reduced mod `p` lies in `R^p`.
*Proof:* Let `G = Σ cᵢ x^(mᵢ)` be the lift used by the code. By Fermat, `cᵢ^p = cᵢ + p εᵢ` with
`εᵢ ∈ Z`. Then `Σ cᵢ^p x^(p mᵢ) = φ(G) + p Σ εᵢ x^(p mᵢ)`, so
`Δ(g) = (G^p − φ(G))/p − Σ εᵢ x^(p mᵢ) = −δ(G) − Σ ε̄ᵢ x^(p mᵢ)` mod `p`. The last sum lies in
`R^p`. By (c), replacing `G` with any other lift `A` changes `δ` by an element of `R^p`. ∎

### 3.4 Lemma 3: the key identity

Let `f ∈ R`. There is a `b ∈ R` with

  `Δ(f^(p-1)) = −(f^(p-2))^p · Δ(f) + b^p`.

*Proof.* Let `F` be the code's lift of `f`. Then `F^(p-1)` is *a* lift of `f^(p-1)`, though not
necessarily the one `Δ₁l` uses. Working mod `p` and mod `R^p` (both sides of each step differ by
an element of `R^p`):

- `Δ(f^(p-1)) ≡ −δ(F^(p-1))` by Lemma 2(d) with `A = F^(p-1)`;
- `−δ(F^(p-1)) ≡ −(p-1) · F^(p(p-2)) · δ(F) ≡ f^(p(p-2)) · δ(F)` by Lemma 2(b), since
  `p − 1 ≡ −1`;
- `δ(F) ≡ −Δ(f)` by Lemma 2(d) again. Multiplying a congruence mod `R^p` by `f^(p(p-2))`, which
  is itself in `R^p`, keeps it a congruence mod `R^p`.

So `Δ(f^(p-1)) ≡ −f^(p(p-2)) Δ(f) = −(f^(p-2))^p Δ(f) mod R^p`. Every element of `R^p` is some
`b^p`. ∎

(For `p = 2`, `f^(p-2) = 1` and the statement reads `Δ(f) = −Δ(f) + b²`, which is consistent
because `−1 = 1` in `F_2`.)

### 3.5 Lemma 4: the two maps differ by a multiple of `u(g)`

For every `g ∈ R`, `θ_old(g) = θ_new(g) + b · u(g)`, with `b` from Lemma 3.

*Proof.* By Lemma 3 and the additivity of `u`,
`θ_old(g) = u(−(f^(p-2))^p Δ(f) g) + u(b^p g)`. By Lemma 1, this is
`−f^(p-2) u(Δ(f) g) + b·u(g) = θ_new(g) + b·u(g)`. ∎

### 3.6 Lemma 5: on the relevant polynomials, "`g ∈ m^[p]`" means "`u(g) = 0`"

Let `g` be homogeneous of degree `n(p-1)` (or zero). Then `inPowerOfVariableIdeal(p, p, g)` is
true if and only if `u(g) = 0`. Both are equivalent to "the coefficient of `x^((p-1)·1)` in `g` is
0".

*Proof.* `inPowerOfVariableIdeal` returns false exactly when some monomial of `g` has all exponents
`< p`, that is `≤ p − 1`. Such a monomial has degree `n(p-1)` only if every exponent equals `p − 1`.
For `u`: a monomial `x^m` of `g` survives only if `m = (p-1)·1 + p·j` with `j ≥ 0`. Comparing
degrees gives `|j| = 0`, so `m = (p-1)·1`, and `u(g)` is that monomial's coefficient (a constant). ∎

### 3.7 Lemma 6: degrees are preserved

If `f` is homogeneous of degree `n` and `g` is homogeneous of degree `n(p-1)` (or zero), then
`θ_new(g)` is homogeneous of degree `n(p-1)` (or zero). `f^(p-1)` has degree `n(p-1)`.

*Proof.* Every term of `F^p` and of `Σ cᵢ^p x^(p mᵢ)` has degree `np`, so `Δ(f)` is homogeneous
of degree `np` (or zero). Then `Δ(f)·g` has degree `np + n(p-1)`. `u` sends a monomial of degree
`D` with `m ≡ (p-1)·1` to one of degree `(D − n(p-1))/p`, so `u(Δ(f) g)` has degree `n`.
Multiplying by `f^(p-2)` gives degree `n + n(p-2) = n(p-1)`. ∎

### 3.8 Theorem

For homogeneous `f` of degree `n` and any `cutoff`, `height_fast(p, f, cutoff)` equals
`quasiFSplitHeight_CY_lift(p, f, cutoff)`.

*Proof.* Both start with the same F-split test and return 1 together. Otherwise
`f^(p-1) ∈ m^[p]`, so by Lemma 5 `u(f^(p-1)) = 0`. Lemma 4 then gives
`θ_old(f^(p-1)) = θ_new(f^(p-1))`, so both start with the same `g₂`.

Suppose both have the same `g_k`. By Lemma 6 (applied to `θ_new`, and so to `θ_old` on these
inputs), `g_k` is homogeneous of degree `n(p-1)` or zero. Both versions run the same two tests on
`g_k` and stop with the same return value if either fires. If neither fires, then `g_k ∈ m^[p]`,
so `u(g_k) = 0` by Lemma 5, and Lemma 4 gives `θ_old(g_k) = θ_new(g_k)`. So `g_{k+1}` is the same
too.

By induction the two runs see identical sequences, make identical decisions, and return the same
value, including `cutoff + 1` when the loop runs out. ∎

## 4. Numerical check of the key identity

Lemma 3 is the only step with real algebra in it, so I checked it directly. For random quartics
over `F_p`, I computed `E = Δ₁l(p, f^(p-1)) + (f^(p-2))^p · Δ₁l(p, f)` with the actual `Qfs` code
and tested whether every exponent in `E` is divisible by `p` (that is, `E ∈ R^p`):

| p | random quartics | `E ∈ R^p` | `E ≠ 0` |
|---|---|---|---|
| 2 | 30 | 30 | 0 |
| 3 | 30 | 30 | 30 |
| 5 | 15 | 15 | 15 |
| 7 | 3 | 3 | 3 |

The last column shows that the correction term is real: the two maps `θ` are genuinely different
for `p ≥ 3`, and only agree on inputs with `u(g) = 0`. The end-to-end comparison in `PLAN.md`
(666 non-F-split quartics, zero height mismatches) tests the whole theorem.

## 5. Scope

- The proof shows that the two pieces of code compute the same number. It says nothing about
  whether the original code matches the KTY definition of height; that question is unchanged.
- It does not need `n = d = 4`; it holds for any `n` with `f` homogeneous of degree `n`.
