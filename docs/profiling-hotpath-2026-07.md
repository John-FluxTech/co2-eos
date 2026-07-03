# Hot-path profile: where `properties_from_rho_u` spends its time

Measured 2026-07-03 on Apple M2 Pro (CPU), float64, JAX 0.10.0, co2-eos 0.2.0
(commit 6b0a326). Harness: `bench/profile_hotpath.py`. Batch points sampled
uniformly over the consumers' operating box **T ∈ [290, 350] K, ρ ∈ [60, 700]
kg/m³**, `u` derived from Span-Wagner so (ρ, u) is exactly consistent.

Consumer context: 1DSim3 measures this call at **379 µs/eval at N=64** (83.5 %
of its RHS). Standalone here it is **325 µs at N=64** — consistent; the
remainder is consumer-side glue. All timings are medians, `block_until_ready`.

## Component timings

Each row: one jitted call over the batch (µs/call). Components are jitted
separately, so each carries its own ~4 µs dispatch floor and the columns
overlap (Newton and the property pass share the δ-invariant envelope prep when
fused); the split below the table accounts for that.

| N    | full   | newton | seed | state(T,ρ) | thermo | bundle | visc | chi_ref |
|------|--------|--------|------|-----------|--------|--------|------|---------|
| 64   | 325    | 208    | 18   | 210       | 115    | 101    | 21   | 88      |
| 128  | 548    | 372    | 22   | 324       | 181    | 157    | 29   | 138     |
| 256  | 1006   | 702    | 25   | 576       | 310    | 274    | 41   | 241     |
| 1024 | 3428   | 2710   | 48   | 1622      | 771    | 719    | 92   | 677     |

- `full` = `properties_from_rho_u`; `newton` = `temperature_from_rho_u`
  (seed + 5 fixed Newton steps); `state(T,ρ)` = thermo + transport at known T;
  `thermo` = one full analytic bundle + ideal + property algebra; `bundle` =
  `residual_derivs` alone; `chi_ref` = the residual bundle at τ_ref that the
  Huber critical enhancement evaluates per point.

## Where the time goes (N=1024, dispatch amortized: 3.35 µs/point)

| Component | ≈ share |
|---|---|
| Newton (ρ,u)→T: 5 fixed iterations, τ-side transcendentals | **~60–70 %** |
| Huber critical-enhancement reference bundle (`chi_ref`) | **~20 %** |
| Final property bundle + ideal + property algebra | ~15 % |
| Transport minus chi_ref (viscosity, λ0, λres, OS assembly) | ~5 % |
| Seed table (2× searchsorted + bilinear gather) | ~1 % |

Newton standalone is 79 % of `full`; fused, its δ-envelope prep is CSE-shared
with the final bundle, so its marginal cost is somewhat lower — the loop body
(5 × τ-transcendentals) is the dominant cost either way.

## Newton convergence (the 5 iterations are a seed problem)

|T_k − T*| over 40k envelope points, **single-phase only** (33.7k points):

| k (fixed iters) | max [K] | p99.9 | p50 |
|---|---|---|---|
| 0 (seed) | 5.0 | 4.9 | 3.0e-3 |
| 1 | 3.6e-2 | 3.3e-2 | 1.4e-8 |
| 2 | 2.1e-4 | 1.3e-6 | 5.7e-14 |
| 3 | 1.4e-6 | 5.1e-13 | 5.7e-14 |
| 4 | 6.2e-11 | 4.7e-13 | 5.7e-14 |
| 5 | 9.1e-13 | 4.7e-13 | 5.7e-14 |

- The shipped bilinear seed is excellent in the bulk (p50 3 mK) but has a
  **near-critical band ~5 K off** (p99 4.4 K) — that band alone is why 5
  iterations are needed. The k=3 worst point is T=304.13 K, ρ=451 kg/m³, i.e.
  the immediate critical neighbourhood where Cv is large and Newton contracts
  slowest.
- **The convergence tail lives entirely inside the two-phase dome** (16 % of
  the uniform box, 0 % of consumers' actual states): there u(T,ρ) on the
  unstable branch is non-monotone and the inversion is ill-posed — documented
  behaviour, unchanged. Single-phase max after 5 iters: 9e-13 K.
- Consequence: a seed accurate to ~10 mK near-critical turns 5 iterations into
  2 (bulk) – 3 (critical neighbourhood) for free.

## Structural findings (what the µs are made of)

1. **Array-exponent `pow` defeats XLA strength reduction.** The residual sums
   are evaluated as `tau ** _AR_T`, `delta ** _AR_D`, … with *array* exponents:
   ~40 scalar pow/exp per Newton iteration per point and ~150 per full bundle,
   at ~10–15 ns each on M2 Pro. XLA cannot specialize a vectorized pow whose
   exponent is a tensor. But the Span-Wagner exponents are structured:
   - every τ-exponent is a multiple of ¼ (16 distinct values) → all τ-powers
     follow from `sqrt(sqrt(τ))` + ~15 multiplies;
   - every δ-exponent is a small integer (d ≤ 10, l ≤ 6) → multiply ladders;
   - the 27 `exp(-δ^l)` envelopes take only **6 distinct values** (l = 1..6);
   - the 5 Gaussian exponentials share (η, β, γ) pairwise → 4 distinct;
   - non-analytic terms: 2 distinct Δ (terms 40/41 share θ, Δ), Δ^0.875 =
     Δ^(7/8) is a sqrt-ladder, only Δ^0.925 and s^(5/3) need real pow/cbrt;
     the Ψ exponentials collapse to 2.
   A restructured kernel needs **~15–20 transcendentals per full bundle**
   (vs ~150–190) and ~10 per Newton iteration (vs ~45) — bit-identical math,
   no accuracy budget consumed.
2. **`chi_ref` is a 1-D function evaluated as a 2-D bundle.** The critical
   enhancement needs dP/dρ at fixed T_ref = 456.19 K — a smooth function of δ
   alone — yet `transport.py` evaluates the full 6-derivative residual bundle
   at (τ_ref, δ) per point per eval (~20 % of the whole hot path). A
   precomputed 1-D polynomial in δ (fit offline to ~1e-13) eliminates it.
3. **Dispatch floor** is ~4 µs/call on this machine — negligible at the
   consumer's N inside their own jit, visible in standalone component rows.

## Optimization directions this licenses (measured-before-committed)

| Direction | Mechanism | Projected effect | Accuracy cost |
|---|---|---|---|
| A. Transcendental economization | ¼-integer τ-ladders, δ-integer ladders, shared exponentials, grouped-by-l envelopes | 3–5× on Newton body and bundles | **zero** (same math) |
| B. chi_ref → 1-D δ-polynomial | precomputed offline fit | removes ~20 % of path | ~1e-13 (fit to round-off) |
| C. Seed table upgrade + fewer fixed iters | denser near-critical table → 2–3 iters instead of 5; opt-in warm-start API for time-integrating consumers | up to ~2× on Newton share | zero (polish sets accuracy) |
| D. Tensor-spline/Chebyshev surrogate of the surface | replace transcendentals entirely | further ~2× over A, at best | real, must be budgeted; near-critical 2nd-derivative fit is the hard part |

A+B+C are exact or round-off-level and compound to a projected ~3.5–5× on the
full path; D is only worth its risk if A+B+C measure short of the ≥2× target
(they should not).
