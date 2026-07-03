# Changelog

All notable changes to co2-eos are documented here.

## [0.3.0] — 2026-07-03

Performance release: the `(ρ, u)` hot path is **2.6× faster at N = 64 and
6.3× faster at N = 1024 on CPU** than 0.2.0 (7.6–19.9× vs the 0.1 autodiff
path), with **round-off-level accuracy consumed** — every property ≤ 1.3e-10
and every (ρ, u)-derivative ≤ 1.2e-11 max relative vs an independent
Span-Wagner oracle chain, both AD modes (`docs/validation-fastpath-2026-07.md`).
**No API changes**; `properties_from_rho_u` signature and `phase_hint`
semantics are untouched, and there is no accuracy/speed switch — the fast path
is exact re-association of the same Span-Wagner arithmetic.

### Changed

- **`helmholtz.py` residual kernels are transcendental-economized.** The
  Span-Wagner exponent structure (¼-integer τ-powers, small-integer δ-powers,
  6 distinct `exp(-δ^l)` envelopes, pairwise-shared Gaussian exponentials,
  shared non-analytic θ/Δ with `Δ^(7/8)` as a sqrt ladder) replaces the
  array-exponent `pow` formulation: a full 6-derivative bundle now costs
  ~12 exps + ~8 sqrts + 3 pows per point instead of ~150 pow + ~45 exp.
  Structural assumptions are asserted at import against the `span_wagner`
  coefficient tables (still the single source of truth).
- **The (ρ, u) → T Newton is 3 fixed unrolled steps** (was 5 via
  `fori_loop`). Seed table v2: 256×512, dome-interior rows filled by
  interpolation (unconverged unstable-branch nodes had been poisoning
  bilinear cells near the dome boundary), stored float32, uniform-grid direct
  indexing, all four bilinear corners from one gather. Measured
  max |T − T*| = 1.0e-12 K over 40k single-phase envelope samples — identical
  to the previous 5-step accuracy.
- **The Huber critical-enhancement reference compressibility is a
  precomputed degree-100 Chebyshev in δ** (2.8e-13 max rel over δ ∈ [0, 2.75];
  `co2_eos/data/chi_ref_cheb.npz`, generator
  `scripts/generate_chi_ref_table.py`) — it is a function of δ alone and had
  been costing a full residual bundle at τ_ref per point (~20 % of the path).
  Exact-bundle fallback if the data file is absent.
- Viscosity / conductivity correlations use the same ladder treatment
  (Rainwater-Friend exponents are multiples of ¼; λ_res is a δ-ladder).

### Added

- `bench/profile_hotpath.py` (component profile, HLO transcendental census,
  Newton convergence), `bench/validate_fastpath.py` (independent-oracle
  validation harness), `bench/proto_*.py` (measured prototypes, including the
  **rejected** tensor-Chebyshev surface surrogate — with the critical point
  inside the operating envelope the non-analytic terms defeat spectral
  convergence), and `tests/test_fastpath_validation.py` pinning the fast path
  at round-off in CI.
- Reports: `docs/profiling-hotpath-2026-07.md`,
  `docs/validation-fastpath-2026-07.md`.

## [0.2.0] — 2026-06-24

Performance redesign of the hot path: hand-coded analytic α-derivatives, a fused
`(ρ, u) → {all properties}` entry point, and a table-seeded fixed-iteration
inversion. **Accuracy and gradients are unchanged** — the redesign is validated
to match the previous autodiff EOS and CoolProp to the same tight tolerances.
The public surface gains the fused hot-path functions; the scalar state
functions keep their signatures (see "Migrating" below).

### Why

In an EOS-bound finite-volume simulation (the conserved state is `(ρ, ρu, E)`),
recovering primitives every RHS evaluation was ~95 % of the cost, and inside
that the `T(ρ, u)` inversion alone dominated. Profiling on CPU put
`temperature_from_Du` at **92 %** of the per-step EOS cost. Two things drove it:
every Newton iteration computed Cv by *nested* `jax.grad` of `α`, and the
`while_loop` Newton ran the whole `vmap` batch to the batch-maximum iteration
count (a long tail from a crude initial guess).

### Added

- **`properties_from_rho_u(rho, u, phase_hint=SUPERCRITICAL)`** — the new primary
  hot-path entry point. Batched / array-native (pass 1-D arrays, get a dict of
  1-D arrays). Solves T once from `u(T, ρ) = u` and reuses the α-derivatives to
  return `{temperature, density, pressure, cv, cp, speed_of_sound, enthalpy,
  internal_energy, entropy, gibbs_energy, viscosity, thermal_conductivity}`.
  `density` and `internal_energy` echo the inputs exactly.
- **`temperature_from_rho_u(rho, u, phase_hint=SUPERCRITICAL)`** — batched lean
  inversion returning T only. Correct jvp/vjp via the implicit function theorem.
- **`co2_eos.helmholtz`** — the analytic Helmholtz core: `residual_derivs`,
  `ideal_derivs` (value + all first/second τ,δ derivatives in one fused pass),
  plus the δ-precomputed τ-derivative path used by the Newton inner loop.
- **`co2_eos.core`** — the fused property kernel and the table-seeded inversion.
- **`co2_eos/data/seed_table.npz`** + `scripts/generate_seed_table.py` — a
  precomputed `(ρ, u) → T₀` bilinear seed table. It is a convergence accelerator
  only: the Newton polish sets accuracy and the IFT JVP sets gradients, so the
  table carries no accuracy or differentiability risk.
- `bench/` — profiling and before/after benchmark scripts (CPU and the V100S
  `gpu_bench.sh` harness for flux-compute).

### Changed

- **Derivatives are now analytic, not autodiff.** `α_δ, α_τ, α_δδ, α_ττ, α_δτ`
  are hand-coded and share each term's value, so the transcendentals are
  evaluated once per term instead of being recomputed by nested `jax.grad`.
  Validated against the autodiff derivatives of `span_wagner.alphar`/`alpha0` to
  < 3e-12 relative across the regime and near-critical stress points.
- **The `(ρ, u) → T` inversion** now uses the table seed + a fixed, unrolled
  analytic-Cv Newton (no `while_loop`), with the δ-invariant envelopes
  precomputed once per solve. Branchless and uniform across a `vmap` batch.
- **`transport.thermal_conductivity`** no longer builds its own `jax.grad`
  chain; it takes the analytic reduced derivatives. The fused `(ρ, u)` path feeds
  it the derivatives already computed at `(T, ρ)`, so the critical-enhancement
  term costs only a reference-temperature δ-derivative pair.
- **State functions are analytic-backed.** `properties`, `state_from_PT`,
  `state_from_Ph`, `state_from_Du`, `density_from_PT` keep their signatures but
  derive properties through the analytic kernel.

### Performance (CPU, Apple M-class, float64, N = 4096 batched)

| stage | before (autodiff) | after (analytic) | speedup |
|---|---:|---:|---:|
| `T(ρ, u)` inversion | 19.8 ms | 4.1 ms | **4.8×** |
| property bundle | 3.4 ms | 1.6 ms | 2.1× |
| full hot path (T + P + μ + k) | 21.7 ms | 6.0 ms | **3.6×** |

On the OVH Tesla V100S the redesign is 1.3–2.2× faster at these (latency-bound)
batch sizes and scales to **7.7× (inversion) / 4.9× (full path)** once
compute-bound (N ≳ 64k, measured to N = 2²⁰). Full tables in `bench/PROFILING.md`.

### Accuracy (unchanged)

- Round-trip `T → (ρ, u) → T` ≤ **1.2e-12 K** across the regime and the
  subcritical liquid/vapor branches (requirement: ≤ 1e-8 K).
- Fused properties match `properties(T, ρ)` at the recovered T to < 1e-9.
- CoolProp agreement is identical to v0.1 (same Span-Wagner polynomial); the
  `tests/test_accuracy_vs_coolprop.py` tolerances are unchanged.
- jvp == vjp == central finite differences for every inversion.

### Kept

- Scalar `properties`, `state_from_PT`, `state_from_Ph`, `state_from_Du`,
  `density_from_PT`; the saturation curve API; `viscosity` / `thermal_conductivity`.
- `co2_eos.span_wagner` (autodiff EOS) and `co2_eos.inversions` (original Newton
  solvers) remain importable — they are the validation ground truth and the
  benchmark "before" baseline.

## [0.1.0]

Initial release: pure-JAX Span-Wagner (1996) EOS with autodiff-derived
properties, transport correlations, saturation table, and phase-aware inversions.
