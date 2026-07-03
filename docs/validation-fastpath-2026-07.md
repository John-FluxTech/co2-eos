# Validation report: the economized (ρ, u) hot path vs the Span-Wagner oracle

Date 2026-07-03 · co2-eos 0.2.x (post-economization) · float64 · JAX 0.10.0.
Harness: `bench/validate_fastpath.py`; CI subset: `tests/test_fastpath_validation.py`.

## What is being validated against what

The shipped hot path (`properties_from_rho_u` → economized `helmholtz`
kernels, 3-iteration table-seeded Newton, Chebyshev chi_ref, ladder
transport) is compared against an **independent reference chain sharing no
code with it**: temperature recovered by a to-convergence while-loop Newton
(tol 1e-13 K, crude seed, no table) on the autodiff oracle
`span_wagner._scalar_internal_energy`; every property from the
`span_wagner` autodiff scalar functions; transport from the literal
Laesecke-Muzny / Huber formulas with plain `pow`, chi_ref via oracle
autodiff. Reference derivatives are assembled by the implicit function
theorem from oracle partials (`dq/du|ρ = q_T/cv`,
`dq/dρ|u = q_ρ − q_T·u_ρ/cv`) — not finite differences — so the comparison
floor is round-off, not FD noise.

## Envelope

35,160 single-phase samples: uniform over T ∈ [290, 350] K,
ρ ∈ [60, 700] kg/m³, plus a near-critical corridor (T ∈ [304.2, 310] K,
ρ ∈ [380, 560] kg/m³), a tight critical approach (T ∈ Tc + [0.02, 1] K,
ρ ∈ ρc ± 30 kg/m³) and the 1DSim3 spec point (6.736 MPa / 316.65 K).
Two-phase-dome (ρ, u) states are excluded: there the single-phase inversion
is documented unstable-branch extrapolation (unchanged behaviour).

## Results

Property values (fast path vs oracle chain), max / p99 relative error:

| quantity | max rel | p99 rel |
|---|---|---|
| temperature | 3.6e-15 | 1.9e-15 |
| pressure | 3.4e-14 | 1.5e-14 |
| cv | 2.0e-12 | 3.5e-13 |
| cp | 1.3e-10 | 1.2e-11 |
| speed_of_sound | 9.4e-13 | 1.8e-13 |
| enthalpy | 1.1e-14 | 4.8e-15 |
| entropy | 7.4e-15 | 3.3e-15 |
| viscosity | 1.5e-13 | 9.3e-14 |
| thermal_conductivity | 8.5e-11 | 6.5e-12 |

Derivatives ∂q/∂ρ|u and ∂q/∂u|ρ for every quantity, **forward (jvp) and
reverse (vjp) mode both checked**, 800 points, worse-of-both-modes max
relative error: all quantities ≤ **1.2e-11** (largest: thermal_conductivity
∂/∂ρ 1.2e-11; temperature/pressure ≈ 2e-13). The custom_jvp
implicit-function-theorem path is exact in both modes.

The cp and λ maxima sit at the near-critical cp divergence, where the
denominator `1 + 2δαʳ_δ + δ²αʳ_δδ → 0` amplifies round-off identically in
any evaluation order; the p99 columns show the typical level.

## Accuracy budget

The fast path is a **re-association of the same Span-Wagner arithmetic**
(quarter-integer power ladders, shared exponentials, precomputed 1-D
Chebyshev for the T_ref compressibility at 2.8e-13, Newton polished to
round-off: max |T − T*| = 1.0e-12 K over 40k envelope samples). The budget
it consumes is round-off-level:

- consumers integrate at rtol 1e-6 to 1e-9;
- worst measured property error is 1.3e-10 (a single near-critical cp
  point), typical ≤ 1e-12;
- margin to the tightest consumer tolerance: **≥ 1 order at the single
  worst point, ≥ 3–6 orders everywhere else**.

Because nothing is fitted on the surface itself (the two precomputed
artifacts — seed table and chi_ref Chebyshev — are a convergence
accelerator and a 2.8e-13 fit of a fixed 1-D curve), the fast path **is**
the default; the `span_wagner` autodiff implementation stays in the repo as
the reference oracle and test truth. There is no accuracy/speed switch to
choose.

A tensor-spline / Chebyshev surrogate of the full surface was prototyped
and rejected with measurements (`bench/proto_chebsurf.py`): with the
critical point inside the operating box, the non-analytic terms defeat
spectral convergence (α_ττ error O(1) at degree 80), and a smooth-part-only
fit tops out at ~1e-7..1e-8 — consistent with the published near-critical
accuracy collapse of SBTL/TTSE-class table methods.

## CI guardrails (tests/test_fastpath_validation.py)

- every property ≤ 5e-10 vs the oracle chain on a 1.5k envelope sample;
- every derivative, both AD modes, ≤ 1e-9 vs IFT-oracle references;
- chi_ref Chebyshev ≤ 5e-12 vs the exact bundle over δ ∈ [0, 2.75];
- 3-iteration inversion ≤ 1e-11 K vs the generating temperatures.

Plus the pre-existing suites: analytic α-derivatives vs `jax.grad` of the
oracle, accuracy vs CoolProp 7.2.0 across three regions, jvp/vjp checks
through every inversion, dome detection, and the CoolProp-free-runtime
guard.
