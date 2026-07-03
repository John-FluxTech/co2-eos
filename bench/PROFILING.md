# EOS hot-path profiling & benchmark report

Where `properties_from_rho_u` spends its time and what each design round
bought. Workload: the consumers' operating envelope (T ∈ [290, 350] K,
ρ ∈ [60, 700] kg/m³, supercritical near-critical CO₂), float64, batched.
Scripts in this directory reproduce every number:

- `profile_hotpath.py` — component timing inside the hot path, HLO
  transcendental census, Newton convergence vs iteration count
- `compare.py` — v0.1 autodiff baseline vs the current path at N = 64–4096
- `validate_fastpath.py` — accuracy of every property + derivative vs an
  independent Span-Wagner oracle chain (report:
  `docs/validation-fastpath-2026-07.md`)
- `proto_econ.py`, `proto_seed_v2.py`, `proto_chebsurf.py` — the measured
  prototypes behind the current design (incl. the rejected Chebyshev-surface
  surrogate)
- `seed_study.py`, `fixed_iters_study.py`, `seed_table_study.py` — seed /
  iteration design studies

## 1. The design

The hot path solves `T(ρ, u)` by a table-seeded, fixed-count, unrolled
analytic-Cv Newton, then derives every thermodynamic and transport property
from one shared α-derivative bundle. Three structural facts keep the
arithmetic cheap — all asserted at import against the Span-Wagner coefficient
tables (`helmholtz.py`):

1. **Transcendental economization.** Every Span-Wagner τ-exponent is a
   multiple of ¼ (→ two sqrts + a multiply ladder, zero `pow`), every
   δ-exponent a small integer (→ multiply ladders); the 27 `exp(-δ^l)`
   envelopes take 6 distinct values; the Gaussian exponentials deduplicate to
   4; the non-analytic terms share θ/Δ with `Δ^(7/8)` as a sqrt ladder. A full
   6-derivative bundle costs ~12 exps + ~8 sqrts + 3 pows per point (the
   naive array-exponent form pays ~150 `pow` + ~45 `exp`, which XLA cannot
   strength-reduce because the exponents are tensors).
2. **The Newton is 3 fixed steps.** The dome-safe 256×512 `(ρ, u) → T₀` seed
   table (float32, uniform-grid indexing, one gather for all four bilinear
   corners) lands close enough that 3 unrolled steps reach round-off:
   max |T − T*| = 1.0e-12 K over 40k single-phase envelope samples. Fixed
   iteration keeps the kernel branchless and batch-uniform.
3. **The Huber chi_ref bundle is a 1-D fit.** The conductivity critical
   enhancement's reference compressibility depends on δ alone; a precomputed
   degree-100 Chebyshev (2.8e-13 max rel, `co2_eos/data/chi_ref_cheb.npz`)
   replaces a full residual bundle at τ_ref per point. Transport correlations
   get the same ladder treatment (the Rainwater-Friend exponents are
   multiples of ¼).

A tensor-spline/Chebyshev surrogate of the whole surface was prototyped and
**rejected with measurements** (`proto_chebsurf.py`): the critical point sits
inside the operating box and the non-analytic terms defeat spectral
convergence (α_ττ fit error O(1) at degree 80; smooth-part-only fits top out
at ~1e-7). Everything shipped is exact re-association or a ≤3e-13 fit of a
fixed 1-D curve — accuracy budget consumed: round-off.

## 2. Where the time went (previous analytic path, M2 Pro CPU)

`profile_hotpath.py` on the pre-economization path
(`docs/profiling-hotpath-2026-07.md` is the full report), N = 1024 asymptote,
3.35 µs/point:

| component | share |
|---|---|
| Newton: 5 fixed iterations, ~45 array-exponent pow/exp each | ~60–70 % |
| Huber chi_ref reference bundle (δ-only function, evaluated as 2-D) | ~20 % |
| final property bundle + ideal + property algebra | ~15 % |
| transport minus chi_ref | ~5 % |

The 5 iterations were a seed problem: unconverged unstable-branch nodes in
the dome interior of the seed table poisoned bilinear cells near the dome
boundary (a ~5 K seed band exactly in the near-critical approach).
Single-phase, the tail was fine — the fix (fill dome rows by interpolation)
plus doubling the grid dropped the requirement to 3 steps at identical
accuracy.

## 3. Before / after (CPU, Apple M2 Pro, float64)

`properties_from_rho_u`, µs per batched call, median; "before" = the previous
analytic path (v0.2, commit 6b0a326), "after" = the economized path:

| N | before | after | speedup |
|---:|---:|---:|---:|
| 64   | 325  | 123  | **2.6×** |
| 128  | 548  | 182  | **3.0×** |
| 256  | 1006 | 265  | **3.8×** |
| 1024 | 3428 | 544  | **6.3×** |

Per-point asymptote: 3.35 µs → 0.53 µs. At the consumers' N = 64–128 the
call carries ~10–15 µs of fixed per-call cost (dispatch + small-batch op
overhead), which consumers calling from inside their own `jit` do not pay in
full.

Against the v0.1 autodiff baseline (`compare.py`, same machine, same day):

| N | invert v0.1 | invert now | × | full v0.1 | full now | × |
|---:|---:|---:|---:|---:|---:|---:|
| 64   | 855   | 72  | 12.0 | 937   | 123  | **7.6×** |
| 256  | 2595  | 114 | 22.8 | 2908  | 278  | **10.5×** |
| 1024 | 10486 | 255 | 41.1 | 11271 | 566  | **19.9×** |
| 4096 | 19474 | 618 | 31.5 | 20595 | 1321 | **15.6×** |

## 4. V100S GPU (OVH Tesla V100S-PCIE-32GB, via flux-compute)

Reproduce with:

```bash
flux-compute run --cloud flux-ovh --upload . \
    --script bench/gpu_bench.sh --fetch "bench-out:gpu-results"
```

`properties_from_rho_u`, µs/call; "v0.2" from the recorded campaign in
`gpu-results/compare_gpu_scaling.json`, "now" from
`gpu-results/compare_gpu_scaling_econ.json` (same script, same flavor):

| N | v0.1 autodiff | v0.2 analytic | now | v0.2→now | v0.1→now |
|---:|---:|---:|---:|---:|---:|
| 64        | 955    | 689   | 409   | **1.7×** | 2.3× |
| 4096      | 1490   | 829   | 347   | **2.4×** | 4.3× |
| 16384     | 3367   | 1258  | 427   | **2.9×** | 7.9× |
| 65536     | 8682   | 2761  | 680   | **4.1×** | 12.8× |
| 262144    | 39500  | 9313  | 4241  | **2.2×** | 9.3× |
| 1048576   | 159885 | 33780 | 17165 | **2.0×** | 9.3× |

The inversion alone reaches **63.7×** vs the v0.1 autodiff path at N = 2²⁰
(2.14 ms for a million points — 2.0 ns/point).

**Reading the GPU numbers:**

- Below N ≈ 4096 the V100S is launch-bound: wall time is flat in N. The
  economization cut the *floor itself* from ~690 µs to ~330–410 µs — fewer
  ops means fewer fused kernels — but a floor remains. There is no separate
  GPU kernel strategy to dispatch to at these sizes; the fix for a
  small-batch RHS is architectural (batch more work per launch), not
  per-backend kernels. The library's branchless fixed-iteration design is
  already the right shape for both backends.
- **The CPU/GPU crossover moved in CPU's favor**: CPU now does N = 64 in
  123 µs and N = 1024 in 544 µs vs the GPU's ~300–410 µs floor, putting the
  crossover at N ≈ 700–1000 (was ~128–256). Consumers at N = 64–256 field
  sizes should run this EOS on CPU.
- At saturation (N = 2²⁰) the GPU runs the full recovery at **16 ns/point**
  (CPU: ~320 ns/point) — batched evaluation remains where the GPU pays.

## 5. Accuracy & gradients

Validated against an independent oracle chain (Span-Wagner autodiff +
literal-formula transport + to-convergence Newton) over 35k single-phase
envelope samples including the near-critical approach and the 1DSim3 spec
point — `docs/validation-fastpath-2026-07.md`:

- every property ≤ 1.3e-10 max rel (typical ≤ 1e-12; the max is round-off
  amplification at the near-critical cp divergence);
- every ∂/∂ρ|u and ∂/∂u|ρ derivative ≤ 1.2e-11, forward **and** reverse AD;
- 3-step inversion ≤ 1.0e-12 K; chi_ref Chebyshev ≤ 2.8e-13;
- analytic α-derivatives match `jax.grad` of the oracle to ~1e-12;
- CoolProp 7.2.0 agreement unchanged (`tests/test_accuracy_vs_coolprop.py`);
- consumers integrate at rtol 1e-6..1e-9 → ≥3–6 orders of margin (≥1 order
  at the single worst cp point).
