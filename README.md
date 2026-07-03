# CO2-EOS

[![tests](https://github.com/FluxTech-UG/co2-eos/actions/workflows/test.yml/badge.svg)](https://github.com/FluxTech-UG/co2-eos/actions/workflows/test.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FluxTech-UG/co2-eos/blob/main/examples/launch_demo.ipynb)

Differentiable CO₂ thermodynamic properties in JAX.

CO2-EOS is a pure-JAX implementation of the Span-Wagner equation of state for carbon dioxide, with hand-coded analytic α-derivatives. Every property evaluation is JIT-compiled, vectorisable with `vmap`, and fully differentiable with `jax.grad`, enabling gradient-based optimisation of any system that depends on CO₂ thermodynamics.

```python
import co2_eos as co2
import jax, jax.numpy as jnp

# Simulation hot path: full state from the conserved variables (ρ, u).
# Batched/array-native — solves T once and reuses the α-derivatives for
# every property. This is the call a finite-volume code makes each RHS eval.
rho = jnp.array([300.0, 350.0, 420.0])      # kg/m³
u   = jnp.array([3.62e5, 3.58e5, 3.42e5])   # J/kg  (specific internal energy)
state = co2.properties_from_rho_u(rho, u)   # dict of arrays: T, P, cp, w, μ, k, …
print(state['temperature'], state['pressure'], state['speed_of_sound'])
# -> T ≈ [307.5, 315.4, 315.6] K, P ≈ [7.6, 8.8, 9.1] MPa  (supercritical)

# Properties from (T, ρ)  (scalar; vmap for batches)
state = co2.properties(T=310.0, rho=350.0)
print(state['pressure'], state['cp'], state['speed_of_sound'])

# State from (P, T): phase-aware, robust near the critical point
state = co2.state_from_PT(P=8e6, T=310.0)

# Differentiate anything: e.g. ∂ρ/∂T at constant P
# (peaks near the Widom line at these conditions)
drho_dT = jax.grad(lambda T: co2.state_from_PT(P=8e6, T=T)['density'])(310.0)

# Vectorise the scalar functions across conditions
T_array = jnp.linspace(280, 340, 100)
states = jax.vmap(lambda T: co2.state_from_PT(P=8e6, T=T))(T_array)
```

## Try it without cloning

[`examples/launch_demo.ipynb`](examples/launch_demo.ipynb) is the launch demo: validation against CoolProp, the `state_from_PT` CPU/GPU benchmarks, and a worked example of gradient-based optimisation through `state_from_PT`. (The v0.2 hot-path `(ρ, u)` benchmarks are in [`bench/PROFILING.md`](bench/PROFILING.md).) [Read it on GitHub](https://github.com/FluxTech-UG/co2-eos/blob/main/examples/launch_demo.ipynb) for a static render, or [open it on Colab](https://colab.research.google.com/github/FluxTech-UG/co2-eos/blob/main/examples/launch_demo.ipynb) and switch the runtime to T4 GPU to reproduce the comparison in about 15 minutes, most of that being the CPU baseline.

## Why this exists

If you're building CO₂ system models in Python and need thermodynamic properties, CoolProp is the standard choice. It's excellent: accurate, well-tested, and covers 110+ fluids.

But CoolProp is a C++ library with Python bindings. You can't `jax.grad` through it. You can't JIT-compile a simulation loop that calls it. And the per-call overhead from the Python↔C++ boundary adds up fast in tight integration loops.

CO2-EOS solves this for CO₂ by implementing the same reference EOS (Span-Wagner 1996) directly in JAX:

- **Differentiable.** `jax.grad` through any property, any inversion, any combination. No finite differences. The inversions carry hand-written `custom_jvp` rules (implicit function theorem), so forward- and reverse-mode gradients are exact and cheap; validated against central finite differences.
- **Fast.** JIT-compiled and vectorisable, with hand-coded, transcendental-economized analytic α-derivatives: the Span-Wagner exponent structure (¼-integer τ-powers, integer δ-powers, shared exponential envelopes) is evaluated by sqrt/multiply ladders, so a full six-derivative bundle costs ~12 exps + 3 pows per point instead of ~150 array-exponent pows — the same arithmetic, re-associated, exact to round-off. For simulation codes that carry `(ρ, u)` as conserved variables, `properties_from_rho_u` fuses the whole primitive recovery: a dome-safe table seed plus three unrolled analytic-Cv Newton steps recover `T(ρ, u)` to 1e-12 K, and one shared derivative bundle feeds every property including the transport critical enhancement. On an Apple M2 Pro (float64) the full recovery runs **123 µs for a 64-point batch and 0.53 µs/point at large batches — 7.6–19.9× faster than the v0.1 autodiff path** and 2.6–6.3× faster than v0.2. On an OVH Tesla V100S the inversion reaches **63.7×** vs v0.1 at a 2²⁰-point batch (2 ns/point; full recovery 16 ns/point), and the launch-latency floor at small batches dropped ~1.7× (`bench/compare.py`, `bench/PROFILING.md`, validation in `docs/validation-fastpath-2026-07.md`).
- **Also fast for `(P, T)`.** The `state_from_PT` workflow that mirrors `PropsSI` includes an iterative density solve and flattens to a few tens of microseconds per point at large batches, roughly 2.7× faster than `CoolProp.PropsSI` in a Python loop on the same inputs (10,000-point batch, Apple M2 Pro; see `examples/launch_demo.ipynb`, section 2). On a Colab T4 GPU it runs at 27,800 states/sec on a 10⁶-point batch (36 μs/pt) versus 1,600 states/sec on Colab CPU (615 μs/pt), a 17× speedup. Its ceiling is the density solve's data-dependent `while_loop` (lockstep across GPU warps); the feed-forward `(ρ, u)` hot path above does not have that bottleneck.
- **Composable.** `jit`, `vmap`, `grad`, `custom_vjp`: the full JAX transformation stack works. Embed property evaluations inside your own JIT-compiled simulation and differentiate end-to-end.
- **Phase-aware.** Robust inversions near the critical point using Halley's method with step damping and bisection fallback. Two-phase dome detection that avoids convergence to thermodynamically unstable spinodal states.

## What's included

**Equation of state**: Full Span-Wagner (1996) formulation for CO₂. Helmholtz free energy as A(T, ρ), with all thermodynamic properties derived from hand-coded analytic α-derivatives (validated against `jax.grad` to < 3e-12):

- Pressure, internal energy, enthalpy, entropy
- Isochoric and isobaric heat capacities (Cv, Cp)
- Speed of sound
- Gibbs energy

**Transport properties:**

- Viscosity: Laesecke & Muzny (2017), including dilute-gas, initial-density, and residual contributions, plus critical enhancement
- Thermal conductivity: Huber, Sykioti, Assael & Perkins (2016), with critical enhancement via the simplified crossover model

**Saturation curve:**

- Saturation pressure, densities, enthalpies, and entropies as functions of T or P
- Cubic spline interpolation of a precomputed Maxwell-construction table, JIT-compilable and differentiable

**State inversions:**

- `properties_from_rho_u(rho, u)` / `temperature_from_rho_u(rho, u)`: the batched simulation hot path. Carries (ρ, u) as conserved variables; solves `T(ρ, u)` once with a table-seeded fixed-iteration analytic-Cv Newton, then derives every property (and viscosity / thermal conductivity) in one fused pass. `custom_jvp` (implicit function theorem) for exact forward- and reverse-mode gradients.
- `state_from_Du(rho, u)`: scalar form of the above (vmap for batches).
- `state_from_PT(P, T)`: Halley's method with pressure-aware critical-region initial guess, step damping, and bisection safety net
- `state_from_Ph(P, h, phase_hint=)` (phase-aware): detects the two-phase dome at subcritical pressures and returns saturation properties directly, bypassing single-phase Newton iteration inside the dome

## Migrating to v0.2

v0.2 is a performance redesign. **Accuracy and gradients are unchanged** and the
scalar state functions keep their signatures, so most code needs no changes. The
one thing worth adopting is the fused hot path.

If your simulation carried `(ρ, u)` and recovered primitives with separate calls:

```python
# v0.1 — separate calls, each re-evaluating α-derivatives
from co2_eos.inversions import temperature_from_Du, SUPERCRITICAL
from co2_eos import span_wagner as sw, transport
T  = temperature_from_Du(rho, u, phase)     # nested-autodiff Cv, while_loop
P  = sw.pressure(T, rho)
mu = transport.viscosity(T, rho)
k  = transport.thermal_conductivity(T, rho)
```

replace it with the single fused call:

```python
# v0.2 — one fused pass: solve T once, reuse the α-derivatives
import co2_eos as co2
s = co2.properties_from_rho_u(rho, u)       # phase_hint defaults to SUPERCRITICAL
T, P, mu, k = s["temperature"], s["pressure"], s["viscosity"], s["thermal_conductivity"]
# also: s["cp"], s["speed_of_sound"], s["enthalpy"], s["entropy"], s["gibbs_energy"], …
```

Notes:

- `properties_from_rho_u` / `temperature_from_rho_u` are **batched** (pass 1-D
  arrays). The scalar `state_from_Du(rho, u)` returns the same dict for a single
  point.
- `phase_hint` is accepted for API symmetry but unused by the `(ρ, u)` inversion
  (`u` is monotone in `T` at fixed `ρ`, so no phase branch is needed).
- The old modules `co2_eos.span_wagner` and `co2_eos.inversions` still work; they
  are the validation ground truth and the benchmark baseline.

## Valid range

CO2-EOS covers the full fluid region of the Span-Wagner EOS:

- **Temperature:** 216.6 K (triple point) to 1100 K
- **Pressure:** up to 800 MPa
- **Saturation curve:** triple point to within 1 mK of the critical point (304.1282 K, 7.3773 MPa)

Transport properties follow the valid ranges of their respective correlations (broadly: gas and liquid up to 100 MPa, with reduced accuracy in the immediate critical region for thermal conductivity).

## Installation

```bash
pip install co2-eos
```

Requires Python ≥ 3.10 and JAX ≥ 0.4. No other dependencies.

To run the validation tests (which compare against CoolProp):

```bash
git clone https://github.com/FluxTech-UG/co2-eos.git
cd co2-eos
pip install -e ".[test]"
pytest tests/ -v
```

The `[test]` extra adds CoolProp (validation only, not a runtime
dependency) and pytest. The `[dev]` extra additionally pulls in scipy for
regenerating the saturation table from scratch via
`scripts/generate_saturation_table.py`.

## Validation

CO2-EOS is validated against CoolProp across the full valid range. The test suite covers:

- Single-phase properties on a dense (T, ρ) grid spanning gas, liquid, and supercritical regions
- Saturation curve properties from triple point to near-critical
- Transport properties (viscosity and thermal conductivity) across all phases
- Inversions (P,T → ρ), (P,h → state), (ρ,u → state) including near-critical conditions
- Comparison of all Helmholtz derivatives against CoolProp's analytical values

Both libraries implement the same Span-Wagner polynomial and the same reference transport correlations, so empirical agreement across the validation grid is at machine precision for the primary quantities and only loosens where physics (not formulation) amplifies floating-point noise:

- **Density and viscosity:** below 10⁻¹² relative error everywhere on the grid (limited by floating-point evaluation order).
- **cp, speed of sound, thermal conductivity:** below 10⁻⁹ in the bulk; up to a few × 10⁻⁸ for the speed of sound near the Widom line and within ~10 % of the saturation curve, and up to a few × 10⁻⁶ for cp at the very tip of its peak just above the critical pressure. There the inversion's residual ρ-error is amplified through ∂q/∂ρ, which is locally enormous (cp climbs above 30,000 J/(kg·K) at the peak, so a ~0.1 J/(kg·K) absolute difference still rounds to ~10⁻⁶ relative); at the same (T, ρ) the underlying polynomials still agree at machine precision.

Run the validation suite:

```bash
pytest tests/ -v
```

## Scope and philosophy

CO2-EOS is a CO₂-first library. Rather than expanding to cover more fluids, we go deeper on CO₂, improving accuracy, coverage, and robustness in the regions that matter for real engineering: near-critical, transcritical, and two-phase.

There are excellent general-purpose thermodynamic libraries: CoolProp for broad fluid coverage, teqp for autodiff-native multi-model EOS evaluation, FeOs for SAFT models and density functional theory, and Clapeyron.jl in the Julia ecosystem. CO2-EOS complements these by providing a JAX-native path for the single fluid where differentiability, speed, and near-critical robustness matter most.

Contributions that extend CO₂ capability are welcome:

- Better transport property models as new correlations are published
- Improved near-critical formulations (crossover EOS, scaled equations)
- Mixture models where CO₂ is the primary component (CO₂ + impurities for CCS, CO₂ + lubricant for heat pumps)
- Fitting to experimental data for conditions where Span-Wagner accuracy is insufficient

## Physical references

The implementations follow these published formulations:

- **EOS:** Span, R. and Wagner, W. (1996). "A New Equation of State for Carbon Dioxide." *J. Phys. Chem. Ref. Data*, 25(6), 1509–1596.
- **Viscosity:** Laesecke, A. and Muzny, C. D. (2017). "Reference Correlation for the Viscosity of Carbon Dioxide." *J. Phys. Chem. Ref. Data*, 46, 013107.
- **Thermal conductivity:** Huber, M. L., Sykioti, E. A., Assael, M. J., and Perkins, R. A. (2016). "Reference Correlation of the Thermal Conductivity of Carbon Dioxide from the Triple Point to 1100 K and up to 200 MPa." *J. Phys. Chem. Ref. Data*, 45, 013102.

Solver design is informed by:

- Bell, I. H. et al. (2014). CoolProp solver architecture: phase-aware initial guesses and fallback chains.
- Bell, I. H., Deiters, U. K., and Leal, A. M. M. (2022). "Implementing an Equation of State without Derivatives: teqp." *Ind. Eng. Chem. Res.*, 61(17), 6010–6027.
- Bell, I. H. and Alpert, B. K. (2018). "Exceptionally Reliable Density-Solving Algorithms." *Fluid Phase Equilibria*, 477, 87–97.

## Acknowledgements

CO2-EOS was developed by Claude and John Patrick Therrien during research on transcritical CO₂ thermoacoustic engines at FluxTech UG in Berlin. The Helmholtz-first autodiff design was validated against CoolProp and informed by the solver strategies documented in teqp and the CoolProp codebase.

## License

Apache-2.0
