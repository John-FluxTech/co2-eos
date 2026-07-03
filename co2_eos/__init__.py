"""CO2-EOS: Differentiable CO₂ thermodynamic properties in JAX.

Pure-JAX Span-Wagner (1996) equation of state for carbon dioxide with hand-coded
analytic α-derivatives.  JIT-compilable, ``vmap``-able, fully differentiable.

Public API
==========

Simulation hot path — full state from the conserved variables (ρ, u).  These are
**batched / array-native**: pass 1-D arrays, get back a dict of 1-D arrays.  A
single fused pass solves T once and reuses the α-derivatives for every property::

    properties_from_rho_u(rho, u, phase_hint=SUPERCRITICAL) -> dict
    temperature_from_rho_u(rho, u, phase_hint=SUPERCRITICAL) -> T   # T only

State functions — **scalar** in their inputs (0-d arrays or Python floats); wrap
with ``jax.vmap`` for batches::

    properties(T, rho)              — full state from (T, ρ)
    state_from_PT(P, T)             — full state from (P, T)
    state_from_Ph(P, h)             — full state from (P, h)
    state_from_Du(rho, u)           — full state from (ρ, u)   (scalar form)
    density_from_PT(P, T)           — ρ from (P, T)

Each full-state dict has keys: ``temperature``, ``density``, ``pressure``,
``cv``, ``cp``, ``speed_of_sound``, ``enthalpy``, ``internal_energy``,
``entropy``, ``gibbs_energy``, ``viscosity``, ``thermal_conductivity``.

Saturation curve (cubic-spline interpolation of a precomputed Maxwell table)::

    saturation_pressure(T) / saturation_temperature(P)
    saturation_densities(T) / saturation_densities_P(P)
    saturation_enthalpies(T) / saturation_enthalpies_P(P)
    saturation_entropies(T) / saturation_entropies_P(P)

Transport (scalar; vmap for batches)::

    viscosity(T, rho)               → μ  [Pa·s]
    thermal_conductivity(T, rho)    → λ  [W/(m·K)]

Phase hints::

    LIQUID = 0, VAPOR = 1, SUPERCRITICAL = 2 (alias: AUTO)

See README "Migrating to v0.2" for the move from the old per-quantity calls to
the fused ``properties_from_rho_u``.
"""

import jax
import jax.numpy as jnp

# Enable float64 for the entire library (also done by submodules).
jax.config.update("jax_enable_x64", True)

__version__ = "0.3.0"

# ── Phase hint constants ────────────────────────────────────────────────────
LIQUID = 0
VAPOR = 1
SUPERCRITICAL = 2
AUTO = SUPERCRITICAL  # alias: auto-pick in the dome / supercritical-style guess

# ── Internal modules ────────────────────────────────────────────────────────
from co2_eos import span_wagner as _sw
from co2_eos import transport as _tr
from co2_eos import saturation as _sat
from co2_eos import inversions as _inv
from co2_eos import core as _core

# ── Re-exports: saturation curve ────────────────────────────────────────────
saturation_pressure = _sat.saturation_pressure
saturation_temperature = _sat.saturation_temperature
saturation_densities = _sat.saturation_densities
saturation_densities_P = _sat.saturation_densities_P
saturation_enthalpies = _sat.saturation_enthalpies_T
saturation_enthalpies_P = _sat.saturation_enthalpies
saturation_entropies = _sat.saturation_entropies
saturation_entropies_P = _sat.saturation_entropies_P


# ── Transport (scalar form; vmap for batches) ───────────────────────────────

@jax.jit
def viscosity(T, rho):
    """Dynamic viscosity μ [Pa·s] at scalar (T, ρ). vmap for batches."""
    T = jnp.asarray(T, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    return _tr._scalar_viscosity(T, rho)


@jax.jit
def thermal_conductivity(T, rho):
    """Thermal conductivity λ [W/(m·K)] at scalar (T, ρ). vmap for batches."""
    T = jnp.asarray(T, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    return _tr._scalar_thermal_conductivity(T, rho)


# ═══════════════════════════════════════════════════════════════════════════
# Hot path: full state from conserved (ρ, u) — batched / array-native
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def properties_from_rho_u(rho, u, phase_hint=SUPERCRITICAL):
    """Full thermodynamic + transport state from (ρ, u). Batched.

    The simulation hot path: solves T once from u(T,ρ)=u via the analytic-Cv
    fixed-iteration Newton, then derives every property reusing the
    α-derivatives.  ``rho`` and ``u`` are 1-D arrays of equal length; the result
    is a dict of 1-D arrays.  ``density`` and ``internal_energy`` echo the
    inputs exactly.

    Args:
        rho: Density [kg/m³], 1-D array.
        u: Specific internal energy [J/kg], 1-D array.
        phase_hint: accepted for API symmetry; the single-phase (ρ, u)
            inversion is phase-agnostic (u is monotone in T at fixed ρ on a
            single-phase branch).

    Note: this is a single-phase EOS inversion. A (ρ, u) state inside the
    two-phase dome (subcritical T, ρ between ρ_v and ρ_l) is not a single-phase
    state; the EOS returns its unstable single-phase extrapolation there. The
    near-critical supercritical operating regime has no dome.
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    phase = jnp.broadcast_to(jnp.asarray(phase_hint, dtype=jnp.int32),
                             jnp.shape(rho))
    return jax.vmap(_core._state_from_rho_u)(rho, u, phase)


@jax.jit
def temperature_from_rho_u(rho, u, phase_hint=SUPERCRITICAL):
    """Temperature T [K] such that u(T,ρ)=u at fixed ρ. Batched.

    The lean inversion (no property derivation) — use when only T is needed.
    Differentiable: correct jvp/vjp via the implicit function theorem.
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    phase = jnp.broadcast_to(jnp.asarray(phase_hint, dtype=jnp.int32),
                             jnp.shape(rho))
    return jax.vmap(_core._temperature_from_Du)(rho, u, phase)


# ═══════════════════════════════════════════════════════════════════════════
# State functions — scalar in their inputs (vmap for batches)
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def properties(T, rho):
    """Full thermodynamic + transport state at scalar (T, ρ).

    Returns a dict (see module docstring for keys). vmap for batches.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    return _core._state_from_T_rho(T, rho)


@jax.jit
def state_from_PT(P, T, phase_hint=AUTO):
    """Full state at (P, T). Phase-aware Halley solve for ρ, then derive all.

    ``pressure`` in the returned dict echoes the input ``P`` exactly.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    rho = _inv._density_from_PT(T, P, phase)
    state = _core._state_from_T_rho(T, rho)
    state["pressure"] = P
    return state


@jax.jit
def density_from_PT(P, T, phase_hint=AUTO):
    """Density [kg/m³] at scalar (P, T). Phase-aware Halley solve."""
    P = jnp.asarray(P, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    return _inv._density_from_PT(T, P, phase)


@jax.jit
def state_from_Ph(P, h, phase_hint=AUTO):
    """Full state at (P, h). Dome-aware at subcritical P; else 2-D Newton.

    ``pressure`` and ``enthalpy`` echo the inputs.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    T, rho = _inv._state_from_Ph(P, h, phase)
    state = _core._state_from_T_rho(T, rho)
    state["pressure"] = P
    state["enthalpy"] = h
    return state


@jax.jit
def state_from_Du(rho, u, phase_hint=AUTO):
    """Full state at scalar (ρ, u). Scalar form of ``properties_from_rho_u``.

    ``density`` and ``internal_energy`` echo the inputs.
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    return _core._state_from_rho_u(rho, u, phase)


__all__ = [
    "__version__",
    "LIQUID", "VAPOR", "SUPERCRITICAL", "AUTO",
    "properties_from_rho_u", "temperature_from_rho_u",
    "properties",
    "state_from_PT", "state_from_Ph", "state_from_Du",
    "density_from_PT",
    "saturation_pressure", "saturation_temperature",
    "saturation_densities", "saturation_densities_P",
    "saturation_enthalpies", "saturation_enthalpies_P",
    "saturation_entropies", "saturation_entropies_P",
    "viscosity", "thermal_conductivity",
]
