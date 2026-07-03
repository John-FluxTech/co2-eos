"""Redesigned high-performance EOS core — analytic derivatives, fused passes.

This module is the fast path of co2-eos.  It replaces the ``jax.grad`` chain in
``span_wagner`` / ``inversions`` with the hand-coded analytic derivatives in
``helmholtz``, and fuses the simulation hot path into a single entry point:

    properties_from_rho_u(rho, u, phase_hint) -> dict

which solves T once from (ρ, e) and reuses the α-derivatives to produce every
thermodynamic and transport property.  This is the call a (ρ, ρu, E) finite
volume code makes on every RHS evaluation.

Design:
  * The Newton inner loop touches only τ-derivatives (`helmholtz.residual_tau_*`)
    with the δ-dependent envelopes precomputed once (they are loop-invariant at
    fixed ρ) — all u and Cv need at fixed density.
  * A precomputed (ρ, u) → T₀ table (dome-safe, uniform grid) seeds Newton so
    that a short FIXED, unrolled iteration (no `lax.while_loop`) reaches
    float64 round-off — measured 8.5e-13 K max over the single-phase envelope
    at 3 steps.  Fixed iteration keeps the GPU kernel uniform and branchless;
    the table seed is purely a convergence accelerator (the polish sets
    accuracy and the IFT JVP sets gradients), so it carries no accuracy or
    differentiability risk.
  * Gradients use the implicit function theorem via `custom_jvp`, with the
    Jacobian entries (Cv, ∂u/∂ρ) taken analytically — correct jvp and vjp.

All functions are scalar in (T, ρ) / (ρ, u); the public wrappers in
``__init__`` vmap them over a batch.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from co2_eos.span_wagner import TC, RHOC, R, T_TRIPLE
from co2_eos import helmholtz as hz
from co2_eos import transport as _tr

# Fixed Newton iterations for the (ρ, u) → T inversion.  With the dome-filled
# 256×512 table seed, 3 analytic-Newton steps reach float64 round-off over the
# single-phase envelope (measured max 8.5e-13 K on 40k samples incl. the
# critical neighbourhood; see bench/proto_seed_v2.py and tests).  The affine
# fallback seed is coarse (±tens of K), so it gets a longer fixed iteration.
_NEWTON_ITERS_TABLE = 3
_NEWTON_ITERS_AFFINE = 8
_T_MIN = T_TRIPLE
_T_MAX = 800.0


# ═══════════════════════════════════════════════════════════════════════════
# (ρ, u) → T₀ seed table  (bilinear; convergence accelerator only)
# ═══════════════════════════════════════════════════════════════════════════
# Loaded eagerly at import (outside any JIT trace) so the arrays are captured as
# constants.  Tolerant of a missing file so scripts/generate_seed_table.py can
# import this module to BUILD the table — in that case the affine fallback seed
# is used (the table is never needed to evaluate u/Cv).

_SEED_PATH = Path(__file__).parent / "data" / "seed_table.npz"
# Affine fallback seed T0 = a + b·u (regime fit), used if the table is absent.
_AFFINE_A, _AFFINE_B = 255.6, 1.656e-4

_HAVE_SEED = False
try:
    _seed = np.load(_SEED_PATH)
    # Stored float32 (seed-only precision; halves the embedded-constant
    # footprint, which XLA:CPU gathers care about).  Kept flat so the four
    # bilinear corners come from ONE gather of (base + static offsets) —
    # four separate gather ops cost ~1.3 µs each per call.
    _SEED_T0_FLAT = jnp.asarray(_seed["T0_table"], dtype=jnp.float32).ravel()
    _np_rho = np.asarray(_seed["rho_grid"])
    _np_u = np.asarray(_seed["u_grid"])
    _SEED_NRHO = int(_np_rho.shape[0])
    _SEED_NU = int(_np_u.shape[0])
    # The grids are uniform (linspace) — index by arithmetic, not searchsorted.
    _SEED_R_LO = float(_np_rho[0])
    _SEED_R_STEP = float(_np_rho[1] - _np_rho[0])
    _SEED_U_LO = float(_np_u[0])
    _SEED_U_STEP = float(_np_u[1] - _np_u[0])
    for _g, _step in ((_np_rho, _SEED_R_STEP), (_np_u, _SEED_U_STEP)):
        if not np.allclose(np.diff(_g), _step, rtol=1e-12):
            raise ValueError("seed table grids must be uniform")
    _HAVE_SEED = True
except FileNotFoundError:
    pass


def _seed_T(rho, u):
    """Initial T guess for the (ρ, u) inversion via the bilinear table.

    Uniform-grid direct indexing (the generator writes linspace grids), with
    the query clamped to the table box — outside it the Newton polish still
    owns the accuracy, the seed is only a starting point.
    """
    if not _HAVE_SEED:
        return jnp.clip(_AFFINE_A + _AFFINE_B * u, _T_MIN, _T_MAX)
    fi = jnp.clip((rho - _SEED_R_LO) / _SEED_R_STEP, 0.0,
                  _SEED_NRHO - 1 - 1e-9)
    fj = jnp.clip((u - _SEED_U_LO) / _SEED_U_STEP, 0.0, _SEED_NU - 1 - 1e-9)
    ri = fi.astype(jnp.int32)
    uj = fj.astype(jnp.int32)
    fr = fi - ri
    fu = fj - uj
    base = ri * _SEED_NU + uj
    corners = _SEED_T0_FLAT[base + jnp.array([0, 1, _SEED_NU, _SEED_NU + 1],
                                             dtype=jnp.int32)]
    c00, c01, c10, c11 = (corners[0].astype(jnp.float64),
                          corners[1].astype(jnp.float64),
                          corners[2].astype(jnp.float64),
                          corners[3].astype(jnp.float64))
    return ((c00 * (1.0 - fr) + c10 * fr) * (1.0 - fu)
            + (c01 * (1.0 - fr) + c11 * fr) * fu)


# ═══════════════════════════════════════════════════════════════════════════
# Thermodynamic property kernel (analytic, single fused pass)
# ═══════════════════════════════════════════════════════════════════════════

def _thermo(T, rho):
    """All eight Helmholtz-derived properties at scalar (T, ρ).

    Returns (P, Cv, Cp, w, h, u, s, g) plus the reduced residual derivatives
    (αʳ_δ, αʳ_δδ, αʳ_ττ, αʳ_δτ) and ideal α⁰_ττ, so the transport layer can
    reuse them without recomputing residual_derivs.
    """
    tau = TC / T
    delta = rho / RHOC
    ar, ar_d, ar_t, ar_dd, ar_tt, ar_dt = hz.residual_derivs(tau, delta)
    a0, a0_t, a0_tt = hz.ideal_derivs(tau, delta)

    sum_tt = a0_tt + ar_tt
    P = rho * R * T * (1.0 + delta * ar_d)
    cv = -R * tau ** 2 * sum_tt
    num = (1.0 + delta * ar_d - delta * tau * ar_dt) ** 2
    den = 1.0 + 2.0 * delta * ar_d + delta ** 2 * ar_dd
    cp = cv + R * num / den
    w_sq = R * T * (den - num / (tau ** 2 * sum_tt))
    w = jnp.sqrt(jnp.maximum(w_sq, 0.0))
    sum_t = a0_t + ar_t
    h = R * T * (tau * sum_t + delta * ar_d + 1.0)
    u = R * T * tau * sum_t
    s = R * (tau * sum_t - a0 - ar)
    g = R * T * (1.0 + delta * ar_d + a0 + ar)
    aux = (ar_d, ar_dd, ar_tt, ar_dt, a0_tt)
    return (P, cv, cp, w, h, u, s, g), aux


# ── Lean (u, Cv) for the Newton inner loop ──────────────────────────────────

def _u_and_cv(T, rho):
    """Internal energy u [J/kg] and Cv [J/(kg·K)] at scalar (T, ρ).

    Only τ-derivatives are needed at fixed ρ.  Note R·T·τ = R·Tc, so u depends
    on T solely through α_τ(τ).
    """
    tau = TC / T
    delta = rho / RHOC
    a0_t, a0_tt = hz.ideal_tau_only(tau)
    ar_t, ar_tt = hz.residual_tau_derivs(tau, delta)
    u = R * TC * (a0_t + ar_t)              # R·T·τ = R·Tc
    cv = -R * tau ** 2 * (a0_tt + ar_tt)
    return u, cv


# ═══════════════════════════════════════════════════════════════════════════
# Newton inversion T(ρ, u) at fixed ρ
# ═══════════════════════════════════════════════════════════════════════════

def _solve_T(rho, u):
    """Fixed-iteration Newton solve for T given (ρ, u). Scalar.

    δ-invariant envelopes are precomputed once; each step evaluates only the
    τ-dependent transcendentals.  The short fixed count is unrolled (no loop
    construct) so XLA fuses the steps and the kernel stays branchless.
    """
    delta = rho / RHOC
    dstate = hz.residual_tau_prep(delta)
    T = _seed_T(rho, u)
    iters = _NEWTON_ITERS_TABLE if _HAVE_SEED else _NEWTON_ITERS_AFFINE

    for _ in range(iters):
        tau = TC / T
        a0_t, a0_tt = hz.ideal_tau_only(tau)
        ar_t, ar_tt = hz.residual_tau_fast(tau, dstate)
        f = R * TC * (a0_t + ar_t) - u
        cv = -R * tau ** 2 * (a0_tt + ar_tt)
        cv = jnp.where(jnp.abs(cv) > 1e-30, cv, 1e-30)
        T = jnp.clip(T - f / cv, _T_MIN, _T_MAX)
    return T


# ── Implicit differentiation via custom_jvp ─────────────────────────────────

@jax.custom_jvp
def _temperature_from_Du(rho, u, phase_hint):
    """T such that u(T, ρ) = u_target at fixed ρ. Scalar. (phase_hint unused.)"""
    return _solve_T(rho, u)


@_temperature_from_Du.defjvp
def _temperature_from_Du_jvp(primals, tangents):
    """dT = (du − (∂u/∂ρ)·dρ) / Cv, with Cv and ∂u/∂ρ analytic.

    ∂u/∂ρ |_T = R·T·τ·αʳ_δτ / ρc = R·Tc·αʳ_δτ / ρc.
    """
    rho, u, phase_hint = primals
    drho, du, _ = tangents
    T_star = _temperature_from_Du(rho, u, phase_hint)

    tau = TC / T_star
    delta = rho / RHOC
    _, _, _, _, _, ar_dt = hz.residual_derivs(tau, delta)
    a0_t, a0_tt = hz.ideal_tau_only(tau)
    _, ar_tt = hz.residual_tau_derivs(tau, delta)
    cv = -R * tau ** 2 * (a0_tt + ar_tt)
    du_drho = R * TC * ar_dt / RHOC

    cv_safe = jnp.where(jnp.abs(cv) > 1e-30, cv, 1e-30)
    dT = (du - du_drho * drho) / cv_safe
    return T_star, dT


# ═══════════════════════════════════════════════════════════════════════════
# Fused state from (ρ, u): the simulation hot path
# ═══════════════════════════════════════════════════════════════════════════

def _state_from_T_rho(T, rho):
    """Scalar fused state at (T, ρ): every thermo + transport property.

    One analytic residual-derivative pass feeds both the thermodynamics and the
    transport critical-enhancement term (viscosity needs no Helmholtz
    derivatives; conductivity reuses the bundle plus a reference-T δ-pair).

    Keys: temperature, density, pressure, internal_energy, cv, cp,
    speed_of_sound, enthalpy, entropy, gibbs_energy, viscosity,
    thermal_conductivity.
    """
    (P, cv, cp, w, h, u, s, g), aux = _thermo(T, rho)
    ar_d, ar_dd, ar_tt, ar_dt, a0_tt = aux
    mu = _tr._scalar_viscosity(T, rho)
    lam = _tr._thermal_conductivity_shared(T, rho, mu, ar_d, ar_dd, ar_tt,
                                           ar_dt, a0_tt)
    return {
        "temperature": T,
        "density": rho,
        "pressure": P,
        "internal_energy": u,
        "cv": cv,
        "cp": cp,
        "speed_of_sound": w,
        "enthalpy": h,
        "entropy": s,
        "gibbs_energy": g,
        "viscosity": mu,
        "thermal_conductivity": lam,
    }


def _state_from_rho_u(rho, u, phase_hint):
    """Scalar fused state from (ρ, u): solve T once, derive everything.

    The simulation hot path.  Echoes ρ and u exactly as the input constraint.
    """
    T = _temperature_from_Du(rho, u, phase_hint)
    st = _state_from_T_rho(T, rho)
    st["density"] = rho
    st["internal_energy"] = u
    return st
