"""Generate the (ρ, u) → T Newton seed table for the inversion hot path.

Stores T₀ on a regular (ρ, u) grid: at each node we solve u(T, ρ) = u for T
offline.  At runtime the inversion seeds Newton by bilinear interpolation of
this table at the query (ρ, u).  The seed lands within a fraction of a kelvin
across the operating regime, so a handful of analytic-Newton steps reach
float64 round-off.

The table is purely a convergence accelerator: the final T comes from the
Newton polish and the gradients come from the implicit-function-theorem JVP, so
the table's resolution affects speed only, never accuracy or differentiability.
Nodes that fall in the (physically unreachable) unstable interior of the
two-phase dome — where the single-phase EOS has Cv < 0 — never seed a real
single-phase query; the regular (ρ, u) grid + bilinear interpolation is robust
to them.

Range covers the near-critical supercritical operating regime plus the
subcritical liquid/vapor points the test suite exercises:

    ρ ∈ [60, 1050] kg/m³ ,  T ∈ [255, 345] K  (u-grid spans this box)

Run:  python scripts/generate_seed_table.py
Writes co2_eos/data/seed_table.npz
"""

import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from co2_eos import core
from co2_eos import saturation as sat
from co2_eos.span_wagner import T_TRIPLE, TC

RHO_LO, RHO_HI, NRHO = 60.0, 1050.0, 256
T_LO, T_HI = 255.0, 345.0
NU = 512

OUT = Path(__file__).resolve().parents[1] / "co2_eos" / "data" / "seed_table.npz"


def _u_cv(T, rho):
    return jax.vmap(lambda t, r: core._u_and_cv(t, r))(T, rho)


def _solve_T(rho, u):
    """Robust offline solve: damped Newton from the supercritical side.

    Also returns the final residual and Cv so nodes where the solve is not a
    converged, stable single-phase root can be flagged and filled.
    """
    def one(rho, u):
        def step(_, T):
            fu, cv = core._u_and_cv(T, rho)
            cv = jnp.where(jnp.abs(cv) > 1e-30, cv, 1e-30)
            dT = jnp.clip((fu - u) / cv, -40.0, 40.0)   # damp to stay stable
            return jnp.clip(T - dT, T_LO, T_HI)
        T = jax.lax.fori_loop(0, 80, step, jnp.float64(320.0))
        fu, cv = core._u_and_cv(T, rho)
        return T, fu - u, cv
    return jax.jit(jax.vmap(one))(rho, u)


def main():
    rho_grid = np.linspace(RHO_LO, RHO_HI, NRHO)

    # u-grid bounds from single-phase states only.  Inside the two-phase dome
    # the single-phase EOS has Cv < 0 and wildly negative u; those states are
    # physically unreachable for a single-phase (ρ, u) query, so we exclude
    # them when sizing the grid (otherwise one outlier ruins the resolution).
    # A state is single-phase iff T ≥ Tc or ρ ≤ ρ_v(T) or ρ ≥ ρ_l(T).
    T_scan = np.linspace(T_LO, T_HI, 181)
    RR, TT = np.meshgrid(rho_grid, T_scan, indexing="ij")
    Tflat = TT.ravel(); Rflat = RR.ravel()
    rho_l, rho_v = sat.saturation_densities(jnp.asarray(np.clip(Tflat, None, TC - 1e-3)))
    rho_l = np.asarray(rho_l); rho_v = np.asarray(rho_v)
    single_phase = (Tflat >= TC) | (Rflat <= rho_v) | (Rflat >= rho_l)
    u_s, _ = _u_cv(jnp.asarray(Tflat), jnp.asarray(Rflat))
    u_s = np.asarray(u_s)[single_phase]
    u_lo = float(u_s.min())
    u_hi = float(u_s.max())
    u_grid = np.linspace(u_lo, u_hi, NU)

    RR, UU = np.meshgrid(rho_grid, u_grid, indexing="ij")
    T0, resid, cv = _solve_T(jnp.asarray(RR.ravel()), jnp.asarray(UU.ravel()))
    T0 = np.array(T0).reshape(NRHO, NU)

    # Dome-safe fill: nodes whose solve did not converge to a stable
    # single-phase root (dome interior: u(T) non-monotone, Cv <= 0) would
    # poison the bilinear cells that straddle the dome boundary.  Replace them
    # by 1-D interpolation along the u-axis — the dome is a u-interval at
    # fixed rho, so the fill is smooth, and those nodes never seed a real
    # single-phase query (the seed is a convergence accelerator only).
    bad = ((np.abs(np.array(resid)) > 1.0) | (np.array(cv) <= 0.0)
           ).reshape(NRHO, NU)
    print(f"  dome/unconverged nodes filled: {bad.sum()} of {bad.size}")
    for i in range(NRHO):
        b = bad[i]
        if b.any() and (~b).any():
            T0[i, b] = np.interp(u_grid[b], u_grid[~b], T0[i, ~b])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # The T0 table is stored float32: it is a Newton SEED only (the polish
    # sets accuracy), f32 quantization adds ~3e-5 K to a seed whose bilinear
    # interpolation error is orders larger — and the halved footprint keeps
    # the embedded constant below the XLA:CPU size threshold above which
    # gathers are dispatched to the parallel executor (~25 us/call overhead).
    np.savez(
        OUT,
        rho_grid=rho_grid.astype(np.float64),
        u_grid=u_grid.astype(np.float64),
        T0_table=T0.astype(np.float32),
    )
    print(f"wrote {OUT}")
    print(f"  grid: {NRHO} ρ × {NU} u  over ρ[{RHO_LO},{RHO_HI}] T[{T_LO},{T_HI}]")
    print(f"  u range: [{u_lo:.4e}, {u_hi:.4e}] J/kg")
    print(f"  T0 range: [{T0.min():.2f}, {T0.max():.2f}] K")


if __name__ == "__main__":
    main()
