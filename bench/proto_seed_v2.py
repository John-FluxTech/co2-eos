"""Prototype C: seed-table v2 — dome-safe fill + denser grid -> fewer iters.

The shipped 128x256 bilinear seed table solves dome-interior nodes on the
unstable branch (damped Newton on a non-monotone u(T)); the garbage corners
poison neighbouring single-phase cells, giving a ~5 K seed band near the
dome/critical boundary and forcing 5 Newton iterations.

v2: 256x512 uniform grid over the same box; nodes whose solve did not converge
(or landed on cv<=0) are filled by 1-D interpolation along the u-axis (the dome
is a u-interval at fixed rho, so row fill is smooth and harmless — those nodes
never seed a real single-phase query).

Measures: seed error and |T_k - T*| vs k on dense single-phase envelope
samples, for the shipped table vs v2.

Run:  python bench/proto_seed_v2.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from co2_eos import span_wagner as sw
from co2_eos import helmholtz as hz
from co2_eos import saturation as sat
from co2_eos import core

RHO_LO, RHO_HI, NRHO = 60.0, 1050.0, 256
T_LO, T_HI = 255.0, 345.0
NU = 512


def build_v2():
    rho_grid = np.linspace(RHO_LO, RHO_HI, NRHO)

    # u-range from single-phase states (same logic as the shipped generator)
    T_scan = np.linspace(T_LO, T_HI, 181)
    RR, TT = np.meshgrid(rho_grid, T_scan, indexing="ij")
    Tf, Rf = TT.ravel(), RR.ravel()
    rho_l, rho_v = sat.saturation_densities(
        jnp.asarray(np.clip(Tf, None, sw.TC - 1e-3)))
    single = (Tf >= sw.TC) | (Rf <= np.asarray(rho_v)) | (Rf >= np.asarray(rho_l))
    u_all = np.asarray(jax.vmap(core._u_and_cv)(jnp.asarray(Tf), jnp.asarray(Rf))[0])
    u_grid = np.linspace(u_all[single].min(), u_all[single].max(), NU)

    RR, UU = np.meshgrid(rho_grid, u_grid, indexing="ij")

    def solve(rho, u):
        def step(_, T):
            fu, cv = core._u_and_cv(T, rho)
            cv = jnp.where(jnp.abs(cv) > 1e-30, cv, 1e-30)
            dT = jnp.clip((fu - u) / cv, -40.0, 40.0)
            return jnp.clip(T - dT, T_LO, T_HI)
        T = jax.lax.fori_loop(0, 80, step, jnp.float64(320.0))
        fu, cv = core._u_and_cv(T, rho)
        return T, fu - u, cv

    T0, resid, cv = jax.jit(jax.vmap(solve))(
        jnp.asarray(RR.ravel()), jnp.asarray(UU.ravel()))
    T0 = np.array(T0).reshape(NRHO, NU)
    bad = ((np.abs(np.asarray(resid)) > 1.0) | (np.asarray(cv) <= 0.0)
           ).reshape(NRHO, NU)
    print(f"v2 nodes: {NRHO*NU}, bad (dome/unconverged): {bad.sum()}")

    # row fill along u
    for i in range(NRHO):
        b = bad[i]
        if b.any() and (~b).any():
            T0[i, b] = np.interp(u_grid[b], u_grid[~b], T0[i, ~b])
    return jnp.asarray(rho_grid), jnp.asarray(u_grid), jnp.asarray(T0)


def make_seed_fn(rho_grid, u_grid, T0):
    nr, nu = T0.shape
    r_lo, r_step = float(rho_grid[0]), float(rho_grid[1] - rho_grid[0])
    u_lo, u_step = float(u_grid[0]), float(u_grid[1] - u_grid[0])

    def seed(rho, u):
        fi = jnp.clip((rho - r_lo) / r_step, 0.0, nr - 1 - 1e-9)
        fj = jnp.clip((u - u_lo) / u_step, 0.0, nu - 1 - 1e-9)
        i = fi.astype(jnp.int32)
        j = fj.astype(jnp.int32)
        fr = fi - i
        fu = fj - j
        c00 = T0[i, j]
        c01 = T0[i, j + 1]
        c10 = T0[i + 1, j]
        c11 = T0[i + 1, j + 1]
        return ((c00 * (1 - fr) + c10 * fr) * (1 - fu)
                + (c01 * (1 - fr) + c11 * fr) * fu)
    return seed


def conv_stats(seed_fn, label, n=40000):
    rng = np.random.default_rng(7)
    T = jnp.asarray(rng.uniform(290.0, 350.0, n))
    rho = jnp.asarray(rng.uniform(60.0, 700.0, n))
    u = sw.internal_energy(T, rho)
    Tn, rhon = np.asarray(T), np.asarray(rho)
    rl, rv = sat.saturation_densities(jnp.asarray(np.clip(Tn, 220.0, 304.0)))
    sp = ~((Tn < sw.TC) & (rhon > np.asarray(rv)) & (rhon < np.asarray(rl)))

    def solve_k(rho, u, k):
        dstate = hz.residual_tau_prep(rho / sw.RHOC)
        Tk = seed_fn(rho, u)
        for _ in range(k):
            ta = sw.TC / Tk
            a0_t, a0_tt = hz.ideal_tau_only(ta)
            ar_t, ar_tt = hz.residual_tau_fast(ta, dstate)
            f = sw.R * sw.TC * (a0_t + ar_t) - u
            cv = -sw.R * ta ** 2 * (a0_tt + ar_tt)
            cv = jnp.where(jnp.abs(cv) > 1e-30, cv, 1e-30)
            Tk = jnp.clip(Tk - f / cv, core._T_MIN, core._T_MAX)
        return Tk

    print(f"\n{label}:")
    for k in range(0, 5):
        Tk = jax.jit(jax.vmap(lambda r, uu: solve_k(r, uu, k)))(rho, u)
        e = np.abs(np.asarray(Tk) - Tn)[sp]
        print(f"  k={k}: max={e.max():.3e}  p99.9={np.percentile(e,99.9):.3e}"
              f"  p50={np.percentile(e,50):.3e} [K]")


def main():
    rho_grid, u_grid, T0 = build_v2()
    seed_v2 = make_seed_fn(rho_grid, u_grid, T0)
    conv_stats(core._seed_T, "shipped table (128x256, searchsorted)")
    conv_stats(seed_v2, "v2 table (256x512, dome-filled, uniform-index)")

    # timing of the seed lookup itself
    rng = np.random.default_rng(0)
    for n in (64, 1024):
        rho = jnp.asarray(rng.uniform(60.0, 700.0, n))
        u = sw.internal_energy(jnp.asarray(rng.uniform(290.0, 350.0, n)), rho)
        for name, fn in [("shipped", core._seed_T), ("v2", seed_v2)]:
            f = jax.jit(jax.vmap(fn))
            jax.block_until_ready(f(rho, u))
            ts = []
            for _ in range(200):
                t0 = time.perf_counter()
                jax.block_until_ready(f(rho, u))
                ts.append(time.perf_counter() - t0)
            print(f"  N={n} seed {name}: {np.median(ts)*1e6:.1f} us")


if __name__ == "__main__":
    main()
