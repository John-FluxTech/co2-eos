"""Prototype D probe: can a tensor-Chebyshev surface replace the residual sum?

Fits 2-D tensor Chebyshev expansions of alpha^r over the consumers' envelope
(tau in [Tc/350, Tc/290], delta in [60,700]/rhoc — the critical point (1,1) is
INSIDE this box) and measures max relative error of the fitted second
derivatives (alpha_tt, alpha_dd — what Cv / sound speed need) on a dense grid.
Also fits the smooth part alone (poly+exp+gauss, non-analytic terms excluded)
to locate the wall.

The decision this feeds: a surrogate surface is only worth its accuracy budget
if it beats the exact economized kernel (~4x, zero budget) by enough at the
accuracy the consumers need (<=1e-6 on properties -> ~1e-8 on alpha 2nd derivs).

Run:  python bench/proto_chebsurf.py
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from co2_eos import span_wagner as sw
from co2_eos import helmholtz as hz

TAU_LO, TAU_HI = sw.TC / 350.0, sw.TC / 290.0
D_LO, D_HI = 60.0 / sw.RHOC, 700.0 / sw.RHOC

C = np.polynomial.chebyshev


def cheb_nodes(n, lo, hi):
    x = np.cos(np.pi * (np.arange(n + 1) + 0.5) / (n + 1))
    return 0.5 * (x + 1) * (hi - lo) + lo, x


def alphar_parts(tau, delta):
    """(full alpha_r, smooth part = poly+exp+gauss only) and their tt/dd."""
    full = hz.residual_derivs(tau, delta)

    # smooth part: subtract the non-analytic contribution by evaluating the
    # NA terms via span_wagner's building blocks — easiest: full minus NA,
    # where NA comes from autodiff of the NA-only sum.
    def na_only(t, d):
        dm1 = d - 1.0
        s = dm1 * dm1
        tm1 = t - 1.0
        theta = -tm1 + sw._NA_BIG_A * s ** hz._NA_P
        Delta = theta ** 2 + sw._NA_BIG_B * s ** sw._NA_A
        Db = jnp.maximum(Delta, 1e-300)
        Psi = jnp.exp(-sw._NA_BIG_C * s - sw._NA_BIG_D * tm1 ** 2)
        return jnp.sum(sw._NA_N * Db ** sw._NA_B * d * Psi)

    na_tt = jax.grad(jax.grad(na_only, 0), 0)(tau, delta)
    na_dd = jax.grad(jax.grad(na_only, 1), 1)(tau, delta)
    ar_tt_s = full[4] - na_tt
    ar_dd_s = full[3] - na_dd
    return full[4], full[3], ar_tt_s, ar_dd_s


def fit_and_test(values_fn, label, degrees=(20, 40, 60, 80)):
    # dense test grid (incl. tight ring around the critical point)
    rng = np.random.default_rng(1)
    tt = np.concatenate([rng.uniform(TAU_LO, TAU_HI, 4000),
                         1.0 + rng.uniform(-2e-3, 2e-3, 1000)])
    dd = np.concatenate([rng.uniform(D_LO, D_HI, 4000),
                         1.0 + rng.uniform(-0.05, 0.05, 1000)])
    tt = np.clip(tt, TAU_LO, TAU_HI)
    dd = np.clip(dd, D_LO, D_HI)
    ref_tt, ref_dd = values_fn(jnp.asarray(tt), jnp.asarray(dd))
    ref_tt, ref_dd = np.asarray(ref_tt), np.asarray(ref_dd)

    print(f"\n{label}:")
    for deg in degrees:
        tn, tx = cheb_nodes(deg, TAU_LO, TAU_HI)
        dn, dx = cheb_nodes(deg, D_LO, D_HI)
        TT, DD = np.meshgrid(tn, dn, indexing="ij")
        # fit alpha_r values on the tensor node grid
        vals = np.asarray(values_fn.__wrapped_value__(
            jnp.asarray(TT.ravel()), jnp.asarray(DD.ravel()))).reshape(TT.shape)
        # tensor Chebyshev coefficients: 1-D fit along each axis
        c1 = np.stack([C.chebfit(tx, vals[i, :] * 0 + vals[i, :], deg)
                       for i in range(len(tn))])  # placeholder; replaced below
        # -- proper 2-step tensor fit --
        A_t = np.stack([C.chebfit(dx, vals[i, :], deg) for i in range(len(tn))])
        coef = np.stack([C.chebfit(tx, A_t[:, j], deg) for j in range(deg + 1)],
                        axis=1)  # coef[i,j]: T_i(tau) T_j(delta)

        # second-derivative coefficient tensors (w.r.t. scaled variables)
        st = 2.0 / (TAU_HI - TAU_LO)
        sd = 2.0 / (D_HI - D_LO)
        c_tt = C.chebder(coef, 2, axis=0) * st ** 2
        c_dd = C.chebder(coef, 2, axis=1) * sd ** 2

        xt = (2 * tt - (TAU_LO + TAU_HI)) / (TAU_HI - TAU_LO)
        xd = (2 * dd - (D_LO + D_HI)) / (D_HI - D_LO)
        fit_tt = C.chebval2d(xt, xd, c_tt)
        fit_dd = C.chebval2d(xt, xd, c_dd)

        def relerr(a, b):
            return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-3))

        print(f"  deg={deg:>3}: alpha_tt max rel err = {relerr(fit_tt, ref_tt):.3e}"
              f"   alpha_dd = {relerr(fit_dd, ref_dd):.3e}")


def main():
    v = jax.jit(jax.vmap(alphar_parts))

    full_val = jax.jit(jax.vmap(lambda t, d: hz.residual_derivs(t, d)[0]))

    def smooth_val_scalar(t, d):
        ar = hz.residual_derivs(t, d)[0]
        dm1 = d - 1.0
        s = dm1 * dm1
        tm1 = t - 1.0
        theta = -tm1 + sw._NA_BIG_A * s ** hz._NA_P
        Delta = theta ** 2 + sw._NA_BIG_B * s ** sw._NA_A
        Db = jnp.maximum(Delta, 1e-300)
        Psi = jnp.exp(-sw._NA_BIG_C * s - sw._NA_BIG_D * tm1 ** 2)
        return ar - jnp.sum(sw._NA_N * Db ** sw._NA_B * d * Psi)

    smooth_val = jax.jit(jax.vmap(smooth_val_scalar))

    def full_second(t, d):
        r = v(t, d)
        return r[0], r[1]
    full_second.__wrapped_value__ = full_val

    def smooth_second(t, d):
        r = v(t, d)
        return r[2], r[3]
    smooth_second.__wrapped_value__ = smooth_val

    fit_and_test(full_second, "FULL alpha_r (incl. non-analytic terms)")
    fit_and_test(smooth_second, "SMOOTH part only (poly+exp+gauss)")


if __name__ == "__main__":
    main()
