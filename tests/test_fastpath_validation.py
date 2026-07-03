"""The economized (ρ, u) fast path must match the independent oracle chain.

The reference is built in ``bench/validate_fastpath.py`` from the
``span_wagner`` autodiff oracle plus literal-formula transport — sharing no
code with the economized kernels.  These tests run the same comparison on
smaller samples so CI keeps the fast path pinned at round-off accuracy.
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos
from co2_eos import span_wagner as sw
from co2_eos import helmholtz as hz
from co2_eos import transport as tr

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validate_fastpath", _REPO / "bench" / "validate_fastpath.py")
vf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vf)


@pytest.fixture(scope="module")
def envelope():
    rho, u, T = vf.envelope_single_phase(1500, seed=23)
    return rho, u, T


def test_properties_vs_oracle(envelope):
    """Every property ≤ 5e-10 rel vs the independent chain (measured ~1e-10)."""
    rho, u, _ = envelope
    fast = co2_eos.properties_from_rho_u(rho, u)
    ref = jax.jit(jax.vmap(vf.reference_state))(rho, u)
    for q in vf.QUANTITIES:
        e = vf.rel(np.asarray(fast[q]), np.asarray(ref[q]), vf.FLOORS[q])
        assert e.max() < 5e-10, f"{q}: max rel {e.max():.3e}"


def test_derivatives_vs_ift_oracle(envelope):
    """d(q)/d(ρ,u), forward and reverse, ≤ 1e-9 vs IFT-oracle references."""
    rho, u, _ = envelope
    rho_d, u_d = rho[:60], u[:60]
    for q in vf.QUANTITIES:
        def f(r, uu, q=q):
            return co2_eos.state_from_Du(r, uu)[q]
        dr_rev, du_rev = jax.jit(jax.vmap(jax.grad(f, argnums=(0, 1))))(rho_d, u_d)
        dr_fwd = jax.jit(jax.vmap(
            lambda r, uu: jax.jvp(f, (r, uu), (1.0, 0.0))[1]))(rho_d, u_d)
        du_fwd = jax.jit(jax.vmap(
            lambda r, uu: jax.jvp(f, (r, uu), (0.0, 1.0))[1]))(rho_d, u_d)
        dr_ref, du_ref = jax.jit(jax.vmap(
            lambda r, uu: vf.reference_derivs(r, uu, q)))(rho_d, u_d)
        dr_ref, du_ref = np.asarray(dr_ref), np.asarray(du_ref)
        scale_r = np.maximum(np.abs(dr_ref), np.abs(du_ref) * 1e-3 + 1e-30)
        scale_u = np.maximum(np.abs(du_ref), np.abs(dr_ref) * 1e-3 + 1e-30)
        for got_r, got_u, mode in ((dr_rev, du_rev, "rev"),
                                   (dr_fwd, du_fwd, "fwd")):
            er = np.max(np.abs(np.asarray(got_r) - dr_ref) / scale_r)
            eu = np.max(np.abs(np.asarray(got_u) - du_ref) / scale_u)
            assert er < 1e-9 and eu < 1e-9, \
                f"{q} ({mode}): d/drho {er:.3e}, d/du {eu:.3e}"


def test_chi_ref_cheb_matches_exact_bundle():
    """The precomputed Chebyshev equals the exact bundle at τ_ref ≤ 5e-12."""
    assert tr._HAVE_CHI_CHEB, "chi_ref_cheb.npz missing from package data"
    delta = jnp.asarray(np.linspace(1e-6, 2.75, 2000))
    tau_ref = sw.TC / tr._T_REF

    def exact(d):
        _, ar_d, _, ar_dd, _, _ = hz.residual_derivs(tau_ref, d)
        return 1.0 + 2.0 * d * ar_d + d * d * ar_dd

    ex = np.asarray(jax.jit(jax.vmap(exact))(delta))
    got = np.asarray(jax.jit(jax.vmap(tr._dpdrho_ref_reduced))(delta))
    assert np.max(np.abs(got - ex) / np.abs(ex)) < 5e-12


def test_inversion_roundoff_at_three_iterations(envelope):
    """3 fixed Newton steps from the v2 seed reach ≤ 1e-11 K single-phase."""
    rho, u, T_true = envelope
    T = co2_eos.temperature_from_rho_u(rho, u)
    err = np.abs(np.asarray(T) - np.asarray(T_true))
    assert err.max() < 1e-11, f"max |T - T*| = {err.max():.3e} K"
