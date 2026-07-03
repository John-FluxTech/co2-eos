"""Generate the chi_ref Chebyshev table for the thermal-conductivity
critical enhancement.

The Olchowy-Sengers enhancement (Huber et al. 2016) needs the reduced
compressibility factor at the fixed reference temperature T_ref = 456.19 K:

    f(δ) = 1 + 2δ·αʳ_δ(τ_ref, δ) + δ²·αʳ_δδ(τ_ref, δ)

— a smooth function of δ alone (τ_ref is a constant).  Evaluating the full
6-derivative residual bundle at τ_ref per point was ~20 % of the fused hot
path; a degree-100 Chebyshev fit reproduces f to 3e-13 max relative error over
δ ∈ [0, 2.75] (ρ up to ~1286 kg/m³, beyond the inversion bisection bound) and
evaluates in ~100 fma by Clenshaw recurrence.

Run:  python scripts/generate_chi_ref_table.py
Writes co2_eos/data/chi_ref_cheb.npz
"""

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from co2_eos import span_wagner as sw
from co2_eos import helmholtz as hz

T_REF = 456.19            # K — Huber et al. (2016); must match transport._T_REF
DELTA_LO, DELTA_HI = 0.0, 2.75
DEGREE = 100

OUT = Path(__file__).resolve().parents[1] / "co2_eos" / "data" / "chi_ref_cheb.npz"


def f_exact(delta):
    tau_ref = sw.TC / T_REF
    _, ar_d, _, ar_dd, _, _ = hz.residual_derivs(tau_ref, delta)
    return 1.0 + 2.0 * delta * ar_d + delta * delta * ar_dd


def main():
    k = np.arange(DEGREE + 1)
    x = np.cos(np.pi * (k + 0.5) / (DEGREE + 1))          # Chebyshev nodes
    d = 0.5 * (x + 1.0) * (DELTA_HI - DELTA_LO) + DELTA_LO
    y = np.asarray(jax.jit(jax.vmap(f_exact))(jnp.asarray(d)))
    coefs = np.polynomial.chebyshev.chebfit(x, y, DEGREE)

    # verify on a dense grid before writing
    dg = np.linspace(DELTA_LO + 1e-9, DELTA_HI, 200001)
    ex = np.asarray(jax.jit(jax.vmap(f_exact))(jnp.asarray(dg)))
    xg = (2.0 * dg - (DELTA_LO + DELTA_HI)) / (DELTA_HI - DELTA_LO)
    ap = np.polynomial.chebyshev.chebval(xg, coefs)
    err = np.max(np.abs(ap - ex) / np.abs(ex))
    if err > 1e-11:
        raise RuntimeError(f"chi_ref Chebyshev fit too loose: {err:.3e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, coefs=coefs.astype(np.float64),
             delta_lo=np.float64(DELTA_LO), delta_hi=np.float64(DELTA_HI),
             t_ref=np.float64(T_REF))
    print(f"wrote {OUT}")
    print(f"  degree {DEGREE}, delta in [{DELTA_LO}, {DELTA_HI}]")
    print(f"  max rel err vs exact bundle: {err:.3e}")


if __name__ == "__main__":
    main()
