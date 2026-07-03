"""Profile INSIDE properties_from_rho_u: where the per-eval microseconds go.

Components timed separately (all jitted, vmapped over the batch, float64):

  full        properties_from_rho_u(rho, u)          — the consumer hot path
  newton      temperature_from_rho_u(rho, u)         — seed + 5 fixed Newton steps
  seed        the bilinear (rho, u) -> T0 table seed alone
  state_Trho  _state_from_T_rho(T, rho)              — thermo + transport at known T
  thermo      _thermo(T, rho)                        — one full analytic bundle + ideal
  bundle      residual_derivs(tau, delta)            — the 6-derivative residual bundle
  visc        _scalar_viscosity(T, rho)
  chi_ref     residual_derivs at tau_ref             — the Huber critical-enhancement
                                                       reference bundle (delta-only fn!)

Plus:
  * an HLO op census of the compiled hot path (pow / exp / log / sqrt / atan
    counts per batch, i.e. the transcendental budget per point),
  * Newton convergence from the shipped seed table: max/p99 |T_k - T*| after
    k = 0..5 fixed iterations over the operating envelope.

Envelope sampled: T in [290, 350] K, rho in [60, 700] kg/m3 (the consumers'
declared operating box), u derived from Span-Wagner so (rho, u) is consistent.

Run:  python bench/profile_hotpath.py [--json OUT.json] [--repeat 200]
"""

import argparse
import json
import re
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import co2_eos
from co2_eos import span_wagner as sw
from co2_eos import transport as tr
from co2_eos import helmholtz as hz
from co2_eos import core

NS = [64, 128, 256, 1024]
REPEAT = 200

T_LO, T_HI = 290.0, 350.0
RHO_LO, RHO_HI = 60.0, 700.0


def envelope(n, seed=0):
    """(T, rho, u) sampled uniformly in the consumers' operating box."""
    rng = np.random.default_rng(seed)
    T = jnp.asarray(rng.uniform(T_LO, T_HI, size=n))
    rho = jnp.asarray(rng.uniform(RHO_LO, RHO_HI, size=n))
    u = sw.internal_energy(T, rho)
    return T, rho, jnp.asarray(u)


def time_call(fn, *args, repeat=REPEAT):
    """Compile + warm up, then median seconds/call over `repeat` runs."""
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ── Component kernels (jit + vmap over the batch) ───────────────────────────

f_full = co2_eos.properties_from_rho_u
f_newton = co2_eos.temperature_from_rho_u
f_seed = jax.jit(jax.vmap(core._seed_T))
f_state_Trho = jax.jit(jax.vmap(core._state_from_T_rho))
f_thermo = jax.jit(jax.vmap(core._thermo))
f_visc = jax.jit(jax.vmap(tr._scalar_viscosity))


@jax.jit
def f_bundle(tau, delta):
    return jax.vmap(hz.residual_derivs)(tau, delta)


_TAU_REF = sw.TC / tr._T_REF


@jax.jit
def f_chi_ref(delta):
    """Exactly the extra Helmholtz work the critical enhancement does."""
    return jax.vmap(lambda d: hz.residual_derivs(_TAU_REF, d))(delta)


# ── HLO transcendental census ───────────────────────────────────────────────

_OPS = ("power", "exponential", "log", "sqrt", "rsqrt", "atan2", "tanh",
        "cbrt", "divide")


def hlo_census(fn, *args):
    """Count transcendental-ish ops in the optimized HLO of fn(*args)."""
    txt = jax.jit(fn).lower(*args).compile().as_text()
    counts = {}
    for op in _OPS:
        # HLO lines look like:  %foo = f64[64]{0} power(...)
        counts[op] = len(re.findall(rf"= [a-z0-9\[\]{{}},]+ {op}\(", txt))
    return counts


# ── Newton convergence from the shipped seed ────────────────────────────────

def newton_convergence(n=20000, seed=7):
    """|T_k - T_true| percentiles after k fixed Newton steps from the table seed."""
    T_true, rho, u = envelope(n, seed=seed)

    def solve_k(rho, u, k):
        delta = rho / sw.RHOC
        dstate = hz.residual_tau_prep(delta)
        T = core._seed_T(rho, u)

        def step(_, T):
            tau = sw.TC / T
            a0_t, a0_tt = hz.ideal_tau_only(tau)
            ar_t, ar_tt = hz.residual_tau_fast(tau, dstate)
            f = sw.R * sw.TC * (a0_t + ar_t) - u
            cv = -sw.R * tau ** 2 * (a0_tt + ar_tt)
            cv = jnp.where(jnp.abs(cv) > 1e-30, cv, 1e-30)
            return jnp.clip(T - f / cv, core._T_MIN, core._T_MAX)

        return jax.lax.fori_loop(0, k, step, T)

    out = {}
    for k in range(0, 6):
        Tk = jax.jit(jax.vmap(lambda r, uu: solve_k(r, uu, k)))(rho, u)
        err = np.abs(np.asarray(Tk) - np.asarray(T_true))
        out[k] = {"max": float(err.max()),
                  "p99": float(np.percentile(err, 99)),
                  "p50": float(np.percentile(err, 50))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--repeat", type=int, default=REPEAT)
    args = ap.parse_args()

    dev = jax.devices()[0]
    print(f"JAX {jax.__version__}  device={dev} (platform={dev.platform})  x64={jax.config.jax_enable_x64}")
    results = {"device": str(dev), "jax": jax.__version__, "by_N": {}}

    cols = ["full", "newton", "seed", "state_Trho", "thermo", "bundle",
            "visc", "chi_ref"]
    header = f"{'N':>6} | " + " ".join(f"{c:>10}" for c in cols) + "   (us/call)"
    print("\n" + header)
    print("-" * len(header))

    for n in NS:
        T, rho, u = envelope(n)
        tau = sw.TC / T
        delta = rho / sw.RHOC
        r = {
            "full": time_call(f_full, rho, u, repeat=args.repeat),
            "newton": time_call(f_newton, rho, u, repeat=args.repeat),
            "seed": time_call(f_seed, rho, u, repeat=args.repeat),
            "state_Trho": time_call(f_state_Trho, T, rho, repeat=args.repeat),
            "thermo": time_call(f_thermo, T, rho, repeat=args.repeat),
            "bundle": time_call(f_bundle, tau, delta, repeat=args.repeat),
            "visc": time_call(f_visc, T, rho, repeat=args.repeat),
            "chi_ref": time_call(f_chi_ref, delta, repeat=args.repeat),
        }
        print(f"{n:>6} | " + " ".join(f"{r[c]*1e6:>10.1f}" for c in cols))
        results["by_N"][n] = r

    # Derived split at N=64 (the consumers' size)
    r = results["by_N"][64]
    full = r["full"]
    transport = r["state_Trho"] - r["thermo"]
    print(f"\nSplit at N=64 (of {full*1e6:.1f} us total):")
    for name, val in [
        ("newton (seed + 5 fixed iters)", r["newton"]),
        ("  of which seed table", r["seed"]),
        ("thermo bundle (residual+ideal+props)", r["thermo"]),
        ("transport (visc + conductivity)", transport),
        ("  of which chi_ref reference bundle", r["chi_ref"]),
    ]:
        print(f"  {name:<40} {val*1e6:>8.1f} us  ({val/full*100:>5.1f}%)")
    resid = full - r["newton"] - r["state_Trho"]
    print(f"  {'unattributed (dispatch, dict, misc)':<40} {resid*1e6:>8.1f} us  ({resid/full*100:>5.1f}%)")

    # HLO census
    print("\nHLO transcendental census (per batch call):")
    T, rho, u = envelope(64)
    census = {}
    for name, fn, fnargs in [
        ("properties_from_rho_u N=64", lambda a, b: co2_eos.properties_from_rho_u(a, b), (rho, u)),
        ("temperature_from_rho_u N=64", lambda a, b: co2_eos.temperature_from_rho_u(a, b), (rho, u)),
        ("state_from_T_rho N=64", lambda a, b: jax.vmap(core._state_from_T_rho)(a, b), (T, rho)),
    ]:
        c = hlo_census(fn, *fnargs)
        census[name] = c
        tot = sum(v for k, v in c.items() if k != "divide")
        print(f"  {name:<32} " + "  ".join(f"{k}={v}" for k, v in c.items())
              + f"   [transcendental lines={tot}]")
    results["hlo_census"] = census

    # Newton convergence
    print("\nNewton |T_k - T*| from shipped seed table (20k envelope points):")
    conv = newton_convergence()
    for k, s in conv.items():
        print(f"  k={k}:  max={s['max']:.3e}  p99={s['p99']:.3e}  p50={s['p50']:.3e}  [K]")
    results["newton_convergence"] = conv

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
