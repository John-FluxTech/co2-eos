"""Validation harness: the fast (ρ, u) path vs an independent reference chain.

Reference = the Span-Wagner oracle (``span_wagner``'s autodiff derivatives of
α) plus the transport correlations written straight from the papers with
plain ``pow`` calls — sharing NO code with the economized ``helmholtz`` /
``core`` / ``transport`` kernels under test.  Temperature is recovered by a
to-convergence while-loop Newton (tol 1e-13 K, 60 iteration cap) seeded
crudely, independent of the seed table.

Checks over the consumers' operating envelope (T ∈ [290, 350] K,
ρ ∈ [60, 700] kg/m³, single-phase only — dome states are documented
unstable-branch extrapolation), dense-sampled including a near-critical ring
and the 1DSim3 spec point (6.736 MPa / 316.65 K):

  1. every property returned by ``properties_from_rho_u`` vs reference
     (max + p99 relative error);
  2. every (∂/∂ρ, ∂/∂u) derivative of every property, forward (jvp) AND
     reverse (vjp) mode, vs implicit-function-theorem references assembled
     from oracle partials;
  3. the accuracy-budget statement vs the consumers' integrator rtol.

Run:  python bench/validate_fastpath.py [--n 40000] [--nderiv 800]
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import co2_eos
from co2_eos import span_wagner as sw
from co2_eos import saturation as sat

# ═════════════════════════════════════════════════════════════════════════
# Independent reference chain (span_wagner autodiff + literal correlations)
# ═════════════════════════════════════════════════════════════════════════

_u_ref = sw._scalar_internal_energy
_cv_ref_fn = jax.grad(_u_ref, argnums=0)                 # du/dT = cv


def _T_from_rho_u_ref(rho, u):
    """To-convergence Newton on the oracle u(T, ρ); crude seed, no table."""
    def cond(state):
        T, i, done = state
        return jnp.logical_and(i < 60, jnp.logical_not(done))

    def body(state):
        T, i, _ = state
        f = _u_ref(T, rho) - u
        fp = _cv_ref_fn(T, rho)
        step = f / jnp.where(jnp.abs(fp) > 1e-30, fp, 1e-30)
        Tn = jnp.clip(T - step, sw.T_TRIPLE, 800.0)
        return (Tn, i + 1, jnp.abs(step) < 1e-13 * jnp.maximum(jnp.abs(Tn), 1.0))

    T0 = jnp.clip(u / (3.5 * sw.R), sw.T_TRIPLE, 800.0)
    T, _, _ = jax.lax.while_loop(cond, body, (T0, jnp.int32(0), jnp.bool_(False)))
    return T


# Transport, literal formulas (Laesecke & Muzny 2017; Huber et al. 2016),
# plain pow everywhere; chi_ref via the oracle's autodiff derivatives.
_A0 = np.array([1749.354893188350, -369.069300007128, 5423856.34887691,
                -2.21283852168356, -269503.247933569, 73145.021531826,
                5.34368649509278])
_RF_B = np.array([-19.572881, 219.73999, -1015.3226, 2471.0125, -3375.1717,
                  2491.6597, -787.26086, 14.085455, -0.34664158])
_RF_T = np.array([0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.5, -5.5])
_LAM0_L = np.array([0.0151874307, 0.0280674040, 0.0228564190, -0.00741624210])
_LAM_B = np.array([0.0100128, 0.0560488, -0.081162, 0.0624337, -0.0206336,
                   0.00253248, 0.00430829, -0.0358563, 0.067148, -0.0522855,
                   0.0174571, -0.00196414])

_M = 0.0440098
_R_MOLAR = 8.31451
_RHOC_MOLAR = 10624.9063
_PC = 7377300.0
_TT_TRIPLE = 216.592
_RHO_TL = 1178.53
_ETA_TL = (_RHO_TL ** (2.0 / 3.0) * (_R_MOLAR * _TT_TRIPLE) ** 0.5
           / (_M ** (1.0 / 6.0) * 84446887.43579945))

def _visc_ref(T, rho):
    T6 = T ** (1.0 / 6.0)
    T3 = T ** (1.0 / 3.0)
    den = (_A0[0] + _A0[1] * T6 + _A0[2] * jnp.exp(_A0[3] * T3)
           + (_A0[4] + _A0[5] * T3) / jnp.exp(T3) + _A0[6] * jnp.sqrt(T))
    eta0 = 0.0010055 * jnp.sqrt(T) / den
    T_star = T / 200.76
    B_eta = 6.02214129e23 * (3.78421e-10) ** 3 * jnp.sum(
        jnp.asarray(_RF_B) * T_star ** jnp.asarray(_RF_T))
    eta_init = eta0 * B_eta * rho / _M
    Tr = T / _TT_TRIPLE
    rhor = rho / _RHO_TL
    eta_res = _ETA_TL * (0.360603235428487 * Tr * rhor ** 3
                         + (rhor ** 2 + rhor ** 8.06282737481277)
                         / (Tr - 0.121550806591497))
    return eta0 + eta_init + eta_res


def _cond_ref(T, rho, mu):
    tau = sw.TC / T
    lam0 = tau ** (-0.5) / (_LAM0_L[0] + _LAM0_L[1] * tau
                            + _LAM0_L[2] * tau ** 2
                            + _LAM0_L[3] * tau ** 3) / 1000.0
    delta = rho / 467.6
    D = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], dtype=np.float64)
    Tt = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1], dtype=np.float64)
    lam_res = jnp.sum(jnp.asarray(_LAM_B) * delta ** jnp.asarray(D)
                      * tau ** jnp.asarray(Tt))

    # critical enhancement, oracle partials: dP/dρ_molar = (dP/dρ_mass)·M
    rho_molar = rho / _M
    d_red = rho_molar / _RHOC_MOLAR
    dPdrho_T = jax.grad(lambda rr: sw._scalar_pressure(T, rr))(rho) * _M
    chi = _PC / _RHOC_MOLAR ** 2 * rho_molar / dPdrho_T
    T_REF = 456.19
    dPdrho_ref = jax.grad(lambda rr: sw._scalar_pressure(T_REF, rr))(rho) * _M
    chi_ref = _PC / _RHOC_MOLAR ** 2 * rho_molar / dPdrho_ref * T_REF / T
    diff = chi - chi_ref

    cv_molar = sw._scalar_cv(T, rho) * _M
    cp_molar = sw._scalar_cp(T, rho) * _M

    zeta = 1.5e-10 * (jnp.maximum(diff, 0.0) / 0.052) ** (0.63 / 1.239)
    qd_zeta = 2.5e9 * zeta
    omega = (2.0 / jnp.pi) * ((cp_molar - cv_molar) / cp_molar
                              * jnp.arctan(qd_zeta)
                              + cv_molar / cp_molar * qd_zeta)
    omega0 = (2.0 / jnp.pi) * (1.0 - jnp.exp(
        -1.0 / (1.0 / jnp.maximum(qd_zeta, 1e-300)
                + qd_zeta ** 2 / (3.0 * d_red ** 2))))
    lam_c = (rho_molar * cp_molar * 1.02 * 1.3806488e-23 * T
             / (6.0 * jnp.pi * mu * jnp.maximum(zeta, 1e-300))
             * (omega - omega0))
    lam_c = jnp.where(diff > 0.0, lam_c, 0.0)
    return lam0 + lam_res + lam_c


def reference_state(rho, u):
    """Full reference state dict at scalar (ρ, u) from the oracle chain."""
    T = _T_from_rho_u_ref(rho, u)
    mu = _visc_ref(T, rho)
    return {
        "temperature": T,
        "pressure": sw._scalar_pressure(T, rho),
        "cv": sw._scalar_cv(T, rho),
        "cp": sw._scalar_cp(T, rho),
        "speed_of_sound": sw._scalar_speed_of_sound(T, rho),
        "enthalpy": sw._scalar_enthalpy(T, rho),
        "entropy": sw._scalar_entropy(T, rho),
        "gibbs_energy": sw._scalar_gibbs_energy(T, rho),
        "viscosity": mu,
        "thermal_conductivity": _cond_ref(T, rho, mu),
    }


# IFT reference derivatives: for any property q(T, ρ) the (ρ, u) derivatives
# are  dq/du|ρ = q_T / cv   and   dq/dρ|u = q_ρ - q_T · u_ρ / cv,
# with every partial from the oracle by autodiff.
def reference_derivs(rho, u, qname):
    T = _T_from_rho_u_ref(rho, u)
    q_of = {
        "temperature": lambda Tq, rq: Tq,
        "pressure": sw._scalar_pressure,
        "cv": sw._scalar_cv,
        "cp": sw._scalar_cp,
        "speed_of_sound": sw._scalar_speed_of_sound,
        "enthalpy": sw._scalar_enthalpy,
        "entropy": sw._scalar_entropy,
        "viscosity": _visc_ref,
        "thermal_conductivity": lambda Tq, rq: _cond_ref(Tq, rq, _visc_ref(Tq, rq)),
    }[qname]
    q_T = jax.grad(q_of, argnums=0)(T, rho)
    q_r = jax.grad(q_of, argnums=1)(T, rho)
    cv = _cv_ref_fn(T, rho)
    u_r = jax.grad(_u_ref, argnums=1)(T, rho)
    return q_r - q_T * u_r / cv, q_T / cv          # (∂q/∂ρ|u, ∂q/∂u|ρ)


# ═════════════════════════════════════════════════════════════════════════
# Envelope sampling (single-phase, incl. near-critical ring + spec point)
# ═════════════════════════════════════════════════════════════════════════

def envelope_single_phase(n, seed=17):
    rng = np.random.default_rng(seed)
    n_ring = max(n // 8, 64)
    T = np.concatenate([
        rng.uniform(290.0, 350.0, n - 2 * n_ring),
        rng.uniform(304.2, 310.0, n_ring),           # near-critical corridor
        304.1282 + rng.uniform(0.02, 1.0, n_ring),   # tight approach, T > Tc
        [316.65],                                     # spec point T
    ])
    rho = np.concatenate([
        rng.uniform(60.0, 700.0, n - 2 * n_ring),
        rng.uniform(380.0, 560.0, n_ring),
        467.6 + rng.uniform(-30.0, 30.0, n_ring),
        [co2_eos.density_from_PT(6.736e6, 316.65)],  # spec point rho
    ])
    rl, rv = sat.saturation_densities(jnp.asarray(np.clip(T, 220.0, 304.0)))
    dome = (T < sw.TC) & (rho > np.asarray(rv)) & (rho < np.asarray(rl))
    T, rho = T[~dome], rho[~dome]
    u = sw.internal_energy(jnp.asarray(T), jnp.asarray(rho))
    return jnp.asarray(rho), jnp.asarray(u), jnp.asarray(T)


QUANTITIES = ["temperature", "pressure", "cv", "cp", "speed_of_sound",
              "enthalpy", "entropy", "viscosity", "thermal_conductivity"]


def rel(a, b, floor):
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


# Denominator floors: h, u, s cross zero inside the envelope (IIR reference
# state), so pure relative error blows up at the zero crossing; floor at a
# characteristic magnitude instead.
FLOORS = {"temperature": 1.0, "pressure": 1e4, "cv": 1.0, "cp": 1.0,
          "speed_of_sound": 1.0, "enthalpy": 1e4, "entropy": 1e2,
          "viscosity": 1e-7, "thermal_conductivity": 1e-4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--nderiv", type=int, default=800)
    args = ap.parse_args()

    rho, u, _ = envelope_single_phase(args.n)
    print(f"envelope samples (single-phase): {rho.shape[0]}")

    fast = co2_eos.properties_from_rho_u(rho, u)
    ref = jax.jit(jax.vmap(reference_state))(rho, u)

    print("\nProperty values, fast path vs independent oracle chain:")
    print(f"{'quantity':>22} | {'max rel':>10} {'p99 rel':>10}")
    worst = 0.0
    for q in QUANTITIES:
        e = rel(np.asarray(fast[q]), np.asarray(ref[q]), FLOORS[q])
        worst = max(worst, e.max())
        print(f"{q:>22} | {e.max():>10.2e} {np.percentile(e, 99):>10.2e}")

    # ── derivatives: forward (jacfwd/jvp) and reverse (grad/vjp) ──
    nd = args.nderiv
    rho_d, u_d = rho[:nd], u[:nd]

    def fast_scalar(q):
        def f(r, uu):
            st = co2_eos.state_from_Du(r, uu)
            if q in st:
                return st[q]
            raise KeyError(q)
        return f

    print("\nDerivatives d(q)/d(rho, u), fwd AND rev, vs IFT oracle:")
    print(f"{'quantity':>22} | {'d/drho max':>11} {'d/du max':>11}  (worse of fwd/rev)")
    worst_d = 0.0
    for q in QUANTITIES:
        f = fast_scalar(q)
        g_rev = jax.jit(jax.vmap(jax.grad(f, argnums=(0, 1))))
        g_fwd = jax.jit(jax.vmap(lambda r, uu: (
            jax.jvp(f, (r, uu), (1.0, 0.0))[1],
            jax.jvp(f, (r, uu), (0.0, 1.0))[1])))
        dr_rev, du_rev = g_rev(rho_d, u_d)
        dr_fwd, du_fwd = g_fwd(rho_d, u_d)
        dr_ref, du_ref = jax.jit(jax.vmap(
            lambda r, uu: reference_derivs(r, uu, q)))(rho_d, u_d)
        dr_ref, du_ref = np.asarray(dr_ref), np.asarray(du_ref)
        scale_r = np.maximum(np.abs(dr_ref), np.abs(du_ref) * 1e-3 + 1e-30)
        scale_u = np.maximum(np.abs(du_ref), np.abs(dr_ref) * 1e-3 + 1e-30)
        e_r = max(np.max(np.abs(np.asarray(dr_rev) - dr_ref) / scale_r),
                  np.max(np.abs(np.asarray(dr_fwd) - dr_ref) / scale_r))
        e_u = max(np.max(np.abs(np.asarray(du_rev) - du_ref) / scale_u),
                  np.max(np.abs(np.asarray(du_fwd) - du_ref) / scale_u))
        worst_d = max(worst_d, e_r, e_u)
        print(f"{q:>22} | {e_r:>11.2e} {e_u:>11.2e}")

    print(f"\nWorst property error:   {worst:.2e}")
    print(f"Worst derivative error: {worst_d:.2e}")
    print("Consumers' integrator rtol: 1e-6 .. 1e-9")


if __name__ == "__main__":
    main()
