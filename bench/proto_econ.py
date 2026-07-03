"""Prototype A/B: transcendental-economized Span-Wagner kernels, measured.

A. `residual_derivs_econ` / `residual_tau_prep_econ` + `residual_tau_fast_econ`
   — the exact Span-Wagner sums restructured so the transcendental count per
   point drops from ~150 pow + ~45 exp to ~6 pow + ~12 exp + sqrt ladders:
     * every tau exponent is a multiple of 1/4  -> 2 sqrts + multiply ladder;
     * every delta exponent is a small integer  -> multiply ladder;
     * the 27 exp(-delta^l) envelopes have only 6 distinct values (l = 1..6);
     * the 5 Gaussian exponentials share (eta, beta, gamma) pairwise -> 4;
     * non-analytic terms: one shared theta, two distinct Delta; Delta^(7/8)
       via sqrt ladder; only Delta^0.925, s^(5/3), s^(2/3) remain real pows.
   Same math, same guards — differences are round-off only.

B. chi_ref -> precomputed 1-D Chebyshev in delta: the Huber critical
   enhancement's dp/drho|_(T_ref) is a function of delta alone; fit it offline
   and evaluate by Clenshaw (no Helmholtz bundle at tau_ref).

Validates both against the shipped kernels, then times: bundle, 5-iter Newton
inner loop, and a mock fused hot path (seed + 3 econ iters + econ bundle +
transport with the Chebyshev chi_ref).

Run:  python bench/proto_econ.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from co2_eos import span_wagner as sw
from co2_eos import helmholtz as hz
from co2_eos import transport as tr
from co2_eos import core
import co2_eos

# ── Static coefficient views (Python floats — folded at trace time) ─────────

_N = np.asarray(sw._AR_N)
_D = np.asarray(sw._AR_D).astype(int)
_T = np.asarray(sw._AR_T)
_L = np.asarray(sw._AR_L).astype(int)

_GN = np.asarray(sw._AR_GAUSS_N)
_GD = np.asarray(sw._AR_GAUSS_D).astype(int)
_GT = np.asarray(sw._AR_GAUSS_T).astype(int)
_GETA = np.asarray(sw._AR_GAUSS_ETA)
_GBETA = np.asarray(sw._AR_GAUSS_BETA)
_GGAMMA = np.asarray(sw._AR_GAUSS_GAMMA)
_GEPS = np.asarray(sw._AR_GAUSS_EPS)

_NAN_ = np.asarray(sw._NA_N)
_NAA = np.asarray(sw._NA_A)
_NAB = np.asarray(sw._NA_B)
_NA_P = float(np.asarray(hz._NA_P)[0])   # 1/(2 beta) = 5/3, same for all three
assert np.all(np.asarray(hz._NA_P) == _NA_P)
_NA_BIGA = float(np.asarray(sw._NA_BIG_A)[0])   # 0.7, shared
_NA_BIGB = np.asarray(sw._NA_BIG_B)
_NA_BIGC = np.asarray(sw._NA_BIG_C)
_NA_BIGD = float(np.asarray(sw._NA_BIG_D)[0])   # 275, shared

assert np.all(np.asarray(sw._NA_BETA) == 0.3) and np.all(
    np.asarray(sw._NA_BIG_A) == 0.7) and np.all(
    np.asarray(sw._NA_BIG_D) == 275.0), "NA sharing assumptions violated"
assert _NA_BIGB[0] == _NA_BIGB[1] and _NAA[0] == _NAA[1], \
    "terms 40/41 no longer share Delta"

_TPOWS = sorted(set(_T.tolist()) | set(np.asarray(sw._AR_GAUSS_T).tolist()))
assert all(abs(t * 4 - round(t * 4)) < 1e-12 for t in _TPOWS), \
    "tau exponent not a multiple of 1/4"


def _tau_ladder(tau):
    """All distinct tau^t (t in _TPOWS) via 2 sqrts + multiplies."""
    st = jnp.sqrt(tau)            # tau^0.5
    qt = jnp.sqrt(st)             # tau^0.25
    p = {0.0: jnp.ones_like(tau), 0.25: qt, 0.5: st, 0.75: st * qt, 1.0: tau}
    # integer powers by largest-prior decomposition
    for t in _TPOWS:
        if t in p:
            continue
        if t == int(t):
            t_int = int(t)
            half = t_int // 2
            if float(half) in p and float(t_int - half) in p:
                p[t] = p[float(half)] * p[float(t_int - half)]
            else:
                # build from largest available key
                avail = max(k for k in p if k <= t and k == int(k))
                acc = p[avail]
                rem = t_int - int(avail)
                while rem > 0:
                    step = max(k for k in p if k <= rem and k == int(k) and k > 0)
                    acc = acc * p[step]
                    rem -= int(step)
                p[t] = acc
        else:
            base = float(int(t))
            frac = t - base
            if base not in p:
                # ensure integer part exists (recursion by construction order)
                raise RuntimeError(f"ladder ordering issue at t={t}")
            p[t] = p[base] * p[frac]
    return p


def _delta_ladder(delta, dmax):
    """delta^1..dmax by successive multiplication (dict of int -> value)."""
    p = {0: jnp.ones_like(delta), 1: delta}
    for k in range(2, dmax + 1):
        p[k] = p[k - 1] * delta
    return p


# ═════════════════════════════════════════════════════════════════════════
# Full 6-derivative bundle, economized
# ═════════════════════════════════════════════════════════════════════════

def residual_derivs_econ(tau, delta):
    """(ar, ar_d, ar_t, ar_dd, ar_tt, ar_dt) — exact SW96, economized."""
    inv_d = 1.0 / delta
    inv_d2 = inv_d * inv_d
    inv_t = 1.0 / tau
    inv_t2 = inv_t * inv_t

    tp = _tau_ladder(tau)
    dp = _delta_ladder(delta, 10)
    # 6 distinct exponential envelopes E_l = exp(-delta^l), l = 1..6
    E = {0: jnp.ones_like(delta)}
    for l in range(1, 7):
        E[l] = jnp.exp(-dp[l])

    ar = ar_d = ar_t = ar_dd = ar_tt = ar_dt = 0.0
    # ── polynomial + exponential terms (static unroll, coefficients folded) ──
    for k in range(len(_N)):
        n, d, t, l = float(_N[k]), int(_D[k]), float(_T[k]), int(_L[k])
        Tk = n * dp[d] * tp[t] * E[l]
        ld = l * dp[l] if l else 0.0         # l * delta^l  (0 when l = 0)
        D1 = (d - ld) * inv_d
        D1p = (-d - (l * (l - 1) * dp[l] if l else 0.0)) * inv_d2
        ar += Tk
        ar_d += Tk * D1
        ar_t += Tk * (t * inv_t)
        ar_dd += Tk * (D1 * D1 + D1p)
        ar_tt += Tk * (t * (t - 1.0) * inv_t2)
        ar_dt += Tk * D1 * (t * inv_t)

    # ── Gaussian terms: 4 distinct exponentials ──
    dme = delta - 1.0                        # eps = 1 for all five terms
    dme2 = dme * dme
    tm = [tau - float(g) for g in _GGAMMA]
    e_gauss = {}
    for k in range(5):
        key = (float(_GETA[k]), float(_GBETA[k]), float(_GGAMMA[k]))
        if key not in e_gauss:
            e_gauss[key] = jnp.exp(-key[0] * dme2 - key[1] * tm[k] * tm[k])
    for k in range(5):
        n, d, t = float(_GN[k]), int(_GD[k]), float(_GT[k])
        eta, beta = float(_GETA[k]), float(_GBETA[k])
        key = (eta, beta, float(_GGAMMA[k]))
        G = n * dp[d] * tp[float(t)] * e_gauss[key]
        Gd = d * inv_d - 2.0 * eta * dme
        Gdp = -d * inv_d2 - 2.0 * eta
        Gt = t * inv_t - 2.0 * beta * tm[k]
        Gtp = -t * inv_t2 - 2.0 * beta
        ar += G
        ar_d += G * Gd
        ar_t += G * Gt
        ar_dd += G * (Gd * Gd + Gdp)
        ar_tt += G * (Gt * Gt + Gtp)
        ar_dt += G * Gd * Gt

    # ── Non-analytic terms: shared theta; 2 distinct Delta; sqrt-ladder F ──
    s = dme * dme
    tm1 = tau - 1.0
    s2 = s * s
    s3 = s2 * s
    abs_dm1 = jnp.abs(dme)
    s_p = s ** _NA_P                          # s^(5/3)   (1 pow)
    s_pm1 = s ** (_NA_P - 1.0)                # s^(2/3)   (1 pow)
    sa = {3.5: s3 * abs_dm1, 3.0: s3}         # s^a by multiplies
    sam1 = {3.5: s2 * abs_dm1, 3.0: s2}       # s^(a-1)

    theta = -tm1 + _NA_BIGA * s_p
    th_d = 2.0 * _NA_BIGA * _NA_P * dme * s_pm1
    th_dd = 2.0 * _NA_BIGA * _NA_P * (2.0 * _NA_P - 1.0) * s_pm1
    th2 = theta * theta
    De_t = -2.0 * theta
    De_tt = 2.0

    exp_tm1 = jnp.exp(-_NA_BIGD * tm1 * tm1)
    e_cs = {}
    for k in range(3):
        C = float(_NA_BIGC[k])
        if C not in e_cs:
            e_cs[C] = jnp.exp(-C * s)

    def F_ladder78(Db):
        """Db^(7/8) by 3 sqrts + 3 multiplies."""
        r8 = jnp.sqrt(jnp.sqrt(jnp.sqrt(Db)))
        q2 = r8 * r8
        q4 = q2 * q2
        return q4 * q2 * r8

    Delta_cache = {}
    for k in range(3):
        a, b = float(_NAA[k]), float(_NAB[k])
        B, C = float(_NA_BIGB[k]), float(_NA_BIGC[k])
        dkey = (a, B)
        if dkey not in Delta_cache:
            Delta = th2 + B * sa[a]
            Db = jnp.maximum(Delta, 1e-300)
            De_d = 2.0 * theta * th_d + 2.0 * B * a * dme * sam1[a]
            De_dd = (2.0 * th_d * th_d + 2.0 * theta * th_dd
                     + 2.0 * B * a * (2.0 * a - 1.0) * sam1[a])
            De_dt = -2.0 * th_d
            Delta_cache[dkey] = (Db, De_d, De_dd, De_dt)
        Db, De_d, De_dd, De_dt = Delta_cache[dkey]

        F = F_ladder78(Db) if b == 0.875 else Db ** b
        Fm1 = F / Db
        Fm2 = Fm1 / Db
        F_d = b * Fm1 * De_d
        F_t = b * Fm1 * De_t
        F_dd = b * (b - 1.0) * Fm2 * De_d * De_d + b * Fm1 * De_dd
        F_tt = b * (b - 1.0) * Fm2 * De_t * De_t + b * Fm1 * De_tt
        F_dt = b * (b - 1.0) * Fm2 * De_d * De_t + b * Fm1 * De_dt

        Psi = e_cs[C] * exp_tm1
        Ps_d = -2.0 * C * dme * Psi
        Ps_t = -2.0 * _NA_BIGD * tm1 * Psi
        Ps_dd = (4.0 * C * C * s - 2.0 * C) * Psi
        Ps_tt = (4.0 * _NA_BIGD * _NA_BIGD * tm1 * tm1 - 2.0 * _NA_BIGD) * Psi
        Ps_dt = 4.0 * C * _NA_BIGD * dme * tm1 * Psi

        n = float(_NAN_[k])
        W = F * delta * Psi
        W_d = F_d * delta * Psi + F * Psi + F * delta * Ps_d
        W_t = F_t * delta * Psi + F * delta * Ps_t
        W_dd = (F_dd * delta * Psi + 2.0 * F_d * Psi + 2.0 * F_d * delta * Ps_d
                + 2.0 * F * Ps_d + F * delta * Ps_dd)
        W_tt = F_tt * delta * Psi + 2.0 * F_t * delta * Ps_t + F * delta * Ps_tt
        W_dt = (F_dt * delta * Psi + F_d * delta * Ps_t + F_t * Psi + F * Ps_t
                + F_t * delta * Ps_d + F * delta * Ps_dt)

        ar += n * W
        ar_d += n * W_d
        ar_t += n * W_t
        ar_dd += n * W_dd
        ar_tt += n * W_tt
        ar_dt += n * W_dt

    return ar, ar_d, ar_t, ar_dd, ar_tt, ar_dt


# ═════════════════════════════════════════════════════════════════════════
# Newton inner loop: delta-invariant prep + tau-only fast eval, economized
# ═════════════════════════════════════════════════════════════════════════

# Distinct tau exponents of the poly-exp part, and per-exponent delta-side
# coefficient assembly:  ar_poly(tau; delta) = sum_j Ct[j] * tau^t_j
_TPOLY = sorted(set(_T.tolist()))
_TP_IDX = {t: j for j, t in enumerate(_TPOLY)}
_W_T = np.array([t for t in _TPOLY])
_W_TT = np.array([t * (t - 1.0) for t in _TPOLY])


def residual_tau_prep_econ(delta):
    """delta-invariant state for the tau-only inner loop (economized)."""
    dp = _delta_ladder(delta, 10)
    E = {0: jnp.ones_like(delta)}
    for l in range(1, 7):
        E[l] = jnp.exp(-dp[l])
    # coefficient of tau^t_j:  sum over terms with that t of n*delta^d*E_l
    Ct = [0.0] * len(_TPOLY)
    for k in range(len(_N)):
        Ct[_TP_IDX[float(_T[k])]] += float(_N[k]) * dp[int(_D[k])] * E[int(_L[k])]
    Ct = jnp.stack([jnp.asarray(c) for c in Ct])

    dme = delta - 1.0
    dme2 = dme * dme
    Cg = jnp.stack([float(_GN[k]) * dp[int(_GD[k])]
                    * jnp.exp(-float(_GETA[k]) * dme2) for k in range(5)])

    s = dme2
    s_p = s ** _NA_P
    Bsa = {}
    for k in range(3):
        a, B = float(_NAA[k]), float(_NA_BIGB[k])
        if (a, B) not in Bsa:
            sa = s * s * s * (jnp.abs(dme) if a == 3.5 else 1.0)
            Bsa[(a, B)] = B * sa
    e_cs = {float(C): jnp.exp(-float(C) * s) for C in set(_NA_BIGC.tolist())}
    return Ct, Cg, s_p, Bsa, e_cs, delta


def residual_tau_fast_econ(tau, dstate):
    """(ar_t, ar_tt) from precomputed delta-state — economized per-iteration."""
    Ct, Cg, s_p, Bsa, e_cs, delta = dstate
    inv_t = 1.0 / tau
    inv_t2 = inv_t * inv_t

    tp = _tau_ladder(tau)
    taupows = jnp.stack([tp[t] for t in _TPOLY])
    base = Ct * taupows
    p_t = jnp.sum(base * jnp.asarray(_W_T)) * inv_t
    p_tt = jnp.sum(base * jnp.asarray(_W_TT)) * inv_t2

    # Gaussian: 4 distinct tau exponentials
    g_t = 0.0
    g_tt = 0.0
    e_g = {}
    for k in range(5):
        beta, gamma = float(_GBETA[k]), float(_GGAMMA[k])
        if (beta, gamma) not in e_g:
            tmg = tau - gamma
            e_g[(beta, gamma)] = (tmg, jnp.exp(-beta * tmg * tmg))
    for k in range(5):
        t, beta, gamma = float(_GT[k]), float(_GBETA[k]), float(_GGAMMA[k])
        tmg, eg = e_g[(beta, gamma)]
        gk = Cg[k] * tp[t] * eg
        Gt = t * inv_t - 2.0 * beta * tmg
        Gtp = -t * inv_t2 - 2.0 * beta
        g_t += gk * Gt
        g_tt += gk * (Gt * Gt + Gtp)

    # Non-analytic: one exp + two sqrt-ladders + one pow per iteration
    tm1 = tau - 1.0
    theta = -tm1 + _NA_BIGA * s_p
    De_t = -2.0 * theta
    De_tt = 2.0
    exp_tm1 = jnp.exp(-_NA_BIGD * tm1 * tm1)

    na_t = 0.0
    na_tt = 0.0
    F_cache = {}
    for k in range(3):
        a, b = float(_NAA[k]), float(_NAB[k])
        B, C = float(_NA_BIGB[k]), float(_NA_BIGC[k])
        dkey = (a, B)
        if dkey not in F_cache:
            Delta = theta * theta + Bsa[dkey]
            F_cache[dkey] = jnp.maximum(Delta, 1e-300)
        Db = F_cache[dkey]
        if b == 0.875:
            r8 = jnp.sqrt(jnp.sqrt(jnp.sqrt(Db)))
            q2 = r8 * r8
            q4 = q2 * q2
            F = q4 * q2 * r8
        else:
            F = Db ** b
        Fm1 = F / Db
        Fm2 = Fm1 / Db
        F_t = b * Fm1 * De_t
        F_tt = b * (b - 1.0) * Fm2 * De_t * De_t + b * Fm1 * De_tt
        Psi = e_cs[C] * exp_tm1
        Ps_t = -2.0 * _NA_BIGD * tm1 * Psi
        Ps_tt = (4.0 * _NA_BIGD * _NA_BIGD * tm1 * tm1 - 2.0 * _NA_BIGD) * Psi
        W_t = F_t * delta * Psi + F * delta * Ps_t
        W_tt = F_tt * delta * Psi + 2.0 * F_t * delta * Ps_t + F * delta * Ps_tt
        na_t += float(_NAN_[k]) * W_t
        na_tt += float(_NAN_[k]) * W_tt

    return p_t + g_t + na_t, p_tt + g_tt + na_tt


# ═════════════════════════════════════════════════════════════════════════
# B. chi_ref -> 1-D Chebyshev in delta
# ═════════════════════════════════════════════════════════════════════════

_TAU_REF = sw.TC / tr._T_REF
_CHI_DELTA_LO, _CHI_DELTA_HI = 1e-4, 2.7


def _dpdrho_ref_exact(delta):
    _, ar_d, _, ar_dd, _, _ = hz.residual_derivs(_TAU_REF, delta)
    return 1.0 + 2.0 * delta * ar_d + delta * delta * ar_dd


def fit_chi_ref_cheb(deg):
    k = np.arange(deg + 1)
    x = np.cos(np.pi * (k + 0.5) / (deg + 1))          # Chebyshev nodes
    d = 0.5 * (x + 1.0) * (_CHI_DELTA_HI - _CHI_DELTA_LO) + _CHI_DELTA_LO
    y = np.asarray(jax.vmap(_dpdrho_ref_exact)(jnp.asarray(d)))
    c = np.polynomial.chebyshev.chebfit(x, y, deg)
    return jnp.asarray(c)


def chi_ref_cheb(delta, coefs):
    """Clenshaw evaluation of the fitted dp/drho|_(T_ref) factor."""
    x = (2.0 * delta - (_CHI_DELTA_LO + _CHI_DELTA_HI)) / (_CHI_DELTA_HI - _CHI_DELTA_LO)
    b1 = jnp.zeros_like(delta)
    b2 = jnp.zeros_like(delta)
    for c in coefs[:0:-1]:
        b1, b2 = 2.0 * x * b1 - b2 + c, b1
    return x * b1 - b2 + coefs[0]


# ═════════════════════════════════════════════════════════════════════════
# Validation + timing
# ═════════════════════════════════════════════════════════════════════════

def _grid(n=300):
    """(tau, delta) covering envelope + margins incl. tight near-critical."""
    rng = np.random.default_rng(3)
    T = np.concatenate([
        rng.uniform(220.0, 800.0, n),
        rng.uniform(290.0, 350.0, n),
        304.1282 + rng.uniform(-0.5, 0.5, n),
    ])
    rho = np.concatenate([
        rng.uniform(1.0, 1200.0, n),
        rng.uniform(60.0, 700.0, n),
        467.6 + rng.uniform(-30.0, 30.0, n),
    ])
    return jnp.asarray(sw.TC / T), jnp.asarray(rho / sw.RHOC)


def time_call(fn, *args, repeat=200):
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    tau, delta = _grid()

    # ── validate the full bundle ──
    ref = jax.jit(jax.vmap(hz.residual_derivs))(tau, delta)
    new = jax.jit(jax.vmap(residual_derivs_econ))(tau, delta)
    names = ["ar", "ar_d", "ar_t", "ar_dd", "ar_tt", "ar_dt"]
    print("bundle econ vs shipped (max |diff| / scale):")
    for nm, a, b in zip(names, ref, new):
        a, b = np.asarray(a), np.asarray(b)
        scale = np.maximum(np.abs(a), 1e-6)
        print(f"  {nm:>6}: {np.max(np.abs(a - b) / scale):.3e}")

    # ── validate the tau-only pair ──
    def tau_pair_old(t, d):
        return hz.residual_tau_fast(t, hz.residual_tau_prep(d))

    def tau_pair_new(t, d):
        return residual_tau_fast_econ(t, residual_tau_prep_econ(d))

    r_old = jax.jit(jax.vmap(tau_pair_old))(tau, delta)
    r_new = jax.jit(jax.vmap(tau_pair_new))(tau, delta)
    for nm, a, b in zip(["ar_t", "ar_tt"], r_old, r_new):
        a, b = np.asarray(a), np.asarray(b)
        scale = np.maximum(np.abs(a), 1e-6)
        print(f"  tau-only {nm:>6}: {np.max(np.abs(a - b) / scale):.3e}")

    # ── chi_ref Chebyshev ──
    dgrid = jnp.asarray(np.linspace(_CHI_DELTA_LO, _CHI_DELTA_HI, 200001))
    exact = jax.jit(jax.vmap(_dpdrho_ref_exact))(dgrid)
    for deg in (20, 30, 40, 50):
        c = fit_chi_ref_cheb(deg)
        approx = jax.jit(lambda d: chi_ref_cheb(d, c))(dgrid)
        rel = np.max(np.abs(np.asarray(approx) - np.asarray(exact))
                     / np.abs(np.asarray(exact)))
        print(f"  chi_ref cheb deg={deg}: max rel err = {rel:.3e}")

    # ── timing: bundle + Newton loop ──
    print("\ntimings (us/call, median):")
    for n in (64, 1024):
        tt, dd = tau[:n], delta[:n]
        f_old = jax.jit(jax.vmap(hz.residual_derivs))
        f_new = jax.jit(jax.vmap(residual_derivs_econ))
        t_old = time_call(f_old, tt, dd)
        t_new = time_call(f_new, tt, dd)
        print(f"  N={n:>5} bundle: shipped {t_old*1e6:8.1f}  econ {t_new*1e6:8.1f}"
              f"   ({t_old/t_new:.2f}x)")

        def newton_old(rho, u):
            dstate = hz.residual_tau_prep(rho / sw.RHOC)
            T = core._seed_T(rho, u)
            for _ in range(5):
                ta = sw.TC / T
                a0_t, a0_tt = hz.ideal_tau_only(ta)
                ar_t, ar_tt = hz.residual_tau_fast(ta, dstate)
                f = sw.R * sw.TC * (a0_t + ar_t) - u
                cv = -sw.R * ta ** 2 * (a0_tt + ar_tt)
                T = jnp.clip(T - f / cv, core._T_MIN, core._T_MAX)
            return T

        def newton_new(rho, u):
            dstate = residual_tau_prep_econ(rho / sw.RHOC)
            T = core._seed_T(rho, u)
            for _ in range(5):
                ta = sw.TC / T
                a0_t, a0_tt = hz.ideal_tau_only(ta)
                ar_t, ar_tt = residual_tau_fast_econ(ta, dstate)
                f = sw.R * sw.TC * (a0_t + ar_t) - u
                cv = -sw.R * ta ** 2 * (a0_tt + ar_tt)
                T = jnp.clip(T - f / cv, core._T_MIN, core._T_MAX)
            return T

        rng = np.random.default_rng(0)
        Ts = jnp.asarray(rng.uniform(290.0, 350.0, n))
        rs = jnp.asarray(rng.uniform(60.0, 700.0, n))
        us = sw.internal_energy(Ts, rs)
        fo = jax.jit(jax.vmap(newton_old))
        fn_ = jax.jit(jax.vmap(newton_new))
        # equality check
        d_newton = np.max(np.abs(np.asarray(fo(rs, us)) - np.asarray(fn_(rs, us))))
        t_o = time_call(fo, rs, us)
        t_n = time_call(fn_, rs, us)
        print(f"  N={n:>5} newton5: shipped {t_o*1e6:8.1f}  econ {t_n*1e6:8.1f}"
              f"   ({t_o/t_n:.2f}x)   maxdiff={d_newton:.2e} K")


if __name__ == "__main__":
    main()
