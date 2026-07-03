"""Span-Wagner (1996) reduced Helmholtz energy with hand-coded analytic
derivatives — pure JAX, transcendental-economized.

This is the numerical core of the EOS hot path.  It computes α and all the
derivatives a property/inversion pass needs — α_δ, α_τ, α_δδ, α_ττ, α_δτ —
analytically, and exploits the *structure* of the Span-Wagner exponents so the
per-point transcendental count collapses:

  * every τ-exponent in the polynomial/exponential/Gaussian terms is a
    multiple of ¼ → all τ-powers come from ``sqrt(sqrt(τ))`` plus a multiply
    ladder (zero ``pow`` calls);
  * every δ-exponent is a small integer (d ≤ 10, l ≤ 6) → multiply ladders;
  * the 27 ``exp(-δ^l)`` envelopes take only six distinct values (l = 1..6);
  * the five Gaussian exponentials share (η, β, γ) pairwise → four distinct;
  * the non-analytic terms share θ (A, β identical across terms) and terms
    40/41 share Δ; Δ^(7/8) is a sqrt ladder; only Δ^0.925, s^(5/3) and
    s^(2/3) remain as real ``pow`` calls.

The structural facts are asserted at import against the coefficient tables in
``span_wagner`` (the single source of truth) — if the tables ever change shape,
import fails loudly rather than computing a stale factorization.

The result is the same Span-Wagner sum re-associated: agreement with the
autodiff derivatives of ``span_wagner.alphar`` / ``alpha0`` is round-off level
(~1e-12; see ``tests/test_analytic_derivs.py``), and every function remains
JIT-compilable, vmappable and differentiable to higher order.

Convention: τ = Tc/T (reduced inverse temperature), δ = ρ/ρc (reduced
density).  Derivatives are plain partials w.r.t. τ and δ (not the δ·∂/∂δ
reduced form), matching the formulas in ``span_wagner``.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Pull the coefficient tables straight from span_wagner so there is a single
# source of truth for the constants.
from co2_eos.span_wagner import (
    _A0_A1, _A0_A2, _A0_LOGTAU, _A0_PE_N, _A0_PE_T,
    _AR_N, _AR_D, _AR_T, _AR_L,
    _AR_GAUSS_N, _AR_GAUSS_D, _AR_GAUSS_T,
    _AR_GAUSS_ETA, _AR_GAUSS_BETA, _AR_GAUSS_GAMMA, _AR_GAUSS_EPS,
    _NA_N, _NA_A, _NA_B, _NA_BETA, _NA_BIG_A, _NA_BIG_B, _NA_BIG_C, _NA_BIG_D,
)

# ── Static (Python-scalar) views of the tables — folded into the trace ─────
_PN = np.asarray(_AR_N)
_PD = np.asarray(_AR_D).astype(int)
_PT = np.asarray(_AR_T)
_PL = np.asarray(_AR_L).astype(int)

_GN = np.asarray(_AR_GAUSS_N)
_GD = np.asarray(_AR_GAUSS_D).astype(int)
_GT = np.asarray(_AR_GAUSS_T)
_GETA = np.asarray(_AR_GAUSS_ETA)
_GBETA = np.asarray(_AR_GAUSS_BETA)
_GGAMMA = np.asarray(_AR_GAUSS_GAMMA)

_NAN = np.asarray(_NA_N)
_NAA = np.asarray(_NA_A)
_NAB = np.asarray(_NA_B)
_NABIGB = np.asarray(_NA_BIG_B)
_NABIGC = np.asarray(_NA_BIG_C)

# Structural facts the economization relies on — asserted, not assumed.
assert np.all(_PD == np.asarray(_AR_D)) and _PD.max() <= 10, "δ-exponent not a small integer"
assert np.all(_PL == np.asarray(_AR_L)) and _PL.max() <= 6, "l-exponent not a small integer"
assert np.all(np.abs(_PT * 4 - np.round(_PT * 4)) < 1e-12), "τ-exponent not a multiple of ¼"
assert np.all(np.abs(_GT * 4 - np.round(_GT * 4)) < 1e-12)
assert np.all(np.asarray(_AR_GAUSS_EPS) == 1.0), "Gaussian ε ≠ 1"
_NA_P_SCALAR = float(1.0 / (2.0 * np.asarray(_NA_BETA)[0]))
assert np.all(np.asarray(_NA_BETA) == np.asarray(_NA_BETA)[0]), "NA β not shared"
_NA_BIGA_SCALAR = float(np.asarray(_NA_BIG_A)[0])
assert np.all(np.asarray(_NA_BIG_A) == _NA_BIGA_SCALAR), "NA A not shared"
_NA_BIGD_SCALAR = float(np.asarray(_NA_BIG_D)[0])
assert np.all(np.asarray(_NA_BIG_D) == _NA_BIGD_SCALAR), "NA D not shared"
assert set(np.round(_NAA * 2).astype(int)) <= {6, 7}, "NA a must be 3 or 3.5"
assert _NAB[0] == 0.875 and _NAB[2] == 0.875, "Δ^(7/8) sqrt ladder assumption"

# Reciprocal of (2β) for the non-analytic θ exponent, precomputed (array view
# kept for external callers/tests; the kernels use the shared scalar).
_NA_P = 1.0 / (2.0 * _NA_BETA)

# Distinct τ-exponents of the polynomial+exponential terms, and per-exponent
# static weights for the τ-derivative dot products.
_TPOLY = sorted(set(_PT.tolist()))
_TP_IDX = {t: j for j, t in enumerate(_TPOLY)}
_TALL = sorted(set(_TPOLY) | set(_GT.tolist()))
_W_T = np.array(_TPOLY)                       # t_j
_W_TT = np.array([t * (t - 1.0) for t in _TPOLY])


def _tau_ladder(tau):
    """All distinct τ^t needed by the terms, via 2 sqrts + multiplies.

    Every exponent is a multiple of ¼, so τ^t = τ^int · (τ^¼)^frac with the
    integer parts built by a multiply chain over increasing exponents.
    """
    st = jnp.sqrt(tau)
    qt = jnp.sqrt(st)
    p = {0.0: jnp.ones_like(tau), 0.25: qt, 0.5: st, 0.75: st * qt, 1.0: tau}
    for t in _TALL:
        if t in p:
            continue
        if t == int(t):
            t_int = int(t)
            half = float(t_int // 2)
            other = float(t_int - t_int // 2)
            if half in p and other in p:
                p[t] = p[half] * p[other]
            else:
                avail = max(k for k in p if k == int(k) and 0 < k <= t)
                acc, rem = p[avail], t_int - int(avail)
                while rem > 0:
                    step = max(k for k in p if k == int(k) and 0 < k <= rem)
                    acc = acc * p[step]
                    rem -= int(step)
                p[t] = acc
        else:
            base, frac = float(int(t)), t - int(t)
            p[t] = p[base] * p[frac]        # base already built (sorted order)
    return p


def _delta_ladder(delta, dmax=10):
    """δ^0..δ^dmax by successive multiplication."""
    p = {0: jnp.ones_like(delta), 1: delta}
    for k in range(2, dmax + 1):
        p[k] = p[k - 1] * delta
    return p


def _exp_envelopes(dp):
    """The six distinct exponential envelopes E_l = exp(-δ^l), l = 0..6."""
    E = {0: jnp.ones_like(dp[1])}
    for l in range(1, 7):
        E[l] = jnp.exp(-dp[l])
    return E


# ═══════════════════════════════════════════════════════════════════════════
# Ideal-gas part α⁰ — τ derivatives (δ part is only ln δ, not needed here)
# ═══════════════════════════════════════════════════════════════════════════

def ideal_derivs(tau, delta):
    """Return (α⁰, α⁰_τ, α⁰_ττ) at scalar (τ, δ).

    α⁰ = a1 + a2·τ + L·ln τ + ln δ + Σ nk·ln(1 - exp(-θk·τ))
    Only the τ-derivatives and the value are used by the property/inversion
    formulas (the ideal δ-dependence cancels out of every measurable property
    except through the additive ln δ in the value, which entropy/Gibbs need).
    """
    L = _A0_LOGTAU
    e = jnp.exp(-_A0_PE_T * tau)            # exp(-θk τ)
    one_minus_e = 1.0 - e

    a0 = (_A0_A1 + _A0_A2 * tau + L * jnp.log(tau) + jnp.log(delta)
          + jnp.sum(_A0_PE_N * jnp.log(one_minus_e)))

    # d/dτ ln(1-e) = θ·e/(1-e)
    a0_t = (_A0_A2 + L / tau
            + jnp.sum(_A0_PE_N * _A0_PE_T * e / one_minus_e))

    # d²/dτ² ln(1-e) = -θ²·e/(1-e)²
    a0_tt = (-L / tau ** 2
             - jnp.sum(_A0_PE_N * _A0_PE_T ** 2 * e / one_minus_e ** 2))

    return a0, a0_t, a0_tt


def ideal_tau_only(tau):
    """Return (α⁰_τ, α⁰_ττ) — the only ideal quantities the Newton needs.

    Independent of δ, so cheaper than ``ideal_derivs`` for the inner loop.
    """
    L = _A0_LOGTAU
    e = jnp.exp(-_A0_PE_T * tau)
    one_minus_e = 1.0 - e
    a0_t = (_A0_A2 + L / tau
            + jnp.sum(_A0_PE_N * _A0_PE_T * e / one_minus_e))
    a0_tt = (-L / tau ** 2
             - jnp.sum(_A0_PE_N * _A0_PE_T ** 2 * e / one_minus_e ** 2))
    return a0_t, a0_tt


# ═══════════════════════════════════════════════════════════════════════════
# Non-analytic terms — shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pow78(Db):
    """Db^(7/8) by three sqrts and three multiplies (Db ≥ 1e-300 guarded)."""
    r8 = jnp.sqrt(jnp.sqrt(jnp.sqrt(Db)))
    q2 = r8 * r8
    q4 = q2 * q2
    return q4 * q2 * r8


def _na_delta_state(delta):
    """δ-side quantities of the non-analytic terms (loop-invariant at fixed ρ).

    Returns (s, dme, s_p, Bsa, e_cs) where Bsa[(a, B)] = B·s^a and
    e_cs[C] = exp(-C·s) for the distinct (a, B) / C among the three terms.
    """
    dme = delta - 1.0
    s = dme * dme
    s_p = s ** _NA_P_SCALAR                  # s^(5/3)
    s3 = s * s * s
    abs_dm1 = jnp.abs(dme)
    sa = {3.5: s3 * abs_dm1, 3.0: s3}        # s^a by multiplies
    Bsa = {}
    for k in range(3):
        key = (float(_NAA[k]), float(_NABIGB[k]))
        if key not in Bsa:
            Bsa[key] = key[1] * sa[key[0]]
    e_cs = {float(C): jnp.exp(-float(C) * s) for C in sorted(set(_NABIGC.tolist()))}
    return s, dme, s_p, Bsa, e_cs


# ═══════════════════════════════════════════════════════════════════════════
# Residual part αʳ — full analytic derivative bundle (one economized pass)
# ═══════════════════════════════════════════════════════════════════════════

def residual_derivs(tau, delta):
    """Return (αʳ, αʳ_δ, αʳ_τ, αʳ_δδ, αʳ_ττ, αʳ_δτ) at scalar (τ, δ).

    All six are accumulated from the same per-term values; the transcendental
    budget per point is ~12 exps + ~8 sqrts + 3 pows (see module docstring).
    """
    inv_d = 1.0 / delta
    inv_d2 = inv_d * inv_d
    inv_t = 1.0 / tau
    inv_t2 = inv_t * inv_t

    tp = _tau_ladder(tau)
    dp = _delta_ladder(delta)
    E = _exp_envelopes(dp)

    ar = ar_d = ar_t = ar_dd = ar_tt = ar_dt = 0.0

    # ── Polynomial + exponential terms (static unroll; constants folded) ──
    for k in range(len(_PN)):
        n, d, t, l = float(_PN[k]), int(_PD[k]), float(_PT[k]), int(_PL[k])
        Tk = n * dp[d] * tp[t] * E[l]
        ld = l * dp[l] if l else 0.0         # l·δ^l (0 when l = 0)
        D1 = (d - ld) * inv_d
        D1p = (-d - (l * (l - 1) * dp[l] if l else 0.0)) * inv_d2
        Tt = t * inv_t
        ar += Tk
        ar_d += Tk * D1
        ar_t += Tk * Tt
        ar_dd += Tk * (D1 * D1 + D1p)
        ar_tt += Tk * (t * (t - 1.0) * inv_t2)
        ar_dt += Tk * D1 * Tt

    # ── Gaussian bell-shaped terms (four distinct exponentials) ──
    dme = delta - 1.0                        # ε = 1 for all five terms
    dme2 = dme * dme
    e_g = {}
    for k in range(5):
        key = (float(_GETA[k]), float(_GBETA[k]), float(_GGAMMA[k]))
        if key not in e_g:
            tmg = tau - key[2]
            e_g[key] = (tmg, jnp.exp(-key[0] * dme2 - key[1] * tmg * tmg))
    for k in range(5):
        n, d, t = float(_GN[k]), int(_GD[k]), float(_GT[k])
        eta, beta = float(_GETA[k]), float(_GBETA[k])
        tmg, eg = e_g[(eta, beta, float(_GGAMMA[k]))]
        G = n * dp[d] * tp[t] * eg
        Gd = d * inv_d - 2.0 * eta * dme
        Gdp = -d * inv_d2 - 2.0 * eta
        Gt = t * inv_t - 2.0 * beta * tmg
        Gtp = -t * inv_t2 - 2.0 * beta
        ar += G
        ar_d += G * Gd
        ar_t += G * Gt
        ar_dd += G * (Gd * Gd + Gdp)
        ar_tt += G * (Gt * Gt + Gtp)
        ar_dt += G * Gd * Gt

    # ── Non-analytic terms (shared θ; terms 40/41 share Δ) ──
    s, dme, s_p, Bsa, e_cs = _na_delta_state(delta)
    s_pm1 = s ** (_NA_P_SCALAR - 1.0)        # s^(2/3)
    s2 = s * s
    sam1 = {3.5: s2 * jnp.abs(dme), 3.0: s2}  # s^(a-1) by multiplies
    tm1 = tau - 1.0

    theta = -tm1 + _NA_BIGA_SCALAR * s_p
    th_d = 2.0 * _NA_BIGA_SCALAR * _NA_P_SCALAR * dme * s_pm1
    th_dd = 2.0 * _NA_BIGA_SCALAR * _NA_P_SCALAR * (2.0 * _NA_P_SCALAR - 1.0) * s_pm1
    De_t = -2.0 * theta
    De_tt = 2.0
    exp_tm1 = jnp.exp(-_NA_BIGD_SCALAR * tm1 * tm1)

    Delta_cache = {}
    for k in range(3):
        a, b = float(_NAA[k]), float(_NAB[k])
        B, Cc = float(_NABIGB[k]), float(_NABIGC[k])
        dkey = (a, B)
        if dkey not in Delta_cache:
            Delta = theta * theta + Bsa[dkey]
            Db = jnp.maximum(Delta, 1e-300)
            De_d = 2.0 * theta * th_d + 2.0 * B * a * dme * sam1[a]
            De_dd = (2.0 * th_d * th_d + 2.0 * theta * th_dd
                     + 2.0 * B * a * (2.0 * a - 1.0) * sam1[a])
            De_dt = -2.0 * th_d
            Delta_cache[dkey] = (Db, De_d, De_dd, De_dt)
        Db, De_d, De_dd, De_dt = Delta_cache[dkey]

        F = _pow78(Db) if b == 0.875 else Db ** b
        Fm1 = F / Db
        Fm2 = Fm1 / Db
        F_d = b * Fm1 * De_d
        F_t = b * Fm1 * De_t
        F_dd = b * (b - 1.0) * Fm2 * De_d * De_d + b * Fm1 * De_dd
        F_tt = b * (b - 1.0) * Fm2 * De_t * De_t + b * Fm1 * De_tt
        F_dt = b * (b - 1.0) * Fm2 * De_d * De_t + b * Fm1 * De_dt

        Psi = e_cs[Cc] * exp_tm1
        Ps_d = -2.0 * Cc * dme * Psi
        Ps_t = -2.0 * _NA_BIGD_SCALAR * tm1 * Psi
        Ps_dd = (4.0 * Cc * Cc * s - 2.0 * Cc) * Psi
        Ps_tt = (4.0 * _NA_BIGD_SCALAR * _NA_BIGD_SCALAR * tm1 * tm1
                 - 2.0 * _NA_BIGD_SCALAR) * Psi
        Ps_dt = 4.0 * Cc * _NA_BIGD_SCALAR * dme * tm1 * Psi

        n = float(_NAN[k])
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


# ═══════════════════════════════════════════════════════════════════════════
# τ-only inner-loop pair: δ-invariant prep + per-iteration fast eval
# ═══════════════════════════════════════════════════════════════════════════

def residual_tau_prep(delta):
    """Precompute the δ-invariant state for the τ-derivative inner loop.

    At fixed ρ (the Newton solves T at fixed density) every δ-dependent power
    and exponential is loop-invariant.  The polynomial+exponential terms
    collapse to one coefficient per distinct τ-exponent:
    αʳ_poly(τ; δ) = Σ_j Ct[j]·τ^t_j with Ct[j] = Σ_{terms with t_j} n·δ^d·E_l.
    """
    dp = _delta_ladder(delta)
    E = _exp_envelopes(dp)
    Ct = [0.0] * len(_TPOLY)
    for k in range(len(_PN)):
        Ct[_TP_IDX[float(_PT[k])]] += float(_PN[k]) * dp[int(_PD[k])] * E[int(_PL[k])]
    Ct = jnp.stack([jnp.asarray(c) for c in Ct])

    dme = delta - 1.0
    dme2 = dme * dme
    Cg = jnp.stack([float(_GN[k]) * dp[int(_GD[k])]
                    * jnp.exp(-float(_GETA[k]) * dme2) for k in range(5)])

    _, _, s_p, Bsa, e_cs = _na_delta_state(delta)
    return Ct, Cg, s_p, Bsa, e_cs, delta


def residual_tau_fast(tau, dstate):
    """Return (αʳ_τ, αʳ_ττ) from precomputed δ-state ``dstate``.

    Identical result to ``residual_tau_derivs(tau, delta)`` but per call only
    the τ-side transcendentals are evaluated: ~6 exps + 8 sqrts + 1 pow.
    """
    Ct, Cg, s_p, Bsa, e_cs, delta = dstate
    inv_t = 1.0 / tau
    inv_t2 = inv_t * inv_t

    tp = _tau_ladder(tau)
    taupows = jnp.stack([tp[t] for t in _TPOLY])
    base = Ct * taupows
    p_t = jnp.sum(base * _W_T) * inv_t
    p_tt = jnp.sum(base * _W_TT) * inv_t2

    g_t = 0.0
    g_tt = 0.0
    e_g = {}
    for k in range(5):
        key = (float(_GBETA[k]), float(_GGAMMA[k]))
        if key not in e_g:
            tmg = tau - key[1]
            e_g[key] = (tmg, jnp.exp(-key[0] * tmg * tmg))
    for k in range(5):
        t, beta = float(_GT[k]), float(_GBETA[k])
        tmg, eg = e_g[(beta, float(_GGAMMA[k]))]
        gk = Cg[k] * tp[t] * eg
        Gt = t * inv_t - 2.0 * beta * tmg
        Gtp = -t * inv_t2 - 2.0 * beta
        g_t += gk * Gt
        g_tt += gk * (Gt * Gt + Gtp)

    tm1 = tau - 1.0
    theta = -tm1 + _NA_BIGA_SCALAR * s_p
    De_t = -2.0 * theta
    De_tt = 2.0
    exp_tm1 = jnp.exp(-_NA_BIGD_SCALAR * tm1 * tm1)

    na_t = 0.0
    na_tt = 0.0
    Db_cache = {}
    for k in range(3):
        a, b = float(_NAA[k]), float(_NAB[k])
        B, Cc = float(_NABIGB[k]), float(_NABIGC[k])
        dkey = (a, B)
        if dkey not in Db_cache:
            Db_cache[dkey] = jnp.maximum(theta * theta + Bsa[dkey], 1e-300)
        Db = Db_cache[dkey]
        F = _pow78(Db) if b == 0.875 else Db ** b
        Fm1 = F / Db
        Fm2 = Fm1 / Db
        F_t = b * Fm1 * De_t
        F_tt = b * (b - 1.0) * Fm2 * De_t * De_t + b * Fm1 * De_tt
        Psi = e_cs[Cc] * exp_tm1
        Ps_t = -2.0 * _NA_BIGD_SCALAR * tm1 * Psi
        Ps_tt = (4.0 * _NA_BIGD_SCALAR * _NA_BIGD_SCALAR * tm1 * tm1
                 - 2.0 * _NA_BIGD_SCALAR) * Psi
        W_t = F_t * delta * Psi + F * delta * Ps_t
        W_tt = F_tt * delta * Psi + 2.0 * F_t * delta * Ps_t + F * delta * Ps_tt
        na_t += float(_NAN[k]) * W_t
        na_tt += float(_NAN[k]) * W_tt

    return p_t + g_t + na_t, p_tt + g_tt + na_tt


def residual_tau_derivs(tau, delta):
    """Return (αʳ_τ, αʳ_ττ) — the residual quantities the Newton needs.

    Computes only the τ-derivatives; the δ-dependent envelopes are
    loop-invariant at fixed ρ and XLA hoists them across an unrolled Newton.
    """
    return residual_tau_fast(tau, residual_tau_prep(delta))
