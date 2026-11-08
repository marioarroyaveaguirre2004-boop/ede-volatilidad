"""
validate.py — Monte Carlo contra fórmulas cerradas.

Uso:   python validate.py            (≈ 1 min)
       python validate.py --quick    (menos trayectorias)

Para cada modelo compara media y desviación de V_T con la teoría, y para cada
combinación (deriva × tendencia) compara E[X_T] con la fórmula exacta.
Sale con código 1 si algo se aleja más de 4 errores estándar (o 3 % relativo).
"""
from __future__ import annotations

import sys

import numpy as np

from models import MODELS, SimConfig, make_noise, simulate, theory_V, theory_X_mean

QUICK = "--quick" in sys.argv
N = 8_000 if QUICK else 30_000
SEED = 2026


def default_cfg(model: str, **kw) -> SimConfig:
    vp = {p.key: p.default for p in MODELS[model].params}
    kw.setdefault("rho", -0.5 if model != "hull_white" else 0.0)
    return SimConfig(model=model, vp=vp, **kw)


def se_std(x: np.ndarray) -> float:
    """Error estándar de la desviación muestral, usando la curtosis muestral
    (importante en colas pesadas como la log-normal de Hull-White)."""
    s = x.std()
    k = ((x - x.mean()) ** 4).mean() / s**4
    return s * np.sqrt(max(k - 1.0, 1e-9) / (4 * len(x)))


def trend_params(drift: str, trend: str, x0: float = 100.0) -> dict:
    lvl = 1.0 if drift == "usual" else x0
    if trend == "const":
        return {"mu": 0.06 if drift == "usual" else 1.2 * x0}
    if trend == "periodic":
        return ({"mu": 0.05, "amp": 0.30, "period": 1.0, "phase": 0.0} if drift == "usual"
                else {"mu": 1.1 * x0, "amp": 0.3 * x0, "period": 1.0, "phase": 0.0})
    return ({"mu0": 0.05, "m": 0.10, "a": 1.5, "s": 0.20} if drift == "usual"
            else {"mu0": x0, "m": 1.2 * x0, "a": 1.5, "s": 0.3 * x0})


fails = 0


def check(label: str, theo: float, sim: float, se: float, rel_tol: float = 0.03) -> None:
    global fails
    z = (sim - theo) / se if se > 0 else 0.0
    rel = abs(sim - theo) / max(abs(theo), 1e-12)
    ok = abs(z) < 4 or rel < rel_tol * 0.3   # 4 SE, o muy cerca en términos relativos
    ok = ok and rel < rel_tol * 3
    fails += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {label:<34} teoría={theo:>12.6f}  sim={sim:>12.6f}  z={z:+5.2f}  rel={rel:6.2%}")


print(f"N = {N} trayectorias, semilla {SEED}\n")

# ── 1. Momentos de V_T ───────────────────────────────────────────────────────
print("1) Momentos de V_T")
for key in MODELS:
    cfg = default_cfg(key, T=1.0, steps_per_year=252,
                      drift="usual", trend="const", tp={"mu": 0.05})
    Z = make_noise(SEED, N, cfg.n_steps)
    res = simulate(cfg, Z)
    th = theory_V(cfg, res.t)
    vT = res.V[-1]
    print(f" {MODELS[key].name}")
    if th is None:
        print("  · sin forma cerrada (solo simulación): "
              f"E[V_T]≈{vT.mean():.5f}, SD≈{vT.std():.5f}, mín={vT.min():.5f}")
        continue
    mean, var = th
    check("E[V_T]", mean[-1], vT.mean(), vT.std() / np.sqrt(N))
    check("SD[V_T]", np.sqrt(var[-1]), vT.std(), se_std(vT), rel_tol=0.04)

# ── 2. E[X_T] para todas las combinaciones ───────────────────────────────────
print("\n2) E[X_T] — deriva × tendencia (modelo Heston, ρ = −0.5)")
for drift in ("usual", "mr"):
    for trend in ("const", "periodic", "stoch"):
        cfg = default_cfg("heston", T=2.0, steps_per_year=252, drift=drift,
                          alpha=3.0, trend=trend, tp=trend_params(drift, trend))
        Z = make_noise(SEED + 1, N, cfg.n_steps)
        res = simulate(cfg, Z)
        theo = theory_X_mean(cfg, res.t)[-1]
        xT = res.X[-1]
        check(f"{drift:<5} + {trend:<8}", theo, xT.mean(), xT.std() / np.sqrt(N))

print("\n3) E[X_T] en los otros modelos (deriva usual, μ constante, ρ = 0)")
for key in ("hull_white", "stein_stein", "nelson", "scott"):
    cfg = default_cfg(key, T=1.0, drift="usual", trend="const", tp={"mu": 0.06}, rho=0.0)
    Z = make_noise(SEED + 2, N, cfg.n_steps)
    res = simulate(cfg, Z)
    xT = res.X[-1]
    check(MODELS[key].name, theory_X_mean(cfg, res.t)[-1], xT.mean(), xT.std() / np.sqrt(N))

# ── 3. Invariantes numéricos ─────────────────────────────────────────────────
print("\n4) Robustez numérica (parámetros extremos, sin NaN/inf)")
for key, spec in MODELS.items():
    vp = {p.key: p.hi for p in spec.params}
    cfg = SimConfig(model=key, vp=vp, v0=0.5, rho=1.0, T=5.0, steps_per_year=52,
                    drift="mr", alpha=10.0, trend="stoch",
                    tp={"mu0": 100.0, "m": 150.0, "a": 10.0, "s": 100.0})
    res = simulate(cfg, make_noise(7, 400, cfg.n_steps))
    bad = not (np.isfinite(res.X).all() and np.isfinite(res.V).all())
    fails += int(bad)
    print(f"  {'✗' if bad else '✓'} {spec.name:<14} extremos → "
          f"{'hay NaN/inf' if bad else 'todo finito'}")

print("\n" + ("TODO OK" if fails == 0 else f"{fails} comprobaciones fallaron"))
sys.exit(1 if fails else 0)
