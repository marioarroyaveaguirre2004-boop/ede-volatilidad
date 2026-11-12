"""
models.py — Motor numérico del laboratorio de volatilidad estocástica (EDE).

Todos los modelos comparten la misma ecuación de precio

    usual :  dX = μ_t X dt + √V_t X dB¹
    redef.:  dX = α(μ_t − X) dt + √V_t X dB²          (reversión a la media)

y se distinguen por la ecuación de la varianza V_t:

    Hull & White : dV = θ V dt + σ V dB²
    Stein & Stein: dV = κ(θ − V) dt + σ dB²
    Nelson       : dV = κ(θ − V) dt + σ V dB²
    Heston       : dV = κ(θ − V) dt + σ √V dB²
    Scott        : dV = κ(θ − (√V − η)²) dt + σ √V dB²

con d⟨B¹,B²⟩ = ρ dt.  La tendencia μ_t puede ser constante, una función
periódica del tiempo o un proceso estocástico (Ornstein-Uhlenbeck).

El módulo no depende de Streamlit: se puede importar desde un notebook o desde
los scripts de validación.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.linalg import expm


# ═════════════════════════════════════════════════════════════════════════════
#  Especificación de los modelos (la UI se genera a partir de aquí)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    lo: float
    hi: float
    default: float
    step: float
    help: str = ""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    v_latex: str          # ecuación de V_t (LaTeX)
    params: tuple[ParamSpec, ...]
    summary: str          # una línea para la guía
    mean_reverting: bool  # ¿V_t revierte a la media?
    positivity: str       # qué pasa con V_t ≥ 0
    closed_form: str      # qué momentos de V_t son exactos


_kappa = ParamSpec("kappa", r"$\kappa$ — velocidad de reversión", 0.1, 10.0, 2.0, 0.1,
                   "Qué tan rápido V_t es jalada hacia θ.")
_theta_lr = ParamSpec("theta", r"$\theta$ — nivel de largo plazo de $V$", 0.005, 0.30, 0.04, 0.005,
                      "Varianza de largo plazo (0.04 ⇒ volatilidad 20 %).")

MODELS: dict[str, ModelSpec] = {
    "hull_white": ModelSpec(
        key="hull_white",
        name="Hull & White",
        v_latex=r"dV_t = \theta\,V_t\,dt + \sigma\,V_t\,dB^{(2)}_t",
        params=(
            ParamSpec("theta", r"$\theta$ — tasa de crecimiento de $V$", -1.0, 1.0, 0.0, 0.05,
                      "V_t es un movimiento browniano geométrico: E[V_t] = V₀ e^{θt}."),
            ParamSpec("sigma", r"$\sigma$ — volatilidad de la varianza", 0.0, 2.0, 0.8, 0.05,
                      "Qué tan errática es la varianza."),
        ),
        summary="Varianza log-normal (MBG). No revierte a la media.",
        mean_reverting=False,
        positivity="V_t > 0 siempre (exponencial de algo).",
        closed_form="Media, varianza y densidad de V_t (log-normal).",
    ),
    "stein_stein": ModelSpec(
        key="stein_stein",
        name="Stein & Stein",
        v_latex=r"dV_t = \kappa(\theta - V_t)\,dt + \sigma\,dB^{(2)}_t",
        params=(
            _kappa, _theta_lr,
            ParamSpec("sigma", r"$\sigma$ — volatilidad de la varianza", 0.0, 0.5, 0.06, 0.005,
                      "Ruido aditivo constante: V_t es un proceso de Ornstein-Uhlenbeck."),
        ),
        summary="Vₜ es un Ornstein-Uhlenbeck (ruido aditivo): gaussiana, con reversión.",
        mean_reverting=True,
        positivity="V_t puede ser negativa; en X_t se usa √max(V_t, 0).",
        closed_form="Media, varianza y densidad de V_t (normal).",
    ),
    "nelson": ModelSpec(
        key="nelson",
        name="Nelson",
        v_latex=r"dV_t = \kappa(\theta - V_t)\,dt + \sigma\,V_t\,dB^{(2)}_t",
        params=(
            _kappa, _theta_lr,
            ParamSpec("sigma", r"$\sigma$ — volatilidad de la varianza", 0.0, 3.0, 1.0, 0.05,
                      "Ruido proporcional a V_t (difusión GARCH)."),
        ),
        summary="Reversión a la media con ruido multiplicativo (límite continuo del GARCH).",
        mean_reverting=True,
        positivity="V_t > 0 si θ > 0 (el esquema usado preserva la positividad).",
        closed_form="Media y varianza de V_t (sistema lineal de momentos).",
    ),
    "heston": ModelSpec(
        key="heston",
        name="Heston",
        v_latex=r"dV_t = \kappa(\theta - V_t)\,dt + \sigma\sqrt{V_t}\,dB^{(2)}_t",
        params=(
            _kappa, _theta_lr,
            ParamSpec("sigma", r"$\sigma$ — volatilidad de la varianza", 0.01, 1.5, 0.3, 0.01,
                      "Ruido proporcional a √V (proceso CIR)."),
        ),
        summary="Vₜ es un CIR: reversión a la media y ruido ∝ √V. El modelo de referencia.",
        mean_reverting=True,
        positivity="V_t ≥ 0 en el modelo continuo; en el esquema se trunca a 0.",
        closed_form="Media, varianza y densidad de V_t (χ² no central).",
    ),
    "scott": ModelSpec(
        key="scott",
        name="Scott",
        v_latex=(r"dV_t = \kappa\big(\theta - (\sqrt{V_t}-\eta)^2\big)\,dt"
                 r" + \sigma\sqrt{V_t}\,dB^{(2)}_t"),
        params=(
            _kappa,
            ParamSpec("theta", r"$\theta$ — amplitud de la reversión", 0.001, 0.20, 0.0225, 0.001,
                      "Los equilibrios de √V están en η ± √θ."),
            ParamSpec("eta", r"$\eta$ — desplazamiento de $\sqrt{V}$", 0.0, 0.5, 0.05, 0.005,
                      "Centro de la parábola: la deriva se anula en √V = η ± √θ."),
            ParamSpec("sigma", r"$\sigma$ — volatilidad de la varianza", 0.01, 1.5, 0.3, 0.01,
                      "Se asumió ruido σ√V (ver nota en la guía)."),
        ),
        summary="La reversión es cuadrática en √V: la volatilidad es jalada hacia η + √θ.",
        mean_reverting=True,
        positivity="V_t ≥ 0 en el modelo continuo; en el esquema se trunca a 0.",
        closed_form="Ninguno (la deriva depende de √V): solo simulación.",
    ),
}

# Modelos cuyo esquema numérico deja V_t con valores negativos que hay que truncar
_TRUNCATED = {"heston", "scott"}


# ═════════════════════════════════════════════════════════════════════════════
#  Configuración de una simulación
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SimConfig:
    model: str = "heston"
    vp: dict = field(default_factory=dict)      # parámetros de V (kappa, theta, sigma, eta)
    v0: float = 0.04
    x0: float = 100.0
    rho: float = 0.0
    T: float = 1.0
    steps_per_year: int = 252
    drift: str = "usual"                        # "usual" | "mr"
    alpha: float = 3.0                          # velocidad de reversión de X (solo drift="mr")
    trend: str = "const"                        # "const" | "periodic" | "stoch"
    tp: dict = field(default_factory=dict)      # parámetros de μ

    @property
    def n_steps(self) -> int:
        return max(2, int(round(self.T * self.steps_per_year)))


# ═════════════════════════════════════════════════════════════════════════════
#  Ruido: misma semilla ⇒ mismas trayectorias al mover parámetros
# ═════════════════════════════════════════════════════════════════════════════
def make_noise(seed: int, n_paths: int, n_steps: int) -> np.ndarray:
    """
    Normales estándar de forma (3, n_steps, n_paths): [B¹, B²⊥, W³].

    Cada trayectoria i y cada canal c tiene su propio generador, así que
    añadir trayectorias o alargar T *no cambia* las que ya existían.
    """
    Z = np.empty((3, n_steps, n_paths), dtype=np.float32)
    for i in range(n_paths):
        for c in range(3):
            Z[c, :, i] = np.random.default_rng([int(seed), i, c]).standard_normal(
                n_steps, dtype=np.float32)
    return Z


# ═════════════════════════════════════════════════════════════════════════════
#  Tendencia μ
# ═════════════════════════════════════════════════════════════════════════════
def mu_det(cfg: SimConfig, t: np.ndarray) -> np.ndarray:
    """μ(t) para las tendencias deterministas (constante o periódica)."""
    tp = cfg.tp
    if cfg.trend == "const":
        return np.full_like(t, tp["mu"], dtype=float)
    if cfg.trend == "periodic":
        return tp["mu"] + tp["amp"] * np.sin(2 * np.pi * t / tp["period"] + tp["phase"])
    raise ValueError("mu_det solo aplica a tendencias deterministas")


def mu_mean(cfg: SimConfig, t: np.ndarray) -> np.ndarray:
    """E[μ_t] para cualquiera de las tres tendencias."""
    if cfg.trend in ("const", "periodic"):
        return mu_det(cfg, t)
    tp = cfg.tp
    return tp["m"] + (tp["mu0"] - tp["m"]) * np.exp(-tp["a"] * t)


def mu_std(cfg: SimConfig, t: np.ndarray) -> np.ndarray | None:
    """Desviación estándar exacta de μ_t (solo OU); None si es determinista."""
    if cfg.trend != "stoch":
        return None
    a, s = cfg.tp["a"], cfg.tp["s"]
    return s * np.sqrt((1 - np.exp(-2 * a * t)) / (2 * a))


# ═════════════════════════════════════════════════════════════════════════════
#  Simulación
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SimResult:
    t: np.ndarray            # (n+1,)
    X: np.ndarray            # (n+1, N)
    V: np.ndarray            # (n+1, N)   (ya truncada a ≥0 en Heston y Scott)
    Mu: np.ndarray | None    # (n+1, N) solo si la tendencia es estocástica


def simulate(cfg: SimConfig, Z: np.ndarray) -> SimResult:
    """
    Simula X_t, V_t (y μ_t) con el ruido Z de make_noise.

    Esquemas
    --------
    V  : exacto para Hull-White (MBG) y Stein-Stein (OU); splitting
         multiplicativo-aditivo para Nelson (positivo); Euler con *full
         truncation* para Heston y Scott.
    μ_t: transición exacta del OU.
    X  : log-Euler (exacto si V es constante en el paso) en la versión usual;
         splitting "GBM con deriva −α + empuje hacia μ" en la redefinición.
    """
    n, N = Z.shape[1], Z.shape[2]
    dt = cfg.T / n
    sq = np.sqrt(dt)
    rho = cfg.rho
    rho_c = np.sqrt(max(0.0, 1.0 - rho * rho))
    p, m = cfg.vp, cfg.model
    t = np.linspace(0.0, cfg.T, n + 1)

    X = np.empty((n + 1, N))
    V = np.empty((n + 1, N))
    X[0], V[0] = cfg.x0, cfg.v0

    stoch = cfg.trend == "stoch"
    if stoch:
        Mu = np.empty((n + 1, N))
        Mu[0] = cfg.tp["mu0"]
        a, mm, s = cfg.tp["a"], cfg.tp["m"], cfg.tp["s"]
        ea = np.exp(-a * dt)
        sd_mu = s * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
        mu_grid = None
    else:
        Mu = None
        mu_grid = mu_det(cfg, t[:-1])

    alpha = cfg.alpha
    usual = cfg.drift == "usual"
    push = 1.0 - np.exp(-alpha * dt)

    # constantes por modelo
    if m in ("stein_stein", "nelson", "heston", "scott"):
        kap, th = p["kappa"], p["theta"]
    sig = p["sigma"]
    if m == "stein_stein":
        e_k = np.exp(-kap * dt)
        sd_v = sig * np.sqrt((1 - np.exp(-2 * kap * dt)) / (2 * kap))
    if m == "nelson":
        e_k = np.exp(-kap * dt)

    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(n):
            z1 = Z[0, k].astype(np.float64)
            z2 = rho * z1 + rho_c * Z[1, k]
            v = V[k]
            vp = np.maximum(v, 0.0)
            sv = np.sqrt(vp)
            mu_k = Mu[k] if stoch else mu_grid[k]

            # ── precio ──────────────────────────────────────────────────────
            if usual:
                X[k + 1] = X[k] * np.exp((mu_k - 0.5 * vp) * dt + sv * sq * z1)
            else:
                X[k + 1] = (X[k] * np.exp((-alpha - 0.5 * vp) * dt + sv * sq * z1)
                            + mu_k * push)

            # ── varianza ────────────────────────────────────────────────────
            if m == "hull_white":
                V[k + 1] = v * np.exp((p["theta"] - 0.5 * sig**2) * dt + sig * sq * z2)
            elif m == "stein_stein":
                V[k + 1] = th + (v - th) * e_k + sd_v * z2
            elif m == "nelson":
                V[k + 1] = v * np.exp((-kap - 0.5 * sig**2) * dt + sig * sq * z2) + th * (1 - e_k)
            elif m == "heston":
                V[k + 1] = v + kap * (th - vp) * dt + sig * sv * sq * z2
            elif m == "scott":
                V[k + 1] = v + kap * (th - (sv - p["eta"]) ** 2) * dt + sig * sv * sq * z2
            else:
                raise ValueError(f"modelo desconocido: {m}")

            # ── tendencia estocástica ───────────────────────────────────────
            if stoch:
                Mu[k + 1] = mm + (Mu[k] - mm) * ea + sd_mu * Z[2, k]

    if m in _TRUNCATED:
        V = np.maximum(V, 0.0)
    return SimResult(t=t, X=X, V=V, Mu=Mu)


# ═════════════════════════════════════════════════════════════════════════════
#  Teoría
# ═════════════════════════════════════════════════════════════════════════════
def theory_V(cfg: SimConfig, t: np.ndarray):
    """(media, varianza) exactas de V_t, o None si el modelo no tiene forma cerrada."""
    p, m, v0 = cfg.vp, cfg.model, cfg.v0
    if m == "hull_white":
        th, s = p["theta"], p["sigma"]
        mean = v0 * np.exp(th * t)
        var = v0**2 * np.exp(2 * th * t) * np.expm1(s**2 * t)
        return mean, var
    if m == "stein_stein":
        k, th, s = p["kappa"], p["theta"], p["sigma"]
        mean = th + (v0 - th) * np.exp(-k * t)
        var = s**2 / (2 * k) * (1 - np.exp(-2 * k * t))
        return mean, var
    if m == "heston":
        k, th, s = p["kappa"], p["theta"], p["sigma"]
        e1 = np.exp(-k * t)
        mean = th + (v0 - th) * e1
        var = (v0 * s**2 / k * (e1 - e1**2) + th * s**2 / (2 * k) * (1 - e1) ** 2)
        return mean, var
    if m == "nelson":
        k, th, s = p["kappa"], p["theta"], p["sigma"]
        # y = (1, E[V], E[V²]);  y' = A y
        A = np.array([[0.0, 0.0, 0.0],
                      [k * th, -k, 0.0],
                      [0.0, 2 * k * th, -(2 * k - s**2)]])
        dt = t[1] - t[0]
        step = expm(A * dt)
        y = np.array([1.0, v0, v0**2])
        m1, m2 = np.empty_like(t), np.empty_like(t)
        for i in range(len(t)):
            m1[i], m2[i] = y[1], y[2]
            y = step @ y
        return m1, np.maximum(m2 - m1**2, 0.0)
    return None  # scott


def theory_V_pdf(cfg: SimConfig, t: float, grid: np.ndarray):
    """Densidad exacta de V_t sobre `grid` (None si no existe en forma cerrada)."""
    p, m, v0 = cfg.vp, cfg.model, cfg.v0
    if t <= 0:
        return None
    if m == "hull_white":
        s = p["sigma"] * np.sqrt(t)
        if s < 1e-9:
            return None
        loc = np.log(v0) + (p["theta"] - 0.5 * p["sigma"] ** 2) * t
        return stats.lognorm.pdf(grid, s=s, scale=np.exp(loc))
    if m == "stein_stein":
        mean, var = theory_V(cfg, np.array([0.0, t]))
        sd = np.sqrt(var[1])
        if sd < 1e-12:
            return None
        return stats.norm.pdf(grid, loc=mean[1], scale=sd)
    if m == "heston":
        k, th, s = p["kappa"], p["theta"], p["sigma"]
        c = s**2 * (1 - np.exp(-k * t)) / (4 * k)
        d = 4 * k * th / s**2
        lam = v0 * np.exp(-k * t) / c
        return stats.ncx2.pdf(grid / c, d, lam) / c
    return None


def _forced_ode(alpha: float, x0: float, forcing: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Resuelve E' = α(f(t) − E), E(0)=x0, con f lineal a trozos (exacto en cada paso)."""
    E = np.empty_like(t)
    E[0] = x0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        eh = np.exp(-alpha * h)
        slope = (forcing[i + 1] - forcing[i]) / h
        E[i + 1] = eh * E[i] + forcing[i] * (1 - eh) + slope * (h - (1 - eh) / alpha)
    return E


def theory_X_mean(cfg: SimConfig, t: np.ndarray) -> np.ndarray:
    """
    E[X_t], suponiendo que la integral estocástica ∫√V X dB¹ es una martingala
    verdadera y que W³ (la tendencia estocástica) es independiente de B¹, B².
    """
    tp = cfg.tp
    if cfg.drift == "mr":
        return _forced_ode(cfg.alpha, cfg.x0, mu_mean(cfg, t), t)

    # deriva usual:  E[X_t] = X₀ · E[exp(∫μ)]
    if cfg.trend == "const":
        return cfg.x0 * np.exp(tp["mu"] * t)
    if cfg.trend == "periodic":
        P, A, ph = tp["period"], tp["amp"], tp["phase"]
        integral = tp["mu"] * t + A * P / (2 * np.pi) * (np.cos(ph) - np.cos(2 * np.pi * t / P + ph))
        return cfg.x0 * np.exp(integral)
    a, s, m0, mm = tp["a"], tp["s"], tp["mu0"], tp["m"]
    M = mm * t + (m0 - mm) * (1 - np.exp(-a * t)) / a
    var = s**2 / a**2 * (t - 2 * (1 - np.exp(-a * t)) / a + (1 - np.exp(-2 * a * t)) / (2 * a))
    return cfg.x0 * np.exp(M + 0.5 * var)
