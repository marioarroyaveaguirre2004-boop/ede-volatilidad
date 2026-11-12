"""
Laboratorio de volatilidad estocástica — Ecuaciones Diferenciales Estocásticas (EAFIT)

Hull & White · Stein & Stein · Nelson · Heston · Scott
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models import (
    MODELS, SimConfig, make_noise, mu_det, mu_mean, mu_std, simulate,
    theory_V, theory_V_pdf, theory_X_mean,
)
from ui import PLOT_CONFIG, inject_css, palette, ptitle, section, slider, stats, style

st.set_page_config(page_title="Laboratorio de volatilidad estocástica",
                   page_icon="📈", layout="wide")
P = palette()
inject_css(P)

TREND_NAME = {"const": r"Constante $\mu$", "periodic": r"Periódica $\mu(t)$", "stoch": r"Estocástica $\mu_t$"}
TREND_TAG = {
    "const": ("good", "observable · estimable ✓"),
    "periodic": ("info", "inobservable · sí estimable ✓"),
    "stoch": ("warn", "inobservable · no estimable (?)"),
}


@st.cache_resource(max_entries=2, show_spinner=False)
def get_noise(seed: int, n_paths: int, n_steps: int) -> np.ndarray:
    return make_noise(seed, n_paths, n_steps)


# ═════════════════════════════════════════════════════════════════════════════
#  Panel de control
# ═════════════════════════════════════════════════════════════════════════════
def trend_controls(drift: str, trend: str, x0: float) -> dict:
    """Sliders de μ. Con reversión a la media, μ es un *nivel* (en unidades de precio)."""
    mr = drift == "mr"
    tag = f"{drift}_{trend}" + (f"_{x0:g}" if mr else "")   # clave nueva si cambia X₀ (cambian los rangos)

    def sl(label, lo, hi, default, step, name, help=None):
        return slider(label, lo, hi, default, step, key=f"{tag}_{name}", help=help)

    per = ("1 año = ciclo estacional (viento, calor, agua…).",)
    if mr:
        lo, hi, st_ = 0.2 * x0, 3.0 * x0, x0 / 100
        lvl = "Nivel (en unidades de precio) hacia el que X_t es atraído."
        if trend == "const":
            return {"mu": sl(r"$\mu$ — nivel de reversión", lo, hi, 1.2 * x0, st_, "mu", lvl)}
        if trend == "periodic":
            return {
                "mu": sl(r"$\bar\mu$ — nivel medio", lo, hi, x0, st_, "mu", lvl),
                "amp": sl(r"$A$ — amplitud", 0.0, x0, 0.25 * x0, st_, "amp"),
                "period": sl(r"$P$ — periodo (años)", 0.1, 5.0, 1.0, 0.05, "period", *per),
                "phase": sl(r"$\varphi$ — fase", 0.0, 6.28, 0.0, 0.05, "phase"),
            }
        return {
            "mu0": sl(r"$\mu_0$ — nivel inicial", lo, hi, x0, st_, "mu0"),
            "m": sl(r"$m$ — nivel de largo plazo de $\mu_t$", lo, hi, 1.2 * x0, st_, "m"),
            "a": sl(r"$a$ — velocidad de reversión de $\mu_t$", 0.05, 10.0, 1.5, 0.05, "a"),
            "s": sl(r"$s$ — volatilidad de $\mu_t$", 0.0, x0, 0.3 * x0, x0 / 100, "s"),
        }
    if trend == "const":
        return {"mu": sl(r"$\mu$ — tasa de crecimiento", -0.5, 0.5, 0.05, 0.01, "mu")}
    if trend == "periodic":
        return {
            "mu": sl(r"$\bar\mu$ — tasa media", -0.5, 0.5, 0.05, 0.01, "mu"),
            "amp": sl(r"$A$ — amplitud", 0.0, 1.0, 0.30, 0.01, "amp"),
            "period": sl(r"$P$ — periodo (años)", 0.1, 5.0, 1.0, 0.05, "period", *per),
            "phase": sl(r"$\varphi$ — fase", 0.0, 6.28, 0.0, 0.05, "phase"),
        }
    return {
        "mu0": sl(r"$\mu_0$ — tasa inicial", -0.5, 0.5, 0.05, 0.01, "mu0"),
        "m": sl(r"$m$ — tasa de largo plazo de $\mu_t$", -0.5, 0.5, 0.10, 0.01, "m"),
        "a": sl(r"$a$ — velocidad de reversión de $\mu_t$", 0.05, 10.0, 1.5, 0.05, "a"),
        "s": sl(r"$s$ — volatilidad de $\mu_t$", 0.0, 1.0, 0.20, 0.01, "s"),
    }


if "seed" not in st.session_state:
    st.session_state["seed"] = 2026

with st.sidebar:
    section("Modelo de volatilidad")
    model_key = st.pills("Modelo", list(MODELS), key="model", default="hull_white",
                         format_func=lambda k: MODELS[k].name, label_visibility="collapsed",
                         width="stretch") or "hull_white"
    spec = MODELS[model_key]

    section("Tendencia de Xₜ")
    drift = st.radio("Término de tendencia", ["usual", "mr"], key="drift", label_visibility="collapsed",
                     format_func=lambda k: (r"Usual  ·  $\mu_t X_t\,dt$" if k == "usual"
                                            else r"Reversión a la media  ·  $\alpha(\mu_t - X_t)\,dt$"))
    alpha = 3.0
    if drift == "mr":
        alpha = slider(r"$\alpha$ — velocidad de reversión de $X$", 0.1, 15.0, 3.0, 0.1, key="alpha",
                       help="Con α grande, X_t pega rápido a μ_t.")
    x0 = slider(r"$X_0$ — precio inicial", 10.0, 1000.0, 100.0, 10.0, key="x0")
    trend = st.radio("Tipo de μ", ["const", "periodic", "stoch"], key="trend", horizontal=True,
                     label_visibility="collapsed", format_func=lambda k: TREND_NAME[k])
    tp = trend_controls(drift, trend, x0)

    section(f"Varianza — {spec.name}")
    v0 = slider(r"$V_0$ — varianza inicial", 0.001, 0.50, 0.04, 0.001, key="v0",
                help="0.04 ⇒ volatilidad inicial √V₀ = 20 %.")
    vp = {p.key: slider(p.label, p.lo, p.hi, p.default, p.step, key=f"{model_key}_{p.key}", help=p.help or None)
          for p in spec.params}
    rho = slider(r"$\rho$ — correlación $B^{(1)}$, $B^{(2)}$", -0.99, 0.99,
                 0.0 if model_key == "hull_white" else -0.5, 0.01, key=f"rho_{model_key}",
                 help="ρ < 0: efecto apalancamiento (el precio cae cuando sube la volatilidad).")

    section("Simulación")
    with st.expander("Horizonte, pasos, trayectorias y semilla", expanded=False):
        T = slider("Horizonte $T$ (años)", 0.25, 5.0, 1.0, 0.25, key="T")
        spy = st.select_slider("Pasos por año", [52, 126, 252, 504], 252, key="spy")
        n_paths = slider("Trayectorias simuladas", 100, 2000, 500, 100, key="n_paths")
        n_show = slider("Trayectorias dibujadas", 1, 100, 40, 1, key="n_show")
        c1, c2 = st.columns([1.4, 1], vertical_alignment="bottom")
        c1.number_input("Semilla", 0, 10**6, step=1, key="seed")
        c2.button("Nueva", width="stretch",
                  on_click=lambda: st.session_state.update(seed=int(np.random.randint(0, 10**6))))
        frozen = st.checkbox("Congelar el ruido", True, key="frozen",
                             help="Al mover un parámetro, las trayectorias se deforman en lugar de "
                                  "cambiar por completo. Así se ve el efecto del parámetro y no el azar.")

    section("Vista")
    show_band = st.toggle("Banda media ± 2σ", True, key="band")
    show_theory = st.toggle("Curvas teóricas", True, key="theory")
    lock = st.toggle("Fijar ejes Y", False, key="lock",
                     help="Congela la escala vertical al activarlo, para ver cómo se mueve la nube "
                          "al cambiar un parámetro.")
    c1, c2 = st.columns(2)
    pin = c1.button("📌 Fijar", width="stretch",
                    help="Guarda la media y la banda actuales como línea gris punteada.")
    unpin = c2.button("Quitar", width="stretch")

seed = int(st.session_state["seed"]) if frozen else int(np.random.randint(0, 10**6))
n_paths, n_show = int(n_paths), int(n_show)

# ═════════════════════════════════════════════════════════════════════════════
#  Simulación
# ═════════════════════════════════════════════════════════════════════════════
cfg = SimConfig(model=model_key, vp=vp, v0=v0, x0=x0, rho=rho, T=T, steps_per_year=spy,
                drift=drift, alpha=alpha, trend=trend, tp=tp)
n_steps = cfg.n_steps
Z = get_noise(seed, n_paths, n_steps)
res = simulate(cfg, Z)
t, X, V, Mu = res.t, res.X, res.V, res.Mu
dt = T / n_steps

x_mean, x_sd = X.mean(axis=1), X.std(axis=1)
v_mean, v_sd = V.mean(axis=1), V.std(axis=1)
th_x = theory_X_mean(cfg, t)
th_v = theory_V(cfg, t)

if not lock:
    st.session_state.pop("ylock", None)
ylock = st.session_state.setdefault("ylock", {}) if lock else None

if unpin:
    st.session_state.pop("ref", None)
if pin:
    st.session_state["ref"] = dict(
        label=f"{spec.name} · ρ = {rho:g} · " + " · ".join(f"{k} = {v:g}" for k, v in vp.items()),
        t=t.copy(), x_mean=x_mean.copy(), x_sd=x_sd.copy(), v_mean=v_mean.copy(), v_sd=v_sd.copy(), T=T)
ref = st.session_state.get("ref")
if ref is not None and abs(ref["T"] - T) > 1e-9:
    ref = None   # otro horizonte: la referencia ya no es comparable

# ═════════════════════════════════════════════════════════════════════════════
#  Gráficas
# ═════════════════════════════════════════════════════════════════════════════
def decimate(n: int, max_pts: int = 400) -> np.ndarray:
    return np.unique(np.linspace(0, n, min(n + 1, max_pts)).astype(int))


IDX = decimate(n_steps)


def line_chart(name: str, Y: np.ndarray, mean: np.ndarray, sd: np.ndarray,
               th_mean=None, th_sd=None, ref_mean=None, ref_sd=None,
               ytitle: str = "", fmt: str = ".3f", height: int = 420, paths: bool = True,
               floor0: bool = False):
    tt = t[IDX]
    fig = go.Figure()
    hv = f"%{{y:{fmt}}}"

    if paths and n_show > 1:
        Ys = Y[IDX][:, 1:min(n_show, Y.shape[1])]
        if Ys.shape[1] > 0:
            xs = np.tile(np.append(tt, np.nan), Ys.shape[1])
            ys = np.vstack([Ys, np.full((1, Ys.shape[1]), np.nan)]).T.ravel()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Trayectorias",
                                     line=dict(color=P["path"], width=1), hoverinfo="skip"))
    if paths:
        fig.add_trace(go.Scatter(x=tt, y=Y[IDX, 0], mode="lines", name="Trayectoria destacada",
                                 line=dict(color=P["violet"], width=1.6), hoverinfo="skip"))

    lo_b, hi_b = mean - 2 * sd, mean + 2 * sd
    if floor0:
        lo_b = np.maximum(lo_b, 0.0)
    if show_band:
        fig.add_trace(go.Scatter(x=tt, y=lo_b[IDX], mode="lines", line=dict(color=P["primary"], width=.7, dash="dot"),
                                 showlegend=False, hovertemplate="Media − 2σ: " + hv + "<extra></extra>"))
        fig.add_trace(go.Scatter(x=tt, y=hi_b[IDX], mode="lines", name="Media ± 2σ (sim.)",
                                 line=dict(color=P["primary"], width=.7, dash="dot"),
                                 fill="tonexty", fillcolor=P["band"],
                                 hovertemplate="Media + 2σ: " + hv + "<extra></extra>"))

    if ref_mean is not None:
        fig.add_trace(go.Scatter(x=tt, y=ref_mean[IDX], mode="lines", name="Referencia fijada",
                                 line=dict(color=P["ref"], width=2, dash="dash"), hovertemplate=hv))
        if show_band and ref_sd is not None:
            for sgn in (-1, 1):
                fig.add_trace(go.Scatter(x=tt, y=(ref_mean + sgn * 2 * ref_sd)[IDX], mode="lines",
                                         line=dict(color=P["ref"], width=.7, dash="dot"),
                                         showlegend=False, hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=tt, y=mean[IDX], mode="lines", name="Media simulada",
                             line=dict(color=P["ink"], width=2.4), hovertemplate=hv))
    if show_theory and th_mean is not None:
        fig.add_trace(go.Scatter(x=tt, y=th_mean[IDX], mode="lines", name="Media teórica",
                                 line=dict(color=P["orange"], width=2.4, dash="dash"), hovertemplate=hv))
        if show_band and th_sd is not None:
            for sgn in (-1, 1):
                fig.add_trace(go.Scatter(x=tt, y=(th_mean + sgn * 2 * th_sd)[IDX], mode="lines",
                                         name="Teórica ± 2σ", legendgroup="thband", showlegend=(sgn == 1),
                                         line=dict(color=P["orange"], width=1.1, dash="dot"), hovertemplate=hv))

    shown = [lo_b[IDX].min() if show_band else mean.min(), hi_b[IDX].max() if show_band else mean.max()]
    if paths:
        shown += [Y[IDX][:, :n_show].min(), Y[IDX][:, :n_show].max()]
    lo, hi = float(np.nanmin(shown)), float(np.nanmax(shown))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    rng = [lo - pad, hi + pad]
    if ylock is not None:
        ylock.setdefault(name, rng)
        rng = ylock[name]

    style(fig, P, height, "Tiempo t (años)", ytitle)
    fig.update_xaxes(range=[0, T])
    fig.update_yaxes(range=rng)
    st.plotly_chart(fig, key=f"chart_{name}", config={**PLOT_CONFIG, "toImageButtonOptions": {"scale": 2, "filename": name}})


# ═════════════════════════════════════════════════════════════════════════════
#  Encabezado
# ═════════════════════════════════════════════════════════════════════════════
hl, hr = st.columns([3, 2], vertical_alignment="center")
hl.markdown('<p class="hdr-title">Laboratorio de volatilidad estocástica</p>'
            '<p class="hdr-sub">Ecuaciones Diferenciales Estocásticas · Ingeniería Matemática, EAFIT</p>',
            unsafe_allow_html=True)
drift_txt = "deriva usual" if drift == "usual" else "reversión a la media"
trend_txt = {"const": "μ constante", "periodic": "μ(t) periódica", "stoch": "μₜ estocástica"}[trend]
hr.markdown(f'<div class="chips"><span class="chip on">{spec.name}</span>'
            f'<span class="chip">{drift_txt}</span><span class="chip">{trend_txt}</span>'
            f'<span class="chip">ρ = {rho:g}</span></div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  Ecuaciones
# ═════════════════════════════════════════════════════════════════════════════
x_eq = (r"dX_t &= \mu_t\,X_t\,dt + \sqrt{V_t}\,X_t\,dB^{(1)}_t" if drift == "usual"
        else r"dX_t &= \alpha\,(\mu_t - X_t)\,dt + \sqrt{V_t}\,X_t\,dB^{(1)}_t")
v_eq = spec.v_latex.replace("dV_t =", "dV_t &=", 1)
mu_eq = {
    "const": r"\mu_t &\equiv \mu",
    "periodic": r"\mu_t &= \mu(t) = \bar\mu + A\,\sin\!\Big(\tfrac{2\pi t}{P}+\varphi\Big)",
    "stoch": r"d\mu_t &= a\,(m-\mu_t)\,dt + s\,dW^{(3)}_t",
}[trend]
corr_eq = r"d\langle B^{(1)},B^{(2)}\rangle_t &= \rho\,dt"

with st.container(key="eq_card"):
    left, right = st.columns([3, 2], vertical_alignment="center")
    left.latex(r"\begin{aligned}" + r" \\[2pt] ".join([x_eq, v_eq, mu_eq, corr_eq]) + r"\end{aligned}")
    cls, txt = TREND_TAG[trend]
    mu_sym = {"const": "μ", "periodic": "μ(t)", "stoch": "μₜ"}[trend]
    right.markdown(
        f'<div class="eq-side"><b>{spec.name}.</b> {spec.summary}<br><br>'
        f'<b>Tendencia {mu_sym}</b> &nbsp;<span class="tag {cls}">{txt}</span><br><br>'
        f'<span class="note">Vₜ es inobservable ⇒ sus parámetros se obtienen por calibración.</span></div>',
        unsafe_allow_html=True)

if model_key in ("hull_white", "nelson") and rho > 0:
    st.warning("Con ρ > 0 en este modelo el proceso puede ser solo una martingala *local*: "
               "la media teórica de $X_t$ puede quedar por encima de la simulada.", icon="⚠️")
if model_key == "stein_stein":
    st.caption("En Stein & Stein, $V_t$ es un OU y puede ser negativa; en $X_t$ se usa $\\sqrt{\\max(V_t,0)}$.")
if model_key == "scott":
    st.caption("En Scott la difusión de $V_t$ se tomó como $\\sigma\\sqrt{V_t}$ — ver la pestaña Guía.")

# ═════════════════════════════════════════════════════════════════════════════
#  Estadísticas
# ═════════════════════════════════════════════════════════════════════════════
xT, vT = X[-1], V[-1]
stats([
    ("E[X_T] simulado", f"{xT.mean():,.2f}", f"{xT.mean() - th_x[-1]:+.2f} frente a la teoría"),
    ("Desv. est. de X_T", f"{xT.std():,.2f}", f"{xT.std() / x0:.1%} de X₀"),
    ("E[V_T]", f"{vT.mean():.4f}", f"volatilidad ≈ {np.sqrt(max(vT.mean(), 0)):.1%}"),
    ("P( X_T < X₀ )", f"{(xT < x0).mean():.1%}", f"{n_paths:,} trayectorias"),
])

# ═════════════════════════════════════════════════════════════════════════════
#  Pestañas
# ═════════════════════════════════════════════════════════════════════════════
tab_paths, tab_dist, tab_guide = st.tabs(["Trayectorias", "Distribución y teoría", "Guía"])

with tab_paths:
    ptitle("Precio Xₜ")
    line_chart("precio", X, x_mean, x_sd, th_mean=th_x,
               ref_mean=None if ref is None else ref["x_mean"], ref_sd=None if ref is None else ref["x_sd"],
               ytitle="X_t", fmt=",.2f", height=420)
    if ref is not None:
        st.caption(f"Referencia fijada: {ref['label']}")

    has_mu_panel = trend != "const"
    cols = st.columns(2, gap="large") if has_mu_panel else [st.container()]
    with cols[0]:
        ptitle("Varianza Vₜ")
        line_chart("varianza", V, v_mean, v_sd,
                   th_mean=None if th_v is None else th_v[0], th_sd=None if th_v is None else np.sqrt(th_v[1]),
                   ref_mean=None if ref is None else ref["v_mean"], ref_sd=None if ref is None else ref["v_sd"],
                   ytitle="V_t", fmt=".4f", height=340, floor0=model_key != "stein_stein")
        if th_v is None:
            st.caption("Scott no tiene momentos de $V_t$ en forma cerrada: solo se muestra la simulación.")
    if has_mu_panel:
        with cols[1]:
            if trend == "periodic":
                ptitle("Tendencia μ(t)")
                mu_curve = mu_det(cfg, t)
                fig = go.Figure(go.Scatter(x=t, y=mu_curve, mode="lines", name="μ(t)",
                                           line=dict(color=P["orange"], width=2.4), hovertemplate="%{y:.3f}"))
                style(fig, P, 340, "Tiempo t (años)", "μ(t)", legend=False)
                fig.update_xaxes(range=[0, T])
                if ylock is not None:
                    pad = 0.08 * (mu_curve.max() - mu_curve.min() + 1e-9)
                    ylock.setdefault("mu", [mu_curve.min() - pad, mu_curve.max() + pad])
                    fig.update_yaxes(range=ylock["mu"])
                st.plotly_chart(fig, key="chart_mu_det", config=PLOT_CONFIG)
                st.caption("Determinista y periódica: la «tendencia funcional periódica» de los mercados de energía.")
            else:
                ptitle("Tendencia μₜ (estocástica)")
                line_chart("tendencia", Mu, Mu.mean(axis=1), Mu.std(axis=1),
                           th_mean=mu_mean(cfg, t), th_sd=mu_std(cfg, t), ytitle="μ_t", fmt=".3f", height=340)

with tab_dist:
    t_sel = slider("Instante $t$ (años)", float(dt), float(T), float(T), float(dt), key=f"tsel_{T}_{spy}", fmt="%.3f")
    k = int(round(t_sel / dt))
    xs, vs = X[k], V[k]
    N = xs.size

    def hist_chart(data, name, mean_sim, mean_th, pdf_fn=None, nb=60):
        lo, hi = float(data.min()), float(np.percentile(data, 99.5))
        if hi <= lo:
            hi = lo + 1e-9
        edges = np.linspace(lo, hi, nb + 1)
        counts, _ = np.histogram(data, bins=edges)
        w = edges[1] - edges[0]
        dens = counts / (N * w)
        centers = 0.5 * (edges[1:] + edges[:-1])
        ymax = float(dens.max()) * 1.12
        fig = go.Figure()
        fig.add_trace(go.Bar(x=centers, y=dens, width=w * 0.94, name="Simulación",
                             marker=dict(color=P["primary"], opacity=0.45, line=dict(width=0)),
                             hovertemplate="%{x:.4f}: %{y:.3f}<extra></extra>"))
        if show_theory and pdf_fn is not None:
            grid = np.linspace(max(lo, 1e-12) if name == "V" and lo <= 0 else lo, hi, 400)
            pdf = pdf_fn(grid)
            if pdf is not None and np.isfinite(pdf).any():
                ymax = max(ymax, float(np.nanmax(pdf[np.isfinite(pdf)])) * 1.05)
                fig.add_trace(go.Scatter(x=grid, y=pdf, mode="lines", name="Densidad teórica",
                                         line=dict(color=P["orange"], width=2.4)))
        fig.add_trace(go.Scatter(x=[mean_sim] * 2, y=[0, ymax], mode="lines", name="Media simulada",
                                 line=dict(color=P["ink"], width=1.8)))
        if show_theory and mean_th is not None:
            fig.add_trace(go.Scatter(x=[mean_th] * 2, y=[0, ymax], mode="lines", name="Media teórica",
                                     line=dict(color=P["orange"], width=1.8, dash="dash")))
        style(fig, P, 360, f"{name}_t en t = {t_sel:.2f}", "Densidad")
        fig.update_layout(bargap=0, hovermode="closest")
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(range=[0, ymax])
        st.plotly_chart(fig, key=f"hist_{name}", config=PLOT_CONFIG)
        out = int((data > hi).sum())
        if out:
            st.caption(f"Se muestra hasta el percentil 99.5 ({out} de {N} valores quedaron a la derecha).")

    hx, hv_ = st.columns(2, gap="large")
    with hx:
        ptitle("Distribución de Xₜ")
        hist_chart(xs, "X", float(xs.mean()), float(th_x[k]))
    with hv_:
        ptitle("Distribución de Vₜ")
        pdf_fn = (lambda g: theory_V_pdf(cfg, t_sel, g)) if th_v is not None else None
        hist_chart(vs, "V", float(vs.mean()), None if th_v is None else float(th_v[0][k]), pdf_fn)

    ptitle("Teoría frente a simulación")
    st.write("")

    def se_std(a):
        sd = a.std()
        if sd == 0:
            return 0.0
        kurt = ((a - a.mean()) ** 4).mean() / sd**4
        return float(sd * np.sqrt(max(kurt - 1.0, 1e-9) / (4 * a.size)))

    def fmt(x, nd=5):
        return "—" if x is None else f"{x:,.{nd}f}"

    def row(label, theo, sim, se):
        z = None if theo is None or se == 0 else (sim - theo) / se
        return {"Magnitud": label, "Teoría": fmt(theo), "Simulación": fmt(sim),
                "Error estándar": fmt(se), "z": "—" if z is None else f"{z:+.2f}"}

    rows = [
        row("E[X_t]", float(th_x[k]), float(xs.mean()), float(xs.std() / np.sqrt(N))),
        row("SD[X_t]", None, float(xs.std()), se_std(xs)),
        row("E[V_t]", None if th_v is None else float(th_v[0][k]), float(vs.mean()), float(vs.std() / np.sqrt(N))),
        row("SD[V_t]", None if th_v is None else float(np.sqrt(th_v[1][k])), float(vs.std()), se_std(vs)),
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("|z| < 3: simulación y teoría compatibles dentro del ruido de Monte Carlo. «—»: sin forma cerrada. "
               "Con pocas trayectorias y colas pesadas (Hull & White con σ grande) la desviación muestral es ruidosa; "
               "sube el número de trayectorias para afinarla.")

with tab_guide:
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("### Cómo usarla en clase")
        st.markdown(
            """
1. **Elige un modelo** con los chips de arriba y mira sus ecuaciones.
2. **Mueve σ, θ, κ, ρ…** — con el ruido congelado, las mismas trayectorias se *estiran o contraen*: se ve el efecto del parámetro y no el azar.
3. **📌 Fija una referencia** (media y banda en gris) y cambia un parámetro para comparar antes/después.
4. **Fijar ejes Y** mantiene la escala vertical quieta mientras mueves un parámetro.
5. Cambia a **reversión a la media** $\\alpha(\\mu_t - X_t)\\,dt$ y prueba las tres tendencias: constante, periódica (estacional) y estocástica.
6. En **Distribución y teoría** desliza el tiempo $t$ para ver cómo se ensancha la distribución y si coincide con la fórmula exacta.

> Streamlit actualiza al **soltar** el slider, no mientras lo arrastras.
"""
        )
        st.markdown("### Clasificación de la tendencia (pizarra)")
        st.markdown(
            """
| Tendencia | Naturaleza | Modelo en la app |
|---|---|---|
| $\\mu$ | observable · estimable ✓ | constante |
| $\\mu(t)$ | inobservable · **sí** estimable ✓ | periódica determinista |
| $\\mu_t$ | inobservable · **no** estimable (?) | OU estocástico |

Mercados de energía: variables climáticas (viento, calor, agua) y energéticas (petróleo, gas, carbón, biocombustibles) ⇒ **tendencia funcional periódica**. De ahí la *redefinición del término de tendencia*: $dX_t = \\alpha(\\mu_t - X_t)\\,dt + \\sqrt{V_t}\\,X_t\\,dB_t$.
"""
        )
    with g2:
        st.markdown("### Los cinco modelos")
        for m in MODELS.values():
            with st.expander(m.name):
                st.latex(m.v_latex)
                st.markdown(f"- **Idea:** {m.summary}\n- **Positividad:** {m.positivity}\n"
                            f"- **Teoría exacta en la app:** {m.closed_form}")

    st.markdown("### Métodos numéricos")
    st.markdown(
        """
| Pieza | Esquema |
|---|---|
| $V_t$ Hull & White | exacto (movimiento browniano geométrico) |
| $V_t$ Stein & Stein, $\\mu_t$ | exacto (transición de un Ornstein-Uhlenbeck) |
| $V_t$ Nelson | *splitting* multiplicativo + aditivo (preserva la positividad) |
| $V_t$ Heston, Scott | Euler con **full truncation** ($V^+=\\max(V,0)$ en deriva y difusión) |
| $X_t$ usual | log-Euler: $X_{k+1}=X_k\\exp\\{(\\mu-\\tfrac12 V)\\Delta t+\\sqrt V\\,\\Delta B^{(1)}\\}$ |
| $X_t$ con reversión | *splitting*: GBM con deriva $-\\alpha$ + empuje exacto hacia $\\mu$ |
| Correlación | $\\Delta B^{(2)} = \\rho\\,\\Delta B^{(1)} + \\sqrt{1-\\rho^2}\\,\\Delta B^\\perp$ |
| Ruido | un generador por trayectoria ⇒ añadir trayectorias o alargar $T$ no cambia las existentes |

**Teoría.** $E[X_t]$ asume que $\\int\\!\\sqrt V X\\,dB^{(1)}$ es martingala verdadera y que $W^{(3)}$ es independiente de $B^{(1)},B^{(2)}$. Con deriva usual y $\\mu_t$ OU: $E[X_t]=X_0\\,e^{M(t)+\\frac12\\mathrm{Var}(\\int_0^t\\mu)}$. Con reversión: $E[X_t]'=\\alpha(E[\\mu_t]-E[X_t])$. Para Heston, $V_t$ es un $\\chi^2$ no central escalado.

**Verificación.** `python validate.py` compara por Monte Carlo (30 000 trayectorias) contra estas fórmulas.
"""
    )
    st.info(
        "**Nota sobre Scott.** En la foto del tablero, la difusión de $V_t$ aparece como "
        "$\\sqrt{V_t}\\,X_t\\,dB^{(2)}_t$. Aquí se usó $\\sigma\\sqrt{V_t}\\,dB^{(2)}_t$ (como en Heston): "
        "con un factor $X_t$ la varianza dependería del nivel del precio. Conviene confirmarlo con el profesor.",
        icon="📝")
