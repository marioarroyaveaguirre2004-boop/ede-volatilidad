"""
ui.py — Tema visual del laboratorio: paleta, CSS, helpers de interfaz y estilo de gráficas.

Todo lo que es "cómo se ve" vive aquí; app.py se ocupa de qué se muestra.
"""
from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"


# ═════════════════════════════════════════════════════════════════════════════
#  Paleta — una sola fuente de verdad para CSS y Plotly
# ═════════════════════════════════════════════════════════════════════════════
def palette() -> dict:
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    if dark:
        return dict(
            dark=True,
            bg="#141416", surface="#1c1c1f", surface2="#232327",
            ink="#ededea", ink2="#b9b9b3", muted="#8c8c86", hair="#2c2c30", hair2="#3a3a40",
            primary="#818cf8", primary_soft="rgba(129,140,248,.14)",
            orange="#f0803f", violet="#a78bfa",
            path="rgba(129,140,248,.26)", band="rgba(129,140,248,.14)", ref="#8c8c86",
            tint="linear-gradient(135deg, rgba(129,140,248,.10), rgba(129,140,248,.03))",
            good="#34d399", warn="#fbbf24", bad="#f87171",
        )
    return dict(
        dark=False,
        bg="#fbfbf9", surface="#ffffff", surface2="#f1f1ec",
        ink="#1b1f24", ink2="#4b5059", muted="#7c8089", hair="#e6e6df", hair2="#d6d6ce",
        primary="#4f46e5", primary_soft="rgba(79,70,229,.10)",
        orange="#e8632b", violet="#7c3aed",
        path="rgba(79,70,229,.20)", band="rgba(79,70,229,.10)", ref="#8a8a84",
        tint="linear-gradient(135deg, rgba(79,70,229,.08), rgba(79,70,229,.02))",
        good="#15803d", warn="#b45309", bad="#b91c1c",
    )


# ═════════════════════════════════════════════════════════════════════════════
#  CSS
# ═════════════════════════════════════════════════════════════════════════════
def inject_css(P: dict) -> None:
    st.markdown(
        f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");
:root {{
  --ink: {P['ink']}; --ink2: {P['ink2']}; --muted: {P['muted']};
  --hair: {P['hair']}; --hair2: {P['hair2']}; --primary: {P['primary']};
  --primary-soft: {P['primary_soft']}; --surface: {P['surface']}; --surface2: {P['surface2']};
}}
/* ── tipografía ─────────────────────────────────────────────────────────── */
html, body, .stApp, [data-testid="stSidebar"], [data-testid="stMarkdownContainer"],
.stMarkdown, .stCaption, button, input, label, p, li, td, th, div[data-baseweb] {{
  font-family: {FONT} !important;
}}
.stApp {{ font-feature-settings: "cv11", "ss01"; }}
h1, h2, h3, h4 {{ letter-spacing: -0.015em; font-weight: 650 !important; }}
code, pre, .stCode {{ font-family: ui-monospace, "JetBrains Mono", Menlo, monospace !important; }}
/* ── chrome de Streamlit ────────────────────────────────────────────────── */
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--hair); }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 1.2rem; }}
hr {{ margin: .6rem 0 .9rem; border-color: var(--hair) !important; }}
/* ── etiquetas de sección en la barra lateral ──────────────────────────── */
.sec {{
  font-size: .68rem; font-weight: 650; letter-spacing: .11em; text-transform: uppercase;
  color: var(--muted); margin: 1.1rem 0 .35rem;
}}
.sec:first-child {{ margin-top: 0; }}
/* ── sliders: fila etiqueta + valor, pista limpia, sin burbuja ─────────── */
[class*="st-key-sl_"] [data-testid="stVerticalBlock"] {{ gap: 0 !important; }}
[class*="st-key-sl_"] [data-testid="stHorizontalBlock"] {{ gap: .5rem; align-items: baseline; flex-wrap: nowrap; }}
[class*="st-key-sl_"] [data-testid="stHorizontalBlock"] > *:first-child {{ flex: 1 1 0 !important; min-width: 0; width: auto !important; }}
[class*="st-key-sl_"] [data-testid="stHorizontalBlock"] > *:last-child {{ flex: 0 0 auto !important; }}
[class*="st-key-sl_"] [data-testid="stMarkdownContainer"] p {{ margin: 0; font-size: .86rem; color: var(--ink2); }}
[class*="st-key-sl_"] .val {{
  font-variant-numeric: tabular-nums; font-weight: 600; font-size: .86rem; color: var(--ink);
  text-align: right; white-space: nowrap;
}}
[class*="st-key-sl_"] [data-testid="stSlider"] {{ margin-top: -.35rem; }}
[data-testid="stSliderThumbValue"] {{ display: none !important; }}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{
  font-size: .66rem; color: var(--muted); font-variant-numeric: tabular-nums;
}}
[data-testid="stSlider"] [role="slider"] {{
  width: 14px !important; height: 14px !important; box-shadow: 0 1px 3px rgba(0,0,0,.18);
  border: 2px solid var(--surface) !important;
}}
[data-testid="stSlider"] [data-testid="stTickBar"] {{ margin-top: .1rem; }}
/* ── chips (pills) ─────────────────────────────────────────────────────── */
[data-testid="stPills"] button {{ font-size: .82rem; padding: .28rem .7rem; border-radius: 999px; }}
/* ── radios más compactos ───────────────────────────────────────────────── */
[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {{ font-size: .86rem; color: var(--ink2); }}
[data-testid="stRadio"] label {{ font-size: .86rem; }}
/* ── toggles y botones ─────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label p, [data-testid="stToggle"] label p {{ font-size: .86rem; }}
[data-testid="stBaseButton-secondary"] {{ font-size: .82rem; border-color: var(--hair2); }}
/* ── encabezado ─────────────────────────────────────────────────────────── */
.hdr-title {{ font-size: 1.9rem !important; font-weight: 700; letter-spacing: -.02em; line-height: 1.15; margin: 0; }}
.hdr-sub {{ color: var(--muted) !important; font-size: .9rem !important; margin: .25rem 0 0; }}
.chips {{ display: flex; gap: .4rem; flex-wrap: wrap; justify-content: flex-end; padding-top: .35rem; }}
.chip {{
  font-size: .74rem; font-weight: 600; padding: .28rem .65rem; border-radius: 999px;
  border: 1px solid var(--hair2); color: var(--ink2); background: var(--surface); white-space: nowrap;
}}
.chip.on {{ background: var(--primary); border-color: var(--primary); color: #fff; }}
/* ── tarjeta de ecuaciones ─────────────────────────────────────────────── */
.st-key-eq_card {{
  background: {P['tint']}; border: 1px solid var(--hair); border-radius: 14px;
  padding: 1.1rem 1.4rem .6rem; margin: 1rem 0 .4rem;
}}
.st-key-eq_card .katex-display {{ margin: .2rem 0 .6rem; }}
.st-key-eq_card .katex {{ font-size: 1.12em; }}
.eq-side {{ font-size: .88rem; color: var(--ink2); line-height: 1.5; }}
.eq-side b {{ color: var(--ink); }}
.tag {{ display: inline-block; font-size: .72rem; font-weight: 600; padding: .14rem .5rem; border-radius: 6px;
        background: var(--surface2); color: var(--ink2); border: 1px solid var(--hair); }}
.tag.good {{ color: {P['good']}; }} .tag.warn {{ color: {P['warn']}; }} .tag.info {{ color: var(--primary); }}
/* ── estadísticas sin caja ─────────────────────────────────────────────── */
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); margin: .4rem 0 .2rem; }}
.stat {{ padding: .55rem 1.1rem .55rem 0; border-right: 1px solid var(--hair); }}
.stat + .stat {{ padding-left: 1.1rem; }}
.stat:last-child {{ border-right: 0; }}
.stat .k {{ font-size: .72rem; color: var(--muted); font-weight: 500; letter-spacing: .02em; }}
.stat .v {{ font-size: 1.55rem; font-weight: 650; letter-spacing: -.02em; line-height: 1.2; margin-top: .1rem;
            font-variant-numeric: tabular-nums; color: var(--ink); }}
.stat .d {{ font-size: .74rem; color: var(--muted); margin-top: .1rem; font-variant-numeric: tabular-nums; }}
@media (max-width: 900px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} .stat:nth-child(2) {{ border-right: 0; }} }}
/* ── pestañas, títulos de gráfica, notas ───────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {{ font-size: .9rem; font-weight: 550; padding: .5rem .2rem; }}
.ptitle {{ font-size: 1.02rem; font-weight: 650; margin: .9rem 0 -.2rem; letter-spacing: -.01em; }}
.note {{ font-size: .8rem; color: var(--muted); }}
[data-testid="stAlert"] {{ border-radius: 10px; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--hair); border-radius: 10px; overflow: hidden; }}
[data-testid="stExpander"] details {{ border-color: var(--hair) !important; border-radius: 10px; }}
[data-testid="stExpander"] summary p {{ font-size: .86rem; font-weight: 550; }}
</style>
""",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers de interfaz
# ═════════════════════════════════════════════════════════════════════════════
def section(text: str) -> None:
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def _decimals(step: float) -> int:
    if step >= 1:
        return 0
    return max(0, -int(math.floor(math.log10(step) + 1e-9)))


def slider(label: str, lo: float, hi: float, default: float, step: float, key: str,
           help: str | None = None, fmt: str | None = None) -> float:
    """Slider con la etiqueta a la izquierda y el valor actual alineado a la derecha."""
    lo, hi, default, step = float(lo), float(hi), float(default), float(step)
    cur = float(st.session_state.get(key, default))
    cur = min(max(cur, lo), hi)
    nd = _decimals(step) if fmt is None else None
    txt = (fmt % cur) if fmt else (f"{cur:,.{nd}f}")
    with st.container(key=f"sl_{key}"):
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="bottom", gap=None):
            st.markdown(label, help=help, width="content")
            st.markdown(f'<span class="val">{txt}</span>', unsafe_allow_html=True, width="content")
        return st.slider(label, lo, hi, default, step, key=key, label_visibility="collapsed")


def stats(items: list[tuple[str, str, str | None]]) -> None:
    """Fila de estadísticas sin caja: (etiqueta, valor, detalle)."""
    cells = "".join(
        f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div>'
        + (f'<div class="d">{d}</div>' if d else "") + "</div>"
        for k, v, d in items)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def ptitle(text: str) -> None:
    st.markdown(f'<div class="ptitle">{text}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Estilo de gráficas
# ═════════════════════════════════════════════════════════════════════════════
def style(fig: go.Figure, P: dict, height: int, xtitle: str = "", ytitle: str = "",
          legend: bool = True) -> go.Figure:
    axis = dict(
        showgrid=True, gridcolor=P["hair"], gridwidth=1, zeroline=False,
        showline=True, linecolor=P["hair2"], linewidth=1, ticks="",
        tickfont=dict(size=11, color=P["muted"]),
        title=dict(font=dict(size=12, color=P["muted"])),
    )
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=8, b=8),
        font=dict(family=FONT, size=12, color=P["ink2"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(bgcolor=P["surface"], bordercolor=P["hair2"],
                        font=dict(family=FONT, size=12, color=P["ink"])),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, traceorder="normal", font=dict(size=11, color=P["ink2"]),
                    bgcolor="rgba(0,0,0,0)", itemwidth=30),
        xaxis={**axis, "title": {**axis["title"], "text": xtitle}, "showgrid": False},
        yaxis={**axis, "title": {**axis["title"], "text": ytitle}},
    )
    return fig


PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
               "toImageButtonOptions": {"scale": 2}}
