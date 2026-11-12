"""
ui.py — Tema visual del laboratorio: paleta, CSS, helpers de interfaz y estilo de gráficas.

Todo lo que es "cómo se ve" vive aquí; app.py se ocupa de qué se muestra.
"""
from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
DISPLAY = "Fraunces, Georgia, serif"


# ═════════════════════════════════════════════════════════════════════════════
#  Paleta — una sola fuente de verdad para CSS y Plotly
# ═════════════════════════════════════════════════════════════════════════════
def palette(dark: bool = False) -> dict:
    """La app se sirve siempre con el tema claro definido en .streamlit/config.toml.

    No se usa st.context.theme: en Streamlit Cloud puede reportar «dark» (por la
    preferencia del sistema) mientras la página se pinta con el tema claro del
    archivo de configuración, y los textos quedan invisibles.
    """
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
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap");
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
section[data-testid="stSidebar"] {{ border-right: 1px solid var(--hair); width: 23.5rem !important; min-width: 23.5rem !important; }}
section[data-testid="stSidebar"] > div:first-child {{ width: 23.5rem !important; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 1.2rem; }}
hr {{ margin: .6rem 0 .9rem; border-color: var(--hair) !important; }}
/* ── marca ──────────────────────────────────────────────────────────────── */
.brand {{ display: flex; align-items: center; gap: .7rem; margin: 0 0 1.4rem; }}
.brand .mark {{ width: 36px; height: 36px; border-radius: 10px; background: var(--primary); color: #fff; display: grid; place-items: center;
               font-family: {DISPLAY}; font-size: 1.25rem; font-weight: 600; box-shadow: 0 4px 12px rgba(79,70,229,.25); }}
.brand .t {{ font-family: {DISPLAY}; font-size: 1.02rem; font-weight: 600; line-height: 1.1; }}
.brand .s {{ font-size: .7rem; color: var(--muted); letter-spacing: .04em; margin-top: .15rem; }}
[data-testid="stSegmentedControl"] button {{ font-size: .78rem; padding: .3rem .45rem; white-space: nowrap; }}
.nt {{ text-transform: none; }}
[data-testid="stSegmentedControl"] {{ margin-bottom: .2rem; }}
/* ── etiquetas de sección en la barra lateral ──────────────────────────── */
.sec {{
  font-size: .66rem; font-weight: 650; letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted); margin: 1.3rem 0 .45rem; padding-top: .9rem; border-top: 1px solid var(--hair);
}}
.sec.first {{ border-top: 0; padding-top: 0; margin-top: 0; }}
.sec:first-child {{ margin-top: 0; }}
/* ── sliders: fila etiqueta + valor, pista limpia, sin burbuja ─────────── */
[class*="st-key-sl_"] [data-testid="stVerticalBlock"] {{ gap: 0 !important; }}
[class*="st-key-sl_"] [data-testid="stHorizontalBlock"] {{ gap: .5rem; align-items: baseline; flex-wrap: nowrap; }}
[class*="st-key-sl_"] [data-testid="stHorizontalBlock"] > *:first-child {{ flex: 1 1 0 !important; min-width: 0; width: auto !important; }}
[class*="st-key-sl_"] [data-testid="stHorizontalBlock"] > *:last-child {{ flex: 0 0 auto !important; }}
[class*="st-key-sl_"] [data-testid="stMarkdownContainer"] p {{ margin: 0; font-size: .86rem; color: var(--ink2); }}
[class*="st-key-sl_"] .val {{
  font-variant-numeric: tabular-nums; font-weight: 600; font-size: .86rem; color: inherit;
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
.eyebrow {{ font-size: .7rem !important; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--primary) !important; margin: 0 0 .3rem !important; }}
.hdr-title {{ font-family: {DISPLAY} !important; font-size: 2.3rem !important; font-weight: 600; letter-spacing: -.02em; line-height: 1.1; margin: 0 !important; font-variation-settings: "opsz" 144; }}
.hdr-sub {{ color: var(--muted) !important; font-size: .92rem !important; margin: .4rem 0 0 !important; }}
.hdr-rule {{ height: 1px; background: linear-gradient(90deg, var(--primary) 0 18%, var(--hair) 18% 100%); margin: 1.1rem 0 1rem; }}
.chips {{ display: flex; gap: .4rem; flex-wrap: wrap; justify-content: flex-end; padding-top: .35rem; }}
.chip {{
  font-size: .74rem; font-weight: 600; padding: .28rem .65rem; border-radius: 999px;
  border: 1px solid color-mix(in srgb, currentColor 22%, transparent); color: inherit; opacity: .85;
  background: color-mix(in srgb, currentColor 5%, transparent); white-space: nowrap;
}}
.chip.on {{ background: var(--primary); border-color: var(--primary); color: #fff; opacity: 1; }}
/* ── tarjeta de ecuaciones ─────────────────────────────────────────────── */
.st-key-eq_card {{
  background: {P['tint']}; border: 1px solid color-mix(in srgb, currentColor 12%, transparent); border-radius: 16px;
  padding: 1.3rem 1.6rem 1.3rem; margin: 0 0 .6rem;
}}
.st-key-eq_card [data-testid="stHorizontalBlock"] {{ align-items: center; }}
.st-key-eq_card [data-testid="stColumn"]:last-child {{ border-left: 1px solid color-mix(in srgb, currentColor 12%, transparent); padding-left: 1.6rem; }}
.st-key-eq_card .katex-display {{ margin: 0; }}
.st-key-eq_card .katex {{ font-size: 1.14em; }}
.dl {{ display: grid; grid-template-columns: auto 1fr; gap: .55rem 1rem; font-size: .86rem; line-height: 1.45; align-items: baseline; }}
.dl dt {{ font-size: .68rem; font-weight: 650; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin: 0; padding-top: .15rem; }}
.dl dd {{ margin: 0; }}
.dl dd b {{ font-weight: 650; }}
.tag {{ display: inline-block; font-size: .72rem; font-weight: 600; padding: .14rem .5rem; border-radius: 6px;
        background: var(--surface2); color: var(--ink2); border: 1px solid var(--hair); }}
.tag.good {{ color: {P['good']}; }} .tag.warn {{ color: {P['warn']}; }} .tag.info {{ color: var(--primary); }}
/* ── estadísticas sin caja ─────────────────────────────────────────────── */
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); margin: .6rem 0 .4rem; padding: .3rem 0; border-top: 1px solid var(--hair); border-bottom: 1px solid var(--hair); }}
.stat {{ padding: .55rem 1.1rem .55rem 0; border-right: 1px solid color-mix(in srgb, currentColor 14%, transparent); }}
.stat + .stat {{ padding-left: 1.1rem; }}
.stat:last-child {{ border-right: 0; }}
.stat .k {{ font-size: .72rem; color: var(--muted); font-weight: 500; letter-spacing: .02em; }}
.stat .v {{ font-family: {DISPLAY}; font-size: 1.75rem; font-weight: 500; letter-spacing: -.01em; line-height: 1.15; margin-top: .15rem;
            font-variant-numeric: tabular-nums; color: inherit; font-variation-settings: "opsz" 72; }}
.stat .d {{ font-size: .74rem; color: var(--muted); margin-top: .1rem; font-variant-numeric: tabular-nums; }}
@media (max-width: 900px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} .stat:nth-child(2) {{ border-right: 0; }} }}
/* ── pestañas, títulos de gráfica, notas ───────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {{ font-size: .9rem; font-weight: 550; padding: .5rem .2rem; }}
.ptitle {{ font-family: {DISPLAY}; font-size: 1.2rem; font-weight: 600; margin: 1rem 0 -.2rem; letter-spacing: -.01em; font-variation-settings: "opsz" 48; }}
.ptitle small {{ font-family: {FONT}; font-size: .74rem; font-weight: 500; color: var(--muted); margin-left: .6rem; letter-spacing: .02em; }}
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
def section(text: str, first: bool = False) -> None:
    st.markdown(f'<div class="sec{" first" if first else ""}">{text}</div>', unsafe_allow_html=True)


def brand() -> None:
    st.markdown('<div class="brand"><div class="mark">σ</div><div><div class="t">Laboratorio EDE</div>'
                '<div class="s">ECUACIONES DIFERENCIALES ESTOCÁSTICAS</div></div></div>', unsafe_allow_html=True)


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
            st.markdown(label, width="content")
            st.markdown(f'<span class="val" title="{help or ""}">{txt}</span>', unsafe_allow_html=True, width="content")
        return st.slider(label, lo, hi, default, step, key=key, label_visibility="collapsed")


def stats(items: list[tuple[str, str, str | None]]) -> None:
    """Fila de estadísticas sin caja: (etiqueta, valor, detalle)."""
    cells = "".join(
        f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div>'
        + (f'<div class="d">{d}</div>' if d else "") + "</div>"
        for k, v, d in items)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def ptitle(text: str, sub: str = "") -> None:
    st.markdown(f'<div class="ptitle">{text}' + (f'<small>{sub}</small>' if sub else '') + '</div>', unsafe_allow_html=True)


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
