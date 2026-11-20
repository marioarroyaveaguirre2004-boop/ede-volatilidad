# Laboratorio de volatilidad estocástica

App interactiva (Streamlit) para el trabajo final de **Ecuaciones Diferenciales Estocásticas**.
Reúne los cinco modelos vistos en clase — **Hull & White, Stein & Stein, Nelson, Heston y Scott** —
con la deriva usual `μ_t X_t dt` y la redefinida con reversión a la media `α(μ_t − X_t) dt`,
y tres tipos de tendencia: constante `μ`, periódica `μ(t)` y estocástica `μ_t`.

## Ejecutar en local

```bash
python -m venv .venv && source .venv/bin/activate     # opcional
pip install -r requirements.txt
streamlit run app.py
```

## Qué incluye

| | |
|---|---|
| Panel de control | modelo, deriva usual / reversión, tipo de μ, κ, θ, σ, η, ρ, V₀, X₀, T, nº de trayectorias; sliders para los parámetros que se exploran en vivo (σ, θ, κ, η, ρ, α) y cajas numéricas para los que se fijan una vez |
| Tendencia por fórmula | μ(t) escrita a mano (`0.05 + 0.3*sin(2*pi*t)`) con un diccionario de funciones seguro (`formula.py`) |
| Comparar | dos modelos con el mismo ruido browniano, lado a lado |
| Animación | botón ▶ para ver las trayectorias avanzar desde t = 0 |
| Ruido congelado | misma semilla al mover un parámetro: las trayectorias se deforman, no cambian |
| Ecuaciones | el sistema completo en LaTeX, según las opciones elegidas |
| Banda ±2σ | media simulada ± 2 desviaciones, contra la media teórica |
| Distribución | histograma en cualquier instante t, con densidad teórica (HW, Stein-Stein, Heston) |
| Teoría vs simulación | tabla con error estándar y z |
| Referencia | 📌 fija la media y la banda actuales para comparar antes/después |

## Verificación

```bash
python validate.py          # Monte Carlo (30 000 trayectorias) vs fórmulas cerradas
python tests/test_app.py    # recorre la interfaz: 5 modelos × 2 derivas × 3 tendencias
```

## Desplegar gratis en Streamlit Community Cloud

1. Crea un repositorio en GitHub y sube **todo el contenido de esta carpeta** (`app.py`, `models.py`,
   `requirements.txt`, `.streamlit/`, …).
2. Entra a <https://share.streamlit.io> e inicia sesión con GitHub.
3. **Create app → Deploy a public app from GitHub**: elige el repo, rama `main`, archivo principal `app.py`.
4. **Deploy**. En 1–2 minutos tendrás una URL pública.
## Estructura

```
app.py          interfaz (Streamlit + Plotly)
models.py       motor numérico: modelos, simulación, teoría (sin dependencias de Streamlit)
formula.py      intérprete seguro de μ(t) escrita por el usuario
ui.py           tema visual: paleta, CSS, helpers de interfaz
validate.py     validación Monte Carlo
tests/          pruebas de la interfaz
```
