"""
Recorre la interfaz con el framework de pruebas de Streamlit:
las 5 × 2 × 3 combinaciones modelo × deriva × tendencia, más cambios de
parámetros, referencia fijada y ejes fijos. Falla si alguna ejecución lanza excepción.

Uso:  python tests/test_app.py        (o:  pytest tests/)
"""
from __future__ import annotations

import itertools
import pathlib
import sys

from streamlit.testing.v1 import AppTest

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
sys.path.insert(0, str(pathlib.Path(APP).parent))

from models import MODELS  # noqa: E402


def fresh() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_default_runs():
    at = fresh()
    assert "Laboratorio de volatilidad" in "".join(m.value for m in at.markdown)


def test_all_combinations():
    at = fresh()
    for model, drift, trend in itertools.product(MODELS, ("usual", "mr"), ("const", "periodic", "stoch")):
        at.pills(key="model").set_value(model)
        at.segmented_control(key="drift").set_value(drift)
        at.segmented_control(key="trend").set_value(trend)
        at.run()
        assert not at.exception, (model, drift, trend, [e.value for e in at.exception])


def test_extreme_sliders():
    at = fresh()
    at.pills(key="model").set_value("heston")
    at.run()
    at.slider(key="heston_sigma").set_value(1.5)
    at.slider(key="heston_kappa").set_value(0.1)
    at.slider(key="rho_heston").set_value(0.99)
    at.slider(key="T").set_value(5.0)
    at.select_slider(key="spy").set_value(504)
    at.slider(key="n_paths").set_value(2000)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_changing_x0_in_mean_reversion():
    at = fresh()
    at.segmented_control(key="drift").set_value("mr")
    at.segmented_control(key="trend").set_value("periodic")
    at.run()
    at.slider(key="x0").set_value(500.0)   # los rangos de μ cambian con X₀
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_lock_and_reference():
    at = fresh()
    at.pills(key="model").set_value("heston")
    at.toggle(key="lock").set_value(True)
    at.run()
    at.button[1].click()  # (0 = nueva semilla, 1 = fijar referencia)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.slider(key="heston_sigma").set_value(0.8)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_new_seed_and_unfrozen_noise():
    at = fresh()
    at.button[0].click()
    at.run()
    assert not at.exception
    at.checkbox(key="frozen").set_value(False)
    at.run()
    assert not at.exception


def test_compare_and_animate():
    at = fresh()
    at.toggle(key="animate").set_value(True)
    at.toggle(key="compare").set_value(True)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.pills(key="model_b").set_value("scott")
    at.segmented_control(key="trend").set_value("stoch")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    # el modelo A pasa a ser el que era B: la app debe elegir otro B sin romperse
    at.pills(key="model").set_value("scott")
    at.run()
    assert not at.exception, [e.value for e in at.exception]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"✓ {fn.__name__}")
    print("\nTODO OK")
