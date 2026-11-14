"""
formula.py — Intérprete seguro de fórmulas μ(t) escritas por el usuario.

Acepta expresiones como  0.05 + 0.3*sin(2*pi*t)  o  0.1*exp(-t) + step(t-0.5)
y las evalúa de forma vectorizada sobre un arreglo de tiempos. No usa eval():
se recorre el árbol sintáctico y solo se permiten números, la variable t,
las constantes y las funciones del diccionario FUNCS.
"""
from __future__ import annotations

import ast
import math

import numpy as np

FUNCS = {
    # nombre: (función numpy, descripción)
    "sin": (np.sin, "seno"), "cos": (np.cos, "coseno"), "tan": (np.tan, "tangente"),
    "exp": (np.exp, "exponencial"), "log": (np.log, "logaritmo natural"), "sqrt": (np.sqrt, "raíz cuadrada"),
    "abs": (np.abs, "valor absoluto"), "sign": (np.sign, "signo"),
    "sinh": (np.sinh, "seno hiperbólico"), "cosh": (np.cosh, "coseno hiperbólico"), "tanh": (np.tanh, "tangente hiperbólica"),
    "floor": (np.floor, "parte entera"), "ceil": (np.ceil, "techo"),
    "step": (lambda x: np.heaviside(np.asarray(x, dtype=float), 1.0), "escalón: 0 si x<0, 1 si x≥0"),
    "max": (np.maximum, "máximo de dos"), "min": (np.minimum, "mínimo de dos"),
    "mod": (np.mod, "residuo"),
}
CONSTS = {"pi": math.pi, "e": math.e}

_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
        ast.Div: np.divide, ast.Pow: np.power, ast.Mod: np.mod}
_UN = {ast.USub: np.negative, ast.UAdd: np.positive}


class FormulaError(ValueError):
    pass


def _eval(node, t):
    if isinstance(node, ast.Expression):
        return _eval(node.body, t)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "t":
            return t
        if node.id in CONSTS:
            return CONSTS[node.id]
        raise FormulaError(f"nombre desconocido: «{node.id}» (usa t, pi, e o una función del diccionario)")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left, t), _eval(node.right, t))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand, t))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in FUNCS:
            raise FormulaError(f"función desconocida: «{name}»")
        if node.keywords:
            raise FormulaError("no se admiten argumentos con nombre")
        args = [_eval(a, t) for a in node.args]
        try:
            return FUNCS[name][0](*args)
        except TypeError:
            raise FormulaError(f"número de argumentos incorrecto en {name}()")
    raise FormulaError("expresión no permitida: usa números, t, + − × ÷ ^ y las funciones del diccionario")


def compile_formula(expr: str):
    """Devuelve una función f(t) vectorizada, o lanza FormulaError con un mensaje legible."""
    src = (expr or "").strip().replace("^", "**")
    if not src:
        raise FormulaError("escribe una fórmula, por ejemplo 0.05 + 0.3*sin(2*pi*t)")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"error de sintaxis cerca de la posición {e.offset}") from None

    def f(t):
        with np.errstate(all="ignore"):
            out = np.broadcast_to(np.asarray(_eval(tree, np.asarray(t, dtype=float)), dtype=float), np.shape(t)).copy()
        if not np.all(np.isfinite(out)):
            raise FormulaError("la fórmula produce valores infinitos o indefinidos en [0, T]")
        return out

    f(np.linspace(0.0, 1.0, 5))   # validación temprana
    return f


def help_table() -> str:
    rows = "\n".join(f"| `{k}` | {d} |" for k, (_, d) in FUNCS.items())
    return ("| Función | Significado |\n|---|---|\n" + rows +
            "\n| `pi`, `e` | constantes |\n| `t` | tiempo en años |\n| `+ − * / ^` | operaciones |")
