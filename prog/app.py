from flask import Flask, render_template, request
import math
from methods.rk import runge_kutta_system
from methods.adams import adams_moulton_system
from methods.gauss import gaussian_elimination
from methods.diff import num_diff_formula
from methods.bisection import bisection_method
from methods.iteration import simple_iteration
from methods.newton import newton_method
from methods.krylov import krylov_method

app = Flask(__name__)

# Разрешённые имена для eval: все из math без приватных
allowed_names = {name: getattr(math, name)
                 for name in dir(math)
                 if not name.startswith("_")}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/runge-kutta", methods=["GET", "POST"])
def runge_kutta():
    result = error = None
    if request.method == "POST":
        try:
            x0 = float(request.form["x0"])
            y10 = float(request.form["y10"])
            y20 = float(request.form["y20"])
            h  = float(request.form["h"])
            n  = int(request.form["n"])
            f1_expr = request.form["f1"]
            f2_expr = request.form["f2"]

            # создаём f1 и f2 с локальным x,y1,y2 и доступными sin,cos,...
            def make_sys_fn(expr):
                return lambda x, y1, y2: eval(
                    expr,
                    {"__builtins__": None},
                    {**allowed_names, "x": x, "y1": y1, "y2": y2}
                )

            f1 = make_sys_fn(f1_expr)
            f2 = make_sys_fn(f2_expr)

            result = runge_kutta_system(f1, f2, x0, y10, y20, h, n)
        except Exception as e:
            error = str(e)
    return render_template("runge_kutta.html", result=result, error=error)

@app.route("/adams", methods=["GET", "POST"])
def adams():
    result = error = None
    if request.method == "POST":
        try:
            x0 = float(request.form["x0"])
            y10 = float(request.form["y10"])
            y20 = float(request.form["y20"])
            h  = float(request.form["h"])
            n  = int(request.form["n"])
            f1_expr = request.form["f1"]
            f2_expr = request.form["f2"]

            def make_sys_fn(expr):
                return lambda x, y1, y2: eval(
                    expr,
                    {"__builtins__": None},
                    {**allowed_names, "x": x, "y1": y1, "y2": y2}
                )

            f1 = make_sys_fn(f1_expr)
            f2 = make_sys_fn(f2_expr)

            result = adams_moulton_system(f1, f2, x0, y10, y20, h, n)
        except Exception as e:
            error = str(e)
    return render_template("adams.html", result=result, error=error)

@app.route("/gauss", methods=["GET", "POST"])
def gauss():
    result = error = None
    if request.method == "POST":
        try:
            n = int(request.form["n"])
            A = [
                [float(request.form[f"a_{i}_{j}"]) for j in range(n)]
                for i in range(n)
            ]
            b = [float(request.form[f"b_{i}"]) for i in range(n)]
            result = gaussian_elimination(A, b)
        except Exception as e:
            error = str(e)
    return render_template("gauss.html", result=result, error=error)

@app.route("/diff", methods=["GET", "POST"])
def diff():
    formula = error = None
    k = None
    if request.method == "POST":
        try:
            k = int(request.form["k"])
            p = int(request.form["p"])
            r = int(request.form["r"])
            s = int(request.form["s"])
            if r + s + 1 != k + p:
                raise ValueError("Требуется r + s + 1 = k + p")
            formula = num_diff_formula(k, p, r, s)
        except Exception as e:
            error = str(e)
    return render_template("diff.html", formula=formula, error=error, k=k)

@app.route("/bisect", methods=["GET", "POST"])
def bisect():
    iterations = error = None
    if request.method == "POST":
        try:
            a   = float(request.form["a"])
            b   = float(request.form["b"])
            eps = float(request.form["eps"])
            expr = request.form["f"]
            # f(x) с доступными sin, cos, exp и т.д.
            f = lambda x: eval(
                expr,
                {"__builtins__": None},
                {**allowed_names, "x": x}
            )
            iterations = bisection_method(f, a, b, eps)
        except Exception as e:
            error = str(e)
    return render_template("bisect.html", iterations=iterations, error=error)

@app.route("/iter", methods=["GET","POST"])
def simple_iter():
    iterations = error = None
    if request.method == "POST":
        try:
            n   = int(request.form["n"])
            # Собираем матрицу A и векторы b, x0
            A  = [[float(request.form[f"a_{i}_{j}"]) for j in range(n)] for i in range(n)]
            b  = [float(request.form[f"b_{i}"])          for i in range(n)]
            x0 = [float(request.form[f"x0_{i}"])         for i in range(n)]
            eps = float(request.form["eps"])
            iterations = simple_iteration(A, b, x0, eps)
        except Exception as e:
            error = str(e)
    return render_template("iter.html", iterations=iterations, error=error)

@app.route("/newton", methods=["GET", "POST"])
def newton():
    iterations = error = None
    if request.method == "POST":
        try:
            x0   = float(request.form["x0"])
            eps  = float(request.form["eps"])
            f_expr  = request.form["f"]
            df_expr = request.form["df"]

            f = lambda x: eval(
                f_expr,
                {"__builtins__": None},
                {**allowed_names, "x": x}
            )
            df = lambda x: eval(
                df_expr,
                {"__builtins__": None},
                {**allowed_names, "x": x}
            )

            iterations = newton_method(f, df, x0, eps)
        except Exception as e:
            error = str(e)
    return render_template("newton.html", iterations=iterations, error=error)

@app.route("/krylov", methods=["GET", "POST"])
def krylov():
    result = error = None
    if request.method == "POST":
        try:
            n = int(request.form["n"])
            A = [[float(request.form[f"a_{i}_{j}"]) for j in range(n)]
                 for i in range(n)]
            f = [float(request.form[f"f_{i}"]) for i in range(n)]
            V, b, coeffs, eigvals = krylov_method(A, f)
            result = {"V": V, "b": b, "coeffs": coeffs, "eigvals": eigvals}
        except Exception as e:
            error = str(e)
    return render_template("krylov.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
test
