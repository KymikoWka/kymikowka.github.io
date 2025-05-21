from methods.gauss import gaussian_elimination
import numpy as np

def krylov_method(A, f):
    """
    Решение полной проблемы собственных значений методом Крылова.
    A — матрица (n×n), f — начальный вектор длины n.
    Возвращает:
      V     — матрица системы Крылова (n×n),
      b     — вектор правой части (длина n),
      coeffs — коэффициенты характеристического многочлена,
      eigvals — список собственных значений.
    Характеристический многочлен: p(λ) = λ^n + coeffs[0]·λ^{n-1} + ... + coeffs[n-1].
    Собственные значения находятся как корни этого многочлена.
    """
    n = len(A)
    # Построим векторы Krylov: v0=f, v1=A f, ..., v_n = A^n f
    vs = [f[:] ]
    for i in range(1, n+1):
        prev = vs[i-1]
        v = [sum(A[row][col] * prev[col] for col in range(n)) for row in range(n)]
        vs.append(v)

    # Матрица системы Крылова: V[i][j] = vs[j][i], j=0..n-1, i=0..n-1
    V = [[vs[j][i] for j in range(n)] for i in range(n)]
    # Правая часть: b[i] = -vs[n][i]
    b = [-vs[n][i] for i in range(n)]

    # Решаем V · coeffs = b
    coeffs = gaussian_elimination(V, b)

    # Находим собственные значения как корни полинома
    # p(λ) = λ^n + coeffs[0] λ^{n-1} + ... + coeffs[n-1]
    poly = [1.0] + coeffs
    eigvals = np.roots(poly).tolist()

    return V, b, coeffs, eigvals