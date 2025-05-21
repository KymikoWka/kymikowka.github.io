import math
from methods.gauss import gaussian_elimination

def num_diff_formula(k, p, r, s):
    """
    Построение коэффициентов сходящейся формулы численного
    дифференцирования порядка k (k-я производная)
    с точностью p и узлами от -r до +s.
    Возвращает список пар (узел, коэффициент).
    """
    # узлы j = -r, …, 0, …, +s
    nodes = list(range(-r, s+1))
    N = len(nodes)          # число неизвестных
    M = k + p               # число уравнений

    # матрица моментов: A[i][j] = (nodes[j])**i, i=0..M-1
    A = [[nodes[j]**i for j in range(N)] for i in range(M)]

    # правая часть: b[i] = 0, кроме i=k → k!
    b = [0.0]*M
    b[k] = math.factorial(k)

    # решаем A x = b
    coeffs = gaussian_elimination(A, b)

    # возвращаем [(node, coeff), …]
    return list(zip(nodes, coeffs))
