def simple_iteration(A, b, x0, eps, max_iter=200):
    """
    Решение Ax=b методом простой (последовательной) итерации:
      x^{k+1} = D^{-1}(b − (L+U)x^k)
    где A=D−(L+U).
    Возвращает список кортежей (k, x_vector, ||x^{k+1}−x^k||∞).
    """
    n = len(b)
    # D⁻¹
    D_inv = [1.0 / A[i][i] for i in range(n)]
    # Матрица C = −D⁻¹·(A−diag(A))
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i][j] = -A[i][j] * D_inv[i]
    # Вектор d = D⁻¹·b
    d = [b[i] * D_inv[i] for i in range(n)]

    x = x0[:]
    iterations = []
    for k in range(1, max_iter+1):
        # x_{k+1} = d + C·x_k
        x_new = [d[i] + sum(C[i][j]*x[j] for j in range(n)) for i in range(n)]
        # норма расхождения
        norm = max(abs(x_new[i] - x[i]) for i in range(n))
        iterations.append((k, x_new, norm))
        if norm < eps:
            break
        x = x_new
    return iterations
