def gaussian_elimination(A, b):
    """
    Решение Ax = b методом Гаусса с выбором главного элемента по столбцу.
    A — список списков (n×n), b — список длины n.
    Возвращает список x.
    """
    n = len(b)
    # копируем, чтобы не портить исходные
    a = [row[:] for row in A]
    f = b[:]

    # прямой ход
    for k in range(n):
        # находим строку с максимальным |a[i][k]|
        max_row = max(range(k, n), key=lambda i: abs(a[i][k]))
        if abs(a[max_row][k]) < 1e-12:
            raise ZeroDivisionError("Нулевой ведущий элемент при k=%d" % k)
        # меняем местами k-ю и max_row
        a[k], a[max_row] = a[max_row], a[k]
        f[k], f[max_row] = f[max_row], f[k]
        # исключаем
        for i in range(k+1, n):
            m = a[i][k] / a[k][k]
            f[i] -= m * f[k]
            for j in range(k, n):
                a[i][j] -= m * a[k][j]

    # обратный ход
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        s = f[i]
        for j in range(i+1, n):
            s -= a[i][j] * x[j]
        x[i] = s / a[i][i]
    return x
