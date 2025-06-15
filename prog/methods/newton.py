def newton_method(f, df, x0, eps, max_iter=100):
    """
    Метод Ньютона для поиска корня уравнения f(x)=0.
    f   — функция,
    df  — производная f,
    x0  — начальное приближение,
    eps — точность по изменению x,
    max_iter — максимальное число итераций.
    Возвращает список кортежей (k, x_k, f(x_k), f'(x_k), \|x_k - x_{k-1}\|).
    """
    iterations = []
    x = x0
    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError(f"Производная равна нулю на итерации {k}")
        x_new = x - fx / dfx
        diff = abs(x_new - x)
        iterations.append((k, x_new, fx, dfx, diff))
        if diff < eps:
            break
        x = x_new
    return iterations
